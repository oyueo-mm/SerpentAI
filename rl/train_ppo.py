"""PPO training with a curriculum, self-play pool, supervisor and resumable checkpoints.

    python -m rl.train_ppo --variant random --steps 2000000 --run ppo-random-v1
    python -m rl.train_ppo --variant bc     --init rl/runs/il-v1/bc-r0.zip --steps 2000000 --run ppo-bc-v1
    python -m rl.train_ppo --variant dagger --init rl/runs/il-v1/bc-r4.zip --steps 2000000 --run ppo-dagger-v1

All variants use the same step budget, curriculum schedule, rollout seeds, hyper-parameters and frozen
observation normalisation (statistics of the expert dataset), so they differ only in the initial policy.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure

from rl.common import (EVAL_SEED_BASE, REPO, RUNS, RunningStats, export_model, new_model, save_json, summarize_episodes,
                       evaluate_in_node, vecnormalize_for)
from rl.envs import SerpentVecEnv
from rl.supervisor import Supervisor, SupervisorCallback, SupervisorConfig

#: Curriculum: (stage, scenario, fraction of the step budget, gate on the stage's own scenario).
CURRICULUM = [
    ("A", "A-food", 0.08, {"metric": "eaten", "min": 150.0}),
    ("B", "B-obstacles", 0.08, {"metric": "deathRate", "max": 0.6}),
    ("C", "C-moving", 0.14, {"metric": "deathRate", "max": 0.7}),
    ("D", "D-mixed", 0.30, {"metric": "survivalS", "min": 40.0}),
    ("E", "E-attack", 0.15, {"metric": "survivalS", "min": 35.0}),
    ("F", "F-selfplay", 0.25, None),
]
VAL_SEEDS = [EVAL_SEED_BASE + 5_000 + i for i in range(24)]
GATE_SEEDS = [EVAL_SEED_BASE + 7_000 + i for i in range(16)]


def atomic_save(model: MaskablePPO, path: Path) -> None:
    tmp = path.with_suffix(".tmp.zip")
    model.save(tmp)
    os.replace(tmp, path)


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run = RUNS / args.run
        self.run.mkdir(parents=True, exist_ok=True)
        st = np.load(args.obs_stats)
        self.rms = RunningStats(st["mean"], st["var"])
        self.state_path = self.run / "state.json"
        self.state: dict[str, Any] = json.loads(self.state_path.read_text()) if args.resume and self.state_path.exists() else {
            "steps": 0, "stage": 0, "budgets": [int(f * args.steps) for _, _, f, _ in CURRICULUM], "gates": [], "snapshots": [], "curve": []}
        self.supervisor = Supervisor(SupervisorConfig(), self.run / "supervisor.jsonl")
        self.venv: SerpentVecEnv | None = None
        self.model: MaskablePPO | None = None
        self.vn = None

    # ------------------------------------------------------------------ helpers
    def export(self, model_id: str, label: str, out_dir: Path, extra: dict[str, Any] | None = None) -> Path:
        return export_model(self.model.policy, self.rms, model_id, label, "ppo",
                            {"run": self.args.run, "variant": self.args.variant, "init": self.args.init, "steps": self.state["steps"], **(extra or {})}, out_dir=out_dir)

    def eval_current(self, scenario: str, seeds: list[int], extra_env: dict[str, Any] | None = None) -> dict[str, Any]:
        f = self.export(f"{self.args.run}-tmp", "tmp", self.run / "tmp")
        eps = evaluate_in_node(f, f"{self.args.run}-tmp", {"scenario": scenario, "maxSteps": 1200, **(extra_env or {})}, seeds)
        return summarize_episodes(eps)

    def validate(self, model: MaskablePPO, step: int) -> tuple[float, dict[str, Any]]:
        s = self.eval_current("D-mixed", VAL_SEEDS)
        rec = {"step": step, "return": s["return"]["mean"], "survivalS": s["survivalS"]["mean"], "maxLength": s["maxLength"]["mean"],
               "deathRate": s["deathRate"]["mean"], "kills": s["kills"]["mean"], "stage": CURRICULUM[self.state["stage"]][0]}
        self.state["curve"].append(rec)
        print(f"[validate] step {step}: return {rec['return']:.2f} survival {rec['survivalS']:.1f}s maxLen {rec['maxLength']:.0f} death {rec['deathRate']:.2f}", flush=True)
        return rec["return"], rec

    def build(self, scenario: str) -> None:
        a = self.args
        self.venv = SerpentVecEnv({"scenario": scenario, "maxSteps": 1200}, num_bridges=a.bridges, envs_per_bridge=a.envs_per_bridge, seed=a.seed * 10_007)
        self.vn = vecnormalize_for(self.venv, self.rms, training=False)
        ckpt = self.run / "latest.zip"
        if self.model is None:
            if a.resume and ckpt.exists():
                self.model = MaskablePPO.load(ckpt, env=self.vn, device="cpu")
            elif a.variant == "random":
                self.model = new_model(self.vn, a.seed)
            else:
                self.model = MaskablePPO.load(a.init, env=self.vn, device="cpu")
                self.model.set_random_seed(a.seed)
            self.model.set_logger(configure(str(self.run / "logs"), ["csv"]))
        else:
            self.model.set_env(self.vn)

    # ------------------------------------------------------------------ main loop
    def train(self) -> None:
        a = self.args
        t0 = time.time()
        pool_ids: list[str] = []
        while self.state["stage"] < len(CURRICULUM):
            si = self.state["stage"]
            name, scenario, _, gate = CURRICULUM[si]
            budget = self.state["budgets"][si]
            self.build(scenario)
            print(f"== stage {name} ({scenario}): {budget} steps", flush=True)
            if name == "F":
                pool_ids = self.refresh_pool(pool_ids)
            cb_sup = SupervisorCallback.make(self.supervisor, self.validate, a.eval_every, self.run / "preserved",
                                             on_new_best=lambda m, step: atomic_save(m, self.run / "best.zip"))
            trainer = self

            class Periodic(BaseCallback):
                def __init__(self):
                    super().__init__()
                    self.next_ckpt = trainer.state["steps"] + a.ckpt_every
                    self.next_snap = trainer.state["steps"] + a.snap_every

                def _on_step(self) -> bool:
                    return True

                def _on_rollout_end(self) -> None:
                    trainer.state["steps"] = int(self.model.num_timesteps)
                    if self.model.num_timesteps >= self.next_ckpt:
                        self.next_ckpt += a.ckpt_every
                        atomic_save(self.model, trainer.run / "latest.zip")
                        save_json(trainer.state_path, trainer.state)
                    if name == "F" and self.model.num_timesteps >= self.next_snap:
                        self.next_snap += a.snap_every
                        trainer.refresh_pool(pool_ids)

            self.model.learn(total_timesteps=budget, callback=CallbackList([cb_sup, Periodic()]), reset_num_timesteps=False, progress_bar=False)
            self.state["steps"] = int(self.model.num_timesteps)
            # Gate: check the stage's own skill before moving on; a failed gate extends the stage once,
            # borrowing the extra steps from the final stage so every variant keeps the same total budget.
            if gate:
                s = self.eval_current(scenario, GATE_SEEDS)
                v = s[gate["metric"]]["mean"]
                ok = v >= gate["min"] if "min" in gate else v <= gate["max"]
                ext = sum(1 for g in self.state["gates"] if g["stage"] == name)
                self.state["gates"].append({"stage": name, "metric": gate["metric"], "value": v, "pass": ok, "step": self.state["steps"]})
                print(f"[gate] stage {name}: {gate['metric']} = {v:.3f} → {'pass' if ok else 'FAIL'}", flush=True)
                if not ok and ext == 0:
                    extra = int(0.1 * a.steps)
                    self.state["budgets"][si] = extra
                    self.state["budgets"][-1] = max(int(0.05 * a.steps), self.state["budgets"][-1] - extra)
                    atomic_save(self.model, self.run / "latest.zip")
                    save_json(self.state_path, self.state)
                    self.venv.close()
                    continue
            if name == "E":
                f = self.export(f"{a.run}-stageE", f"{a.run} stage E", self.run / "snapshots")
                self.state["snapshots"].append(str(f))
            self.state["stage"] += 1
            atomic_save(self.model, self.run / "latest.zip")
            save_json(self.state_path, self.state)
            self.venv.close()
        self.state["seconds"] = self.state.get("seconds", 0) + time.time() - t0
        self.finish()

    def refresh_pool(self, pool_ids: list[str]) -> list[str]:
        """Self-play pool: snapshot the current policy, register all snapshots and give each bridge a different pair."""
        a = self.args
        snap_id = f"{a.run}-snap{len(self.state['snapshots'])}"
        f = self.export(snap_id, snap_id, self.run / "snapshots")
        self.state["snapshots"].append(str(f))
        model_files = [json.loads(Path(p).read_text()) for p in self.state["snapshots"]]
        rng = np.random.default_rng(len(self.state["snapshots"]))
        for b in self.venv.bridges:
            for m in model_files:
                b.add_model(m)
            ids = [m["id"] for m in model_files]
            pick = list(rng.choice(ids, size=min(2, len(ids)), replace=False))
            b.set_env({"scenario": "F-selfplay", "maxSteps": 1200, "extraOpponents": pick})
        return [m["id"] for m in model_files]

    def finish(self) -> None:
        a = self.args
        best = self.run / "best.zip"
        final_model = self.model
        if best.exists():
            self.model = MaskablePPO.load(best, device="cpu")
        out = REPO / "public" / "models"
        label = {"random": "PPO (무작위 초기화)", "bc": "PPO (BC 초기화)", "dagger": "PPO (BC+DAgger 초기화)"}[a.variant]
        self.export(a.model_id or a.run, label, out, {"selected": "best validation" if best.exists() else "final", "bestStep": self.supervisor.best_step})
        self.model = final_model
        self.export(f"{a.run}-final", f"{label} final", self.run / "export")
        summary = {"run": a.run, "variant": a.variant, "init": a.init, "steps": self.state["steps"], "budget": a.steps, "seconds": self.state.get("seconds"),
                   "curve": self.state["curve"], "gates": self.state["gates"], "bestStep": self.supervisor.best_step, "bestScore": self.supervisor.best_score,
                   "supervisor": self.supervisor.events, "anomalies": self.supervisor.anomaly_counts, "hyper": vars(a)}
        save_json(self.run / "summary.json", summary)
        save_json(REPO / "public" / "data" / "rl" / f"{a.run}.json", {k: v for k, v in summary.items() if k != "hyper"} | {"hyper": {k: str(v) for k, v in vars(a).items()}})
        print(f"done: {self.state['steps']} steps in {summary['seconds']:.0f}s; best validation {self.supervisor.best_score:.3f} at {self.supervisor.best_step}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["random", "bc", "dagger"], required=True)
    ap.add_argument("--init", default=None, help="BC/DAgger checkpoint (.zip) for the bc/dagger variants")
    ap.add_argument("--obs-stats", default=str(RUNS / "il-v1" / "obs_stats.npz"))
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--bridges", type=int, default=7)
    ap.add_argument("--envs-per-bridge", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=100_000)
    ap.add_argument("--ckpt-every", type=int, default=100_000)
    ap.add_argument("--snap-every", type=int, default=150_000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.variant != "random" and not args.init:
        ap.error("--init is required for bc/dagger variants")
    Trainer(args).train()


if __name__ == "__main__":
    main()
