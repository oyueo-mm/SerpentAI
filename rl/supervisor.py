"""Training supervisor (docs/REINFORCEMENT_LEARNING.md §Supervisor).

Watching a loss go down is not evidence of a better policy. The supervisor tracks rollout statistics and
held-out validation scores, flags behavioural anomalies, and intervenes when validation keeps getting
worse: it records a warning, preserves the current checkpoint, compares with the best one, halves the
learning rate and tightens the KL target, and finally rolls back to the last validated checkpoint.
Every decision is appended to a JSONL log with its reason.

The class is framework-free (unit-tested with synthetic streams in rl/tests/test_supervisor.py);
SupervisorCallback adapts it to Stable-Baselines3.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

TURNS = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])


@dataclass
class SupervisorConfig:
    #: Validation is "worse" when below best × (1 − tolerance).
    tolerance: float = 0.08
    #: Consecutive worse validations before reducing the learning rate.
    patience_lr: int = 2
    #: Further consecutive worse validations (after the LR cut) before rolling back.
    patience_rollback: int = 2
    lr_factor: float = 0.5
    min_lr: float = 2e-5
    kl_factor: float = 0.5
    min_kl: float = 0.005
    # Anomaly thresholds on episode behaviour (fractions are per decision).
    spin_frac: float = 0.6
    min_cells_per_min: float = 4.0
    wall_near_frac: float = 0.35
    boost_frac: float = 0.5
    min_eaten_per_min: float = 15.0
    passive_survival_s: float = 90.0
    #: Reward per decision above this is physically implausible under reward-v1 → possible exploit.
    max_reward_rate: float = 0.12
    #: Training-mix vs held-out-opponents score ratio above which the policy looks opponent-specific.
    overfit_ratio: float = 1.6


@dataclass
class Intervention:
    kind: str
    reason: str
    data: dict[str, Any] = field(default_factory=dict)


class Supervisor:
    def __init__(self, config: SupervisorConfig | None = None, log_path: Path | None = None):
        self.c = config or SupervisorConfig()
        self.log_path = log_path
        self.best_score = -math.inf
        self.best_step = -1
        self.worse_streak = 0
        self.lr_cut_done = 0
        self.events: list[dict[str, Any]] = []
        self.anomaly_counts: dict[str, int] = {}

    def _log(self, kind: str, **data: Any) -> dict[str, Any]:
        rec = {"t": time.time(), "kind": kind, **data}
        self.events.append(rec)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=float) + "\n")
        return rec

    # ---------------------------------------------------------------- anomalies
    def check_episodes(self, step: int, episodes: list[dict[str, Any]], action_counts: np.ndarray | None = None) -> list[str]:
        """Behavioural anomaly detectors over a batch of finished episodes. Returns the anomaly names found."""
        if not episodes:
            return []
        c = self.c
        found: list[str] = []
        mins = np.array([max(e["survivalS"], 1.0) / 60 for e in episodes])
        spin = np.mean([e.get("spinFrac", 0.0) for e in episodes])
        cells_pm = np.mean([e.get("cells", 0) / m for e, m in zip(episodes, mins)])
        wall = np.mean([e.get("wallNearFrac", 0.0) for e in episodes])
        boost = np.mean([e["boostS"] / max(e["survivalS"], 1e-6) for e in episodes])
        eaten_pm = np.mean([e["eaten"] / m for e, m in zip(episodes, mins)])
        surv = np.mean([e["survivalS"] for e in episodes])
        rate = max(e["r"] / max(e["l"], 1) for e in episodes)
        if spin > c.spin_frac and cells_pm < c.min_cells_per_min:
            found.append("spinning_in_place")
        if wall > c.wall_near_frac:
            found.append("wall_hugging")
        if boost > c.boost_frac and eaten_pm < c.min_eaten_per_min * 2:
            found.append("pointless_boost")
        if surv > c.passive_survival_s and eaten_pm < c.min_eaten_per_min:
            found.append("passive_ignores_food")
        if rate > c.max_reward_rate:
            found.append("implausible_reward_rate")
        kills_h = sum(e.get("killsBy", {}).get("headon", 0) for e in episodes)
        kills_b = sum(e.get("killsBy", {}).get("body", 0) for e in episodes)
        if kills_h + kills_b >= 10 and kills_h > 3 * kills_b:
            found.append("headon_rule_farming")
        if action_counts is not None and action_counts.sum() > 0:
            p = action_counts / action_counts.sum()
            if p.max() > 0.85:
                found.append("action_collapse")
        for a in found:
            self.anomaly_counts[a] = self.anomaly_counts.get(a, 0) + 1
        if found:
            self._log("anomaly", step=step, anomalies=found, stats={"spinFrac": spin, "cellsPerMin": cells_pm, "wallNearFrac": wall,
                                                                    "boostFrac": boost, "eatenPerMin": eaten_pm, "survivalS": surv, "maxRewardRate": rate})
        return found

    def check_generalisation(self, step: int, train_score: float, heldout_score: float) -> list[str]:
        if heldout_score > 0 and train_score / heldout_score > self.c.overfit_ratio:
            self._log("anomaly", step=step, anomalies=["opponent_specific"], stats={"train": train_score, "heldout": heldout_score})
            return ["opponent_specific"]
        return []

    # ---------------------------------------------------------------- validation & interventions
    def observe_validation(self, step: int, score: float, lr: float, target_kl: float | None, details: dict[str, Any] | None = None) -> list[Intervention]:
        c = self.c
        out: list[Intervention] = []
        self._log("validation", step=step, score=score, best=self.best_score, lr=lr, target_kl=target_kl, details=details or {})
        if score > self.best_score:
            self.best_score, self.best_step = score, step
            self.worse_streak = 0
            self.lr_cut_done = 0
            out.append(Intervention("new_best", f"validation {score:.3f} is the best so far", {"step": step}))
            return out
        if score >= self.best_score - abs(self.best_score) * c.tolerance:
            self.worse_streak = 0  # within tolerance of the best: not a regression
            return out
        self.worse_streak += 1
        out.append(Intervention("warning", f"validation {score:.3f} below best {self.best_score:.3f} (step {self.best_step}) for {self.worse_streak} evaluation(s)",
                                {"step": step, "best": self.best_score}))
        out.append(Intervention("preserve", "keep the current checkpoint for comparison", {"step": step}))
        if self.worse_streak == c.patience_lr and lr > c.min_lr:
            new_lr = max(c.min_lr, lr * c.lr_factor)
            new_kl = max(c.min_kl, (target_kl or 0.03) * c.kl_factor)
            self.lr_cut_done += 1
            out.append(Intervention("reduce_lr", f"{self.worse_streak} consecutive worse validations: lr {lr:.2e} → {new_lr:.2e}, target_kl {target_kl} → {new_kl:.4f}",
                                    {"lr": new_lr, "target_kl": new_kl}))
        elif self.worse_streak >= c.patience_lr + c.patience_rollback:
            out.append(Intervention("rollback", f"still worse after the LR cut: restore the validated checkpoint from step {self.best_step}", {"to_step": self.best_step}))
            self.worse_streak = 0
        for iv in out:
            if iv.kind != "warning":
                self._log("intervention", step=step, action=iv.kind, reason=iv.reason, **{k: v for k, v in iv.data.items() if k != "step"})
        return out


class SupervisorCallback:
    """Stable-Baselines3 adapter (constructed lazily to keep this module importable without SB3)."""

    @staticmethod
    def make(supervisor: Supervisor, validate: Callable[[Any, int], tuple[float, dict[str, Any]]], eval_every: int, snapshot_dir: Path,
             on_new_best: Callable[[Any, int], None] | None = None):
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.utils import get_schedule_fn

        class _CB(BaseCallback):
            def __init__(self):
                super().__init__()
                self.next_eval = eval_every
                self.episodes: list[dict[str, Any]] = []
                self.actions = np.zeros(18)
                self.best_state: dict[str, Any] | None = None
                self.best_rms: Any = None
                self.history: list[dict[str, Any]] = []

            def _on_step(self) -> bool:
                for inf in self.locals.get("infos", []):
                    if "episode_info" in inf:
                        self.episodes.append(inf["episode_info"])
                acts = self.locals.get("actions")
                if acts is not None:
                    self.actions += np.bincount(np.asarray(acts).reshape(-1), minlength=18)[:18]
                return True

            def _on_rollout_end(self) -> None:
                if len(self.episodes) >= 16:
                    supervisor.check_episodes(self.num_timesteps, self.episodes, self.actions)
                    eps = self.episodes
                    self.history.append({"step": self.num_timesteps, "episodes": len(eps), "return": float(np.mean([e["r"] for e in eps])),
                                         "survivalS": float(np.mean([e["survivalS"] for e in eps])), "maxLength": float(np.mean([e["maxLength"] for e in eps])),
                                         "deathRate": float(np.mean([e["death"] is not None for e in eps])),
                                         "boostShare": float(self.actions[1::2].sum() / max(1, self.actions.sum()))})
                    self.episodes, self.actions = [], np.zeros(18)
                if self.num_timesteps >= self.next_eval:
                    self.next_eval += eval_every
                    model = self.model
                    score, details = validate(model, self.num_timesteps)
                    lr = float(model.lr_schedule(1.0))
                    ivs = supervisor.observe_validation(self.num_timesteps, score, lr, model.target_kl, details)
                    for iv in ivs:
                        if iv.kind == "new_best":
                            self.best_state = copy.deepcopy(model.policy.state_dict())
                            env = model.get_env()
                            self.best_rms = copy.deepcopy(getattr(env, "obs_rms", None))
                            if on_new_best:
                                on_new_best(model, self.num_timesteps)
                        elif iv.kind == "preserve":
                            snapshot_dir.mkdir(parents=True, exist_ok=True)
                            model.save(snapshot_dir / f"preserved-{self.num_timesteps}.zip")
                        elif iv.kind == "reduce_lr":
                            model.lr_schedule = get_schedule_fn(iv.data["lr"])
                            model.learning_rate = iv.data["lr"]
                            model.target_kl = iv.data["target_kl"]
                        elif iv.kind == "rollback" and self.best_state is not None:
                            model.policy.load_state_dict(self.best_state)
                            env = model.get_env()
                            if self.best_rms is not None and hasattr(env, "obs_rms"):
                                env.obs_rms = copy.deepcopy(self.best_rms)

        return _CB()
