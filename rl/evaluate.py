"""Final evaluation of every policy on fresh seeds (never used for training, selection or validation).

    python -m rl.evaluate --episodes 100 --out public/data/rl/final-eval.json

Also publishes the imitation models next to the PPO models and rewrites public/models/index.json.
Learned policies run through the exported JSON + TypeScript inference (exactly what the browser runs).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from rl.common import HELDOUT_OPPONENTS, MODELS_DIR, REPO, RUNS, evaluate_in_node, save_json, summarize_episodes
from rl.envs import EVAL_SEED_BASE

FINAL_SEED_BASE = EVAL_SEED_BASE + 20_000
ORDER = ["ppo-dagger-v1", "ppo-bc-v1", "ppo-random-v1", "il-v1-dagger-r2", "il-v1-bc"]


def publish_imitation(best_round: int) -> None:
    src = RUNS / "il-v1" / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["il-v1-bc.json", f"il-v1-dagger-r{best_round}.json"]:
        if (src / name).exists():
            shutil.copy(src / name, MODELS_DIR / name)


def write_index() -> list[str]:
    ids = []
    for f in sorted(MODELS_DIR.glob("*.json")):
        if f.name == "index.json":
            continue
        d = json.loads(f.read_text())
        if d.get("format") == "serpent-mlp":
            ids.append((d["id"], f.name))
    ids.sort(key=lambda x: ORDER.index(x[0]) if x[0] in ORDER else 99)
    (MODELS_DIR / "index.json").write_text(json.dumps({"models": [{"id": i, "file": n} for i, n in ids]}, indent=1))
    return [i for i, _ in ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default=str(REPO / "public" / "data" / "rl" / "final-eval.json"))
    ap.add_argument("--heuristics", default="mpc,hybrid,survival,aggressive,potential,greedy,random")
    args = ap.parse_args()
    il = json.loads((RUNS / "il-v1" / "summary.json").read_text())
    publish_imitation(il["bestRound"])
    model_ids = write_index()
    seeds = [FINAL_SEED_BASE + i for i in range(args.episodes)]
    held_seeds = seeds[: max(10, args.episodes // 2)]
    policies = [(m, "learned") for m in model_ids] + [(h, "mpc" if h.startswith("mpc") else "heuristic") for h in args.heuristics.split(",") if h]
    out = {"seeds": [seeds[0], seeds[-1]], "episodes": args.episodes, "heldoutEpisodes": len(held_seeds), "policies": []}
    t0 = time.time()
    for pid, kind in policies:
        mf = MODELS_DIR / f"{pid}.json" if kind == "learned" else None
        if mf is not None and not mf.exists():
            mf = next((MODELS_DIR / e["file"] for e in json.loads((MODELS_DIR / "index.json").read_text())["models"] if e["id"] == pid), None)
        res = {"D-mixed": summarize_episodes(evaluate_in_node(mf, pid, {"scenario": "D-mixed", "maxSteps": 1200}, seeds))}
        for name, opps in HELDOUT_OPPONENTS.items():
            res[f"heldout:{name}"] = summarize_episodes(evaluate_in_node(mf, pid, {"scenario": "D-mixed", "opponents": opps, "maxSteps": 1200}, held_seeds))
        out["policies"].append({"id": pid, "kind": kind, "results": res})
        print(f"{pid:18s} survival {res['D-mixed']['survivalS']['mean']:6.1f}s  death {res['D-mixed']['deathRate']['mean']:.2f}  "
              f"maxLen {res['D-mixed']['maxLength']['mean']:6.0f}  | mpc-heavy {res['heldout:mpc-heavy']['survivalS']['mean']:5.1f}s  swarm {res['heldout:swarm']['survivalS']['mean']:5.1f}s  ({time.time() - t0:.0f}s)", flush=True)
    out["seconds"] = time.time() - t0
    save_json(Path(args.out), out)


if __name__ == "__main__":
    main()
