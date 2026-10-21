"""Diagnostic for the condition-F collapse: same seeds, shifted colours, with vs. without the thickness/shadow change.

    python -m vision.diag_theme --model public/models/ppo-dagger-v1.json --episodes 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl.common import save_json

from vision.bridge.policy import MlpPolicy
from vision.sim2visual import S2V_SEED_BASE, summarize, vision_condition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", default="public/data/vision/theme-diagnostic.json")
    args = ap.parse_args()
    seeds = [S2V_SEED_BASE + i for i in range(args.episodes)]
    eps, _ = vision_condition(Path(args.model), seeds, "realtime", "shift-thin", Path("vision/sessions/diag"), "F-thin", MlpPolicy(args.model))
    out = {"theme": "shift-thin", "seeds": seeds, "summary": summarize(eps), "episodes": eps}
    save_json(Path(args.out), out)
    print(json.dumps(out["summary"], indent=1, default=float))


if __name__ == "__main__":
    main()
