"""Sim-to-Visual transfer experiment (docs/EXPERIMENTS.md §Sim-to-Visual).

The same exported policy is evaluated on the same seeds and opponents under six conditions:
  A  exact engine state                         (Node, obs from the engine)
  B  exact state + measured control latency     (Node, actionDelay = latency measured in E)
  C  exact state + observation noise            (Node, obsNoise σ matched to the vision error measured in E)
  D  screen-based, lockstep (no latency)        (browser + vision, the game waits for each decision)
  E  screen-based, real time                    (browser + vision, the game runs at 1×)
  F  screen-based, real time, shifted visuals   (browser + vision, held-out 'shift' theme)

    python -m vision.sim2visual --model public/models/ppo-dagger-v1.json --episodes 8
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from rl.common import evaluate_in_node, summarize_episodes
from vision.bridge.policy import MlpPolicy
from vision.bridge.runner import DEFAULT_BOTS, run_session
from vision.calibration.analyze import analyze

S2V_SEED_BASE = 930_000
REPO = Path(__file__).resolve().parent.parent


def expand(bots: str) -> list[str]:
    out = []
    for part in bots.split(","):
        pid, n = part.split(":")
        out += [pid] * int(n)
    return out


def ci(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=float)
    if len(a) == 0:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
    m = float(a.mean())
    h = 1.96 * float(a.std(ddof=1)) / math.sqrt(len(a)) if len(a) > 1 else 0.0
    return {"mean": m, "lo": m - h, "hi": m + h, "n": int(len(a))}


def node_condition(model_file: Path, model_id: str, seeds: list[int], opponents: list[str], **env: Any) -> list[dict]:
    eps = evaluate_in_node(model_file, model_id, {"scenario": "D-mixed", "opponents": opponents, "maxSteps": 1200, **env}, seeds)
    return [{"seed": e["seed"], "survivalS": e["survivalS"], "died": e["death"] is not None, "maxLength": e["maxLength"], "eaten": e["eaten"],
             "boostShare": float(np.sum(e["actions"][1::2]) / max(1, np.sum(e["actions"])))} for e in eps]


def vision_condition(model_file: Path, seeds: list[int], mode: str, theme: str, sessions: Path, tag: str, policy: MlpPolicy) -> tuple[list[dict], list[dict]]:
    eps, reports = [], []
    for s in seeds:
        rec = sessions / f"{tag}-{s}"
        summary = run_session(str(model_file), s, mode=mode, theme=theme, bots=DEFAULT_BOTS, record=rec, gt_record=True,
                              max_seconds=125.0 if mode == "realtime" else 600.0)
        rep = analyze(rec, policy)
        ep = rep["episode"]
        # Lockstep runs are capped at 1200 decisions (120 s of game time) like the Node episodes.
        eps.append({"seed": s, "survivalS": min(120.0, ep["survivalS"]), "died": ep["died"], "maxLength": ep["maxLength"], "eaten": ep["eaten"],
                    "boostShare": ep["boostShare"], "endReason": summary["endReason"], "controlHz": ep["controlHz"]})
        reports.append(rep)
        print(f"  {tag} seed {s}: survival {eps[-1]['survivalS']:.1f}s died={ep['died']} maxLen {ep['maxLength']:.0f} ({summary['endReason']})", flush=True)
    return eps, reports


def summarize(eps: list[dict]) -> dict[str, Any]:
    return {"episodes": len(eps), "survivalS": ci([e["survivalS"] for e in eps]), "deathRate": float(np.mean([e["died"] for e in eps])) if eps else math.nan,
            "maxLength": ci([e["maxLength"] for e in eps]), "eaten": ci([e["eaten"] for e in eps]), "boostShare": ci([e["boostShare"] for e in eps])}


def paired(a: list[dict], b: list[dict], key: str) -> dict[str, float]:
    da = {e["seed"]: e[key] for e in a}
    return ci([e[key] - da[e["seed"]] for e in b if e["seed"] in da])


def merge_reports(reports: list[dict]) -> dict[str, Any]:
    def mean_of(path: list[str]) -> float:
        vals = []
        for r in reports:
            x: Any = r
            for p in path:
                x = x.get(p, {}) if isinstance(x, dict) else {}
            if isinstance(x, (int, float)) and math.isfinite(x):
                vals.append(float(x))
        return float(np.mean(vals)) if vals else math.nan
    return {
        "headingErrDeg": mean_of(["perception", "headingDeg", "median"]),
        "zoomRelErr": mean_of(["perception", "zoomRel", "median"]),
        "lengthRelErr": mean_of(["perception", "lengthRel", "median"]),
        "oppPosErrU": mean_of(["perception", "oppPosU", "median"]),
        "obsMAE": {g: mean_of(["obsMAE", g]) for g in ["self", "rayBody", "rayWall", "foodSum", "foodNear", "opponents"]},
        "actionAgreement": mean_of(["actionAgreement"]),
        "latencyMs": {k: mean_of(["latencyMs", k, "median"]) for k in ["sendToPage", "pageToApplied", "sendToDrawn"]},
        "timingMs": {k: mean_of(["timingMs", k, "mean"]) for k in ["captureMs", "detectMs", "policyMs", "inputMs", "stepMs"]},
        "displayLagMs": mean_of(["perception", "displayLagMs", "median"]),
        "controlHz": float(np.mean([r["episode"]["controlHz"] for r in reports])) if reports else math.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--sessions", default="vision/sessions/s2v")
    ap.add_argument("--out", default="public/data/vision/sim2visual.json")
    ap.add_argument("--conditions", default="E,D,F,A,B,C")
    args = ap.parse_args()
    model_file = Path(args.model)
    model_id = json.loads(model_file.read_text())["id"]
    policy = MlpPolicy(model_file)
    seeds = [S2V_SEED_BASE + i for i in range(args.episodes)]
    opps = expand(DEFAULT_BOTS)
    sessions = Path(args.sessions)
    t0 = time.time()
    res: dict[str, Any] = {"model": model_id, "seeds": seeds, "bots": DEFAULT_BOTS, "conditions": {}, "measured": {}}
    eps: dict[str, list[dict]] = {}
    conds = args.conditions.split(",")
    if "E" in conds:
        print("== E: vision, real time", flush=True)
        eps["E"], rep_e = vision_condition(model_file, seeds, "realtime", "default", sessions, "E", policy)
        res["measured"]["E"] = merge_reports(rep_e)
    if "D" in conds:
        print("== D: vision, lockstep", flush=True)
        eps["D"], rep_d = vision_condition(model_file, seeds, "lockstep", "default", sessions, "D", policy)
        res["measured"]["D"] = merge_reports(rep_d)
    if "F" in conds:
        print("== F: vision, real time, shifted theme", flush=True)
        eps["F"], rep_f = vision_condition(model_file, seeds, "realtime", "shift", sessions, "F", policy)
        res["measured"]["F"] = merge_reports(rep_f)
    # Latency and noise for B / C come from the measured real-time vision runs (E).
    m = res["measured"].get("E", {})
    total_ms = sum(v for v in [m.get("displayLagMs", 0), m.get("timingMs", {}).get("stepMs", 0), m.get("latencyMs", {}).get("pageToApplied", 0)] if np.isfinite(v))
    delay_ticks = int(round(total_ms / (1000 / 30))) if total_ms else 4
    om = m.get("obsMAE", {})
    feat = [om.get(k, math.nan) for k in ["rayBody", "rayWall", "foodSum", "foodNear", "opponents"]]
    sigma = float(np.nanmean(feat) * math.sqrt(math.pi / 2)) if np.isfinite(np.nanmean(feat)) else 0.05
    res["derived"] = {"delayTicks": delay_ticks, "delayMs": total_ms, "obsNoiseSigma": sigma}
    if "A" in conds:
        print("== A: exact state", flush=True)
        eps["A"] = node_condition(model_file, model_id, seeds, opps)
    if "B" in conds:
        print(f"== B: exact state + {delay_ticks} ticks latency", flush=True)
        eps["B"] = node_condition(model_file, model_id, seeds, opps, actionDelay=delay_ticks)
        res["latencySweep"] = {d: summarize(node_condition(model_file, model_id, seeds, opps, actionDelay=d)) for d in [0, 3, 6, 9, 12]}
    if "C" in conds:
        print(f"== C: exact state + obs noise σ={sigma:.3f}", flush=True)
        eps["C"] = node_condition(model_file, model_id, seeds, opps, obsNoise=sigma)
        res["noiseSweep"] = {f"{s:.2f}": summarize(node_condition(model_file, model_id, seeds, opps, obsNoise=s)) for s in [0.0, 0.02, 0.05, 0.1, 0.2]}
    for k, v in eps.items():
        res["conditions"][k] = {**summarize(v), "episodesDetail": v}
        if "A" in eps and k != "A":
            res["conditions"][k]["pairedVsA"] = {"survivalS": paired(eps["A"], v, "survivalS"), "maxLength": paired(eps["A"], v, "maxLength")}
    res["seconds"] = time.time() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "episodesDetail"} for k, v in res["conditions"].items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
