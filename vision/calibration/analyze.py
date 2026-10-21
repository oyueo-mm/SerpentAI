"""Calibration analysis of a recorded Vision Runner session (runner log + the page's ground-truth log).

    python -m vision.calibration.analyze vision/sessions/<run> [--policy model.json]

Everything is put on one timeline (epoch ms on the same machine):
  runner capture time → tick displayed at that moment (page draw log) → exact engine state of that tick;
  runner input send time → page input receipt → first tick that applied it → time that tick was drawn.
Ground truth is used here only, after the run, to measure errors and latencies.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from vision.bridge.policy import MlpPolicy
from vision.observation.encode import IDX_FOOD_NEAR, IDX_FOOD_SUM, IDX_OPP, IDX_RAY_BODY, IDX_RAY_WALL, IDX_SELF, VIEW_RADIUS

GROUPS = {"self": IDX_SELF, "rayBody": IDX_RAY_BODY, "rayWall": IDX_RAY_WALL, "foodSum": IDX_FOOD_SUM, "foodNear": IDX_FOOD_NEAR, "opponents": IDX_OPP}


def ang(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def stat(xs: list[float]) -> dict[str, float]:
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if len(a) == 0:
        return {"mean": math.nan, "median": math.nan, "p90": math.nan, "n": 0}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.quantile(a, 0.9)), "n": int(len(a))}


def load_session(root: Path) -> tuple[list[dict], dict, dict]:
    log = [json.loads(l) for l in (root / "log.jsonl").read_text().splitlines() if l.strip()]
    gt = json.loads((root / "groundtruth.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    return log, gt, summary


def analyze(root: Path, policy: MlpPolicy | None = None) -> dict[str, Any]:
    log, gt, summary = load_session(root)
    frames = {f["tick"]: f for f in gt["frames"]}
    draws = [d for d in gt["draws"] if not d.get("label")]
    draw_t = [d["epochMs"] for d in draws]
    inputs = gt["inputs"]
    out: dict[str, Any] = {"session": root.name, "summary": summary}

    # ---- per-step perception errors at the displayed tick
    err = {k: [] for k in ["headingDeg", "zoomRel", "lengthRel", "oppPosU", "oppHeadingDeg", "displayLagMs", "turnRateErr", "speedErr"]}
    obs_err = []
    action_agree = []
    tagged: dict[str, list[float]] = {}
    prev = None
    for r in log:
        if r.get("action") is None:
            continue
        k = bisect.bisect_right(draw_t, r["tCap"]) - 1
        if k < 0:
            continue
        d = draws[k]
        f = frames.get(d["tick"])
        if f is None or not f["self"]["alive"]:
            continue
        err["displayLagMs"].append(r["tCap"] - d["epochMs"])
        cam = d["cam"]
        err["headingDeg"].append(math.degrees(ang(r["heading"], f["self"]["heading"])))
        err["zoomRel"].append(abs(r["zoom"] - cam["zoom"]) / cam["zoom"])
        err["lengthRel"].append(abs(r["length"] - f["self"]["length"]) / f["self"]["length"])
        w, h = r.get("w", d["w"]), r.get("h", d["h"])
        others = [(((o["x"] - cam["cx"]) * cam["zoom"] + w / 2), ((o["y"] - cam["cy"]) * cam["zoom"] + h / 2), o) for o in f["others"]]
        for hx, hy, hh, hr in r.get("headList", []):
            best = min(others, key=lambda g: math.hypot(g[0] - hx, g[1] - hy), default=None)
            if best and math.hypot(best[0] - hx, best[1] - hy) < 3 * best[2]["radius"] * cam["zoom"]:
                err["oppPosU"].append(math.hypot(best[0] - hx, best[1] - hy) / cam["zoom"])
                err["oppHeadingDeg"].append(math.degrees(ang(hh, best[2]["heading"])))
        if f["obs"] is not None:
            vis = np.asarray(r["obs"], dtype=np.float32)
            ex = np.asarray(f["obs"], dtype=np.float32)
            ex[3], ex[4] = vis[3], vis[4]   # previous-action features come from the runner itself
            e = np.abs(vis - ex)
            obs_err.append(e)
            # Situations that are hard for vision.
            tags = []
            if any(math.hypot(o["x"] - f["self"]["x"], o["y"] - f["self"]["y"]) < VIEW_RADIUS and
                   any(math.hypot(o["x"] - q["x"], o["y"] - q["y"]) < 3 * (o["radius"] + q["radius"]) for q in f["others"] if q is not o) for o in f["others"]):
                tags.append("crossing")
            if any(not (40 < x < w - 40 and 40 < y < h - 40) and -20 < x < w + 20 and -20 < y < h + 20 for x, y, _ in others):
                tags.append("screen_edge")
            if f["self"]["boosting"] and abs(r.get("turn", 0)) >= 0.75:
                tags.append("boost_sharp_turn")
            dense = sum(1 for i in range(0, len(f["food"]), 4) if math.hypot(f["food"][i] - f["self"]["x"], f["food"][i + 1] - f["self"]["y"]) < 200)
            if dense > 40:
                tags.append("food_dense")
            if prev is not None and len(prev["others"]) > len(f["others"]):
                tags.append("after_death")
            for t in tags or ["normal"]:
                tagged.setdefault(t, []).append(float(e.mean()))
            if policy is not None:
                a_state = int(np.argmax(np.where(([True] * 18 if ex[2] > 0.5 else [i % 2 == 0 for i in range(18)]), policy.logits(ex), -np.inf)))
                action_agree.append(1.0 if a_state == r["action"] else 0.0)
        # Motion from the odometry (camera displacement per second) vs engine speed.
        if prev is not None and r.get("cam") and prev_r.get("cam"):
            dt = (r["tCap"] - prev_r["tCap"]) / 1000
            if dt > 0:
                v = math.hypot(r["cam"][0] - prev_r["cam"][0], r["cam"][1] - prev_r["cam"][1]) / dt
                err["speedErr"].append(abs(v - f["self"]["speed"]))
                wv = math.atan2(math.sin(r["heading"] - prev_r["heading"]), math.cos(r["heading"] - prev_r["heading"])) / dt
                fp = prev
                wt = math.atan2(math.sin(f["self"]["heading"] - fp["self"]["heading"]), math.cos(f["self"]["heading"] - fp["self"]["heading"])) / max(1e-3, (f["tick"] - fp["tick"]) / 30)
                err["turnRateErr"].append(abs(wv - wt))
        prev, prev_r = f, r

    E = np.stack(obs_err) if obs_err else np.zeros((0, 84))
    out["perception"] = {k: stat(v) for k, v in err.items()}
    out["obsMAE"] = {g: float(E[:, idx].mean()) if len(E) else math.nan for g, idx in GROUPS.items()}
    out["obsMAEPerFeature"] = E.mean(axis=0).round(4).tolist() if len(E) else []
    out["obsErrorByTag"] = {t: stat(v) for t, v in tagged.items()}
    if action_agree:
        out["actionAgreement"] = float(np.mean(action_agree))

    # ---- control latency chain (heading/boost commands)
    seqs = sorted(inputs, key=lambda x: x["seq"])
    applied_tick: dict[int, int] = {}
    for f in gt["frames"]:
        s = f["appliedSeq"]
        if s not in applied_tick:
            applied_tick[s] = f["tick"]
    draw_of_tick = {}
    for d in draws:
        draw_of_tick.setdefault(d["tick"], d["epochMs"])
    sends = [r["tSend"] for r in log if r.get("tSend")]
    lat = {"sendToPage": [], "pageToApplied": [], "sendToDrawn": []}
    for inp in seqs:
        j = bisect.bisect_right(sends, inp["epochMs"]) - 1
        if j < 0:
            continue
        t_send = sends[j]
        # first tick whose appliedSeq ≥ this seq
        tick = next((applied_tick[s] for s in sorted(applied_tick) if s >= inp["seq"]), None)
        if tick is None or tick not in draw_of_tick:
            continue
        t_tick = draw_of_tick[tick]
        lat["sendToPage"].append(inp["epochMs"] - t_send)
        lat["pageToApplied"].append(t_tick - inp["epochMs"])
        lat["sendToDrawn"].append(t_tick - t_send)
    out["latencyMs"] = {k: stat(v) for k, v in lat.items() if v}
    out["timingMs"] = {k: stat([r.get(k) for r in log]) for k in ["captureMs", "detectMs", "policyMs", "inputMs", "stepMs"]}
    # Episode outcome from the engine log.
    alive = [f for f in gt["frames"] if f["self"]["alive"]]
    lens = [f["self"]["length"] for f in gt["frames"]]
    gained = sum(max(0.0, b - a) for a, b in zip(lens, lens[1:]))
    out["episode"] = {"survivalS": (alive[-1]["tick"] - gt["frames"][0]["tick"] + 1) / 30 if alive else 0.0, "died": not gt["frames"][-1]["self"]["alive"],
                      "maxLength": max(lens), "eaten": gained, "boostShare": float(np.mean([r.get("boost", False) for r in log if r.get("action") is not None])),
                      "controlHz": len([r for r in log if r.get("action") is not None]) / max(1e-6, summary["seconds"])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pol = MlpPolicy(args.policy) if args.policy else None
    rep = analyze(Path(args.session), pol)
    print(json.dumps({k: v for k, v in rep.items() if k != "obsMAEPerFeature"}, indent=1, default=float))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
