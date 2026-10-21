"""Vision accuracy against ground truth on a generated dataset (vision/datagen.py).

    python -m vision.evaluate vision/datasets/val-rand [--segmenter classical|cnn] [--out report.json]

Ground truth is used only here, to score the pipeline: never inside it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from vision.detection.classical import Detections, detect
from vision.observation.encode import (IDX_FOOD_NEAR, IDX_FOOD_SUM, IDX_OPP, IDX_RAY_BODY, IDX_RAY_WALL, IDX_SELF, OBS_SIZE, VIEW_RADIUS,
                                       Sensors, encode_obs)
from vision.observation.from_vision import VisionObserver

CLASSES = ["floor", "wall", "self", "other", "pellet", "eye"]


def angle_err(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def to_screen(cam: dict, w: float, h: float, x: float, y: float) -> tuple[float, float]:
    return (x - cam["cx"]) * cam["zoom"] + w / 2, (y - cam["cy"]) * cam["zoom"] + h / 2


def det_masks(det: Detections, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    h, w = shape
    m: dict[str, np.ndarray] = {}
    m["self"] = det.self_mask if det.self_mask is not None else np.zeros(shape, bool)
    m["other"] = det.obstacle if det.obstacle is not None else np.zeros(shape, bool)
    wall = np.ones(shape, bool)
    x0, y0, x1, y1 = det.floor_rect
    wall[int(max(0, y0)): int(min(h, y1)) if math.isfinite(y1) else h, int(max(0, x0)): int(min(w, x1)) if math.isfinite(x1) else w] = False
    m["wall"] = wall
    pel = np.zeros(shape, bool)
    yy, xx = np.mgrid[0:h, 0:w]
    for p in det.pellets:
        r = max(1.0, p.r_px)
        y_lo, y_hi, x_lo, x_hi = int(max(0, p.y - r - 1)), int(min(h, p.y + r + 2)), int(max(0, p.x - r - 1)), int(min(w, p.x + r + 2))
        sub = (xx[y_lo:y_hi, x_lo:x_hi] - p.x) ** 2 + (yy[y_lo:y_hi, x_lo:x_hi] - p.y) ** 2 <= r * r
        pel[y_lo:y_hi, x_lo:x_hi] |= sub
    m["pellet"] = pel
    return m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else math.nan


def evaluate_dir(root: Path, detector: Callable[[np.ndarray], Detections] = detect, limit: int | None = None) -> dict[str, Any]:
    files = sorted(p for p in root.glob("*.json") if p.name != "meta.json")[:limit]
    acc: dict[str, list[float]] = {k: [] for k in ["zoomRelErr", "headingErrDeg", "lengthRelErr", "headPosErrU", "headHeadingErrDeg", "pelletRecall",
                                                    "pelletPrecision", "headRecall", "headPrecision", "detectMs", "okRate"]}
    ious: dict[str, list[float]] = {c: [] for c in ["wall", "self", "other", "pellet"]}
    obs_err: list[np.ndarray] = []
    heading_src: dict[str, int] = {}
    reasons: dict[str, int] = {}
    episodes: dict[str, VisionObserver] = {}
    for f in files:
        meta = json.loads(f.read_text())
        gt, draw = meta["gt"], meta["draw"]
        img = np.asarray(Image.open(f.with_suffix(".png")).convert("RGB"))
        lab = np.asarray(Image.open(f.with_name(f.stem + "_label.png")))
        h, w = img.shape[:2]
        det = detector(img)
        acc["detectMs"].append(det.timings.get("total", math.nan) * 1000)
        acc["okRate"].append(1.0 if det.ok else 0.0)
        if not det.ok:
            reasons[det.reason] = reasons.get(det.reason, 0) + 1
            continue
        heading_src[det.self_heading_src] = heading_src.get(det.self_heading_src, 0) + 1
        cam = draw["cam"]
        acc["zoomRelErr"].append(abs(det.zoom - cam["zoom"]) / cam["zoom"])
        acc["headingErrDeg"].append(math.degrees(angle_err(det.self_heading, gt["self"]["heading"])))
        if det.hud_fraction is not None:
            from vision.detection.classical import length_from_gauge
            acc["lengthRelErr"].append(abs(length_from_gauge(det.hud_fraction) - gt["self"]["length"]) / gt["self"]["length"])
        # Heads: GT heads of others whose head centre is on screen and within the view radius.
        gt_heads = []
        for o in gt["others"]:
            sx, sy = to_screen(cam, w, h, o["x"], o["y"])
            if 4 <= sx < w - 4 and 4 <= sy < h - 4 and math.hypot(o["x"] - gt["self"]["x"], o["y"] - gt["self"]["y"]) < VIEW_RADIUS:
                gt_heads.append((sx, sy, o))
        matched = 0
        for hd in det.heads:
            best = min(gt_heads, key=lambda g: math.hypot(g[0] - hd.x, g[1] - hd.y), default=None)
            if best and math.hypot(best[0] - hd.x, best[1] - hd.y) < 2.5 * best[2]["radius"] * cam["zoom"] + 3:
                matched += 1
                acc["headPosErrU"].append(math.hypot(best[0] - hd.x, best[1] - hd.y) / cam["zoom"])
                acc["headHeadingErrDeg"].append(math.degrees(angle_err(hd.heading, best[2]["heading"])))
        if det.heads:
            acc["headPrecision"].append(matched / len(det.heads))
        if gt_heads:
            acc["headRecall"].append(min(1.0, matched / len(gt_heads)))
        # Pellets.
        food = np.asarray(gt["food"], dtype=float).reshape(-1, 4)
        on = []
        for x, y, r, v in food:
            sx, sy = to_screen(cam, w, h, x, y)
            if 3 <= sx < w - 3 and 3 <= sy < h - 3:
                on.append((sx, sy, r * cam["zoom"]))
        hit = 0
        for p in det.pellets:
            if any(math.hypot(p.x - sx, p.y - sy) < max(3.0, 1.5 * r) for sx, sy, r in on):
                hit += 1
        if det.pellets:
            acc["pelletPrecision"].append(hit / len(det.pellets))
        if on:
            acc["pelletRecall"].append(min(1.0, hit / len(on)))
        masks = det_masks(det, (h, w))
        for c in ious:
            ious[c].append(iou(masks[c], lab == CLASSES.index(c)))
        # Observation error vs the exact obs-v1 of the same tick (stateful observer per episode).
        ep = f.stem.split("_")[0]
        obsr = episodes.setdefault(ep, VisionObserver())
        s, _ = obsr.observe(det, gt["tick"] / 30.0, 0.0, bool(gt["self"]["boosting"]))
        if s is not None and gt["obs"] is not None:
            vis = encode_obs(s, gt["obs"][3], bool(gt["obs"][4]))
            obs_err.append(np.abs(vis - np.asarray(gt["obs"], dtype=np.float32)))

    def stat(xs: list[float]) -> dict[str, float]:
        a = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
        if len(a) == 0:
            return {"mean": math.nan, "median": math.nan, "p90": math.nan, "n": 0}
        return {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.quantile(a, 0.9)), "n": int(len(a))}

    E = np.stack(obs_err) if obs_err else np.zeros((0, OBS_SIZE))
    groups = {"self": IDX_SELF, "rayBody": IDX_RAY_BODY, "rayWall": IDX_RAY_WALL, "foodSum": IDX_FOOD_SUM, "foodNear": IDX_FOOD_NEAR, "opponents": IDX_OPP}
    return {
        "frames": len(files),
        "metrics": {k: stat(v) for k, v in acc.items()},
        "iou": {c: stat(v) for c, v in ious.items()},
        "obsMAE": {g: float(E[:, idx].mean()) if len(E) else math.nan for g, idx in groups.items()},
        "obsMAEPerFeature": E.mean(axis=0).round(4).tolist() if len(E) else [],
        "headingSource": heading_src,
        "failures": reasons,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--segmenter", default="classical", choices=["classical", "cnn"])
    ap.add_argument("--model", default="vision/models/seg-v1.pt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    detector = detect
    if args.segmenter == "cnn":
        from vision.segmentation.cnn import CnnDetector
        detector = CnnDetector(args.model)
    rep = evaluate_dir(Path(args.root), detector, args.limit)
    rep["segmenter"] = args.segmenter
    txt = json.dumps({k: v for k, v in rep.items() if k != "obsMAEPerFeature"}, indent=1, default=float)
    print(txt)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
