"""Offline calibration fits on the synthetic training set (ground truth is used only here, offline).

    python -m vision.calibration.fit vision/datasets/train

Fits the pellet-radius bias (detected equal-area radius vs. drawn radius, in px) and writes
vision/calibration/vision-calibration.json, which VisionObserver loads at start-up.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from vision.detection.classical import detect

CAL_FILE = Path(__file__).with_name("vision-calibration.json")


def fit_pellet_bias(root: Path, limit: int = 300) -> dict:
    diffs = []
    for f in sorted(p for p in root.glob("*.json") if p.name != "meta.json")[:limit]:
        meta = json.loads(f.read_text())
        cam, gt = meta["draw"]["cam"], meta["gt"]
        img = np.asarray(Image.open(f.with_suffix(".png")).convert("RGB"))
        h, w = img.shape[:2]
        det = detect(img)
        if not det.ok:
            continue
        food = np.asarray(gt["food"], dtype=float).reshape(-1, 4)
        sx = (food[:, 0] - cam["cx"]) * cam["zoom"] + w / 2
        sy = (food[:, 1] - cam["cy"]) * cam["zoom"] + h / 2
        for p in det.pellets:
            d = np.hypot(sx - p.x, sy - p.y)
            k = int(np.argmin(d))
            # Only unambiguous matches: one GT pellet within 2 px and no other within 3× its radius.
            if d[k] < 2.0 and np.sum(d < 3 * food[k, 2] * cam["zoom"]) == 1:
                diffs.append(food[k, 2] * cam["zoom"] - p.r_px)
    a = np.asarray(diffs)
    return {"pelletBiasPx": float(np.median(a)) if len(a) else 0.25, "pelletMatches": int(len(a)),
            "pelletBiasIQR": [float(np.quantile(a, 0.25)), float(np.quantile(a, 0.75))] if len(a) else None, "source": str(root)}


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "vision/datasets/train")
    cal = fit_pellet_bias(root)
    CAL_FILE.write_text(json.dumps(cal, indent=1))
    print(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()
