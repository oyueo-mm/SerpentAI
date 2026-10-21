"""Accuracy report for the web lab: classical CV vs. CNN segmentation on every validation set.

    python -m vision.report [--cnn vision/models/seg-v1.pt]

Writes public/data/vision/accuracy.json. 'val-rand+aug' re-uses val-rand frames with the training-time
photometric/JPEG/occlusion augmentation (fixed seed) to test robustness.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from vision.detection.classical import detect
from vision.evaluate import evaluate_dir
from vision.segmentation.cnn import CnnDetector, augment

REPO = Path(__file__).resolve().parent.parent


class Augmented:
    def __init__(self, inner, seed: int = 0):
        self.inner, self.rng = inner, random.Random(seed)

    def __call__(self, img: np.ndarray):
        x, _ = augment(img, None, self.rng, strength=1.0)
        return self.inner(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="vision/models/seg-v1.pt")
    ap.add_argument("--out", default=str(REPO / "public" / "data" / "vision" / "accuracy.json"))
    args = ap.parse_args()
    detectors = {"classical": detect}
    if Path(args.cnn).exists():
        detectors["cnn"] = CnnDetector(args.cnn)
    rows = []
    for ds in ["val-default", "val-rand", "val-shift", "val-rand+aug"]:
        root = REPO / "vision" / "datasets" / ds.replace("+aug", "")
        if not root.exists():
            continue
        for name, det in detectors.items():
            d = Augmented(det, 7) if ds.endswith("+aug") else det
            rep = evaluate_dir(root, d)
            rep.pop("obsMAEPerFeature", None)
            rows.append({"dataset": ds, "segmenter": name, **rep})
            m = rep["metrics"]
            print(f"{ds:14s} {name:9s} ok {m['okRate']['mean']:.2f} heading {m['headingErrDeg']['median']:.2f}° headR {m['headRecall']['mean']:.2f} "
                  f"pelletR {m['pelletRecall']['mean']:.2f} IoU self {rep['iou']['self']['mean']:.3f} other {rep['iou']['other']['mean']:.3f} "
                  f"rayMAE {rep['obsMAE']['rayBody']:.3f} foodMAE {rep['obsMAE']['foodSum']:.3f} {m['detectMs']['mean']:.0f}ms", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
