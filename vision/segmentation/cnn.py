"""Learned segmentation: a small U-Net (6 classes: floor, wall, self, other, pellet, eye) trained on the
synthetic label renders, plugged into the same geometry stage as the classical detector.

    python -m vision.segmentation.cnn train --data vision/datasets/train --val vision/datasets/val-rand --out vision/models/seg-v1.pt
    python -m vision.evaluate vision/datasets/val-shift --segmenter cnn

Scale (grid period) and the HUD are still read with the classical routines; the CNN replaces the colour
heuristics for bodies, pellets, eyes and walls.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from vision.detection.classical import (Detections, Pellet, estimate_floor_rect, estimate_scale, neck_heading, pair_eyes, read_hud, zoom_prior,
                                        HUD)

CLASSES = ["floor", "wall", "self", "other", "pellet", "eye"]
IGNORE = 255
# Class weights: rare classes (eyes, pellets) matter most for the observation.
CLASS_W = th.tensor([0.3, 1.0, 1.0, 1.0, 3.0, 6.0])


def block(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                         nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    def __init__(self, w: tuple[int, int, int, int] = (12, 24, 48, 64), n_cls: int = 6):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = block(3, w[0]), block(w[0], w[1]), block(w[1], w[2]), block(w[2], w[3])
        self.d3, self.d2, self.d1 = block(w[3] + w[2], w[2]), block(w[2] + w[1], w[1]), block(w[1] + w[0], w[0])
        self.head = nn.Conv2d(w[0], n_cls, 1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        d3 = self.d3(th.cat([F.interpolate(e4, size=e3.shape[2:], mode="nearest"), e3], 1))
        d2 = self.d2(th.cat([F.interpolate(d3, size=e2.shape[2:], mode="nearest"), e2], 1))
        d1 = self.d1(th.cat([F.interpolate(d2, size=e1.shape[2:], mode="nearest"), e1], 1))
        return self.head(d1)


# ------------------------------------------------------------------ augmentation (shared with evaluation)
def augment(img: np.ndarray, lab: np.ndarray | None, rng: random.Random, strength: float = 1.0) -> tuple[np.ndarray, np.ndarray | None]:
    """Photometric changes, sensor noise, JPEG compression and partial occlusion (occluded label = ignore)."""
    x = img.astype(np.float32)
    if rng.random() < 0.8 * strength:
        x = x * rng.uniform(0.8, 1.15) + rng.uniform(-18, 18)
    if rng.random() < 0.5 * strength:
        x = x + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 7), x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.5 * strength:
        ok, enc = cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, rng.randint(45, 90)])
        x = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].copy()
    if lab is not None:
        lab = lab.copy()
    if rng.random() < 0.4 * strength:
        h, w = x.shape[:2]
        for _ in range(rng.randint(1, 2)):
            rw, rh = rng.randint(w // 12, w // 5), rng.randint(h // 12, h // 5)
            rx, ry = rng.randint(0, w - rw), rng.randint(0, h - rh - 40)
            x[ry:ry + rh, rx:rx + rw] = rng.choice([(40, 44, 52), (230, 232, 236), (120, 124, 130)])
            if lab is not None:
                lab[ry:ry + rh, rx:rx + rw] = IGNORE
    return x, lab


# ------------------------------------------------------------------ data
def list_samples(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*.json") if p.name != "meta.json")


def load_sample(p: Path) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(Image.open(p.with_suffix(".png")).convert("RGB"))
    lab = np.asarray(Image.open(p.with_name(p.stem + "_label.png")))
    return img, lab


def to_tensor(img: np.ndarray) -> th.Tensor:
    return th.from_numpy(img).permute(2, 0, 1).float().div(255.0)


def random_crop(img: np.ndarray, lab: np.ndarray, size: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h, w = lab.shape
    # Bias crops towards rare classes (eyes, pellets) half of the time.
    if rng.random() < 0.5:
        ys, xs = np.nonzero(lab == 5)
        if len(ys):
            k = rng.randrange(len(ys))
            cy, cx = ys[k], xs[k]
            y0 = int(np.clip(cy - rng.randint(16, size - 16), 0, h - size))
            x0 = int(np.clip(cx - rng.randint(16, size - 16), 0, w - size))
            return img[y0:y0 + size, x0:x0 + size], lab[y0:y0 + size, x0:x0 + size]
    y0, x0 = rng.randint(0, h - size), rng.randint(0, w - size)
    return img[y0:y0 + size, x0:x0 + size], lab[y0:y0 + size, x0:x0 + size]


def train(data: Path, val: Path, out: Path, epochs: int = 12, crop: int = 192, batch: int = 16, lr: float = 2e-3, seed: int = 0, threads: int = 0) -> dict:
    rng = random.Random(seed)
    th.manual_seed(seed)
    if threads > 0:
        th.set_num_threads(threads)  # leave cores for concurrent RL training
    samples = [load_sample(p) for p in list_samples(data)]
    vals = [load_sample(p) for p in list_samples(val)][:60]
    model = TinyUNet()
    opt = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(1, (len(samples) * 3) // batch)
    sched = th.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for _ in range(steps_per_epoch):
            xs, ys = [], []
            for _ in range(batch):
                img, lab = samples[rng.randrange(len(samples))]
                ci, cl = random_crop(img, lab, crop, rng)
                ci, cl = augment(ci, cl, rng)
                xs.append(to_tensor(ci))
                ys.append(th.from_numpy(cl.astype(np.int64)))
            x, y = th.stack(xs), th.stack(ys)
            loss = F.cross_entropy(model(x), y, weight=CLASS_W, ignore_index=IGNORE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        m = evaluate_iou(model, vals)
        hist.append({"epoch": ep, "loss": tot / steps_per_epoch, "valIoU": m, "seconds": time.time() - t0})
        print(json.dumps(hist[-1], default=float), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    th.save({"state": model.state_dict(), "classes": CLASSES, "history": hist, "train": str(data), "val": str(val), "crop": crop}, out)
    return {"history": hist}


@th.no_grad()
def predict(model: TinyUNet, img: np.ndarray) -> np.ndarray:
    model.eval()
    h, w = img.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    x = to_tensor(np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge"))[None]
    return model(x)[0, :, :h, :w].argmax(0).numpy().astype(np.uint8)


def evaluate_iou(model: TinyUNet, vals: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    inter = np.zeros(len(CLASSES))
    union = np.zeros(len(CLASSES))
    for img, lab in vals:
        p = predict(model, img)
        for c in range(len(CLASSES)):
            a, b = p == c, lab == c
            inter[c] += (a & b).sum()
            union[c] += (a | b).sum()
    return {c: float(inter[i] / union[i]) if union[i] else math.nan for i, c in enumerate(CLASSES)}


# ------------------------------------------------------------------ detector
class CnnDetector:
    """Same Detections contract as vision.detection.classical.detect, with CNN masks."""

    def __init__(self, path: str | Path):
        ck = th.load(path, map_location="cpu", weights_only=False)
        self.model = TinyUNet()
        self.model.load_state_dict(ck["state"])
        self.model.eval()

    def __call__(self, rgb: np.ndarray) -> Detections:
        t0 = time.perf_counter()
        h, w = rgb.shape[:2]
        det = Detections(ok=False, reason="", width=w, height=h)
        det.hud_fraction, det.can_boost = read_hud(rgb)
        seg = predict(self.model, rgb)
        m = HUD["margin"]
        seg[max(0, h - m - HUD["barH"] - 10):, : m + HUD["barW"] + 30] = 0
        det.timings["cnn"] = time.perf_counter() - t0
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        H, V = hsv[..., 0].astype(np.float32), hsv[..., 2].astype(np.float32)
        wall = seg == 1
        body = (seg == 2) | (seg == 3)
        det.floor_rect = estimate_floor_rect(wall, np.zeros_like(wall))
        x0, y0, x1, y1 = det.floor_rect
        inside = np.zeros((h, w), bool)
        inside[int(max(0, y0)) + 2: int(min(h, y1)) - 2, int(max(0, x0)) + 2: int(min(w, x1)) - 2] = True
        notfloor = cv2.dilate((seg != 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        det.zoom, det.zoom_conf, det.grid_phase = estimate_scale(V, inside & ~notfloor, zoom_prior(w, h))
        if not np.isfinite(det.zoom):
            det.reason = "grid_not_found"
            return det
        n, lab, stats, cents = cv2.connectedComponentsWithStats((seg == 4).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            det.pellets.append(Pellet(float(cents[i][0]), float(cents[i][1]), math.sqrt(stats[i, cv2.CC_STAT_AREA] / math.pi)))
        cx, cy = w / 2, h / 2
        self_mask = seg == 2
        if not self_mask[max(0, int(cy) - 4): int(cy) + 5, max(0, int(cx) - 4): int(cx) + 5].any():
            det.reason = "self_not_at_centre"
            return det
        det.self_mask, det.obstacle = self_mask, seg == 3
        dist = cv2.distanceTransform(body.astype(np.uint8), cv2.DIST_L2, 3)
        n, lab, stats, cents = cv2.connectedComponentsWithStats((seg == 5).astype(np.uint8), connectivity=8)
        holes = [(float(cents[i][0]), float(cents[i][1]), math.sqrt(stats[i, cv2.CC_STAT_AREA] / math.pi)) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 2]
        cls = np.where(body, 0, -1).astype(np.int32)
        heads = pair_eyes(holes, body | (seg == 5), cls, H, dist)
        own = [hd for hd in heads if math.hypot(hd.x - cx, hd.y - cy) < max(6.0, 1.2 * hd.radius_px)]
        if own:
            best = min(own, key=lambda hd: math.hypot(hd.x - cx, hd.y - cy))
            det.self_heading, det.self_heading_src, det.self_radius_px = best.heading, "eyes", best.radius_px
            det.heads = [hd for hd in heads if hd is not best]
        else:
            det.heads = heads
            nh = neck_heading(self_mask, cx, cy, float(dist[int(cy), int(cx)]) or 6.0)
            if nh is not None:
                det.self_heading, det.self_heading_src = nh, "neck"
        det.ok = det.self_heading is not None
        det.reason = "" if det.ok else "self_heading_unknown"
        det.timings["total"] = time.perf_counter() - t0
        return det


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="vision/models/seg-v1.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    train(Path(args.data), Path(args.val), Path(args.out), epochs=args.epochs, threads=args.threads)


if __name__ == "__main__":
    main()
