"""Synthetic vision data from the real game page (Playwright + ?view=target).

Every sample stores the rendered frame (PNG), a label render of the same frame (flat class colours, same
geometry) and the exact engine state behind it (camera, heads, pellets, Sensors, obs-v1). Snake 0 is
driven by a built-in pilot policy (data generation only); the game is advanced in lockstep with the
"]" key so the captured frame and its ground truth always refer to the same tick.

    python -m vision.datagen --out vision/datasets/train --episodes 24 --frames 60 --theme rand --theme-base 1000
    python -m vision.datagen --out vision/datasets/val-shift --episodes 4 --frames 60 --theme shift
"""
from __future__ import annotations

import argparse
import io
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

DEFAULT_URL = "http://127.0.0.1:4173/SerpentAI/"
SIZES = [(960, 720), (1280, 720), (800, 800), (1024, 640), (1100, 760)]
PILOTS = ["hybrid", "mpc", "survival", "greedy"]
BOTS = ["hybrid:3,mpc:2,greedy:3,survival:2,aggressive:2", "greedy:6,random:4,aggressive:3", "mpc:4,hybrid:4,potential:3", "aggressive:5,hybrid:5,greedy:4"]

# Label classes (index order is the class id used by the segmentation model).
CLASSES = ["floor", "wall", "self", "other", "pellet", "eye"]
LABEL_RGB = np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.int32)


def label_ids(rgb: np.ndarray) -> np.ndarray:
    """Label render → class ids (nearest class colour; anti-aliased edge pixels go to the nearest)."""
    d = ((rgb[..., None, :].astype(np.int32) - LABEL_RGB[None, None]) ** 2).sum(axis=-1)
    return d.argmin(axis=-1).astype(np.uint8)


def canvas_png(page) -> np.ndarray:
    png = page.locator("canvas").screenshot()
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def wait_drawn(page, tick: int, label: bool, timeout: float = 20.0) -> dict:
    page.wait_for_function("([t, l]) => { const d = window.__serpentGT && window.__serpentGT.draws; if (!d || !d.length) return false; const x = d[d.length-1]; return x.tick === t && x.label === l; }",
                           arg=[tick, label], timeout=timeout * 1000)
    return page.evaluate("() => window.__serpentGT.draws[window.__serpentGT.draws.length - 1]")


def current_gt(page) -> dict:
    return page.evaluate("() => { const f = window.__serpentGT.frames; return f[f.length - 1]; }")


def settle(page, timeout: float = 30.0) -> dict | None:
    """Waits until the worker has processed every queued step (the tick stops advancing), then returns the GT frame."""
    t0 = time.time()
    last, same = None, 0
    while time.time() - t0 < timeout:
        g = current_gt(page)
        tick = g["tick"] if g else None
        same = same + 1 if tick == last else 0
        last = tick
        if same >= 3:
            return g
        page.wait_for_timeout(60)
    return None


def run_episode(pw_browser, out: Path, ep: int, args, rng: random.Random) -> int:
    w, h = rng.choice(SIZES)
    theme = args.theme if args.theme != "rand" else f"rand:{args.theme_base + ep}"
    seed = args.seed_base + ep
    pilot, bots = rng.choice(PILOTS), rng.choice(BOTS)
    ctx = pw_browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=1)
    page = ctx.new_page()
    url = f"{args.url}?view=target&gt=1&autostart=0&respawn=1&seed={seed}&theme={theme}&pilot={pilot}&bots={bots}"
    page.goto(url)
    page.wait_for_function("() => window.__serpentGT && window.__serpentGT.draws.length > 0", timeout=15000)
    # Warm up: let the match develop (bodies grow, deaths, death pellets).
    warm = rng.randint(20, 120)
    for _ in range(warm):
        page.keyboard.press("]")
    settle(page, 60)
    saved = 0
    for k in range(args.frames):
        for _ in range(rng.randint(1, 6)):
            page.keyboard.press("]")
        gt = settle(page)
        if not gt or not gt["self"]["alive"]:
            for _ in range(70):  # wait out the respawn delay
                page.keyboard.press("]")
            settle(page)
            continue
        try:
            draw = wait_drawn(page, gt["tick"], False)
            img = canvas_png(page)
            page.evaluate("() => window.__serpentSetLabel(true)")
            wait_drawn(page, gt["tick"], True)
            lab = label_ids(canvas_png(page))
            page.evaluate("() => window.__serpentSetLabel(false)")
            wait_drawn(page, gt["tick"], False)
        except Exception as err:  # a slow machine can miss a frame: skip the sample, keep the episode
            print(f"  skip sample {k}: {type(err).__name__}", flush=True)
            page.evaluate("() => window.__serpentSetLabel(false)")
            continue
        name = f"e{ep:03d}_f{k:03d}"
        Image.fromarray(img).save(out / f"{name}.png")
        Image.fromarray(lab).save(out / f"{name}_label.png")
        (out / f"{name}.json").write_text(json.dumps({"gt": gt, "draw": draw, "theme": theme, "seed": seed, "pilot": pilot, "bots": bots, "size": [w, h]}))
        saved += 1
    ctx.close()
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--theme", default="rand", help="rand | default | shift")
    ap.add_argument("--theme-base", type=int, default=1000)
    ap.add_argument("--seed-base", type=int, default=50_000)
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed_base)
    t0 = time.time()
    total = 0
    with sync_playwright() as p:
        b = p.chromium.launch()
        for ep in range(args.episodes):
            total += run_episode(b, out, ep, args, rng)
            print(f"episode {ep + 1}/{args.episodes}: {total} frames, {time.time() - t0:.0f}s", flush=True)
        b.close()
    (out / "meta.json").write_text(json.dumps({"frames": total, "args": vars(args), "classes": CLASSES, "seconds": time.time() - t0}))


if __name__ == "__main__":
    main()
