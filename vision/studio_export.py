"""Exports a recorded Vision Runner session for the read-only web Studio.

    python -m vision.studio_export vision/sessions/s2v/E-930000 [--max-steps 300] [--scale 0.5]

Writes public/data/vision/studio/session.json and downscaled JPEG frames. Per-step errors against the
ground-truth log are added for display only (they were not available to the policy).
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "public" / "data" / "vision" / "studio"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--frame-every", type=int, default=2)
    args = ap.parse_args()
    root = Path(args.session)
    log = [json.loads(l) for l in (root / "log.jsonl").read_text().splitlines() if l.strip()]
    summary = json.loads((root / "summary.json").read_text())
    gt = json.loads((root / "groundtruth.json").read_text()) if (root / "groundtruth.json").exists() else None
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "frames").mkdir(parents=True)
    frames_by_tick = {f["tick"]: f for f in gt["frames"]} if gt else {}
    draws = [d for d in gt["draws"] if not d.get("label")] if gt else []
    draw_t = [d["epochMs"] for d in draws]
    steps = []
    w = h = None
    for r in log:
        if r.get("action") is None:
            continue
        if len(steps) >= args.max_steps:
            break
        w, h = r.get("w", 960), r.get("h", 720)
        frame_name = None
        src = root / "frames" / f"{r['step'] + 1:05d}.jpg"
        if len(steps) % args.frame_every == 0 and src.exists():
            im = Image.open(src)
            im = im.resize((int(im.width * args.scale), int(im.height * args.scale)), Image.LANCZOS)
            frame_name = f"frames/{r['step'] + 1:05d}.jpg"
            im.save(OUT / frame_name, quality=72)
        err = None
        if gt:
            k = bisect.bisect_right(draw_t, r["tCap"]) - 1
            f = frames_by_tick.get(draws[k]["tick"]) if k >= 0 else None
            if f and f["obs"] is not None:
                ex = np.asarray(f["obs"], dtype=float)
                vis = np.asarray(r["obs"], dtype=float)
                ex[3], ex[4] = vis[3], vis[4]
                dh = math.degrees(abs(math.atan2(math.sin(r["heading"] - f["self"]["heading"]), math.cos(r["heading"] - f["self"]["heading"]))))
                err = {"headingDeg": round(dh, 2), "obsMAE": round(float(np.abs(vis - ex).mean()), 4)}
        steps.append({k: r.get(k) for k in ["step", "heading", "zoom", "headingSrc", "headList", "obs", "probs", "action", "turn", "boost", "conf", "length",
                                            "captureMs", "detectMs", "policyMs", "stepMs"]} | {"t": r["tCap"], "frame": frame_name, "err": err})
    out = {"session": root.name, "model": summary.get("model"), "mode": summary.get("mode"), "theme": summary.get("theme", "default"), "w": w, "h": h,
           "scale": args.scale, "summary": summary, "steps": steps}
    (OUT / "session.json").write_text(json.dumps(out, separators=(",", ":"), default=float))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"exported {len(steps)} steps, {sum(1 for s in steps if s['frame'])} frames, {size / 1e6:.1f} MB → {OUT}")


if __name__ == "__main__":
    main()
