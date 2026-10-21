"""Classical computer-vision baseline: one RGB frame of the SerpentAI canvas → detections.

Only what is drawn on screen is used (docs/VISION.md):
- scale (px per world unit) from the period of the floor grid (lines every 100 u);
- arena walls as an axis-aligned rectangle (the camera never rotates);
- snakes and pellets from saturated pixels split into hue classes; small round blobs are pellets;
- heads from eye pairs, found as small enclosed holes in the saturated mask;
- own length and boost availability from the canvas HUD gauge.
The own head is at the screen centre because the game camera follows it.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

GRID_SPACING = 100.0
HUD = dict(margin=16, barW=180, barH=10, dotR=6)
HUD_FILL = np.array([14, 165, 233])
HUD_TRACK = np.array([226, 232, 240])
HUD_BOOST_ON = np.array([16, 185, 129])
HUD_BOOST_OFF = np.array([148, 163, 184])
SAT_MIN = 90      # HSV saturation (0–255) above which a pixel is a snake/pellet colour
VAL_MIN = 40


@dataclass
class Head:
    x: float              # screen px
    y: float
    heading: float        # rad, screen axes (= world axes)
    radius_px: float
    hue: float
    eye_dist: float
    conf: float


@dataclass
class Pellet:
    x: float
    y: float
    r_px: float


@dataclass
class Detections:
    ok: bool
    reason: str
    width: int
    height: int
    zoom: float = math.nan            # px per world unit
    zoom_conf: float = 0.0
    grid_phase: tuple[float, float] = (math.nan, math.nan)  # px, phase of the grid in x / y (mod period)
    self_heading: float | None = None
    self_heading_src: str = "none"    # 'eyes' | 'neck' | 'none'
    self_radius_px: float = math.nan
    self_hue: float = math.nan
    hud_fraction: float | None = None
    can_boost: bool | None = None
    #: Arena floor rectangle in screen px: (x0, y0, x1, y1); a side is ±inf when the wall is not visible.
    floor_rect: tuple[float, float, float, float] = (-math.inf, -math.inf, math.inf, math.inf)
    pellets: list[Pellet] = field(default_factory=list)
    heads: list[Head] = field(default_factory=list)
    obstacle: np.ndarray | None = None   # bool mask of other snakes' bodies
    self_mask: np.ndarray | None = None
    timings: dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------ HUD
def read_hud(rgb: np.ndarray) -> tuple[float | None, bool | None]:
    h = rgb.shape[0]
    m, bw, bh = HUD["margin"], HUD["barW"], HUD["barH"]
    y = h - m - bh // 2 - 1
    if y < 0 or rgb.shape[1] < m + bw + 30:
        return None, None
    row = rgb[y, m:m + bw].astype(np.int32)
    d_fill = np.abs(row - HUD_FILL).sum(axis=1)
    d_track = np.abs(row - HUD_TRACK).sum(axis=1)
    known = (d_fill < 25) | (d_track < 25)
    if known.mean() < 0.8:
        return None, None  # HUD covered or not drawn
    frac = float(((d_fill < d_track) & (d_fill < 25)).sum()) / bw
    dot = rgb[y, m + bw + 14].astype(np.int32)
    on, off = np.abs(dot - HUD_BOOST_ON).sum(), np.abs(dot - HUD_BOOST_OFF).sum()
    can_boost = bool(on < off) if min(on, off) < 90 else None
    return frac, can_boost


def length_from_gauge(f: float) -> float:
    return 120.0 * 100.0 ** f


# ------------------------------------------------------------------ grid scale
def _period(profile: np.ndarray, lo: float, hi: float) -> tuple[float, float, float]:
    """Dominant period of a 1-D profile in [lo, hi] px: (period, strength, phase px)."""
    x = profile - np.convolve(profile, np.ones(31) / 31, mode="same")
    x[:15] = 0
    x[-15:] = 0
    n = len(x)
    if n < 3 * lo:
        return math.nan, 0.0, math.nan
    ac = np.correlate(x, x, mode="full")[n - 1:]
    if ac[0] <= 0:
        return math.nan, 0.0, math.nan
    ac = ac / ac[0]
    lo_i, hi_i = int(max(2, lo)), int(min(n // 2, hi))
    if hi_i <= lo_i + 2:
        return math.nan, 0.0, math.nan
    seg = ac[lo_i:hi_i]
    k = int(np.argmax(seg)) + lo_i
    strength = float(ac[k])
    # Sub-pixel refinement (parabola through the peak), then average with the 2nd harmonic when present.
    if 1 <= k < len(ac) - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        den = a - 2 * b + c
        p = k + (0.5 * (a - c) / den if den != 0 else 0.0)
    else:
        p = float(k)
    k2 = int(round(2 * p))
    if k2 + 1 < len(ac) and ac[k2] > 0.3 * strength:
        a, b, c = ac[k2 - 1], ac[k2], ac[k2 + 1]
        den = a - 2 * b + c
        p2 = k2 + (0.5 * (a - c) / den if den != 0 else 0.0)
        p = (p + p2 / 2) / 2
    # Phase: position of the grid lines (dark dips) modulo the period.
    idx = np.arange(n)
    z = np.sum(-x * np.exp(-2j * np.pi * idx / p))
    phase = (np.angle(z) / (2 * np.pi) * p) % p
    return float(p), strength, float(phase)


def estimate_scale(v: np.ndarray, valid: np.ndarray, zoom_range: tuple[float, float]) -> tuple[float, float, tuple[float, float]]:
    """Zoom (px/u) from the floor grid period, its confidence, and the grid phase (px) in x and y."""
    lo, hi = GRID_SPACING * zoom_range[0], GRID_SPACING * zoom_range[1]
    vv = np.where(valid, v, 0.0)
    cnt_c, cnt_r = valid.sum(axis=0), valid.sum(axis=1)
    colp = vv.sum(axis=0) / np.maximum(cnt_c, 1)
    rowp = vv.sum(axis=1) / np.maximum(cnt_r, 1)
    # Columns/rows with too few floor pixels get the mean value (no spurious dips).
    colp[cnt_c < 10] = colp[cnt_c >= 10].mean() if (cnt_c >= 10).any() else 0.0
    rowp[cnt_r < 10] = rowp[cnt_r >= 10].mean() if (cnt_r >= 10).any() else 0.0
    px, sx, phx = _period(colp, lo, hi)
    py, sy, phy = _period(rowp, lo, hi)
    cands = [(p, s) for p, s in ((px, sx), (py, sy)) if np.isfinite(p) and s > 0.15]
    if not cands:
        return math.nan, 0.0, (math.nan, math.nan)
    if len(cands) == 2 and abs(px - py) / max(px, py) > 0.05:
        p = cands[0][0] if sx >= sy else cands[1][0]
        conf = 0.4 * max(sx, sy)
    else:
        w = sum(s for _, s in cands)
        p = sum(pp * s for pp, s in cands) / w
        conf = min(1.0, w / len(cands) + (0.2 if len(cands) == 2 else 0.0))
    return p / GRID_SPACING, float(min(1.0, conf)), (phx, phy)


def zoom_prior(width: int, height: int) -> tuple[float, float]:
    """The game camera shows a circle of 480–600 u around the head on the shorter screen side."""
    m = min(width, height)
    return 0.9 * m / (2 * 600), 1.1 * m / (2 * 480)


# ------------------------------------------------------------------ walls
def estimate_floor_rect(wall_like: np.ndarray, sat: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned arena floor rectangle from columns/rows that are mostly wall-coloured."""
    usable = ~sat
    h, w = wall_like.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        colf = (wall_like & usable).sum(axis=0) / np.maximum(1, usable.sum(axis=0))
        rowf = (wall_like & usable).sum(axis=1) / np.maximum(1, usable.sum(axis=1))
    def edge(frac: np.ndarray, from_start: bool) -> float:
        wallish = frac > 0.6
        if from_start:
            if not wallish[0]:
                return -math.inf
            k = int(np.argmin(wallish)) if not wallish.all() else len(frac)
            return float(k)
        if not wallish[-1]:
            return math.inf
        k = int(np.argmin(wallish[::-1])) if not wallish.all() else len(frac)
        return float(len(frac) - k)
    return edge(colf, True), edge(rowf, True), edge(colf, False), edge(rowf, False)


# ------------------------------------------------------------------ main
def detect(rgb: np.ndarray, zoom_range: tuple[float, float] | None = None) -> Detections:
    t0 = time.perf_counter()
    h, w = rgb.shape[:2]
    det = Detections(ok=False, reason="", width=w, height=h)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0].astype(np.float32), hsv[..., 1], hsv[..., 2]
    hud_mask = np.zeros((h, w), bool)
    m = HUD["margin"]
    hud_mask[max(0, h - m - HUD["barH"] - 10):, : m + HUD["barW"] + 30] = True
    det.hud_fraction, det.can_boost = read_hud(rgb)
    sat = (S > SAT_MIN) & (V > VAL_MIN) & ~hud_mask
    det.timings["masks"] = time.perf_counter() - t0

    # Floor brightness and wall-like pixels (unsaturated and clearly darker than the floor).
    floor_v = float(np.percentile(V[~sat], 75)) if (~sat).any() else 255.0
    dark = (~sat) & (V < floor_v - 22)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    det.floor_rect = estimate_floor_rect(dark, sat)
    x0, y0, x1, y1 = det.floor_rect
    inside = np.zeros((h, w), bool)
    inside[int(max(0, y0)) + 2: int(min(h, y1)) - 2, int(max(0, x0)) + 2: int(min(w, x1)) - 2] = True

    # Scale from the grid inside the floor, away from saturated pixels.
    sat_d = cv2.dilate(sat.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    det.zoom, det.zoom_conf, det.grid_phase = estimate_scale(V.astype(np.float32), inside & ~sat_d & ~hud_mask, zoom_range or zoom_prior(w, h))
    det.timings["scale"] = time.perf_counter() - t0
    if not np.isfinite(det.zoom):
        det.reason = "grid_not_found"
        return det

    # Hue classes: peaks of the hue histogram of saturated pixels (circular, 180 bins).
    ys, xs = np.nonzero(sat)
    hues = H[ys, xs]
    if hues.size < 50:
        det.reason = "no_snake_pixels"
        return det
    hist = np.bincount(hues.astype(np.int32), minlength=180).astype(np.float64)
    sm = np.convolve(np.concatenate([hist[-4:], hist, hist[:4]]), np.ones(5) / 5, mode="same")[4:-4]
    peaks = [i for i in range(180) if sm[i] >= sm[i - 1] and sm[i] > sm[(i + 1) % 180] and sm[i] > max(30, 0.002 * hues.size)]
    peaks = np.array(peaks if peaks else [int(np.argmax(sm))], dtype=np.float32)
    dh = np.abs(hues[:, None] - peaks[None, :])
    dh = np.minimum(dh, 180 - dh)
    cls = np.full((h, w), -1, dtype=np.int32)
    cls[ys, xs] = np.argmin(dh, axis=1)
    det.timings["hue"] = time.perf_counter() - t0

    # Components per hue class (inside the class bounding box); pellets vs bodies by size and roundness.
    zoom = det.zoom
    pellet_max_px = (2 + 1.2 * math.sqrt(20)) * zoom + 1.5
    body = np.zeros((h, w), bool)
    dist = cv2.distanceTransform(sat.astype(np.uint8), cv2.DIST_L2, 3)
    pix_cls = cls[ys, xs]
    sat_ii = cv2.integral(sat.astype(np.uint8))
    for c in range(len(peaks)):
        sel = pix_cls == c
        if not sel.any():
            continue
        cy0, cy1, cx0, cx1 = ys[sel].min(), ys[sel].max() + 1, xs[sel].min(), xs[sel].max() + 1
        mc = (cls[cy0:cy1, cx0:cx1] == c).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(mc, connectivity=8)
        area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        r_eq = np.sqrt(area / math.pi)
        extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        pellet = (r_eq <= pellet_max_px) & (extent <= 2.6 * r_eq + 2) & (area < math.pi * (pellet_max_px + 1) ** 2)
        pellet[0] = False
        for i in np.flatnonzero(pellet):
            px_, py_ = float(cents[i][0] + cx0), float(cents[i][1] + cy0)
            # A pellet is an isolated disc on the floor: few saturated pixels around it besides its own.
            # (Small fragments at the edge of a body fall in a neighbouring hue class and fail this test.)
            R = int(r_eq[i] + 3)
            ya, yb, xa, xb = max(0, int(py_) - R), min(h, int(py_) + R + 1), max(0, int(px_) - R), min(w, int(px_) + R + 1)
            around = sat_ii[yb, xb] - sat_ii[ya, xb] - sat_ii[yb, xa] + sat_ii[ya, xa] - area[i]
            ring = (yb - ya) * (xb - xa) - area[i]
            if ring > 0 and around / ring > 0.25:
                continue
            det.pellets.append(Pellet(px_, py_, float(r_eq[i])))
        is_body = ~pellet
        is_body[0] = False
        if is_body.any():
            body[cy0:cy1, cx0:cx1] |= is_body[lab]
    det.timings["components"] = time.perf_counter() - t0

    # Own body: the hue class under the screen centre.
    cx, cy = w / 2, h / 2
    ci, cj = int(cy), int(cx)
    win = cls[max(0, ci - 4): ci + 5, max(0, cj - 4): cj + 5]
    vals = win[win >= 0]
    if vals.size == 0:
        det.reason = "self_not_at_centre"
        return det
    self_cls = int(np.bincount(vals).argmax())
    det.self_hue = float(peaks[self_cls])
    self_mask = body & (cls == self_cls)
    det.self_mask = self_mask
    det.obstacle = body & ~self_mask
    det.self_radius_px = float(dist[ci, cj]) if sat[ci, cj] else math.nan

    # Eyes = small enclosed holes in the saturated mask.
    nsat = (~sat).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(nsat, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    holes = []
    for i in range(1, n):
        if i in border:
            continue
        area = stats[i, cv2.CC_STAT_AREA]
        if 2 <= area <= 900:
            holes.append((float(cents[i][0]), float(cents[i][1]), math.sqrt(area / math.pi)))
    det.timings["holes"] = time.perf_counter() - t0
    heads = pair_eyes(holes, body, cls, H, dist)
    det.heads = heads

    # Own heading: eye pair whose head centre is nearest the screen centre.
    own = [hd for hd in heads if math.hypot(hd.x - cx, hd.y - cy) < max(6.0, 1.2 * hd.radius_px)]
    if own:
        best = min(own, key=lambda hd: math.hypot(hd.x - cx, hd.y - cy))
        det.self_heading, det.self_heading_src = best.heading, "eyes"
        det.self_radius_px = best.radius_px
        det.heads = [hd for hd in heads if hd is not best]
    else:
        nh = neck_heading(self_mask, cx, cy, det.self_radius_px if np.isfinite(det.self_radius_px) else 8 * zoom)
        if nh is not None:
            det.self_heading, det.self_heading_src = nh, "neck"
    det.ok = det.self_heading is not None
    det.reason = "" if det.ok else "self_heading_unknown"
    det.timings["total"] = time.perf_counter() - t0
    return det


def pair_eyes(holes: list[tuple[float, float, float]], body: np.ndarray, cls: np.ndarray, H: np.ndarray, dist: np.ndarray) -> list[Head]:
    """Pairs eye holes into heads. Eye centres sit at ±0.46 r beside the axis, 0.38 r ahead of the head centre."""
    h, w = body.shape
    used: set[int] = set()
    heads: list[Head] = []
    order = sorted(range(len(holes)), key=lambda i: -holes[i][2])
    for a in order:
        if a in used:
            continue
        xa, ya, ra = holes[a]
        best, bd = -1, math.inf
        for b in order:
            if b == a or b in used:
                continue
            xb, yb, rb = holes[b]
            d = math.hypot(xb - xa, yb - ya)
            rm = (ra + rb) / 2
            if 1.6 * rm <= d <= 4.5 * rm and abs(ra - rb) <= 0.6 * max(ra, rb) + 1 and d < bd:
                best, bd = b, d
        if best < 0:
            continue
        xb, yb, rb = holes[best]
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        r_px = bd / 0.92
        ex, ey = (xb - xa) / bd, (yb - ya) / bd
        nx, ny = -ey, ex  # one perpendicular; decide the forward sign from where the body is
        sc = 0
        for sgn in (1, -1):
            cnt = 0
            for k in range(1, 7):
                px_, py_ = int(mx + sgn * nx * r_px * 0.6 * k), int(my + sgn * ny * r_px * 0.6 * k)
                if 0 <= px_ < w and 0 <= py_ < h and body[py_, px_]:
                    cnt += 1
            sc += sgn * cnt
        if sc == 0:
            continue
        fx, fy = (-nx, -ny) if sc > 0 else (nx, ny)   # forward points away from the body
        hx, hy = mx - fx * 0.38 * r_px, my - fy * 0.38 * r_px
        ix, iy = int(round(hx)), int(round(hy))
        if not (0 <= ix < w and 0 <= iy < h) or not body[iy, ix]:
            continue
        used.update((a, best))
        heads.append(Head(x=hx, y=hy, heading=math.atan2(fy, fx), radius_px=max(r_px, float(dist[iy, ix])), hue=float(H[iy, ix]), eye_dist=bd,
                          conf=min(1.0, abs(sc) / 6)))
    return heads


def neck_heading(self_mask: np.ndarray, cx: float, cy: float, r_px: float) -> float | None:
    """Fallback heading: opposite of the mean direction of own-body pixels in an annulus behind the head."""
    h, w = self_mask.shape
    R = int(max(6, 4 * r_px))
    y0, y1, x0, x1 = max(0, int(cy) - R), min(h, int(cy) + R + 1), max(0, int(cx) - R), min(w, int(cx) + R + 1)
    ys, xs = np.nonzero(self_mask[y0:y1, x0:x1])
    if len(xs) < 10:
        return None
    dx, dy = xs + x0 - cx, ys + y0 - cy
    d = np.hypot(dx, dy)
    sel = (d > 1.5 * r_px) & (d < 4 * r_px)
    if sel.sum() < 5:
        return None
    return math.atan2(-dy[sel].mean(), -dx[sel].mean())
