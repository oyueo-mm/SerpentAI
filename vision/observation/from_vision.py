"""Detections (+ tracking) → Sensors → obs-v1, using only on-screen information.

The obstacle rays, wall rays, food sectors and opponent features follow the definitions in
src/observation/sensors.ts, measured in pixels and converted with the zoom estimated from the grid.
Nothing here can read the engine: the only inputs are the frame's Detections, the frame timestamp
and the policy's own previous action.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vision.detection.classical import Detections, length_from_gauge
from vision.observation.encode import (FOOD_RANGE, FOOD_SECTORS, OPP_K, OPP_RANGE, RAY_COUNT, RAY_RANGE, RULES, Opponent, Sensors,
                                       encode_obs, radius_for, ray_angle, sector_of)
from vision.tracking.tracker import Odometry, Tracker

TURN_LEVELS = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]


#: Pellet radius correction (px): the saturation threshold clips the anti-aliased rim of a disc, so the
#: equal-area radius of the detected blob is slightly smaller than the drawn one. Fitted on the training
#: dataset by vision/calibration/fit.py (overwritten by vision/calibration/vision-calibration.json).
PELLET_R_BIAS_PX = 0.25


def pellet_value(r_u: float) -> float:
    """Inverse of pelletRadius(v) = 2 + 1.2·√v."""
    return max(0.0, (r_u - 2.0) / 1.2) ** 2


def ray_distances(mask: np.ndarray | None, cx: float, cy: float, heading: float, zoom: float, max_u: float = RAY_RANGE) -> list[float]:
    """Distance (u) along each of the 16 rays from the head centre to the first obstacle pixel."""
    out = [max_u] * RAY_COUNT
    if mask is None:
        return out
    h, w = mask.shape
    steps = np.arange(1.0, max_u * zoom, 1.0)
    for i in range(RAY_COUNT):
        a = heading + ray_angle(i)
        xs = np.round(cx + np.cos(a) * steps).astype(np.int32)
        ys = np.round(cy + np.sin(a) * steps).astype(np.int32)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        hit = np.zeros_like(ok)
        hit[ok] = mask[ys[ok], xs[ok]]
        k = int(np.argmax(hit)) if hit.any() else -1
        if k >= 0:
            out[i] = min(max_u, steps[k] / zoom)
    return out


def wall_distances(rect: tuple[float, float, float, float], cx: float, cy: float, heading: float, zoom: float) -> list[float]:
    """Distance (u) along each ray to the arena boundary (screen rectangle), capped at RAY_RANGE."""
    x0, y0, x1, y1 = rect
    out = []
    for i in range(RAY_COUNT):
        a = heading + ray_angle(i)
        dx, dy = math.cos(a), math.sin(a)
        t = math.inf
        if dx > 1e-9 and math.isfinite(x1):
            t = min(t, (x1 - cx) / dx)
        if dx < -1e-9 and math.isfinite(x0):
            t = min(t, (x0 - cx) / dx)
        if dy > 1e-9 and math.isfinite(y1):
            t = min(t, (y1 - cy) / dy)
        if dy < -1e-9 and math.isfinite(y0):
            t = min(t, (y0 - cy) / dy)
        out.append(min(RAY_RANGE, max(0.0, t / zoom)))
    return out


@dataclass
class VisionObserver:
    """Stateful frame-to-observation converter (zoom smoothing, odometry, opponent tracking)."""
    zoom_alpha: float = 0.3
    zoom: float = math.nan
    odo: Odometry = field(default_factory=Odometry)
    tracker: Tracker = field(default_factory=Tracker)
    last_t: float | None = None
    last_length: float = RULES["startLength"]
    last_heading: float | None = None
    pellet_bias_px: float = PELLET_R_BIAS_PX

    def observe(self, det: Detections, t: float, prev_turn: float, prev_boost: bool, frame_age: float = 0.0) -> tuple[Sensors | None, dict]:
        info: dict = {"reason": det.reason}
        if not det.ok or not np.isfinite(det.zoom):
            return None, info
        # Zoom: smoothed across frames, trusting confident estimates more.
        if not np.isfinite(self.zoom) or abs(det.zoom - self.zoom) / self.zoom > 0.25:
            self.zoom = det.zoom
        else:
            a = self.zoom_alpha * max(0.2, det.zoom_conf)
            self.zoom = (1 - a) * self.zoom + a * det.zoom
        zoom = self.zoom
        cx, cy = det.width / 2, det.height / 2
        heading = float(det.self_heading)
        # Own length/boost from the HUD gauge (fallback: last known).
        length = length_from_gauge(det.hud_fraction) if det.hud_fraction is not None else self.last_length
        self.last_length = length
        radius = radius_for(length)

        # Odometry prior from our own command: speed × dt along the heading.
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        speed = RULES["boostSpeed"] if (prev_boost and length > RULES["boostMinLength"]) else RULES["baseSpeed"]
        prior = np.array([math.cos(heading), math.sin(heading)]) * speed * dt
        cam = self.odo.update(det.grid_phase, zoom, prior)
        self.last_t = t

        ray_body = ray_distances(det.obstacle, cx, cy, heading, zoom)
        ray_wall = wall_distances(det.floor_rect, cx, cy, heading, zoom)

        fx, fy = math.cos(heading), math.sin(heading)
        rx, ry = -fy, fx
        food_sum = [0.0] * FOOD_SECTORS
        food_near = [FOOD_RANGE] * FOOD_SECTORS
        for p in det.pellets:
            dx, dy = (p.x - cx) / zoom, (p.y - cy) / zoom
            d = math.hypot(dx, dy)
            if d >= FOOD_RANGE:
                continue
            s = sector_of(math.atan2(dx * rx + dy * ry, dx * fx + dy * fy))
            food_sum[s] += pellet_value((p.r_px + self.pellet_bias_px) / zoom)
            food_near[s] = min(food_near[s], d)

        meas = []
        for hd in det.heads:
            wx, wy = cam[0] + (hd.x - cx) / zoom, cam[1] + (hd.y - cy) / zoom
            meas.append((wx, wy, hd.hue, hd.radius_px / zoom, hd.heading))
        tracks = self.tracker.update(t, meas)
        opps = []
        for hd, tr in zip(det.heads, tracks):
            dx, dy = (hd.x - cx) / zoom, (hd.y - cy) / zoom
            d = math.hypot(dx, dy)
            if d >= OPP_RANGE:
                continue
            rel = math.atan2(math.sin(hd.heading - heading), math.cos(hd.heading - heading))
            spd = tr.speed if tr is not None and tr.hits >= 3 else RULES["baseSpeed"]
            spd = min(max(spd, 0.6 * RULES["baseSpeed"]), 1.2 * RULES["boostSpeed"])
            opps.append((d, Opponent(dx * fx + dy * fy, dx * rx + dy * ry, rel, spd, hd.radius_px / zoom)))
        opps.sort(key=lambda p: p[0])

        conf = 1.0
        conf *= 0.5 + 0.5 * min(1.0, det.zoom_conf)
        conf *= {"eyes": 1.0, "neck": 0.6}.get(det.self_heading_src, 0.2)
        conf *= 1.0 if det.hud_fraction is not None else 0.8
        s = Sensors(length=length, radius=radius, ray_body=ray_body, ray_wall=ray_wall, food_sum=food_sum, food_near=food_near,
                    opponents=[o for _, o in opps[:OPP_K]], confidence=conf, frame_age=frame_age)
        info.update(zoom=zoom, heading=heading, heading_src=det.self_heading_src, cam=cam.tolist(), tracks=len(self.tracker.tracks), conf=conf)
        self.last_heading = heading
        return s, info

    def obs(self, s: Sensors, prev_turn: float, prev_boost: bool) -> np.ndarray:
        return encode_obs(s, prev_turn, prev_boost)
