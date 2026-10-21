"""Multi-object tracking of opponent heads and visual odometry of the camera.

Positions are tracked in an "odometry frame": world units relative to wherever the camera was when
tracking started. The camera displacement between frames is measured from the shift of the floor grid
(phase change of the 100 u grid, unwrapped with the policy's own motion prior), so the tracker never
needs the true camera position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GRID = 100.0


@dataclass
class Track:
    id: int
    x: np.ndarray            # state [px, py, vx, vy] in odometry-frame u, u/s
    P: np.ndarray            # covariance
    hue: float
    radius: float            # u
    heading: float           # rad (from the last eye detection)
    hits: int = 1
    misses: int = 0
    last_t: float = 0.0

    @property
    def speed(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))


@dataclass
class Odometry:
    """Camera position in the odometry frame, from grid phase shifts."""
    cam: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_phase: tuple[float, float] | None = None
    last_zoom: float = math.nan
    fails: int = 0

    def update(self, phase: tuple[float, float], zoom: float, prior: np.ndarray) -> np.ndarray:
        """Advances the camera by the grid shift; `prior` is the expected displacement (u) from own motion."""
        if self.last_phase is None or not all(np.isfinite(phase)) or not np.isfinite(zoom):
            self.cam = self.cam + prior
            self.last_phase, self.last_zoom = phase, zoom
            self.fails += 0 if self.last_phase is None else 1
            return self.cam
        d = np.zeros(2)
        for k in range(2):
            # Grid lines move by −Δcam·zoom on screen: Δcam = −Δphase / zoom (mod 100 u), unwrapped near the prior.
            raw = -(phase[k] - self.last_phase[k]) / zoom
            n = round((prior[k] - raw) / GRID)
            d[k] = raw + n * GRID
            if abs(d[k] - prior[k]) > 0.6 * GRID:
                d[k] = prior[k]
        self.cam = self.cam + d
        self.last_phase, self.last_zoom = phase, zoom
        return self.cam


class Tracker:
    def __init__(self, gate: float = 70.0, hue_gate: float = 18.0, max_misses: int = 4):
        self.tracks: list[Track] = []
        self.next_id = 0
        self.gate, self.hue_gate, self.max_misses = gate, hue_gate, max_misses
        self.q = 400.0   # process noise (accel², u²/s⁴ scale)
        self.r = 9.0     # measurement noise (u²)
        self.stats = {"created": 0, "lost": 0, "matched": 0}

    def predict(self, t: float) -> None:
        for tr in self.tracks:
            dt = max(1e-3, t - tr.last_t)
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
            tr.x = F @ tr.x
            tr.P = F @ tr.P @ F.T + G @ G.T * self.q
            tr.last_t = t

    def update(self, t: float, meas: list[tuple[float, float, float, float, float]]) -> list[Track | None]:
        """meas: (x, y, hue, radius_u, heading) in the odometry frame. Returns the track for each measurement."""
        self.predict(t)
        assigned: list[Track | None] = [None] * len(meas)
        pairs = []
        for i, (mx, my, mh, _, _) in enumerate(meas):
            for tr in self.tracks:
                d = math.hypot(mx - tr.x[0], my - tr.x[1])
                dh = min(abs(mh - tr.hue), 180 - abs(mh - tr.hue))
                if d < self.gate and dh < self.hue_gate:
                    pairs.append((d + 2 * dh, i, tr))
        pairs.sort(key=lambda p: p[0])
        used_t: set[int] = set()
        for _, i, tr in pairs:
            if assigned[i] is not None or tr.id in used_t:
                continue
            mx, my, mh, rad, hd = meas[i]
            Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
            S = Hm @ tr.P @ Hm.T + np.eye(2) * self.r
            K = tr.P @ Hm.T @ np.linalg.inv(S)
            tr.x = tr.x + K @ (np.array([mx, my]) - Hm @ tr.x)
            tr.P = (np.eye(4) - K @ Hm) @ tr.P
            tr.hue, tr.radius, tr.heading = 0.7 * tr.hue + 0.3 * mh, 0.7 * tr.radius + 0.3 * rad, hd
            tr.hits += 1
            tr.misses = 0
            assigned[i] = tr
            used_t.add(tr.id)
            self.stats["matched"] += 1
        for i, a in enumerate(assigned):
            if a is None:
                mx, my, mh, rad, hd = meas[i]
                # New track: initial velocity along the eye heading at cruise speed (most snakes cruise).
                tr = Track(self.next_id, np.array([mx, my, 150 * math.cos(hd), 150 * math.sin(hd)]), np.diag([9.0, 9.0, 2500.0, 2500.0]), mh, rad, hd, last_t=t)
                self.next_id += 1
                self.tracks.append(tr)
                assigned[i] = tr
                self.stats["created"] += 1
        for tr in self.tracks:
            if tr.id not in used_t and tr.hits > 0 and tr not in assigned:
                tr.misses += 1
        lost = [tr for tr in self.tracks if tr.misses > self.max_misses]
        self.stats["lost"] += len(lost)
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
        return assigned
