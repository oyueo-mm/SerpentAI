"""Python port of the obs-v1 encoder (src/observation/obs.ts) and the Sensors contract
(src/observation/sensors.ts). The Vision pipeline estimates a `Sensors` from pixels and encodes it with
exactly the same arithmetic the engine uses, so the policy network cannot tell the two sources apart.
Parity with the TypeScript encoder is tested on engine fixtures (vision/tests/test_encode.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

OBS_VERSION = "obs-v1"
VIEW_RADIUS = 480.0
RAY_COUNT = 16
RAY_RANGE = 400.0
FOOD_SECTORS = 8
FOOD_RANGE = 400.0
OPP_K = 4
OPP_RANGE = 450.0
SELF_FEATURES = 6
OPP_FEATURES = 7
OBS_SIZE = SELF_FEATURES + RAY_COUNT * 2 + FOOD_SECTORS * 2 + OPP_K * OPP_FEATURES + 2

RULES = dict(startLength=240.0, radiusMax=30.0, boostMinLength=160.0, baseSpeed=150.0, boostSpeed=300.0,
             radiusBase=10.0, radiusExponent=0.35, maxTurnRate=3.2)
TURN_LEVELS = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]


@dataclass
class Opponent:
    x: float           # egocentric, u (+x = own heading)
    y: float           # egocentric, u (+y = 90° clockwise on screen)
    rel_heading: float  # rad
    speed: float       # u/s
    radius: float      # u


@dataclass
class Sensors:
    length: float
    radius: float
    ray_body: list[float]
    ray_wall: list[float]
    food_sum: list[float]
    food_near: list[float]
    opponents: list[Opponent] = field(default_factory=list)
    confidence: float = 1.0
    frame_age: float = 0.0

    @staticmethod
    def from_ts(d: dict) -> "Sensors":
        """From the JSON a TypeScript Sensors serialises to."""
        return Sensors(length=d["length"], radius=d["radius"], ray_body=list(d["rayBody"]), ray_wall=list(d["rayWall"]),
                       food_sum=list(d["foodSum"]), food_near=list(d["foodNear"]),
                       opponents=[Opponent(o["x"], o["y"], o["relHeading"], o["speed"], o["radius"]) for o in d["opponents"]],
                       confidence=d["confidence"], frame_age=d["frameAge"])


def radius_for(length: float) -> float:
    R = RULES
    return min(R["radiusMax"], R["radiusBase"] * (max(length, 1.0) / R["startLength"]) ** R["radiusExponent"])


def ray_angle(i: int) -> float:
    return i * 2 * math.pi / RAY_COUNT


def sector_of(phi: float) -> int:
    w = 2 * math.pi / FOOD_SECTORS
    a = math.fmod(phi + w / 2, 2 * math.pi)
    if a < 0:
        a += 2 * math.pi
    return min(FOOD_SECTORS - 1, int(math.floor(a / w)))


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def encode_obs(s: Sensors, prev_turn: float, prev_boost: bool) -> np.ndarray:
    R = RULES
    o = np.zeros(OBS_SIZE, dtype=np.float32)
    k = 0

    def put(v: float) -> None:
        nonlocal k
        o[k] = v
        k += 1

    put(_clip(math.log(max(s.length, 1) / R["startLength"]) / math.log(50), -1, 1))
    put(_clip(s.radius / R["radiusMax"], 0, 1))
    put(1.0 if s.length > R["boostMinLength"] else 0.0)
    put(_clip(prev_turn, -1, 1))
    put(1.0 if prev_boost else 0.0)
    put(_clip((s.length - R["boostMinLength"]) / 200, 0, 1))
    for i in range(RAY_COUNT):
        put(_clip(s.ray_body[i] / RAY_RANGE, 0, 1))
        put(_clip(s.ray_wall[i] / RAY_RANGE, 0, 1))
    for i in range(FOOD_SECTORS):
        put(_clip(math.log1p(max(0.0, s.food_sum[i])) / math.log1p(200), 0, 1))
        put(_clip(s.food_near[i] / FOOD_RANGE, 0, 1))
    span = R["boostSpeed"] - R["baseSpeed"]
    for j in range(OPP_K):
        if j >= len(s.opponents):
            k += OPP_FEATURES
            continue
        p = s.opponents[j]
        put(1.0)
        put(_clip(p.x / OPP_RANGE, -1, 1))
        put(_clip(p.y / OPP_RANGE, -1, 1))
        put(math.cos(p.rel_heading))
        put(math.sin(p.rel_heading))
        put(_clip((p.speed - R["baseSpeed"]) / span, -1, 2))
        put(_clip(math.log2(max(p.radius, 1e-3) / max(s.radius, 1e-3)) / 2, -1, 1))
    put(_clip(s.confidence, 0, 1))
    put(_clip(s.frame_age, 0, 1))
    assert k == OBS_SIZE
    return o


def feature_names() -> list[str]:
    n = ["self.log_length", "self.radius", "self.can_boost", "self.prev_turn", "self.prev_boost", "self.boost_fuel"]
    for i in range(RAY_COUNT):
        n += [f"ray{i}.body", f"ray{i}.wall"]
    for i in range(FOOD_SECTORS):
        n += [f"food{i}.sum", f"food{i}.near"]
    for j in range(OPP_K):
        n += [f"opp{j}.present", f"opp{j}.x", f"opp{j}.y", f"opp{j}.cos", f"opp{j}.sin", f"opp{j}.speed", f"opp{j}.radius_ratio"]
    return n + ["conf.confidence", "conf.frame_age"]


#: Index groups used by error analyses.
IDX_SELF = list(range(0, 6))
IDX_RAY_BODY = list(range(6, 38, 2))
IDX_RAY_WALL = list(range(7, 38, 2))
IDX_FOOD_SUM = list(range(38, 54, 2))
IDX_FOOD_NEAR = list(range(39, 54, 2))
IDX_OPP = list(range(54, 82))
