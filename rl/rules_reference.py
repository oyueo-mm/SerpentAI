"""Pure-Python reference implementation of rules-v1 (Method B of docs/REINFORCEMENT_LEARNING.md).

It is a line-by-line port of src/sim/world.ts and src/sim/body.ts without the spatial grid (brute-force
candidate search; outcomes do not depend on iteration order because every tie is broken by id). It
exists to (1) cross-check the TypeScript engine against golden trajectories with explicit tolerances
and (2) measure what training on a separate Python engine would cost. Training uses the TypeScript
engine through the bridge (Method A); see the throughput comparison in docs/BENCHMARK.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

M32 = 0xFFFFFFFF

DEFAULT_RULES = dict(
    worldSize=3000, tickRate=30, baseSpeed=150, boostSpeed=300, maxTurnRate=3.2, startLength=240, minLength=120,
    boostMinLength=160, boostCost=45, boostDropFraction=0.5, boostPelletValue=6, radiusBase=10, radiusExponent=0.35,
    radiusMax=30, pathSpacing=6, maxSubstepDistance=5, foodTarget=1200, foodSpawnPerTick=4, foodValueMin=2,
    foodValueMax=6, eatReach=6, deathDropFraction=0.6, deathPelletSpacingMin=8, gridCell=96,
)


def imul(a: int, b: int) -> int:
    return (a * b) & M32


class Rng:
    """sfc32 with splitmix32 seeding — bit-identical to src/sim/rng.ts."""

    def __init__(self, seed: int | list[int]):
        if isinstance(seed, int):
            s = seed & M32
            words = []
            for _ in range(4):
                s = (s + 0x9E3779B9) & M32
                z = s
                z = imul(z ^ (z >> 16), 0x85EBCA6B)
                z = imul(z ^ (z >> 13), 0xC2B2AE35)
                words.append((z ^ (z >> 16)) & M32)
            self.a, self.b, self.c, self.d = words
            for _ in range(12):
                self.next_u32()
        else:
            self.a, self.b, self.c, self.d = [w & M32 for w in seed]

    def next_u32(self) -> int:
        t = (((self.a + self.b) & M32) + self.d) & M32
        self.d = (self.d + 1) & M32
        self.a = (self.b ^ (self.b >> 9)) & M32
        self.b = (self.c + ((self.c << 3) & M32)) & M32
        self.c = (((self.c << 21) & M32) | (self.c >> 11)) & M32
        self.c = (self.c + t) & M32
        return t

    def next(self) -> float:
        return self.next_u32() / 4294967296

    def range(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.next()

    def int(self, lo: int, hi: int) -> int:
        return lo + math.floor(self.next() * (hi - lo + 1))

    @staticmethod
    def derive(seed: int, stream: int) -> "Rng":
        return Rng((imul(seed & M32, 0x9E3779B1) ^ imul((stream + 1) & M32, 0x85EBCA77)) & M32)


TAU = math.pi * 2


def wrap_angle(a: float) -> float:
    x = math.fmod(a, TAU)
    if x <= -math.pi:
        x += TAU
    elif x > math.pi:
        x -= TAU
    return x


def dist_point_seg_sq(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = ((px - ax) * dx + (py - ay) * dy) / l2 if l2 > 0 else 0.0
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    cx, cy = ax + t * dx - px, ay + t * dy - py
    return cx * cx + cy * cy


def closest_approach_sq(a0x, a0y, a1x, a1y, b0x, b0y, b1x, b1y) -> float:
    px, py = a0x - b0x, a0y - b0y
    vx, vy = (a1x - a0x) - (b1x - b0x), (a1y - a0y) - (b1y - b0y)
    vv = vx * vx + vy * vy
    t = min(1.0, max(0.0, -(px * vx + py * vy) / vv)) if vv > 1e-12 else 0.0
    cx, cy = px + vx * t, py + vy * t
    return cx * cx + cy * cy


def integrate_arc(x, y, theta, v, omega, h):
    th1 = theta + omega * h
    if abs(omega * h) < 1e-9:
        thm = theta + 0.5 * omega * h
        nx, ny = x + v * h * math.cos(thm), y + v * h * math.sin(thm)
    else:
        k = v / omega
        nx = x + k * (math.sin(th1) - math.sin(theta))
        ny = y - k * (math.cos(th1) - math.cos(theta))
    return nx, ny, wrap_angle(th1)


def radius_for(R, L) -> float:
    return min(R["radiusMax"], R["radiusBase"] * (max(L, 1) / R["startLength"]) ** R["radiusExponent"])


def pellet_radius(v) -> float:
    return 2 + 1.2 * math.sqrt(v)


class Body:
    def __init__(self):
        self.pts: list[float] = []
        self.chain = 0.0

    def init_straight(self, hx, hy, th, length, spacing):
        self.pts, self.chain = [], 0.0
        bx, by = -math.cos(th), -math.sin(th)
        d, prev = length, -1.0
        while d > 1e-9:
            self.pts += [hx + bx * d, hy + by * d]
            if prev >= 0:
                self.chain += prev - d
            prev = d
            d -= spacing

    @property
    def n(self) -> int:
        return len(self.pts) // 2

    def total(self, hx, hy) -> float:
        return self.chain + math.hypot(hx - self.pts[-2], hy - self.pts[-1])

    def advance(self, hx, hy, spacing):
        d = math.hypot(hx - self.pts[-2], hy - self.pts[-1])
        if d >= spacing:
            self.pts += [hx, hy]
            self.chain += d

    def trim(self, hx, hy, length):
        total = self.total(hx, hy)
        while total > length:
            excess = total - length
            if self.n >= 2:
                x0, y0, x1, y1 = self.pts[0], self.pts[1], self.pts[2], self.pts[3]
                s0 = math.hypot(x1 - x0, y1 - y0)
                if s0 <= excess:
                    del self.pts[:2]
                    self.chain -= s0
                    total -= s0
                    continue
                f = excess / s0
                self.pts[0], self.pts[1] = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
                self.chain -= excess
            else:
                x0, y0 = self.pts[0], self.pts[1]
                s0 = math.hypot(hx - x0, hy - y0)
                f = min(1.0, excess / s0) if s0 > 0 else 0.0
                self.pts[0], self.pts[1] = x0 + (hx - x0) * f, y0 + (hy - y0) * f
            break


@dataclass
class Snake:
    id: int
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    length: float = 0.0
    alive: bool = True
    boosting: bool = False
    speed: float = 0.0
    drop_acc: float = 0.0
    death_cause: str | None = None
    killer: int = -1
    kills: int = 0
    eaten_tick: float = 0.0
    body: Body = field(default_factory=Body)


class World:
    def __init__(self, rules: dict | None = None, seed: int = 0, fill_food: bool = True):
        self.R = {**DEFAULT_RULES, **(rules or {})}
        R = self.R
        self.substeps = max(1, math.ceil(R["boostSpeed"] / R["tickRate"] / R["maxSubstepDistance"] - 1e-9))
        self.dt = 1 / R["tickRate"]
        self.tick = 0
        self.snakes: list[Snake] = []
        self.food: dict[int, list] = {}  # id -> [x, y, value, r]  (dict keeps insertion = id order)
        self.next_food_id = 0
        self.food_rng = Rng.derive(seed, 1)
        self.spawn_rng = Rng.derive(seed, 2)
        self.deaths: list[dict] = []
        if fill_food:
            while len(self.food) < R["foodTarget"]:
                self._spawn_natural()

    def add_pellet(self, x, y, value):
        W = self.R["worldSize"]
        r = pellet_radius(value)
        self.food[self.next_food_id] = [min(W - r, max(r, x)), min(W - r, max(r, y)), value, r]
        self.next_food_id += 1

    def _spawn_natural(self):
        W, m = self.R["worldSize"], 20
        x = self.food_rng.range(m, W - m)
        y = self.food_rng.range(m, W - m)
        v = self.food_rng.int(self.R["foodValueMin"], self.R["foodValueMax"])
        self.add_pellet(x, y, v)

    def add_snake(self, x, y, heading, length=None):
        R = self.R
        s = Snake(id=len(self.snakes))
        s.x, s.y, s.heading = x, y, wrap_angle(heading)
        s.length = R["startLength"] if length is None else length
        s.speed = R["baseSpeed"]
        s.body.init_straight(x, y, s.heading, s.length, R["pathSpacing"])
        self.snakes.append(s)
        return s

    def segments(self):
        out = []
        for s in self.snakes:
            if not s.alive:
                continue
            r = radius_for(self.R, s.length)
            p = s.body.pts
            n = s.body.n
            for i in range(n - 1):
                out.append((s.id, p[2 * i], p[2 * i + 1], p[2 * i + 2], p[2 * i + 3], r))
            out.append((s.id, p[-2], p[-1], s.x, s.y, r))
        return out

    def step(self, controls):
        R = self.R
        h = self.dt / self.substeps
        self.deaths = []
        omegas = []
        for s in self.snakes:
            s.eaten_tick = 0.0
            if not s.alive:
                omegas.append(0.0)
                continue
            c = controls[s.id] if s.id < len(controls) else None
            w = 0.0
            if c and c.get("targetHeading") is not None:
                w = wrap_angle(c["targetHeading"] - s.heading) / self.dt
            elif c and c.get("turn") is not None:
                w = max(-1.0, min(1.0, c["turn"])) * R["maxTurnRate"]
            omegas.append(max(-R["maxTurnRate"], min(R["maxTurnRate"], w)))
        n = len(self.snakes)
        px, py = [0.0] * n, [0.0] * n
        for _ in range(self.substeps):
            for s in self.snakes:
                if not s.alive:
                    continue
                px[s.id], py[s.id] = s.x, s.y
                c = controls[s.id] if s.id < len(controls) else None
                boost = bool(c and c.get("boost")) and s.length > R["boostMinLength"]
                s.boosting = boost
                s.speed = R["boostSpeed"] if boost else R["baseSpeed"]
                s.x, s.y, s.heading = integrate_arc(s.x, s.y, s.heading, s.speed, omegas[s.id], h)
                s.body.advance(s.x, s.y, R["pathSpacing"])
                if boost:
                    new_l = max(R["boostMinLength"], s.length - R["boostCost"] * h)
                    s.drop_acc += (s.length - new_l) * R["boostDropFraction"]
                    s.length = new_l
                s.body.trim(s.x, s.y, s.length)
                while s.drop_acc >= R["boostPelletValue"]:
                    s.drop_acc -= R["boostPelletValue"]
                    jr = radius_for(R, s.length) * 0.5
                    tx, ty = s.body.pts[0], s.body.pts[1]
                    jx = self.food_rng.range(-jr, jr)
                    jy = self.food_rng.range(-jr, jr)
                    self.add_pellet(tx + jx, ty + jy, R["boostPelletValue"])
            segs = self.segments()
            dying = self._detect(px, py, segs)
            self._eat(px, py, {d["id"] for d in dying})
            for d in dying:
                self._kill(d)
        self.tick += 1
        spawn = R["foodSpawnPerTick"]
        while spawn > 0 and len(self.food) < R["foodTarget"]:
            self._spawn_natural()
            spawn -= 1

    def _detect(self, px, py, segs):
        R, W = self.R, self.R["worldSize"]
        alive = [s for s in self.snakes if s.alive]
        rad = {s.id: radius_for(R, s.length) for s in alive}
        head_on: dict[int, list[int]] = {}
        for a in range(len(alive)):
            for b in range(a + 1, len(alive)):
                i, j = alive[a], alive[b]
                thr = rad[i.id] + rad[j.id]
                if abs(i.x - j.x) > thr + 20 or abs(i.y - j.y) > thr + 20:
                    continue
                if closest_approach_sq(px[i.id], py[i.id], i.x, i.y, px[j.id], py[j.id], j.x, j.y) < thr * thr:
                    head_on.setdefault(i.id, []).append(j.id)
                    head_on.setdefault(j.id, []).append(i.id)
        out = []
        for s in alive:
            r = rad[s.id]
            partners = head_on.get(s.id, [])
            cause, killer, loss = None, -1, -1
            for j in partners:
                o = self.snakes[j]
                if s.length < o.length:
                    loss = j if loss < 0 else loss
                elif s.length == o.length and loss < 0:
                    loss = -2 - j
            if loss != -1:
                cause, killer = "headon", (loss if loss >= 0 else -1)
            else:
                best_pen, best_owner = 0.0, -1
                for (o, ax, ay, bx, by, sr) in segs:
                    if o == s.id or o in partners:
                        continue
                    thr = r + sr
                    d2 = dist_point_seg_sq(s.x, s.y, ax, ay, bx, by)
                    if d2 < thr * thr:
                        pen = thr - math.sqrt(d2)
                        if pen > best_pen or (pen == best_pen and o < best_owner):
                            best_pen, best_owner = pen, o
                if best_owner >= 0:
                    cause, killer = "body", best_owner
                elif s.x < r or s.y < r or s.x > W - r or s.y > W - r:
                    cause = "wall"
            if cause:
                out.append({"id": s.id, "cause": cause, "killer": killer, "length": s.length})
        return out

    def _eat(self, px, py, dead):
        R = self.R
        claims: dict[int, tuple[int, float]] = {}
        for s in self.snakes:
            if not s.alive or s.id in dead:
                continue
            r = radius_for(R, s.length)
            x0, y0 = px[s.id], py[s.id]
            for fid, (x, y, v, pr) in self.food.items():
                lim = r + pr + R["eatReach"]
                if abs(x - s.x) > lim + 20 and abs(x - x0) > lim + 20:
                    continue
                d2 = dist_point_seg_sq(x, y, x0, y0, s.x, s.y)
                if d2 >= lim * lim:
                    continue
                prev = claims.get(fid)
                if prev is None or d2 < prev[1] or (d2 == prev[1] and s.id < prev[0]):
                    claims[fid] = (s.id, d2)
        for fid in sorted(claims):
            sid = claims[fid][0]
            v = self.food[fid][2]
            s = self.snakes[sid]
            s.length += v
            s.eaten_tick += v
            del self.food[fid]

    def _kill(self, d):
        R = self.R
        s = self.snakes[d["id"]]
        s.alive = False
        s.boosting = False
        s.death_cause = d["cause"]
        s.killer = d["killer"]
        if d["killer"] >= 0:
            self.snakes[d["killer"]].kills += 1
        self.deaths.append(d)
        r = radius_for(R, s.length)
        spacing = max(R["deathPelletSpacingMin"], r)
        pts = [(s.x, s.y)]
        ax, ay, carry = s.x, s.y, 0.0
        p = s.body.pts
        for i in range(s.body.n - 1, -1, -1):
            bx, by = p[2 * i], p[2 * i + 1]
            seg = math.hypot(bx - ax, by - ay)
            t = spacing - carry
            while t <= seg:
                pts.append((ax + (bx - ax) * (t / seg), ay + (by - ay) * (t / seg)))
                t += spacing
            carry = seg - (t - spacing)
            ax, ay = bx, by
        value = (R["deathDropFraction"] * s.length) / len(pts)
        for (x, y) in pts:
            jx = self.food_rng.range(-r * 0.5, r * 0.5)
            jy = self.food_rng.range(-r * 0.5, r * 0.5)
            self.add_pellet(x + jx, y + jy, value)
