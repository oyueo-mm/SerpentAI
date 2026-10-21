import { rayBox, rayCapsule } from '../geometry/geom';
import { radiusFor } from '../sim/config';
import type { Pellet, Snake, World } from '../sim/world';

/**
 * Raw egocentric sensor readings — the quantities the obs-v1 vector is built from. Every field is
 * defined on things visible on screen within VIEW_RADIUS of the head (the game camera guarantees that
 * circle is always on screen), so the Vision pipeline can estimate the same struct from pixels.
 *
 * Egocentric frame: origin = own head centre, +x = heading, +y = 90° clockwise of heading on screen
 * (world y points down), angles φ measured from +x towards +y.
 */
export const VIEW_RADIUS = 480;
export const RAY_COUNT = 16;
export const RAY_RANGE = 400;
export const FOOD_SECTORS = 8;
export const FOOD_RANGE = 400;
export const OPP_K = 4;
export const OPP_RANGE = 450;

export interface OpponentReading {
  /** Egocentric head position (u). */
  x: number;
  y: number;
  /** Opponent heading relative to own heading (rad). */
  relHeading: number;
  /** Head speed (u/s). */
  speed: number;
  /** Body radius (u). */
  radius: number;
}

export interface Sensors {
  /** Own body length L (u) — read from the HUD length gauge in Vision mode. */
  length: number;
  /** Own body radius (u). */
  radius: number;
  /** Distance (u) along each ray to the nearest other-snake body surface; RAY_RANGE if none. */
  rayBody: number[];
  /** Distance (u) along each ray to the arena wall; RAY_RANGE if beyond. */
  rayWall: number[];
  /** Total pellet value per food sector within FOOD_RANGE. */
  foodSum: number[];
  /** Distance (u) to the nearest pellet centre per sector; FOOD_RANGE if none. */
  foodNear: number[];
  /** Up to OPP_K nearest other heads within OPP_RANGE, nearest first. */
  opponents: OpponentReading[];
  /** Estimate confidence in [0, 1] (1 for exact engine state). */
  confidence: number;
  /** Age of the frame the reading came from (s); 0 for engine state. */
  frameAge: number;
}

export function rayAngle(i: number): number {
  return (i * 2 * Math.PI) / RAY_COUNT;
}

/** Food sector index for an egocentric angle; sector 0 is centred on the heading. */
export function sectorOf(phi: number): number {
  const w = (2 * Math.PI) / FOOD_SECTORS;
  let a = (phi + w / 2) % (2 * Math.PI);
  if (a < 0) a += 2 * Math.PI;
  return Math.min(FOOD_SECTORS - 1, Math.floor(a / w));
}

/** Exact sensor readings for `self` from the engine state (State mode). */
export function senseWorld(world: World, self: Snake): Sensors {
  world.ensureGrid();
  const R = world.rules, g = world.segGrid;
  const hx = self.x, hy = self.y, th = self.heading;
  const fx = Math.cos(th), fy = Math.sin(th);
  const rx = -fy, ry = fx;
  const rayBody = new Array<number>(RAY_COUNT).fill(RAY_RANGE);
  const rayWall = new Array<number>(RAY_COUNT);
  const dirs: [number, number][] = [];
  for (let i = 0; i < RAY_COUNT; i++) {
    const a = th + rayAngle(i);
    const dx = Math.cos(a), dy = Math.sin(a);
    dirs.push([dx, dy]);
    rayWall[i] = Math.min(RAY_RANGE, rayBox(hx, hy, dx, dy, R.worldSize));
  }
  const reach = RAY_RANGE + R.radiusMax;
  g.query(hx - reach, hy - reach, hx + reach, hy + reach, (k) => {
    const o = g.owner[k];
    if (o === self.id || !world.snakes[o].alive) return;
    for (let i = 0; i < RAY_COUNT; i++) {
      const t = rayCapsule(hx, hy, dirs[i][0], dirs[i][1], g.ax[k], g.ay[k], g.bx[k], g.by[k], g.rad[k]);
      if (t < rayBody[i]) rayBody[i] = t;
    }
  });

  const foodSum = new Array<number>(FOOD_SECTORS).fill(0);
  const foodNear = new Array<number>(FOOD_SECTORS).fill(FOOD_RANGE);
  // Visit pellets in id order so the float sums do not depend on grid bucket order.
  const near: Pellet[] = [];
  world.queryFood(hx - FOOD_RANGE, hy - FOOD_RANGE, hx + FOOD_RANGE, hy + FOOD_RANGE, (p) => near.push(p));
  near.sort((a, b) => a.id - b.id);
  for (const p of near) {
    const dx = p.x - hx, dy = p.y - hy;
    const d = Math.hypot(dx, dy);
    if (d >= FOOD_RANGE) continue;
    const s = sectorOf(Math.atan2(dx * rx + dy * ry, dx * fx + dy * fy));
    foodSum[s] += p.value;
    if (d < foodNear[s]) foodNear[s] = d;
  }

  const opponents: (OpponentReading & { d: number; id: number })[] = [];
  for (const o of world.snakes) {
    if (o.id === self.id || !o.alive) continue;
    const dx = o.x - hx, dy = o.y - hy;
    const d = Math.hypot(dx, dy);
    if (d >= OPP_RANGE) continue;
    let rel = o.heading - th;
    rel = Math.atan2(Math.sin(rel), Math.cos(rel));
    opponents.push({ x: dx * fx + dy * fy, y: dx * rx + dy * ry, relHeading: rel, speed: o.speed, radius: radiusFor(R, o.length), d, id: o.id });
  }
  opponents.sort((a, b) => a.d - b.d || a.id - b.id);

  return {
    length: self.length,
    radius: radiusFor(R, self.length),
    rayBody, rayWall, foodSum, foodNear,
    opponents: opponents.slice(0, OPP_K).map(({ x, y, relHeading, speed, radius }) => ({ x, y, relHeading, speed, radius })),
    confidence: 1,
    frameAge: 0,
  };
}
