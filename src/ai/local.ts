import { distPointSegSq, integrateArc } from '../geometry/geom';
import { VIEW_RADIUS } from '../observation/sensors';
import { radiusFor } from '../sim/config';
import type { Pellet, Snake, World } from '../sim/world';

/**
 * Shared prediction helpers for the heuristic and MPC experts. They only use state within VIEW_RADIUS
 * of the own head (what a player sees on screen), never anything off-screen.
 *
 * Prediction model ("pm-v1"): own head follows the engine's exact arc kinematics; other bodies are
 * static; other heads move straight at their current speed and leave a body capsule behind them.
 */
export const CLEAR_CAP = 80;

export interface PredictedHead {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  length: number;
  heading: number;
  speed: number;
}

export function visibleHeads(world: World, self: Snake, radius = VIEW_RADIUS): PredictedHead[] {
  const out: PredictedHead[] = [];
  for (const o of world.snakes) {
    if (!o.alive || o.id === self.id) continue;
    if (Math.hypot(o.x - self.x, o.y - self.y) > radius) continue;
    out.push({ id: o.id, x: o.x, y: o.y, vx: Math.cos(o.heading) * o.speed, vy: Math.sin(o.heading) * o.speed, r: radiusFor(world.rules, o.length), length: o.length, heading: o.heading, speed: o.speed });
  }
  return out;
}

export function visiblePellets(world: World, self: Snake, radius = VIEW_RADIUS): Pellet[] {
  const out: Pellet[] = [];
  world.queryFood(self.x - radius, self.y - radius, self.x + radius, self.y + radius, (p) => {
    if (Math.hypot(p.x - self.x, p.y - self.y) <= radius) out.push(p);
  });
  out.sort((a, b) => a.id - b.id);
  return out;
}

/**
 * Clearance (u) of a head circle of radius r at (x, y) at prediction time t: distance to the nearest
 * visible body surface, predicted head capsule or wall, capped at CLEAR_CAP. Negative = collision.
 */
export function clearanceAt(world: World, self: Snake, x: number, y: number, r: number, heads: PredictedHead[], t: number, view = VIEW_RADIUS): number {
  const W = world.rules.worldSize, g = world.segGrid;
  let best = Math.min(CLEAR_CAP, x - r, y - r, W - x - r, W - y - r);
  const reach = r + world.rules.radiusMax + CLEAR_CAP;
  const vr2 = (view + 40) * (view + 40);
  g.query(x - reach, y - reach, x + reach, y + reach, (k) => {
    const o = g.owner[k];
    if (o === self.id || !world.snakes[o].alive) return;
    const ax = g.ax[k] - self.x, ay = g.ay[k] - self.y;
    if (ax * ax + ay * ay > vr2) return;
    const d = Math.sqrt(distPointSegSq(x, y, g.ax[k], g.ay[k], g.bx[k], g.by[k])) - g.rad[k] - r;
    if (d < best) best = d;
  });
  for (const h of heads) {
    const d = Math.sqrt(distPointSegSq(x, y, h.x, h.y, h.x + h.vx * t, h.y + h.vy * t)) - h.r - r;
    if (d < best) best = d;
  }
  return best;
}

export interface Rollout {
  path: [number, number][];
  /** Time (s) of the first predicted collision, Infinity if none within the horizon. */
  hitTime: number;
  hit?: [number, number];
  /** Minimum clearance along the path (u). */
  minClear: number;
  /** Σ max(0, margin − clearance) / margin over steps. */
  clearancePenalty: number;
  /** Total value of distinct pellets the mouth would sweep. */
  food: number;
  lengthLost: number;
  endX: number;
  endY: number;
  endHeading: number;
}

export interface RolloutSpec {
  /** Turn level for t < switchTime, then turn2 (if given). */
  turn: number;
  turn2?: number;
  switchTime?: number;
  boost: boolean;
  horizon: number;
  step: number;
  margin: number;
  /** Perception radius for obstacles (default VIEW_RADIUS). */
  view?: number;
}

/** Simulates the own head under a piecewise-constant turn command with the pm-v1 model. */
export function rollout(world: World, self: Snake, heads: PredictedHead[], pellets: Pellet[], spec: RolloutSpec): Rollout {
  const R = world.rules;
  let x = self.x, y = self.y, th = self.heading, L = self.length;
  const r = radiusFor(R, L);
  const tmp = new Float64Array(3);
  const path: [number, number][] = [[x, y]];
  let hitTime = Infinity, hit: [number, number] | undefined, minClear = CLEAR_CAP, pen = 0, food = 0, lost = 0;
  const eaten = new Set<number>();
  const n = Math.max(1, Math.round(spec.horizon / spec.step));
  for (let i = 1; i <= n; i++) {
    const t = i * spec.step;
    const turn = spec.turn2 !== undefined && (t - spec.step) >= (spec.switchTime ?? Infinity) ? spec.turn2 : spec.turn;
    const boosting = spec.boost && L > R.boostMinLength;
    const v = boosting ? R.boostSpeed : R.baseSpeed;
    const px = x, py = y;
    integrateArc(x, y, th, v, turn * R.maxTurnRate, spec.step, tmp);
    x = tmp[0]; y = tmp[1]; th = tmp[2];
    if (boosting) {
      const nl = Math.max(R.boostMinLength, L - R.boostCost * spec.step);
      lost += L - nl;
      L = nl;
    }
    path.push([x, y]);
    // Food swept by the mouth on this step.
    const lim0 = r + R.eatReach;
    for (const p of pellets) {
      if (eaten.has(p.id)) continue;
      const lim = lim0 + p.r;
      if (distPointSegSq(p.x, p.y, px, py, x, y) < lim * lim) { eaten.add(p.id); food += p.value; }
    }
    const c = clearanceAt(world, self, x, y, r, heads, t, spec.view);
    if (c < minClear) minClear = c;
    if (c < spec.margin) pen += (spec.margin - c) / spec.margin;
    if (c <= 0) { hitTime = t; hit = [x, y]; break; }
  }
  return { path, hitTime, hit, minClear, clearancePenalty: pen, food, lengthLost: lost, endX: x, endY: y, endHeading: th };
}

/** Heading (rad) from the head to a world point. */
export function headingTo(self: Snake, x: number, y: number): number {
  return Math.atan2(y - self.y, x - self.x);
}

/** Absolute angle between a heading and the direction to a point. */
export function angleOff(heading: number, fromX: number, fromY: number, toX: number, toY: number): number {
  const a = Math.atan2(toY - fromY, toX - fromX) - heading;
  return Math.abs(Math.atan2(Math.sin(a), Math.cos(a)));
}
