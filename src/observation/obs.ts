import { DEFAULT_RULES, type RulesConfig } from '../sim/config';
import { FOOD_RANGE, FOOD_SECTORS, OPP_K, OPP_RANGE, RAY_COUNT, RAY_RANGE, type Sensors } from './sensors';

/**
 * obs-v1: fixed-length egocentric observation vector. The same encoder (this file, and its Python port
 * vision/observation/encode.py) turns Sensors from either the engine (State mode) or the Vision
 * pipeline into the policy input, so the network never knows where the numbers came from.
 *
 * Layout (84 floats):
 *   [0..5]   self: log-length, radius, can-boost, previous turn, previous boost, boost fuel
 *   [6..37]  16 rays × (body distance, wall distance), each distance / RAY_RANGE, 1 = nothing in range
 *   [38..53] 8 food sectors × (log1p(value sum)/log1p(200), nearest distance / FOOD_RANGE)
 *   [54..81] 4 opponents × (present, x/R, y/R, cos rel heading, sin rel heading, speed level, radius ratio)
 *   [82..83] confidence, frame age (s, clipped to 1)
 */
export const OBS_VERSION = 'obs-v1';
export const SELF_FEATURES = 6;
export const OPP_FEATURES = 7;
export const OBS_SIZE = SELF_FEATURES + RAY_COUNT * 2 + FOOD_SECTORS * 2 + OPP_K * OPP_FEATURES + 2;

export interface PrevAction {
  turn: number;
  boost: boolean;
}

const clip = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

export function encodeObs(s: Sensors, prev: PrevAction, rules: RulesConfig = DEFAULT_RULES): Float32Array {
  const o = new Float32Array(OBS_SIZE);
  let k = 0;
  o[k++] = clip(Math.log(Math.max(s.length, 1) / rules.startLength) / Math.log(50), -1, 1);
  o[k++] = clip(s.radius / rules.radiusMax, 0, 1);
  o[k++] = s.length > rules.boostMinLength ? 1 : 0;
  o[k++] = clip(prev.turn, -1, 1);
  o[k++] = prev.boost ? 1 : 0;
  o[k++] = clip((s.length - rules.boostMinLength) / 200, 0, 1);
  for (let i = 0; i < RAY_COUNT; i++) {
    o[k++] = clip(s.rayBody[i] / RAY_RANGE, 0, 1);
    o[k++] = clip(s.rayWall[i] / RAY_RANGE, 0, 1);
  }
  for (let i = 0; i < FOOD_SECTORS; i++) {
    o[k++] = clip(Math.log1p(Math.max(0, s.foodSum[i])) / Math.log1p(200), 0, 1);
    o[k++] = clip(s.foodNear[i] / FOOD_RANGE, 0, 1);
  }
  const speedSpan = rules.boostSpeed - rules.baseSpeed;
  for (let j = 0; j < OPP_K; j++) {
    const p = s.opponents[j];
    if (!p) { k += OPP_FEATURES; continue; }
    o[k++] = 1;
    o[k++] = clip(p.x / OPP_RANGE, -1, 1);
    o[k++] = clip(p.y / OPP_RANGE, -1, 1);
    o[k++] = Math.cos(p.relHeading);
    o[k++] = Math.sin(p.relHeading);
    o[k++] = clip((p.speed - rules.baseSpeed) / speedSpan, -1, 2);
    o[k++] = clip(Math.log2(Math.max(p.radius, 1e-3) / Math.max(s.radius, 1e-3)) / 2, -1, 1);
  }
  o[k++] = clip(s.confidence, 0, 1);
  o[k++] = clip(s.frameAge, 0, 1);
  if (k !== OBS_SIZE) throw new Error(`obs layout error: ${k} != ${OBS_SIZE}`);
  return o;
}

/** Human-readable feature names, in vector order (exported to docs and Python). */
export function obsFeatureNames(): string[] {
  const n = ['self.log_length', 'self.radius', 'self.can_boost', 'self.prev_turn', 'self.prev_boost', 'self.boost_fuel'];
  for (let i = 0; i < RAY_COUNT; i++) n.push(`ray${i}.body`, `ray${i}.wall`);
  for (let i = 0; i < FOOD_SECTORS; i++) n.push(`food${i}.sum`, `food${i}.near`);
  for (let j = 0; j < OPP_K; j++) n.push(`opp${j}.present`, `opp${j}.x`, `opp${j}.y`, `opp${j}.cos`, `opp${j}.sin`, `opp${j}.speed`, `opp${j}.radius_ratio`);
  n.push('conf.confidence', 'conf.frame_age');
  return n;
}
