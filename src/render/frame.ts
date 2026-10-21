import type { DecisionDebug } from '../ai/types';
import { integrateArc } from '../geometry/geom';
import { decodeAction } from '../policy/action';
import { radiusFor } from '../sim/config';
import type { Match } from '../sim/match';

/** Render snapshot of one snake. body = [x0, y0, …] tail → head (the head point is last). */
export interface SnakeFrame {
  id: number;
  label: string;
  policy: string;
  skin: number;
  x: number;
  y: number;
  heading: number;
  length: number;
  radius: number;
  alive: boolean;
  boosting: boolean;
  speed: number;
  /** Turn level currently applied (act-v1 decoded) or the human's turn rate fraction. */
  turn: number;
  body: Float32Array;
  kills: number;
  lives: number;
}

export interface FrameDebug {
  /** Per-snake decision debug (only for the snakes requested). */
  decisions: Record<number, DecisionDebug | undefined>;
  /** Head positions of the focused snake over the last seconds (actual motion trail). */
  trail: number[];
  /** Extrapolation of each visible snake's current command for 1 s (exact arc kinematics). */
  extrapolations: Record<number, number[]>;
  gridCell: number;
}

export interface Frame {
  tick: number;
  time: number;
  worldSize: number;
  tickRate: number;
  maxTurnRate: number;
  boostMinLength: number;
  snakes: SnakeFrame[];
  /** Pellets: [x, y, r, kind(0 natural, 1 boost, 2 death), colourSeed] × n. */
  food: Float32Array;
  focusId: number;
  debug?: FrameDebug;
  ended: boolean;
}

/** Builds a render frame from the live match (runs in the worker). */
export function buildFrame(match: Match, focusId: number, opts: { debug: boolean; trail: number[] }): Frame {
  const w = match.world;
  const snakes: SnakeFrame[] = w.snakes.map((s) => {
    const b = s.body;
    const n = b.vertexCount;
    const body = new Float32Array(s.alive ? (n + 1) * 2 : 0);
    if (s.alive) {
      for (let i = 0; i < n; i++) { body[2 * i] = b.vx(i); body[2 * i + 1] = b.vy(i); }
      body[2 * n] = s.x; body[2 * n + 1] = s.y;
    }
    const a = match.agents[s.id];
    return {
      id: s.id, label: s.label, policy: a.spec.policy, skin: s.skin, x: s.x, y: s.y, heading: s.heading, length: s.length,
      radius: radiusFor(w.rules, s.length), alive: s.alive, boosting: s.boosting, speed: s.speed,
      turn: a.isHuman ? s.omega / w.rules.maxTurnRate : decodeAction(a.held).turn, body, kills: a.record.kills, lives: s.lives,
    };
  });
  const food = new Float32Array(w.food.size * 5);
  let k = 0;
  for (const p of w.food.values()) {
    food[k++] = p.x; food[k++] = p.y; food[k++] = p.r;
    food[k++] = p.kind === 'natural' ? 0 : p.kind === 'boost' ? 1 : 2;
    food[k++] = p.id % 997;
  }
  let debug: FrameDebug | undefined;
  if (opts.debug) {
    const decisions: Record<number, DecisionDebug | undefined> = {};
    const extrapolations: Record<number, number[]> = {};
    const tmp = new Float64Array(3);
    for (const s of w.snakes) {
      if (!s.alive) continue;
      decisions[s.id] = match.agents[s.id].debug;
      const pts: number[] = [s.x, s.y];
      let x = s.x, y = s.y, th = s.heading;
      for (let i = 0; i < 15; i++) {
        integrateArc(x, y, th, s.speed, s.omega, 1 / 15, tmp);
        x = tmp[0]; y = tmp[1]; th = tmp[2];
        pts.push(x, y);
      }
      extrapolations[s.id] = pts;
    }
    debug = { decisions, trail: opts.trail.slice(), extrapolations, gridCell: w.rules.gridCell };
  }
  return { tick: w.tick, time: w.time, worldSize: w.rules.worldSize, tickRate: w.rules.tickRate, maxTurnRate: w.rules.maxTurnRate, boostMinLength: w.rules.boostMinLength, snakes, food, focusId, debug, ended: match.ended };
}
