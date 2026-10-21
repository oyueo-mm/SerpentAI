import { NUM_ACTIONS, decodeAction } from '../policy/action';
import { VIEW_RADIUS } from '../observation/sensors';
import { rollout, visibleHeads, visiblePellets, type PredictedHead } from './local';
import type { Decision, DecisionContext, MpcCandidate, Policy } from './types';
import type { Pellet, Snake, World } from '../sim/world';

/**
 * Model Predictive Control over act-v1. Each candidate holds one act-v1 action (turn level + boost) for
 * the first segment and, with depth 2, continues with one of `turns2` for the rest of the horizon.
 * Trajectories are predicted with the pm-v1 model (src/ai/local.ts) and scored by separate terms:
 *
 *   total = w.food·food + w.terminal·terminal − w.collision·collision − w.clearance·clearance − w.boost·boost
 *
 * Only the first action is executed; the plan is recomputed at the next decision (receding horizon).
 */
export interface MpcConfig {
  horizon: number;
  step: number;
  depth: 1 | 2;
  /** Fraction of the horizon covered by the first segment (depth 2). */
  split: number;
  turns2: number[];
  margin: number;
  weights: { food: number; collision: number; clearance: number; boost: number; terminal: number };
}

export const DEFAULT_MPC: MpcConfig = {
  horizon: 1.5,
  step: 1 / 15,
  depth: 2,
  split: 1 / 3,
  turns2: [-1, -0.5, 0, 0.5, 1],
  margin: 15,
  weights: { food: 1, collision: 1000, clearance: 4, boost: 0.5, terminal: 1 },
};

/** Value of food near the end of the plan (a cheap estimate of what lies beyond the horizon) minus a wall term. */
function terminalValue(world: World, x: number, y: number, pellets: Pellet[]): number {
  let v = 0;
  for (const p of pellets) {
    const d = Math.hypot(p.x - x, p.y - y);
    if (d < 160) v += p.value / (1 + d / 40);
  }
  const W = world.rules.worldSize;
  const wall = Math.min(x, y, W - x, W - y);
  if (wall < 150) v -= (150 - wall) * 0.2;
  return v;
}

export class MpcPolicy implements Policy {
  readonly id: string;
  readonly kind = 'mpc' as const;

  constructor(readonly config: MpcConfig = DEFAULT_MPC, id = 'mpc') {
    this.id = id;
  }

  /** Evaluates every candidate; exported separately so tests and the overlay can inspect the plan. */
  plan(world: World, self: Snake, heads?: PredictedHead[], view = VIEW_RADIUS): { best: number; candidates: MpcCandidate[] } {
    const c = this.config;
    const hs = heads ?? visibleHeads(world, self, view);
    const pellets = visiblePellets(world, self, view);
    const near = pellets.filter((p) => Math.hypot(p.x - self.x, p.y - self.y) < c.horizon * world.rules.boostSpeed + 40);
    const canBoost = self.length > world.rules.boostMinLength;
    const candidates: MpcCandidate[] = [];
    for (let a = 0; a < NUM_ACTIONS; a++) {
      const { turn, boost } = decodeAction(a);
      if (boost && !canBoost) continue; // masked
      const conts = c.depth === 2 ? c.turns2 : [undefined];
      let best: MpcCandidate | undefined;
      for (const turn2 of conts) {
        const roll = rollout(world, self, hs, near, { turn, turn2, switchTime: c.horizon * c.split, boost, horizon: c.horizon, step: c.step, margin: c.margin, view });
        const w = c.weights;
        const food = roll.food;
        const collision = roll.hitTime < Infinity ? 1 + (c.horizon - roll.hitTime) / c.horizon : 0;
        const clearance = roll.clearancePenalty * c.step;
        const boostCost = roll.lengthLost;
        const terminal = roll.hitTime < Infinity ? 0 : terminalValue(world, roll.endX, roll.endY, pellets);
        const total = w.food * food + w.terminal * terminal - w.collision * collision - w.clearance * clearance - w.boost * boostCost;
        const cand: MpcCandidate = {
          action: a, turn2, path: roll.path.filter((_, i) => i % 2 === 0 || i === roll.path.length - 1),
          hit: roll.hit, hitTime: roll.hitTime < Infinity ? roll.hitTime : undefined,
          cost: { food, collision, clearance, boost: boostCost, terminal, total },
        };
        if (!best || total > best.cost.total) best = cand;
      }
      candidates.push(best!);
    }
    let bi = 0;
    for (let i = 1; i < candidates.length; i++) if (candidates[i].cost.total > candidates[bi].cost.total) bi = i;
    return { best: bi, candidates };
  }

  decide(ctx: DecisionContext): Decision {
    const t0 = globalThis.performance?.now() ?? 0;
    const { best, candidates } = this.plan(ctx.world, ctx.self, undefined, ctx.sensing ?? VIEW_RADIUS);
    const evalMs = (globalThis.performance?.now() ?? 0) - t0;
    const chosen = candidates[best];
    const reason = chosen.cost.collision > 0 ? 'evade' : chosen.cost.food > 0 ? 'food' : 'cruise';
    return {
      action: chosen.action,
      debug: { policy: this.id, reason, target: chosen.path[chosen.path.length - 1], mpc: { candidates, best, horizon: this.config.horizon, evalMs } },
    };
  }
}
