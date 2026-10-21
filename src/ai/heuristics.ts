import { wrapAngle } from '../geometry/geom';
import { FOOD_SECTORS, RAY_COUNT, RAY_RANGE, rayAngle } from '../observation/sensors';
import { TURN_LEVELS, controlToAction, decodeAction, encodeAction } from '../policy/action';
import { radiusFor } from '../sim/config';
import { angleOff, headingTo, rollout, visibleHeads, visiblePellets, type PredictedHead, type Rollout } from './local';
import type { Decision, DecisionContext, Policy } from './types';

const steer = (ctx: DecisionContext, heading: number, boost = false) =>
  controlToAction({ targetHeading: heading, boost }, ctx.self.heading, ctx.world.rules, ctx.holdTicks);

/** Random: holds a random turn level for a random time, occasionally boosts. Baseline only. */
export class RandomPolicy implements Policy {
  readonly id = 'random';
  readonly kind = 'heuristic' as const;
  private current = encodeAction(0, false);
  private holdLeft = 0;

  decide(ctx: DecisionContext): Decision {
    if (this.holdLeft <= 0) {
      const turn = TURN_LEVELS[ctx.rng.int(0, TURN_LEVELS.length - 1)];
      this.current = encodeAction(turn, ctx.rng.next() < 0.08);
      this.holdLeft = ctx.rng.int(3, 15);
    }
    this.holdLeft--;
    return { action: this.current, debug: { policy: this.id, reason: 'random' } };
  }

  getState() { return { current: this.current, holdLeft: this.holdLeft }; }
  setState(s: unknown) { Object.assign(this, s as object); }
}

/** Greedy: chases the best value/distance pellet; avoids walls only (bodies are ignored on purpose). */
export class GreedyPolicy implements Policy {
  readonly id = 'greedy';
  readonly kind = 'heuristic' as const;

  decide(ctx: DecisionContext): Decision {
    const { self, world } = ctx;
    const W = world.rules.worldSize;
    const r = radiusFor(world.rules, self.length);
    const fx = self.x + Math.cos(self.heading) * 90, fy = self.y + Math.sin(self.heading) * 90;
    if (fx < r + 10 || fy < r + 10 || fx > W - r - 10 || fy > W - r - 10) {
      return { action: steer(ctx, headingTo(self, W / 2, W / 2)), debug: { policy: this.id, reason: 'wall', target: [W / 2, W / 2] } };
    }
    let best: [number, number] | undefined, bestScore = -Infinity;
    for (const p of visiblePellets(world, self)) {
      const d = Math.hypot(p.x - self.x, p.y - self.y);
      // Pellets behind need a U-turn: discount by the turn needed.
      const off = angleOff(self.heading, self.x, self.y, p.x, p.y);
      const score = p.value / (d + 30 + off * 60);
      if (score > bestScore) { bestScore = score; best = [p.x, p.y]; }
    }
    if (!best) return { action: steer(ctx, headingTo(self, W / 2, W / 2)), debug: { policy: this.id, reason: 'explore', target: [W / 2, W / 2] } };
    return { action: steer(ctx, headingTo(self, best[0], best[1])), debug: { policy: this.id, reason: 'food', target: best } };
  }
}

/**
 * Survival: purely reactive on the 16 sensor rays. Clearance per ray = min(body, wall); while the way
 * ahead is clear it drifts towards the richest food sector, otherwise it turns towards the clearest ray
 * that is reachable (penalising large turns), and boosts away from a head closing in from behind.
 */
export class SurvivalPolicy implements Policy {
  readonly id = 'survival';
  readonly kind = 'heuristic' as const;

  decide(ctx: DecisionContext): Decision {
    const { self, world } = ctx;
    const s = ctx.sensors();
    const R = world.rules;
    const rMin = R.baseSpeed / R.maxTurnRate; // minimum turning radius
    // A thin body can slip between two rays: a direction is only as clear as its neighbours allow.
    const raw = s.rayBody.map((b, i) => Math.min(b, s.rayWall[i]));
    const clear = raw.map((c, i) => Math.min(c, 1.6 * raw[(i + 1) % RAY_COUNT], 1.6 * raw[(i + RAY_COUNT - 1) % RAY_COUNT]));
    const danger = clear.map((c) => 1 - Math.min(1, c / RAY_RANGE));
    // Forward cone (rays 15, 0, 1) must have room for a full-rate turn plus margin.
    const fwd = Math.min(clear[RAY_COUNT - 1], clear[0], clear[1]);
    const safe = fwd > 2.2 * rMin + s.radius * 2;
    let target: number;
    let reason: string;
    if (safe) {
      let bs = 0, bestSec = -1;
      for (let i = 0; i < FOOD_SECTORS; i++) if (s.foodSum[i] > bs) { bs = s.foodSum[i]; bestSec = i; }
      const secAngle = bestSec >= 0 ? (bestSec * 2 * Math.PI) / FOOD_SECTORS : 0;
      // Only drift towards food in directions that are themselves clear (ray 2k is sector k's centre).
      const rayIdx = bestSec >= 0 ? bestSec * (RAY_COUNT / FOOD_SECTORS) : 0;
      target = clear[rayIdx] > 1.5 * rMin ? wrapAngle(secAngle) : 0;
      reason = bestSec >= 0 ? 'food' : 'cruise';
    } else {
      let best = 0, bestScore = -Infinity;
      for (let i = 0; i < RAY_COUNT; i++) {
        const phi = wrapAngle(rayAngle(i));
        const score = clear[i] - 70 * Math.abs(phi);
        if (score > bestScore) { bestScore = score; best = i; }
      }
      target = wrapAngle(rayAngle(best));
      reason = 'evade';
    }
    // A head closing in from behind (rel. position behind us, pointing at us) → boost away.
    let boost = false;
    for (const o of s.opponents) {
      const behind = o.x < 0 && Math.hypot(o.x, o.y) < 160;
      const pointing = Math.cos(Math.atan2(-o.y, -o.x) - o.relHeading) > 0.8;
      if (behind && pointing && s.length > R.boostMinLength + 40) boost = true;
    }
    const heading = self.heading + target;
    const tx = self.x + Math.cos(heading) * 150, ty = self.y + Math.sin(heading) * 150;
    return { action: steer(ctx, heading, boost), debug: { policy: this.id, reason: boost ? 'flee' : reason, danger, target: [tx, ty] } };
  }
}

/** Earliest collision (s) for each of the 9 turn levels held over `horizon`, under the pm-v1 model. */
function safetyScan(ctx: DecisionContext, heads: PredictedHead[], boost: boolean, horizon = 1.0): Rollout[] {
  const pellets = visiblePellets(ctx.world, ctx.self, 300);
  return TURN_LEVELS.map((turn) => rollout(ctx.world, ctx.self, heads, pellets, { turn, boost, horizon, step: 1 / 15, margin: 12 }));
}

/** Picks the turn level whose rollout ends closest (in angle) to `heading`, among the given rollouts. */
function bestTowards(ctx: DecisionContext, rolls: { roll: Rollout; turn: number }[], heading: number): number {
  let best = 0, bestScore = Infinity;
  rolls.forEach(({ roll }, i) => {
    const off = Math.abs(wrapAngle(roll.endHeading - heading));
    const toward = angleOff(heading, ctx.self.x, ctx.self.y, roll.endX, roll.endY);
    const score = off + 0.5 * toward;
    if (score < bestScore) { bestScore = score; best = i; }
  });
  return rolls[best].turn;
}

/**
 * Aggressive: picks a smaller (or similar) visible head, predicts its straight-line motion and aims at a
 * point just in front of it so its path runs into our body; boosts when the intercept is close.
 * Falls back to an evasive turn when its own short-horizon rollout predicts a collision.
 */
export class AggressivePolicy implements Policy {
  readonly id = 'aggressive';
  readonly kind = 'heuristic' as const;

  decide(ctx: DecisionContext): Decision {
    const { self, world } = ctx;
    const R = world.rules;
    const heads = visibleHeads(world, self);
    const myR = radiusFor(R, self.length);
    let target: PredictedHead | undefined, bestScore = -Infinity;
    for (const h of heads) {
      if (h.r > myR * 1.25) continue;
      const d = Math.hypot(h.x - self.x, h.y - self.y);
      const score = -d + 80 * (myR - h.r) / myR;
      if (score > bestScore) { bestScore = score; target = h; }
    }
    let aim: [number, number] | undefined, boost = false, reason = 'forage', intercept: [number, number] | undefined;
    if (target) {
      // Solve τ ≈ |aim(τ) − self| / v with a few fixed-point iterations.
      const lead = 3 * target.r + 25;
      const mySpeed = self.length > R.boostMinLength + 60 ? R.boostSpeed : R.baseSpeed;
      let tau = 0;
      let ax = target.x, ay = target.y;
      for (let i = 0; i < 4; i++) {
        const hx = target.x + target.vx * tau, hy = target.y + target.vy * tau;
        const sp = Math.hypot(target.vx, target.vy) || 1;
        ax = hx + (target.vx / sp) * lead; ay = hy + (target.vy / sp) * lead;
        tau = Math.hypot(ax - self.x, ay - self.y) / mySpeed;
      }
      aim = [ax, ay];
      intercept = [ax, ay];
      const d = Math.hypot(ax - self.x, ay - self.y);
      boost = d < 320 && angleOff(self.heading, self.x, self.y, ax, ay) < Math.PI / 4 && self.length > R.boostMinLength + 60;
      reason = 'attack';
    } else {
      // Collect the biggest visible pellet (death remains first).
      let bs = -Infinity;
      for (const p of visiblePellets(world, self)) {
        const sc = (p.kind === 'death' ? 2 : 1) * p.value / (Math.hypot(p.x - self.x, p.y - self.y) + 40);
        if (sc > bs) { bs = sc; aim = [p.x, p.y]; }
      }
      if (!aim) aim = [R.worldSize / 2, R.worldSize / 2];
    }
    const desired = headingTo(self, aim[0], aim[1]);
    // Safety: keep the attack only if its rollout is collision-free; otherwise take the safest turn.
    const want = controlToAction({ targetHeading: desired, boost }, self.heading, R, ctx.holdTicks);
    const { turn } = decodeAction(want);
    const check = rollout(world, self, heads, [], { turn, boost, horizon: 0.8, step: 1 / 15, margin: 6 });
    if (check.hitTime < Infinity) {
      const scan = safetyScan(ctx, heads, false, 0.8);
      let bi = 0;
      scan.forEach((r, i) => { if (r.hitTime > scan[bi].hitTime || (r.hitTime === scan[bi].hitTime && r.minClear > scan[bi].minClear)) bi = i; });
      return { action: encodeAction(TURN_LEVELS[bi], false), debug: { policy: this.id, reason: 'evade', target: aim, intercept } };
    }
    return { action: want, debug: { policy: this.id, reason, target: aim, intercept } };
  }
}

/**
 * Artificial potential field: pellets attract (value / d²), visible body segments and walls repel
 * (1 / gap²). Steers along the net force. Prone to local minima between symmetric obstacles — kept
 * as-is so the tournament can show it.
 */
export class PotentialFieldPolicy implements Policy {
  readonly id = 'potential';
  readonly kind = 'heuristic' as const;

  decide(ctx: DecisionContext): Decision {
    const { self, world } = ctx;
    const R = world.rules, W = R.worldSize, g = world.segGrid;
    const r = radiusFor(R, self.length);
    let fx = Math.cos(self.heading) * 0.002, fy = Math.sin(self.heading) * 0.002; // small forward bias
    for (const p of visiblePellets(world, self, 350)) {
      const dx = p.x - self.x, dy = p.y - self.y;
      const d = Math.max(10, Math.hypot(dx, dy));
      const k = (p.value / (d * d)) / d;
      fx += dx * k; fy += dy * k;
    }
    const REP = 150;
    world.ensureGrid();
    g.query(self.x - REP, self.y - REP, self.x + REP, self.y + REP, (k) => {
      if (g.owner[k] === self.id || !world.snakes[g.owner[k]].alive) return;
      // Closest point on the segment.
      const ax = g.ax[k], ay = g.ay[k], bx = g.bx[k], by = g.by[k];
      const sx = bx - ax, sy = by - ay, l2 = sx * sx + sy * sy;
      let t = l2 > 0 ? ((self.x - ax) * sx + (self.y - ay) * sy) / l2 : 0;
      t = Math.max(0, Math.min(1, t));
      const cx = ax + sx * t, cy = ay + sy * t;
      const dx = self.x - cx, dy = self.y - cy, d = Math.hypot(dx, dy);
      if (d > REP || d < 1e-6) return;
      const gap = Math.max(2, d - g.rad[k] - r);
      const m = 6 / (gap * gap) / d;
      fx += dx * m; fy += dy * m;
    });
    const walls: [number, number, number][] = [[self.x - r, 1, 0], [W - self.x - r, -1, 0], [self.y - r, 0, 1], [W - self.y - r, 0, -1]];
    for (const [gap, nx, ny] of walls) if (gap < REP) { const m = 10 / Math.max(2, gap) ** 2; fx += nx * m; fy += ny * m; }
    const heading = Math.atan2(fy, fx);
    return { action: steer(ctx, heading), debug: { policy: this.id, reason: 'field', target: [self.x + Math.cos(heading) * 120, self.y + Math.sin(heading) * 120] } };
  }
}

/**
 * Hybrid: a safety layer (pm-v1 rollouts of every turn level, with and without boost) filters the
 * actions; among the safe ones it pursues, in priority order, an interception of a smaller nearby head,
 * the best food (death pellets weigh double), or the arena centre. Boosts only when safe and worthwhile.
 */
export class HybridPolicy implements Policy {
  readonly id = 'hybrid';
  readonly kind = 'heuristic' as const;

  decide(ctx: DecisionContext): Decision {
    const { self, world } = ctx;
    const R = world.rules;
    const heads = visibleHeads(world, self);
    const myR = radiusFor(R, self.length);
    const canBoost = self.length > R.boostMinLength + 40;
    const plain = safetyScan(ctx, heads, false, 1.0);
    const boosted = canBoost ? safetyScan(ctx, heads, true, 1.0) : [];

    // Goal selection.
    let goal: [number, number] | undefined, reason = 'food', wantBoost = false, intercept: [number, number] | undefined;
    for (const h of heads) {
      if (h.r >= myR * 0.95) continue;
      const d = Math.hypot(h.x - self.x, h.y - self.y);
      if (d > 260) continue;
      const sp = Math.hypot(h.vx, h.vy) || 1;
      const tau = d / R.boostSpeed;
      const ax = h.x + h.vx * tau + (h.vx / sp) * (3 * h.r + 25), ay = h.y + h.vy * tau + (h.vy / sp) * (3 * h.r + 25);
      if (angleOff(self.heading, self.x, self.y, ax, ay) < Math.PI / 2) {
        goal = [ax, ay]; intercept = goal; reason = 'attack'; wantBoost = canBoost && d < 220;
        break;
      }
    }
    if (!goal) {
      let bs = -Infinity;
      for (const p of visiblePellets(world, self)) {
        const d = Math.hypot(p.x - self.x, p.y - self.y);
        const off = angleOff(self.heading, self.x, self.y, p.x, p.y);
        const sc = (p.kind === 'death' ? 2 : 1) * p.value / (d + 40 + 50 * off);
        if (sc > bs) { bs = sc; goal = [p.x, p.y]; }
      }
      if (!goal) { goal = [R.worldSize / 2, R.worldSize / 2]; reason = 'explore'; }
    }
    const desired = headingTo(self, goal[0], goal[1]);

    const options = [
      ...plain.map((roll, i) => ({ roll, turn: TURN_LEVELS[i], boost: false })),
      ...boosted.map((roll, i) => ({ roll, turn: TURN_LEVELS[i], boost: true })),
    ];
    const safe = options.filter((o) => o.roll.hitTime === Infinity && o.roll.minClear > 4);
    let chosen: { turn: number; boost: boolean };
    if (safe.length === 0) {
      // No safe option: maximise time to impact, then clearance.
      const best = options.reduce((a, b) => (b.roll.hitTime > a.roll.hitTime || (b.roll.hitTime === a.roll.hitTime && b.roll.minClear > a.roll.minClear) ? b : a));
      chosen = best;
      reason = 'evade';
    } else {
      const pool = safe.filter((o) => o.boost === wantBoost);
      const use = pool.length ? pool : safe.filter((o) => !o.boost).length ? safe.filter((o) => !o.boost) : safe;
      const turn = bestTowards(ctx, use.map((o) => ({ roll: o.roll, turn: o.turn })), desired);
      chosen = use.find((o) => o.turn === turn)!;
      if (safe.length < options.length / 3) reason = reason === 'attack' ? 'attack' : 'caution';
    }
    return { action: encodeAction(chosen.turn, chosen.boost), debug: { policy: this.id, reason, target: goal, intercept } };
  }
}

export const HEURISTIC_IDS = ['random', 'greedy', 'survival', 'aggressive', 'potential', 'hybrid'] as const;

export function createHeuristic(id: (typeof HEURISTIC_IDS)[number]): Policy {
  switch (id) {
    case 'random': return new RandomPolicy();
    case 'greedy': return new GreedyPolicy();
    case 'survival': return new SurvivalPolicy();
    case 'aggressive': return new AggressivePolicy();
    case 'potential': return new PotentialFieldPolicy();
    case 'hybrid': return new HybridPolicy();
  }
}
