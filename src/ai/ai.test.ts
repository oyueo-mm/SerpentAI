import { describe, expect, it } from 'vitest';
import { NUM_ACTIONS, decodeAction } from '../policy/action';
import { Match } from '../sim/match';
import { Rng } from '../sim/rng';
import { World } from '../sim/world';
import { senseWorld } from '../observation/sensors';
import { HEURISTIC_IDS, createHeuristic } from './heuristics';
import { rollout, visibleHeads } from './local';
import { DEFAULT_MPC, MpcPolicy } from './mpc';
import type { DecisionContext, Policy } from './types';

function ctxFor(world: World, id: number, rng = new Rng(1), holdTicks = 3): DecisionContext {
  return { world, self: world.snakes[id], rng, holdTicks, sensors: () => senseWorld(world, world.snakes[id]), prevAction: 8 };
}

function emptyWorld(): World {
  return new World({ foodTarget: 0 }, 5, { fillFood: false });
}

/** Runs one snake under a policy (decisions every 3 ticks) and returns whether it is still alive. */
function survive(world: World, id: number, policy: Policy, seconds: number): boolean {
  const rng = new Rng(2);
  let action = 8;
  for (let t = 0; t < seconds * 30; t++) {
    if (t % 3 === 0) action = policy.decide(ctxFor(world, id, rng)).action;
    const controls = world.snakes.map((s) => (s.id === id ? decodeAction(action) : {}));
    world.step(controls);
    if (!world.snakes[id].alive) return false;
  }
  return true;
}

describe('policy interface', () => {
  it('every built-in policy returns a valid act-v1 action, deterministically', () => {
    for (const id of [...HEURISTIC_IDS, 'mpc'] as const) {
      const mk = () => {
        const w = new World({}, 77);
        for (let i = 0; i < 8; i++) w.addSnake();
        for (let t = 0; t < 30; t++) w.step([]);
        return w;
      };
      const p1 = id === 'mpc' ? new MpcPolicy() : createHeuristic(id);
      const p2 = id === 'mpc' ? new MpcPolicy() : createHeuristic(id);
      const w1 = mk(), w2 = mk();
      for (let k = 0; k < 8; k++) {
        const a = p1.decide(ctxFor(w1, k, new Rng(k))).action;
        const b = p2.decide(ctxFor(w2, k, new Rng(k))).action;
        expect(a, id).toBeGreaterThanOrEqual(0);
        expect(a, id).toBeLessThan(NUM_ACTIONS);
        expect(a, id).toBe(b);
      }
    }
  });
});

describe('MPC', () => {
  it('predicts its own trajectory exactly as the engine executes it (pm-v1 kinematics)', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 1500, y: 1500, heading: 0.4, length: 500 });
    for (const [turn, boost] of [[0.75, false], [-0.5, true], [1, true]] as const) {
      const roll = rollout(w, s, [], [], { turn, boost, horizon: 1.0, step: 1 / 30, margin: 0 });
      for (let t = 1; t <= 30; t++) {
        w.step([{ turn, boost }]);
        expect(Math.hypot(s.x - roll.path[t][0], s.y - roll.path[t][1])).toBeLessThan(1e-6);
      }
    }
  });

  it('turns away from a wall dead ahead and survives', () => {
    const w = emptyWorld();
    w.addSnake({ x: 2800, y: 1500, heading: 0 });
    expect(survive(w, 0, new MpcPolicy(), 8)).toBe(true);
  });

  it('avoids a long body lying across its path', () => {
    const w = emptyWorld();
    w.addSnake({ x: 1500, y: 1650, heading: -Math.PI / 2 });
    w.addSnake({ x: 2000, y: 1500, heading: 0, length: 1200 }); // body along y = 1500 from x = 800
    expect(survive(w, 0, new MpcPolicy(), 6)).toBe(true);
  });

  it('never plans a boost when boosting is unavailable (masked)', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 1500, y: 1500, heading: 0, length: w.rules.boostMinLength });
    const { candidates } = new MpcPolicy().plan(w, s);
    expect(candidates.length).toBe(9);
    expect(candidates.every((c) => !decodeAction(c.action).boost)).toBe(true);
  });

  it('reports the candidates it actually evaluated, and executes the best one', () => {
    const w = new World({}, 3);
    for (let i = 0; i < 6; i++) w.addSnake();
    const mpc = new MpcPolicy();
    const d = mpc.decide(ctxFor(w, 0));
    const dbg = d.debug!.mpc!;
    expect(dbg.candidates.length).toBe(NUM_ACTIONS);
    const best = dbg.candidates[dbg.best];
    expect(d.action).toBe(best.action);
    for (const c of dbg.candidates) expect(c.cost.total).toBeLessThanOrEqual(best.cost.total);
    // Re-planning from the same state reproduces the same costs (the overlay is the real computation).
    const again = mpc.plan(w, w.snakes[0]);
    expect(again.candidates.map((c) => c.cost.total)).toEqual(dbg.candidates.map((c) => c.cost.total));
    for (const c of dbg.candidates) if (c.hit) expect(c.cost.collision).toBeGreaterThan(0);
    expect(DEFAULT_MPC.weights.collision).toBeGreaterThan(100 * DEFAULT_MPC.weights.food);
  });

  it('sees only heads within the view radius', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 500, y: 500, heading: 0 });
    w.addSnake({ x: 800, y: 500, heading: 0 });
    w.addSnake({ x: 2500, y: 2500, heading: 0 });
    expect(visibleHeads(w, s).map((h) => h.id)).toEqual([1]);
  });
});

describe('heuristics', () => {
  it('survival turns away from a wall', () => {
    const w = emptyWorld();
    w.addSnake({ x: 2750, y: 1500, heading: 0 });
    expect(survive(w, 0, createHeuristic('survival'), 8)).toBe(true);
  });

  it('greedy heads for and eats a nearby pellet', () => {
    const w = emptyWorld();
    w.addSnake({ x: 1500, y: 1500, heading: 0 });
    w.addPellet(1500, 1700, 6, 'natural', -1);
    survive(w, 0, createHeuristic('greedy'), 4);
    expect(w.snakes[0].stats.pellets).toBe(1);
  });

  it('aggressive computes an intercept in front of a smaller head', () => {
    const w = emptyWorld();
    w.addSnake({ x: 1000, y: 1500, heading: 0, length: 900 });
    w.addSnake({ x: 1250, y: 1300, heading: Math.PI / 2, length: 240 });
    const d = createHeuristic('aggressive').decide(ctxFor(w, 0));
    expect(d.debug?.reason).toBe('attack');
    const [ix, iy] = d.debug!.intercept!;
    expect(iy).toBeGreaterThan(1300); // ahead of the target along its heading (+y)
    expect(Math.abs(ix - 1250)).toBeLessThan(1);
  });

  it('hybrid and potential field avoid the wall', () => {
    for (const id of ['hybrid', 'potential'] as const) {
      const w = emptyWorld();
      w.addSnake({ x: 2750, y: 1500, heading: 0 });
      expect(survive(w, 0, createHeuristic(id), 8), id).toBe(true);
    }
  });
});

describe('match layer', () => {
  const cfg = { seed: 21, agents: ['random', 'greedy', 'survival', 'aggressive', 'potential', 'hybrid', 'mpc'].map((policy) => ({ policy })), respawn: true };

  it('is reproducible and survives save/restore mid-match (policy state included)', () => {
    const a = new Match(cfg), b = new Match(cfg);
    for (let t = 0; t < 400; t++) { a.step(); b.step(); }
    expect(JSON.stringify(a.toJSON())).toBe(JSON.stringify(b.toJSON()));
    const snap = JSON.parse(JSON.stringify(a.toJSON()));
    for (let t = 0; t < 400; t++) a.step();
    const c = Match.fromJSON(snap);
    for (let t = 0; t < 400; t++) c.step();
    expect(JSON.stringify(c.toJSON())).toBe(JSON.stringify(a.toJSON()));
  });

  it('holds actions between decisions and applies them after the configured delay', () => {
    const m = new Match({ seed: 4, agents: [{ policy: 'random' }], decisionInterval: 3, actionDelay: 5 });
    const applied: number[] = [], decided: number[] = [];
    for (let t = 0; t < 60; t++) {
      decided.push(m.agents[0].lastDecided);
      m.step();
      applied.push(m.agents[0].held);
    }
    // The decision taken at tick 3k (visible as lastDecided from tick 3k + 1) is applied at tick 3k + 5.
    for (let k = 1; k < 15; k++) expect(applied[3 * k + 5]).toBe(decided[3 * k + 1]);
  });

  it('respawns dead snakes and keeps per-agent records', () => {
    const m = new Match({ ...cfg, agents: Array.from({ length: 10 }, () => ({ policy: 'random' })), respawnDelay: 1 });
    for (let t = 0; t < 1800; t++) m.step();
    const deaths = m.agents.reduce((a, x) => a + x.record.deaths, 0);
    expect(deaths).toBeGreaterThan(0);
    expect(m.world.snakes.some((s) => s.lives > 1)).toBe(true);
  });
});
