import { describe, expect, it } from 'vitest';
import { distPointSegSq, wrapAngle } from '../geometry/geom';
import { DEFAULT_RULES, radiusFor, type RulesConfig } from './config';
import { Rng } from './rng';
import { World, type Control } from './world';

const R = DEFAULT_RULES;

/** Empty arena (no natural food) for controlled physics experiments. */
function emptyWorld(overrides: Partial<RulesConfig> = {}, seed = 1): World {
  return new World({ foodTarget: 0, ...overrides }, seed, { fillFood: false });
}

function run(w: World, ticks: number, controls: (Control | undefined)[] | ((t: number) => (Control | undefined)[])): void {
  for (let t = 0; t < ticks; t++) w.step(typeof controls === 'function' ? controls(t) : controls);
}

function bodyMaxSegment(w: World, id: number): { maxCommitted: number; neck: number } {
  const s = w.snakes[id], b = s.body;
  let maxCommitted = 0;
  for (let i = 0; i < b.vertexCount - 1; i++) maxCommitted = Math.max(maxCommitted, Math.hypot(b.vx(i + 1) - b.vx(i), b.vy(i + 1) - b.vy(i)));
  return { maxCommitted, neck: Math.hypot(s.x - b.lastX(), s.y - b.lastY()) };
}

describe('motion', () => {
  it('moves exactly baseSpeed u/s in a straight line', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 500, y: 1500, heading: 0 });
    run(w, R.tickRate * 2, []);
    expect(s.x).toBeCloseTo(500 + 2 * R.baseSpeed, 9);
    expect(s.y).toBeCloseTo(1500, 9);
  });

  it('limits the heading change per tick to maxTurnRate·dt', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 1500, y: 1500, heading: 0 });
    const before = s.heading;
    w.step([{ targetHeading: Math.PI / 2 }]);
    expect(wrapAngle(s.heading - before)).toBeCloseTo(R.maxTurnRate / R.tickRate, 12);
    // A small target change is reached exactly in one tick.
    w.step([{ targetHeading: s.heading + 0.01 }]);
    w.step([{ targetHeading: s.heading - 0.02 }]);
    expect(Math.abs(s.omega)).toBeLessThanOrEqual(R.maxTurnRate + 1e-12);
  });

  it('a constant full turn traces a circle of radius v/ω', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 1500, y: 1500, heading: 0 });
    const rad = R.baseSpeed / R.maxTurnRate;
    // Turning clockwise (+) on a y-down screen: the centre lies at heading + π/2.
    const cx = 1500, cy = 1500 + rad;
    for (let t = 0; t < 120; t++) {
      w.step([{ turn: 1 }]);
      expect(Math.hypot(s.x - cx, s.y - cy)).toBeCloseTo(rad, 6);
    }
    // Every committed vertex lies on the circle; the interpolated tail vertex lies on a chord, so it is
    // inside by at most the sagitta of one committed segment.
    const b = s.body;
    for (let i = 1; i < b.vertexCount; i++) expect(Math.abs(Math.hypot(b.vx(i) - cx, b.vy(i) - cy) - rad)).toBeLessThan(1e-6);
    const chord = R.pathSpacing + R.maxSubstepDistance;
    expect(rad - Math.hypot(b.vx(0) - cx, b.vy(0) - cy)).toBeLessThanOrEqual((chord * chord) / (8 * rad) + 1e-9);
  });

  it('keeps the body contiguous under alternating hard turns and boosts', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 1500, y: 1500, heading: 0, length: 600 });
    run(w, 600, (t) => [{ turn: (Math.floor(t / 7) % 2 ? 1 : -1), boost: t % 50 < 20 }]);
    const { maxCommitted, neck } = bodyMaxSegment(w, s.id);
    expect(maxCommitted).toBeLessThanOrEqual(R.pathSpacing + R.maxSubstepDistance + 1e-9);
    expect(neck).toBeLessThan(R.pathSpacing);
  });

  it('body arc length equals L once the tail has caught up', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 1500, y: 1500, heading: 0 });
    run(w, 90, [{ turn: 0.4 }]);
    expect(s.body.totalLength(s.x, s.y)).toBeCloseTo(s.length, 9);
  });

  it('is independent of the tick rate for constant controls', () => {
    const a = emptyWorld({ tickRate: 30 });
    const b = emptyWorld({ tickRate: 60 });
    const sa = a.addSnake({ x: 1500, y: 1500, heading: 0.3 });
    const sb = b.addSnake({ x: 1500, y: 1500, heading: 0.3 });
    run(a, 45, [{ turn: 0.7 }]);
    run(b, 90, [{ turn: 0.7 }]);
    expect(sa.x).toBeCloseTo(sb.x, 8);
    expect(sa.y).toBeCloseTo(sb.y, 8);
    expect(wrapAngle(sa.heading - sb.heading)).toBeCloseTo(0, 10);
    // Body shapes coincide: every vertex of one body is on the other's polyline.
    const pb = sb.body;
    for (let i = 0; i < sa.body.vertexCount; i += 5) {
      let best = Infinity;
      for (let k = 0; k < pb.vertexCount - 1; k++) best = Math.min(best, distPointSegSq(sa.body.vx(i), sa.body.vy(i), pb.vx(k), pb.vy(k), pb.vx(k + 1), pb.vy(k + 1)));
      expect(Math.sqrt(best)).toBeLessThan(0.15);
    }
  });
});

describe('food and growth', () => {
  it('absorbs a pellet on its path and grows by its value', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 500, y: 1500, heading: 0 });
    const p = w.addPellet(560, 1500 + 12, 5, 'natural', -1);
    run(w, 15, []);
    expect(w.food.has(p.id)).toBe(false);
    expect(s.length).toBe(R.startLength + 5);
    expect(s.stats.pellets).toBe(1);
  });

  it('a fast head does not skip a pellet between two sub-steps (swept mouth)', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 500, y: 1500, heading: 0, length: 400 });
    w.addPellet(503.3, 1500 + s.radius(R) + 2, 2, 'natural', -1);
    run(w, 3, [{ boost: true }]);
    expect(w.food.size).toBe(0);
  });

  it('growth extends the body as the head advances (tail waits)', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 500, y: 1500, heading: 0 });
    w.addPellet(510, 1500, 30, 'natural', -1);
    w.step([]);
    expect(s.length).toBe(R.startLength + 30);
    const tail0 = s.body.tailX();
    run(w, 3, []);
    expect(s.body.tailX()).toBeCloseTo(tail0, 9); // ≤ 4 ticks · 5 u = 20 u < 30 u since eating: tail has not moved yet
    run(w, 10, []);
    expect(s.body.totalLength(s.x, s.y)).toBeCloseTo(s.length, 9);
  });

  it('pellets claimed by two heads go to the closest one, deterministically', () => {
    const w = emptyWorld();
    // Heads 30 u apart (no contact: 2r = 20 u); both mouths reach the pellets (r + pellet r + reach ≈ 20.4 u).
    const a = w.addSnake({ x: 1000, y: 1000, heading: 0 });
    const b = w.addSnake({ x: 1000, y: 1030, heading: 0 });
    w.addPellet(1003, 1014, 4, 'natural', -1); // 14 u from a, 16 u from b → a
    w.addPellet(1003, 1017, 4, 'natural', -1); // 17 u from a, 13 u from b → b
    w.step([]);
    expect(a.length).toBe(R.startLength + 4);
    expect(b.length).toBe(R.startLength + 4);
    expect(w.food.size).toBe(0);
  });
});

describe('boost', () => {
  it('doubles speed and costs boostCost u/s, dropping half of it as pellets', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 500, y: 1500, heading: 0, length: 400 });
    run(w, R.tickRate, [{ boost: true }]);
    expect(s.x).toBeCloseTo(500 + R.boostSpeed, 6);
    expect(s.length).toBeCloseTo(400 - R.boostCost, 9);
    const dropped = [...w.food.values()].reduce((a, p) => a + p.value, 0);
    expect(dropped + s.dropAcc).toBeCloseTo(R.boostCost * R.boostDropFraction, 9);
  });

  it('cannot boost at or below boostMinLength, and length never drops below it', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 500, y: 1500, heading: 0, length: R.boostMinLength + 3 });
    run(w, R.tickRate, [{ boost: true }]);
    expect(s.length).toBe(R.boostMinLength);
    const x = s.x;
    w.step([{ boost: true }]);
    expect(s.boosting).toBe(false);
    expect(s.x - x).toBeCloseTo(R.baseSpeed / R.tickRate, 9);
  });
});

describe('collisions', () => {
  it('a head running into another body dies and credits the owner', () => {
    const w = emptyWorld();
    const a = w.addSnake({ x: 1000, y: 900, heading: Math.PI / 2 });
    const b = w.addSnake({ x: 1200, y: 1000, heading: 0, length: 600 }); // body along y = 1000, x ∈ [600, 1200]
    run(w, 30, []);
    expect(a.alive).toBe(false);
    expect(a.deathCause).toBe('body');
    expect(a.killer).toBe(b.id);
    expect(b.stats.kills).toBe(1);
    expect(b.alive).toBe(true);
  });

  it('ignores its own body (self-collision is off in rules-v1)', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 1500, y: 1500, heading: 0, length: 2000 });
    run(w, 300, [{ turn: 1 }]); // tight circles overlap its own body many times
    expect(s.alive).toBe(true);
  });

  it('head-on with equal lengths kills both; unequal kills the shorter', () => {
    const w = emptyWorld();
    const a = w.addSnake({ x: 1000, y: 1000, heading: 0 });
    const b = w.addSnake({ x: 1100, y: 1000, heading: Math.PI });
    run(w, 15, []);
    expect([a.alive, b.alive]).toEqual([false, false]);
    expect([a.deathCause, b.deathCause]).toEqual(['headon', 'headon']);

    const w2 = emptyWorld();
    const c = w2.addSnake({ x: 1000, y: 1000, heading: 0, length: 300 });
    const d = w2.addSnake({ x: 1100, y: 1000, heading: Math.PI, length: 260 });
    run(w2, 15, []);
    expect(c.alive).toBe(true);
    expect(d.alive).toBe(false);
    expect(d.killer).toBe(c.id);
  });

  it('dies at the wall when the head circle touches it', () => {
    const w = emptyWorld();
    const s = w.addSnake({ x: 2950, y: 1500, heading: 0 });
    run(w, 30, []);
    expect(s.alive).toBe(false);
    expect(s.deathCause).toBe('wall');
    expect(R.worldSize - s.x).toBeLessThanOrEqual(radiusFor(R, s.length) + 1e-9);
    expect(R.worldSize - s.x).toBeGreaterThan(radiusFor(R, s.length) - R.maxSubstepDistance);
  });

  /** Fires a boosting head across the thinnest moving body at 12 sub-step phases × 4 angles; returns survivors. */
  function crossThinBody(overrides: Partial<RulesConfig>): { survivors: string[]; crossings: number } {
    const survivors: string[] = [];
    let crossings = 0;
    for (let phase = 0; phase < 12; phase++) {
      for (const angle of [Math.PI / 2, Math.PI / 3, Math.PI / 5, (2 * Math.PI) / 3]) {
        const w = emptyWorld(overrides);
        const RR = w.rules;
        // Thinnest legal body (L = minLength) moving right along y = 1000 at base speed.
        const thin = w.addSnake({ x: 1700, y: 1000, heading: 0, length: RR.minLength });
        const gap = 120 + phase * 0.83 * (RR.boostSpeed / 300); // sub-step phase offsets relative to the body line
        const T = gap / (RR.boostSpeed * Math.sin(angle)); // time for the crosser to reach y = 1000
        const xAim = 1700 - RR.minLength / 2 + RR.baseSpeed * T; // middle of the moving body at that moment
        const a = w.addSnake({ x: xAim - RR.boostSpeed * Math.cos(angle) * T, y: 1000 - gap, heading: angle, length: 900 });
        run(w, 40, [{}, { boost: true }]);
        const tag = `phase ${phase} angle ${angle.toFixed(2)}`;
        if (a.alive || a.killer !== thin.id) survivors.push(tag);
        else expect(a.deathCause, tag).toBe('body');
        crossings++;
      }
    }
    return { survivors, crossings };
  }

  it('never tunnels: a boosting head always dies crossing the thinnest body, at any phase and angle', () => {
    const r = crossThinBody({});
    expect(r.crossings).toBe(48);
    expect(r.survivors).toEqual([]);
  });

  it('sub-stepping is what prevents tunnelling at extreme speed (2400 u/s boost)', () => {
    // 80 u per tick > the 2(r_a + r_b) ≈ 48 u contact band: one discrete check per tick tunnels…
    const without = crossThinBody({ boostSpeed: 2400, maxSubstepDistance: 1000 });
    expect(without.survivors.length).toBeGreaterThan(0);
    // …while the fixed sub-step bound (≤ 5 u per check) catches every crossing.
    const withSub = crossThinBody({ boostSpeed: 2400 });
    expect(withSub.survivors).toEqual([]);
  });

  it('turns a dead snake into pellets worth deathDropFraction of its length', () => {
    const w = emptyWorld();
    const a = w.addSnake({ x: 1000, y: 900, heading: Math.PI / 2, length: 500 });
    w.addSnake({ x: 1200, y: 1000, heading: 0, length: 600 });
    run(w, 30, []);
    expect(a.alive).toBe(false);
    const fromA = [...w.food.values()].filter((p) => p.source === a.id && p.kind === 'death');
    expect(fromA.length).toBeGreaterThan(10);
    expect(fromA.reduce((s, p) => s + p.value, 0)).toBeCloseTo(R.deathDropFraction * 500, 6);
  });
});

/** Random controls for every snake, from a dedicated RNG. */
function randomControls(rng: Rng, n: number): Control[] {
  return Array.from({ length: n }, () => ({ turn: rng.range(-1, 1), boost: rng.next() < 0.15 }));
}

function populated(seed: number, snakes = 16): World {
  const w = new World({}, seed);
  for (let i = 0; i < snakes; i++) w.addSnake();
  return w;
}

describe('determinism and persistence', () => {
  it('same seed + same inputs → identical state', () => {
    const a = populated(42), b = populated(42);
    const ra = new Rng(7), rb = new Rng(7);
    for (let t = 0; t < 600; t++) {
      a.step(randomControls(ra, a.snakes.length));
      b.step(randomControls(rb, b.snakes.length));
    }
    expect(JSON.stringify(a.toJSON())).toBe(JSON.stringify(b.toJSON()));
    expect(a.snakes.some((s) => !s.alive)).toBe(true); // the run actually exercised deaths
  });

  it('different seeds diverge', () => {
    expect(JSON.stringify(populated(1).toJSON())).not.toBe(JSON.stringify(populated(2).toJSON()));
  });

  it('save → restore → continue matches the uninterrupted run bit-for-bit', () => {
    const a = populated(9);
    const rng = new Rng(3);
    const inputs = Array.from({ length: 900 }, () => randomControls(rng, a.snakes.length));
    for (let t = 0; t < 400; t++) a.step(inputs[t]);
    const snap = JSON.parse(JSON.stringify(a.toJSON()));
    for (let t = 400; t < 900; t++) a.step(inputs[t]);
    const b = World.fromJSON(snap);
    for (let t = 400; t < 900; t++) b.step(inputs[t]);
    expect(JSON.stringify(b.toJSON())).toBe(JSON.stringify(a.toJSON()));
  });

  it('rejects snapshots from another rules version', () => {
    const snap = { ...populated(1).toJSON(), rulesVersion: 'rules-v0' };
    expect(() => World.fromJSON(snap)).toThrow(/rules-v0/);
  });
});

describe('invariant fuzzing (random play, 16 snakes)', () => {
  /**
   * Exact per-tick mass law, with M = Σ living lengths + Σ pellet values + Σ living boost accumulators:
   *   M' = M + new natural pellets − (1 − boostDropFraction)·boost loss − Σ dead (L + acc) + deathDropFraction·Σ dead L.
   * Eating moves mass from pellets to snakes, so a pellet eaten twice (or a phantom pellet) breaks the law.
   */
  it('conserves mass exactly, keeps bodies valid and inside the arena', () => {
    for (const seed of [11, 12, 13]) {
      const w = populated(seed);
      const rng = new Rng(seed * 31);
      const R2 = w.rules;
      const mass = () => w.aliveSnakes().reduce((a, s) => a + s.length + s.dropAcc, 0) + [...w.food.values()].reduce((a, p) => a + p.value, 0);
      let deaths = 0;
      for (let t = 0; t < 900; t++) {
        const before = new Map(w.aliveSnakes().map((s) => [s.id, s.length]));
        const nextIdBefore = w.nextFoodId;
        const m0 = mass();
        w.step(randomControls(rng, w.snakes.length));
        let expected = m0;
        for (const p of w.food.values()) if (p.id >= nextIdBefore && p.kind === 'natural') expected += p.value;
        const deadLen = new Map(w.deaths.map((d) => [d.id, d.length]));
        for (const [id, L0] of before) {
          const s = w.snakes[id];
          const Lend = deadLen.get(id) ?? s.length;
          const boostLoss = L0 + s.eatenThisTick - Lend;
          expect(boostLoss).toBeGreaterThanOrEqual(-1e-9);
          expected -= (1 - R2.boostDropFraction) * boostLoss;
          if (deadLen.has(id)) expected -= Lend + s.dropAcc - R2.deathDropFraction * Lend;
        }
        deaths += w.deaths.length;
        expect(Math.abs(mass() - expected)).toBeLessThan(1e-6);
        for (const s of w.aliveSnakes()) {
          const r = s.radius(R2);
          expect(Math.min(s.x, s.y)).toBeGreaterThanOrEqual(r);
          expect(Math.max(s.x, s.y)).toBeLessThanOrEqual(R2.worldSize - r);
          expect(s.body.totalLength(s.x, s.y)).toBeLessThanOrEqual(s.length + 1e-6);
          expect(s.length).toBeGreaterThanOrEqual(R2.minLength);
        }
      }
      expect(deaths).toBeGreaterThan(0);
    }
  });
});
