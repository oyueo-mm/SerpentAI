import { describe, expect, it } from 'vitest';
import { ACTION_VERSION, NUM_ACTIONS, actionMask, controlToAction, decodeAction, encodeAction } from '../policy/action';
import { DEFAULT_RULES } from '../sim/config';
import { World } from '../sim/world';
import { Pacer } from '../worker/pacing';
import { Rng } from '../sim/rng';
import { OBS_SIZE, OBS_VERSION, encodeObs, obsFeatureNames } from './obs';
import { senseWorld } from './sensors';

describe('obs-v1 contract', () => {
  it('has 84 named features, all within the declared box', () => {
    expect(OBS_VERSION).toBe('obs-v1');
    expect(OBS_SIZE).toBe(84);
    expect(obsFeatureNames()).toHaveLength(OBS_SIZE);
    const w = new World({}, 4);
    for (let i = 0; i < 12; i++) w.addSnake();
    for (let t = 0; t < 200; t++) w.step(w.snakes.map(() => ({ turn: 0.3 })));
    for (const s of w.aliveSnakes()) {
      const o = encodeObs(senseWorld(w, s), { turn: 0.5, boost: true });
      for (const v of o) { expect(v).toBeGreaterThanOrEqual(-1); expect(v).toBeLessThanOrEqual(2); }
    }
  });

  it('is egocentric: mirroring/rotating the whole scene leaves the observation unchanged', () => {
    // Two worlds that are the same scene rotated by 90° about the arena centre.
    const C = DEFAULT_RULES.worldSize / 2;
    const rot = (x: number, y: number): [number, number] => [C - (y - C), C + (x - C)];
    const a = new World({ foodTarget: 0 }, 1, { fillFood: false });
    const b = new World({ foodTarget: 0 }, 1, { fillFood: false });
    const scene = [{ x: 1500, y: 1500, h: 0.3 }, { x: 1700, y: 1450, h: 2.0 }, { x: 1350, y: 1650, h: -1.0 }];
    for (const s of scene) {
      a.addSnake({ x: s.x, y: s.y, heading: s.h });
      const [x, y] = rot(s.x, s.y);
      b.addSnake({ x, y, heading: s.h + Math.PI / 2 });
    }
    for (const [x, y, v] of [[1600, 1500, 4], [1500, 1300, 6], [1420, 1580, 3]]) {
      a.addPellet(x, y, v, 'natural', -1);
      const [rx, ry] = rot(x, y);
      b.addPellet(rx, ry, v, 'natural', -1);
    }
    const oa = encodeObs(senseWorld(a, a.snakes[0]), { turn: 0, boost: false });
    const ob = encodeObs(senseWorld(b, b.snakes[0]), { turn: 0, boost: false });
    for (let i = 0; i < OBS_SIZE; i++) {
      if (obsFeatureNames()[i].endsWith('.wall')) continue; // walls are not part of the rotated scene's symmetry
      expect(ob[i], obsFeatureNames()[i]).toBeCloseTo(oa[i], 4);
    }
  });
});

describe('act-v1 contract', () => {
  it('round-trips all 18 actions and masks boost', () => {
    expect(ACTION_VERSION).toBe('act-v1');
    for (let a = 0; a < NUM_ACTIONS; a++) {
      const { turn, boost } = decodeAction(a);
      expect(encodeAction(turn, boost)).toBe(a);
    }
    expect(actionMask(false).filter(Boolean)).toHaveLength(9);
    expect(() => decodeAction(18)).toThrow();
  });

  it('maps a target heading to the turn level that reaches it over the hold', () => {
    const R = DEFAULT_RULES;
    const quarter = (R.maxTurnRate * 3) / R.tickRate * 0.5; // half the max turn over 3 ticks
    expect(decodeAction(controlToAction({ targetHeading: quarter }, 0, R, 3)).turn).toBe(0.5);
    expect(decodeAction(controlToAction({ targetHeading: Math.PI }, 0, R, 3)).turn).toBe(1);
  });
});

describe('pacing', () => {
  it('runs speed × tickRate ticks per second regardless of how real time is sliced', () => {
    for (const slices of [[16, 16, 16], [5, 40, 3, 100, 8], [33, 33, 33, 33]]) {
      const p = new Pacer(2, 30);
      let now = 0, ticks = 0;
      p.reset(now);
      const total = slices.reduce((a, b) => a + b, 0) * 10;
      while (now < total) for (const s of slices) { now += s; ticks += p.due(now, 1000); }
      expect(Math.abs(ticks - (now / 1000) * 2 * 30)).toBeLessThanOrEqual(1);
    }
  });

  it('the simulation outcome does not depend on frame timing (fixed ticks only)', () => {
    const run = (chunks: number[]) => {
      const w = new World({}, 8);
      for (let i = 0; i < 6; i++) w.addSnake();
      const rng = new Rng(1);
      for (const n of chunks) for (let k = 0; k < n; k++) w.step(w.snakes.map(() => ({ turn: rng.range(-1, 1) })));
      return JSON.stringify(w.toJSON());
    };
    expect(run([300])).toBe(run([1, 7, 92, 200]));
  });
});
