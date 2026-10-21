import { describe, expect, it } from 'vitest';
import { createHeuristic } from '../ai/heuristics';
import { MlpModel, type MlpModelFile } from '../policy/mlp';
import { AutoQuality, autoLevel } from '../render/quality';
import { Match } from '../sim/match';
import { Rng } from '../sim/rng';
import { World } from '../sim/world';
import { OBS_SIZE, OBS_V2_SIZE, encodeObs, encodeObsV2 } from './obs';
import { OBS_V1_RANGES, rangesFor, senseWorld } from './sensors';

function busyWorld(): World {
  const w = new World({}, 12);
  for (let i = 0; i < 14; i++) w.addSnake();
  for (let t = 0; t < 90; t++) w.step(w.snakes.map(() => ({ turn: 0.4 })));
  return w;
}

describe('sensing ranges (obs-v1 fixed, heuristics configurable)', () => {
  it('legacy readings are exactly obs-v1 (no ranges field) and the setting is clamped to 200–480 u', () => {
    const w = busyWorld();
    const s = w.aliveSnakes()[0];
    expect(senseWorld(w, s)).not.toHaveProperty('ranges');
    expect(senseWorld(w, s, rangesFor(undefined))).toEqual(senseWorld(w, s));
    expect(rangesFor(1000)).toEqual({ ray: 480, food: 480, opp: 480 });
    expect(rangesFor(50)).toEqual({ ray: 200, food: 200, opp: 200 });
    expect(rangesFor(undefined)).toEqual(OBS_V1_RANGES);
  });

  it('a wider range sees farther; obs-v1 refuses non-obs-v1 readings; obs-v2 normalises by the range', () => {
    const w = busyWorld();
    for (const s of w.aliveSnakes()) {
      const v1 = senseWorld(w, s), wide = senseWorld(w, s, rangesFor(480));
      for (let i = 0; i < 16; i++) expect(wide.rayBody[i]).toBeLessThanOrEqual(480);
      expect(wide.foodSum.reduce((a, b) => a + b, 0)).toBeGreaterThanOrEqual(v1.foodSum.reduce((a, b) => a + b, 0) - 1e-9);
      expect(() => encodeObs(wide, { turn: 0, boost: false })).toThrow(/obs-v1/);
      const o2 = encodeObsV2(wide, { turn: 0, boost: false });
      expect(o2.length).toBe(OBS_V2_SIZE);
      expect(OBS_V2_SIZE).toBe(OBS_SIZE + 1);
      expect(o2[OBS_SIZE]).toBe(1);
      for (let i = 0; i < 16; i++) expect(o2[6 + 2 * i]).toBeCloseTo(Math.min(1, wide.rayBody[i] / 480), 5);
    }
  });

  it('learned obs-v1 models decide identically whatever sensing distance the match sets for heuristics', () => {
    const r = new Rng(3);
    const layer = (i: number, o: number, act: 'tanh' | 'none') => ({ w: Array.from({ length: o }, () => Array.from({ length: i }, () => r.normal() * 0.3)), b: new Array(o).fill(0), act });
    const file: MlpModelFile = { format: 'serpent-mlp', id: 'm', label: 'm', algo: 'ppo', obsVersion: 'obs-v1', actionVersion: 'act-v1', rulesVersion: 'rules-v1',
      obsMean: null, obsVar: null, obsEps: 1e-8, obsClip: 10, layers: [layer(84, 32, 'tanh'), layer(32, 18, 'none')], meta: {} };
    const models = new Map([['m', new MlpModel(file)]]);
    const run = (sensing?: number) => {
      const m = new Match({ seed: 5, agents: [{ policy: 'm' }, { policy: 'greedy' }, { policy: 'hybrid' }], sensing }, models);
      m.step();
      return JSON.stringify(m.agents[0].debug?.probs);
    };
    expect(run(200)).toBe(run(undefined));
    expect(run(480)).toBe(run(undefined));
  });

  it('heuristics really sense less at a shorter distance', () => {
    const w = new World({ foodTarget: 0 }, 1, { fillFood: false });
    w.addSnake({ x: 1500, y: 1500, heading: 0 });
    w.addSnake({ x: 1500, y: 1800, heading: 0, length: 900 }); // body ~300 u to the right of the first head
    const ctx = (sensing?: number) => ({ world: w, self: w.snakes[0], rng: new Rng(1), holdTicks: 3, prevAction: 8, sensing,
      sensors: () => senseWorld(w, w.snakes[0], rangesFor(sensing)) });
    const near = createHeuristic('survival').decide(ctx(200)).debug!.danger!;
    const far = createHeuristic('survival').decide(ctx(480)).debug!.danger!;
    expect(Math.max(...near)).toBe(0);           // 300 u away: invisible at 200 u
    expect(Math.max(...far)).toBeGreaterThan(0); // visible at 480 u
  });
});

describe('graphics quality (render-only)', () => {
  it('auto starts high on desktop, medium on phones, low on low-end phones', () => {
    expect(autoLevel({ coarsePointer: false, cores: 4 })).toBe('high');
    expect(autoLevel({ coarsePointer: true, cores: 8, memoryGB: 8 })).toBe('medium');
    expect(autoLevel({ coarsePointer: true, cores: 4 })).toBe('low');
    expect(autoLevel({ coarsePointer: true, cores: 8, memoryGB: 2 })).toBe('low');
  });

  it('auto steps down after two slow readings and never oscillates back', () => {
    const a = new AutoQuality('high');
    expect(a.observe(40)).toBe('high');
    expect(a.observe(41)).toBe('medium');
    expect(a.observe(60)).toBe('medium');
    a.observe(30); expect(a.observe(30)).toBe('low');
    a.observe(5); expect(a.observe(5)).toBe('low');
    expect(a.observe(60)).toBe('low');
  });
});
