// Performance benchmark: node dist-node/benchmark.mjs [--out public/data/benchmark.json] [--quick]
import { cpus, totalmem, platform, release } from 'node:os';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { performance } from 'node:perf_hooks';
import { createPolicy } from '../ai/registry';
import { encodeObs } from '../observation/obs';
import { senseWorld } from '../observation/sensors';
import { MlpModel, type MlpModelFile } from '../policy/mlp';
import { Rng } from '../sim/rng';
import { World } from '../sim/world';
import { LearnerEnv } from '../../rl/bridge/envs';
import { loadModelDir } from './models';

const quick = process.argv.includes('--quick');
const outIdx = process.argv.indexOf('--out');
const out = outIdx >= 0 ? process.argv[outIdx + 1] : undefined;

function time(fn: () => void, minMs = quick ? 150 : 600): { perCallMs: number; calls: number } {
  for (let i = 0; i < 3; i++) fn();
  let calls = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < minMs) { fn(); calls++; }
  return { perCallMs: (performance.now() - t0) / calls, calls };
}

function populated(n: number, length: number, seed = 1): World {
  const w = new World({}, seed);
  for (let i = 0; i < n; i++) w.addSnake({ length });
  return w;
}

/**
 * Mean wall time of world.step() alone under random steering. Dead snakes are respawned outside the
 * timed region (spawn search is not part of a tick's cost), so the snake count stays constant.
 */
function timeSteps(w: World, rng: Rng, minMs = quick ? 150 : 600): number {
  let spent = 0, n = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < minMs) {
    const controls = w.snakes.map(() => ({ turn: rng.range(-0.6, 0.6), boost: false }));
    const a = performance.now();
    w.step(controls);
    spent += performance.now() - a;
    n++;
    for (const s of w.snakes) if (!s.alive) w.respawn(s.id, { length: s.stats.maxLength });
  }
  return spent / n;
}

const results: Record<string, unknown> = {};

// 1. Engine tick cost vs number of snakes (random steering, respawn keeps the count fixed).
const bySnakes: { snakes: number; tickMs: number; ups: number }[] = [];
for (const n of [5, 10, 20, 30, 50]) {
  const w = populated(n, 400);
  const tickMs = timeSteps(w, new Rng(n));
  bySnakes.push({ snakes: n, tickMs, ups: 1000 / tickMs });
}
results.engineBySnakes = bySnakes;

// 2. Engine tick cost vs body length (20 snakes).
const byLength: { length: number; vertices: number; tickMs: number }[] = [];
for (const L of [240, 1000, 3000, 6000]) {
  const w = populated(20, L);
  const vertices = w.snakes.reduce((a, s) => a + s.body.vertexCount, 0);
  byLength.push({ length: L, vertices, tickMs: timeSteps(w, new Rng(L)) });
}
results.engineByLength = byLength;

// 3. Collision phase alone: grid rebuild + death detection (per sub-step), 20 snakes × L = 1000.
{
  const w = populated(20, 1000);
  const anyW = w as unknown as { detectDeaths(px: Float64Array, py: Float64Array): unknown };
  const px = new Float64Array(20), py = new Float64Array(20);
  w.snakes.forEach((s) => { px[s.id] = s.x; py[s.id] = s.y; });
  const rebuild = time(() => w.rebuildGrid());
  const detect = time(() => anyW.detectDeaths(px, py));
  results.collision = { snakes: 20, length: 1000, rebuildMs: rebuild.perCallMs, detectMs: detect.perCallMs, segments: w.segGrid.count };
}

// 4. Policy decision time in a 15-snake arena.
{
  const w = populated(15, 600, 3);
  for (let t = 0; t < 60; t++) w.step([]);
  const ids = ['random', 'greedy', 'survival', 'aggressive', 'potential', 'hybrid', 'mpc-lite', 'mpc'];
  const decisions: { policy: string; decideMs: number }[] = [];
  for (const id of ids) {
    const p = createPolicy(id, new Map());
    const s = w.snakes.find((x) => x.alive)!;
    const rng = new Rng(5);
    const r = time(() => p.decide({ world: w, self: s, rng, holdTicks: 3, sensors: () => senseWorld(w, s), prevAction: 8 }));
    decisions.push({ policy: id, decideMs: r.perCallMs });
  }
  results.decisions = decisions;
  const s = w.snakes.find((x) => x.alive)!;
  results.observation = { senseAndEncodeMs: time(() => encodeObs(senseWorld(w, s), { turn: 0, boost: false })).perCallMs };
}

// 5. MLP inference (shipped models if present, else a random 84-128-128-18 network).
{
  const models = loadModelDir('public/models');
  let model = [...models.values()][0];
  if (!model) {
    const r = new Rng(9);
    const layer = (i: number, o: number, act: 'tanh' | 'none') => ({ w: Array.from({ length: o }, () => Array.from({ length: i }, () => r.normal() * 0.1)), b: new Array(o).fill(0), act });
    const f: MlpModelFile = { format: 'serpent-mlp', id: 'bench', label: 'bench', algo: 'ppo', obsVersion: 'obs-v1', actionVersion: 'act-v1', rulesVersion: 'rules-v1', obsMean: null, obsVar: null, obsEps: 1e-8, obsClip: 10, layers: [layer(84, 128, 'tanh'), layer(128, 128, 'tanh'), layer(128, 18, 'none')], meta: {} };
    model = new MlpModel(f);
  }
  const x = new Float32Array(84).fill(0.3);
  results.mlp = { model: model.file.id, layers: model.file.layers.map((l) => `${l.w[0].length}→${l.b.length}`), inferMs: time(() => model.logits(x)).perCallMs };
}

// 6. RL environment step throughput (one env, single thread) per scenario.
{
  const envRes: { scenario: string; stepsPerSec: number; ticksPerSec: number }[] = [];
  for (const scenario of ['A-food', 'C-moving', 'D-mixed'] as const) {
    const e = new LearnerEnv({ scenario }, new Map());
    e.reset(1);
    const rng = new Rng(2);
    const r = time(() => { const res = e.step(rng.int(0, 17)); if (res.info) e.reset(rng.nextU32()); }, quick ? 300 : 1500);
    envRes.push({ scenario, stepsPerSec: 1000 / r.perCallMs, ticksPerSec: 3000 / r.perCallMs });
  }
  results.envThroughput = envRes;
}

const env = { node: process.version, platform: `${platform()} ${release()}`, cpu: cpus()[0]?.model ?? 'unknown', cores: cpus().length, memGB: Math.round(totalmem() / 2 ** 30), date: new Date().toISOString() };
console.log(JSON.stringify({ env, ...results }, null, 2));
if (out) {
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify({ env, ...results }, null, 1));
  console.log(`wrote ${out}`);
}
