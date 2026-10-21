// Headless tournament CLI: node dist-node/tournament.mjs [--rounds N] [--copies K] [--duration S] [--policies a,b,c] [--models dir] [--out file]
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { BUILTIN_POLICIES } from '../ai/registry';
import { runTournament } from '../sim/tournament';
import { loadModelDir } from './models';

function arg(name: string, def?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}

const models = loadModelDir(arg('models', 'public/models')!);
const policies = (arg('policies') ?? [...BUILTIN_POLICIES.filter((p) => p !== 'mpc-lite'), ...models.keys()].join(',')).split(',').filter(Boolean);
const rounds = Number(arg('rounds', '10'));
const copies = Number(arg('copies', '2'));
const durationS = Number(arg('duration', '90'));
const foodTarget = arg('food') ? Number(arg('food')) : undefined;
const seedBase = arg('seed') ? Number(arg('seed')) : undefined;
const out = arg('out');

console.log(`tournament: ${policies.join(', ')} · ${rounds} rounds · ${copies} copies · ${durationS}s${foodTarget ? ` · food ${foodTarget}` : ''}`);
const res = runTournament({ policies, rounds, copies, durationS, seedBase, rules: foodTarget ? { foodTarget } : undefined }, models, (i, r) => {
  process.stdout.write(`  round ${i + 1}/${rounds} seed ${r.seed}: ${(r.ticks / 30).toFixed(0)}s, winner ${r.entries.find((e) => e.rank === 1)?.policy}\n`);
});
const f = (x: { mean: number; lo: number; hi: number }, d = 1) => `${x.mean.toFixed(d)} [${x.lo.toFixed(d)}, ${x.hi.toFixed(d)}]`;
console.log('\npolicy        win%            rank           survival s        max length            kills          death%  wall%');
for (const s of [...res.summary].sort((a, b) => a.rank.mean - b.rank.mean)) {
  console.log(`${s.policy.padEnd(13)} ${f({ mean: s.winRate.mean * 100, lo: s.winRate.lo * 100, hi: s.winRate.hi * 100 }, 0).padEnd(15)} ${f(s.rank).padEnd(14)} ${f(s.survivalS).padEnd(17)} ${f(s.maxLength, 0).padEnd(21)} ${f(s.kills, 2).padEnd(14)} ${(s.deathRate.mean * 100).toFixed(0).padStart(5)} ${(s.wallDeathRate.mean * 100).toFixed(0).padStart(6)}`);
}
console.log(`\nwall time ${(res.wallMs / 1000).toFixed(1)} s`);
if (out) {
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify({ ...res, generatedAt: new Date().toISOString(), node: process.version, platform: process.platform }, null, 1));
  console.log(`wrote ${out}`);
}
