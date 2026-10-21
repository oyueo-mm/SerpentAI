/**
 * Engine bridge for Python (Method A). Protocol on stdin/stdout, little-endian:
 *   request  = u32 length + UTF-8 JSON {cmd, …}
 *   response = u32 length + UTF-8 JSON header [+ u32 length + float32 payload when header.bin is set]
 * Observations travel as one float32 block (numEnvs × OBS_SIZE, followed by terminal observations).
 */
import type { ModelStore } from '../../src/ai/registry';
import { OBS_SIZE, OBS_VERSION, encodeObs } from '../../src/observation/obs';
import { ACTION_VERSION, NUM_ACTIONS, decodeAction } from '../../src/policy/action';
import { MlpModel, type MlpModelFile } from '../../src/policy/mlp';
import { RULES_VERSION } from '../../src/sim/config';
import { senseWorld } from '../../src/observation/sensors';
import { LearnerEnv, REWARD_VERSION, type EnvConfig, type EpisodeInfo } from './envs';

const VERSIONS = { rules: RULES_VERSION, obs: OBS_VERSION, action: ACTION_VERSION, reward: REWARD_VERSION, obsSize: OBS_SIZE, numActions: NUM_ACTIONS };

type Request = { cmd: string } & Record<string, unknown>;

const models: ModelStore = new Map();
let envs: LearnerEnv[] = [];
let baseSeed = 0;
let episodeCounter = 0;

function nextSeed(): number {
  return (baseSeed + episodeCounter++) >>> 0;
}

function u32(n: number): Buffer {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(n, 0);
  return b;
}

function write(header: Record<string, unknown>, bin?: Float32Array): void {
  const json = Buffer.from(JSON.stringify(bin ? { ...header, bin: bin.byteLength } : header), 'utf8');
  const parts: Buffer[] = [u32(json.length), json];
  if (bin) parts.push(u32(bin.byteLength), Buffer.from(bin.buffer, bin.byteOffset, bin.byteLength));
  process.stdout.write(Buffer.concat(parts));
}

function pack(rows: Float32Array[]): Float32Array {
  const out = new Float32Array(rows.length * OBS_SIZE);
  rows.forEach((r, i) => out.set(r, i * OBS_SIZE));
  return out;
}

/** Runs full episodes with a policy driving the learner; learned models see env.observe() (noise included). */
function evaluate(policy: string, env: EnvConfig, seeds: number[], stochastic: boolean): (EpisodeInfo & { actions: number[] })[] {
  const model = models.get(policy);
  const e = new LearnerEnv(model ? env : { ...env, learnerPolicy: policy }, models);
  const out: (EpisodeInfo & { actions: number[] })[] = [];
  for (const seed of seeds) {
    let obs = e.reset(seed);
    const counts = new Array(NUM_ACTIONS).fill(0);
    let u = (seed * 2654435761) >>> 0;
    for (;;) {
      let a = 8;
      if (model) {
        const logits = model.logits(obs);
        const mask = Array.from({ length: NUM_ACTIONS }, (_, i) => e.canBoost() || (i & 1) === 0);
        if (stochastic) {
          let m = -Infinity;
          for (let i = 0; i < NUM_ACTIONS; i++) if (mask[i] && logits[i] > m) m = logits[i];
          const p = Array.from(logits, (l, i) => (mask[i] ? Math.exp(l - m) : 0));
          const z = p.reduce((x, y) => x + y, 0);
          u = (Math.imul(u, 1664525) + 1013904223) >>> 0;
          let r = (u / 4294967296) * z;
          a = NUM_ACTIONS - 1;
          for (let i = 0; i < NUM_ACTIONS; i++) { r -= p[i]; if (r < 0) { a = i; break; } }
        } else {
          let best = -Infinity;
          for (let i = 0; i < NUM_ACTIONS; i++) if (mask[i] && logits[i] > best) { best = logits[i]; a = i; }
        }
      }
      const r = e.step(a);
      counts[r.taken]++;
      obs = r.obs;
      if (r.info) { out.push({ ...r.info, actions: counts }); break; }
    }
  }
  return out;
}

function handle(req: Request): void {
  switch (req.cmd) {
    case 'init': {
      const n = Number(req.numEnvs);
      if (!Number.isInteger(n) || n < 1 || n > 512) throw new Error('numEnvs must be an integer in [1, 512]');
      baseSeed = Number(req.seed ?? 0) >>> 0;
      episodeCounter = 0;
      envs = Array.from({ length: n }, () => new LearnerEnv(req.env as EnvConfig, models));
      write({ ok: true, ...VERSIONS });
      return;
    }
    case 'setEnv':
      for (const e of envs) e.setConfig(req.env as EnvConfig);
      write({ ok: true });
      return;
    case 'addModel': {
      const m = new MlpModel(req.model as MlpModelFile);
      models.set(m.file.id, m);
      write({ ok: true, id: m.file.id });
      return;
    }
    case 'reset': {
      const seeds = Array.isArray(req.seeds) ? (req.seeds as number[]) : null;
      const rows = envs.map((e, i) => e.reset(seeds ? seeds[i] : nextSeed()));
      write({ ok: true, seeds: envs.map((e) => e.seed), canBoost: envs.map((e) => e.canBoost()), experts: envs.map((e) => e.expertAction()) }, pack(rows));
      return;
    }
    case 'step': {
      const actions = req.actions as number[];
      if (!Array.isArray(actions) || actions.length !== envs.length) throw new Error('actions length must equal numEnvs');
      const rows: Float32Array[] = [], terminal: Float32Array[] = [];
      const rewards: number[] = [], dones: boolean[] = [], truncated: boolean[] = [], taken: number[] = [], canBoost: boolean[] = [], experts: number[] = [];
      const infos: ((EpisodeInfo & { terminal_index: number }) | null)[] = [];
      envs.forEach((e, i) => {
        const r = e.step(actions[i]);
        rewards.push(r.reward);
        taken.push(r.taken);
        dones.push(r.terminated || r.truncated);
        truncated.push(r.truncated);
        if (r.info) {
          terminal.push(r.obs);
          infos.push({ ...r.info, terminal_index: terminal.length - 1 });
          rows.push(e.reset(nextSeed()));
          canBoost.push(e.canBoost());
          experts.push(e.expertAction());
        } else {
          infos.push(null);
          rows.push(r.obs);
          canBoost.push(r.canBoost);
          experts.push(r.expert);
        }
      });
      write({ ok: true, rewards, dones, truncated, infos, taken, canBoost, experts, terminals: terminal.length }, pack([...rows, ...terminal]));
      return;
    }
    case 'eval': {
      const res = evaluate(String(req.policy), req.env as EnvConfig, req.seeds as number[], req.stochastic === true);
      write({ ok: true, episodes: res });
      return;
    }
    case 'obs': {
      // Exact sensors + obs for env i (fixtures for the Python encoder port).
      const e = envs[Number(req.index ?? 0)];
      const s = e.match.world.snakes[0];
      const sensors = senseWorld(e.match.world, s);
      write({ ok: true, sensors, obs: Array.from(encodeObs(sensors, decodeAction(8), e.match.world.rules)) });
      return;
    }
    case 'snapshot': {
      const e = envs[Number(req.index ?? 0)];
      if (!e) throw new Error('no such env');
      write({ ok: true, match: e.match.toJSON() });
      return;
    }
    case 'close':
      write({ ok: true });
      process.exit(0);
    // eslint-disable-next-line no-fallthrough
    default:
      throw new Error(`unknown command ${req.cmd}`);
  }
}

let buffer: Buffer = Buffer.alloc(0);
process.stdin.on('data', (chunk: Buffer) => {
  buffer = buffer.length ? Buffer.concat([buffer, chunk]) : chunk;
  while (buffer.length >= 4) {
    const len = buffer.readUInt32LE(0);
    if (buffer.length < 4 + len) break;
    const text = buffer.subarray(4, 4 + len).toString('utf8');
    buffer = buffer.subarray(4 + len);
    try {
      handle(JSON.parse(text) as Request);
    } catch (err) {
      write({ ok: false, error: err instanceof Error ? err.stack ?? err.message : String(err) });
    }
  }
});
process.stdin.on('end', () => process.exit(0));
