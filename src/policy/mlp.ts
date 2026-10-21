import { ACTION_VERSION, NUM_ACTIONS, actionMask, decodeAction } from './action';
import { OBS_SIZE, OBS_VERSION, encodeObs } from '../observation/obs';
import { senseWorld } from '../observation/sensors';
import { RULES_VERSION } from '../sim/config';
import type { Decision, DecisionContext, Policy } from '../ai/types';

/**
 * Exported policy network (rl/export.py writes this format). Inference here must match PyTorch to
 * within 1e-4 on the same obs-v1 input (checked by src/policy/mlp.test.ts against Python fixtures).
 */
export interface MlpLayer {
  /** Weight matrix, out × in (row-major nested arrays). */
  w: number[][];
  b: number[];
  act: 'tanh' | 'relu' | 'none';
}

export interface MlpModelFile {
  format: 'serpent-mlp';
  id: string;
  label: string;
  /** Training algorithm family shown in the UI: never a heuristic. */
  algo: 'bc' | 'dagger' | 'ppo';
  obsVersion: string;
  actionVersion: string;
  rulesVersion: string;
  /** Observation normalisation (VecNormalize-style): (obs − mean) / sqrt(var + eps), clipped. */
  obsMean: number[] | null;
  obsVar: number[] | null;
  obsEps: number;
  obsClip: number;
  layers: MlpLayer[];
  /** Free-form provenance written by the exporter (steps, seeds, date, init checkpoint …). */
  meta: Record<string, unknown>;
}

export class MlpModel {
  constructor(readonly file: MlpModelFile) {
    if (file.format !== 'serpent-mlp') throw new Error(`${file.id}: not a serpent-mlp file`);
    if (file.obsVersion !== OBS_VERSION) throw new Error(`${file.id}: model expects ${file.obsVersion}, encoder is ${OBS_VERSION}`);
    if (file.actionVersion !== ACTION_VERSION) throw new Error(`${file.id}: model uses ${file.actionVersion}, game uses ${ACTION_VERSION}`);
    if (file.layers[0]?.w[0]?.length !== OBS_SIZE) throw new Error(`${file.id}: first layer expects ${file.layers[0]?.w[0]?.length} inputs, obs is ${OBS_SIZE}`);
    if (file.layers[file.layers.length - 1].b.length !== NUM_ACTIONS) throw new Error(`${file.id}: output size is not ${NUM_ACTIONS}`);
  }

  normalize(obs: ArrayLike<number>): Float64Array {
    const f = this.file;
    const x = new Float64Array(obs.length);
    for (let i = 0; i < obs.length; i++) {
      let v = obs[i];
      if (f.obsMean && f.obsVar) v = (v - f.obsMean[i]) / Math.sqrt(f.obsVar[i] + f.obsEps);
      x[i] = Math.max(-f.obsClip, Math.min(f.obsClip, v));
    }
    return x;
  }

  /** Raw action logits for an obs-v1 vector. */
  logits(obs: ArrayLike<number>): Float64Array {
    let x = this.normalize(obs);
    for (const layer of this.file.layers) {
      const y = new Float64Array(layer.b.length);
      for (let o = 0; o < y.length; o++) {
        const row = layer.w[o];
        let s = layer.b[o];
        for (let i = 0; i < row.length; i++) s += row[i] * x[i];
        y[o] = layer.act === 'tanh' ? Math.tanh(s) : layer.act === 'relu' ? (s > 0 ? s : 0) : s;
      }
      x = y;
    }
    return x;
  }
}

export function maskedSoftmax(logits: ArrayLike<number>, mask: boolean[]): number[] {
  let m = -Infinity;
  for (let i = 0; i < logits.length; i++) if (mask[i] && logits[i] > m) m = logits[i];
  const e = Array.from(logits, (l, i) => (mask[i] ? Math.exp(l - m) : 0));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

/** Policy backed by an exported network; deterministic arg-max by default. */
export class MlpPolicy implements Policy {
  readonly kind = 'learned' as const;
  readonly id: string;

  constructor(readonly model: MlpModel, readonly stochastic = false) {
    this.id = model.file.id;
  }

  decide(ctx: DecisionContext): Decision {
    const prev = decodeAction(ctx.prevAction);
    // Always obs-v1 sensors with obs-v1 ranges, whatever sensing distance the match configures for
    // heuristics: a model must see exactly the inputs it was trained on.
    const obs = encodeObs(senseWorld(ctx.world, ctx.self), prev, ctx.world.rules);
    return this.decideFromObs(obs, ctx.self.length > ctx.world.rules.boostMinLength, ctx.rng.next());
  }

  /** Same decision from a ready obs vector (used by the Vision bridge parity tests and the RL bridge). */
  decideFromObs(obs: ArrayLike<number>, boostAvailable: boolean, u = 0.5): Decision {
    const probs = maskedSoftmax(this.model.logits(obs), actionMask(boostAvailable));
    let action = 0;
    if (this.stochastic) {
      let acc = 0;
      action = probs.length - 1;
      for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (u < acc) { action = i; break; } }
    } else {
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[action]) action = i;
    }
    return { action, debug: { policy: this.id, reason: 'learned', probs } };
  }
}

export function checkRulesVersion(f: MlpModelFile): string | null {
  return f.rulesVersion === RULES_VERSION ? null : `${f.id} was trained on ${f.rulesVersion} (engine: ${RULES_VERSION})`;
}
