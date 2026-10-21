import type { Sensors } from '../observation/sensors';
import type { Rng } from '../sim/rng';
import type { Snake, World } from '../sim/world';

export type PolicyKind = 'heuristic' | 'mpc' | 'learned' | 'human';

export interface DecisionContext {
  world: World;
  self: Snake;
  /** Per-snake RNG stream owned by the match (serialized with it). */
  rng: Rng;
  /** Ticks the chosen action will be held (the decision interval). */
  holdTicks: number;
  /** Exact engine sensors for this snake (computed lazily, cached per decision). */
  sensors: () => Sensors;
  /** Previously executed act-v1 action. */
  prevAction: number;
}

/** One candidate trajectory the MPC actually evaluated (for the explanation overlay). */
export interface MpcCandidate {
  /** act-v1 action of the first segment (and second-segment turn for depth-2 plans). */
  action: number;
  turn2?: number;
  /** Predicted head positions (world u), every rollout step. */
  path: [number, number][];
  /** Predicted collision point, if any. */
  hit?: [number, number];
  hitTime?: number;
  cost: { food: number; collision: number; clearance: number; boost: number; terminal: number; total: number };
}

export interface MpcDebug {
  candidates: MpcCandidate[];
  best: number;
  horizon: number;
  evalMs: number;
}

export interface DecisionDebug {
  policy: string;
  /** Short reason key (see i18n: t.reason). */
  reason?: string;
  /** World point the policy is steering to. */
  target?: [number, number];
  /** Predicted interception point used by an attack. */
  intercept?: [number, number];
  /** Per-ray danger in [0, 1] used by reactive policies. */
  danger?: number[];
  mpc?: MpcDebug;
  /** Action probabilities of a learned policy. */
  probs?: number[];
  value?: number;
}

export interface Decision {
  /** act-v1 action index. */
  action: number;
  debug?: DecisionDebug;
}

export interface Policy {
  /** Stable id used in tournaments and datasets, e.g. 'hybrid', 'mpc', 'ppo-random-v1'. */
  readonly id: string;
  readonly kind: PolicyKind;
  decide(ctx: DecisionContext): Decision;
  /** Internal state (for policies with memory) so matches stay reproducible after save/restore. */
  getState?(): unknown;
  setState?(s: unknown): void;
}
