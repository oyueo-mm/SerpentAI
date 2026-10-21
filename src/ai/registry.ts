import { HEURISTIC_IDS, createHeuristic } from './heuristics';
import { DEFAULT_MPC, MpcPolicy } from './mpc';
import { MlpPolicy, type MlpModel } from '../policy/mlp';
import type { Policy } from './types';

export type ModelStore = Map<string, MlpModel>;

/** Built-in (non-learned) policy ids, in display order. */
export const BUILTIN_POLICIES = [...HEURISTIC_IDS, 'mpc', 'mpc-lite'] as const;
export type BuiltinPolicyId = (typeof BUILTIN_POLICIES)[number];

export function isBuiltin(id: string): id is BuiltinPolicyId {
  return (BUILTIN_POLICIES as readonly string[]).includes(id);
}

/**
 * Creates a policy by id. Learned ids must be present in `models` (loaded from exported files): a
 * missing model is an error, never silently replaced by a heuristic.
 */
export function createPolicy(id: string, models: ModelStore): Policy {
  if ((HEURISTIC_IDS as readonly string[]).includes(id)) return createHeuristic(id as (typeof HEURISTIC_IDS)[number]);
  if (id === 'mpc') return new MpcPolicy(DEFAULT_MPC, 'mpc');
  if (id === 'mpc-lite') return new MpcPolicy({ ...DEFAULT_MPC, depth: 1, horizon: 1.0 }, 'mpc-lite');
  const m = models.get(id);
  if (!m) throw new Error(`unknown policy or missing model: ${id}`);
  return new MlpPolicy(m);
}
