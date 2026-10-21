import { BUILTIN_POLICIES } from '../ai/registry';
import type { AgentSpec, MatchConfig } from '../sim/match';
import { DEFAULT_RULES } from '../sim/config';
import { NO_DEBUG, type DebugFlags } from '../render/renderer';

export type Mode = 'play' | 'watch' | 'tournament' | 'lab' | 'vision' | 'calibration';
export const MODES: Mode[] = ['play', 'watch', 'tournament', 'lab', 'vision', 'calibration'];

export interface ArenaSettings {
  seed: number;
  /** Number of bots per policy id (built-ins and learned model ids). */
  bots: Record<string, number>;
  foodTarget: number;
  decisionInterval: number;
  respawn: boolean;
}

export interface UiSettings {
  speed: number;
  debug: DebugFlags;
  labels: boolean;
  rightOpen: boolean;
  settingsOpen: boolean;
}

export const DEFAULT_UI: UiSettings = { speed: 1, debug: { ...NO_DEBUG }, labels: true, rightOpen: false, settingsOpen: false };

export function defaultArena(mode: 'play' | 'watch', learned: string[] = []): ArenaSettings {
  const bots: Record<string, number> = { random: 1, greedy: 2, survival: 2, aggressive: 2, potential: 2, hybrid: 2, mpc: 2 };
  for (const id of learned.slice(0, 2)) bots[id] = 1;
  return { seed: mode === 'play' ? 1 : 7, bots, foodTarget: DEFAULT_RULES.foodTarget, decisionInterval: 3, respawn: true };
}

/** Bot ids in a stable display order. */
export function botOrder(bots: Record<string, number>): string[] {
  const builtin = BUILTIN_POLICIES.filter((p) => p in bots);
  const learned = Object.keys(bots).filter((p) => !(BUILTIN_POLICIES as readonly string[]).includes(p)).sort();
  return [...builtin, ...learned];
}

/** Match config for the arena: in Play the human is snake 0; bots follow in policy order. */
export function toMatchConfig(mode: 'play' | 'watch', a: ArenaSettings): MatchConfig {
  const agents: AgentSpec[] = [];
  if (mode === 'play') agents.push({ policy: 'human', label: '나', skin: 0 });
  for (const id of botOrder(a.bots)) {
    for (let i = 0; i < (a.bots[id] ?? 0); i++) agents.push({ policy: id, label: `${id}${a.bots[id] > 1 ? `-${i + 1}` : ''}`, skin: agents.length });
  }
  return { seed: a.seed, agents, rules: { foodTarget: a.foodTarget }, decisionInterval: a.decisionInterval, respawn: a.respawn, respawnDelay: 2 };
}
