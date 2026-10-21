import type { ModelStore } from '../ai/registry';
import type { RulesConfig } from './config';
import { Match } from './match';
import { meanCI, wilson, type Interval } from './stats';

/**
 * Free-for-all tournament. Every round puts `copies` snakes of each policy into one arena (no respawn)
 * and runs until time is up or one snake is left. Seat order is rotated every round, so each policy
 * spawns from every spawn slot equally often; seeds come from a range never used for training
 * (EVAL_SEED_BASE) unless overridden.
 */
export const EVAL_SEED_BASE = 900_000;

export interface TournamentConfig {
  policies: string[];
  rounds: number;
  copies?: number;
  seedBase?: number;
  durationS?: number;
  rules?: Partial<RulesConfig>;
  decisionInterval?: number;
}

export interface EntryResult {
  policy: string;
  rank: number;
  alive: boolean;
  survivalS: number;
  finalLength: number;
  maxLength: number;
  eaten: number;
  kills: number;
  death: 'wall' | 'body' | 'headon' | null;
  boostS: number;
}

export interface RoundResult {
  seed: number;
  ticks: number;
  entries: EntryResult[];
}

export interface PolicySummary {
  policy: string;
  entries: number;
  winRate: Interval;
  rank: Interval;
  survivalS: Interval;
  finalLength: Interval;
  maxLength: Interval;
  eaten: Interval;
  kills: Interval;
  deathRate: Interval;
  wallDeathRate: Interval;
  collisionDeathRate: Interval;
  /** Length eaten per second of boost used (NaN if never boosted). */
  boostShare: Interval;
}

export interface TournamentResult {
  config: Required<Omit<TournamentConfig, 'rules'>> & { rules?: Partial<RulesConfig> };
  rounds: RoundResult[];
  summary: PolicySummary[];
  wallMs: number;
}

export function runRound(cfg: TournamentConfig, round: number, models: ModelStore): RoundResult {
  const copies = cfg.copies ?? 1;
  const seed = (cfg.seedBase ?? EVAL_SEED_BASE) + round;
  const P = cfg.policies.length;
  const seats: string[] = [];
  for (let c = 0; c < copies; c++) for (let i = 0; i < P; i++) seats.push(cfg.policies[(i + round) % P]);
  const match = new Match({
    seed, rules: cfg.rules, decisionInterval: cfg.decisionInterval ?? 3, respawn: false, endWhenOneLeft: true,
    durationTicks: Math.round((cfg.durationS ?? 90) * 30 * ((cfg.rules?.tickRate ?? 30) / 30)),
    agents: seats.map((policy, i) => ({ policy, label: `${policy}#${i}` })),
  }, models);
  while (!match.ended) match.step();
  const rate = match.world.rules.tickRate;
  const ranks = new Map(match.rankings().map((r) => [r.snakeId, r]));
  return {
    seed,
    ticks: match.tick,
    entries: match.agents.map((a) => {
      const s = match.world.snakes[a.snakeId];
      const r = ranks.get(a.snakeId)!;
      return {
        policy: a.spec.policy, rank: r.rank, alive: s.alive, survivalS: r.survivalTicks / rate, finalLength: s.alive ? s.length : 0,
        maxLength: a.record.maxLength, eaten: a.record.eaten, kills: a.record.kills, death: s.alive ? null : s.deathCause, boostS: a.record.boostTicks / rate,
      };
    }),
  };
}

export function summarize(policies: string[], rounds: RoundResult[]): PolicySummary[] {
  return policies.map((policy) => {
    const es = rounds.flatMap((r) => r.entries.filter((e) => e.policy === policy));
    const n = es.length;
    const deaths = es.filter((e) => !e.alive);
    const boosted = es.filter((e) => e.boostS > 0);
    return {
      policy, entries: n,
      winRate: wilson(es.filter((e) => e.rank === 1).length, n),
      rank: meanCI(es.map((e) => e.rank)),
      survivalS: meanCI(es.map((e) => e.survivalS), 0),
      finalLength: meanCI(es.map((e) => e.finalLength), 0),
      maxLength: meanCI(es.map((e) => e.maxLength), 0),
      eaten: meanCI(es.map((e) => e.eaten), 0),
      kills: meanCI(es.map((e) => e.kills), 0),
      deathRate: wilson(deaths.length, n),
      wallDeathRate: wilson(deaths.filter((e) => e.death === 'wall').length, n),
      collisionDeathRate: wilson(deaths.filter((e) => e.death !== 'wall').length, n),
      boostShare: meanCI(boosted.map((e) => e.eaten / e.boostS), 0),
    };
  });
}

export function runTournament(cfg: TournamentConfig, models: ModelStore = new Map(), onRound?: (i: number, r: RoundResult) => void): TournamentResult {
  const t0 = Date.now();
  const rounds: RoundResult[] = [];
  for (let i = 0; i < cfg.rounds; i++) {
    const r = runRound(cfg, i, models);
    rounds.push(r);
    onRound?.(i, r);
  }
  return {
    config: { policies: cfg.policies, rounds: cfg.rounds, copies: cfg.copies ?? 1, seedBase: cfg.seedBase ?? EVAL_SEED_BASE, durationS: cfg.durationS ?? 90, decisionInterval: cfg.decisionInterval ?? 3, rules: cfg.rules },
    rounds, summary: summarize(cfg.policies, rounds), wallMs: Date.now() - t0,
  };
}
