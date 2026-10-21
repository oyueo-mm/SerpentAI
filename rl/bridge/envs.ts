import type { ModelStore } from '../../src/ai/registry';
import { createPolicy } from '../../src/ai/registry';
import type { Policy } from '../../src/ai/types';
import { OBS_SIZE, encodeObs } from '../../src/observation/obs';
import { senseWorld } from '../../src/observation/sensors';
import { NOOP_ACTION, actionToControl, decodeAction } from '../../src/policy/action';
import { Match, type AgentSpec } from '../../src/sim/match';
import { Rng } from '../../src/sim/rng';

/**
 * Single-learner RL environment over the real engine. The learner is snake 0; every env step is one
 * policy decision held for `decisionInterval` ticks. All rules, sensing and obs encoding are the same
 * code the browser runs.
 */
export const REWARD_VERSION = 'reward-v1';

export interface RewardConfig {
  /** Per unit of length change (eaten − boost loss). */
  length: number;
  kill: number;
  death: number;
  /** Per decision survived. */
  alive: number;
}

export const DEFAULT_REWARD: RewardConfig = { length: 0.01, kill: 0.5, death: -1, alive: 0.002 };

export type Scenario = 'A-food' | 'B-obstacles' | 'C-moving' | 'D-mixed' | 'E-attack' | 'F-selfplay';

export interface EnvConfig {
  scenario: Scenario;
  /** Overrides the scenario's opponent list (policy ids; learned ids must be added with addModel). */
  opponents?: string[];
  /** Extra opponents appended to the scenario list (e.g. self-play snapshots). */
  extraOpponents?: string[];
  maxSteps?: number;
  decisionInterval?: number;
  foodTarget?: number;
  /** Built-in policy that labels every visited state (BC / DAgger); does not affect the episode. */
  expert?: string;
  /** If set, this built-in policy drives the learner and Python's actions are ignored (expert rollouts). */
  learnerPolicy?: string;
  /** Ticks between a decision and its effect (Sim-to-Visual condition B). */
  actionDelay?: number;
  /** Std of Gaussian noise added to observation features 6.. (condition C); 0 = exact. */
  obsNoise?: number;
  reward?: Partial<RewardConfig>;
}

export const SCENARIOS: Record<Scenario, string[]> = {
  'A-food': [],
  'B-obstacles': ['obstacle', 'obstacle', 'obstacle', 'obstacle', 'obstacle', 'obstacle'],
  'C-moving': ['random', 'random', 'random', 'greedy', 'greedy', 'greedy', 'potential'],
  'D-mixed': ['greedy', 'greedy', 'survival', 'survival', 'aggressive', 'aggressive', 'potential', 'potential', 'hybrid', 'hybrid', 'mpc-lite'],
  'E-attack': ['aggressive', 'aggressive', 'aggressive', 'hybrid', 'hybrid', 'mpc-lite', 'greedy', 'greedy', 'random'],
  'F-selfplay': ['greedy', 'survival', 'aggressive', 'potential', 'hybrid', 'hybrid', 'mpc-lite'],
};

export interface EpisodeInfo {
  /** Episode return under reward-v1. */
  r: number;
  /** Decisions taken. */
  l: number;
  survivalS: number;
  finalLength: number;
  maxLength: number;
  eaten: number;
  kills: number;
  death: string | null;
  boostS: number;
  seed: number;
  scenario: Scenario;
}

export interface StepResult {
  obs: Float32Array;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  /** act-v1 action actually executed (differs from the request when learnerPolicy drives). */
  taken: number;
  canBoost: boolean;
  expert: number;
  info: EpisodeInfo | null;
}

/** Training-only obstacle bot: a long snake circling slowly (Stage B). */
class ObstaclePolicy implements Policy {
  readonly id = 'obstacle';
  readonly kind = 'heuristic' as const;
  decide() { return { action: 2 * 6 }; } // turn 0.5, no boost
}

function makePolicy(id: string, models: ModelStore): Policy {
  return id === 'obstacle' ? new ObstaclePolicy() : createPolicy(id, models);
}

export class LearnerEnv {
  config!: EnvConfig;
  match!: Match;
  seed = 0;
  private steps = 0;
  private ret = 0;
  private prevAction = NOOP_ACTION;
  private expertPolicy: Policy | null = null;
  private learnerPolicy: Policy | null = null;
  private noiseRng = new Rng(1);
  private expertRng = new Rng(2);
  private reward: RewardConfig = DEFAULT_REWARD;

  constructor(config: EnvConfig, private readonly models: ModelStore) {
    this.setConfig(config);
  }

  setConfig(c: EnvConfig): void {
    if (!SCENARIOS[c.scenario]) throw new Error(`unknown scenario ${c.scenario}`);
    this.config = c;
    this.reward = { ...DEFAULT_REWARD, ...(c.reward ?? {}) };
    this.expertPolicy = c.expert ? makePolicy(c.expert, this.models) : null;
    this.learnerPolicy = c.learnerPolicy ? makePolicy(c.learnerPolicy, this.models) : null;
  }

  get decisionInterval(): number {
    return this.config.decisionInterval ?? 3;
  }

  reset(seed: number): Float32Array {
    this.seed = seed >>> 0;
    const c = this.config;
    const opponents = [...(c.opponents ?? SCENARIOS[c.scenario]), ...(c.extraOpponents ?? [])];
    const agents: AgentSpec[] = [{ policy: 'human', label: 'learner', skin: 0 }, ...opponents.map((p, i) => ({ policy: p, label: `${p}-${i}`, skin: i + 1 }))];
    // 'obstacle' is not a registry id: build the match with placeholders and swap the policies in.
    this.match = new Match({
      seed: this.seed, agents: agents.map((a) => (a.policy === 'obstacle' ? { ...a, policy: 'random' } : a)),
      rules: { foodTarget: c.foodTarget ?? 1200 }, decisionInterval: this.decisionInterval, actionDelay: c.actionDelay ?? 0, respawn: true, respawnDelay: 2,
    }, this.models);
    agents.forEach((a, i) => {
      if (a.policy === 'obstacle') {
        this.match.agents[i].policy = new ObstaclePolicy();
        const s = this.match.world.snakes[i];
        this.match.world.respawn(i, { length: 1400, x: s.x, y: s.y, heading: s.heading });
      }
    });
    this.steps = 0;
    this.ret = 0;
    this.prevAction = NOOP_ACTION;
    this.noiseRng = Rng.derive(this.seed, 77);
    this.expertRng = Rng.derive(this.seed, 78);
    return this.observe();
  }

  private learner() {
    return this.match.world.snakes[0];
  }

  observe(): Float32Array {
    const s = this.learner();
    const o = encodeObs(senseWorld(this.match.world, s), decodeAction(this.prevAction), this.match.world.rules);
    const sd = this.config.obsNoise ?? 0;
    if (sd > 0) for (let i = 6; i < o.length - 2; i++) o[i] = Math.max(-1, Math.min(2, o[i] + sd * this.noiseRng.normal()));
    return o;
  }

  canBoost(): boolean {
    const s = this.learner();
    return s.alive && s.length > this.match.world.rules.boostMinLength;
  }

  /** Expert label for the current state (computed on the live state with a dedicated RNG stream). */
  expertAction(): number {
    if (!this.expertPolicy) return -1;
    const w = this.match.world, s = this.learner();
    return this.expertPolicy.decide({ world: w, self: s, rng: this.expertRng, holdTicks: this.decisionInterval, sensors: () => senseWorld(w, s), prevAction: this.prevAction }).action;
  }

  step(action: number): StepResult {
    const w = this.match.world;
    const s = this.learner();
    let taken = action;
    if (this.learnerPolicy) {
      taken = this.learnerPolicy.decide({ world: w, self: s, rng: this.expertRng, holdTicks: this.decisionInterval, sensors: () => senseWorld(w, s), prevAction: this.prevAction }).action;
    }
    if (!Number.isInteger(taken) || taken < 0 || taken >= 18) throw new Error(`invalid action ${taken}`);
    // Masked boost is executed as the same turn without boost (the engine would refuse it anyway).
    if ((taken & 1) === 1 && !this.canBoost()) taken -= 1;
    const L0 = s.length;
    const kills0 = this.match.agents[0].record.kills;
    const delay = this.config.actionDelay ?? 0;
    const a0 = this.match.agents[0];
    if (delay > 0) a0.queue.push({ tick: w.tick + delay, action: taken });
    let died = false;
    for (let k = 0; k < this.decisionInterval; k++) {
      if (delay > 0) {
        while (a0.queue.length && a0.queue[0].tick <= w.tick) a0.held = a0.queue.shift()!.action;
      } else a0.held = taken;
      this.match.humanControl = actionToControl(a0.held);
      this.match.step();
      if (!s.alive) { died = true; break; }
    }
    this.prevAction = taken;
    this.steps++;
    const R = this.reward;
    const kills = this.match.agents[0].record.kills - kills0;
    const dL = (died ? (w.deaths.find((d) => d.id === 0)?.length ?? L0) : s.length) - L0;
    let r = R.length * dL + R.kill * kills;
    if (died) r += R.death; else r += R.alive;
    this.ret += r;
    const truncated = !died && this.steps >= (this.config.maxSteps ?? 1200);
    const done = died || truncated;
    const obs = died ? new Float32Array(OBS_SIZE) : this.observe();
    const rec = this.match.agents[0].record;
    const info: EpisodeInfo | null = done ? {
      r: this.ret, l: this.steps, survivalS: (died ? s.deathTick - s.spawnTick : w.tick - s.spawnTick) / w.rules.tickRate,
      finalLength: died ? 0 : s.length, maxLength: rec.maxLength, eaten: rec.eaten, kills: rec.kills, death: died ? s.deathCause : null,
      boostS: rec.boostTicks / w.rules.tickRate, seed: this.seed, scenario: this.config.scenario,
    } : null;
    return { obs, reward: r, terminated: died, truncated, taken, canBoost: !died && this.canBoost(), expert: done ? -1 : this.expertAction(), info };
  }
}
