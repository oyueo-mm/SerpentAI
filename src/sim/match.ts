import { createPolicy, type ModelStore } from '../ai/registry';
import type { DecisionDebug, Policy } from '../ai/types';
import { senseWorld, type Sensors } from '../observation/sensors';
import { NOOP_ACTION, actionToControl, decodeAction } from '../policy/action';
import type { RulesConfig } from './config';
import { Rng } from './rng';
import { World, type Control, type WorldJSON } from './world';

export interface AgentSpec {
  /** Policy id ('human', 'random', 'greedy', …, 'mpc', or a learned model id). */
  policy: string;
  label?: string;
  skin?: number;
}

export interface MatchConfig {
  seed: number;
  agents: AgentSpec[];
  rules?: Partial<RulesConfig>;
  /** Policy decisions every N ticks (3 → 10 Hz at 30 ticks/s); the action is held in between. */
  decisionInterval?: number;
  /** Ticks between a decision and the moment it takes effect (control latency experiments). */
  actionDelay?: number;
  /** Dead snakes come back after respawnDelay seconds. */
  respawn?: boolean;
  respawnDelay?: number;
  /** Snake ids that never respawn even when respawn is on (e.g. the Vision Bridge's player). */
  respawnExclude?: number[];
  /** Match length in ticks (0 = endless). */
  durationTicks?: number;
  /** End when at most one snake is alive (tournament rounds without respawn). */
  endWhenOneLeft?: boolean;
}

/** Cumulative per-agent record across lives. */
export interface AgentRecord {
  deaths: number;
  deathsBy: { wall: number; body: number; headon: number };
  kills: number;
  eaten: number;
  maxLength: number;
  /** Alive ticks summed over lives. */
  aliveTicks: number;
  boostTicks: number;
  decisions: number;
  /** Histogram over act-v1 actions actually executed. */
  actionCounts: number[];
}

interface Pending {
  tick: number;
  action: number;
}

export class AgentRuntime {
  policy: Policy | null;
  rng: Rng;
  /** Action currently applied to the engine. */
  held = NOOP_ACTION;
  /** Most recent decision (may still be in the delay queue). */
  lastDecided = NOOP_ACTION;
  queue: Pending[] = [];
  debug: DecisionDebug | undefined;
  record: AgentRecord;
  respawnAt = -1;

  constructor(readonly snakeId: number, readonly spec: AgentSpec, policy: Policy | null, seed: number) {
    this.policy = policy;
    this.rng = Rng.derive(seed, 1000 + snakeId);
    this.record = { deaths: 0, deathsBy: { wall: 0, body: 0, headon: 0 }, kills: 0, eaten: 0, maxLength: 0, aliveTicks: 0, boostTicks: 0, decisions: 0, actionCounts: new Array(18).fill(0) };
  }

  get isHuman(): boolean {
    return this.spec.policy === 'human';
  }
}

export interface MatchJSON {
  format: 'serpent-match';
  config: MatchConfig;
  world: WorldJSON;
  agents: { rng: number[]; held: number; lastDecided: number; queue: Pending[]; record: AgentRecord; respawnAt: number; policyState?: unknown }[];
}

export interface Ranking {
  snakeId: number;
  label: string;
  policy: string;
  rank: number;
  alive: boolean;
  length: number;
  survivalTicks: number;
}

export class Match {
  readonly world: World;
  readonly agents: AgentRuntime[] = [];
  readonly decisionInterval: number;
  readonly actionDelay: number;
  /** Control for 'human' agents (set by the UI every frame). */
  humanControl: Control = {};
  ended = false;

  constructor(readonly config: MatchConfig, models: ModelStore = new Map(), world?: World) {
    this.decisionInterval = Math.max(1, config.decisionInterval ?? 3);
    this.actionDelay = Math.max(0, config.actionDelay ?? 0);
    this.world = world ?? new World(config.rules ?? {}, config.seed);
    config.agents.forEach((spec, i) => {
      if (!world) this.world.addSnake({ label: spec.label ?? spec.policy, skin: spec.skin ?? i });
      const policy = spec.policy === 'human' ? null : createPolicy(spec.policy, models);
      const rt = new AgentRuntime(i, spec, policy, config.seed);
      rt.record.maxLength = this.world.snakes[i].length;
      this.agents.push(rt);
    });
  }

  get tick(): number {
    return this.world.tick;
  }

  /** Sensors for the decision context (cached per call site). */
  private sensorsFor(id: number): () => Sensors {
    let cache: Sensors | undefined;
    return () => (cache ??= senseWorld(this.world, this.world.snakes[id]));
  }

  /** Advances one tick: due decisions → delayed actions take effect → engine tick → bookkeeping. */
  step(): void {
    if (this.ended) return;
    const w = this.world;
    const decide = w.tick % this.decisionInterval === 0;
    const controls: Control[] = [];
    for (const a of this.agents) {
      const s = w.snakes[a.snakeId];
      if (!s.alive) { controls.push({}); continue; }
      if (a.isHuman) { controls.push(this.humanControl); continue; }
      if (decide && a.policy) {
        const d = a.policy.decide({ world: w, self: s, rng: a.rng, holdTicks: this.decisionInterval, sensors: this.sensorsFor(a.snakeId), prevAction: a.lastDecided });
        a.debug = d.debug;
        a.lastDecided = d.action;
        a.record.decisions++;
        a.queue.push({ tick: w.tick + this.actionDelay, action: d.action });
      }
      while (a.queue.length && a.queue[0].tick <= w.tick) a.held = a.queue.shift()!.action;
      a.record.actionCounts[a.held]++;
      controls.push(actionToControl(a.held));
    }
    w.step(controls);

    for (const a of this.agents) {
      const s = w.snakes[a.snakeId];
      if (s.alive) {
        a.record.aliveTicks++;
        a.record.eaten += s.eatenThisTick;
        if (s.boosting) a.record.boostTicks++;
        a.record.maxLength = Math.max(a.record.maxLength, s.length);
      }
    }
    for (const d of w.deaths) {
      const a = this.agents[d.id];
      a.record.deaths++;
      a.record.deathsBy[d.cause]++;
      a.record.eaten += w.snakes[d.id].eatenThisTick;
      if (d.killer >= 0) this.agents[d.killer].record.kills++;
      a.queue = [];
      a.held = NOOP_ACTION;
      a.lastDecided = NOOP_ACTION;
      if (this.config.respawn && !this.config.respawnExclude?.includes(d.id)) a.respawnAt = w.tick + Math.round((this.config.respawnDelay ?? 2) * w.rules.tickRate);
    }
    if (this.config.respawn) {
      for (const a of this.agents) {
        if (a.respawnAt >= 0 && w.tick >= a.respawnAt && !w.snakes[a.snakeId].alive) {
          w.respawn(a.snakeId);
          a.respawnAt = -1;
        }
      }
    }
    const dur = this.config.durationTicks ?? 0;
    if (dur > 0 && w.tick >= dur) this.ended = true;
    if (this.config.endWhenOneLeft && w.snakes.filter((s) => s.alive).length <= 1) this.ended = true;
  }

  /** Final standings: living snakes by length, then the dead by how long they lasted. */
  rankings(): Ranking[] {
    const w = this.world;
    const rows = this.agents.map((a) => {
      const s = w.snakes[a.snakeId];
      return { snakeId: s.id, label: s.label, policy: a.spec.policy, alive: s.alive, length: s.length, survivalTicks: (s.alive ? w.tick : s.deathTick) - s.spawnTick, rank: 0 };
    });
    rows.sort((a, b) => (a.alive !== b.alive ? (a.alive ? -1 : 1) : a.alive ? b.length - a.length || a.snakeId - b.snakeId : b.survivalTicks - a.survivalTicks || b.length - a.length || a.snakeId - b.snakeId));
    rows.forEach((r, i) => (r.rank = i + 1));
    return rows;
  }

  /** Current act-v1 turn/boost being applied for an agent (for HUD/overlays). */
  heldCommand(id: number): { turn: number; boost: boolean } {
    return decodeAction(this.agents[id].held);
  }

  toJSON(): MatchJSON {
    return {
      format: 'serpent-match',
      config: this.config,
      world: this.world.toJSON(),
      agents: this.agents.map((a) => ({ rng: a.rng.state(), held: a.held, lastDecided: a.lastDecided, queue: a.queue.map((q) => ({ ...q })), record: structuredClone(a.record), respawnAt: a.respawnAt, policyState: a.policy?.getState?.() })),
    };
  }

  static fromJSON(o: MatchJSON, models: ModelStore = new Map()): Match {
    if (o.format !== 'serpent-match') throw new Error('not a serpent-match snapshot');
    const m = new Match(o.config, models, World.fromJSON(o.world));
    o.agents.forEach((s, i) => {
      const a = m.agents[i];
      a.rng = new Rng(s.rng);
      a.held = s.held;
      a.lastDecided = s.lastDecided;
      a.queue = s.queue.map((q) => ({ ...q }));
      a.record = structuredClone(s.record);
      a.respawnAt = s.respawnAt;
      if (s.policyState !== undefined) a.policy?.setState?.(structuredClone(s.policyState));
    });
    return m;
  }
}
