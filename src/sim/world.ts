import { closestApproachSq, distPointSegSq, integrateArc, wrapAngle } from '../geometry/geom';
import { PointGrid, SegmentGrid } from '../geometry/grid';
import { BodyPath } from './body';
import { RULES_VERSION, makeRules, pelletRadius, radiusFor, substepsPerTick, type RulesConfig } from './config';
import { Rng } from './rng';

export type DeathCause = 'wall' | 'body' | 'headon';

/**
 * Per-tick steering input (the official input contract shared by humans, policies and the Vision
 * Bridge). Either `turn` ∈ [-1, 1] (fraction of maxTurnRate; + = clockwise on screen, since y points
 * down) or `targetHeading` (rad; turned towards at most maxTurnRate). Missing control = straight, no boost.
 */
export interface Control {
  turn?: number;
  targetHeading?: number;
  boost?: boolean;
}

export interface SnakeStats {
  /** Total pellet value eaten (u of length). */
  eaten: number;
  pellets: number;
  kills: number;
  maxLength: number;
  boostTicks: number;
}

export interface Pellet {
  id: number;
  x: number;
  y: number;
  value: number;
  r: number;
  kind: 'natural' | 'boost' | 'death';
  /** Snake that produced it (-1 for natural food). */
  source: number;
}

export interface DeathEvent {
  tick: number;
  id: number;
  cause: DeathCause;
  /** Snake credited with the kill (-1 for walls and head-on ties). */
  killer: number;
  length: number;
  x: number;
  y: number;
}

export interface SnakeSpawn {
  label?: string;
  skin?: number;
  length?: number;
  x?: number;
  y?: number;
  heading?: number;
}

export class Snake {
  x = 0;
  y = 0;
  heading = 0;
  length = 0;
  alive = true;
  /** Boost actually applied in the last sub-step. */
  boosting = false;
  /** Turn rate (rad/s) applied in the last tick. */
  omega = 0;
  speed = 0;
  /** Accumulated boost loss not yet dropped as a pellet. */
  dropAcc = 0;
  spawnTick = 0;
  deathTick = -1;
  deathCause: DeathCause | null = null;
  killer = -1;
  lives = 1;
  /** Value eaten during the most recent tick (for rewards and statistics). */
  eatenThisTick = 0;
  body = new BodyPath();
  stats: SnakeStats = { eaten: 0, pellets: 0, kills: 0, maxLength: 0, boostTicks: 0 };

  constructor(readonly id: number, public label: string, public skin: number) {}

  radius(rules: RulesConfig): number {
    return radiusFor(rules, this.length);
  }
}

interface SnakeJSON {
  id: number; label: string; skin: number; x: number; y: number; heading: number; length: number; alive: boolean;
  boosting: boolean; omega: number; speed: number; dropAcc: number; spawnTick: number; deathTick: number;
  deathCause: DeathCause | null; killer: number; lives: number; eatenThisTick: number;
  body: { pts: number[]; chainLen: number }; stats: SnakeStats;
}

export interface WorldJSON {
  format: 'serpent-world';
  rulesVersion: string;
  rules: RulesConfig;
  seed: number;
  tick: number;
  snakes: SnakeJSON[];
  food: Pellet[];
  nextFoodId: number;
  foodRng: number[];
  spawnRng: number[];
}

export interface WorldOptions {
  /** Fill the arena with natural food at creation (default true). */
  fillFood?: boolean;
}

/**
 * Deterministic arena simulation. One call to step() advances one fixed tick of 1/tickRate s, split into
 * a fixed number of sub-steps so no head moves more than maxSubstepDistance between collision checks.
 *
 * Order inside every sub-step (all snakes simultaneously, iterated in id order):
 *   1. move heads along exact arcs, apply boost cost, update bodies (commit vertex, trim tail)
 *   2. rebuild the body segment grid
 *   3. detect deaths from the post-move state (walls, head-on, head-into-body), all at once
 *   4. surviving heads eat pellets touched by their swept mouth (closest head wins, ties → lower id)
 *   5. dead snakes turn into pellets
 */
export class World {
  readonly rules: RulesConfig;
  readonly substeps: number;
  readonly dt: number;
  tick = 0;
  snakes: Snake[] = [];
  food = new Map<number, Pellet>();
  nextFoodId = 0;
  foodRng: Rng;
  spawnRng: Rng;
  /** Deaths that happened during the most recent tick. */
  deaths: DeathEvent[] = [];
  readonly segGrid: SegmentGrid;
  private readonly foodGrid: PointGrid;
  private gridDirty = true;
  private readonly tmp = new Float64Array(3);
  /** Largest pellet radius ever added (a query bound only; never affects outcomes). */
  private maxPelletR = 0;

  constructor(rules: Partial<RulesConfig> | RulesConfig, readonly seed: number, opts: WorldOptions = {}) {
    this.rules = makeRules(rules);
    this.substeps = substepsPerTick(this.rules);
    this.dt = 1 / this.rules.tickRate;
    this.foodRng = Rng.derive(seed, 1);
    this.spawnRng = Rng.derive(seed, 2);
    this.segGrid = new SegmentGrid(this.rules.worldSize, this.rules.gridCell);
    this.foodGrid = new PointGrid(this.rules.worldSize, 64);
    if (opts.fillFood ?? true) {
      while (this.food.size < this.rules.foodTarget) this.spawnNaturalPellet();
    }
  }

  get time(): number {
    return this.tick * this.dt;
  }

  aliveSnakes(): Snake[] {
    return this.snakes.filter((s) => s.alive);
  }

  // ------------------------------------------------------------------ food
  addPellet(x: number, y: number, value: number, kind: Pellet['kind'], source: number): Pellet {
    const W = this.rules.worldSize;
    const r = pelletRadius(value);
    const p: Pellet = { id: this.nextFoodId++, x: Math.min(W - r, Math.max(r, x)), y: Math.min(W - r, Math.max(r, y)), value, r, kind, source };
    this.food.set(p.id, p);
    this.foodGrid.insert(p.id, p.x, p.y);
    if (r > this.maxPelletR) this.maxPelletR = r;
    return p;
  }

  removePellet(id: number): void {
    if (this.food.delete(id)) this.foodGrid.remove(id);
  }

  private spawnNaturalPellet(): void {
    const W = this.rules.worldSize, m = 20;
    const x = this.foodRng.range(m, W - m), y = this.foodRng.range(m, W - m);
    const v = this.foodRng.int(this.rules.foodValueMin, this.rules.foodValueMax);
    this.addPellet(x, y, v, 'natural', -1);
  }

  /** Visits pellets within the box (in arbitrary order). */
  queryFood(minX: number, minY: number, maxX: number, maxY: number, fn: (p: Pellet) => void): void {
    this.foodGrid.query(minX, minY, maxX, maxY, (id) => {
      const p = this.food.get(id);
      if (p) fn(p);
    });
  }

  // ------------------------------------------------------------------ snakes
  /** Adds a snake; position/heading are drawn from the spawn RNG unless given. */
  addSnake(spec: SnakeSpawn = {}): Snake {
    const s = new Snake(this.snakes.length, spec.label ?? `S${this.snakes.length}`, spec.skin ?? this.snakes.length);
    this.snakes.push(s);
    this.placeSnake(s, spec);
    return s;
  }

  /** Brings a dead snake back as a fresh life at a safe spawn point. */
  respawn(id: number, spec: SnakeSpawn = {}): Snake {
    const s = this.snakes[id];
    if (!s) throw new Error(`no snake ${id}`);
    s.lives++;
    this.placeSnake(s, spec);
    return s;
  }

  private placeSnake(s: Snake, spec: SnakeSpawn): void {
    const L = spec.length ?? this.rules.startLength;
    let x = spec.x, y = spec.y, heading = spec.heading;
    if (x === undefined || y === undefined) {
      const p = this.findSpawnPoint(L);
      x = p.x; y = p.y;
      if (heading === undefined) heading = p.heading;
    }
    if (heading === undefined) heading = this.spawnRng.range(-Math.PI, Math.PI);
    s.x = x; s.y = y; s.heading = wrapAngle(heading);
    s.length = L;
    s.alive = true;
    s.boosting = false;
    s.omega = 0;
    s.speed = this.rules.baseSpeed;
    s.dropAcc = 0;
    s.spawnTick = this.tick;
    s.deathTick = -1;
    s.deathCause = null;
    s.killer = -1;
    s.eatenThisTick = 0;
    s.body = new BodyPath();
    s.body.initStraight(x, y, s.heading, L, this.rules.pathSpacing);
    s.stats = { eaten: 0, pellets: 0, kills: 0, maxLength: L, boostTicks: 0 };
    this.gridDirty = true;
  }

  /**
   * Picks a spawn point ≥ 300 u from the walls, heading roughly towards the centre, whose head and
   * straight initial body keep the largest clearance from living snakes (first candidate with ≥ 250 u wins).
   */
  findSpawnPoint(length: number): { x: number; y: number; heading: number } {
    this.ensureGrid();
    const W = this.rules.worldSize;
    const margin = Math.min(300, W / 4);
    let best = { x: W / 2, y: W / 2, heading: 0 }, bestClear = -1;
    for (let attempt = 0; attempt < 60; attempt++) {
      const x = this.spawnRng.range(margin, W - margin), y = this.spawnRng.range(margin, W - margin);
      const toC = Math.atan2(W / 2 - y, W / 2 - x);
      const heading = wrapAngle(toC + this.spawnRng.range(-Math.PI / 3, Math.PI / 3));
      const tx = x - Math.cos(heading) * length, ty = y - Math.sin(heading) * length;
      if (tx < 30 || ty < 30 || tx > W - 30 || ty > W - 30) continue;
      let clear = Infinity;
      for (const o of this.snakes) {
        if (!o.alive) continue;
        clear = Math.min(clear, Math.sqrt(distPointSegSq(o.x, o.y, x, y, tx, ty)));
      }
      const R = 260;
      this.segGrid.query(Math.min(x, tx) - R, Math.min(y, ty) - R, Math.max(x, tx) + R, Math.max(y, ty) + R, (k) => {
        if (!this.snakes[this.segGrid.owner[k]].alive) return;
        const d = Math.sqrt(distPointSegSq(this.segGrid.ax[k], this.segGrid.ay[k], x, y, tx, ty)) - this.segGrid.rad[k];
        if (d < clear) clear = d;
      });
      if (clear > bestClear) { bestClear = clear; best = { x, y, heading }; }
      if (clear >= 250) break;
    }
    return best;
  }

  // ------------------------------------------------------------------ grid
  /** Rebuilds the body segment grid from living snakes (each body = committed vertices + head). */
  rebuildGrid(): void {
    const g = this.segGrid;
    g.clear();
    for (const s of this.snakes) {
      if (!s.alive) continue;
      const r = s.radius(this.rules);
      const b = s.body, n = b.vertexCount;
      for (let i = 0; i < n - 1; i++) g.add(s.id, i, b.vx(i), b.vy(i), b.vx(i + 1), b.vy(i + 1), r);
      g.add(s.id, n - 1, b.vx(n - 1), b.vy(n - 1), s.x, s.y, r);
    }
    this.gridDirty = false;
  }

  /** Makes the segment grid reflect the current state (call before spatial queries from outside). */
  ensureGrid(): void {
    if (this.gridDirty) this.rebuildGrid();
  }

  // ------------------------------------------------------------------ step
  /** Advances one tick. controls[id] is the input for snake id (missing = straight, no boost). */
  step(controls: readonly (Control | undefined)[]): void {
    const R = this.rules;
    const h = this.dt / this.substeps;
    this.deaths = [];
    const omegas: number[] = [];
    for (const s of this.snakes) {
      s.eatenThisTick = 0;
      if (!s.alive) { omegas.push(0); continue; }
      const c = controls[s.id];
      let w = 0;
      if (c?.targetHeading !== undefined && Number.isFinite(c.targetHeading)) {
        w = wrapAngle(c.targetHeading - s.heading) / this.dt;
      } else if (c?.turn !== undefined && Number.isFinite(c.turn)) {
        w = Math.max(-1, Math.min(1, c.turn)) * R.maxTurnRate;
      }
      w = Math.max(-R.maxTurnRate, Math.min(R.maxTurnRate, w));
      s.omega = w;
      omegas.push(w);
    }
    const n = this.snakes.length;
    const px = new Float64Array(n), py = new Float64Array(n);

    for (let sub = 0; sub < this.substeps; sub++) {
      // 1. motion
      for (const s of this.snakes) {
        if (!s.alive) continue;
        px[s.id] = s.x; py[s.id] = s.y;
        const wantBoost = controls[s.id]?.boost === true;
        const boost = wantBoost && s.length > R.boostMinLength;
        s.boosting = boost;
        s.speed = boost ? R.boostSpeed : R.baseSpeed;
        integrateArc(s.x, s.y, s.heading, s.speed, omegas[s.id], h, this.tmp);
        s.x = this.tmp[0]; s.y = this.tmp[1]; s.heading = this.tmp[2];
        s.body.advance(s.x, s.y, R.pathSpacing);
        if (boost) {
          const newL = Math.max(R.boostMinLength, s.length - R.boostCost * h);
          s.dropAcc += (s.length - newL) * R.boostDropFraction;
          s.length = newL;
        }
        s.body.trim(s.x, s.y, s.length);
        while (s.dropAcc >= R.boostPelletValue) {
          s.dropAcc -= R.boostPelletValue;
          const jr = s.radius(R) * 0.5;
          this.addPellet(s.body.tailX() + this.foodRng.range(-jr, jr), s.body.tailY() + this.foodRng.range(-jr, jr), R.boostPelletValue, 'boost', s.id);
        }
      }
      // 2. geometry index
      this.rebuildGrid();
      // 3. deaths (decided from one consistent post-move state, then applied together)
      const dying = this.detectDeaths(px, py);
      // 4. eating by survivors
      this.eat(px, py, dying);
      // 5. apply deaths
      for (const d of dying) this.kill(d);
      if (dying.length) this.gridDirty = true;
    }
    for (const s of this.snakes) {
      if (!s.alive) continue;
      if (s.boosting) s.stats.boostTicks++;
      if (s.length > s.stats.maxLength) s.stats.maxLength = s.length;
    }
    this.tick++;
    let spawn = R.foodSpawnPerTick;
    while (spawn-- > 0 && this.food.size < R.foodTarget) this.spawnNaturalPellet();
  }

  private detectDeaths(px: Float64Array, py: Float64Array): DeathEvent[] {
    const R = this.rules, W = R.worldSize, g = this.segGrid;
    const alive = this.snakes.filter((s) => s.alive);
    const rad = new Map<number, number>();
    for (const s of alive) rad.set(s.id, s.radius(R));
    // Head-on contacts: swept closest approach of the two heads over the sub-step.
    const headOn = new Map<number, number[]>();
    for (let a = 0; a < alive.length; a++) {
      for (let b = a + 1; b < alive.length; b++) {
        const i = alive[a], j = alive[b];
        const thr = rad.get(i.id)! + rad.get(j.id)!;
        if (Math.abs(i.x - j.x) > thr + 20 || Math.abs(i.y - j.y) > thr + 20) continue;
        if (closestApproachSq(px[i.id], py[i.id], i.x, i.y, px[j.id], py[j.id], j.x, j.y) < thr * thr) {
          (headOn.get(i.id) ?? headOn.set(i.id, []).get(i.id)!).push(j.id);
          (headOn.get(j.id) ?? headOn.set(j.id, []).get(j.id)!).push(i.id);
        }
      }
    }
    const out: DeathEvent[] = [];
    for (const s of alive) {
      const r = rad.get(s.id)!;
      const partners = headOn.get(s.id) ?? [];
      let cause: DeathCause | null = null, killer = -1;
      // Head-on: the shorter snake dies; equal lengths → both die (no kill credit).
      let headOnLoss = -1;
      for (const j of partners) {
        const o = this.snakes[j];
        if (s.length < o.length) { headOnLoss = headOnLoss < 0 ? j : headOnLoss; }
        else if (s.length === o.length && headOnLoss < 0) headOnLoss = -2 - j;
      }
      if (headOnLoss !== -1) {
        cause = 'headon';
        killer = headOnLoss >= 0 ? headOnLoss : -1;
      } else {
        // Head into another body (capsule overlap), ignoring head-on partners and itself.
        let bestPen = 0, bestOwner = -1;
        const reach = r + R.radiusMax;
        g.query(s.x - reach, s.y - reach, s.x + reach, s.y + reach, (k) => {
          const o = g.owner[k];
          if (o === s.id || partners.includes(o)) return;
          const thr = r + g.rad[k];
          const d2 = distPointSegSq(s.x, s.y, g.ax[k], g.ay[k], g.bx[k], g.by[k]);
          if (d2 < thr * thr) {
            const pen = thr - Math.sqrt(d2);
            if (pen > bestPen || (pen === bestPen && o < bestOwner)) { bestPen = pen; bestOwner = o; }
          }
        });
        if (bestOwner >= 0) { cause = 'body'; killer = bestOwner; }
        else if (s.x < r || s.y < r || s.x > W - r || s.y > W - r) cause = 'wall';
      }
      if (cause) out.push({ tick: this.tick, id: s.id, cause, killer, length: s.length, x: s.x, y: s.y });
    }
    return out;
  }

  private eat(px: Float64Array, py: Float64Array, dying: DeathEvent[]): void {
    const R = this.rules;
    const dead = new Set(dying.map((d) => d.id));
    // pellet id → [snake id, squared distance]
    const claims = new Map<number, [number, number]>();
    for (const s of this.snakes) {
      if (!s.alive || dead.has(s.id)) continue;
      const r = s.radius(R);
      const reach = r + R.eatReach + this.maxPelletR;
      const x0 = px[s.id], y0 = py[s.id];
      this.foodGrid.query(Math.min(x0, s.x) - reach, Math.min(y0, s.y) - reach, Math.max(x0, s.x) + reach, Math.max(y0, s.y) + reach, (id) => {
        const p = this.food.get(id)!;
        const lim = r + p.r + R.eatReach;
        const d2 = distPointSegSq(p.x, p.y, x0, y0, s.x, s.y);
        if (d2 >= lim * lim) return;
        const prev = claims.get(id);
        if (!prev || d2 < prev[1] || (d2 === prev[1] && s.id < prev[0])) claims.set(id, [s.id, d2]);
      });
    }
    const ids = [...claims.keys()].sort((a, b) => a - b);
    for (const id of ids) {
      const p = this.food.get(id)!;
      const s = this.snakes[claims.get(id)![0]];
      s.length += p.value;
      s.eatenThisTick += p.value;
      s.stats.eaten += p.value;
      s.stats.pellets++;
      this.removePellet(id);
    }
  }

  private kill(d: DeathEvent): void {
    const R = this.rules;
    const s = this.snakes[d.id];
    s.alive = false;
    s.deathTick = this.tick;
    s.deathCause = d.cause;
    s.killer = d.killer;
    s.boosting = false;
    if (d.killer >= 0) this.snakes[d.killer].stats.kills++;
    this.deaths.push(d);
    // Lay pellets along the body from head to tail.
    const r = s.radius(R);
    const spacing = Math.max(R.deathPelletSpacingMin, r);
    const pts: [number, number][] = [];
    const b = s.body;
    let ax = s.x, ay = s.y, carry = 0;
    pts.push([ax, ay]);
    for (let i = b.vertexCount - 1; i >= 0; i--) {
      const bx = b.vx(i), by = b.vy(i);
      const seg = Math.hypot(bx - ax, by - ay);
      let t = spacing - carry;
      while (t <= seg) {
        pts.push([ax + (bx - ax) * (t / seg), ay + (by - ay) * (t / seg)]);
        t += spacing;
      }
      carry = seg - (t - spacing);
      ax = bx; ay = by;
    }
    const value = (R.deathDropFraction * s.length) / pts.length;
    for (const [x, y] of pts) {
      this.addPellet(x + this.foodRng.range(-r * 0.5, r * 0.5), y + this.foodRng.range(-r * 0.5, r * 0.5), value, 'death', s.id);
    }
  }

  // ------------------------------------------------------------------ persistence
  toJSON(): WorldJSON {
    return {
      format: 'serpent-world',
      rulesVersion: RULES_VERSION,
      rules: { ...this.rules },
      seed: this.seed,
      tick: this.tick,
      snakes: this.snakes.map((s) => ({
        id: s.id, label: s.label, skin: s.skin, x: s.x, y: s.y, heading: s.heading, length: s.length, alive: s.alive,
        boosting: s.boosting, omega: s.omega, speed: s.speed, dropAcc: s.dropAcc, spawnTick: s.spawnTick, deathTick: s.deathTick,
        deathCause: s.deathCause, killer: s.killer, lives: s.lives, eatenThisTick: s.eatenThisTick,
        body: s.body.toJSON(), stats: { ...s.stats },
      })),
      food: [...this.food.values()].map((p) => ({ ...p })),
      nextFoodId: this.nextFoodId,
      foodRng: this.foodRng.state(),
      spawnRng: this.spawnRng.state(),
    };
  }

  static fromJSON(o: WorldJSON): World {
    if (o.format !== 'serpent-world') throw new Error('not a serpent-world snapshot');
    if (o.rulesVersion !== RULES_VERSION) throw new Error(`snapshot uses ${o.rulesVersion}, engine is ${RULES_VERSION}`);
    const w = new World(o.rules, o.seed, { fillFood: false });
    w.tick = o.tick;
    for (const j of o.snakes) {
      const s = new Snake(j.id, j.label, j.skin);
      Object.assign(s, { x: j.x, y: j.y, heading: j.heading, length: j.length, alive: j.alive, boosting: j.boosting, omega: j.omega,
        speed: j.speed, dropAcc: j.dropAcc, spawnTick: j.spawnTick, deathTick: j.deathTick, deathCause: j.deathCause, killer: j.killer,
        lives: j.lives, eatenThisTick: j.eatenThisTick });
      s.body = BodyPath.fromJSON(j.body);
      s.stats = { ...j.stats };
      w.snakes.push(s);
    }
    for (const p of o.food) {
      w.food.set(p.id, { ...p });
      w.foodGrid.insert(p.id, p.x, p.y);
      w.maxPelletR = Math.max(w.maxPelletR, p.r);
    }
    w.nextFoodId = o.nextFoodId;
    w.foodRng = new Rng(o.foodRng);
    w.spawnRng = new Rng(o.spawnRng);
    w.gridDirty = true;
    return w;
  }

  clone(): World {
    return World.fromJSON(this.toJSON());
  }
}
