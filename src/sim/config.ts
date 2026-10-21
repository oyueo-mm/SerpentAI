/**
 * Central rule constants (rules-v1). Units: world units (u), seconds (s), radians (rad).
 * Body length L is measured as arc length of the body path in u; a snake's score equals floor(L).
 * Any change to a value here that alters outcomes must bump RULES_VERSION (docs/RULES.md).
 */
export const RULES_VERSION = 'rules-v1';

export interface RulesConfig {
  /** Square arena side; the playable area is [0, worldSize]². */
  worldSize: number;
  /** Fixed simulation ticks per simulated second (independent of render FPS). */
  tickRate: number;
  /** Cruise speed (u/s). */
  baseSpeed: number;
  /** Boost speed (u/s). */
  boostSpeed: number;
  /** Maximum heading change rate (rad/s), same with or without boost. */
  maxTurnRate: number;
  startLength: number;
  /** Length never drops below this (boost is cut off before it could). */
  minLength: number;
  /** Boost is only available while L > boostMinLength. */
  boostMinLength: number;
  /** Length spent per second of boost (u/s). */
  boostCost: number;
  /** Fraction of the boost cost dropped back into the world as pellets at the tail. */
  boostDropFraction: number;
  /** Value of one boost pellet (u of length). */
  boostPelletValue: number;
  /** Body radius r(L) = min(radiusMax, radiusBase · (L / startLength)^radiusExponent). */
  radiusBase: number;
  radiusExponent: number;
  radiusMax: number;
  /** A new body vertex is committed once the head is this far (u) from the previous one. */
  pathSpacing: number;
  /** Upper bound on head displacement per sub-step (u); sets the fixed sub-step count. */
  maxSubstepDistance: number;
  /** Natural food is replenished towards this many pellets. */
  foodTarget: number;
  /** Max natural pellets spawned per tick while below target. */
  foodSpawnPerTick: number;
  /** Natural pellet values are uniform integers in [foodValueMin, foodValueMax] (u of length). */
  foodValueMin: number;
  foodValueMax: number;
  /** Extra reach (u) of the mouth beyond head radius + pellet radius. */
  eatReach: number;
  /** Fraction of a dead snake's length converted to pellets. */
  deathDropFraction: number;
  /** Death pellets are laid every max(deathPelletSpacingMin, r) u along the body. */
  deathPelletSpacingMin: number;
  /** Spatial grid cell size (u). */
  gridCell: number;
}

export const DEFAULT_RULES: Readonly<RulesConfig> = Object.freeze({
  worldSize: 3000,
  tickRate: 30,
  baseSpeed: 150,
  boostSpeed: 300,
  maxTurnRate: 3.2,
  startLength: 240,
  minLength: 120,
  boostMinLength: 160,
  boostCost: 45,
  boostDropFraction: 0.5,
  boostPelletValue: 6,
  radiusBase: 10,
  radiusExponent: 0.35,
  radiusMax: 30,
  pathSpacing: 6,
  maxSubstepDistance: 5,
  foodTarget: 1200,
  foodSpawnPerTick: 4,
  foodValueMin: 2,
  foodValueMax: 6,
  eatReach: 6,
  deathDropFraction: 0.6,
  deathPelletSpacingMin: 8,
  gridCell: 96,
});

export function makeRules(overrides: Partial<RulesConfig> = {}): RulesConfig {
  const r = { ...DEFAULT_RULES, ...overrides };
  if (r.minLength >= r.boostMinLength) throw new Error('minLength must be below boostMinLength');
  if (r.tickRate <= 0 || r.worldSize <= 0) throw new Error('tickRate and worldSize must be positive');
  return r;
}

/** Body radius (u) for body length L. */
export function radiusFor(rules: RulesConfig, length: number): number {
  return Math.min(rules.radiusMax, rules.radiusBase * Math.pow(Math.max(length, 1) / rules.startLength, rules.radiusExponent));
}

/** Inverse of radiusFor below the cap (used by vision to estimate length from thickness). */
export function lengthForRadius(rules: RulesConfig, radius: number): number {
  return rules.startLength * Math.pow(radius / rules.radiusBase, 1 / rules.radiusExponent);
}

/** Pellet radius (u) for a pellet of the given value. */
export function pelletRadius(value: number): number {
  return 2 + 1.2 * Math.sqrt(value);
}

/** Fixed sub-steps per tick so no head moves more than maxSubstepDistance per sub-step. */
export function substepsPerTick(rules: RulesConfig): number {
  return Math.max(1, Math.ceil(rules.boostSpeed / rules.tickRate / rules.maxSubstepDistance - 1e-9));
}
