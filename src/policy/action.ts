import type { RulesConfig } from '../sim/config';
import type { Control } from '../sim/world';

/**
 * act-v1: 18 discrete actions = 9 turn levels × {no boost, boost}; index = turnIndex · 2 + boost.
 * The turn level is a fraction of maxTurnRate (+ = clockwise on screen).
 */
export const ACTION_VERSION = 'act-v1';
export const TURN_LEVELS = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1] as const;
export const NUM_ACTIONS = TURN_LEVELS.length * 2;
/** "Straight, no boost" — the safe default when no decision is available. */
export const NOOP_ACTION = 4 * 2;

export function decodeAction(a: number): { turn: number; boost: boolean } {
  if (!Number.isInteger(a) || a < 0 || a >= NUM_ACTIONS) throw new Error(`action ${a} outside act-v1`);
  return { turn: TURN_LEVELS[a >> 1], boost: (a & 1) === 1 };
}

export function encodeAction(turn: number, boost: boolean): number {
  let best = 0;
  for (let i = 1; i < TURN_LEVELS.length; i++) if (Math.abs(TURN_LEVELS[i] - turn) < Math.abs(TURN_LEVELS[best] - turn)) best = i;
  return best * 2 + (boost ? 1 : 0);
}

export function actionToControl(a: number): Control {
  const { turn, boost } = decodeAction(a);
  return { turn, boost };
}

/** Boost is available only while L > boostMinLength (same rule the engine applies). */
export function canBoost(rules: RulesConfig, length: number): boolean {
  return length > rules.boostMinLength;
}

/** Valid-action mask: boost actions are masked out when boost is unavailable. */
export function actionMask(boostAvailable: boolean): boolean[] {
  return Array.from({ length: NUM_ACTIONS }, (_, a) => boostAvailable || (a & 1) === 0);
}

/**
 * Maps a free-form control (e.g. a heuristic's target heading) to the nearest act-v1 action, assuming the
 * action is held for `holdTicks` ticks (the decision interval): the turn level that reaches the target
 * heading by the end of the hold, saturated at ±1.
 */
export function controlToAction(c: Control, heading: number, rules: RulesConfig, holdTicks = 1): number {
  let turn = 0;
  if (c.targetHeading !== undefined) {
    const d = Math.atan2(Math.sin(c.targetHeading - heading), Math.cos(c.targetHeading - heading));
    turn = Math.max(-1, Math.min(1, (d * rules.tickRate) / (rules.maxTurnRate * Math.max(1, holdTicks))));
  } else if (c.turn !== undefined) turn = c.turn;
  return encodeAction(turn, c.boost === true);
}
