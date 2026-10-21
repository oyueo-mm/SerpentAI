/**
 * Wall-clock pacing for the simulation worker. The simulation always advances in whole fixed ticks;
 * pacing only decides how many ticks to run per slice of real time, so the outcome never depends on
 * frame rate or slice timing (see pacing.test.ts).
 */
export const SPEED_MIN = 0.1;
export const SPEED_MAX = 64;
/** Sentinel for "as fast as possible". */
export const SPEED_MAX_MODE = Number.POSITIVE_INFINITY;

export class Pacer {
  private owed = 0;
  private last = -1;
  /** Measured simulated seconds per real second (EMA). */
  actualSpeed = 0;
  /** Ticks per real second (EMA). */
  ups = 0;

  constructor(public speed: number, readonly tickRate: number) {}

  reset(now: number): void {
    this.last = now;
    this.owed = 0;
  }

  /** Ticks to run now; never more than maxTicks (a backlog beyond that is dropped, not accumulated). */
  due(now: number, maxTicks: number): number {
    if (this.last < 0) this.last = now;
    const dt = Math.max(0, Math.min(0.25, (now - this.last) / 1000));
    this.last = now;
    if (!Number.isFinite(this.speed)) return maxTicks;
    this.owed += dt * this.speed * this.tickRate;
    const n = Math.min(maxTicks, Math.floor(this.owed));
    this.owed -= n;
    if (this.owed > maxTicks) this.owed = 0;
    return n;
  }

  /** Records that `ticks` ran over `ms` of real time (for the measured speed readout). */
  observe(ticks: number, ms: number): void {
    if (ms <= 0) return;
    const ups = (ticks * 1000) / ms;
    const a = 0.15;
    this.ups = this.ups === 0 ? ups : this.ups * (1 - a) + ups * a;
    this.actualSpeed = this.ups / this.tickRate;
  }
}

/** Log-scale slider mapping between [0, 1] and [SPEED_MIN, SPEED_MAX]. */
export function sliderToSpeed(p: number): number {
  return SPEED_MIN * Math.pow(SPEED_MAX / SPEED_MIN, Math.max(0, Math.min(1, p)));
}

export function speedToSlider(s: number): number {
  return Math.log(Math.max(SPEED_MIN, Math.min(SPEED_MAX, s)) / SPEED_MIN) / Math.log(SPEED_MAX / SPEED_MIN);
}
