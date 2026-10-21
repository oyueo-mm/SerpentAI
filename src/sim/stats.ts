/** Small statistics helpers for evaluation reports (95 % intervals). */

export interface Interval {
  mean: number;
  lo: number;
  hi: number;
  n: number;
}

// Two-sided 97.5 % Student-t quantiles for small samples; ≥ 120 uses the normal 1.96.
const T975: Record<number, number> = { 1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.06, 30: 2.042, 40: 2.021, 60: 2.0, 120: 1.98 };

export function tQuantile(df: number): number {
  if (df >= 120) return 1.96;
  const keys = Object.keys(T975).map(Number).sort((a, b) => a - b);
  for (const k of keys) if (df <= k) return T975[k];
  return 1.96;
}

/** Mean with a t-based 95 % confidence interval. */
export function meanCI(xs: number[]): Interval {
  const n = xs.length;
  if (n === 0) return { mean: NaN, lo: NaN, hi: NaN, n };
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  if (n === 1) return { mean, lo: mean, hi: mean, n };
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1));
  const h = tQuantile(n - 1) * sd / Math.sqrt(n);
  return { mean, lo: mean - h, hi: mean + h, n };
}

/** Wilson score 95 % interval for a proportion k/n. */
export function wilson(k: number, n: number): Interval {
  if (n === 0) return { mean: NaN, lo: NaN, hi: NaN, n };
  const z = 1.96, p = k / n;
  const den = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / den;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / den;
  return { mean: p, lo: Math.max(0, c - h), hi: Math.min(1, c + h), n };
}
