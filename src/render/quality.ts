/**
 * Graphics quality (render-only). The simulation runs in the worker and never sees these settings, so
 * game speed, physics and AI results are identical at every level.
 */
export type QualityLevel = 'high' | 'medium' | 'low';
export type QualitySetting = 'auto' | QualityLevel;

export interface RenderQuality {
  level: QualityLevel;
  /** Max device-pixel ratio of the canvas backing store. */
  dprCap: number;
  /** Curve samples per body vertex (0 = plain polyline). */
  smoothSamples: number;
  /** Darker outline stroke under each body. */
  outline: boolean;
  /** Boost glow stroke. */
  glow: boolean;
  /** Frame cap (0 = display rate). */
  maxFps: number;
}

export const QUALITY: Record<QualityLevel, RenderQuality> = {
  high: { level: 'high', dprCap: 2, smoothSamples: 3, outline: true, glow: true, maxFps: 0 },
  medium: { level: 'medium', dprCap: 1.5, smoothSamples: 1, outline: true, glow: false, maxFps: 60 },
  low: { level: 'low', dprCap: 1, smoothSamples: 0, outline: false, glow: false, maxFps: 30 },
};

/** FPS each level aims for. */
export const TARGET_FPS: Record<QualityLevel, number> = { high: 60, medium: 60, low: 30 };

export interface DeviceHints {
  coarsePointer: boolean;
  cores: number;
  memoryGB?: number;
}

/** Starting level for 'auto': phones start at medium, low-end phones (≤ 4 cores or ≤ 3 GB) at low. */
export function autoLevel(d: DeviceHints): QualityLevel {
  if (!d.coarsePointer) return 'high';
  if (d.cores <= 4 || (d.memoryGB !== undefined && d.memoryGB <= 3)) return 'low';
  return 'medium';
}

export function deviceHints(): DeviceHints {
  const nav = typeof navigator !== 'undefined' ? (navigator as Navigator & { deviceMemory?: number }) : undefined;
  return {
    coarsePointer: typeof window !== 'undefined' && !!window.matchMedia?.('(pointer: coarse)').matches,
    cores: nav?.hardwareConcurrency ?? 8,
    memoryGB: nav?.deviceMemory,
  };
}

/**
 * 'auto' adaptation: steps the level down after two consecutive FPS readings below 80 % of the target.
 * It never steps back up on its own (no oscillation); choosing a level manually overrides it.
 */
export class AutoQuality {
  private slow = 0;

  constructor(public level: QualityLevel) {}

  /** Feeds one FPS reading (≈ 1 s of animated frames); returns the possibly lowered level. */
  observe(fps: number): QualityLevel {
    if (!(fps > 0)) return this.level;
    if (fps < 0.8 * TARGET_FPS[this.level]) this.slow++;
    else this.slow = 0;
    if (this.slow >= 2 && this.level !== 'low') {
      this.level = this.level === 'high' ? 'medium' : 'low';
      this.slow = 0;
    }
    return this.level;
  }
}

const KEY = 'serpent.quality';

export function loadQualitySetting(): QualitySetting {
  try {
    const v = localStorage.getItem(KEY);
    if (v === 'auto' || v === 'high' || v === 'medium' || v === 'low') return v;
  } catch { /* storage unavailable */ }
  return 'auto';
}

export function saveQualitySetting(q: QualitySetting): void {
  try { localStorage.setItem(KEY, q); } catch { /* storage unavailable */ }
}
