import { Rng } from '../sim/rng';
import { DEFAULT_THEME, type Theme } from './renderer';

/**
 * Visual themes. 'default' is the game look; 'shift' is a held-out domain shift used for Sim-to-Visual
 * condition F (never used to generate training images); 'rand:<seed>' is domain randomisation for the
 * synthetic vision dataset.
 */
function hsl(h: number, s: number, l: number): string {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    const c = l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return Math.round(255 * c).toString(16).padStart(2, '0');
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

export const SHIFT_THEME: Theme = {
  ...DEFAULT_THEME,
  background: '#fdf6e3',
  grid: '#eadfc4',
  wall: '#b8b2a7',
  wallEdge: '#8a8478',
  snakes: DEFAULT_THEME.snakes.map((_, i) => hsl((i * 137.5 + 20) % 360, 0.62, 0.42)),
  thickness: 1.12,
  shadow: true,
};

export function randomTheme(seed: number): Theme {
  const r = new Rng(seed);
  const bgL = r.range(0.93, 0.99), bgH = r.range(0, 360), bgS = r.range(0, 0.35);
  const hueBase = r.range(0, 360);
  return {
    ...DEFAULT_THEME,
    background: hsl(bgH, bgS, bgL),
    grid: hsl(bgH, bgS, bgL - r.range(0.04, 0.09)),
    wall: hsl(r.range(0, 360), r.range(0, 0.15), r.range(0.62, 0.8)),
    wallEdge: hsl(0, 0, r.range(0.45, 0.6)),
    snakes: DEFAULT_THEME.snakes.map((_, i) => hsl((hueBase + i * 137.5) % 360, r.range(0.55, 0.85), r.range(0.35, 0.52))),
    pellets: DEFAULT_THEME.pellets.map((_, i) => hsl((hueBase + 40 + i * 45) % 360, r.range(0.6, 0.9), r.range(0.45, 0.6))),
    outlineDarken: r.range(0.2, 0.45),
    thickness: r.range(0.9, 1.1),
    shadow: r.next() < 0.4,
  };
}

export function themeFromParam(p: string | null): Theme {
  if (!p || p === 'default') return DEFAULT_THEME;
  if (p === 'shift') return SHIFT_THEME;
  const m = /^rand:(\d+)$/.exec(p);
  if (m) return randomTheme(Number(m[1]));
  return DEFAULT_THEME;
}
