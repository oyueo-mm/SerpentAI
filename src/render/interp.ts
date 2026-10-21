import type { Frame } from './frame';

/**
 * Display-only interpolation between two consecutive frames: each snake is drawn at
 * A + α(B − A) by translating frame B's geometry by (α − 1)(head_B − head_A). The simulation state is
 * untouched; this only removes the 30 Hz stepping from the picture at normal speed.
 */
export function interpolateFrame(a: Frame, b: Frame, alpha: number): Frame {
  if (b.tick - a.tick !== 1 && b.tick - a.tick !== 2) return b;
  const prev = new Map(a.snakes.map((s) => [s.id, s]));
  const k = Math.max(0, Math.min(1, alpha)) - 1;
  return {
    ...b,
    snakes: b.snakes.map((s) => {
      const p = prev.get(s.id);
      if (!p || !p.alive || !s.alive || Math.hypot(s.x - p.x, s.y - p.y) > 40) return s;
      const dx = k * (s.x - p.x), dy = k * (s.y - p.y);
      const body = new Float32Array(s.body.length);
      for (let i = 0; i < body.length; i += 2) { body[i] = s.body[i] + dx; body[i + 1] = s.body[i + 1] + dy; }
      return { ...s, x: s.x + dx, y: s.y + dy, body };
    }),
  };
}
