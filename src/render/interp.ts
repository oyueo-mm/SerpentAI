import { wrapAngle } from '../geometry/geom';
import type { Frame } from './frame';
import { trimHead } from './path';

/**
 * Display-only interpolation between two consecutive ticks A → B, at fraction α ∈ [0, 1].
 *
 * The body is moved *along its own path*, not translated: frame B's polyline is shortened at the head
 * end by (1 − α)·|head_B − head_A| (the head slides back onto the path it actually travelled) and the
 * tail is drawn (1 − α) of the way back towards tail_A. The result is always a single contiguous path
 * whose length stays between the two ticks' lengths. The simulation state is never modified.
 */
export function interpolateFrame(a: Frame, b: Frame, alpha: number): Frame {
  const dt = b.tick - a.tick;
  if (dt < 1 || dt > 2) return b;
  const prev = new Map(a.snakes.map((s) => [s.id, s]));
  const k = 1 - Math.max(0, Math.min(1, alpha));
  if (k === 0) return b;
  return {
    ...b,
    snakes: b.snakes.map((s) => {
      const p = prev.get(s.id);
      if (!p || !p.alive || !s.alive || p.lives !== s.lives || s.body.length < 4 || p.body.length < 4) return s;
      const dHead = Math.hypot(s.x - p.x, s.y - p.y);
      if (dHead > 40) return s;
      let body = trimHead(s.body, k * dHead);
      const tbx = s.body[0], tby = s.body[1], tax = p.body[0], tay = p.body[1];
      const dTail = Math.hypot(tax - tbx, tay - tby);
      if (dTail > 1e-6 && dTail < 40) {
        const ext = new Float32Array(body.length + 2);
        ext[0] = tbx + (tax - tbx) * k;
        ext[1] = tby + (tay - tby) * k;
        ext.set(body, 2);
        body = ext;
      }
      const n = body.length;
      return { ...s, x: body[n - 2], y: body[n - 1], heading: s.heading - k * wrapAngle(s.heading - p.heading), body };
    }),
  };
}
