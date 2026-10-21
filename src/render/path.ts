/**
 * Render-only geometry on body polylines ([x0, y0, x1, y1, …], tail → head). Nothing here feeds back into
 * the simulation: collision, eating and observations always use the engine's exact BodyPath.
 */

export function pathLength(p: ArrayLike<number>): number {
  let L = 0;
  for (let i = 2; i < p.length; i += 2) L += Math.hypot(p[i] - p[i - 2], p[i + 1] - p[i - 1]);
  return L;
}

/** Removes `cut` units of arc length from the head end; the new head lies on the original polyline. */
export function trimHead(p: ArrayLike<number>, cut: number): Float32Array {
  const n = p.length >> 1;
  if (cut <= 0 || n < 2) return Float32Array.from(p as ArrayLike<number>);
  let left = cut;
  let i = n - 1;
  while (i > 0) {
    const seg = Math.hypot(p[2 * i] - p[2 * i - 2], p[2 * i + 1] - p[2 * i - 1]);
    if (seg >= left) {
      const f = seg > 0 ? left / seg : 0;
      const out = new Float32Array(2 * (i + 1));
      for (let k = 0; k < 2 * i; k++) out[k] = p[k];
      out[2 * i] = p[2 * i] + (p[2 * i - 2] - p[2 * i]) * f;
      out[2 * i + 1] = p[2 * i + 1] + (p[2 * i - 1] - p[2 * i + 1]) * f;
      return out;
    }
    left -= seg;
    i--;
  }
  return Float32Array.from([p[0], p[1], p[0], p[1]]); // cut longer than the path: degenerate point
}

/**
 * Smooth curve through a body polyline: quadratic Bézier segments between consecutive edge midpoints,
 * with each vertex as control point (a uniform quadratic B-spline), sampled `samples` times per vertex.
 * Tail and head points are kept exactly. The curve never leaves the polyline's convex corners, so it
 * stays within a fraction of one segment of the physical path.
 */
export function smoothPath(p: ArrayLike<number>, samples = 3): number[] {
  const n = p.length >> 1;
  if (n < 3) return Array.from(p as ArrayLike<number>);
  const out: number[] = [p[0], p[1]];
  let mx = (p[0] + p[2]) / 2, my = (p[1] + p[3]) / 2;
  out.push(mx, my);
  for (let i = 1; i < n - 1; i++) {
    const cx = p[2 * i], cy = p[2 * i + 1];
    const nx = (cx + p[2 * i + 2]) / 2, ny = (cy + p[2 * i + 3]) / 2;
    for (let s = 1; s <= samples; s++) {
      const t = s / samples, u = 1 - t;
      out.push(u * u * mx + 2 * u * t * cx + t * t * nx, u * u * my + 2 * u * t * cy + t * t * ny);
    }
    mx = nx; my = ny;
  }
  out.push(p[2 * n - 2], p[2 * n - 1]);
  return out;
}
