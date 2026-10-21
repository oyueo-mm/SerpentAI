/** Planar geometry primitives used by collision, eating, MPC and the observation encoder. */

export const TAU = Math.PI * 2;

/** Wraps an angle to (-π, π]. */
export function wrapAngle(a: number): number {
  let x = a % TAU;
  if (x <= -Math.PI) x += TAU;
  else if (x > Math.PI) x -= TAU;
  return x;
}

/** Squared distance from point P to segment AB. */
export function distPointSegSq(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
  if (t < 0) t = 0; else if (t > 1) t = 1;
  const cx = ax + t * dx - px, cy = ay + t * dy - py;
  return cx * cx + cy * cy;
}

/** Squared distance between segments P1Q1 and P2Q2 (Ericson, Real-Time Collision Detection §5.1.9). */
export function segSegDistSq(
  p1x: number, p1y: number, q1x: number, q1y: number,
  p2x: number, p2y: number, q2x: number, q2y: number,
): number {
  const d1x = q1x - p1x, d1y = q1y - p1y;
  const d2x = q2x - p2x, d2y = q2y - p2y;
  const rx = p1x - p2x, ry = p1y - p2y;
  const a = d1x * d1x + d1y * d1y;
  const e = d2x * d2x + d2y * d2y;
  const f = d2x * rx + d2y * ry;
  const EPS = 1e-12;
  let s: number, t: number;
  if (a <= EPS && e <= EPS) return rx * rx + ry * ry;
  if (a <= EPS) {
    s = 0;
    t = clamp01(f / e);
  } else {
    const c = d1x * rx + d1y * ry;
    if (e <= EPS) {
      t = 0;
      s = clamp01(-c / a);
    } else {
      const b = d1x * d2x + d1y * d2y;
      const denom = a * e - b * b;
      s = denom > EPS ? clamp01((b * f - c * e) / denom) : 0;
      t = (b * s + f) / e;
      if (t < 0) { t = 0; s = clamp01(-c / a); }
      else if (t > 1) { t = 1; s = clamp01((b - c) / a); }
    }
  }
  const cx = p1x + d1x * s - (p2x + d2x * t);
  const cy = p1y + d1y * s - (p2y + d2y * t);
  return cx * cx + cy * cy;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Smallest t ≥ 0 at which the ray O + t·D (|D| = 1) touches the circle (C, r); 0 if O is inside, Infinity if never. */
export function rayCircle(ox: number, oy: number, dx: number, dy: number, cx: number, cy: number, r: number): number {
  const mx = ox - cx, my = oy - cy;
  const c = mx * mx + my * my - r * r;
  if (c <= 0) return 0;
  const b = mx * dx + my * dy;
  if (b > 0) return Infinity;
  const disc = b * b - c;
  if (disc < 0) return Infinity;
  return -b - Math.sqrt(disc);
}

/** Smallest t ≥ 0 at which the ray O + t·D (|D| = 1) touches the capsule of radius r around segment AB. */
export function rayCapsule(
  ox: number, oy: number, dx: number, dy: number,
  ax: number, ay: number, bx: number, by: number, r: number,
): number {
  if (distPointSegSq(ox, oy, ax, ay, bx, by) <= r * r) return 0;
  let best = Math.min(rayCircle(ox, oy, dx, dy, ax, ay, r), rayCircle(ox, oy, dx, dy, bx, by, r));
  const sx = bx - ax, sy = by - ay;
  const len = Math.hypot(sx, sy);
  if (len > 1e-12) {
    const ux = sx / len, uy = sy / len;
    const nx = -uy, ny = ux;
    const s0 = (ox - ax) * nx + (oy - ay) * ny;
    const sn = dx * nx + dy * ny;
    let t = Infinity;
    if (s0 > r && sn < 0) t = (r - s0) / sn;
    else if (s0 < -r && sn > 0) t = (-r - s0) / sn;
    if (t < best) {
      const along = (ox + t * dx - ax) * ux + (oy + t * dy - ay) * uy;
      if (along >= 0 && along <= len) best = t;
    }
  }
  return best;
}

/** Distance along the ray O + t·D (|D| = 1) from inside the square [0, size]² to its boundary. */
export function rayBox(ox: number, oy: number, dx: number, dy: number, size: number): number {
  let t = Infinity;
  if (dx > 1e-12) t = Math.min(t, (size - ox) / dx);
  else if (dx < -1e-12) t = Math.min(t, -ox / dx);
  if (dy > 1e-12) t = Math.min(t, (size - oy) / dy);
  else if (dy < -1e-12) t = Math.min(t, -oy / dy);
  return Math.max(0, t);
}

/**
 * Closest approach of two points moving linearly over t ∈ [0, 1]:
 * A(t) = A0 + t(A1 − A0), B(t) = B0 + t(B1 − B0). Returns the minimum squared distance.
 */
export function closestApproachSq(
  a0x: number, a0y: number, a1x: number, a1y: number,
  b0x: number, b0y: number, b1x: number, b1y: number,
): number {
  const px = a0x - b0x, py = a0y - b0y;
  const vx = (a1x - a0x) - (b1x - b0x), vy = (a1y - a0y) - (b1y - b0y);
  const vv = vx * vx + vy * vy;
  const t = vv > 1e-12 ? clamp01(-(px * vx + py * vy) / vv) : 0;
  const cx = px + vx * t, cy = py + vy * t;
  return cx * cx + cy * cy;
}

/**
 * Exact circular-arc integration of a unicycle: heading θ turning at ω (rad/s) for h seconds at speed v.
 * Writes [x', y', θ'] into out. Independent of how h is subdivided for constant (v, ω).
 */
export function integrateArc(x: number, y: number, theta: number, v: number, omega: number, h: number, out: Float64Array | number[]): void {
  const th1 = theta + omega * h;
  if (Math.abs(omega * h) < 1e-9) {
    const thm = theta + 0.5 * omega * h;
    out[0] = x + v * h * Math.cos(thm);
    out[1] = y + v * h * Math.sin(thm);
  } else {
    const k = v / omega;
    out[0] = x + k * (Math.sin(th1) - Math.sin(theta));
    out[1] = y - k * (Math.cos(th1) - Math.cos(theta));
  }
  out[2] = wrapAngle(th1);
}
