/**
 * Body path: a polyline of committed vertices (tail → newest) followed by the live head point.
 *
 * - A vertex is committed when the head is ≥ pathSpacing from the newest committed vertex, so vertex
 *   density depends on distance travelled, not on the tick rate.
 * - The tail is trimmed so the path's arc length equals the target length L exactly (the tail-most
 *   vertex is moved along its segment). When L grows, the tail simply waits until the head has
 *   travelled the extra distance, so growth never teleports any part of the body.
 * - chainLen (sum of committed segment lengths) is updated incrementally and serialized verbatim so a
 *   restored match continues bit-identically.
 */
export class BodyPath {
  /** Flat [x0, y0, x1, y1, …]; live vertices are pts[start … end). */
  pts: number[] = [];
  start = 0;
  chainLen = 0;

  get vertexCount(): number {
    return (this.pts.length - this.start) >> 1;
  }

  /** x of the i-th live vertex (0 = tail). */
  vx(i: number): number { return this.pts[this.start + 2 * i]; }
  vy(i: number): number { return this.pts[this.start + 2 * i + 1]; }

  lastX(): number { return this.pts[this.pts.length - 2]; }
  lastY(): number { return this.pts[this.pts.length - 1]; }

  /** Straight initial body of arc length `length` behind (hx, hy) for heading θ. */
  initStraight(hx: number, hy: number, theta: number, length: number, spacing: number): void {
    this.pts = [];
    this.start = 0;
    this.chainLen = 0;
    const bx = -Math.cos(theta), by = -Math.sin(theta);
    let d = length;
    let prevD = -1;
    while (d > 1e-9) {
      this.pts.push(hx + bx * d, hy + by * d);
      if (prevD >= 0) this.chainLen += prevD - d;
      prevD = d;
      d -= spacing;
    }
    if (this.pts.length === 0) this.pts.push(hx + bx * length, hy + by * length);
  }

  /** Total arc length including the neck from the newest vertex to the head. */
  totalLength(hx: number, hy: number): number {
    return this.chainLen + Math.hypot(hx - this.lastX(), hy - this.lastY());
  }

  /** Called after the head moved to (hx, hy): commits the head as a vertex once it is far enough away. */
  advance(hx: number, hy: number, spacing: number): void {
    const d = Math.hypot(hx - this.lastX(), hy - this.lastY());
    if (d >= spacing) {
      this.pts.push(hx, hy);
      this.chainLen += d;
    }
  }

  /** Cuts the tail so the total arc length is at most `length`. */
  trim(hx: number, hy: number, length: number): void {
    let total = this.totalLength(hx, hy);
    while (total > length) {
      const excess = total - length;
      if (this.vertexCount >= 2) {
        const x0 = this.vx(0), y0 = this.vy(0), x1 = this.vx(1), y1 = this.vy(1);
        const s0 = Math.hypot(x1 - x0, y1 - y0);
        if (s0 <= excess) {
          this.start += 2;
          this.chainLen -= s0;
          total -= s0;
          continue;
        }
        const f = excess / s0;
        this.pts[this.start] = x0 + (x1 - x0) * f;
        this.pts[this.start + 1] = y0 + (y1 - y0) * f;
        this.chainLen -= excess;
      } else {
        // Only the neck remains: slide the single vertex towards the head.
        const x0 = this.vx(0), y0 = this.vy(0);
        const s0 = Math.hypot(hx - x0, hy - y0);
        const f = s0 > 0 ? Math.min(1, excess / s0) : 0;
        this.pts[this.start] = x0 + (hx - x0) * f;
        this.pts[this.start + 1] = y0 + (hy - y0) * f;
      }
      break;
    }
    if (this.start > 4096 && this.start * 2 > this.pts.length) {
      this.pts = this.pts.slice(this.start);
      this.start = 0;
    }
  }

  /** Tail point (the tail-most vertex). */
  tailX(): number { return this.vx(0); }
  tailY(): number { return this.vy(0); }

  toJSON(): { pts: number[]; chainLen: number } {
    return { pts: this.pts.slice(this.start), chainLen: this.chainLen };
  }

  static fromJSON(o: { pts: number[]; chainLen: number }): BodyPath {
    const b = new BodyPath();
    b.pts = o.pts.slice();
    b.chainLen = o.chainLen;
    return b;
  }

  clone(): BodyPath {
    const b = new BodyPath();
    b.pts = this.pts.slice(this.start);
    b.chainLen = this.chainLen;
    return b;
  }
}
