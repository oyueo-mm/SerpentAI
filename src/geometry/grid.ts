/**
 * Uniform grids over the square arena. SegmentGrid is rebuilt every sub-step from the body polylines;
 * PointGrid holds food pellets and is updated incrementally. Queries visit each item at most once.
 */

export class SegmentGrid {
  readonly cols: number;
  private readonly cells: number[][];
  private touched: number[] = [];
  private cap = 0;
  count = 0;
  ax = new Float64Array(0);
  ay = new Float64Array(0);
  bx = new Float64Array(0);
  by = new Float64Array(0);
  /** Capsule radius of each segment. */
  rad = new Float64Array(0);
  /** Owning snake id. */
  owner = new Int32Array(0);
  /** Index of the segment along its body: 0 = tail-most segment. */
  index = new Int32Array(0);
  private mark = new Int32Array(0);
  private stamp = 0;

  constructor(readonly size: number, readonly cell: number) {
    this.cols = Math.max(1, Math.ceil(size / cell));
    this.cells = Array.from({ length: this.cols * this.cols }, () => []);
    this.grow(1024);
  }

  private grow(n: number): void {
    const cap = Math.max(n, this.cap * 2);
    const f = (old: Float64Array) => { const a = new Float64Array(cap); a.set(old); return a; };
    const i = (old: Int32Array) => { const a = new Int32Array(cap); a.set(old); return a; };
    this.ax = f(this.ax); this.ay = f(this.ay); this.bx = f(this.bx); this.by = f(this.by); this.rad = f(this.rad);
    this.owner = i(this.owner); this.index = i(this.index); this.mark = i(this.mark);
    this.cap = cap;
  }

  clear(): void {
    for (const c of this.touched) this.cells[c].length = 0;
    this.touched.length = 0;
    this.count = 0;
  }

  private cellOf(v: number): number {
    const c = Math.floor(v / this.cell);
    return c < 0 ? 0 : c >= this.cols ? this.cols - 1 : c;
  }

  add(owner: number, index: number, ax: number, ay: number, bx: number, by: number, r: number): void {
    if (this.count >= this.cap) this.grow(this.count + 1);
    const k = this.count++;
    this.ax[k] = ax; this.ay[k] = ay; this.bx[k] = bx; this.by[k] = by; this.rad[k] = r;
    this.owner[k] = owner; this.index[k] = index; this.mark[k] = 0;
    const x0 = this.cellOf(Math.min(ax, bx) - r), x1 = this.cellOf(Math.max(ax, bx) + r);
    const y0 = this.cellOf(Math.min(ay, by) - r), y1 = this.cellOf(Math.max(ay, by) + r);
    for (let cy = y0; cy <= y1; cy++) {
      for (let cx = x0; cx <= x1; cx++) {
        const ci = cy * this.cols + cx;
        const cell = this.cells[ci];
        if (cell.length === 0) this.touched.push(ci);
        cell.push(k);
      }
    }
  }

  /** Calls fn(k) once for every segment whose inflated bbox may overlap the query box. */
  query(minX: number, minY: number, maxX: number, maxY: number, fn: (k: number) => void): void {
    if (++this.stamp > 0x3fffffff) { this.stamp = 1; this.mark.fill(0); }
    const st = this.stamp;
    const x0 = this.cellOf(minX), x1 = this.cellOf(maxX), y0 = this.cellOf(minY), y1 = this.cellOf(maxY);
    for (let cy = y0; cy <= y1; cy++) {
      for (let cx = x0; cx <= x1; cx++) {
        const cell = this.cells[cy * this.cols + cx];
        for (let n = 0; n < cell.length; n++) {
          const k = cell[n];
          if (this.mark[k] === st) continue;
          this.mark[k] = st;
          fn(k);
        }
      }
    }
  }
}

/** Grid of point items keyed by integer id (food pellets). */
export class PointGrid {
  readonly cols: number;
  private readonly cells: number[][];
  private readonly where = new Map<number, number>();

  constructor(readonly size: number, readonly cell: number) {
    this.cols = Math.max(1, Math.ceil(size / cell));
    this.cells = Array.from({ length: this.cols * this.cols }, () => []);
  }

  private cellIndex(x: number, y: number): number {
    let cx = Math.floor(x / this.cell), cy = Math.floor(y / this.cell);
    cx = cx < 0 ? 0 : cx >= this.cols ? this.cols - 1 : cx;
    cy = cy < 0 ? 0 : cy >= this.cols ? this.cols - 1 : cy;
    return cy * this.cols + cx;
  }

  insert(id: number, x: number, y: number): void {
    const c = this.cellIndex(x, y);
    this.cells[c].push(id);
    this.where.set(id, c);
  }

  remove(id: number): void {
    const c = this.where.get(id);
    if (c === undefined) return;
    const cell = this.cells[c];
    const i = cell.indexOf(id);
    if (i >= 0) { cell[i] = cell[cell.length - 1]; cell.pop(); }
    this.where.delete(id);
  }

  /** Visits ids in cells overlapping the box (order is cell order, not id order: callers sort if it matters). */
  query(minX: number, minY: number, maxX: number, maxY: number, fn: (id: number) => void): void {
    const c0 = this.cellIndex(minX, minY), c1 = this.cellIndex(maxX, maxY);
    const x0 = c0 % this.cols, y0 = Math.floor(c0 / this.cols), x1 = c1 % this.cols, y1 = Math.floor(c1 / this.cols);
    for (let cy = y0; cy <= y1; cy++) for (let cx = x0; cx <= x1; cx++) {
      const cell = this.cells[cy * this.cols + cx];
      for (let n = 0; n < cell.length; n++) fn(cell[n]);
    }
  }

  clear(): void {
    for (const c of this.cells) c.length = 0;
    this.where.clear();
  }
}
