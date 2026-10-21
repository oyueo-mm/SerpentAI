/**
 * sfc32 PRNG with a serializable 4×uint32 state. All randomness in the engine and the built-in
 * policies goes through instances of this class so a match is a pure function of (seed, inputs).
 */
export class Rng {
  private a: number;
  private b: number;
  private c: number;
  private d: number;

  constructor(seed: number | readonly number[]) {
    if (typeof seed === 'number') {
      // splitmix32 expands one seed into the four state words.
      let s = seed >>> 0;
      const next = () => {
        s = (s + 0x9e3779b9) >>> 0;
        let z = s;
        z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
        z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
        return (z ^ (z >>> 16)) >>> 0;
      };
      this.a = next(); this.b = next(); this.c = next(); this.d = next();
      for (let i = 0; i < 12; i++) this.nextU32();
    } else {
      // Restoring a saved state: continue exactly where it left off.
      [this.a, this.b, this.c, this.d] = seed.map((v) => v >>> 0);
    }
  }

  nextU32(): number {
    const t = (((this.a + this.b) >>> 0) + this.d) >>> 0;
    this.d = (this.d + 1) >>> 0;
    this.a = (this.b ^ (this.b >>> 9)) >>> 0;
    this.b = (this.c + (this.c << 3)) >>> 0;
    this.c = ((this.c << 21) | (this.c >>> 11)) >>> 0;
    this.c = (this.c + t) >>> 0;
    return t;
  }

  /** Uniform in [0, 1). */
  next(): number {
    return this.nextU32() / 4294967296;
  }

  range(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next();
  }

  /** Uniform integer in [lo, hi] inclusive. */
  int(lo: number, hi: number): number {
    return lo + Math.floor(this.next() * (hi - lo + 1));
  }

  /** Standard normal (Box–Muller). */
  normal(): number {
    const u = 1 - this.next();
    const v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  state(): [number, number, number, number] {
    return [this.a, this.b, this.c, this.d];
  }

  /** Derives an independent stream (e.g. one per snake) without disturbing this one's sequence. */
  static derive(seed: number, stream: number): Rng {
    return new Rng((Math.imul(seed >>> 0, 0x9e3779b1) ^ Math.imul((stream + 1) >>> 0, 0x85ebca77)) >>> 0);
  }
}
