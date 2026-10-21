import { describe, expect, it } from 'vitest';
import { distPointSegSq } from '../geometry/geom';
import { encodeObs } from '../observation/obs';
import { senseWorld } from '../observation/sensors';
import { DEFAULT_RULES } from '../sim/config';
import { Match } from '../sim/match';
import { CameraController } from './camera';
import { buildFrame, type Frame } from './frame';
import { interpolateFrame } from './interp';
import { pathLength, smoothPath, trimHead } from './path';
import { cameraFor } from './renderer';

const R = DEFAULT_RULES;
/** Longest committed segment the engine can produce (pathSpacing + one sub-step). */
const MAX_SEG = R.pathSpacing + R.maxSubstepDistance;
/** The tail moves at most one tick of boosted travel plus the boost length loss per tick. */
const MAX_TAIL_STEP = (R.boostSpeed + R.boostCost) / R.tickRate;

function maxSegment(p: ArrayLike<number>): number {
  let m = 0;
  for (let i = 2; i < p.length; i += 2) m = Math.max(m, Math.hypot(p[i] - p[i - 2], p[i + 1] - p[i - 1]));
  return m;
}

function distToPolyline(x: number, y: number, p: ArrayLike<number>): number {
  let best = Infinity;
  for (let i = 2; i < p.length; i += 2) best = Math.min(best, distPointSegSq(x, y, p[i - 2], p[i - 1], p[i], p[i + 1]));
  return Math.sqrt(best);
}

/** Consecutive frames of a busy match: hard turns, boosts and growth (death pellets, food). */
function frames(ticks: number, seed = 3): Frame[] {
  const m = new Match({ seed, agents: ['aggressive', 'hybrid', 'mpc', 'greedy', 'random', 'survival'].map((policy) => ({ policy })), respawn: true });
  const out: Frame[] = [];
  for (let t = 0; t < ticks; t++) {
    m.step();
    out.push(buildFrame(m, 0, { debug: false, trail: [] }));
  }
  return out;
}

describe('render path continuity (display only)', () => {
  const fr = frames(240);

  it('frame bodies are contiguous and as long as the physical body', () => {
    for (const f of fr) for (const s of f.snakes) {
      if (!s.alive) continue;
      expect(maxSegment(s.body)).toBeLessThanOrEqual(MAX_SEG + 1e-3);
      expect(Math.abs(pathLength(s.body) - s.length)).toBeLessThan(0.05 * s.length); // tail still catching up after growth
      expect(pathLength(s.body)).toBeLessThanOrEqual(s.length + 1e-2);
    }
  });

  it('interpolated bodies stay one contiguous path whose length lies between the two ticks', () => {
    let checked = 0;
    for (let k = 1; k < fr.length; k++) {
      const a = fr[k - 1], b = fr[k];
      let prevHead = new Map<number, number>();
      for (const alpha of [0, 0.25, 0.5, 0.75, 1]) {
        const f = interpolateFrame(a, b, alpha);
        for (const s of f.snakes) {
          const sa = a.snakes[s.id], sb = b.snakes[s.id];
          if (!s.alive || !sa.alive || sa.lives !== sb.lives) continue;
          expect(maxSegment(s.body)).toBeLessThanOrEqual(Math.max(MAX_SEG, MAX_TAIL_STEP) + 1e-3);
          const L = pathLength(s.body), La = pathLength(sa.body), Lb = pathLength(sb.body);
          expect(L).toBeGreaterThanOrEqual(Math.min(La, Lb) - 0.6);
          expect(L).toBeLessThanOrEqual(Math.max(La, Lb) + 0.6);
          // The head moves forward monotonically from tick A to tick B.
          const d = Math.hypot(s.x - sa.x, s.y - sa.y);
          expect(d).toBeGreaterThanOrEqual((prevHead.get(s.id) ?? 0) - 1e-3);
          prevHead.set(s.id, d);
          if (alpha === 1) { expect(s.x).toBeCloseTo(sb.x, 3); expect(s.y).toBeCloseTo(sb.y, 3); }
          if (alpha === 0) expect(d).toBeLessThan(0.5);
          checked++;
        }
      }
      prevHead = new Map();
    }
    expect(checked).toBeGreaterThan(5000);
  });

  it('smooth curves stay on the physical path (render-only) and keep its length', () => {
    for (const f of fr.filter((_, i) => i % 10 === 0)) for (const s of f.snakes) {
      if (!s.alive || s.body.length < 6) continue;
      const sm = smoothPath(s.body);
      expect(sm[0]).toBeCloseTo(s.body[0], 4);
      expect(sm[sm.length - 1]).toBeCloseTo(s.body[s.body.length - 1], 4);
      for (let i = 0; i < sm.length; i += 2) expect(distToPolyline(sm[i], sm[i + 1], s.body)).toBeLessThan(1.0);
      expect(pathLength(sm)).toBeGreaterThan(0.98 * pathLength(s.body));
      expect(maxSegment(sm)).toBeLessThanOrEqual(MAX_SEG);
    }
  });

  it('skips interpolation across jumps (high speed, respawn) instead of drawing a wrong path', () => {
    const a = fr[10], b = fr[14];
    expect(interpolateFrame(a, b, 0.5)).toBe(b);
    expect(trimHead([0, 0, 10, 0], 4)).toEqual(Float32Array.from([0, 0, 6, 0]));
  });
});

describe('camera controller (view only)', () => {
  const f = frames(5)[4];

  it('follow mode equals the game camera until the user zooms', () => {
    const c = new CameraController();
    expect(c.view(f, 1200, 800)).toEqual(cameraFor(f, 1200, 800));
    c.zoomAt(2, 600, 400, 1200, 800);
    const v = c.view(f, 1200, 800);
    expect(v.cx).toBe(cameraFor(f, 1200, 800).cx);
    expect(v.zoom).toBeCloseTo(cameraFor(f, 1200, 800).zoom * 2);
  });

  it('pan switches to a free camera that starts where the user was looking', () => {
    const c = new CameraController();
    const v0 = c.view(f, 1200, 800);
    c.pan(100, -50);
    expect(c.mode).toBe('free');
    const v = c.view(f, 1200, 800);
    expect(v.cx).toBeCloseTo(v0.cx - 100 / v0.zoom);
    expect(v.cy).toBeCloseTo(v0.cy + 50 / v0.zoom);
  });

  it('zooming keeps the world point under the cursor fixed and is clamped', () => {
    const c = new CameraController();
    c.view(f, 1000, 800);
    c.setMode('free');
    const before = c.toWorld(250, 200, 1000, 800);
    c.zoomAt(1.7, 250, 200, 1000, 800);
    c.view(f, 1000, 800);
    const after = c.toWorld(250, 200, 1000, 800);
    expect(after[0]).toBeCloseTo(before[0], 6);
    expect(after[1]).toBeCloseTo(before[1], 6);
    for (let i = 0; i < 50; i++) c.zoomAt(2, 0, 0, 1000, 800);
    expect(c.zoom).toBeLessThanOrEqual(6);
  });

  it('selecting another snake or a new frame does not reset a free camera; F/follow does', () => {
    const c = new CameraController();
    c.view(f, 1200, 800);
    c.pan(300, 0);
    const v = c.view(f, 1200, 800);
    expect(c.view({ ...f, focusId: 3, tick: f.tick + 1 }, 1200, 800)).toEqual(v);
    c.setMode('follow');
    expect(c.view(f, 1200, 800)).toEqual(cameraFor(f, 1200, 800));
  });

  it('the AI observation does not depend on the camera', () => {
    const m = new Match({ seed: 9, agents: [{ policy: 'hybrid' }, { policy: 'mpc' }, { policy: 'greedy' }] });
    for (let t = 0; t < 60; t++) m.step();
    const s = m.world.snakes[0];
    const o1 = encodeObs(senseWorld(m.world, s), { turn: 0, boost: false });
    const c = new CameraController();
    const fr2 = buildFrame(m, 0, { debug: true, trail: [] });
    c.view(fr2, 800, 600); c.pan(500, 500); c.zoomAt(3, 10, 10, 800, 600); c.setMode('overview'); c.view(fr2, 800, 600);
    const o2 = encodeObs(senseWorld(m.world, s), { turn: 0, boost: false });
    expect(Array.from(o2)).toEqual(Array.from(o1));
    // The overlay draws exactly the engine sensors of this tick.
    expect(fr2.debug?.sensors).toEqual(senseWorld(m.world, s));
  });
});
