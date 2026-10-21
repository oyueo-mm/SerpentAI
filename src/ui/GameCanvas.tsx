import { useEffect, useRef, type MutableRefObject } from 'react';
import type { CameraController } from '../render/camera';
import type { Frame } from '../render/frame';
import { interpolateFrame } from '../render/interp';
import { DEFAULT_THEME, cameraFor, render, type Camera, type DebugFlags, type Theme } from '../render/renderer';
import type { Control } from '../sim/world';

interface Props {
  frame: MutableRefObject<Frame | null>;
  prevFrame: MutableRefObject<{ frame: Frame; at: number } | null>;
  frameAt: MutableRefObject<number>;
  /** Smoothed real time between ticks (ms); the interpolation span. */
  tickInterval?: MutableRefObject<number>;
  debug: DebugFlags;
  labels: boolean;
  hud: boolean;
  inspectId: number;
  /**
   * User camera (follow / free / overview, pan, zoom). Without it the canvas uses the fixed follow camera
   * and ignores camera input — the Vision Bridge target page relies on that.
   */
  camera?: CameraController;
  /** Human input: when set, pointer/keyboard steer snake `focus` and the control is reported here. */
  onControl?: (c: Control) => void;
  /** External boost/joystick input (touch controls) merged into the control. */
  touch?: MutableRefObject<{ heading: number | null; boost: boolean }>;
  onSelect?: (id: number) => void;
  onFps?: (fps: number) => void;
  /** ~4 Hz report of the view (for the "AI field of view" panel). */
  onView?: (cam: Camera, width: number, height: number) => void;
  /** Called after each draw with the displayed tick and camera (render log for calibration). */
  onDrawn?: (tick: number, cam: Camera, width: number, height: number) => void;
  theme?: Theme;
  interpolate?: boolean;
  /** Render bodies as smooth curves (render-only; off on the Vision Bridge target page). */
  smooth?: boolean;
  testId?: string;
}

const TAP_SLOP = 6;

/** Full-size game canvas: draws the latest frame every animation frame and turns input into controls. */
export function GameCanvas(p: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const props = useRef(p);
  props.current = p;

  useEffect(() => {
    const el = canvas.current!;
    const ctx = el.getContext('2d', { alpha: false })!;
    let raf = 0;
    let size = { w: 0, h: 0 };
    const pointer = { x: 0, y: 0, active: false, down: false };
    /** Mouse pan in progress (right / middle button). */
    let pan: { id: number; x: number; y: number } | null = null;
    /** Active touches on the canvas (camera gestures). */
    const touches = new Map<number, { x: number; y: number; x0: number; y0: number }>();
    /** Left-button / touch press for click-to-select (selects only if the pointer did not move). */
    let press: { id: number; x: number; y: number; moved: boolean } | null = null;
    /** Heading being steered to; frozen while the camera is being moved. */
    let lastHeading: number | null = null;
    const keys = new Set<string>();
    let lastSent = '';
    let frames = 0, fpsT = performance.now(), viewT = 0;

    const resize = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const r = el.getBoundingClientRect();
      size = { w: r.width, h: r.height };
      el.width = Math.round(r.width * dpr);
      el.height = Math.round(r.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    const cameraBusy = () => pan !== null || touches.size > 0;
    const capture = (id: number) => { try { el.setPointerCapture(id); } catch { /* pointer already gone */ } };

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const P = props.current;
      let f = P.frame.current;
      if (!f || size.w === 0) return;
      const prev = P.prevFrame.current;
      if (P.interpolate !== false && prev && prev.frame.tick < f.tick) {
        const span = P.tickInterval?.current ?? P.frameAt.current - prev.at;
        if (span > 0 && span < 250) f = interpolateFrame(prev.frame, f, (performance.now() - P.frameAt.current) / span);
      }
      const cam = P.camera ? P.camera.view(f, size.w, size.h) : cameraFor(f, size.w, size.h, false);
      render(ctx, f, cam, size.w, size.h, { theme: P.theme ?? DEFAULT_THEME, debug: P.debug, hud: P.hud, labels: P.labels, inspectId: P.inspectId, smooth: P.smooth });
      P.onDrawn?.(f.tick, cam, size.w, size.h);
      const now = performance.now();
      if (P.onView && now - viewT > 250) { viewT = now; P.onView(cam, size.w, size.h); }
      // Human control: touch joystick > keyboard turn > pointer heading. Camera gestures never steer.
      if (P.onControl) {
        const focus = f.snakes.find((s) => s.id === f!.focusId);
        let c: Control = {};
        const left = keys.has('ArrowLeft') || keys.has('KeyA'), right = keys.has('ArrowRight') || keys.has('KeyD');
        const boost = keys.has('Space') || pointer.down || (P.touch?.current.boost ?? false);
        if (P.touch?.current.heading != null) c = { targetHeading: P.touch.current.heading };
        else if (left !== right) c = { turn: left ? -1 : 1 };
        else if (pointer.active && focus) {
          if (!cameraBusy()) {
            const sx = (focus.x - cam.cx) * cam.zoom + size.w / 2, sy = (focus.y - cam.cy) * cam.zoom + size.h / 2;
            if (Math.hypot(pointer.x - sx, pointer.y - sy) > 4) lastHeading = Math.atan2(pointer.y - sy, pointer.x - sx);
          }
          if (lastHeading !== null) c = { targetHeading: lastHeading };
        }
        c.boost = boost;
        const key = JSON.stringify(c);
        if (key !== lastSent) { lastSent = key; P.onControl(c); }
      }
      frames++;
      if (now - fpsT > 1000) { P.onFps?.((frames * 1000) / (now - fpsT)); frames = 0; fpsT = now; }
    };
    raf = requestAnimationFrame(draw);

    const local = (e: PointerEvent | WheelEvent) => {
      const r = el.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };

    const select = (x: number, y: number) => {
      const P = props.current;
      const f = P.frame.current;
      if (!P.onSelect || !f) return;
      const cam = P.camera?.last ?? cameraFor(f, size.w, size.h, false);
      const wx = (x - size.w / 2) / cam.zoom + cam.cx, wy = (y - size.h / 2) / cam.zoom + cam.cy;
      let best = -1, bd = 60 / cam.zoom;
      for (const s of f.snakes) if (s.alive) { const d = Math.hypot(s.x - wx, s.y - wy); if (d < bd) { bd = d; best = s.id; } }
      if (best >= 0) P.onSelect(best);
    };

    const onDown = (e: PointerEvent) => {
      const P = props.current;
      const { x, y } = local(e);
      if (e.pointerType === 'touch') {
        if (!P.camera && !P.onSelect) return;
        capture(e.pointerId);
        touches.set(e.pointerId, { x, y, x0: x, y0: y });
        if (touches.size === 1) press = { id: e.pointerId, x, y, moved: false };
        else press = null;
        return;
      }
      if ((e.button === 1 || e.button === 2) && P.camera) {
        e.preventDefault();
        capture(e.pointerId);
        pan = { id: e.pointerId, x, y };
        return;
      }
      if (e.button === 0) {
        if (P.onControl) pointer.down = true;
        press = { id: e.pointerId, x, y, moved: false };
      }
    };

    const onMove = (e: PointerEvent) => {
      const P = props.current;
      const { x, y } = local(e);
      if (press && press.id === e.pointerId && Math.hypot(x - press.x, y - press.y) > TAP_SLOP) press.moved = true;
      if (e.pointerType === 'touch') {
        const t = touches.get(e.pointerId);
        if (!t || !P.camera) return;
        if (touches.size === 1) {
          if (press?.moved) P.camera.pan(x - t.x, y - t.y);
        } else if (touches.size === 2) {
          const [a, b] = [...touches.entries()].map(([id, v]) => (id === e.pointerId ? { ...v, x, y } : v));
          const [oa, ob] = [...touches.values()];
          const d = Math.hypot(a.x - b.x, a.y - b.y), d0 = Math.hypot(oa.x - ob.x, oa.y - ob.y);
          const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2, mx0 = (oa.x + ob.x) / 2, my0 = (oa.y + ob.y) / 2;
          if (P.camera.mode !== 'free') P.camera.setMode('free');
          P.camera.pan(mx - mx0, my - my0);
          if (d0 > 10 && d > 10) P.camera.zoomAt(d / d0, mx, my, size.w, size.h);
        }
        t.x = x; t.y = y;
        return;
      }
      if (pan && pan.id === e.pointerId && P.camera) {
        P.camera.pan(x - pan.x, y - pan.y);
        pan.x = x; pan.y = y;
        return;
      }
      pointer.x = x; pointer.y = y; pointer.active = true;
    };

    const onUp = (e: PointerEvent) => {
      if (pan && pan.id === e.pointerId) pan = null;
      if (e.pointerType === 'touch') touches.delete(e.pointerId);
      else if (e.button === 0) pointer.down = false;
      if (press && press.id === e.pointerId) {
        if (!press.moved) select(press.x, press.y);
        press = null;
      }
    };

    const onWheel = (e: WheelEvent) => {
      const P = props.current;
      if (!P.camera) return;
      e.preventDefault();
      const { x, y } = local(e);
      P.camera.zoomAt(Math.exp(-e.deltaY * 0.0015), x, y, size.w, size.h);
    };

    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest?.('input, select, textarea')) return;
      if (e.code === 'KeyF' && e.type === 'keydown' && props.current.camera) {
        props.current.camera.setMode('follow');
        return;
      }
      if (['ArrowLeft', 'ArrowRight', 'KeyA', 'KeyD', 'Space'].includes(e.code)) {
        if (e.type === 'keydown') keys.add(e.code); else keys.delete(e.code);
        if (props.current.onControl) e.preventDefault();
      }
    };
    const onBlur = () => { keys.clear(); pointer.down = false; pan = null; touches.clear(); };
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKey);
    window.addEventListener('blur', onBlur);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('wheel', onWheel);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('keyup', onKey);
      window.removeEventListener('blur', onBlur);
    };
  }, []);

  return (
    <canvas
      ref={canvas} data-testid={p.testId ?? 'game-canvas'} className="block h-full w-full touch-none select-none"
      onContextMenu={(e) => e.preventDefault()} onAuxClick={(e) => e.preventDefault()}
    />
  );
}
