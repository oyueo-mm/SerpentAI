import { useEffect, useRef, type MutableRefObject } from 'react';
import type { Frame } from '../render/frame';
import { interpolateFrame } from '../render/interp';
import { DEFAULT_THEME, cameraFor, render, type Camera, type DebugFlags, type Theme } from '../render/renderer';
import type { Control } from '../sim/world';

interface Props {
  frame: MutableRefObject<Frame | null>;
  prevFrame: MutableRefObject<{ frame: Frame; at: number } | null>;
  frameAt: MutableRefObject<number>;
  debug: DebugFlags;
  labels: boolean;
  overview: boolean;
  hud: boolean;
  inspectId: number;
  /** Human input: when set, pointer/keyboard steer snake `focus` and the control is reported here. */
  onControl?: (c: Control) => void;
  /** External boost/joystick input (touch controls) merged into the control. */
  touch?: MutableRefObject<{ heading: number | null; boost: boolean }>;
  onSelect?: (id: number) => void;
  onFps?: (fps: number) => void;
  /** Called after each draw with the displayed tick and camera (render log for calibration). */
  onDrawn?: (tick: number, cam: Camera, width: number, height: number) => void;
  theme?: Theme;
  interpolate?: boolean;
  testId?: string;
}

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
    const keys = new Set<string>();
    let lastSent = '';
    let frames = 0, fpsT = performance.now();

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

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const P = props.current;
      let f = P.frame.current;
      if (!f || size.w === 0) return;
      const prev = P.prevFrame.current;
      if (P.interpolate !== false && prev) {
        const span = P.frameAt.current - prev.at;
        if (span > 0 && span < 200) f = interpolateFrame(prev.frame, f, (performance.now() - P.frameAt.current) / span);
      }
      const cam = cameraFor(f, size.w, size.h, P.overview);
      render(ctx, f, cam, size.w, size.h, { theme: P.theme ?? DEFAULT_THEME, debug: P.debug, hud: P.hud, labels: P.labels, inspectId: P.inspectId });
      P.onDrawn?.(f.tick, cam, size.w, size.h);
      // Human control: keyboard turn wins over pointer heading; touch joystick wins over both.
      if (P.onControl) {
        const focus = f.snakes.find((s) => s.id === f!.focusId);
        let c: Control = {};
        const left = keys.has('ArrowLeft') || keys.has('KeyA'), right = keys.has('ArrowRight') || keys.has('KeyD');
        const boost = keys.has('Space') || pointer.down || (P.touch?.current.boost ?? false);
        if (P.touch?.current.heading != null) c = { targetHeading: P.touch.current.heading };
        else if (left !== right) c = { turn: left ? -1 : 1 };
        else if (pointer.active && focus) {
          const sx = (focus.x - cam.cx) * cam.zoom + size.w / 2, sy = (focus.y - cam.cy) * cam.zoom + size.h / 2;
          if (Math.hypot(pointer.x - sx, pointer.y - sy) > 4) c = { targetHeading: Math.atan2(pointer.y - sy, pointer.x - sx) };
        }
        c.boost = boost;
        const key = JSON.stringify(c);
        if (key !== lastSent) { lastSent = key; P.onControl(c); }
      }
      frames++;
      const now = performance.now();
      if (now - fpsT > 1000) { P.onFps?.((frames * 1000) / (now - fpsT)); frames = 0; fpsT = now; }
    };
    raf = requestAnimationFrame(draw);

    const onMove = (e: PointerEvent) => {
      if (e.pointerType === 'touch') return;
      const r = el.getBoundingClientRect();
      pointer.x = e.clientX - r.left; pointer.y = e.clientY - r.top; pointer.active = true;
    };
    const onDown = (e: PointerEvent) => {
      const P = props.current;
      if (e.pointerType !== 'touch' && P.onControl && e.button === 0) pointer.down = true;
      if (P.onSelect && P.frame.current) {
        const f = P.frame.current, r = el.getBoundingClientRect();
        const cam = cameraFor(f, size.w, size.h, P.overview);
        const x = (e.clientX - r.left - size.w / 2) / cam.zoom + cam.cx, y = (e.clientY - r.top - size.h / 2) / cam.zoom + cam.cy;
        let best = -1, bd = 60 / cam.zoom;
        for (const s of f.snakes) if (s.alive) { const d = Math.hypot(s.x - x, s.y - y); if (d < bd) { bd = d; best = s.id; } }
        if (best >= 0) P.onSelect(best);
      }
    };
    const onUp = () => { pointer.down = false; };
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest?.('input, select, textarea')) return;
      if (['ArrowLeft', 'ArrowRight', 'KeyA', 'KeyD', 'Space'].includes(e.code)) {
        if (e.type === 'keydown') keys.add(e.code); else keys.delete(e.code);
        if (props.current.onControl) e.preventDefault();
      }
    };
    const onBlur = () => { keys.clear(); pointer.down = false; };
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerdown', onDown);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKey);
    window.addEventListener('blur', onBlur);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerdown', onDown);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('keyup', onKey);
      window.removeEventListener('blur', onBlur);
    };
  }, []);

  return <canvas ref={canvas} data-testid={p.testId ?? 'game-canvas'} className="block h-full w-full touch-none select-none" onContextMenu={(e) => e.preventDefault()} />;
}
