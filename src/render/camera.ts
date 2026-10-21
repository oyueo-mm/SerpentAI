import type { Frame } from './frame';
import { cameraFor, type Camera } from './renderer';

/**
 * View-only camera controller for the Play/Watch pages. It never touches the simulation: it only turns
 * the latest frame into a Camera for the renderer. The Vision Bridge target page does not use it (its
 * camera stays the fixed follow camera the perception pipeline relies on).
 *
 *   follow   — centred on the focused head; the zoom is the game's follow zoom × a user factor
 *   free     — user-controlled centre and zoom (panning / wheel / pinch), independent of any snake
 *   overview — the whole arena
 */
export type CameraMode = 'follow' | 'free' | 'overview';

export const ZOOM_FACTOR_MIN = 0.35;
export const ZOOM_FACTOR_MAX = 4;
export const FREE_ZOOM_MIN = 0.05;
export const FREE_ZOOM_MAX = 6;

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

export class CameraController {
  mode: CameraMode = 'follow';
  /** User zoom factor applied to the follow camera. */
  followFactor = 1;
  /** Free-camera state (world units, px per unit). */
  cx = 0;
  cy = 0;
  zoom = 1;
  /** Last view actually rendered (used to start a free camera exactly where the user was looking). */
  last: Camera | null = null;
  private listeners = new Set<(m: CameraMode) => void>();

  onModeChange(fn: (m: CameraMode) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  setMode(m: CameraMode): void {
    if (m === 'free' && this.mode !== 'free' && this.last) {
      this.cx = this.last.cx;
      this.cy = this.last.cy;
      this.zoom = this.last.zoom;
    }
    if (m === 'follow') this.followFactor = 1;
    if (this.mode !== m) {
      this.mode = m;
      for (const fn of this.listeners) fn(m);
    }
  }

  /** The camera for this frame. Pure function of the controller state, the frame and the canvas size. */
  view(frame: Frame, width: number, height: number): Camera {
    let cam: Camera;
    if (this.mode === 'free') cam = { cx: this.cx, cy: this.cy, zoom: this.zoom };
    else if (this.mode === 'overview') cam = cameraFor(frame, width, height, true);
    else {
      const base = cameraFor(frame, width, height, false);
      const hasFocus = frame.snakes.some((s) => s.id === frame.focusId && s.alive);
      cam = hasFocus ? { ...base, zoom: base.zoom * this.followFactor } : base;
    }
    this.last = cam;
    return cam;
  }

  /** Pans by a screen-space drag (px). Switches to the free camera. */
  pan(dxPx: number, dyPx: number): void {
    this.setMode('free');
    this.cx -= dxPx / this.zoom;
    this.cy -= dyPx / this.zoom;
  }

  /**
   * Zooms by `factor` keeping the world point under screen point (sx, sy) fixed (free / overview), or
   * scales the follow zoom (follow mode keeps tracking the head at the centre).
   */
  zoomAt(factor: number, sx: number, sy: number, width: number, height: number): void {
    if (this.mode === 'follow') {
      this.followFactor = clamp(this.followFactor * factor, ZOOM_FACTOR_MIN, ZOOM_FACTOR_MAX);
      return;
    }
    this.setMode('free');
    const z0 = this.zoom;
    const z1 = clamp(z0 * factor, FREE_ZOOM_MIN, FREE_ZOOM_MAX);
    const wx = this.cx + (sx - width / 2) / z0;
    const wy = this.cy + (sy - height / 2) / z0;
    this.zoom = z1;
    this.cx = wx - (sx - width / 2) / z1;
    this.cy = wy - (sy - height / 2) / z1;
  }

  /** Screen point → world point under the last rendered view. */
  toWorld(sx: number, sy: number, width: number, height: number): [number, number] {
    const c = this.last ?? { cx: this.cx, cy: this.cy, zoom: this.zoom };
    return [c.cx + (sx - width / 2) / c.zoom, c.cy + (sy - height / 2) / c.zoom];
  }
}
