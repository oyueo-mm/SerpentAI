import { VIEW_RADIUS, RAY_COUNT, rayAngle } from '../observation/sensors';
import { decodeAction } from '../policy/action';
import type { Frame, SnakeFrame } from './frame';

/**
 * Canvas renderer. Visual design rules that the Vision pipeline relies on (docs/VISION.md):
 * - the camera is centred on the focused head and always shows a circle of VIEW_RADIUS u around it;
 * - the arena floor carries a grid with lines every GRID_SPACING u (scale and motion cue);
 * - outside the arena is a flat wall colour;
 * - snakes are saturated colours with a darker outline; heads carry two white eyes (heading cue);
 * - pellets are small round dots; death pellets are larger;
 * - the HUD length gauge and boost dot sit at fixed pixel positions in the bottom-left corner.
 */
export const GRID_SPACING = 100;
export const HUD = { margin: 16, barW: 180, barH: 10, dotR: 6 } as const;
/** Gauge fill fraction for a length L: log-scaled between 120 u and 12 000 u. */
export function gaugeFraction(length: number): number {
  return Math.max(0, Math.min(1, Math.log(length / 120) / Math.log(100)));
}
export function lengthFromGauge(f: number): number {
  return 120 * Math.pow(100, f);
}

export interface Theme {
  background: string;
  grid: string;
  wall: string;
  wallEdge: string;
  /** Snake body colours indexed by skin (mod length). */
  snakes: string[];
  pellets: string[];
  outlineDarken: number;
  /** Multiplies drawn body thickness (domain randomisation; 1 = physical radius). */
  thickness: number;
  shadow: boolean;
  hudTrack: string;
  hudFill: string;
  hudBoostOn: string;
  hudBoostOff: string;
  /**
   * Label rendering for segmentation ground truth: same geometry, flat class colours
   * (floor white, wall black, own body red, other bodies green, pellets blue, eyes yellow), no HUD.
   */
  label?: boolean;
}

export const LABEL_COLORS = { floor: '#ffffff', wall: '#000000', self: '#ff0000', other: '#00ff00', pellet: '#0000ff', eye: '#ffff00' } as const;

/** The label version of a theme (keeps its thickness so labels match the rendered geometry). */
export function labelTheme(t: Theme): Theme {
  return { ...t, background: LABEL_COLORS.floor, grid: LABEL_COLORS.floor, wall: LABEL_COLORS.wall, wallEdge: LABEL_COLORS.wall,
    pellets: [LABEL_COLORS.pellet], outlineDarken: 0, shadow: false, label: true };
}

export const DEFAULT_THEME: Theme = {
  background: '#f8fafc',
  grid: '#e2e8f0',
  wall: '#cbd5e1',
  wallEdge: '#94a3b8',
  snakes: ['#2563eb', '#0d9488', '#ea580c', '#db2777', '#7c3aed', '#16a34a', '#d97706', '#dc2626', '#0891b2', '#65a30d', '#4f46e5', '#e11d48', '#9333ea', '#ca8a04', '#059669', '#c026d3'],
  pellets: ['#f59e0b', '#ef4444', '#10b981', '#3b82f6', '#a855f7', '#ec4899', '#14b8a6', '#f97316'],
  outlineDarken: 0.35,
  thickness: 1,
  shadow: false,
  hudTrack: '#e2e8f0',
  hudFill: '#0ea5e9',
  hudBoostOn: '#10b981',
  hudBoostOff: '#94a3b8',
};

export interface DebugFlags {
  colliders: boolean;
  centerline: boolean;
  heading: boolean;
  turnRadius: boolean;
  foodRadius: boolean;
  grid: boolean;
  danger: boolean;
  predictions: boolean;
  trail: boolean;
}

export const NO_DEBUG: DebugFlags = { colliders: false, centerline: false, heading: false, turnRadius: false, foodRadius: false, grid: false, danger: false, predictions: false, trail: false };

export interface Camera {
  cx: number;
  cy: number;
  /** Pixels per world unit (CSS pixels). */
  zoom: number;
}

/** Visible radius in u for a snake of length L: VIEW_RADIUS, widening up to 25 % for long snakes. */
export function viewRadiusFor(length: number): number {
  return VIEW_RADIUS * (1 + 0.25 * Math.max(0, Math.min(1, Math.log(length / 240) / Math.log(20))));
}

export function cameraFor(frame: Frame, width: number, height: number, overview = false): Camera {
  const focus = frame.snakes.find((s) => s.id === frame.focusId && s.alive);
  if (!focus || overview) {
    return { cx: frame.worldSize / 2, cy: frame.worldSize / 2, zoom: Math.min(width, height) / (frame.worldSize * 1.04) };
  }
  return { cx: focus.x, cy: focus.y, zoom: Math.min(width, height) / (2 * viewRadiusFor(focus.length)) };
}

function shade(hex: string, f: number): string {
  const n = parseInt(hex.slice(1), 16);
  const ch = (v: number) => Math.round(f < 0 ? v * (1 + f) : v + (255 - v) * f);
  return `rgb(${ch((n >> 16) & 255)}, ${ch((n >> 8) & 255)}, ${ch(n & 255)})`;
}

export interface RenderOptions {
  theme: Theme;
  debug: DebugFlags;
  hud: boolean;
  labels: boolean;
  /** Snake whose decision overlay is drawn (usually the focus). */
  inspectId: number;
}

export function render(ctx: CanvasRenderingContext2D, frame: Frame, cam: Camera, width: number, height: number, o: RenderOptions): void {
  const T = o.theme;
  const z = cam.zoom;
  const W = frame.worldSize;
  const toX = (x: number) => (x - cam.cx) * z + width / 2;
  const toY = (y: number) => (y - cam.cy) * z + height / 2;
  // Wall (everything) then the arena floor.
  ctx.fillStyle = T.wall;
  ctx.fillRect(0, 0, width, height);
  const ax0 = toX(0), ay0 = toY(0), ax1 = toX(W), ay1 = toY(W);
  ctx.fillStyle = T.background;
  ctx.fillRect(ax0, ay0, ax1 - ax0, ay1 - ay0);
  // Floor grid.
  const vx0 = Math.max(0, cam.cx - width / 2 / z), vx1 = Math.min(W, cam.cx + width / 2 / z);
  const vy0 = Math.max(0, cam.cy - height / 2 / z), vy1 = Math.min(W, cam.cy + height / 2 / z);
  ctx.strokeStyle = T.grid;
  ctx.lineWidth = Math.max(1, z * 1.5);
  ctx.beginPath();
  for (let gx = Math.ceil(vx0 / GRID_SPACING) * GRID_SPACING; gx <= vx1; gx += GRID_SPACING) { ctx.moveTo(toX(gx), toY(vy0)); ctx.lineTo(toX(gx), toY(vy1)); }
  for (let gy = Math.ceil(vy0 / GRID_SPACING) * GRID_SPACING; gy <= vy1; gy += GRID_SPACING) { ctx.moveTo(toX(vx0), toY(gy)); ctx.lineTo(toX(vx1), toY(gy)); }
  ctx.stroke();
  ctx.strokeStyle = T.wallEdge;
  ctx.lineWidth = Math.max(1.5, z * 4);
  ctx.strokeRect(ax0, ay0, ax1 - ax0, ay1 - ay0);

  if (o.debug.grid && frame.debug) {
    const c = frame.debug.gridCell;
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.35)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    for (let gx = Math.ceil(vx0 / c) * c; gx <= vx1; gx += c) { ctx.moveTo(toX(gx), toY(vy0)); ctx.lineTo(toX(gx), toY(vy1)); }
    for (let gy = Math.ceil(vy0 / c) * c; gy <= vy1; gy += c) { ctx.moveTo(toX(vx0), toY(gy)); ctx.lineTo(toX(vx1), toY(gy)); }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Pellets.
  const f = frame.food;
  const margin = 30 / z;
  for (let i = 0; i < f.length; i += 5) {
    const x = f[i], y = f[i + 1];
    if (x < vx0 - margin || x > vx1 + margin || y < vy0 - margin || y > vy1 + margin) continue;
    const r = Math.max(1.2, f[i + 2] * z);
    ctx.fillStyle = T.pellets[f[i + 4] % T.pellets.length];
    ctx.beginPath();
    ctx.arc(toX(x), toY(y), r, 0, Math.PI * 2);
    ctx.fill();
    if (o.debug.foodRadius) {
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.6)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  // Snakes: others first, the focused one on top.
  const order = [...frame.snakes].filter((s) => s.alive).sort((a, b) => (a.id === frame.focusId ? 1 : 0) - (b.id === frame.focusId ? 1 : 0) || a.id - b.id);
  for (const s of order) drawSnake(ctx, s, T, z, toX, toY, s.id === frame.focusId);
  if (o.labels) {
    ctx.font = '600 11px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    for (const s of order) {
      if (s.id === frame.focusId) continue;
      ctx.fillStyle = 'rgba(15, 23, 42, 0.55)';
      ctx.fillText(s.label, toX(s.x), toY(s.y) - s.radius * z - 8);
    }
  }

  if (T.label) return;
  drawDebug(ctx, frame, o, z, toX, toY);
  if (o.hud) drawHud(ctx, frame, T, height);
}

function drawSnake(ctx: CanvasRenderingContext2D, s: SnakeFrame, T: Theme, z: number, toX: (x: number) => number, toY: (y: number) => number, isSelf: boolean): void {
  const color = T.label ? (isSelf ? LABEL_COLORS.self : LABEL_COLORS.other) : T.snakes[s.skin % T.snakes.length];
  const b = s.body;
  const n = b.length / 2;
  if (n < 2) return;
  const width = 2 * s.radius * z * T.thickness;
  const path = new Path2D();
  path.moveTo(toX(b[0]), toY(b[1]));
  for (let i = 1; i < n; i++) path.lineTo(toX(b[2 * i]), toY(b[2 * i + 1]));
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  if (T.shadow) {
    ctx.save();
    ctx.translate(width * 0.12, width * 0.18);
    ctx.strokeStyle = 'rgba(15, 23, 42, 0.18)';
    ctx.lineWidth = width;
    ctx.stroke(path);
    ctx.restore();
  }
  if (s.boosting && !T.label) {
    ctx.strokeStyle = shade(color, 0.65);
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = width + Math.max(4, width * 0.45);
    ctx.stroke(path);
    ctx.globalAlpha = 1;
  }
  ctx.strokeStyle = T.label ? color : shade(color, -T.outlineDarken);
  ctx.lineWidth = width + Math.max(1.5, width * 0.14);
  ctx.stroke(path);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke(path);
  // Eyes.
  const hx = toX(s.x), hy = toY(s.y), r = s.radius * z * T.thickness;
  const fx = Math.cos(s.heading), fy = Math.sin(s.heading), rx = -fy, ry = fx;
  for (const side of [-1, 1]) {
    const ex = hx + fx * r * 0.38 + rx * r * 0.46 * side, ey = hy + fy * r * 0.38 + ry * r * 0.46 * side;
    ctx.fillStyle = T.label ? LABEL_COLORS.eye : '#ffffff';
    ctx.beginPath();
    ctx.arc(ex, ey, Math.max(1.5, r * 0.34), 0, Math.PI * 2);
    ctx.fill();
    if (T.label) continue;
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.arc(ex + fx * r * 0.12, ey + fy * r * 0.12, Math.max(0.8, r * 0.16), 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawDebug(ctx: CanvasRenderingContext2D, frame: Frame, o: RenderOptions, z: number, toX: (x: number) => number, toY: (y: number) => number): void {
  const d = o.debug;
  const dbg = frame.debug;
  for (const s of frame.snakes) {
    if (!s.alive) continue;
    const b = s.body, n = b.length / 2;
    if (d.colliders) {
      // Capsules: the exact collision radius around every segment, plus the head circle.
      ctx.strokeStyle = 'rgba(220, 38, 38, 0.55)';
      ctx.lineWidth = 1;
      for (let i = 0; i < n; i += 3) {
        ctx.beginPath();
        ctx.arc(toX(b[2 * i]), toY(b[2 * i + 1]), s.radius * z, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.strokeStyle = 'rgba(220, 38, 38, 0.95)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(toX(s.x), toY(s.y), s.radius * z, 0, Math.PI * 2);
      ctx.stroke();
    }
    if (d.centerline) {
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.8)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(toX(b[0]), toY(b[1]));
      for (let i = 1; i < n; i++) ctx.lineTo(toX(b[2 * i]), toY(b[2 * i + 1]));
      ctx.stroke();
      ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
      for (let i = 0; i < n; i++) ctx.fillRect(toX(b[2 * i]) - 1, toY(b[2 * i + 1]) - 1, 2, 2);
    }
    if (d.heading) {
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(toX(s.x), toY(s.y));
      ctx.lineTo(toX(s.x + Math.cos(s.heading) * 60), toY(s.y + Math.sin(s.heading) * 60));
      ctx.stroke();
    }
    if (d.turnRadius && Math.abs(s.turn) > 1e-3) {
      // Circle the head is currently turning on: radius v / (|turn| · ω_max).
      const rad = s.speed / (Math.abs(s.turn) * frame.maxTurnRate);
      const side = Math.sign(s.turn);
      const cx = s.x - Math.sin(s.heading) * rad * side, cy = s.y + Math.cos(s.heading) * rad * side;
      ctx.strokeStyle = 'rgba(124, 58, 237, 0.5)';
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(toX(cx), toY(cy), rad * z, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (d.predictions && dbg?.extrapolations[s.id] && s.id !== o.inspectId) {
      const p = dbg.extrapolations[s.id];
      ctx.strokeStyle = 'rgba(14, 165, 233, 0.7)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(toX(p[0]), toY(p[1]));
      for (let i = 2; i < p.length; i += 2) ctx.lineTo(toX(p[i]), toY(p[i + 1]));
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  if (d.trail && dbg && dbg.trail.length > 2) {
    ctx.strokeStyle = 'rgba(234, 88, 12, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(toX(dbg.trail[0]), toY(dbg.trail[1]));
    for (let i = 2; i < dbg.trail.length; i += 2) ctx.lineTo(toX(dbg.trail[i]), toY(dbg.trail[i + 1]));
    ctx.stroke();
  }
  const s = frame.snakes.find((x) => x.id === o.inspectId && x.alive);
  const dec = dbg?.decisions[o.inspectId];
  if (!s || !dec) return;
  if (d.danger && dec.danger) {
    for (let i = 0; i < RAY_COUNT; i++) {
      const a = s.heading + rayAngle(i);
      const v = dec.danger[i];
      ctx.strokeStyle = `rgba(${Math.round(40 + 215 * v)}, ${Math.round(160 * (1 - v))}, 80, 0.75)`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(toX(s.x), toY(s.y));
      const len = 400 * (1 - v) + 20;
      ctx.lineTo(toX(s.x + Math.cos(a) * len), toY(s.y + Math.sin(a) * len));
      ctx.stroke();
    }
  }
  if (d.predictions) {
    if (dec.mpc) {
      // Every candidate the MPC evaluated: grey = rejected, red = predicted collision, blue = chosen.
      dec.mpc.candidates.forEach((c, i) => {
        if (i === dec.mpc!.best) return;
        ctx.strokeStyle = c.hit ? 'rgba(220, 38, 38, 0.55)' : 'rgba(100, 116, 139, 0.35)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(toX(c.path[0][0]), toY(c.path[0][1]));
        for (const [x, y] of c.path) ctx.lineTo(toX(x), toY(y));
        ctx.stroke();
        if (c.hit) {
          ctx.fillStyle = 'rgba(220, 38, 38, 0.8)';
          ctx.fillRect(toX(c.hit[0]) - 3, toY(c.hit[1]) - 3, 6, 6);
        }
      });
      const best = dec.mpc.candidates[dec.mpc.best];
      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(toX(best.path[0][0]), toY(best.path[0][1]));
      for (const [x, y] of best.path) ctx.lineTo(toX(x), toY(y));
      ctx.stroke();
      const boost = decodeAction(best.action).boost;
      if (boost) {
        ctx.fillStyle = '#2563eb';
        ctx.font = '600 11px Inter, system-ui, sans-serif';
        ctx.fillText('BOOST', toX(best.path[best.path.length - 1][0]) + 6, toY(best.path[best.path.length - 1][1]));
      }
    }
    if (dec.target) {
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.9)';
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(toX(s.x), toY(s.y));
      ctx.lineTo(toX(dec.target[0]), toY(dec.target[1]));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(toX(dec.target[0]), toY(dec.target[1]), 5, 0, Math.PI * 2);
      ctx.stroke();
    }
    if (dec.intercept) {
      const [ix, iy] = dec.intercept;
      ctx.strokeStyle = '#dc2626';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(toX(ix) - 7, toY(iy) - 7); ctx.lineTo(toX(ix) + 7, toY(iy) + 7);
      ctx.moveTo(toX(ix) + 7, toY(iy) - 7); ctx.lineTo(toX(ix) - 7, toY(iy) + 7);
      ctx.stroke();
    }
  }
}

/** Canvas HUD (part of the rendered image, so screen-based agents can read it). */
function drawHud(ctx: CanvasRenderingContext2D, frame: Frame, T: Theme, height: number): void {
  const s = frame.snakes.find((x) => x.id === frame.focusId);
  if (!s) return;
  const x = HUD.margin, y = height - HUD.margin - HUD.barH;
  ctx.fillStyle = T.hudTrack;
  ctx.fillRect(x, y, HUD.barW, HUD.barH);
  ctx.fillStyle = T.hudFill;
  ctx.fillRect(x, y, HUD.barW * gaugeFraction(s.alive ? s.length : 120), HUD.barH);
  ctx.fillStyle = s.alive && s.length > frame.boostMinLength ? T.hudBoostOn : T.hudBoostOff;
  ctx.beginPath();
  ctx.arc(x + HUD.barW + 14, y + HUD.barH / 2, HUD.dotR, 0, Math.PI * 2);
  ctx.fill();
}
