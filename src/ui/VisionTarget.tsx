import { useEffect, useMemo, useRef, useState } from 'react';
import { NO_DEBUG, labelTheme, type Camera } from '../render/renderer';
import { themeFromParam } from '../render/themes';
import type { MatchConfig } from '../sim/match';
import type { Control } from '../sim/world';
import type { GroundTruth } from '../worker/protocol';
import { GameCanvas } from './GameCanvas';
import { useSimulation } from './useSimulation';

/**
 * Bare game page driven by the Vision Bridge (?view=target). It shows only the game canvas (with the
 * canvas HUD) and accepts only the game's official inputs: pointer heading, Space / mouse button for
 * boost, P to pause, "." to advance one tick, "]" to advance one decision (3 ticks), Enter to start.
 *
 * URL parameters: seed, bots (e.g. "hybrid:4,mpc:2"), speed, theme (default | shift | rand:N),
 * autostart (1 | 0), respawn (1 | 0), selfRespawn (0 = the player never respawns: GAME OVER ends the run),
 * pilot (built-in policy for snake 0 — synthetic data generation only),
 * gt=1 (publish exact state on window.__serpentGT for the calibration recorder; never drawn).
 */
interface GtLog {
  frames: GroundTruth[];
  inputs: { seq: number; epochMs: number; control: Control }[];
  draws: { tick: number; epochMs: number; cam: Camera; w: number; h: number; label: boolean }[];
  config: MatchConfig;
}

declare global {
  interface Window {
    __serpentGT?: GtLog;
    /** Data-generation hook (gt=1 only): redraw the current frame with segmentation label colours. */
    __serpentSetLabel?: (on: boolean) => void;
  }
}

const RING = 20000;
const epoch = () => performance.timeOrigin + performance.now();

function parseBots(s: string | null): Record<string, number> {
  const out: Record<string, number> = {};
  for (const part of (s ?? 'hybrid:3,mpc:2,greedy:3,survival:2,aggressive:2').split(',')) {
    const [id, n] = part.split(':');
    if (id) out[id.trim()] = Math.max(0, Math.min(16, Number(n ?? 1) || 0));
  }
  return out;
}

export function VisionTarget() {
  const params = useMemo(() => new URLSearchParams(window.location.search), []);
  const gt = params.get('gt') === '1';
  const baseTheme = useMemo(() => themeFromParam(params.get('theme')), [params]);
  const [labelMode, setLabelMode] = useState(false);
  const theme = useMemo(() => (labelMode ? labelTheme(baseTheme) : baseTheme), [labelMode, baseTheme]);
  const log = useRef<GtLog | null>(null);
  const sim = useSimulation(gt ? (frames) => {
    const l = log.current;
    if (!l) return;
    l.frames.push(...frames);
    if (l.frames.length > RING) l.frames.splice(0, l.frames.length - RING);
  } : undefined);
  const [started, setStarted] = useState(params.get('autostart') !== '0');
  const seq = useRef(0);

  const config = useMemo<MatchConfig>(() => {
    const bots = parseBots(params.get('bots'));
    const pilot = params.get('pilot');
    const agents = [{ policy: pilot ?? 'human', label: 'player', skin: 0 }];
    for (const [id, n] of Object.entries(bots)) for (let i = 0; i < n; i++) agents.push({ policy: id, label: `${id}-${i + 1}`, skin: agents.length });
    return {
      seed: Number(params.get('seed') ?? 1), agents, decisionInterval: 3, respawn: params.get('respawn') !== '0', respawnDelay: 2,
      respawnExclude: params.get('selfRespawn') === '0' ? [0] : undefined, rules: params.get('food') ? { foodTarget: Number(params.get('food')) } : undefined,
    };
  }, [params]);

  useEffect(() => {
    if (gt) {
      log.current = { frames: [], inputs: [], draws: [], config };
      window.__serpentGT = log.current;
      window.__serpentSetLabel = setLabelMode;
    }
    sim.api.init(config, [], 0, started, gt);
    sim.api.setSpeed(Number(params.get('speed') ?? 1));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'Enter' && !started) { setStarted(true); sim.api.setRunning(true); }
      else if (e.code === 'KeyP') { setStarted((s) => { sim.api.setRunning(!s); return !s; }); }
      else if (e.key === '.') { setStarted(false); sim.api.step(1); }
      else if (e.key === ']') { setStarted(false); sim.api.step(3); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [started, sim.api]);

  const onControl = (c: Control) => {
    seq.current++;
    log.current?.inputs.push({ seq: seq.current, epochMs: epoch(), control: c });
    if (log.current && log.current.inputs.length > RING) log.current.inputs.splice(0, 1000);
    sim.api.human(c, seq.current);
  };

  const me = sim.ui.leaders.find((r) => r.id === 0);
  const dead = me !== undefined && !me.alive && (!config.respawn || config.respawnExclude?.includes(0) === true);
  return (
    <div className="relative h-full w-full bg-slate-100" data-testid="vision-target">
      <GameCanvas
        frame={sim.frame} prevFrame={sim.prevFrame} frameAt={sim.frameAt} debug={NO_DEBUG} labels={false} overview={false} hud inspectId={0}
        onControl={config.agents[0].policy === 'human' ? onControl : undefined} theme={theme} interpolate={false}
        onDrawn={gt ? (tick, cam, w, h) => {
          const l = log.current!;
          const last = l.draws[l.draws.length - 1];
          if (!last || last.tick !== tick || last.label !== labelMode) l.draws.push({ tick, epochMs: epoch(), cam, w, h, label: labelMode });
          if (l.draws.length > RING) l.draws.splice(0, 1000);
        } : undefined}
      />
      {dead && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/70" data-testid="game-over">
          <span className="text-3xl font-bold tracking-widest text-white">GAME OVER</span>
        </div>
      )}
    </div>
  );
}
