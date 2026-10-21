import { useEffect, useMemo, useRef, useState } from 'react';
import type { MlpModelFile } from '../policy/mlp';
import type { Frame } from '../render/frame';
import type { AgentRecord, MatchConfig, MatchJSON } from '../sim/match';
import type { Control } from '../sim/world';
import type { FromWorker, GroundTruth, LeaderRow, SimStats, ToWorker } from '../worker/protocol';

export interface SimUi {
  stats: SimStats;
  leaders: LeaderRow[];
  records: AgentRecord[];
  tick: number;
  time: number;
  ended: boolean;
  error: string | null;
}

/**
 * Owns the simulation worker. The latest frame lives in a ref (read by the canvas every animation
 * frame); a throttled copy of the summary goes into React state ~5 times per second.
 */
export function useSimulation(onGroundTruth?: (gt: GroundTruth[]) => void) {
  const worker = useRef<Worker | null>(null);
  const frame = useRef<Frame | null>(null);
  const prevFrame = useRef<{ frame: Frame; at: number } | null>(null);
  const frameAt = useRef(0);
  /** Smoothed real time between consecutive ticks arriving (ms), used as the interpolation span. */
  const tickInterval = useRef(1000 / 30);
  const savedResolve = useRef<((s: MatchJSON) => void) | null>(null);
  const [ui, setUi] = useState<SimUi>({ stats: { ups: 0, actualSpeed: 0, tickMs: 0, running: false }, leaders: [], records: [], tick: 0, time: 0, ended: false, error: null });
  const lastUi = useRef(0);
  const gtSink = useRef(onGroundTruth);
  gtSink.current = onGroundTruth;

  useEffect(() => {
    const w = new Worker(new URL('../worker/sim.worker.ts', import.meta.url), { type: 'module' });
    worker.current = w;
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (m.type === 'frame') {
        const now = performance.now();
        const cur = frame.current;
        if (cur && m.frame.tick !== cur.tick) {
          // A new tick: the old one becomes the interpolation start. Re-sent frames of the same tick
          // (stats refresh, debug toggles) must not reset the interpolation (that caused the stutter).
          prevFrame.current = { frame: cur, at: frameAt.current };
          const gap = now - frameAt.current;
          if (gap > 0 && gap < 250) tickInterval.current = tickInterval.current * 0.8 + gap * 0.2;
          frameAt.current = now;
        } else if (!cur || m.frame.tick < cur.tick) {
          prevFrame.current = null;
          frameAt.current = now;
        }
        frame.current = m.frame;
        w.postMessage({ type: 'ack' } satisfies ToWorker);
        if (m.gt?.length) gtSink.current?.(m.gt);
        if (now - lastUi.current > 200 || m.frame.ended) {
          lastUi.current = now;
          setUi((u) => ({ ...u, stats: m.stats, leaders: m.leaders, records: m.records, tick: m.frame.tick, time: m.frame.time, ended: m.frame.ended }));
        }
      } else if (m.type === 'saved') {
        savedResolve.current?.(m.snapshot);
        savedResolve.current = null;
      } else if (m.type === 'error') {
        setUi((u) => ({ ...u, error: m.message }));
      }
    };
    return () => w.terminate();
  }, []);

  const api = useMemo(() => {
    const send = (m: ToWorker) => worker.current?.postMessage(m);
    return {
      init: (config: MatchConfig, models: MlpModelFile[], focusId: number, running: boolean, groundTruth = false) => {
        frame.current = null;
        prevFrame.current = null;
        setUi((u) => ({ ...u, error: null }));
        send({ type: 'init', config, models, focusId, running, groundTruth });
      },
      load: (snapshot: MatchJSON, models: MlpModelFile[], focusId: number) => { frame.current = null; prevFrame.current = null; send({ type: 'load', snapshot, models, focusId }); },
      setRunning: (running: boolean) => send({ type: 'running', running }),
      setSpeed: (speed: number) => send({ type: 'speed', speed }),
      step: (ticks = 1) => send({ type: 'step', ticks }),
      human: (control: Control, seq?: number) => send({ type: 'human', control, seq }),
      focus: (id: number) => send({ type: 'focus', id }),
      debug: (enabled: boolean) => send({ type: 'debug', enabled }),
      save: () => new Promise<MatchJSON>((resolve) => { savedResolve.current = resolve; send({ type: 'save' }); }),
    };
  }, []);

  return { api, ui, frame, prevFrame, frameAt, tickInterval };
}

export type Simulation = ReturnType<typeof useSimulation>;
