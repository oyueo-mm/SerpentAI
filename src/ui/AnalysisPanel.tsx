import { useEffect, useState, type MutableRefObject } from 'react';
import type { DecisionDebug } from '../ai/types';
import { t } from '../i18n';
import { TURN_LEVELS, decodeAction } from '../policy/action';
import type { MlpModelFile } from '../policy/mlp';
import type { Frame, SnakeFrame } from '../render/frame';
import type { AgentRecord } from '../sim/match';
import type { LeaderRow, SimStats } from '../worker/protocol';
import { Badge, Section, Stat, formatTime } from './kit';
import { policyName } from './SettingsPanel';

export function policyKind(id: string, models: MlpModelFile[]): { label: string; tone: 'slate' | 'sky' | 'teal' | 'violet' } {
  if (id === 'human') return { label: t.policyKind.human, tone: 'slate' };
  if (id === 'mpc' || id === 'mpc-lite') return { label: t.policyKind.mpc, tone: 'teal' };
  const m = models.find((x) => x.id === id);
  if (m) return { label: `${t.policyKind.learned} · ${m.algo.toUpperCase()}`, tone: 'violet' };
  return { label: t.policyKind.heuristic, tone: 'sky' };
}

function TurnBar({ turn }: { turn: number }) {
  return (
    <div className="relative h-2 w-full rounded-full bg-slate-100" aria-label={`${t.inspector.turn} ${turn.toFixed(2)}`}>
      <div className="absolute top-0 h-2 w-0.5 bg-slate-300" style={{ left: '50%' }} />
      <div className="absolute top-0 h-2 rounded-full bg-sky-500" style={turn >= 0 ? { left: '50%', width: `${turn * 50}%` } : { right: '50%', width: `${-turn * 50}%` }} />
    </div>
  );
}

function actionLabel(a: number): string {
  const { turn, boost } = decodeAction(a);
  const dir = turn === 0 ? '직진' : `${turn < 0 ? '좌' : '우'} ${Math.abs(turn).toFixed(2)}`;
  return `${dir}${boost ? ' +B' : ''}`;
}

function MpcTable({ dec }: { dec: DecisionDebug }) {
  const m = dec.mpc!;
  const rows = m.candidates.map((c, i) => ({ c, i })).sort((a, b) => b.c.cost.total - a.c.cost.total).slice(0, 8);
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px] text-slate-500">
        <span>{t.inspector.mpcTitle}</span>
        <span className="tabular-nums">{t.inspector.mpcMs(m.evalMs)}</span>
      </div>
      <table className="w-full text-[11px] tabular-nums" data-testid="mpc-table">
        <thead><tr className="text-slate-400">{t.inspector.mpcCols.map((c) => <th key={c} className="px-0.5 text-right font-medium first:text-left">{c}</th>)}</tr></thead>
        <tbody>
          {rows.map(({ c, i }) => (
            <tr key={i} className={i === m.best ? 'font-semibold text-sky-700' : c.hit ? 'text-rose-500' : 'text-slate-600'}>
              <td className="px-0.5">{actionLabel(c.action)}</td>
              <td className="px-0.5 text-right">{c.cost.food.toFixed(0)}</td>
              <td className="px-0.5 text-right">{c.cost.terminal.toFixed(1)}</td>
              <td className="px-0.5 text-right">{c.cost.collision > 0 ? c.cost.collision.toFixed(2) : '–'}</td>
              <td className="px-0.5 text-right">{c.cost.clearance.toFixed(2)}</td>
              <td className="px-0.5 text-right">{c.cost.boost.toFixed(0)}</td>
              <td className="px-0.5 text-right">{c.cost.total.toFixed(0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Probs({ probs }: { probs: number[] }) {
  return (
    <div>
      <div className="mb-1 text-[11px] text-slate-500">{t.inspector.probs}</div>
      <div className="grid grid-cols-9 items-end gap-0.5" style={{ height: 56 }} data-testid="action-probs">
        {TURN_LEVELS.map((turn, ti) => (
          <div key={turn} className="flex h-full flex-col justify-end gap-0.5" title={`${actionLabel(ti * 2)} / +B`}>
            <div className="w-full rounded-sm bg-teal-500" style={{ height: `${probs[ti * 2 + 1] * 100}%` }} />
            <div className="w-full rounded-sm bg-sky-500" style={{ height: `${probs[ti * 2] * 100}%` }} />
          </div>
        ))}
      </div>
      <div className="mt-0.5 flex justify-between text-[10px] text-slate-400"><span>좌</span><span>직진</span><span>우</span></div>
    </div>
  );
}

interface Props {
  frame: MutableRefObject<Frame | null>;
  inspectId: number;
  onInspect: (id: number) => void;
  leaders: LeaderRow[];
  records: AgentRecord[];
  stats: SimStats;
  fps: number;
  time: number;
  models: MlpModelFile[];
}

export function AnalysisPanel({ frame, inspectId, onInspect, leaders, records, stats, fps, time, models }: Props) {
  const [snap, setSnap] = useState<{ s?: SnakeFrame; dec?: DecisionDebug }>({});
  useEffect(() => {
    const id = setInterval(() => {
      const f = frame.current;
      if (!f) return;
      setSnap({ s: f.snakes.find((x) => x.id === inspectId), dec: f.debug?.decisions[inspectId] });
    }, 250);
    return () => clearInterval(id);
  }, [frame, inspectId]);
  const s = snap.s, dec = snap.dec, rec = records[inspectId];
  const kind = s ? policyKind(s.policy, models) : null;
  return (
    <div className="flex h-full flex-col" data-testid="analysis-panel">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <Section title={t.inspector.title}>
          {s && kind ? (
            <>
              <div className="flex items-center justify-between gap-2">
                <span className="truncate font-semibold text-slate-800" data-testid="inspect-label">{s.label}</span>
                <span className="flex gap-1"><Badge tone={kind.tone}>{kind.label}</Badge>{policyName(s.policy, models) !== kind.label && <Badge>{policyName(s.policy, models)}</Badge>}</span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <Stat label={t.inspector.length} value={s.alive ? s.length.toFixed(0) : '–'} />
                <Stat label={t.inspector.radius} value={s.alive ? s.radius.toFixed(1) : '–'} />
                <Stat label={t.inspector.speed} value={s.alive ? s.speed.toFixed(0) : '–'} />
              </div>
              <div>
                <div className="mb-1 flex justify-between text-[11px] text-slate-500"><span>{t.inspector.command}</span><span>{s.boosting ? <Badge tone="teal">{t.inspector.boost}</Badge> : null}</span></div>
                <TurnBar turn={s.turn} />
              </div>
              {dec ? (
                <>
                  {dec.reason && <div className="text-xs text-slate-600">{t.inspector.reason}: <span className="font-medium text-slate-800" data-testid="inspect-reason">{t.reason[dec.reason] ?? dec.reason}</span></div>}
                  {dec.mpc && <MpcTable dec={dec} />}
                  {dec.probs && <Probs probs={dec.probs} />}
                </>
              ) : (
                s.policy !== 'human' && <p className="text-xs text-slate-400">{t.inspector.noDecision}</p>
              )}
              {rec && (
                <div className="grid grid-cols-4 gap-2 border-t border-slate-100 pt-2">
                  <Stat label={t.inspector.lives} value={s.lives} />
                  <Stat label={t.inspector.deaths} value={rec.deaths} />
                  <Stat label={t.hud.kills} value={rec.kills} />
                  <Stat label={t.inspector.eaten} value={rec.eaten.toFixed(0)} />
                </div>
              )}
            </>
          ) : <p className="text-xs text-slate-400">{t.common.loading}</p>}
        </Section>
        <Section title={t.leaders.title}>
          <ol className="space-y-0.5" data-testid="leaderboard">
            {leaders.slice(0, 16).map((r, i) => (
              <li key={r.id}>
                <button type="button" onClick={() => onInspect(r.id)} className={`flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left text-xs ${r.id === inspectId ? 'bg-sky-50' : 'hover:bg-slate-50'} ${r.alive ? 'text-slate-700' : 'text-slate-400 line-through'}`}>
                  <span className="w-5 text-right tabular-nums text-slate-400">{i + 1}</span>
                  <span className="flex-1 truncate">{r.label}</span>
                  <span className="tabular-nums">{r.alive ? r.length.toFixed(0) : t.leaders.dead}</span>
                </button>
              </li>
            ))}
          </ol>
        </Section>
        <Section title="성능" defaultOpen={false}>
          <div className="grid grid-cols-2 gap-2">
            <Stat label={t.stats.simTime} value={formatTime(time)} />
            <Stat label={t.stats.ups} value={stats.ups.toFixed(0)} />
            <Stat label={t.stats.fps} value={fps.toFixed(0)} />
            <Stat label={t.stats.tick} value={`${stats.tickMs.toFixed(2)} ms`} />
          </div>
        </Section>
      </div>
    </div>
  );
}
