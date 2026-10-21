import { useState, type KeyboardEvent } from 'react';
import { t } from '../i18n';
import { SPEED_MAX, SPEED_MAX_MODE, SPEED_MIN, sliderToSpeed, speedToSlider } from '../worker/pacing';

const STEPS = 1000;
const QUICK = [0.5, 1, 2, 5, 10];

interface Props {
  /** Target speed; SPEED_MAX_MODE means "as fast as possible". */
  value: number;
  /** Measured simulated seconds per real second (0 while paused). */
  actual: number;
  running: boolean;
  onChange: (speed: number) => void;
}

/** Slider position → speed, snapping to exactly 1× near it so returning to real time is easy. */
function positionToSpeed(pos: number): number {
  const s = sliderToSpeed(pos / STEPS);
  return Math.abs(Math.log(s)) < 0.025 ? 1 : Number(s.toPrecision(3));
}

function RangeInput({ value, onChange, large = false }: { value: number; onChange: (s: number) => void; large?: boolean }) {
  const isMax = value >= SPEED_MAX_MODE;
  const pos = Math.round(speedToSlider(isMax ? SPEED_MAX : value) * STEPS);
  const nudge = (d: number) => onChange(positionToSpeed(Math.min(STEPS, Math.max(0, pos + d))));
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    const step = STEPS / 40, page = STEPS / 10;
    const map: Record<string, () => void> = {
      ArrowRight: () => nudge(step), ArrowUp: () => nudge(step), ArrowLeft: () => nudge(-step), ArrowDown: () => nudge(-step),
      PageUp: () => nudge(page), PageDown: () => nudge(-page), Home: () => onChange(SPEED_MIN), End: () => onChange(SPEED_MAX),
    };
    const fn = map[e.key];
    if (fn) { e.preventDefault(); e.stopPropagation(); fn(); }
  };
  return (
    <input
      type="range" min={0} max={STEPS} step={1} value={pos} aria-label={t.speed.aria} data-testid={large ? 'speed-slider-sheet' : 'speed-slider'}
      aria-valuetext={isMax ? t.speed.max : t.speed.value(value)}
      onChange={(e) => onChange(positionToSpeed(Number(e.target.value)))} onKeyDown={onKey}
      className={`speed-range ${large ? 'speed-range-lg w-full' : 'w-28 lg:w-36'} ${isMax ? 'opacity-50' : ''}`}
    />
  );
}

function Readout({ value, actual, running }: Omit<Props, 'onChange'>) {
  const isMax = value >= SPEED_MAX_MODE;
  const behind = running && !isMax && actual > 0 && actual < value * 0.9;
  return (
    <span className="flex flex-col items-end leading-tight">
      <span className="text-sm font-semibold tabular-nums text-slate-800" data-testid="speed-value">{isMax ? t.speed.max : t.speed.value(value)}</span>
      {running && (isMax || behind) && (
        <span className={`text-[10px] tabular-nums ${behind ? 'text-amber-600' : 'text-slate-400'}`} data-testid="speed-actual">
          {t.speed.actual} {t.speed.value(actual)}
        </span>
      )}
    </span>
  );
}

/**
 * Log-scale speed control: inline slider on wide screens, a chip opening a large slider sheet on
 * phones. Speed only changes how many fixed ticks run per real second, never the outcome.
 */
export function SpeedControl(props: Props) {
  const [open, setOpen] = useState(false);
  const { value, onChange } = props;
  const isMax = value >= SPEED_MAX_MODE;
  return (
    <div className="relative flex items-center">
      <div className="speed-inline items-center gap-2 px-1">
        <span className="whitespace-nowrap text-xs font-medium text-slate-500">{t.speed.label}</span>
        <RangeInput value={value} onChange={onChange} />
        <button type="button" onClick={() => setOpen((o) => !o)} className="min-w-14 rounded-md px-1 py-0.5 text-right hover:bg-slate-100" aria-label={t.speed.sheetTitle} aria-expanded={open}>
          <Readout {...props} />
        </button>
        <button type="button" onClick={() => onChange(1)} data-testid="speed-reset" title={t.speed.reset}
          className={`h-7 rounded-md px-1.5 text-xs font-medium ${value === 1 ? 'text-slate-300' : 'text-sky-700 hover:bg-sky-50'}`}>1×</button>
        <button type="button" onClick={() => onChange(isMax ? 1 : SPEED_MAX_MODE)} data-testid="speed-max" aria-pressed={isMax} title={t.speed.maxNote}
          className={`h-7 rounded-md px-1.5 text-xs font-semibold ${isMax ? 'bg-sky-600 text-white' : 'text-slate-500 hover:bg-slate-100'}`}>MAX</button>
      </div>
      <button type="button" onClick={() => setOpen((o) => !o)} className="speed-chip h-9 items-center rounded-lg px-2 hover:bg-slate-100" aria-label={t.speed.sheetTitle} data-testid="speed-chip">
        <Readout {...props} />
      </button>
      {open && (
        <div className="fixed inset-x-0 bottom-0 z-40 rounded-t-2xl border-t border-slate-200 bg-white p-4 pb-[max(1rem,env(safe-area-inset-bottom))] shadow-2xl sm:absolute sm:inset-x-auto sm:bottom-12 sm:right-0 sm:w-80 sm:rounded-2xl sm:border" role="dialog" aria-label={t.speed.sheetTitle}>
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm font-semibold text-slate-800">{t.speed.sheetTitle}</span>
            <Readout {...props} />
          </div>
          <RangeInput value={value} onChange={onChange} large />
          <div className="mt-3 flex flex-wrap gap-1.5">
            {QUICK.map((q) => (
              <button key={q} type="button" onClick={() => onChange(q)} className={`h-9 rounded-lg px-3 text-sm font-medium ${value === q ? 'bg-sky-600 text-white' : 'bg-slate-100 text-slate-700'}`}>{t.speed.value(q)}</button>
            ))}
            <button type="button" onClick={() => onChange(SPEED_MAX_MODE)} className={`h-9 rounded-lg px-3 text-sm font-semibold ${isMax ? 'bg-sky-600 text-white' : 'bg-slate-100 text-slate-700'}`}>MAX</button>
          </div>
          <p className="mt-2 text-[11px] text-slate-400">{t.speed.maxNote}</p>
          <button type="button" onClick={() => setOpen(false)} className="mt-3 h-10 w-full rounded-lg bg-slate-100 text-sm font-medium text-slate-700">{t.common.close}</button>
        </div>
      )}
    </div>
  );
}
