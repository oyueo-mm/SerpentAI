import { useEffect, useState, type ReactNode } from 'react';
import { t } from '../i18n';

/**
 * Small SVG chart kit for the analysis pages. Colours: categorical slots in fixed order (validated:
 * blue, orange, aqua — all-pairs CVD ΔE ≥ 9.2 on white), single-series charts use slot 1 only. Every chart
 * card has a table view (aqua is < 3:1 on white, so values are always available as text too).
 */
export const SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
const INK = { primary: '#0b0b0b', secondary: '#52514e', muted: '#898781', grid: '#e1e0d9', axis: '#c3c2b7' };

export interface Interval { mean: number; lo: number; hi: number; n?: number }

export function useJson<T>(url: string): { data: T | null; error: string | null } {
  const [state, setState] = useState<{ data: T | null; error: string | null }>({ data: null, error: null });
  useEffect(() => {
    let alive = true;
    fetch(url, { cache: 'no-cache' })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d) => alive && setState({ data: d as T, error: null }))
      .catch((e) => alive && setState({ data: null, error: String(e.message ?? e) }));
    return () => { alive = false; };
  }, [url]);
  return state;
}

export function fmt(v: number | undefined | null, d = 1): string {
  return v === undefined || v === null || !Number.isFinite(v) ? '–' : v.toFixed(d);
}

export function fmtCI(x: Interval | undefined, d = 1, scale = 1): string {
  if (!x || !Number.isFinite(x.mean)) return '–';
  return `${(x.mean * scale).toFixed(d)} [${(x.lo * scale).toFixed(d)}, ${(x.hi * scale).toFixed(d)}]`;
}

export function Card({ title, subtitle, children, table, testId }: { title: string; subtitle?: ReactNode; children: ReactNode; table?: ReactNode; testId?: string }) {
  const [showTable, setShowTable] = useState(false);
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4" data-testid={testId}>
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
          {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
        </div>
        {table && (
          <button type="button" onClick={() => setShowTable((s) => !s)} className="shrink-0 rounded-md px-2 py-1 text-xs text-sky-700 hover:bg-sky-50" aria-pressed={showTable}>
            {showTable ? '차트 보기' : '표 보기'}
          </button>
        )}
      </div>
      {showTable && table ? <div className="overflow-x-auto">{table}</div> : children}
    </section>
  );
}

export function Table({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <table className="w-full text-xs">
      <thead><tr className="border-b border-slate-200 text-left text-slate-500">{head.map((h) => <th key={h} className="px-2 py-1.5 font-medium">{h}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i} className="border-b border-slate-100 last:border-0">{r.map((c, j) => <td key={j} className={`px-2 py-1.5 ${j > 0 ? 'tabular-nums' : ''} text-slate-700`}>{c}</td>)}</tr>)}</tbody>
    </table>
  );
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="mb-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-600">
      {items.map((it) => <span key={it.label} className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: it.color }} />{it.label}</span>)}
    </div>
  );
}

/** Rounds an axis maximum up to 1, 2, 2.5 or 5 × 10^k so the five ticks land on readable values. */
export function niceMax(v: number): number {
  if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

/** Horizontal bars (mean) with 95 % CI whiskers; one row per item, single hue. */
export function CiBars({ rows, unit = '', digits = 1, max }: { rows: { label: string; v: Interval; note?: string }[]; unit?: string; digits?: number; max?: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const finite = rows.filter((r) => Number.isFinite(r.v.mean));
  const top = max ?? niceMax(Math.max(1e-9, ...finite.map((r) => (Number.isFinite(r.v.hi) ? r.v.hi : r.v.mean))));
  const W = 520, L = 132, R = 92, rowH = 26;
  const x = (v: number) => L + (Math.max(0, Math.min(top, v)) / top) * (W - L - R);
  return (
    <svg viewBox={`0 0 ${W} ${rows.length * rowH + 22}`} className="w-full max-w-2xl" role="img" aria-label="막대 차트">
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <g key={f}>
          <line x1={x(top * f)} x2={x(top * f)} y1={0} y2={rows.length * rowH} stroke={INK.grid} strokeWidth={1} />
          <text x={x(top * f)} y={rows.length * rowH + 14} fontSize={10} fill={INK.muted} textAnchor="middle">{fmt(top * f, top < 2 ? 2 : top < 10 ? 1 : 0)}</text>
        </g>
      ))}
      {rows.map((r, i) => {
        const y = i * rowH + 6, h = rowH - 12;
        const ok = Number.isFinite(r.v.mean);
        return (
          <g key={r.label} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
            <rect x={0} y={i * rowH} width={W} height={rowH} fill={hover === i ? '#f1f5f9' : 'transparent'} />
            <text x={L - 8} y={y + h / 2 + 4} fontSize={11} fill={INK.primary} textAnchor="end">{r.label}</text>
            {ok && <rect x={L} y={y} width={Math.max(2, x(r.v.mean) - L)} height={h} rx={4} fill={SERIES[0]} />}
            {ok && Number.isFinite(r.v.lo) && (
              <g stroke={INK.primary} strokeWidth={1.5}>
                <line x1={x(r.v.lo)} x2={x(r.v.hi)} y1={y + h / 2} y2={y + h / 2} />
                <line x1={x(r.v.lo)} x2={x(r.v.lo)} y1={y + 3} y2={y + h - 3} />
                <line x1={x(r.v.hi)} x2={x(r.v.hi)} y1={y + 3} y2={y + h - 3} />
              </g>
            )}
            <text x={W - R + 6} y={y + h / 2 + 4} fontSize={11} fill={INK.secondary}>{ok ? `${fmt(r.v.mean, digits)}${unit}` : '–'}{r.note ? ` ${r.note}` : ''}</text>
            {hover === i && ok && <title>{`${r.label}: ${fmt(r.v.mean, digits)}${unit} (95% CI ${fmt(r.v.lo, digits)}–${fmt(r.v.hi, digits)}, n=${r.v.n ?? '?'})`}</title>}
          </g>
        );
      })}
    </svg>
  );
}

/** Direct end-of-line labels pushed apart vertically so they never overlap (≥ 12 px). */
function endLabels(ls: { label: string; x: number; y: number }[]): { label: string; x: number; y: number }[] {
  const out = [...ls].sort((a, b) => a.y - b.y);
  for (let i = 1; i < out.length; i++) if (out[i].y - out[i - 1].y < 12) out[i] = { ...out[i], y: out[i - 1].y + 12 };
  return out;
}

/** Multi-series line chart with a crosshair tooltip. */
export function Lines({ series, xLabel, yLabel, height = 220 }: { series: { label: string; points: [number, number][] }[]; xLabel: string; yLabel: string; height?: number }) {
  const [hx, setHx] = useState<number | null>(null);
  const all = series.flatMap((s) => s.points);
  if (all.length === 0) return <p className="text-xs text-slate-400">{t.common.notMeasured}</p>;
  const W = 560, L = 44, R = 96, T = 10, B = 30;
  const x0 = Math.min(...all.map((p) => p[0])), x1 = Math.max(...all.map((p) => p[0]));
  const yMin = Math.min(0, ...all.map((p) => p[1])), yMax = niceMax(Math.max(...all.map((p) => p[1])) * 1.02) || 1;
  const X = (v: number) => L + ((v - x0) / Math.max(1e-9, x1 - x0)) * (W - L - R);
  const Y = (v: number) => T + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * (height - T - B);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => yMin + f * (yMax - yMin));
  const xt = [0, 0.25, 0.5, 0.75, 1].map((f) => x0 + f * (x1 - x0));
  const near = (s: { points: [number, number][] }) => (hx === null ? null : s.points.reduce((a, p) => (Math.abs(p[0] - hx) < Math.abs(a[0] - hx) ? p : a), s.points[0]));
  return (
    <div>
      <Legend items={series.map((s, i) => ({ label: s.label, color: SERIES[i % SERIES.length] }))} />
      <svg viewBox={`0 0 ${W} ${height}`} className="w-full max-w-3xl" role="img" aria-label={yLabel}
        onMouseMove={(e) => { const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect(); const px = ((e.clientX - r.left) / r.width) * W; setHx(x0 + ((px - L) / (W - L - R)) * (x1 - x0)); }}
        onMouseLeave={() => setHx(null)}>
        {ticks.map((v) => <g key={v}><line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} stroke={INK.grid} /><text x={L - 6} y={Y(v) + 3} fontSize={10} fill={INK.muted} textAnchor="end">{fmt(v, Math.abs(yMax) < 10 ? 1 : 0)}</text></g>)}
        {xt.map((v) => <text key={v} x={X(v)} y={height - 12} fontSize={10} fill={INK.muted} textAnchor="middle">{v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(0)}k` : fmt(v, x1 - x0 < 2 ? 2 : x1 - x0 < 20 ? 1 : 0)}</text>)}
        <text x={W - R} y={height - 1} fontSize={10} fill={INK.muted} textAnchor="end">{xLabel}</text>
        <line x1={L} x2={W - R} y1={Y(yMin)} y2={Y(yMin)} stroke={INK.axis} />
        {series.map((s, i) => (
          <polyline key={s.label} fill="none" stroke={SERIES[i % SERIES.length]} strokeWidth={2} strokeLinejoin="round" points={s.points.map((p) => `${X(p[0])},${Y(p[1])}`).join(' ')} />
        ))}
        {endLabels(series.filter((s) => s.points.length > 0).map((s) => ({ label: s.label, x: X(s.points[s.points.length - 1][0]) + 6, y: Y(s.points[s.points.length - 1][1]) + 4 }))).map((l) => (
          <text key={l.label} x={l.x} y={l.y} fontSize={10} fill={INK.secondary}>{l.label}</text>
        ))}
        {hx !== null && hx >= x0 && hx <= x1 && (
          <g>
            <line x1={X(hx)} x2={X(hx)} y1={T} y2={height - B} stroke={INK.axis} />
            {series.map((s, i) => { const p = near(s); return p ? <circle key={s.label} cx={X(p[0])} cy={Y(p[1])} r={4} fill={SERIES[i % SERIES.length]} stroke="#fff" strokeWidth={2} /> : null; })}
            <g transform={`translate(${Math.min(X(hx) + 8, W - R - 120)}, ${T + 4})`}>
              <rect width={128} height={14 + series.length * 14} rx={6} fill="#fff" stroke={INK.grid} />
              {series.map((s, i) => { const p = near(s); return <text key={s.label} x={8} y={16 + i * 14} fontSize={10} fill={INK.primary}>{`${s.label}: ${p ? fmt(p[1], 2) : '–'}`}</text>; })}
            </g>
          </g>
        )}
      </svg>
      <p className="text-[11px] text-slate-400">{yLabel}</p>
    </div>
  );
}

/** One horizontal stacked bar per row (e.g. latency components), 2 px gaps between segments. */
export function StackedBars({ rows, parts, unit = 'ms' }: { rows: { label: string; values: number[] }[]; parts: string[]; unit?: string }) {
  const W = 520, L = 132, R = 70, rowH = 28;
  const max = Math.max(1e-9, ...rows.map((r) => r.values.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0)));
  return (
    <div>
      <Legend items={parts.map((p, i) => ({ label: p, color: SERIES[i % SERIES.length] }))} />
      <svg viewBox={`0 0 ${W} ${rows.length * rowH}`} className="w-full max-w-2xl" role="img" aria-label="누적 막대">
        {rows.map((r, i) => {
          let acc = L;
          const tot = r.values.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
          return (
            <g key={r.label}>
              <text x={L - 8} y={i * rowH + rowH / 2 + 4} fontSize={11} fill={INK.primary} textAnchor="end">{r.label}</text>
              {r.values.map((v, j) => {
                if (!Number.isFinite(v) || v <= 0) return null;
                const w = (v / max) * (W - L - R);
                const el = <rect key={j} x={acc} y={i * rowH + 6} width={Math.max(1, w - 2)} height={rowH - 12} rx={j === r.values.length - 1 ? 4 : 0} fill={SERIES[j % SERIES.length]}><title>{`${parts[j]}: ${fmt(v, 1)} ${unit}`}</title></rect>;
                acc += w;
                return el;
              })}
              <text x={W - R + 6} y={i * rowH + rowH / 2 + 4} fontSize={11} fill={INK.secondary}>{fmt(tot, 0)} {unit}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function Page({ title, intro, children }: { title: string; intro?: ReactNode; children: ReactNode }) {
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl space-y-4 p-4 sm:p-6">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
          {intro && <div className="mt-1 max-w-3xl text-sm text-slate-600">{intro}</div>}
        </div>
        {children}
      </div>
    </div>
  );
}

export function Missing({ what, how }: { what: string; how: string }) {
  return (
    <div className="rounded-xl border border-dashed border-slate-300 bg-white p-4 text-sm text-slate-500">
      <p>{what}: {t.common.notMeasured}</p>
      <p className="mt-1 font-mono text-xs text-slate-400">{how}</p>
    </div>
  );
}
