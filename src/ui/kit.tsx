import { useState, type ReactNode } from 'react';

export function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <defs>
        <linearGradient id="sp-body" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#0d9488" />
          <stop offset="100%" stopColor="#2563eb" />
        </linearGradient>
      </defs>
      <path d="M6 25c0-5 5-6 9.5-6S25 17.5 25 13s-4-6-8.5-6S9 8 9 8" fill="none" stroke="url(#sp-body)" strokeWidth="5.5" strokeLinecap="round" />
      <circle cx="8.3" cy="7.2" r="1.3" fill="#fff" />
      <circle cx="10.6" cy="9.1" r="1.3" fill="#fff" />
    </svg>
  );
}

export function Section({ title, children, defaultOpen = true, right }: { title: string; children: ReactNode; defaultOpen?: boolean; right?: ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="border-b border-slate-100 px-4 py-3">
      <div className="flex items-center justify-between">
        <button type="button" onClick={() => setOpen(!open)} aria-expanded={open} className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-500 hover:text-slate-800">
          <span className={`inline-block transition-transform ${open ? 'rotate-90' : ''}`}>›</span>
          {title}
        </button>
        {right}
      </div>
      {open && <div className="mt-2.5 space-y-2.5">{children}</div>}
    </section>
  );
}

export function Toggle({ label, checked, onChange, hint, testId }: { label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string; testId?: string }) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-3 text-sm text-slate-700">
      <span>
        {label}
        {hint && <span className="block text-[11px] leading-tight text-slate-400">{hint}</span>}
      </span>
      <button
        type="button" role="switch" aria-checked={checked} aria-label={label} data-testid={testId} onClick={() => onChange(!checked)}
        className={`relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-colors ${checked ? 'bg-sky-600' : 'bg-slate-200'}`}
      >
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${checked ? 'left-[18px]' : 'left-0.5'}`} />
      </button>
    </label>
  );
}

export function NumberField({ label, value, onChange, min, max, step = 1, testId, suffix }: {
  label: string; value: number; onChange: (v: number) => void; min: number; max: number; step?: number; testId?: string; suffix?: string;
}) {
  return (
    <label className="flex items-center justify-between gap-3 text-sm text-slate-700">
      <span>{label}</span>
      <span className="flex items-center gap-1">
        <input
          type="number" value={value} min={min} max={max} step={step} data-testid={testId} aria-label={label}
          onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) onChange(Math.max(min, Math.min(max, v))); }}
          className="w-20 rounded-md border border-slate-200 bg-white px-2 py-1 text-right text-sm tabular-nums focus:border-sky-400 focus:outline-none"
        />
        {suffix && <span className="text-xs text-slate-400">{suffix}</span>}
      </span>
    </label>
  );
}

export function Badge({ children, tone = 'slate' }: { children: ReactNode; tone?: 'slate' | 'sky' | 'teal' | 'amber' | 'rose' | 'violet' }) {
  const tones = {
    slate: 'bg-slate-100 text-slate-600', sky: 'bg-sky-50 text-sky-700', teal: 'bg-teal-50 text-teal-700',
    amber: 'bg-amber-50 text-amber-700', rose: 'bg-rose-50 text-rose-700', violet: 'bg-violet-50 text-violet-700',
  };
  return <span className={`inline-flex items-center rounded-md px-1.5 py-0.5 text-[11px] font-medium ${tones[tone]}`}>{children}</span>;
}

export function IconButton({ label, onClick, children, active = false, testId, disabled }: { label: string; onClick: () => void; children: ReactNode; active?: boolean; testId?: string; disabled?: boolean }) {
  return (
    <button
      type="button" onClick={onClick} aria-label={label} title={label} data-testid={testId} disabled={disabled} aria-pressed={active}
      className={`flex h-9 min-w-9 items-center justify-center gap-1 rounded-lg px-2 text-sm font-medium transition-colors disabled:opacity-40 ${active ? 'bg-sky-50 text-sky-700' : 'text-slate-600 hover:bg-slate-100'}`}
    >
      {children}
    </button>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] text-slate-400">{label}</div>
      <div className="truncate text-sm font-semibold tabular-nums text-slate-800">{value}</div>
      {sub && <div className="text-[10px] text-slate-400">{sub}</div>}
    </div>
  );
}

export function downloadText(name: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function formatTime(s: number): string {
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${m}:${r.toString().padStart(2, '0')}`;
}

/** Inline SVG icons (stroke style, 18 px). */
export const Icon = {
  play: <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z" /></svg>,
  pause: <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6.5" y="5" width="4" height="14" rx="1" /><rect x="13.5" y="5" width="4" height="14" rx="1" /></svg>,
  step: <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6 5.5v13l9-6.5z" /><rect x="16" y="5" width="2.5" height="14" rx="1" /></svg>,
  reset: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M4 12a8 8 0 1 0 2.4-5.7" /><path d="M4 4v4.5h4.5" /></svg>,
  settings: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M4 7h10M18 7h2M4 17h4M12 17h8" /><circle cx="16" cy="7" r="2" /><circle cx="10" cy="17" r="2" /></svg>,
  chart: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></svg>,
  save: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M12 4v11M7 10l5 5 5-5M5 20h14" /></svg>,
  load: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M12 15V4M7 9l5-5 5 5M5 20h14" /></svg>,
  globe: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="2" /><path d="M4 12h16M12 4v16" /></svg>,
  close: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>,
};
