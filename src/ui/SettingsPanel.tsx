import { BUILTIN_POLICIES } from '../ai/registry';
import { t } from '../i18n';
import type { MlpModelFile } from '../policy/mlp';
import type { DebugFlags } from '../render/renderer';
import { NumberField, Section, Toggle } from './kit';
import type { ArenaSettings, UiSettings } from './settings';

interface Props {
  arena: ArenaSettings;
  onArena: (a: ArenaSettings) => void;
  ui: UiSettings;
  onUi: (u: UiSettings) => void;
  models: MlpModelFile[];
  onApply: () => void;
  onClose: () => void;
}

export function policyName(id: string, models: MlpModelFile[] = []): string {
  return t.policies[id] ?? models.find((m) => m.id === id)?.label ?? id;
}

export function SettingsPanel({ arena, onArena, ui, onUi, models, onApply, onClose }: Props) {
  const ids = [...BUILTIN_POLICIES.filter((p) => p !== 'mpc-lite'), ...models.map((m) => m.id)];
  const setBots = (id: string, n: number) => onArena({ ...arena, bots: { ...arena.bots, [id]: n } });
  const setDebug = (k: keyof DebugFlags, v: boolean) => onUi({ ...ui, debug: { ...ui.debug, [k]: v } });
  return (
    <div className="flex h-full flex-col" data-testid="settings-panel">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-2.5">
        <h2 className="text-sm font-semibold text-slate-800">{t.settings.title}</h2>
        <button type="button" onClick={onClose} className="rounded-md px-2 py-1 text-xs text-slate-500 hover:bg-slate-100">{t.common.close}</button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <Section title={t.settings.bots}>
          {ids.map((id) => (
            <NumberField key={id} label={policyName(id, models)} value={arena.bots[id] ?? 0} min={0} max={8} onChange={(v) => setBots(id, v)} testId={`bots-${id}`} />
          ))}
        </Section>
        <Section title={t.settings.title}>
          <div className="flex items-end gap-2">
            <div className="flex-1"><NumberField label={t.settings.seed} value={arena.seed} min={0} max={2 ** 31 - 1} onChange={(v) => onArena({ ...arena, seed: v })} testId="seed" /></div>
            <button type="button" onClick={() => onArena({ ...arena, seed: Math.floor(Math.random() * 1e6) })} className="h-8 rounded-md bg-slate-100 px-2 text-xs text-slate-600 hover:bg-slate-200">{t.settings.newSeed}</button>
          </div>
          <NumberField label={t.settings.food} value={arena.foodTarget} min={0} max={4000} step={100} onChange={(v) => onArena({ ...arena, foodTarget: v })} />
          <label className="flex items-center justify-between text-sm text-slate-700">
            <span>{t.settings.decision}</span>
            <select value={arena.decisionInterval} onChange={(e) => onArena({ ...arena, decisionInterval: Number(e.target.value) })} className="rounded-md border border-slate-200 bg-white px-2 py-1 text-sm">
              {[1, 2, 3, 6].map((d) => <option key={d} value={d}>{t.settings.decisionHz(30 / d)}</option>)}
            </select>
          </label>
          <Toggle label={t.settings.respawn} checked={arena.respawn} onChange={(v) => onArena({ ...arena, respawn: v })} />
          <p className="text-[11px] text-slate-400">{t.settings.hint}</p>
          <button type="button" onClick={onApply} data-testid="apply-settings" className="h-9 w-full rounded-lg bg-sky-600 text-sm font-semibold text-white hover:bg-sky-700">{t.settings.apply}</button>
        </Section>
        <Section title={t.settings.debug}>
          {(Object.keys(t.debug) as (keyof DebugFlags)[]).map((k) => (
            <Toggle key={k} label={t.debug[k]} checked={ui.debug[k]} onChange={(v) => setDebug(k, v)} testId={`debug-${k}`} />
          ))}
          <Toggle label={t.settings.labels} checked={ui.labels} onChange={(v) => onUi({ ...ui, labels: v })} />
        </Section>
      </div>
    </div>
  );
}
