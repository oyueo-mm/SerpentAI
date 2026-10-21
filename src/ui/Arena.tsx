import { useCallback, useEffect, useRef, useState } from 'react';
import { t } from '../i18n';
import type { MlpModelFile } from '../policy/mlp';
import { RULES_VERSION } from '../sim/config';
import type { MatchJSON } from '../sim/match';
import { AnalysisPanel } from './AnalysisPanel';
import { CameraController, type CameraMode } from '../render/camera';
import type { Camera } from '../render/renderer';
import { GameCanvas } from './GameCanvas';
import { Icon, IconButton, downloadText, formatTime } from './kit';
import { SettingsPanel } from './SettingsPanel';
import { DEFAULT_UI, defaultArena, toMatchConfig, type ArenaSettings, type UiSettings } from './settings';
import { SpeedControl } from './SpeedControl';
import { TouchControls } from './TouchControls';
import { useSimulation } from './useSimulation';

interface ExperimentFile {
  format: 'serpent-experiment';
  version: 1;
  rulesVersion: string;
  savedAt: string;
  mode: 'play' | 'watch';
  arena: ArenaSettings;
  snapshot: MatchJSON;
}

interface Props {
  mode: 'play' | 'watch';
  models: MlpModelFile[];
  active: boolean;
}

/** Play / Watch page. Each arena owns its own simulation worker, so switching pages keeps both matches. */
/** Below the lg breakpoint the drawers overlay the game, so only one may be open at a time. */
const narrow = () => typeof window !== 'undefined' && window.innerWidth < 1024;

export function Arena({ mode, models, active }: Props) {
  const sim = useSimulation();
  // View-only camera: lives outside the simulation and survives selection, speed changes and restarts.
  const [camera] = useState(() => new CameraController());
  const [camMode, setCamMode] = useState<CameraMode>('follow');
  useEffect(() => camera.onModeChange(setCamMode), [camera]);
  const view = useRef<{ cam: Camera; w: number; h: number } | null>(null);
  const learnedIds = models.map((m) => m.id);
  const [arena, setArena] = useState<ArenaSettings>(() => defaultArena(mode, learnedIds));
  const [applied, setApplied] = useState<ArenaSettings>(arena);
  const [ui, setUi] = useState<UiSettings>(() => ({ ...DEFAULT_UI, rightOpen: typeof window !== 'undefined' && window.innerWidth >= 1100 }));
  const [running, setRunning] = useState(true);
  const [inspectId, setInspectId] = useState(0);
  const [fps, setFps] = useState(0);
  const [notice, setNotice] = useState<string | null>(null);
  const [help, setHelp] = useState(mode === 'play');
  const [moreOpen, setMoreOpen] = useState(false);
  const touch = useRef<{ heading: number | null; boost: boolean }>({ heading: null, boost: false });
  const fileInput = useRef<HTMLInputElement>(null);
  const started = useRef(false);

  const focusFor = (a: ArenaSettings) => (mode === 'play' ? 0 : Math.max(0, toMatchConfig(mode, a).agents.findIndex((x) => x.policy === 'mpc')));

  const start = useCallback((a: ArenaSettings) => {
    const focus = focusFor(a);
    sim.api.init(toMatchConfig(mode, a), models, focus, true);
    sim.api.setSpeed(ui.speed);
    setInspectId(focus);
    setRunning(true);
  }, [sim.api, models, ui.speed]); // eslint-disable-line react-hooks/exhaustive-deps

  // Start when the page becomes active; pause when leaving.
  useEffect(() => {
    if (!active) { sim.api.setRunning(false); return; }
    if (!started.current) { started.current = true; start(applied); }
    else sim.api.setRunning(running);
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { if (active) sim.api.debug(ui.rightOpen || Object.values(ui.debug).some(Boolean)); }, [active, ui.rightOpen, ui.debug, sim.api]);
  useEffect(() => { if (active) sim.api.focus(inspectId); }, [active, inspectId, sim.api]);
  useEffect(() => { if (!notice) return; const id = setTimeout(() => setNotice(null), 4000); return () => clearTimeout(id); }, [notice]);
  useEffect(() => { if (sim.ui.ended) setRunning(false); }, [sim.ui.ended]);

  const toggleRun = () => { const r = !running; setRunning(r); sim.api.setRunning(r); };
  const step = () => { setRunning(false); sim.api.step(); };
  const restart = (seedBump = 0) => {
    const next = { ...arena, seed: arena.seed + seedBump };
    setArena(next);
    setApplied(next);
    start(next);
  };
  const setSpeed = (s: number) => { setUi((u) => ({ ...u, speed: s })); sim.api.setSpeed(s); };

  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest?.('input, select, textarea')) return;
      if (e.code === 'KeyP') toggleRun();
      else if (e.key === '.') step();
      else if (e.code === 'Enter' && mode === 'play' && humanDead) restart(1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const save = async () => {
    const snapshot = await sim.api.save();
    const file: ExperimentFile = { format: 'serpent-experiment', version: 1, rulesVersion: RULES_VERSION, savedAt: new Date().toISOString(), mode, arena: applied, snapshot };
    downloadText(`serpent-${mode}-seed${applied.seed}-t${Math.round(sim.ui.time)}.json`, JSON.stringify(file));
    setNotice(t.app.saved);
  };
  const load = async (f: File) => {
    try {
      const exp = JSON.parse(await f.text()) as ExperimentFile;
      if (exp.format !== 'serpent-experiment') throw new Error('serpent-experiment 파일이 아닙니다');
      if (exp.rulesVersion !== RULES_VERSION) throw new Error(`${exp.rulesVersion} 규칙의 파일입니다 (현재 ${RULES_VERSION})`);
      setArena(exp.arena);
      setApplied(exp.arena);
      sim.api.load(exp.snapshot, models, mode === 'play' ? 0 : inspectId);
      sim.api.setRunning(false);
      setRunning(false);
      setNotice(t.app.loadedState);
    } catch (err) {
      setNotice(t.app.loadFailed(err instanceof Error ? err.message : String(err)));
    }
  };

  const me = sim.ui.leaders.find((r) => r.id === 0);
  const humanDead = mode === 'play' && me !== undefined && !me.alive;
  const rank = sim.ui.leaders.filter((r) => r.alive).findIndex((r) => r.id === 0) + 1;
  const coarse = typeof window !== 'undefined' && window.matchMedia?.('(pointer: coarse)').matches;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="relative flex min-h-0 flex-1">
        {ui.settingsOpen && (
          <aside className="absolute inset-y-0 left-0 z-30 w-[min(20rem,90vw)] border-r border-slate-200 bg-white shadow-xl lg:static lg:shadow-none">
            <SettingsPanel arena={arena} onArena={setArena} ui={ui} onUi={setUi} models={models} onApply={() => { restart(0); }} onClose={() => setUi({ ...ui, settingsOpen: false })} />
          </aside>
        )}
        <div className="relative min-w-0 flex-1">
          <GameCanvas
            frame={sim.frame} prevFrame={sim.prevFrame} frameAt={sim.frameAt} tickInterval={sim.tickInterval} debug={ui.debug} labels={ui.labels} hud smooth
            camera={camera} inspectId={inspectId} onControl={mode === 'play' ? sim.api.human : undefined} touch={mode === 'play' ? touch : undefined}
            onSelect={mode === 'watch' ? setInspectId : undefined} onFps={setFps} onView={(cam, w, h) => { view.current = { cam, w, h }; }}
          />
          {camMode !== 'follow' && (
            <div className="pointer-events-none absolute bottom-3 left-1/2 -translate-x-1/2 rounded-lg bg-slate-900/75 px-3 py-1 text-[11px] text-white" data-testid="camera-hint">
              {camMode === 'free' ? t.camera.freeHint : t.camera.overviewHint}
            </div>
          )}
          <div className="pointer-events-none absolute left-3 top-3 flex gap-2" data-testid="hud">
            {mode === 'play' && me && (
              <div className="rounded-lg bg-white/85 px-3 py-1.5 text-xs shadow-sm backdrop-blur">
                <span className="text-slate-400">{t.hud.length}</span> <span className="font-semibold tabular-nums text-slate-800" data-testid="my-length">{me.alive ? me.length.toFixed(0) : 0}</span>
                {me.alive && rank > 0 && <><span className="ml-2 text-slate-400">{t.hud.rank}</span> <span className="font-semibold tabular-nums text-slate-800">{rank}</span></>}
                <span className="ml-2 text-slate-400">{t.hud.kills}</span> <span className="font-semibold tabular-nums text-slate-800">{me.kills}</span>
              </div>
            )}
            <div className="rounded-lg bg-white/85 px-3 py-1.5 text-xs tabular-nums text-slate-600 shadow-sm backdrop-blur" data-testid="sim-time">{formatTime(sim.ui.time)}</div>
          </div>
          {mode === 'play' && help && !coarse && (
            <div className="absolute right-3 top-3 max-w-xs rounded-lg bg-white/90 p-3 text-xs text-slate-600 shadow-sm backdrop-blur" data-testid="controls-help">
              <div className="mb-1 flex items-center justify-between"><span className="font-semibold text-slate-800">{t.controls.title}</span><button type="button" onClick={() => setHelp(false)} className="text-slate-400 hover:text-slate-700" aria-label={t.common.close}>{Icon.close}</button></div>
              <p>{t.controls.mouse}</p><p>{t.controls.keys}</p><p>{t.controls.pause}</p><p>{t.controls.camera}</p>
            </div>
          )}
          {humanDead && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/40 backdrop-blur-[1px]" data-testid="death-overlay">
              <div className="rounded-2xl bg-white p-5 text-center shadow-xl">
                <div className="text-base font-semibold text-slate-800">{t.hud.dead}</div>
                <button type="button" onClick={() => restart(1)} data-testid="respawn" className="mt-3 h-10 rounded-lg bg-sky-600 px-5 text-sm font-semibold text-white hover:bg-sky-700">{t.hud.respawn}</button>
              </div>
            </div>
          )}
          {sim.ui.ended && mode === 'watch' && <div className="absolute inset-x-0 top-14 mx-auto w-fit rounded-lg bg-white px-3 py-1.5 text-sm shadow">{t.hud.ended}</div>}
          {mode === 'play' && coarse && !humanDead && <TouchControls state={touch} />}
          {notice && <div className="absolute bottom-3 left-1/2 z-20 -translate-x-1/2 rounded-lg bg-slate-900/85 px-3 py-1.5 text-xs text-white" role="status">{notice}</div>}
          {sim.ui.error && <div className="absolute inset-x-3 top-14 z-20 rounded-lg bg-rose-50 px-3 py-2 text-xs text-rose-700" role="alert">{t.app.workerError(sim.ui.error)}</div>}
        </div>
        {ui.rightOpen && (
          <aside className="absolute inset-y-0 right-0 z-30 w-[min(20rem,90vw)] border-l border-slate-200 bg-white shadow-xl lg:static lg:shadow-none">
            <AnalysisPanel frame={sim.frame} inspectId={inspectId} onInspect={setInspectId} leaders={sim.ui.leaders} records={sim.ui.records} stats={sim.ui.stats} fps={fps} time={sim.ui.time} models={models} view={view}
              perception={ui.debug.perception} onPerception={(v) => setUi((u) => ({ ...u, debug: { ...u.debug, perception: v } }))} />
          </aside>
        )}
      </div>
      <footer className="flex h-12 shrink-0 items-center gap-0.5 border-t border-slate-200 bg-white px-2" data-testid="dock">
        <IconButton label={running ? t.dock.pause : t.dock.play} onClick={toggleRun} testId="play-pause">{running ? Icon.pause : Icon.play}</IconButton>
        <span className="hidden sm:contents"><IconButton label={t.dock.step} onClick={step} testId="step">{Icon.step}</IconButton></span>
        <IconButton label={t.dock.reset} onClick={() => restart(1)} testId="reset">{Icon.reset}</IconButton>
        <div className="mx-1 hidden h-6 w-px bg-slate-200 sm:block" />
        <SpeedControl value={ui.speed} actual={sim.ui.stats.actualSpeed} running={running} onChange={setSpeed} />
        <div className="flex-1" />
        {/* Phones: one button cycling follow → free → overview (the dock must fit 360 px). */}
        <button type="button" onClick={() => camera.setMode(camMode === 'follow' ? 'free' : camMode === 'free' ? 'overview' : 'follow')}
          className="flex h-9 items-center gap-0.5 rounded-lg px-1.5 text-xs font-medium text-sky-700 hover:bg-sky-50 sm:hidden" data-testid="camera-cycle" title={t.camera.tips[camMode]}
          aria-label={`${t.camera.label}: ${t.camera.modes[camMode]}`}>
          {Icon.globe}{t.camera.modes[camMode]}
        </button>
        <div className="mr-1 hidden rounded-lg bg-slate-100 p-0.5 sm:flex" role="radiogroup" aria-label={t.camera.label} data-testid="camera-mode">
          {(['follow', 'free', 'overview'] as const).map((m) => (
            <button key={m} type="button" role="radio" aria-checked={camMode === m} data-testid={`camera-${m}`} title={t.camera.tips[m]}
              onClick={() => camera.setMode(m)}
              className={`h-7 rounded-md px-2 text-xs font-medium ${camMode === m ? 'bg-white text-sky-700 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}>
              {t.camera.modes[m]}
            </button>
          ))}
        </div>
        <span className="hidden sm:contents">
          <IconButton label={t.dock.save} onClick={() => void save()} testId="save">{Icon.save}</IconButton>
          <IconButton label={t.dock.load} onClick={() => fileInput.current?.click()} testId="load">{Icon.load}</IconButton>
        </span>
        {/* Phones: step / save / load live in a small menu so the dock fits 360 px. */}
        <div className="relative sm:hidden">
          <IconButton label={t.dock.more} onClick={() => setMoreOpen((o) => !o)} active={moreOpen} testId="dock-more">⋯</IconButton>
          {moreOpen && (
            <div className="absolute bottom-11 right-0 z-40 flex gap-1 rounded-xl border border-slate-200 bg-white p-1 shadow-lg" role="menu" data-testid="dock-more-menu">
              <IconButton label={t.dock.step} onClick={() => { setMoreOpen(false); step(); }} testId="step-m">{Icon.step}</IconButton>
              <IconButton label={t.dock.save} onClick={() => { setMoreOpen(false); void save(); }} testId="save-m">{Icon.save}</IconButton>
              <IconButton label={t.dock.load} onClick={() => { setMoreOpen(false); fileInput.current?.click(); }} testId="load-m">{Icon.load}</IconButton>
            </div>
          )}
        </div>
        <input ref={fileInput} type="file" accept="application/json" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) void load(f); e.target.value = ''; }} />
        <IconButton label={t.dock.settings} onClick={() => setUi({ ...ui, settingsOpen: !ui.settingsOpen, rightOpen: narrow() && !ui.settingsOpen ? false : ui.rightOpen })} active={ui.settingsOpen} testId="toggle-settings">{Icon.settings}</IconButton>
        <IconButton label={t.dock.analysis} onClick={() => setUi({ ...ui, rightOpen: !ui.rightOpen, settingsOpen: narrow() && !ui.rightOpen ? false : ui.settingsOpen })} active={ui.rightOpen} testId="toggle-analysis">{Icon.chart}</IconButton>
      </footer>
    </div>
  );
}
