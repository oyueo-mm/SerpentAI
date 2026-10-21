import { Suspense, lazy, useEffect, useState } from 'react';
import { t } from '../i18n';
import type { MlpModelFile } from '../policy/mlp';
import { RULES_VERSION } from '../sim/config';
import { Arena } from './Arena';
import { Badge, Logo } from './kit';
import { loadModels } from './models';
import { MODES, type Mode } from './settings';

// Analysis pages load on demand so the game starts with a smaller bundle.
const TournamentView = lazy(() => import('./TournamentView').then((m) => ({ default: m.TournamentView })));
const LabView = lazy(() => import('./LabView').then((m) => ({ default: m.LabView })));
const VisionView = lazy(() => import('./VisionView').then((m) => ({ default: m.VisionView })));
const CalibrationView = lazy(() => import('./CalibrationView').then((m) => ({ default: m.CalibrationView })));

function modeFromHash(): Mode {
  const h = window.location.hash.replace('#', '');
  return (MODES.includes(h as Mode) ? h : 'play') as Mode;
}

export function App() {
  const [mode, setMode] = useState<Mode>(modeFromHash);
  const [models, setModels] = useState<MlpModelFile[] | null>(null);
  const [modelErrors, setModelErrors] = useState<string[]>([]);
  const [visited, setVisited] = useState<Set<Mode>>(() => new Set([modeFromHash()]));

  useEffect(() => {
    void loadModels(import.meta.env.BASE_URL).then(({ files, errors }) => { setModels(files); setModelErrors(errors); });
    const onHash = () => setMode(modeFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  useEffect(() => { setVisited((v) => (v.has(mode) ? v : new Set([...v, mode]))); }, [mode]);

  const go = (m: Mode) => { window.location.hash = m; setMode(m); };

  return (
    <div className="flex h-full flex-col bg-slate-50 text-slate-800">
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-slate-200 bg-white px-3">
        <a href="#play" onClick={() => go('play')} className="flex items-center gap-2" aria-label="SerpentAI">
          <Logo size={26} />
          <span className="hidden text-sm font-bold tracking-tight sm:inline">SerpentAI</span>
        </a>
        <nav className="flex min-w-0 flex-1 gap-0.5 overflow-x-auto" aria-label="모드">
          {MODES.map((m) => (
            <button
              key={m} type="button" onClick={() => go(m)} data-testid={`mode-${m}`} aria-current={mode === m ? 'page' : undefined}
              className={`h-8 shrink-0 whitespace-nowrap rounded-md px-2.5 text-sm font-medium transition-colors ${mode === m ? 'bg-sky-50 text-sky-700' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-800'}`}
            >
              {t.modes[m]}
            </button>
          ))}
        </nav>
        <span className="hidden md:inline"><Badge>{RULES_VERSION}</Badge></span>
        <a href="https://github.com/oyueo-mm/SerpentAI" className="hidden text-xs text-slate-400 hover:text-slate-700 md:inline" target="_blank" rel="noreferrer">GitHub</a>
      </header>
      {modelErrors.length > 0 && <div className="bg-amber-50 px-3 py-1 text-xs text-amber-700" role="alert" title={modelErrors.join('\n')}>{t.app.modelErrors(modelErrors.length)}</div>}
      <main className="relative min-h-0 flex-1">
        {models === null ? (
          <div className="flex h-full items-center justify-center text-sm text-slate-400">{t.common.loading}</div>
        ) : (
          <>
            {(['play', 'watch'] as const).map((m) => visited.has(m) && (
              <div key={m} className={mode === m ? 'h-full' : 'hidden'} data-testid={`page-${m}`}>
                <Arena mode={m} models={models} active={mode === m} />
              </div>
            ))}
            <Suspense fallback={<div className="p-6 text-sm text-slate-400">{t.common.loading}</div>}>
              {mode === 'tournament' && <TournamentView models={models} />}
              {mode === 'lab' && <LabView models={models} />}
              {mode === 'vision' && <VisionView />}
              {mode === 'calibration' && <CalibrationView />}
            </Suspense>
          </>
        )}
      </main>
    </div>
  );
}
