import { useEffect, useRef, useState } from 'react';
import { BUILTIN_POLICIES } from '../ai/registry';
import type { MlpModelFile } from '../policy/mlp';
import type { PolicySummary, TournamentResult } from '../sim/tournament';
import { Card, CiBars, Missing, Page, Table, fmtCI, useJson } from './charts';
import { policyKind } from './AnalysisPanel';
import { Badge } from './kit';
import { policyName } from './SettingsPanel';

type Metric = 'winRate' | 'rank' | 'survivalS' | 'maxLength' | 'kills' | 'deathRate';
const METRICS: { key: Metric; label: string; unit: string; scale: number; digits: number }[] = [
  { key: 'winRate', label: '승률', unit: '%', scale: 100, digits: 0 },
  { key: 'rank', label: '평균 최종 순위 (낮을수록 좋음)', unit: '', scale: 1, digits: 1 },
  { key: 'survivalS', label: '생존 시간', unit: 's', scale: 1, digits: 1 },
  { key: 'maxLength', label: '최대 길이', unit: '', scale: 1, digits: 0 },
  { key: 'kills', label: '제거한 상대 수', unit: '', scale: 1, digits: 2 },
  { key: 'deathRate', label: '사망률', unit: '%', scale: 100, digits: 0 },
];

function SummaryView({ summary, models }: { summary: PolicySummary[]; models: MlpModelFile[] }) {
  const [metric, setMetric] = useState<Metric>('winRate');
  const m = METRICS.find((x) => x.key === metric)!;
  const sorted = [...summary].sort((a, b) => a.rank.mean - b.rank.mean);
  const rows = sorted.map((s) => ({ label: policyName(s.policy, models), v: { mean: s[metric].mean * m.scale, lo: s[metric].lo * m.scale, hi: s[metric].hi * m.scale, n: s[metric].n } }));
  return (
    <Card title="정책별 성능 (95% 신뢰구간)" subtitle="막대 = 평균, 수염 = 95% 신뢰구간 (비율은 Wilson, 평균은 t-분포)" testId="tournament-summary"
      table={<Table head={['정책', '종류', '승률 %', '평균 순위', '생존 s', '최대 길이', '처치', '사망률 %', '벽 사망 %']}
        rows={sorted.map((s) => [policyName(s.policy, models), policyKind(s.policy, models).label, fmtCI(s.winRate, 0, 100), fmtCI(s.rank), fmtCI(s.survivalS), fmtCI(s.maxLength, 0), fmtCI(s.kills, 2), fmtCI(s.deathRate, 0, 100), fmtCI(s.wallDeathRate, 0, 100)])} />}>
      <div className="mb-3 flex flex-wrap gap-1" role="tablist">
        {METRICS.map((x) => (
          <button key={x.key} type="button" role="tab" aria-selected={metric === x.key} onClick={() => setMetric(x.key)}
            className={`h-7 rounded-md px-2 text-xs font-medium ${metric === x.key ? 'bg-sky-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}>{x.label}</button>
        ))}
      </div>
      <CiBars rows={rows} unit={m.unit} digits={m.digits} max={metric === 'winRate' || metric === 'deathRate' ? 100 : undefined} />
    </Card>
  );
}

function LiveTournament({ models }: { models: MlpModelFile[] }) {
  const [rounds, setRounds] = useState(4);
  const [state, setState] = useState<{ done: number; total: number; summary: PolicySummary[]; wallMs: number } | null>(null);
  const [running, setRunning] = useState(false);
  const worker = useRef<Worker | null>(null);
  useEffect(() => () => worker.current?.terminate(), []);
  const policies = [...BUILTIN_POLICIES.filter((p) => p !== 'mpc-lite'), ...models.map((m) => m.id)];
  const start = () => {
    worker.current?.terminate();
    const w = new Worker(new URL('../worker/tournament.worker.ts', import.meta.url), { type: 'module' });
    worker.current = w;
    setRunning(true);
    setState(null);
    w.onmessage = (e) => { if (e.data.type === 'progress') setState(e.data); else setRunning(false); };
    w.postMessage({ config: { policies, rounds, copies: 1, durationS: 60, seedBase: 950_000 + Math.floor(Math.random() * 10_000) }, models });
  };
  return (
    <Card title="브라우저에서 직접 실행" subtitle={`${policies.length}개 정책 × 1마리, 라운드당 60초, 무작위 시드 — 작은 표본이라 신뢰구간이 넓습니다.`}>
      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-2 text-sm text-slate-700">라운드
          <input type="number" min={1} max={30} value={rounds} onChange={(e) => setRounds(Math.max(1, Math.min(30, Number(e.target.value) || 1)))} className="w-16 rounded-md border border-slate-200 px-2 py-1 text-right text-sm" />
        </label>
        <button type="button" onClick={start} disabled={running} data-testid="run-tournament" className="h-8 rounded-lg bg-sky-600 px-3 text-sm font-semibold text-white disabled:opacity-50">{running ? '실행 중…' : '실행'}</button>
        {state && <span className="text-xs text-slate-500" data-testid="tournament-progress">{state.done}/{state.total} 라운드 · {(state.wallMs / 1000).toFixed(1)}초</span>}
      </div>
      {state && <div className="mt-3"><SummaryView summary={state.summary} models={models} /></div>}
    </Card>
  );
}

export function TournamentView({ models }: { models: MlpModelFile[] }) {
  const { data, error } = useJson<TournamentResult & { generatedAt: string; node: string }>(`${import.meta.env.BASE_URL}data/tournament.json`);
  return (
    <Page title="AI 토너먼트" intro={<>모든 정책이 같은 엔진·같은 인터페이스로 한 경기장에서 경쟁합니다. 좌석(출발 위치)은 라운드마다 회전하고, 시드는 학습에 쓰지 않은 범위(900000번대)입니다. 학습 정책은 내보낸 모델 파일에서 실제로 불러와 실행됩니다.</>}>
      {data ? (
        <>
          <div className="flex flex-wrap gap-2 text-xs text-slate-500">
            <Badge>{data.config.rounds} 라운드</Badge><Badge>정책당 {data.config.copies}마리</Badge><Badge>라운드 {data.config.durationS}초</Badge>
            <Badge>시드 {data.config.seedBase}–{data.config.seedBase + data.config.rounds - 1}</Badge><Badge>{(data.wallMs / 1000).toFixed(0)}초 소요 · Node {data.node}</Badge>
          </div>
          <SummaryView summary={data.summary} models={models} />
          <p className="text-xs text-slate-500">한 경기장에 모든 정책이 함께 있으므로 결과는 이 상대 구성에 대한 성능입니다. 단일 상대에 대한 우위가 일반적인 강함을 뜻하지는 않습니다 — 학습하지 않은 상대 구성에 대한 결과는 강화학습 실험실의 일반화 평가를 보세요.</p>
        </>
      ) : <Missing what="사전 계산된 토너먼트" how={error ? `npm run tournament -- --out public/data/tournament.json (${error})` : '불러오는 중…'} />}
      <LiveTournament models={models} />
    </Page>
  );
}
