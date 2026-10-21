import type { MlpModelFile } from '../policy/mlp';
import { Card, CiBars, Lines, Missing, Page, Table, fmt, fmtCI, useJson, type Interval } from './charts';
import { Badge } from './kit';
import { policyName } from './SettingsPanel';

const B = import.meta.env.BASE_URL;
const VARIANTS = [
  { run: 'ppo-random-v1', label: 'PPO-Random' },
  { run: 'ppo-bc-v1', label: 'PPO-BC' },
  { run: 'ppo-dagger-v1', label: 'PPO-DAgger' },
];

interface EvalSummary { survivalS: Interval; maxLength: Interval; deathRate: Interval; kills: Interval; return: Interval; eaten: Interval; collisionDeathRate: Interval; boostShare: number }
interface ImitationRound { round: number; beta?: number; samples: number; val: Record<string, number>; eval: Record<string, EvalSummary> }
interface Imitation { expert: string; data: { d0: { samples: number; episodes: number; categories: Record<string, number> } }; rounds: ImitationRound[]; expertEval: Record<string, EvalSummary>; bestRound: number; seconds: number }
interface PpoRun { run: string; steps: number; budget: number; seconds: number; curve: { step: number; return: number; survivalS: number; maxLength: number; deathRate: number; stage: string }[];
  gates: { stage: string; metric: string; value: number; pass: boolean; step: number }[]; supervisor: { kind: string; step?: number; action?: string; reason?: string; anomalies?: string[] }[]; anomalies: Record<string, number>; bestStep: number }
interface FinalEval { seeds: [number, number]; episodes: number; policies: { id: string; kind: string; results: Record<string, EvalSummary> }[] }

const CAT_LABEL: Record<string, string> = {
  danger_ahead: '전방 충돌 위험', near_body: '몸통 근접', near_wall: '벽 근처', encircled: '포위', sharp_turn: '급회전', boost: '부스트', food_dense: '먹이 밀집', opponent_close: '상대 근접',
};

function ImitationSection({ il }: { il: Imitation }) {
  const ex = il.expertEval['D-mixed'];
  const rows = [{ label: `전문가 (${il.expert.toUpperCase()})`, v: ex.survivalS },
    ...il.rounds.map((r) => ({ label: r.round === 0 ? 'BC' : `BC+DAgger r${r.round}`, v: r.eval['D-mixed'].survivalS, note: r.round === il.bestRound ? '★' : '' }))];
  return (
    <>
      <Card title="전문가 데이터 구성" subtitle={`${il.data.d0.samples.toLocaleString()}개 상태 · ${il.data.d0.episodes}개 에피소드 · 행동 잡음 ε=0.1 · 위험 상태는 학습 시 가중치 상향`}
        table={<Table head={['상태 유형', '비율 %']} rows={Object.entries(il.data.d0.categories).map(([k, v]) => [CAT_LABEL[k] ?? k, (v * 100).toFixed(1)])} />}>
        <CiBars rows={Object.entries(il.data.d0.categories).map(([k, v]) => ({ label: CAT_LABEL[k] ?? k, v: { mean: v * 100, lo: NaN, hi: NaN } }))} unit="%" digits={1} />
      </Card>
      <Card title="BC · DAgger: 실제 게임 성능 (생존 시간, 120초 상한)" subtitle="검증 시드 60개, D-mixed 상대. ★ = PPO-DAgger 초기화에 쓴 라운드 (검증 생존 시간 최고)"
        table={<Table head={['정책', '샘플', '검증 정확도', '회전 ±1단계 일치', '생존 s', '사망률 %', '최대 길이', '처치']}
          rows={[[`전문가 ${il.expert}`, '–', '–', '–', fmtCI(ex.survivalS), fmtCI(ex.deathRate, 0, 100), fmtCI(ex.maxLength, 0), fmtCI(ex.kills, 2)],
            ...il.rounds.map((r) => { const e = r.eval['D-mixed']; return [r.round === 0 ? 'BC' : `DAgger r${r.round}`, r.samples.toLocaleString(), fmt(r.val.valAccuracy * 100, 1) + '%', fmt(r.val.valTurnWithinOne * 100, 1) + '%', fmtCI(e.survivalS), fmtCI(e.deathRate, 0, 100), fmtCI(e.maxLength, 0), fmtCI(e.kills, 2)]; })]} />}>
        <CiBars rows={rows} unit="s" max={120} />
      </Card>
    </>
  );
}

function PpoSection({ runs }: { runs: (PpoRun | null)[] }) {
  const have = VARIANTS.map((v, i) => ({ ...v, data: runs[i] })).filter((v) => v.data);
  if (have.length === 0) return <Missing what="PPO 학습 기록" how="python -m rl.train_ppo --variant random --steps 2000000 --run ppo-random-v1" />;
  return (
    <>
      <Card title="학습 곡선 — 같은 스텝에서 비교" subtitle="10만 스텝마다 검증 시드 24개로 측정한 평균 리턴 (reward-v1). 세 변형은 스텝 예산·시드·하이퍼파라미터·관측 정규화가 같고 초기 정책만 다릅니다."
        table={<Table head={['변형', '스텝', '리턴', '생존 s', '최대 길이', '사망률', '단계']} rows={have.flatMap((v) => v.data!.curve.map((c) => [v.label, c.step.toLocaleString(), fmt(c.return, 2), fmt(c.survivalS), fmt(c.maxLength, 0), fmt(c.deathRate, 2), c.stage]))} />}>
        <Lines series={have.map((v) => ({ label: v.label, points: v.data!.curve.map((c) => [c.step, c.return] as [number, number]) }))} xLabel="환경 스텝" yLabel="검증 평균 리턴" />
      </Card>
      <Card title="학습 곡선 — 생존 시간" table={<Table head={['변형', '스텝', '생존 s']} rows={have.flatMap((v) => v.data!.curve.map((c) => [v.label, c.step.toLocaleString(), fmt(c.survivalS)]))} />}>
        <Lines series={have.map((v) => ({ label: v.label, points: v.data!.curve.map((c) => [c.step, c.survivalS] as [number, number]) }))} xLabel="환경 스텝" yLabel="검증 평균 생존 시간 (s)" />
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="커리큘럼 단계 관문" subtitle="각 단계를 마칠 때 그 단계 시나리오에서 기준을 확인합니다. 실패하면 한 번 연장(마지막 단계에서 차감 — 총 예산 동일).">
          <Table head={['변형', '단계', '지표', '값', '결과']} rows={have.flatMap((v) => v.data!.gates.map((g) => [v.label, g.stage, g.metric, fmt(g.value, 2), g.pass ? '통과' : '실패']))} />
        </Card>
        <Card title="학습 Supervisor 기록" subtitle="검증 성능이 연속으로 나빠지면 경고 → 체크포인트 보존 → 학습률·KL 제한 축소 → 최고 체크포인트로 복귀. 이상 행동 탐지 횟수 포함.">
          <Table head={['변형', '스텝', '종류', '내용']} rows={have.flatMap((v) => [
            ...v.data!.supervisor.filter((e) => e.kind === 'intervention').map((e) => [v.label, (e.step ?? 0).toLocaleString(), e.action ?? '', e.reason ?? '']),
            ...Object.entries(v.data!.anomalies).map(([k, n]) => [v.label, '–', '이상 징후', `${k} × ${n}`]),
          ])} />
        </Card>
      </div>
    </>
  );
}

function FinalSection({ fe, models }: { fe: FinalEval; models: MlpModelFile[] }) {
  const scenarios = Object.keys(fe.policies[0]?.results ?? {});
  return (
    <>
      {scenarios.map((sc) => (
        <Card key={sc} title={`최종 평가 — ${sc === 'D-mixed' ? '학습 상대 구성 (D-mixed)' : `학습하지 않은 상대: ${sc.replace('heldout:', '')}`}`}
          subtitle={`새 시드 ${fe.seeds[0]}–${fe.seeds[1]} (선택·검증에 쓰지 않음), 에피소드 ${fe.episodes}개, 120초 상한. 막대 = 생존 시간 평균과 95% CI.`}
          table={<Table head={['정책', '종류', '생존 s', '사망률 %', '충돌 사망 %', '최대 길이', '먹은 양', '처치', '리턴', '부스트 비율']}
            rows={fe.policies.map((p) => { const r = p.results[sc]; return [policyName(p.id, models), p.kind, fmtCI(r.survivalS), fmtCI(r.deathRate, 0, 100), fmtCI(r.collisionDeathRate, 0, 100), fmtCI(r.maxLength, 0), fmtCI(r.eaten, 0), fmtCI(r.kills, 2), fmtCI(r.return, 2), fmt(r.boostShare * 100, 1) + '%']; })} />}>
          <CiBars rows={fe.policies.map((p) => ({ label: policyName(p.id, models), v: p.results[sc].survivalS }))} unit="s" max={120} />
        </Card>
      ))}
    </>
  );
}

function ModelCards({ models }: { models: MlpModelFile[] }) {
  if (!models.length) return <Missing what="배포된 학습 모델" how="python -m rl.train_ppo … / public/models/index.json" />;
  return (
    <Card title="브라우저에 배포된 학습 모델" subtitle="게임의 AI 관전·토너먼트에서 이 파일들을 실제로 불러와 추론합니다 (휴리스틱으로 대체하지 않음).">
      <Table head={['모델', '알고리즘', '계약', '구조', '메타데이터']} rows={models.map((m) => [
        m.label, m.algo.toUpperCase(), `${m.obsVersion} · ${m.actionVersion} · ${m.rulesVersion}`, m.layers.map((l) => l.b.length).join('→'),
        Object.entries(m.meta).filter(([k]) => ['steps', 'samples', 'round', 'bestStep', 'init', 'exportedAt'].includes(k)).map(([k, v]) => `${k}=${String(v)}`).join(' · '),
      ])} />
    </Card>
  );
}

export function LabView({ models }: { models: MlpModelFile[] }) {
  const il = useJson<Imitation>(`${B}data/rl/il-v1.json`);
  const r0 = useJson<PpoRun>(`${B}data/rl/${VARIANTS[0].run}.json`);
  const r1 = useJson<PpoRun>(`${B}data/rl/${VARIANTS[1].run}.json`);
  const r2 = useJson<PpoRun>(`${B}data/rl/${VARIANTS[2].run}.json`);
  const fe = useJson<FinalEval>(`${B}data/rl/final-eval.json`);
  return (
    <Page title="강화학습 실험실" intro={<>MPC 전문가의 시연으로 Behavior Cloning을 학습하고, DAgger로 정책이 실제로 방문한 상태에 전문가 레이블을 추가합니다. 그 결과를 PPO의 초기값으로 쓴 경우와 무작위 초기화 PPO를 <b>같은 스텝 예산</b>으로 비교합니다. 모든 수치는 실제 실행 결과이며, 학습 실행은 로컬에서 수행합니다 (<span className="font-mono text-xs">python -m rl.imitate / rl.train_ppo</span>).</>}>
      <div className="flex flex-wrap gap-2"><Badge tone="violet">obs-v1 · act-v1 · reward-v1 · rules-v1</Badge><Badge>MaskablePPO (sb3-contrib)</Badge><Badge>MLP 84→256→256→18 tanh</Badge></div>
      <h3 className="pt-2 text-sm font-semibold text-slate-700">모방학습 (BC · DAgger)</h3>
      {il.data ? <ImitationSection il={il.data} /> : <Missing what="모방학습 결과" how="python -m rl.imitate all --run il-v1" />}
      <h3 className="pt-2 text-sm font-semibold text-slate-700">PPO</h3>
      <PpoSection runs={[r0.data, r1.data, r2.data]} />
      <h3 className="pt-2 text-sm font-semibold text-slate-700">최종 평가와 일반화</h3>
      {fe.data ? <FinalSection fe={fe.data} models={models} /> : <Missing what="최종 평가" how="python -m rl.evaluate --out public/data/rl/final-eval.json" />}
      <ModelCards models={models} />
    </Page>
  );
}
