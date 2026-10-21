import { Card, CiBars, Lines, Missing, Page, StackedBars, Table, fmt, fmtCI, useJson, type Interval } from './charts';
import { Badge } from './kit';

const B = import.meta.env.BASE_URL;
const COND: Record<string, string> = {
  A: 'A 정확한 상태', B: 'B 상태 + 지연', C: 'C 상태 + 관측 잡음', D: 'D 화면 인식 (록스텝)', E: 'E 화면 인식 (실시간)', F: 'F 화면 인식 + 시각 변화',
};

interface Cond { episodes: number; survivalS: Interval; deathRate: number; maxLength: Interval; eaten: Interval; boostShare: Interval; pairedVsA?: { survivalS: Interval; maxLength: Interval } }
interface Measured {
  headingErrDeg: number; zoomRelErr: number; lengthRelErr: number; oppPosErrU: number; obsMAE: Record<string, number>; actionAgreement: number;
  latencyMs: Record<string, number>; timingMs: Record<string, number>; displayLagMs: number; controlHz: number;
}
interface S2V {
  model: string; seeds: number[]; conditions: Record<string, Cond>; measured: Record<string, Measured>; derived: { delayTicks: number; delayMs: number; obsNoiseSigma: number };
  latencySweep?: Record<string, { survivalS: Interval }>; noiseSweep?: Record<string, { survivalS: Interval }>; seconds: number;
}
interface CalibrationFile { tags: Record<string, { mean: number; n: number }>; perFeature: number[]; names: string[] }

export function CalibrationView() {
  const s = useJson<S2V>(`${B}data/vision/sim2visual.json`);
  const cal = useJson<CalibrationFile>(`${B}data/vision/calibration.json`);
  const d = s.data;
  const order = ['A', 'B', 'C', 'D', 'E', 'F'].filter((k) => d?.conditions[k]);
  const vis = ['D', 'E', 'F'].filter((k) => d?.measured[k]);
  return (
    <Page title="Calibration · Sim-to-Visual" intro={<>같은 학습 정책을 같은 시드·같은 상대로 여섯 조건에서 평가해, 성능 저하가 <b>인식 오차</b> 때문인지 <b>제어 지연</b> 때문인지 분리합니다. B의 지연과 C의 잡음 크기는 E에서 실제로 측정한 값으로 정합니다. 화면 인식이 개입하면 경기 진행이 갈라지므로 시점별 일대일 대응은 가정하지 않고, 시드별 결과만 짝지어 비교합니다.</>}>
      {d ? (
        <>
          <div className="flex flex-wrap gap-2"><Badge tone="violet">정책 {d.model}</Badge><Badge>시드 {d.seeds[0]}–{d.seeds[d.seeds.length - 1]}</Badge><Badge>측정 지연 {fmt(d.derived.delayMs, 0)} ms ≈ {d.derived.delayTicks}틱</Badge><Badge>측정 관측 잡음 σ ≈ {fmt(d.derived.obsNoiseSigma, 3)}</Badge></div>
          <Card title="조건별 생존 시간 (120초 상한, 95% CI)" testId="s2v-survival"
            table={<Table head={['조건', '에피소드', '생존 s', 'A 대비 (짝지은 차이)', '사망률 %', '최대 길이', '먹은 양', '부스트 비율 %']}
              rows={order.map((k) => { const c = d.conditions[k]; return [COND[k], c.episodes, fmtCI(c.survivalS), c.pairedVsA ? fmtCI(c.pairedVsA.survivalS) : '–', fmt(c.deathRate * 100, 0), fmtCI(c.maxLength, 0), fmtCI(c.eaten, 0), fmtCI(c.boostShare, 1, 100)]; })} />}>
            <CiBars rows={order.map((k) => ({ label: COND[k], v: d.conditions[k].survivalS }))} unit="s" max={120} />
          </Card>
          <Card title="최대 길이" table={<Table head={['조건', '최대 길이']} rows={order.map((k) => [COND[k], fmtCI(d.conditions[k].maxLength, 0)])} />}>
            <CiBars rows={order.map((k) => ({ label: COND[k], v: d.conditions[k].maxLength }))} digits={0} />
          </Card>
          {vis.length > 0 && (
            <>
              <Card title="제어 루프 지연 구성 (평균, ms)" subtitle="화면에 그려진 시점 → 캡처 → 인식 → 정책 → 입력 → 게임에 반영되기까지. 반영 지연은 페이지의 입력 기록과 틱 기록으로 측정했습니다."
                table={<Table head={['조건', '표시→캡처', '캡처', '인식', '정책', '입력', '입력→반영', '제어 주기 Hz']} rows={vis.map((k) => { const m = d.measured[k]; return [COND[k], fmt(m.displayLagMs), fmt(m.timingMs.captureMs), fmt(m.timingMs.detectMs), fmt(m.timingMs.policyMs, 2), fmt(m.timingMs.inputMs), fmt(m.latencyMs.pageToApplied), fmt(m.controlHz, 1)]; })} />}>
                <StackedBars parts={['표시→캡처', '캡처', '인식', '정책', '입력', '입력→반영']} rows={vis.map((k) => { const m = d.measured[k]; return { label: COND[k], values: [m.displayLagMs, m.timingMs.captureMs, m.timingMs.detectMs, m.timingMs.policyMs, m.timingMs.inputMs, m.latencyMs.pageToApplied] }; })} />
              </Card>
              <Card title="인식 오차 (실제 플레이 중, 사후 Ground Truth 대비)" subtitle="관측 MAE는 obs-v1 정규화 단위(0–1)."
                table={<Table head={['조건', '방향 °', '배율 %', '길이 %', '상대 머리 위치 u', '광선 몸통', '광선 벽', '먹이 합', '먹이 거리', '상대', '행동 일치율 %']}
                  rows={vis.map((k) => { const m = d.measured[k]; return [COND[k], fmt(m.headingErrDeg, 2), fmt(m.zoomRelErr * 100, 2), fmt(m.lengthRelErr * 100, 2), fmt(m.oppPosErrU, 1), fmt(m.obsMAE.rayBody, 3), fmt(m.obsMAE.rayWall, 3), fmt(m.obsMAE.foodSum, 3), fmt(m.obsMAE.foodNear, 3), fmt(m.obsMAE.opponents, 3), fmt(m.actionAgreement * 100, 1)]; })} />}>
                <CiBars rows={vis.flatMap((k) => ['rayBody', 'foodSum', 'opponents'].map((g) => ({ label: `${k} · ${g}`, v: { mean: d.measured[k].obsMAE[g], lo: NaN, hi: NaN } })))} digits={3} />
              </Card>
            </>
          )}
          {(d.latencySweep || d.noiseSweep) && (
            <div className="grid gap-4 lg:grid-cols-2">
              {d.latencySweep && (
                <Card title="지연에 따른 생존 시간 (정확한 상태)" table={<Table head={['지연 틱', '생존 s']} rows={Object.entries(d.latencySweep).map(([k, v]) => [k, fmtCI(v.survivalS)])} />}>
                  <Lines series={[{ label: '생존', points: Object.entries(d.latencySweep).map(([k, v]) => [Number(k), v.survivalS.mean] as [number, number]) }]} xLabel="지연 (틱, 1틱 = 33 ms)" yLabel="평균 생존 시간 (s)" height={180} />
                </Card>
              )}
              {d.noiseSweep && (
                <Card title="관측 잡음에 따른 생존 시간 (정확한 상태 + 가우시안 잡음)" table={<Table head={['σ', '생존 s']} rows={Object.entries(d.noiseSweep).map(([k, v]) => [k, fmtCI(v.survivalS)])} />}>
                  <Lines series={[{ label: '생존', points: Object.entries(d.noiseSweep).map(([k, v]) => [Number(k), v.survivalS.mean] as [number, number]) }]} xLabel="잡음 σ (정규화 단위)" yLabel="평균 생존 시간 (s)" height={180} />
                </Card>
              )}
            </div>
          )}
        </>
      ) : <Missing what="Sim-to-Visual 실험" how="python -m vision.sim2visual --model public/models/<model>.json --episodes 8" />}
      {cal.data && (
        <Card title="오차가 큰 상황 자동 수집" subtitle="실시간 세션의 각 제어 스텝을 상황별로 분류하고, 그 스텝의 관측 MAE 평균을 비교합니다."
          table={<Table head={['상황', '스텝 수', '관측 MAE']} rows={Object.entries(cal.data.tags).map(([k, v]) => [k, v.n, fmt(v.mean, 4)])} />}>
          <CiBars rows={Object.entries(cal.data.tags).map(([k, v]) => ({ label: k, v: { mean: v.mean, lo: NaN, hi: NaN, n: v.n } }))} digits={4} />
        </Card>
      )}
    </Page>
  );
}
