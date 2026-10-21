import { useEffect, useMemo, useState } from 'react';
import { TURN_LEVELS } from '../policy/action';
import { Card, Missing, Page, Table, fmt, useJson } from './charts';
import { Badge } from './kit';

const B = import.meta.env.BASE_URL;
const RAYS = 16;

interface StudioStep {
  step: number; t: number; frame: string | null; heading: number; zoom: number; headingSrc: string; headList: [number, number, number, number][];
  obs: number[]; probs: number[]; action: number; turn: number; boost: boolean; conf: number; length: number;
  captureMs: number; detectMs: number; policyMs: number; stepMs: number; err?: { headingDeg?: number; obsMAE?: number };
}
interface Studio { session: string; model: string; mode: string; theme: string; w: number; h: number; scale: number; summary: Record<string, unknown>; steps: StudioStep[] }

function ObsBars({ obs }: { obs: number[] }) {
  const groups: [string, number, number][] = [['자기', 0, 6], ['광선 몸통', 6, 38], ['먹이', 38, 54], ['상대', 54, 82], ['신뢰도', 82, 84]];
  return (
    <div className="space-y-1.5">
      {groups.map(([name, a, b]) => (
        <div key={name}>
          <div className="text-[10px] text-slate-500">{name}</div>
          <div className="flex h-8 items-end gap-px">
            {obs.slice(a, b).map((v, i) => (
              <div key={i} className="flex-1 rounded-t-sm bg-sky-500" style={{ height: `${Math.max(2, Math.min(1, Math.abs(v)) * 100)}%`, opacity: v < 0 ? 0.45 : 1 }} title={`${a + i}: ${v.toFixed(3)}`} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function Probs({ probs, action }: { probs: number[]; action: number }) {
  return (
    <div>
      <div className="grid grid-cols-9 items-end gap-0.5" style={{ height: 56 }} data-testid="studio-probs">
        {TURN_LEVELS.map((turn, ti) => (
          <div key={turn} className="flex h-full flex-col justify-end gap-0.5">
            <div className={`w-full rounded-sm ${action === ti * 2 + 1 ? 'bg-teal-700' : 'bg-teal-500'}`} style={{ height: `${probs[ti * 2 + 1] * 100}%` }} />
            <div className={`w-full rounded-sm ${action === ti * 2 ? 'bg-sky-800' : 'bg-sky-500'}`} style={{ height: `${probs[ti * 2] * 100}%` }} />
          </div>
        ))}
      </div>
      <div className="mt-0.5 flex justify-between text-[10px] text-slate-400"><span>좌회전</span><span>직진</span><span>우회전</span></div>
      <div className="mt-1 flex gap-3 text-[10px] text-slate-500"><span><span className="mr-1 inline-block h-2 w-2 rounded-sm bg-sky-500" />부스트 없음</span><span><span className="mr-1 inline-block h-2 w-2 rounded-sm bg-teal-500" />부스트</span></div>
    </div>
  );
}

/** Read-only replay of a recorded Vision Runner session: opening it never touches a game or a policy. */
function StudioPlayer({ s }: { s: Studio }) {
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const steps = s.steps;
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setI((k) => (k + 1 < steps.length ? k + 1 : (setPlaying(false), k))), 100);
    return () => clearInterval(id);
  }, [playing, steps.length]);
  const st = steps[i];
  const frameStep = useMemo(() => { for (let k = i; k >= 0; k--) if (steps[k].frame) return steps[k]; return null; }, [i, steps]);
  const W = s.w * s.scale, H = s.h * s.scale, cx = W / 2, cy = H / 2;
  const ray = (k: number) => {
    const a = st.heading + (k * 2 * Math.PI) / RAYS;
    const d = Math.min(st.obs[6 + 2 * k], st.obs[7 + 2 * k]) * 400 * st.zoom * s.scale;
    return [cx + Math.cos(a) * d, cy + Math.sin(a) * d];
  };
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]" data-testid="studio">
      <div>
        <div className="relative overflow-hidden rounded-lg border border-slate-200 bg-slate-100" style={{ aspectRatio: `${W} / ${H}` }}>
          {frameStep?.frame && <img src={`${B}data/vision/studio/${frameStep.frame}`} alt={`프레임 ${frameStep.step}`} className="absolute inset-0 h-full w-full" />}
          <svg viewBox={`0 0 ${W} ${H}`} className="absolute inset-0 h-full w-full">
            {Array.from({ length: RAYS }, (_, k) => { const [x, y] = ray(k); return <line key={k} x1={cx} y1={cy} x2={x} y2={y} stroke="#eb6834" strokeOpacity={0.7} strokeWidth={1} />; })}
            <line x1={cx} y1={cy} x2={cx + Math.cos(st.heading) * 40} y2={cy + Math.sin(st.heading) * 40} stroke="#0b0b0b" strokeWidth={2} />
            {(st.headList ?? []).map(([x, y, hd], k) => (
              <g key={k}>
                <circle cx={x * s.scale} cy={y * s.scale} r={9} fill="none" stroke="#2a78d6" strokeWidth={2} />
                <line x1={x * s.scale} y1={y * s.scale} x2={x * s.scale + Math.cos(hd) * 22} y2={y * s.scale + Math.sin(hd) * 22} stroke="#2a78d6" strokeWidth={2} />
              </g>
            ))}
          </svg>
        </div>
        <div className="mt-2 flex items-center gap-2">
          <button type="button" onClick={() => setPlaying((p) => !p)} className="h-8 rounded-lg bg-sky-600 px-3 text-sm font-semibold text-white" data-testid="studio-play">{playing ? '일시정지' : '재생'}</button>
          <input type="range" min={0} max={steps.length - 1} value={i} onChange={(e) => { setPlaying(false); setI(Number(e.target.value)); }} className="speed-range flex-1" aria-label="재생 위치" />
          <span className="w-20 text-right text-xs tabular-nums text-slate-500">{i + 1} / {steps.length}</span>
        </div>
        <p className="mt-1 text-[11px] text-slate-400">주황 = 정책 입력의 16개 광선(몸통·벽 중 가까운 쪽), 파랑 = 인식된 상대 머리와 방향, 검정 = 추정한 자기 진행 방향.</p>
      </div>
      <div className="space-y-3 text-xs">
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-slate-400">행동</div><div className="font-semibold text-slate-800" data-testid="studio-action">{st.turn === 0 ? '직진' : `${st.turn < 0 ? '좌' : '우'} ${Math.abs(st.turn)}`}{st.boost ? ' + 부스트' : ''}</div></div>
          <div><div className="text-slate-400">관측 신뢰도</div><div className="font-semibold text-slate-800">{fmt(st.conf, 2)} · 방향 출처 {st.headingSrc === 'eyes' ? '눈' : st.headingSrc}</div></div>
          <div><div className="text-slate-400">HUD 길이 추정</div><div className="font-semibold text-slate-800">{fmt(st.length, 0)}</div></div>
          <div><div className="text-slate-400">처리 시간</div><div className="font-semibold tabular-nums text-slate-800">캡처 {fmt(st.captureMs, 0)} · 인식 {fmt(st.detectMs, 0)} · 정책 {fmt(st.policyMs, 1)} · 전체 {fmt(st.stepMs, 0)} ms</div></div>
          {st.err && <div className="col-span-2"><div className="text-slate-400">사후 오차 (Ground Truth, 실행 중에는 사용 안 함)</div><div className="font-semibold text-slate-800">방향 {fmt(st.err.headingDeg, 1)}° · 관측 MAE {fmt(st.err.obsMAE, 3)}</div></div>}
        </div>
        <div><div className="mb-1 text-slate-500">행동 확률</div><Probs probs={st.probs} action={st.action} /></div>
        <div><div className="mb-1 text-slate-500">관측 벡터 obs-v1 (84)</div><ObsBars obs={st.obs} /></div>
      </div>
    </div>
  );
}

function LivePanel() {
  const [status, setStatus] = useState<{ state: string; reason: string; step: number; last: Record<string, unknown> } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const url = 'http://127.0.0.1:8765';
  // Polls only after the user asks to connect, so the page makes no requests to localhost by itself.
  useEffect(() => {
    if (!connected) return;
    let alive = true;
    const poll = () => fetch(`${url}/status`).then((r) => r.json()).then((d) => { if (alive) { setStatus(d); setErr(null); } }).catch(() => { if (alive) { setErr('연결 안 됨'); setConnected(false); } });
    poll();
    const id = setInterval(poll, 1000);
    return () => { alive = false; clearInterval(id); };
  }, [connected]);
  const send = (cmd: string) => fetch(`${url}/${cmd}`, { method: 'POST' }).catch(() => setErr('명령 전송 실패'));
  return (
    <Card title="로컬 Vision Runner 제어" subtitle={<>로컬에서 <span className="font-mono">python -m vision.bridge.runner --model … --wait-start</span> 를 실행하면 이 패널이 127.0.0.1:8765 에 연결됩니다. 브라우저 게임을 화면으로만 보고 조작하며, 허용 목록 밖의 주소는 조작하지 않습니다.</>}>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        {!connected && <button type="button" onClick={() => setConnected(true)} className="h-8 rounded-lg bg-slate-100 px-3 text-sm font-medium text-slate-700 hover:bg-slate-200" data-testid="runner-connect">연결</button>}
        <Badge tone={status?.state === 'running' ? 'teal' : status ? 'slate' : 'amber'}>{status ? status.state : err ?? (connected ? '확인 중' : '미연결')}</Badge>
        {status && <span className="text-xs text-slate-500">step {status.step}{status.reason ? ` · ${status.reason}` : ''}</span>}
        <div className="flex-1" />
        <button type="button" onClick={() => void send('start')} disabled={!status} className="h-8 rounded-lg bg-sky-600 px-3 text-sm font-semibold text-white disabled:opacity-40">Start</button>
        <button type="button" onClick={() => void send('stop')} disabled={!status} className="h-8 rounded-lg bg-slate-200 px-3 text-sm font-semibold text-slate-700 disabled:opacity-40">Stop</button>
        <button type="button" onClick={() => void send('estop')} disabled={!status} className="h-8 rounded-lg bg-rose-600 px-3 text-sm font-semibold text-white disabled:opacity-40">Emergency Stop</button>
      </div>
    </Card>
  );
}

interface AccuracyRow { dataset: string; segmenter: string; frames: number; metrics: Record<string, { mean: number; median: number; p90: number; n: number }>; iou: Record<string, { mean: number }>; obsMAE: Record<string, number> }

function Accuracy({ rows }: { rows: AccuracyRow[] }) {
  const m = (r: AccuracyRow, k: string, f: 'mean' | 'median' = 'median', s = 1, d = 2) => fmt((r.metrics[k]?.[f] ?? NaN) * s, d);
  return (
    <Card title="인식 정확도 — Ground Truth와 비교" subtitle="학습에 쓰지 않은 테마/시드의 합성 검증 세트. 고전 CV(색·형태 규칙)와 학습형 CNN 세그멘테이션(TinyUNet, 6클래스)을 같은 기하 단계에 연결해 비교합니다.">
      <Table head={['데이터셋', '방식', '프레임', '성공률', '배율 오차 %', '자기 방향 °', '상대 머리 재현율', '정밀도', '먹이 재현율', 'IoU 자기', 'IoU 상대', 'IoU 먹이', 'IoU 벽', '광선 MAE', '먹이 MAE', '상대 MAE', '처리 ms']}
        rows={rows.map((r) => [r.dataset, r.segmenter === 'cnn' ? 'CNN' : '고전 CV', r.frames, m(r, 'okRate', 'mean', 100, 0) + '%', m(r, 'zoomRelErr', 'median', 100, 2), m(r, 'headingErrDeg'), m(r, 'headRecall', 'mean'), m(r, 'headPrecision', 'mean'),
          m(r, 'pelletRecall', 'mean'), fmt(r.iou.self?.mean, 3), fmt(r.iou.other?.mean, 3), fmt(r.iou.pellet?.mean, 3), fmt(r.iou.wall?.mean, 3), fmt(r.obsMAE.rayBody, 3), fmt(r.obsMAE.foodSum, 3), fmt(r.obsMAE.opponents, 3), m(r, 'detectMs', 'mean', 1, 0)])} />
    </Card>
  );
}

export function VisionView() {
  const studio = useJson<Studio>(`${B}data/vision/studio/session.json`);
  const acc = useJson<AccuracyRow[]>(`${B}data/vision/accuracy.json`);
  return (
    <Page title="화면 인식 실험실" intro={<>학습된 정책은 엔진 상태 없이 브라우저 화면의 픽셀만으로 관측 벡터를 추정해 행동합니다: <b>캡처 → 컴퓨터 비전 → obs-v1 → 신경망 → 마우스/Space 입력</b>. 배율은 바닥 격자 간격, 자기 방향은 눈의 위치, 자기 길이는 캔버스 HUD 게이지에서 읽습니다. Ground Truth는 정확도 측정에만 쓰입니다.</>}>
      <div className="flex flex-wrap gap-2"><Badge tone="sky">Read-only Studio</Badge><Badge>엔진 상태 접근 차단 (GuardedPage)</Badge><Badge>대상 주소 허용 목록</Badge></div>
      {studio.data ? (
        <Card title={`Studio — 기록된 세션 재생 (${studio.data.model}, ${studio.data.mode === 'lockstep' ? '록스텝' : '실시간'}, 테마 ${studio.data.theme})`} subtitle="재생 전용입니다. 이 화면을 열거나 닫아도 정책 실행이나 게임 상태는 바뀌지 않습니다.">
          <StudioPlayer s={studio.data} />
        </Card>
      ) : <Missing what="기록된 Vision 세션" how="python -m vision.bridge.runner --record … && python -m vision.studio_export …" />}
      {acc.data ? <Accuracy rows={acc.data} /> : <Missing what="인식 정확도" how="python -m vision.report" />}
      <LivePanel />
    </Page>
  );
}
