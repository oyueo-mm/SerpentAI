import { useEffect, useState, type MutableRefObject } from 'react';
import { FOOD_RANGE, FOOD_SECTORS, OPP_K, OPP_RANGE, RAY_COUNT, RAY_RANGE } from '../observation/sensors';
import type { Camera } from '../render/renderer';
import { Section, Toggle } from './kit';

/** Fraction of the visible screen area that lies inside the AI's sensing circle (grid sampling). */
export function sensedFraction(cam: Camera, w: number, h: number, hx: number, hy: number, radius = RAY_RANGE): number {
  const N = 40;
  let inside = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const wx = cam.cx + ((i + 0.5) / N - 0.5) * (w / cam.zoom), wy = cam.cy + ((j + 0.5) / N - 0.5) * (h / cam.zoom);
    if (Math.hypot(wx - hx, wy - hy) <= radius) inside++;
  }
  return inside / (N * N);
}

interface Props {
  view: MutableRefObject<{ cam: Camera; w: number; h: number } | null>;
  /** World position of the inspected snake's head (null if dead). */
  head: () => [number, number] | null;
  perception: boolean;
  onPerception: (v: boolean) => void;
}

/**
 * "AI field of view": what the screen shows vs. what obs-v1 can sense. The observation is computed from
 * the engine around the snake's head and never depends on the camera, so panning or zooming changes only
 * the picture, not the AI's input.
 */
export function AiViewSection({ view, head, perception, onPerception }: Props) {
  const [s, setS] = useState<{ halfW: number; halfH: number; frac: number } | null>(null);
  useEffect(() => {
    const id = setInterval(() => {
      const v = view.current, h = head();
      if (!v) return;
      setS({ halfW: v.w / 2 / v.cam.zoom, halfH: v.h / 2 / v.cam.zoom, frac: h ? sensedFraction(v.cam, v.w, v.h, h[0], h[1]) : NaN });
    }, 500);
    return () => clearInterval(id);
  }, [view, head]);
  const gap = (2 * RAY_RANGE * Math.sin(Math.PI / RAY_COUNT)).toFixed(0);
  return (
    <Section title="AI 시야" defaultOpen={false}>
      <Toggle label="360° 감지 범위 표시" checked={perception} onChange={onPerception} testId="ai-view-toggle" hint="광선·먹이 구역·추적 대상과 감지 범위 밖(어둡게)" />
      <div className="space-y-1.5 text-xs leading-relaxed text-slate-600" data-testid="ai-view-text">
        <p>학습 정책(obs-v1)은 화면 전체가 아니라 머리 주변의 정해진 양만 봅니다. 방향은 <b>360° 전방위</b>입니다.</p>
        <ul className="list-disc space-y-0.5 pl-4">
          <li>장애물: {RAY_COUNT}개 광선(22.5° 간격)으로 반경 {RAY_RANGE} u 안의 가장 가까운 몸통 표면과 벽까지의 거리만 압니다. 광선 사이는 보지 못합니다 — {RAY_RANGE} u 거리에서 광선 사이 간격은 약 {gap} u입니다. 좁은 꼬리 끝이나 광선과 나란한 몸통은 놓칠 수 있습니다.</li>
          <li>먹이: 반경 {FOOD_RANGE} u를 {FOOD_SECTORS}개 구역으로 나눠 구역별 <b>합계와 가장 가까운 거리</b>만 압니다. 개별 먹이 위치는 모릅니다.</li>
          <li>상대: {OPP_RANGE} u 안에서 가장 가까운 머리 {OPP_K}개의 위치·방향·속도·크기만 압니다. 상대의 몸통 모양은 광선으로만 봅니다.</li>
          <li>자기 몸통은 감지하지 않습니다(자기 충돌이 없는 규칙).</li>
        </ul>
        {s && (
          <p className="rounded-md bg-slate-50 px-2 py-1.5 tabular-nums" data-testid="ai-view-numbers">
            지금 화면: 가로 ±{s.halfW.toFixed(0)} u · 세로 ±{s.halfH.toFixed(0)} u
            {Number.isFinite(s.frac) && <> · AI 감지 반경 안: 화면의 {(s.frac * 100).toFixed(0)}%</>}
          </p>
        )}
        <p>카메라를 옮기거나 확대해도 AI가 받는 정보는 변하지 않습니다. 관측은 엔진 상태에서 머리 기준으로 계산되고, 카메라는 그림만 바꿉니다. 기본 추적 카메라도 짧은 변 기준 480–600 u를 보여 주므로 사람은 AI보다 멀리 봅니다.</p>
      </div>
    </Section>
  );
}
