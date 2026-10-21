# 아키텍처

```
 ┌──────────────── Layer A: Game Simulator (TypeScript, src/sim, src/geometry) ───────────────┐
 │ World.step(controls) — 원호 이동 · 몸통 경로 · 먹이 · 부스트 · 캡슐 충돌 · 사망 · 직렬화      │
 └───────────────▲──────────────────────────▲─────────────────────────────▲──────────────────┘
                 │ 같은 코드                 │ 같은 코드 (Node 번들)        │ 같은 코드 (Web Worker)
 ┌───────────────┴─────────┐   ┌────────────┴─────────────┐   ┌───────────┴────────────────────┐
 │ Layer B: Expert Policies│   │ Layer C: Learning (Python)│   │ Layer F: Interactive Lab (React)│
 │ src/ai: 휴리스틱 6종,    │   │ rl/: Gymnasium · SB3 ·    │   │ src/ui: 플레이 · 관전 · 토너먼트 │
 │ MPC (pm-v1 예측 모델)    │   │ BC · DAgger · PPO ·       │   │ · 강화학습 · 화면 인식 · Calib.  │
 └─────────────┬───────────┘   │ Supervisor · export       │   └───────────▲────────────────────┘
               │ act-v1         └────────────┬─────────────┘               │ 캔버스 픽셀만
               ▼                             │ serpent-mlp JSON            │
       Match (src/sim/match.ts)              ▼                 ┌───────────┴────────────────────┐
       결정 주기 · 지연 큐 · 부활 ◀── MlpPolicy (src/policy) ◀─│ Layer E: Vision Bridge (Python) │
                                                              │ vision/bridge: Playwright 캡처 →│
 ┌──────────── Layer D: Visual Perception (Python, vision/) ──│ 인식 → obs → MLP → 마우스/Space │
 │ detection(고전 CV) · segmentation(CNN) · tracking(칼만,     └────────────────────────────────┘
 │ 주행거리) · observation(Detections → Sensors → obs-v1)
 └──────────────────────────────────────────────────────────────────────────────────────────────
```

## 계층 사이의 데이터 계약

| 경계 | 형식 | 정의 |
|---|---|---|
| 정책 → 엔진 | `Control {turn | targetHeading, boost}` / act-v1 정수 0–17 | `src/sim/world.ts`, `src/policy/action.ts` |
| 엔진 → 정책 | `Sensors` → obs-v1 Float32[84] | `src/observation/sensors.ts`, `obs.ts` |
| 화면 → 정책 | `Detections` → `Sensors` → obs-v1 (Python 포트) | `vision/observation/` |
| Python ↔ Node | u32 길이 + JSON 헤더 (+ float32 블록) | `rl/bridge/bridge.ts`, `rl/bridge_client.py` |
| 학습 → 브라우저 | `serpent-mlp` JSON (가중치, 관측 정규화, 계약 버전) | `src/policy/mlp.ts`, `rl/common.py:export_model` |
| 시뮬레이션 → 렌더러 | `Frame` (전송 가능한 Float32Array) | `src/render/frame.ts` |
| 엔진 → 평가 기록 | `GroundTruth` (`?gt=1`일 때만) | `src/worker/protocol.ts` |

버전 문자열 `rules-v1`, `obs-v1`, `act-v1`, `reward-v1`은 모든 경계에서 확인되며, 다르면 불러오기·연결을 거부합니다.

## 실행 구성

- **브라우저**: 시뮬레이션은 Web Worker(`sim.worker.ts`)에서 고정 틱으로 돌고, 메인 스레드는 최신 프레임을 requestAnimationFrame으로 그립니다. 배속은 실제 시간당 실행하는 틱 수만 바꿉니다(목표 배속과 측정 배속을 따로 표시). 1배속에서는 두 프레임 사이를 표시용으로만 보간합니다.
- **학습**: Python이 Node 브리지 프로세스 여러 개(각 8개 환경)를 병렬로 구동합니다. 게임 규칙은 Python에 다시 구현하지 않습니다 (방법 A — [`REINFORCEMENT_LEARNING.md`](REINFORCEMENT_LEARNING.md)의 비교 참조).
- **Vision Bridge**: 별도의 Chromium 창(Playwright)에 `?view=target` 게임을 띄우고, 러너는 그 창의 캔버스 스크린샷과 공식 입력만 씁니다. 러너가 받는 페이지 객체(`GuardedPage`)는 `evaluate` 같은 상태 접근을 예외로 막습니다. Ground Truth는 Calibration 모드에서 **실행이 끝난 뒤** 한 번만 읽습니다.

## 디렉터리

| 경로 | 내용 |
|---|---|
| `src/sim` | World, BodyPath, 규칙 상수, RNG, Match, 토너먼트, 통계 |
| `src/geometry` | 거리·광선·원호 적분, 공간 해시 |
| `src/ai` | 정책 인터페이스, 휴리스틱, MPC, 예측 모델(pm-v1), 레지스트리 |
| `src/observation`, `src/policy` | 관측·행동 계약, MLP 추론 |
| `src/render`, `src/worker`, `src/ui`, `src/i18n` | 렌더러·테마, 워커, React UI, 한국어 문자열 |
| `src/cli` | 토너먼트·벤치마크·골든 궤적 CLI (Node) |
| `rl/` | 브리지, 환경, 모방학습, PPO, Supervisor, 평가, Python 참조 엔진, 테스트 |
| `vision/` | 인식(고전·CNN), 추적, 관측 변환, 러너, Calibration, 데이터 생성, 평가, 테스트 |
| `tests/e2e` | Playwright 브라우저 테스트 |
| `public/models`, `public/data` | 배포되는 모델과 실험 결과(JSON) |
