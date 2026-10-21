# SerpentAI

지렁이 생존 게임을 직접 플레이하고, 휴리스틱·MPC·모방학습·강화학습 AI의 전략과 화면 기반 자동 제어를 실험하는 인터랙티브 연구실.

**Live Demo: https://oyueo-mm.github.io/SerpentAI/**

SerpentAI는 독립적으로 만든 지렁이 생존 게임(결정론적 시뮬레이터)과, 그 위에서 돌아가는 AI 연구 도구 모음입니다. 같은 게임 엔진이 브라우저(Web Worker), 학습(Node 브리지 + Python), 평가에서 그대로 쓰이고, 학습한 정책은 엔진 상태 없이 **브라우저 화면의 픽셀만 보고** 별도의 브라우저 창에서 게임을 조작할 수 있습니다.

> Slither.io 등 외부 게임 서비스와 관계가 없습니다. 실제 외부 서버에 접속하거나 자동 플레이하는 코드는 없으며, 화면 기반 제어는 이 프로젝트의 게임에만 허용됩니다([`docs/REFERENCE_LICENSES.md`](docs/REFERENCE_LICENSES.md)).

## 연구 질문과 결과 요약

| 질문 | 답 (실측) | 자세히 |
|---|---|---|
| **Q1** 휴리스틱·MPC·강화학습 중 무엇이 가장 잘 생존하는가? | **MPC**가 모든 기준에서 1위(새 시드 100개: 리턴 18.1, 사망률 9%, 24마리 토너먼트 평균 순위 3.2). PPO는 Hybrid 등 휴리스틱보다 강하지만(리턴 12.5–14.4 vs 6.9) MPC에는 못 미침 | [실험](docs/EXPERIMENTS.md#q1) |
| **Q2** 전문가 모방학습이 PPO의 학습 효율을 개선하는가? | **초기 효율은 크게 개선**(10만 스텝 검증 리턴 0.4 → 5.7/7.1). 최종 성능 차이는 작고 CI가 겹치며, 모방 초기화 PPO는 더 길게 자라지만 더 자주 죽음. 학습 시드 1개, Supervisor 개입 차이라는 한계 | [실험](docs/EXPERIMENTS.md#q2) |
| **Q3** 내부 상태로 학습한 정책이 화면 인식만으로도 작동하는가? | **그렇다, 시간 조건이 같다면.** 록스텝 화면 기반 플레이(생존 81.2 s)와 정확한 상태(84.5 s)가 구분되지 않음 | [실험](docs/EXPERIMENTS.md#q3q4-sim-to-visual) |
| **Q4** 인식 오차와 제어 지연이 성능에 미치는 영향은? | 실시간 제어에서 크게 저하(54.3 s, 8/8 사망). 측정한 지연(133 ms)·잡음 각각만으로는 설명되지 않음. 학습하지 않은 시각 테마에서 가장 크게 저하(40.9 s) | [Calibration](docs/CALIBRATION.md) |

### Sim-to-Visual (Q3·Q4) — 같은 PPO 정책, 같은 시드 8개

| 조건 | 생존 (s, 95% CI) | 사망 | 최대 길이 |
|---|---|---|---|
| A 정확한 엔진 상태 | 84.5 [49.3, 119.6] | 3/8 | 1210 |
| B 상태 + 측정 지연 133 ms | 76.0 [42.4, 109.7] | 4/8 | 1120 |
| C 상태 + 측정 관측 잡음 | 94.0 [67.2, 120.7] | 3/8 | 1404 |
| **D 화면 픽셀만, 록스텝** | **81.2** [57.5, 104.8] | 5/8 | **1217** |
| E 화면 픽셀만, 실시간 | 54.3 [27.7, 81.0] | 8/8 | 672 |
| F 화면 픽셀만, 실시간, 학습 안 한 테마 | 40.9 [14.9, 67.0] | 8/8 | 549 |

- **Q3**: 엔진 상태로 학습한 정책이 화면 픽셀만으로 플레이해도, 시간 조건이 같으면(D) 정확한 상태(A)와 성능을 구분할 수 없습니다.
- **Q4**: 실시간(E)에서는 크게 떨어집니다. 측정한 지연만(B) 또는 잡음만(C) 넣으면 거의 떨어지지 않으므로, 실시간에서만 생기는 요인들(움직이는 화면의 큰 방향 오차, 마우스 입력 의미 차이, 타이밍 흔들림)이 함께 작용한 것으로 보이며 이 실험으로 분리하지 못했습니다. 실시간 결과는 같은 코드로도 실행마다 크게 달랐습니다(E: 77.8 → 54.3 s). 시각 도메인 변화(F)가 가장 큰 저하 요인입니다.
- 특징별 오차 분석으로 **상대 속도 추정의 편향**(순항 상대를 부스트로 오인)을 찾아 줄였습니다(+0.81 → +0.31). 이 버그로 첫 실행의 F는 17.4 s로 붕괴했습니다.


## 화면 구성

| 모드 | 내용 |
|---|---|
| **직접 플레이** | 사람이 지렁이를 조작해 AI들과 경쟁 |
| **AI 관전** | 휴리스틱·MPC·학습 모델이 경쟁. 지렁이를 클릭하면 판단 근거(MPC 후보 궤적·비용표, 학습 모델의 행동 확률)를 표시. 디버그 그래픽(충돌 캡슐, 중심선, 회전 반경, 공간 해시 격자, 광선 위험도, 예측 경로, 실제 이동 경로, **AI 시야: 360° 감지 범위와 감지되지 않는 영역**). 사람이 보는 화면과 AI 감지 범위의 차이는 [`docs/OBSERVATION_ACTION.md`](docs/OBSERVATION_ACTION.md#사용자가-보는-화면-vs-ai가-감지하는-범위) |
| **토너먼트** | 사전 계산 결과(95% CI) + 브라우저에서 직접 실행 |
| **강화학습 실험실** | 전문가 데이터 구성, BC·DAgger 라운드, PPO 학습 곡선(같은 스텝 비교), 커리큘럼 관문, Supervisor 기록, 최종 평가와 일반화, 배포 모델 |
| **화면 인식** | 기록된 Vision 세션의 읽기 전용 Studio 재생(프레임 + 인식 결과 + 관측 벡터 + 행동 확률 + 지연), 고전 CV vs CNN 정확도, 로컬 Vision Runner 제어(Start/Stop/Emergency Stop) |
| **Calibration** | Sim-to-Visual 조건 A–F 비교, 제어 루프 지연 구성, 인식 오차, 지연·잡음 민감도, 오차가 큰 상황 |

## 조작법

- **마우스**: 머리가 포인터 방향으로 회전 (최대 각속도 제한)
- **← → / A D**: 최대 속도로 회전 · **Space 또는 마우스 버튼**: 부스트 (길이를 소모, 길이 160 이하에서는 불가)
- **P**: 일시정지 · **.**: 한 틱 진행 · 독(dock)의 슬라이더: 로그 스케일 배속(0.1×–64×, 1× 복귀, MAX). 목표 배속과 실제 배속을 따로 표시
- **카메라** (독의 `추적 / 자유 / 전체`): 우클릭 또는 휠 클릭 드래그로 이동(자유 카메라), 휠로 확대·축소, **F**로 추적 복귀. 모바일은 한 손가락 끌기로 이동, 두 손가락으로 확대. 카메라 입력은 조향과 분리되어 있고(드래그 중에는 조향 방향 고정), 지렁이 선택·배속 변경으로 시점이 초기화되지 않습니다. 카메라는 그림만 바꾸며 게임과 AI의 입력에는 영향이 없습니다
- **모바일**: 왼쪽 조이스틱(방향) + 오른쪽 부스트 버튼
- 설정 패널에서 AI 구성·시드·먹이 양·결정 주기를 바꾸고, 경기 상태를 파일로 저장·불러오기

## 시뮬레이터

규칙 전체는 [`docs/RULES.md`](docs/RULES.md), 수식은 [`docs/MATHEMATICS.md`](docs/MATHEMATICS.md).

- 3000 × 3000 u 경기장, 고정 30 틱/s (렌더링 FPS와 무관), 단위: u·s·rad.
- 머리는 **정확한 원호 적분**으로 이동(틱 간격이 달라도 같은 궤적), 몸통은 이동 거리 기준으로 꼭짓점을 확정하는 호 길이 폴리라인이고 꼬리를 정확히 길이 L로 자릅니다.
- 충돌: 머리 원 vs 다른 몸통 **캡슐**, 연속 최근접 거리로 정면 충돌(짧은 쪽 사망, 같으면 둘 다), 벽. 하위 스텝(≤ 5 u)으로 고속 관통을 막고, 모든 사망을 같은 상태에서 판정한 뒤 동시에 적용합니다. 공간 해시(96 u)로 후보를 줄입니다.
- 부스트: 2배 속도, 45 u/s 소모(절반은 먹이로 떨어짐). 사망 시 길이의 60%가 먹이로.
- **결정론**: 같은 시드 + 같은 입력 → 비트 단위로 같은 상태, 저장 → 복원 → 계속도 동일. 독립 Python 구현과 1e-6 u 이내로 일치.

## AI

| 종류 | 정책 |
|---|---|
| 휴리스틱 | Random, Greedy(몸통 무시), Survival(16광선 반응형), Aggressive(진로 차단 요격), Potential Field, Hybrid(안전 계층 + 목표 우선순위) |
| 탐색 | **MPC**: 18 행동 × 5 후속 회전 = 90개 궤적을 1.5초 동안 엔진과 같은 운동학으로 예측, 먹이·말단 가치·충돌·안전 여유·부스트 비용을 따로 계산 ([`docs/AI_ALGORITHMS.md`](docs/AI_ALGORITHMS.md)) |
| 모방학습 | BC (MPC 시연 20만 상태), BC + DAgger 4라운드 ([`docs/IMITATION_LEARNING.md`](docs/IMITATION_LEARNING.md)) |
| 강화학습 | MaskablePPO × 3 (무작위 / BC / BC+DAgger 초기화), 6단계 커리큘럼 + Self-Play, Supervisor ([`docs/REINFORCEMENT_LEARNING.md`](docs/REINFORCEMENT_LEARNING.md)) |

모든 정책은 같은 행동 공간 act-v1(회전 9단계 × 부스트 = 18)과 같은 평가 도구를 씁니다. 학습 정책의 관측 obs-v1(84차원, 자기 중심 좌표)은 **화면에서 볼 수 있는 양으로만** 정의되어 있어, 엔진 상태와 화면 인식 결과가 같은 구조로 들어갑니다 ([`docs/OBSERVATION_ACTION.md`](docs/OBSERVATION_ACTION.md)). 브라우저의 학습 모델은 `public/models/*.json`에서 실제로 불러오며, 휴리스틱을 학습 모델로 표시하지 않습니다.

## 화면 인식과 Vision Bridge

```
별도 Chromium 창의 게임 캔버스 → 캡처 → 컴퓨터 비전 → obs-v1 → 같은 신경망 → 마우스 방향 / Space 키
```

- **고전 CV**: 바닥 격자 주기로 배율, 경기장 사각형으로 벽, 색상 클래스로 몸통·먹이, 몸통 마스크 속 닫힌 구멍(눈) 쌍으로 머리와 방향, HUD 게이지로 자기 길이. 격자 위상으로 카메라 이동을 측정하고 칼만 필터로 상대를 추적합니다.
- **학습형 CNN**: 실제 게임 페이지에서 같은 틱을 일반/라벨 테마로 두 번 렌더링해 만든 합성 데이터(도메인 무작위화 1,199장)로 TinyUNet 세그멘테이션을 학습해 비교.
- **안전**: 대상 주소 허용 목록, 인식 실패·신뢰도 하락·프레임 지연·느린 추론 시 정지, 비상 정지(파일/엔드포인트/웹 패널), 제어 루프는 엔진 상태에 접근할 수 없는 `GuardedPage`만 사용.

자세히: [`docs/VISION.md`](docs/VISION.md), [`docs/CALIBRATION.md`](docs/CALIBRATION.md).

**인식 정확도 (학습에 쓰지 않은 검증 세트, 고전 CV)** — 원본 [`public/data/vision/accuracy.json`](public/data/vision/accuracy.json)

| 세트 | 자기 방향 오차 (중앙값) | 상대 머리 재현율 | 먹이 재현율 | IoU 자기/상대/먹이 | 광선 관측 MAE | 처리 시간 |
|---|---|---|---|---|---|---|
| 기본 테마 | 1.07° | 0.98 | 0.99 | 0.996 / 0.956 / 0.941 | 0.001 | 36 ms |
| 무작위 테마 (미학습 시드) | 1.01° | 1.00 | 0.96 | 0.992 / 0.897 / 0.916 | 0.003 | 35 ms |
| **shift 테마 (전혀 학습 안 함)** | 0.72° | 1.00 | 0.97 | 0.993 / 0.976 / 0.905 | 0.005 | 41 ms |
| 무작위 + 잡음·JPEG·가림 | 1.34° | 0.74 | 0.96 | 0.975 / 0.906 / 0.867 | 0.005 | 38 ms (프레임의 20%는 인식 실패) |

CNN은 상대 몸통 IoU가 무작위 테마에서 더 좋지만(0.936 vs 0.897), 눈 세그멘테이션이 불완전해 머리 재현율이 0.20–0.50으로 낮고, 기본 테마의 연회색 벽을 바닥으로 분류하며(벽 IoU 0.0), 약 10배 느립니다(350–410 ms). 실제 Vision Bridge는 고전 CV를 씁니다.

## 성능 (Intel Core Ultra 7 256V, CPU만)

| 항목 | 측정값 |
|---|---|
| 엔진 (20마리, L = 400) | 0.32 ms/틱 (≈ 3,100 틱/s, 단일 스레드) |
| MPC 결정 1회 (90 궤적) | 0.31 ms |
| MLP 추론 (TypeScript) | 0.067 ms |
| PPO 학습 | 2.04M 스텝 / 19–22분 (56 환경) |
| 고전 CV 인식 / 화면 캡처 | 약 35–44 ms / 약 43 ms (960×720) |

전체: [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## 테스트

| | 개수 | 내용 |
|---|---|---|
| Vitest | 58 | 물리(관통 불가 증명 포함)·질량 보존 퍼징·결정론·저장/복원·악용 탐색·MPC 예측 정확성·관측 회전 불변성·렌더링 경로 연속성과 길이·카메라 |
| pytest `rl/tests` | 20 | Gymnasium/SB3 검사, Python 참조 엔진 동치성(골든 궤적 3종), Supervisor 개입, BC/DAgger/PPO 갱신, TS↔PyTorch 추론 일치 |
| pytest `vision/tests` | 15 | 인코더 동치성, 실제 게임 프레임 인식, 추적, 가림, Vision Bridge 안전 장치·상태 접근 차단 |
| Playwright | 22 (+4 의도적 건너뜀) | 데스크톱 + 모바일: 플레이, 관전, 속도, 저장/불러오기, 카메라(드래그·휠·F·터치, 시뮬레이션 불변), Vision 대상 페이지의 고정 카메라, 분석 페이지가 실제 결과 데이터를 표시, 콘솔 오류, Pages 하위 경로·404 |

목록: [`docs/TESTING.md`](docs/TESTING.md). GitHub Actions가 모두 실행한 뒤 Pages에 배포합니다.

## 실행

```bash
npm ci
npm run dev                      # http://127.0.0.1:5173/SerpentAI/
npm test && npm run build && npm run test:e2e
npm run tournament -- --rounds 30 --out public/data/tournament.json
npm run benchmark -- --out public/data/benchmark.json
```

학습과 Vision Runner는 로컬에서 실행합니다 (CI에서는 작은 스모크 테스트만):

```bash
npm run build:node
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r rl/requirements.txt -r vision/requirements.txt && python -m playwright install chromium
python -m pytest rl/tests vision/tests

python -m rl.imitate all --expert mpc --run il-v1                                   # BC + DAgger
python -m rl.train_ppo --variant dagger --init rl/runs/il-v1/bc-r2.zip --run ppo-dagger-v1
python -m rl.evaluate --episodes 100                                                # 최종 평가

npm run build && npm run preview &                                                  # 게임 서버
python -m vision.bridge.runner --model public/models/ppo-dagger-v1.json --headed   # 화면만 보고 플레이
python -m vision.sim2visual --model public/models/ppo-dagger-v1.json --episodes 8  # 조건 A–F
```

## 제한 사항

- 학습 시드가 변형당 1개라 PPO 변형 간 차이의 시드 분산을 모릅니다. Supervisor의 학습률 개입이 변형마다 달랐습니다.
- 학습한 정책은 전문가 MPC보다 약합니다. MPC의 예측 모델은 상대의 회전·부스트를 예측하지 않습니다.
- 화면 인식은 이 게임의 공개된 그리기 규칙(격자, 눈, HUD, 고정 카메라)에 기대며, 다른 게임에 그대로 적용되지 않습니다. 몸통 아래에 깔린 먹이는 볼 수 없습니다.
- CNN 세그멘테이션은 머리 검출에서 고전 CV보다 나쁘고 실시간으로 쓰기에는 느립니다.
- Vision Bridge의 캡처는 Playwright 스크린샷이며 OS 수준 화면 녹화가 아닙니다. 측정 지연은 이 방식 기준입니다.
- 자기 몸통 충돌은 규칙상 없습니다(rules-v1).

## 참고 프로젝트와 라이선스

설계 원칙은 [Hasty-CR](https://github.com/hastylmao/Hasty-CR)을 참고했고 코드·자산은 복사하지 않았습니다. 사용한 오픈소스와 라이선스는 [`docs/REFERENCE_LICENSES.md`](docs/REFERENCE_LICENSES.md). 이 저장소의 코드는 MIT 라이선스입니다 ([`LICENSE`](LICENSE)).

문서: [아키텍처](docs/ARCHITECTURE.md) · [규칙](docs/RULES.md) · [수학](docs/MATHEMATICS.md) · [AI 알고리즘](docs/AI_ALGORITHMS.md) · [관측·행동](docs/OBSERVATION_ACTION.md) · [모방학습](docs/IMITATION_LEARNING.md) · [강화학습](docs/REINFORCEMENT_LEARNING.md) · [화면 인식](docs/VISION.md) · [Calibration](docs/CALIBRATION.md) · [실험](docs/EXPERIMENTS.md) · [벤치마크](docs/BENCHMARK.md) · [테스트](docs/TESTING.md)
