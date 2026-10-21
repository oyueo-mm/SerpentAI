# 참고 프로젝트와 라이선스

## Hasty-CR (https://github.com/hastylmao/Hasty-CR)

설계 원칙만 참고했으며 **코드나 자산을 복사하지 않았습니다.** 참고한 원칙:

- 결정론적 시뮬레이터와 대량 headless 경기
- 전문가 정책 기반 Behavior Cloning → PPO
- 학습 Supervisor (지표 감시, 개입, 체크포인트 복귀)
- 다양한 상대 정책에 대한 평가
- 화면 인식과 정책 추론의 연결, 관측·행동 계약의 일치
- Calibration과 시뮬레이터 악용 탐색
- 측정한 결과와 한계를 그대로 기록

게임(Clash Royale)이 전혀 다르므로 물리·관측·행동 공간은 모두 SerpentAI용으로 새로 설계했습니다.

## Slither.io

장르적 아이디어(긴 몸통, 부스트, 죽으면 먹이가 됨)만 참고했습니다. 원작의 코드·이미지·로고는 사용하지 않았고, 실제 Slither.io 서버에 접속하거나 통신 프로토콜을 분석하거나 자동 플레이하는 코드는 **없습니다.** Vision Bridge는 대상 주소를 이 프로젝트의 로컬 빌드와 `https://oyueo-mm.github.io/SerpentAI/`로 제한합니다 (`vision/bridge/runner.py:check_target`, 테스트 포함). 이 프로젝트는 Slither.io 또는 그 운영사와 아무 관계가 없습니다.

## 같은 작성자의 이전 프로젝트

UI 디자인 원칙(화이트 기반, 한국어 UI, 접이식 패널, 로그 스케일 속도 슬라이더)과 Node 브리지 구조는 같은 저장소 소유자의 NebulousAI·EvolutionLab 등에서 확립한 방식을 따랐습니다.

## 사용한 오픈소스

| 패키지 | 라이선스 | 용도 |
|---|---|---|
| React, React DOM | MIT | UI |
| Vite, @vitejs/plugin-react | MIT | 빌드 |
| Tailwind CSS | MIT | 스타일 |
| Vitest, Playwright, TypeScript, esbuild | MIT / Apache-2.0 | 테스트·빌드 |
| PyTorch | BSD-3-Clause | 학습 |
| Stable-Baselines3, sb3-contrib | MIT | PPO, MaskablePPO |
| Gymnasium | MIT | 환경 API |
| NumPy | BSD-3-Clause | 수치 |
| OpenCV (opencv-python-headless) | Apache-2.0 | 고전 CV |
| Pillow | MIT-CMU (HPND) | 이미지 입출력 |
| Playwright for Python | Apache-2.0 | 브라우저 캡처·입력 |

충돌 거리 함수 `segSegDistSq`는 Ericson, *Real-Time Collision Detection* (2005) §5.1.9의 공개 알고리즘을 구현한 것입니다.
