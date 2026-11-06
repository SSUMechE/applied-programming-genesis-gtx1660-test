# Applied Programming — GTX 1660 Test

## 전체 맵 운반 시험 · 2026-11-06

쉬움·중간·어려움의 전체 경로를 검사하는 [full_tier 실행 안내](full_tier/README.md)를 추가했습니다. 원래 초기 상태에서 시작하여 로봇 1·2·3개의 운반과 필요한 0·1·2회의 인계를 끝까지 검사합니다. 용기와 구슬 4개의 상태는 인계 중에도 이어집니다. 중간 로봇의 복귀는 기존 경로를 물리 계산 없이 되짚으며, 마지막 로봇은 복귀하지 않습니다.

세 예시 모두 RTX 4090에서 전체 검증을 통과했습니다. CPU 검증과 GPU 실행의 합계는 쉬움 **12분 25초**, 중간 **40분 22초**, 어려움 **52분 8초**였습니다. 관측 최대 프로세스 VRAM은 각각 **3.339 / 3.513 / 3.660 GB**입니다. 이 CPU 비용은 제공된 고정 경로의 검증 비용이며 RRT 경로 탐색 시간은 포함하지 않습니다.

**GTX 1660의 전체 맵 비용은 아직 측정 전입니다.** 아래에 보존된 GTX 1660의 15분 10초 결과는 기존 단일 운반 예시입니다. 새 전체 맵 시험과 구분해서 비교합니다. 30분 기준은 로봇 1개에 적용하며, 2·3개 전체 비용은 별도로 기록합니다. VRAM 5 GB는 관측 기준이며 순간 할당량의 강제 상한이 아닙니다.

기존 저장소 폴더의 Anaconda Prompt에서 다음 명령을 실행한 뒤 [난이도별 CPU·GPU 명령](full_tier/README.md)을 순서대로 실행합니다. 새 예시는 같은 `applied-programming-cuda` 환경을 사용합니다.

```text
git pull --ff-only
cd full_tier
conda activate applied-programming-cuda
python -B check_environment.py --cuda --output output/environment.json
```

이 공개 패키지는 하드웨어 측정용입니다. 경쟁 점수 제출 규칙은 아직 비활성 상태입니다.

## 기존 단일 운반 예시

기존 **`applied-programming-cuda` Conda 환경**에서 실행하는 GPU 시험 예시입니다. 제공된 경로와 구간별 이동 시간이 완성되어 있으므로 코드를 수정하지 않고 실행할 수 있습니다.

하드웨어 비교용 예시는 시작·도착 관절값도 고정 입력으로 사용합니다. PC마다 역기구학으로 다시 계산하지 않으며, 제공된 배치에서 자세·관절 제한·충돌을 검증한 뒤 전체 경로를 검사합니다.

한 로봇암이 용기와 구슬 4개를 운반합니다. 다른 두 팔은 정지한 충돌 형상으로 유지합니다. 물리 시간 간격은 `0.0025 s`, 내부 계산은 16단계입니다. 잡기·놓기는 제공된 절차를 사용하며, 운반·충돌·시료 유지·최종 놓기와 정지를 검사합니다.

**지원 실행 방식:** Windows 10/11의 Anaconda Prompt. 화면을 띄우지 않고 결과 파일을 저장하는 시험입니다. 다른 GPU로도 실행할 수 있으며 실제 GPU 정보가 기록됩니다.

## GTX 1660 검증 결과

2026-11-05에 GTX 1660 PC에서 제공된 예시의 운반·놓기·해제·최종 정지 검사가 완료됐습니다. 실행자가 제공한 최종 콘솔 결과에서 `ACCEPTED_TRANSPORT`, `review_accepted: true`, 네 검사 그룹의 통과를 확인했습니다. 실행 전 업데이트한 코드 버전은 [`510d41d`](https://github.com/SSUMechE/applied-programming-genesis-gtx1660-test/commit/510d41d940ff38107de266c134e233f293513ff9)입니다.

다른 PC에서 같은 예시를 실행한 뒤, 다음 값과 소요 시간·메모리를 비교할 수 있습니다.

| 항목 | 측정값 |
|---|---|
| CPU 경로 검증 | 2.13초 |
| GPU 물리 검증 | 908.18초 · 15분 8초 |
| CPU + GPU 합계 | 910.31초 · **15분 10초** |
| 관측 최대 프로세스 VRAM | **2.302 GB** · 십진 GB |

이 실행은 30분 시간 제한과 관측 VRAM 5 GB 조건을 통과했습니다. VRAM은 표본 최대치이며 순간 최대 할당량을 보장하지 않습니다. 위 값은 제공된 한 경로의 한 번의 실행 결과입니다. PC 구성과 경로·이동 시간이 달라지면 비용도 달라집니다. 측정값과 증거 범위는 [검증 결과 JSON](benchmarks/gtx1660_20261105.json)에 있습니다. GPU 모델명은 실행자 설명에 근거하며, 최종 콘솔 요약을 검토한 결과입니다.

## 실행

Anaconda Prompt에서 실행합니다. 코드 블록 안의 명령만 복사하고, 명령창의 프롬프트나 출력된 JSON을 명령으로 붙여넣지 않습니다. 각 단계가 정상 완료되면 다음 단계로 진행합니다.

처음 내려받는 PC에서는 다음 두 명령을 실행합니다.

```text
git clone https://github.com/SSUMechE/applied-programming-genesis-gtx1660-test.git
cd applied-programming-genesis-gtx1660-test
```

이미 저장소가 있으면 기존 저장소 폴더에서 업데이트합니다.

```text
git pull --ff-only
```

저장소 폴더에서 기존 환경을 활성화하고 확인합니다. 이전 실행 결과가 있으면 아래의 **다시 실행하거나 경로를 바꿀 때**에 따라 새 출력 이름을 사용합니다.

```text
conda activate applied-programming-cuda
python -B check_environment.py --cuda --output output/environment.json
```

환경 검사는 필요한 패키지·소스 버전과 실제 CUDA 동작을 확인하고 `output/environment.json`에 기록합니다. 패키지를 설치하거나 환경을 변경하지 않습니다. 환경 검사가 실패하면 표시된 원인을 확인한 뒤 진행합니다.

먼저 예시 경로를 CPU에서 검증하고, 이 컴퓨터의 소스에 연결된 실행 입력을 만듭니다. 이 단계는 Genesis를 실행하지 않습니다.

```text
python -B -m physics.plan_student_transport --scope one-carry --material beads --strategy student.fixed_example --edge-timing-hook --design example/design.json --wall-limit-s 300 --output output/cpu_example
```

`CPU_GEOMETRIC_ONE_CARRY_PLANNED`가 출력되고 `output/cpu_example/saved_one_carry.json`이 생성되면 GPU 운반 검증을 실행합니다.

```text
python -B -m physics.review_student_transport --saved-carry output/cpu_example/saved_one_carry.json --task cases/three_arm_cell_bead16_v2 --design example/design.json --material beads --execution-backend device --single-active-arm arm_0 --wall-limit-s 1800 --vram-limit-gb 5 --output output/gpu_example
```

GPU 운반 중에는 콘솔 출력이 한동안 없을 수 있습니다. 감독 프로세스가 시간과 해당 실행 프로세스의 VRAM을 계속 확인합니다. 최종 결과가 나올 때까지 기다립니다.

운반 전 경로를 검사하는 별도 GPU 장면은 자세와 충돌 검출만 사용합니다. 이 장면에서는 사용하지 않는 제약력 솔버를 실행하지 않아 GTX 1660의 공유 메모리 한도를 넘는 커널을 피합니다. 실제 운반 장면의 접촉력·중력·16단계 물리 계산과 합격 기준은 그대로 적용합니다.

## 결과 확인

`output/gpu_example/REVIEW_SUPERVISOR.json`에서 다음 값을 확인합니다. 아래 표는 실행의 성공 여부와 다른 컴퓨터의 비용을 비교할 때 사용합니다.

| 필드 | 의미 |
|---|---|
| `outcome` | 최종 판정. 정상 완료는 `ACCEPTED_TRANSPORT` |
| `review_accepted` | 전체 운반·놓기·정지와 자원 조건을 모두 통과했는지 여부 |
| `gpu_review_process_wall_s` | GPU 검증 프로세스에 걸린 실제 시간, 초 |
| `additive_cpu_plus_review_wall_s` | CPU 경로 검증과 GPU 검증의 합산 시간, 초 |
| `process_memory.sampled_peak.GB` | 해당 프로세스에서 관측한 VRAM 최대치, 십진 GB |
| `timed_out` | 시간 제한에 도달했는지 여부 |
| `observed_memory_limit_stop` | 관측한 VRAM이 제한을 넘어 중단했는지 여부 |

`review_accepted: true`여야 완료로 봅니다. CPU 경로 검증만 통과했거나 GPU 실행이 중간에 끝난 경우는 전체 성공이 아닙니다. 수치 오류와 물리적 실패는 별도로 기록됩니다. 오류가 나면 `output/gpu_example.worker.log`도 함께 확인합니다.

작업 프로세스가 실패하면 콘솔 결과의 `worker_failure`에 오류 종류와 메시지가, `worker_log`에 자세한 로그 위치가 표시됩니다.

시간 제한 1,800초에는 앞서 측정한 CPU 검증 비용이 포함됩니다. VRAM 제한은 5 GB입니다. VRAM은 일정 간격으로 관측하므로 기록된 값은 표본 최대치이며, 할당량을 강제로 제한하는 기능은 아닙니다.

다른 컴퓨터의 측정 결과를 비교하려면 이 JSON과 환경 검사 결과를 보관하면 됩니다. 저장소에 업로드할 필요는 없습니다. `output/`은 Git 추적에서 제외되어 있습니다.

## 다시 실행하거나 경로를 바꿀 때

기존 결과를 덮어쓰지 않으므로 새 출력 이름을 사용합니다. 환경 검사를 다시 기록하려면 `output/environment_2.json`처럼 새 파일을 지정합니다. CPU 명령의 출력은 `output/cpu_example_2`로 바꾸고, GPU 명령의 입력도 그 폴더의 `saved_one_carry.json`으로 맞춥니다. GPU 출력은 `output/gpu_example_2`로 지정합니다.

저장된 실행 입력은 생성한 컴퓨터의 경로와 소스에 연결됩니다. 다른 컴퓨터로 옮겼거나 실행 코드를 변경했다면 CPU 명령부터 다시 실행합니다.

실행 코드를 업데이트한 뒤에는 새 출력 이름으로 CPU 명령부터 실행합니다. CPU 단계가 실패하면 콘솔의 `failure`와 `worker_log`에 원인이 표시됩니다.

기준 시험은 제공된 경로·시간·물리 설정을 그대로 사용합니다. 경로를 변경하는 실험은 기준 측정을 마친 뒤 별도로 진행합니다. 이 기존 단일 운반 예시는 첫 번째 팔의 한 운반 구간에 한정됩니다. 세 팔의 순차 인계는 위의 `full_tier`에서 검사합니다. 액체·모래 및 다른 로봇 배치에 대한 검증은 포함하지 않습니다.

## 수정 이력

- [`c1612f1`](https://github.com/SSUMechE/applied-programming-genesis-gtx1660-test/commit/c1612f103ee0e61080ba523619a16417e92e3d04): CPU별 미세한 역기구학 계산 차이로 고정 예시의 끝점 비교가 실패하던 문제를 수정했습니다. 제공된 시작·도착 관절값을 직접 검증하며 경로·시간과 기존 기하 검사를 유지합니다.
- [`510d41d`](https://github.com/SSUMechE/applied-programming-genesis-gtx1660-test/commit/510d41d940ff38107de266c134e233f293513ff9): GTX 1660에서 검사용 장면의 커널이 공유 메모리 66,048바이트를 요청해 65,536바이트 한도를 초과하던 문제를 수정했습니다. 기하 검사용 장면의 불필요한 제약력 계산만 제외하고 충돌 검출과 실제 운반 물리를 유지합니다. GPU 작업 오류도 최종 콘솔 요약에 표시합니다.

현재 `main`에는 두 수정이 모두 포함되어 있습니다.

## 버전과 출처

환경 버전은 `environment_reference.json`과 `requirements.txt`에 기록되어 있습니다. 이 파일들은 기존 환경을 확인하기 위한 기준입니다. 기본 실행 절차에는 `pip install`이나 새 환경 생성이 없습니다.

모델과 외부 소프트웨어의 출처는 [Third-party notices](THIRD_PARTY_NOTICES.md)에 있습니다.
