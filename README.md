# Applied Programming — GTX 1660 Test

기존 **`applied-programming-cuda` Conda 환경**에서 실행하는 GPU 시험 예시입니다. 제공된 경로와 구간별 이동 시간이 완성되어 있으므로 코드를 수정하지 않고 실행할 수 있습니다.

하드웨어 비교용 예시는 시작·도착 관절값도 고정 입력으로 사용합니다. PC마다 역기구학으로 다시 계산하지 않으며, 제공된 배치에서 자세·관절 제한·충돌을 검증한 뒤 전체 경로를 검사합니다.

한 로봇암이 용기와 구슬 4개를 운반합니다. 다른 두 팔은 정지한 충돌 형상으로 유지합니다. 물리 시간 간격은 `0.0025 s`, 내부 계산은 16단계입니다. 잡기·놓기는 제공된 절차를 사용하며, 운반·충돌·시료 유지·최종 놓기와 정지를 검사합니다.

**지원 실행 방식:** Windows 10/11의 Anaconda Prompt. 화면을 띄우지 않고 결과 파일을 저장하는 시험입니다. GTX 1660의 실행 시간은 해당 컴퓨터에서 실제로 측정해야 합니다. 다른 GPU로도 실행할 수 있으며 실제 GPU 정보가 기록됩니다.

## 실행

Anaconda Prompt에서 저장소를 내려받고 기존 환경을 활성화합니다.

```text
git clone https://github.com/SSUMechE/applied-programming-genesis-gtx1660-test.git
cd applied-programming-genesis-gtx1660-test
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

시간 제한 1,800초에는 앞서 측정한 CPU 검증 비용이 포함됩니다. VRAM 제한은 5 GB입니다. VRAM은 일정 간격으로 관측하므로 기록된 값은 표본 최대치이며, 할당량을 강제로 제한하는 기능은 아닙니다.

다른 컴퓨터의 측정 결과를 비교하려면 이 JSON과 환경 검사 결과를 보관하면 됩니다. 저장소에 업로드할 필요는 없습니다. `output/`은 Git 추적에서 제외되어 있습니다.

## 다시 실행하거나 경로를 바꿀 때

기존 결과를 덮어쓰지 않으므로 새 출력 이름을 사용합니다. 환경 검사를 다시 기록하려면 `output/environment_2.json`처럼 새 파일을 지정합니다. CPU 명령의 출력은 `output/cpu_example_2`로 바꾸고, GPU 명령의 입력도 그 폴더의 `saved_one_carry.json`으로 맞춥니다. GPU 출력은 `output/gpu_example_2`로 지정합니다.

저장된 실행 입력은 생성한 컴퓨터의 경로와 소스에 연결됩니다. 다른 컴퓨터로 옮겼거나 실행 코드를 변경했다면 CPU 명령부터 다시 실행합니다.

기존 복제본의 코드를 업데이트할 때는 저장소 폴더에서 `git pull --ff-only`를 실행합니다. 업데이트 뒤에는 새 출력 이름으로 CPU 명령부터 실행합니다. 이전 버전의 `The fixed example requires its supplied layout and endpoints` 오류는 고정 예시의 시작·도착 자세를 다시 계산하던 절차에서 발생할 수 있습니다. 수정 버전은 제공된 관절값을 그대로 검증합니다. CPU 단계가 실패하면 콘솔의 `failure`와 `worker_log`에 원인이 표시됩니다.

기준 시험은 제공된 경로·시간·물리 설정을 그대로 사용합니다. 경로를 변경하는 실험은 기준 측정을 마친 뒤 별도로 진행합니다. 이 저장소의 예시는 첫 번째 팔의 한 운반 구간에 한정됩니다. 세 팔의 순차 인계, 액체·모래 및 다른 로봇 배치에 대한 검증은 포함하지 않습니다.

## 버전과 출처

환경 버전은 `environment_reference.json`과 `requirements.txt`에 기록되어 있습니다. 이 파일들은 기존 환경을 확인하기 위한 기준입니다. 기본 실행 절차에는 `pip install`이나 새 환경 생성이 없습니다.

모델과 외부 소프트웨어의 출처는 [Third-party notices](THIRD_PARTY_NOTICES.md)에 있습니다.
