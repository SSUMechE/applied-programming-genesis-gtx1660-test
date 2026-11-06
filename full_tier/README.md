# Applied Programming — Full Mission Benchmark

This folder measures a complete supplied four-bead transport mission on your computer. Easy uses one arm, Medium uses two, and Hard uses three. All three native robot entities stay in one persistent scene. One actor moves at a time, and the same cup and beads pass through the handoffs.

Use the existing Anaconda Prompt and `applied-programming-cuda` environment. No installation or new environment is required. Start in the existing repository root, then run:

```text
git pull --ff-only
cd full_tier
conda activate applied-programming-cuda
python -B check_environment.py --cuda --output output/environment.json
```

The fixed physics is unchanged: four original beads, `dt = 0.0025 s`, and 16 internal substeps. The supplied design speed scales are Easy 0.75, Medium 0.5, and Hard 0.5. These are route-design inputs.

## Run one difficulty

Run the CPU command first. Continue only after it exits successfully and prints `CPU_GEOMETRIC_TIER_MISSION_PLANNED`. It validates the supplied fixed route and creates a fresh local source-bound bundle. This CPU time measures validation and export, not RRT search.

Then run the matching GPU command. It executes every approach, carry, support, release, handoff and final-rest check from the original initial state. Run one difficulty at a time. Some phases can remain quiet for several minutes.

Use output directories that do not already exist. For another attempt, choose fresh CPU and GPU output names and update `--saved-tier` to the new CPU output. After a code update, repeat CPU export before GPU execution.

### Easy

```text
python -B -m physics.plan_student_transport --scope tier-mission --difficulty easy --task cases/difficulty_20261106/easy_open_v1 --material beads --design example/easy_design.json --strategy supplied_support.fixed_tier_paths --internal-teacher-strategy --post-place-policy kinematic_return --wall-limit-s 300 --output output/cpu_easy
python -B -m physics.review_tier_transport --saved-tier output/cpu_easy/saved_tier_mission.json --difficulty easy --task cases/difficulty_20261106/easy_open_v1 --design example/easy_design.json --rule leaderboard/rules/portable_candidate_n0/ap2026-easy-beads-portable-candidate-n0.json --wall-limit-s 1800 --vram-limit-gb 5 --output output/gpu_easy
```

### Medium

```text
python -B -m physics.plan_student_transport --scope tier-mission --difficulty medium --task cases/difficulty_20261106/medium_staggered_v1 --material beads --design example/medium_design.json --strategy supplied_support.fixed_tier_paths --internal-teacher-strategy --post-place-policy kinematic_return --wall-limit-s 300 --output output/cpu_medium
python -B -m physics.review_tier_transport --saved-tier output/cpu_medium/saved_tier_mission.json --difficulty medium --task cases/difficulty_20261106/medium_staggered_v1 --design example/medium_design.json --rule leaderboard/rules/portable_candidate_n0/ap2026-medium-beads-portable-candidate-n0.json --wall-limit-s 3600 --vram-limit-gb 5 --output output/gpu_medium
```

### Hard

```text
python -B -m physics.plan_student_transport --scope tier-mission --difficulty hard --task cases/difficulty_20261106/hard_industrial_v1 --material beads --design example/hard_design.json --strategy supplied_support.fixed_tier_paths --internal-teacher-strategy --post-place-policy kinematic_return --wall-limit-s 300 --output output/cpu_hard
python -B -m physics.review_tier_transport --saved-tier output/cpu_hard/saved_tier_mission.json --difficulty hard --task cases/difficulty_20261106/hard_industrial_v1 --design example/hard_design.json --rule leaderboard/rules/portable_candidate_n0/ap2026-hard-beads-portable-candidate-n0.json --wall-limit-s 5400 --vram-limit-gb 5 --output output/gpu_hard
```

## Read your result

After the GPU command exits, read its `REVIEW_SUPERVISOR.json`. This example uses Medium. Change the path for Easy, Hard or a new output name.

```text
python -B -c "import json;from pathlib import Path;r=json.loads(Path('output/gpu_medium/REVIEW_SUPERVISOR.json').read_text(encoding='utf-8'));print(json.dumps({'difficulty':r['difficulty'],'status':'ACCEPTED' if r['review_accepted'] else 'NOT_ACCEPTED','full_mission_passed':r['full_mission_passed'],'outcome':r.get('outcome'),'worker_failure':r.get('worker_failure'),'CPU_plus_GPU_wall_s':r['additive_cpu_plus_review_wall_s'],'sampled_process_GB':((r.get('process_memory') or {}).get('sampled_peak') or {}).get('GB')},indent=2))"
```

A completed accepted result has `status: ACCEPTED` and `full_mission_passed: true`. A null memory value means the sample was unavailable, not zero. Recorded host GPU metadata is separate from owned-process memory:

```text
python -B -c "import json;from pathlib import Path;r=json.loads(Path('output/gpu_medium/REVIEW_SUPERVISOR.json').read_text(encoding='utf-8'));print(json.dumps(r['hardware_before'],indent=2))"
```

These commands read local files. Output files are not uploaded automatically.

## Clock and resource limits

Accepted task time starts at native tick 200, immediately after the fixed initial 0.5 s stabilization. It ends at accepted terminal rest. All subsequent physical holds and rests count. Required donor reverse returns advance zero simulation ticks. Their execution still contributes to wall time. There is no final return.

| Difficulty | Arms / handoffs | Task deadline | Complete wall-time ceiling |
| --- | --- | --- | --- |
| Easy | 1 / 0 | 40 s | 1,800 s |
| Medium | 2 / 1 | 85 s | 3,600 s |
| Hard | 3 / 2 | 105 s | 5,400 s |

The wall-time ceiling includes the fresh CPU export and whole GPU review process. The process-memory monitor samples owned-process usage against 5 GB. Sampling does not establish a hard allocation cap.

## Recorded reference runs

Measured hardware: **NVIDIA GeForce RTX 4090**. GTX 1660 full-mission cost remains unmeasured. Run the same commands on that computer to obtain its own complete cost.

| Difficulty | Current-package result | Accepted task time | CPU + GPU wall time | Sampled process peak |
| --- | --- | --- | --- | --- |
| Easy | Full mission accepted and independently audited | 18.4425 s | 12 min 25.179 s | 3.338965 GB |
| Medium | Full mission accepted and independently audited | 56.7625 s | 40 min 22.156 s | 3.512832 GB |
| Hard | Full mission accepted and independently audited | 72.6775 s | 52 min 8.380 s | 3.659829248 GB |

The reference costs include provided-route CPU validation and the entire supervised GPU process. The CPU exports were run concurrently, and unrelated background GPU work was left untouched. These are individual observed runs, not an isolated throughput comparison or a GTX 1660 prediction.

Recorded native plots: [Easy](benchmarks/easy_recorded_outcome.png), [Medium](benchmarks/medium_recorded_outcome.png), [Hard](benchmarks/hard_recorded_outcome.png). Exact numeric summaries and evidence hashes are in [benchmark results](benchmarks/results.json).

The supplied rules remain disabled for competition. Hardware-example completion does not automatically qualify other paths, designs or machines for scoring.
