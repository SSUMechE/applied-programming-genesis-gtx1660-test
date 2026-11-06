"""Supervise an unchanged CPU-planned full bead mission on the actual GPU.

Only complete native evidence can pass. Partial runs, missing memory samples,
source drift and elapsed resource failures remain explicit failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LIMIT_SECONDS = {"easy": 40, "medium": 85, "hard": 105}
MEASUREMENT_WALL_LIMITS = {"easy": 1800., "medium": 3600., "hard": 5400.}


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_record(path):
    if not Path(path).is_file():
        return {}
    try:
        value = json.loads(Path(path).read_bytes())
        if type(value) is not dict:
            raise ValueError("The record must be a JSON object")
        return value
    except (OSError, ValueError) as exc:
        return {"record_read_error": type(exc).__name__ + ": " + str(exc)}


def hardware_snapshot():
    """Read driver metadata without creating a parent CUDA context."""
    command = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return {"command": command, "exit_code": result.returncode,
                "rows": result.stdout.strip().splitlines(), "error": result.stderr.strip(),
                "scope": "Device metadata only. Device total memory is not owned-process memory."}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "error": type(exc).__name__ + ": " + str(exc), "rows": []}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--saved-tier", type=Path, required=True)
    p.add_argument("--difficulty", choices=tuple(LIMIT_SECONDS), required=True)
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--design", type=Path, default=ROOT / "student/three_arm_design.json")
    p.add_argument("--rule", type=Path, help="Frozen full-tier rule for native evidence export. Disabled candidates remain ineligible for scoring.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wall-limit-s", type=float, help="Measurement guard, default 1800/3600/5400 seconds for one/two/three arms. Not an estimated runtime.")
    p.add_argument("--vram-limit-gb", type=float, default=5.)
    p.add_argument("--memory-interval-s", type=float, default=.5)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--gpu-wall-allowance-s", type=float, help=argparse.SUPPRESS)
    return p


def run_worker(args, p):
    if args.gpu_wall_allowance_s is None or args.gpu_wall_allowance_s <= 10:
        p.error("Worker requires its supervised remaining wall allowance")
    from physics import run_student_three_arm as entry
    from physics.review_student_transport import block_online_planning
    old = sys.argv
    sys.argv = [str(ROOT / "physics/run_student_three_arm.py"),
        "--mode", "tier_mission", "--saved-tier", str(args.saved_tier.resolve()),
        "--difficulty", args.difficulty, "--task", str(args.task.resolve()),
        "--design", str(args.design.resolve()), "--output", str(args.output),
        "--contents", "beads", "--grasp-policy", "automatic",
        "--query-fixture-readback", "batched", "--step-limit", "0",
        "--wall-limit-s", str(args.gpu_wall_allowance_s),
        "--rigid16-readback", "native", "--trace-mode", "compact",
        "--execution-backend", "device"]
    with block_online_planning() as calls:
        try:
            return entry.main()
        finally:
            sys.argv = old
            args.output.mkdir(parents=True, exist_ok=True)
            save(args.output / "ONLINE_CALL_GUARD.json", {
                "blocked_online_calls": calls,
                "scope": "Actual full-mission worker calls. Any online planning or IK attempt rejects the run."})


def assess_native(result, plan, calls, *, worker_exit_code, timed_out, sources_current):
    """Check the actual module's typed evidence, never manufacture phase flags."""
    reasons = []
    guarded = (set(calls) == {"plan_student_motion", "plan_joint_motion", "ik"}
               and all(type(v) is int and v == 0 for v in calls.values()))
    required = {
        "worker_exit": worker_exit_code == 0 and not timed_out,
        "status": result.get("status") == "TIER_MISSION_DEVICE_TRANSPORT_SAMPLED_PASS",
        "full_native_mission": result.get("full_mission_passed") is True,
        "sources": sources_current and result.get("source_bindings_unchanged") is True,
        "saved_bundle": result.get("saved_bundle_sha256") == plan.sha256,
        "saved_file": result.get("saved_plan_file_sha256") == plan.file_sha256,
        "online_call_guard": guarded,
        "cleanup": result.get("cleanup_errors") == [],
        "runtime_failure": result.get("failure") is None,
    }
    reasons.extend(name for name, passed in required.items() if not passed)
    evidence_valid = False
    evidence_error = None
    if not reasons:
        try:
            from physics.device_tier_mission import validate_device_tier_completion
            evidence_valid = validate_device_tier_completion(
                result.get("device_tier_execution"), plan_record=plan.record()) is True
        except Exception as exc:
            evidence_error = type(exc).__name__ + ": " + str(exc)
        if not evidence_valid:
            reasons.append("native_completion_evidence")
    return not reasons, {"failed_checks": reasons, "evidence_valid": evidence_valid,
                         "evidence_error": evidence_error}


def classify_outcome(report):
    """Describe the recorded outcome without changing any acceptance decision."""
    if report.get("review_accepted") is True:
        return "ACCEPTED_FULL_MISSION"
    if report.get("source_bindings_unchanged") is not True:
        return "SOURCE_BINDING_FAILED"
    if report.get("timed_out") or report.get("observed_additive_cost_within_budget") is False:
        return "WALL_BUDGET_EXCEEDED"
    memory = report.get("process_memory", {})
    peak = (memory.get("sampled_peak") or {}).get("bytes")
    limit = (memory.get("requested_limit") or {}).get("bytes")
    if (report.get("observed_memory_limit_stop")
            or (type(peak) in (int, float) and type(limit) in (int, float) and peak > limit)):
        return "VRAM_BUDGET_EXCEEDED"
    failure = report.get("worker_failure")
    if isinstance(failure, dict) and failure.get("type") == "PhysicalTransportRejected":
        return "PHYSICAL_REJECTED"
    if failure or report.get("worker_result_read_error") or report.get("exit_code") != 0:
        return "SIMULATION_FAILED"
    if report.get("worker_cleanup_errors"):
        return "CLEANUP_FAILED"
    if report.get("score_export_error"):
        return "RESULT_EXPORT_FAILED"
    if report.get("observed_sampled_process_vram_within_budget") is not True:
        return "MEMORY_MEASUREMENT_INCOMPLETE"
    return "INCOMPLETE_NATIVE_EVIDENCE"


def main(argv=None):
    started = time.perf_counter()
    p = parser()
    args = p.parse_args(argv)
    if args.wall_limit_s is None:
        args.wall_limit_s = MEASUREMENT_WALL_LIMITS[args.difficulty]
    if not 10 <= args.wall_limit_s <= MEASUREMENT_WALL_LIMITS[args.difficulty]:
        p.error("Use a positive review budget within this tier's declared measurement guard")
    if not 0 < args.vram_limit_gb <= 5 or not .1 <= args.memory_interval_s <= 3:
        p.error("Use an observed process limit at most 5 GB and sampling 0.1..3 seconds")
    from physics.tier_mission_plan import TierMissionPlan
    plan = TierMissionPlan.from_file(args.saved_tier)
    record = plan.record()
    if plan.task_contract["difficulty"] != args.difficulty:
        p.error("Difficulty differs from the immutable CPU mission")
    args.output = args.output.resolve()
    if args.worker:
        return run_worker(args, p)
    if os.name != "nt":
        p.error("This resource supervisor requires Windows exact owned-process monitoring")
    cpu_wall = record["cpu_planning_result"]["process_wall_s"]
    if type(cpu_wall) not in (int, float) or not 0 < cpu_wall < args.wall_limit_s - 30:
        p.error("The attributed CPU planning has exhausted this complete review budget")
    memory_path = args.output.with_suffix(".memory_samples.jsonl")
    worker_log = args.output.with_suffix(".worker.log")
    if args.output.exists() or worker_log.exists() or memory_path.exists():
        p.error("Use a fresh review directory, worker log and memory sample file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rule = None
    rule_sources_before = None
    versions = None
    if args.rule is not None:
        from leaderboard.contracts import load_rule, source_snapshot, runtime_snapshot
        rule = load_rule(args.rule)
        if rule["difficulty"] != args.difficulty:
            p.error("Rule difficulty differs from the saved mission")
        rule_sources_before = source_snapshot(rule, ROOT)
        versions = runtime_snapshot(rule)
    hardware = hardware_snapshot()
    cleanup_reserve = 20.
    worker_allowance = args.wall_limit_s - cpu_wall - cleanup_reserve
    command = [sys.executable, "-B", "-m", "physics.review_tier_transport", "--worker",
        "--saved-tier", str(args.saved_tier.resolve()), "--difficulty", args.difficulty,
        "--task", str(args.task.resolve()), "--design", str(args.design.resolve()),
        "--output", str(args.output), "--wall-limit-s", str(args.wall_limit_s),
        "--vram-limit-gb", str(args.vram_limit_gb),
        "--gpu-wall-allowance-s", str(worker_allowance)]
    from physics.review_process_memory import (
        ProcessMemorySampler, creation_identity, kill_owned_tree, wait_owned_worker)
    with worker_log.open("xb") as stream:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=stream,
            stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        identity = None
        try:
            identity = creation_identity(process)
            memory = ProcessMemorySampler(process, memory_path, started=started,
                interval_s=args.memory_interval_s, limit_gb=args.vram_limit_gb)
            memory.start()
        except BaseException:
            if identity is not None:
                kill_owned_tree(process, identity)
            else:
                process.kill()
                process.wait(timeout=3.)
            raise
        print(json.dumps({"event": "Applied Programming full mission worker started",
              "difficulty": args.difficulty, "owned_worker": identity,
              "output": str(args.output)}), flush=True)
        waited = None
        termination = None
        try:
            waited = wait_owned_worker(process, memory, identity,
                deadline=started + worker_allowance)
        finally:
            if process.poll() is None:
                termination = kill_owned_tree(process, identity)
            memory_receipt = memory.finish()
    args.output.mkdir(parents=True, exist_ok=True)
    result = read_record(args.output / "result.json")
    calls = read_record(args.output / "ONLINE_CALL_GUARD.json").get("blocked_online_calls", {})
    try:
        sources_current = plan.verify_sources() is True
    except Exception:
        sources_current = False
    rule_sources_after = None
    if rule is not None:
        try:
            rule_sources_after = source_snapshot(rule, ROOT)
            sources_current = bool(sources_current and rule_sources_after == rule_sources_before
                                   and runtime_snapshot(rule) == versions)
        except Exception:
            sources_current = False
    physical, validation = assess_native(result, plan, calls,
        worker_exit_code=waited["exit_code"], timed_out=waited["timed_out"],
        sources_current=sources_current)
    save(args.output / "REVIEW_PROCESS_MEMORY.json", memory_receipt)
    wall = time.perf_counter() - started
    budget_ok = cpu_wall + wall <= args.wall_limit_s
    memory_ok = memory_receipt.get("observed_sampled_peak_within_limit") is True
    accepted = (physical and budget_ok and memory_ok and not waited["memory_limit_stop"])
    report = {
        "schema_version": "student-full-tier-review-supervisor-v1",
        "application_name": "Applied Programming", "scope": "full_mission",
        "difficulty": args.difficulty, "execution_backend": "device",
        "run_directory": str(args.output), "worker_command": command,
        "robot_dynamics_model": "persistent_three_native_arms_serial_actor",
        "owned_worker": identity, "exit_code": waited["exit_code"],
        "timed_out": waited["timed_out"], "observed_memory_limit_stop": waited["memory_limit_stop"],
        "owned_tree_termination": termination or waited["termination"],
        "physical_transport_accepted": physical, "review_accepted": bool(accepted),
        "worker_status": result.get("status"), "worker_failure": result.get("failure"),
        "worker_result_read_error": result.get("record_read_error"),
        "worker_cleanup_errors": result.get("cleanup_errors"),
        "physical_steps": result.get("physical_steps"),
        "native_evidence_validation": validation, "blocked_online_calls": calls,
        "source_bindings_unchanged": sources_current,
        "saved_bundle_sha256": plan.sha256, "saved_file_sha256": plan.file_sha256,
        "gpu_review_process_wall_s": wall, "attributed_cpu_planning_wall_s": cpu_wall,
        "additive_cpu_plus_review_wall_s": cpu_wall + wall,
        "requested_complete_budget_s": args.wall_limit_s,
        "observed_additive_cost_within_budget": budget_ok,
        "observed_sampled_process_vram_within_budget": memory_ok,
        "process_vram_hard_cap_proven": False, "process_memory": memory_receipt,
        "hardware_before": hardware, "cleanup_reserve_s": cleanup_reserve,
        "timing_scope": "Parent precheck through worker/monitor exit and resource receipt write. Separate source-bound CPU planning attribution.",
        "full_mission_passed": bool(accepted), "student_release_approved": False,
    }
    if rule is not None:
        source_record = {"rule_path": str(args.rule.resolve()), "rule_digest": rule["rule_digest"],
            "source_hashes_before": rule_sources_before, "source_hashes_after": rule_sources_after,
            "runtime_versions": versions, "sources_match": sources_current}
        save(args.output / "REVIEW_SOURCE_BINDINGS.json", source_record)
        report["rule_digest"] = rule["rule_digest"]
        report["rule_qualified"] = rule["qualified"]
        if accepted:
            try:
                from physics.full_tier_result import export_full_review
                report["score_export"] = export_full_review(rule=rule, plan=plan, supervisor=report,
                    native_result=result, native_path=args.output / "result.json",
                    design_path=args.design, task_path=args.task,
                    source_before=rule_sources_before, source_after=rule_sources_after,
                    runtime_versions=versions, output=args.output / "score_evidence",
                    wall_time_provider=lambda: cpu_wall + time.perf_counter() - started)
            except Exception as exc:
                report["score_export_error"] = type(exc).__name__ + ": " + str(exc)
                report["review_accepted"] = False
                if (args.output / "score_evidence").exists():
                    save(args.output / "score_evidence/EXPORT_REVOKED.json", {
                        "reason": "Native evidence export did not close successfully",
                        "error": report["score_export_error"]})
    final_wall = time.perf_counter() - started
    final_cost = cpu_wall + final_wall
    report.update(gpu_review_process_wall_s=final_wall, additive_cpu_plus_review_wall_s=final_cost,
                  observed_additive_cost_within_budget=final_cost <= args.wall_limit_s,
                  timing_scope="Parent startup through native validation and optional score export, before final supervisor serialization. Separate source-bound CPU planning attribution.")
    if final_cost > args.wall_limit_s:
        report["review_accepted"] = False
        report["full_mission_passed"] = False
        if ((args.output / "score_evidence").exists()
                and not (args.output / "score_evidence/EXPORT_REVOKED.json").exists()):
            save(args.output / "score_evidence/EXPORT_REVOKED.json", {
                "reason": "Complete review including export exceeded its measured wall guard",
                "additive_cpu_plus_review_wall_s": final_cost, "limit_s": args.wall_limit_s})
    report["outcome"] = classify_outcome(report)
    save(args.output / "REVIEW_SUPERVISOR.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["review_accepted"] else 2 if physical else 1


if __name__ == "__main__":
    raise SystemExit(main())
