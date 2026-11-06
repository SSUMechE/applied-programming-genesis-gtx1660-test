"""Export only source-bound completed native missions to the score protocol.

Qualification runs may record a disabled candidate. Data validation never
changes that rule's qualification state or makes the result submit-eligible.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import uuid

from leaderboard.contracts import (
    canonical_bytes, digest, require, validate_evaluation_record, validate_rule,
)
from leaderboard.evaluate import NATIVE_BOUND_FIELDS, validate_native_result


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical_bytes(value) + b"\n")


def export_full_review(*, rule, plan, supervisor, native_result, native_path,
                       design_path, task_path, source_before, source_after,
                       runtime_versions, output, wall_time_provider=None):
    """Project recomputed native findings into an independently checked envelope."""
    from physics.device_tier_mission import validate_device_tier_completion
    validate_rule(rule)
    require(rule["schema_version"] == "ap-leaderboard-rule-v2",
            "RULE_VERSION", "Full native tier export requires the typed task-clock rule")
    require(supervisor.get("review_accepted") is True
            and supervisor.get("physical_transport_accepted") is True
            and supervisor.get("full_mission_passed") is True,
            "INCOMPLETE_MISSION", "Only an accepted complete resource-checked review can export", 409)
    require(native_result == json.loads(Path(native_path).read_bytes()),
            "NATIVE_ARTIFACT_CHANGED", "The actual worker result changed before export", 409)
    proof = native_result.get("device_tier_execution")
    require(validate_device_tier_completion(proof, plan_record=plan.record()) is True,
            "NATIVE_PROOF", "Complete original native phase evidence is required", 409)
    require(plan.verify_sources() is True and source_before == source_after == rule["evaluation_sources"]
            and runtime_versions == rule["runtime_versions"],
            "EVALUATION_CODE_MODIFIED", "Native execution sources or runtime do not match this rule", 409)
    contract = plan.task_contract
    terminal = {"support_id": contract["terminal_support_id"],
                "geometry": contract["terminal_support_geometry"],
                "target": contract["terminal_support_target"]}
    require(contract["difficulty"] == rule["difficulty"]
            and contract["source_task_manifest_sha256"] == rule["task_manifest_sha256"]
            and file_sha(Path(task_path) / "initialization.json") == rule["initialization_sha256"]
            and digest(terminal) == rule["terminal_target_sha256"]
            and contract["completion_limit_ticks"] == rule["max_completion_ticks"]
            and contract["required_arm_ids"] == rule["arm_ids"],
            "RULE_INPUT_MISMATCH", "Executed task, terminal or time limit differs from this rule", 409)
    from transport_challenge.three_arm_design import ThreeArmDesign
    design_path = Path(design_path).resolve(strict=True)
    planned = plan.record()
    require(planned["source_pins"].get(str(design_path)) == file_sha(design_path),
            "EVALUATED_INPUT_CHANGED", "The actual student design bytes differ from their planned source pin", 409)
    actual_design = ThreeArmDesign.from_record(json.loads(design_path.read_bytes()),
                                              json.loads((Path(task_path) / "world.json").read_bytes()))
    require(actual_design.sha256 == planned["design_sha256"],
            "EVALUATED_INPUT_CHANGED", "The actual student design changed after planning", 409)
    summary = proof.get("completion_summary")
    required_summary = {"completed_stage_ids", "completed_arm_ids", "handoffs", "donor_returns",
                        "checks", "initial_sample_ids", "final_sample_ids", "final_rest_steps",
                        "completion_ticks", "completion_clock"}
    require(type(summary) is dict and required_summary.issubset(summary),
            "NATIVE_PROOF", "Actual recomputed completion findings are incomplete", 409)
    memory = supervisor["process_memory"]
    resources = {
        "wall_s": supervisor["additive_cpu_plus_review_wall_s"],
        "sampled_peak_process_vram_bytes": int(memory["sampled_peak"]["bytes"]),
        "valid_memory_samples": memory["valid_sample_count"],
        "unknown_memory_samples": memory["unknown_or_invalid_sample_count"],
        "timed_out": supervisor["timed_out"],
        "memory_limit_stop": supervisor["observed_memory_limit_stop"],
    }
    # Input bytes contain the exact student design and immutable CPU path/times.
    # Native receipts remain output evidence, never fabricated input values.
    evaluated_input = {"schema_version": "ap-evaluated-tier-input-v1",
        "difficulty": contract["difficulty"],
        "design": json.loads(Path(design_path).read_bytes()),
        "saved_tier_mission": plan.record()}
    output = Path(output)
    require(not output.exists(), "OUTPUT_EXISTS", "Use a fresh native export directory")
    output.mkdir(parents=True)
    input_path = output / "EVALUATED_INPUT.json"
    write_json(input_path, evaluated_input)
    if wall_time_provider is not None:
        resources["wall_s"] = wall_time_provider()
    report = {"schema_version": "ap-full-mission-evaluation-v2", "run_id": str(uuid.uuid4()),
        "scope": "full_mission", "status": "PASS", "backend": "genesis_cuda",
        "initialization_mode": "original_task_initial", "input_sha256": file_sha(input_path),
        "artifact_sha256": "0" * 64, "resources": resources,
        "source_hashes_before": copy.deepcopy(source_before),
        "source_hashes_after": copy.deepcopy(source_after),
        "runtime_versions": copy.deepcopy(runtime_versions)}
    for name in ("rule_digest", "task_manifest_sha256", "initialization_sha256",
                 "physics_signature_sha256", "terminal_target_sha256", "material"):
        report[name] = rule[name]
    for name in required_summary:
        report[name] = copy.deepcopy(summary[name])
    envelope = {name: copy.deepcopy(report[name]) for name in NATIVE_BOUND_FIELDS}
    envelope.update(schema_version="ap-full-mission-native-result-v2", status="FULL_MISSION_PASS",
        full_mission_passed=native_result["full_mission_passed"],
        source_bindings_unchanged=native_result["source_bindings_unchanged"],
        failure=native_result.get("failure"), cleanup_errors=copy.deepcopy(native_result["cleanup_errors"]),
        completion_clock=copy.deepcopy(report["completion_clock"]))
    validate_native_result(report, envelope)
    result_path = output / "result.json"
    write_json(result_path, envelope)
    report["artifact_sha256"] = file_sha(result_path)
    validate_evaluation_record(rule, report)
    write_json(output / "FULL_MISSION_EVALUATION.json", report)
    audit = {"schema_version": "ap-native-tier-producer-export-v1",
        "native_result_path": str(Path(native_path).resolve()),
        "native_result_sha256": file_sha(native_path),
        "native_execution_proof_sha256": digest(proof),
        "saved_plan_sha256": plan.sha256, "saved_plan_file_sha256": plan.file_sha256,
        "rule_digest": rule["rule_digest"], "rule_qualified": rule["qualified"],
        "score_eligibility": "qualified_rule" if rule["qualified"] else "disabled_qualification_candidate",
        "submitted_to_server": False,
        "wall_scope": "CPU attribution and parent time through native proof validation and input serialization. Supervisor rechecks after export and revokes any over-budget output.",
        "scope": "Projection of validated actual full native findings. No qualification or physical PASS is inserted."}
    write_json(output / "PRODUCER_AUDIT.json", audit)
    return {"output": str(output.resolve()), "run_id": report["run_id"],
            "rule_qualified": rule["qualified"], "completion_ticks": report["completion_ticks"],
            "evaluation_sha256": file_sha(output / "FULL_MISSION_EVALUATION.json")}
