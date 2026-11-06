"""Strict, data-only contracts for student-run full-mission submissions.

Source matching enforces the published no-modification policy. It is not remote
attestation of execution on a student-controlled computer.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import uuid


class ProtocolError(ValueError):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.status = status


def require(condition, code, message, status=400):
    if not condition:
        raise ProtocolError(code, message, status)


def canonical_bytes(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ProtocolError("INVALID_JSON", "Use finite JSON data only.") from exc


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def decode_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "DUPLICATE_KEY", "Duplicate JSON keys are not accepted.")
            result[key] = value
        return result
    def nonfinite(value):
        raise ProtocolError("NONFINITE_VALUE", "NaN and Infinity are not accepted.")
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError("INVALID_JSON", "The request must contain valid JSON.") from exc


def exact_keys(value, keys, label):
    require(type(value) is dict and set(value) == set(keys), "INVALID_SCHEMA",
            label + " has missing or unexpected fields.")


def sha(value, label):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value),
            "INVALID_HASH", label + " must be a SHA-256 hex digest.")


def integer(value, minimum, maximum, label):
    require(type(value) is int and minimum <= value <= maximum, "INVALID_NUMBER",
            label + " must be an integer in the allowed range.")


def finite(value, minimum, maximum, label):
    require(type(value) in (int, float) and math.isfinite(value)
            and minimum <= value <= maximum, "INVALID_NUMBER",
            label + " must be a finite number in the allowed range.")


def safe_relative(name):
    require(type(name) is str and 0 < len(name) <= 240 and "\\" not in name
            and ":" not in name and "\x00" not in name, "INVALID_SOURCE_PATH",
            "Source inventory paths must be relative POSIX paths.")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts
            and path.as_posix() == name and name != ".", "INVALID_SOURCE_PATH",
            "Source inventory paths must remain inside the project.")
    require(path.parts[0] not in ("student", "output", ".git"),
            "STUDENT_INPUT_PINNED", "Do not freeze student edits or generated outputs as supplied code.")
    return path


RULE_FIELDS = {
    "schema_version", "rule_id", "rule_digest", "difficulty", "material",
    "qualified", "qualification_reason", "test_only", "dt_s", "arm_ids",
    "stages", "handoffs", "donor_return_arm_ids", "task_manifest_sha256",
    "initialization_sha256", "physics_signature_sha256", "terminal_target_sha256",
    "sample_count", "minimum_final_rest_steps", "max_wall_s", "max_vram_bytes",
    "evaluation_sources", "runtime_versions",
}
DEADLINE_RULE_FIELDS = RULE_FIELDS | {"completion_clock", "max_completion_ticks"}


def completion_clock_contract():
    """The fixed bead-task clock. This does not qualify a native producer."""
    return {
        "schema_version": "ap-first-motion-final-rest-clock-v1",
        "native_initial_step": 0,
        "initial_stabilization_ticks": 200,
        "first_motion_start_step": 200,
        "end_event": "first_accepted_final_released_rest_post_step",
        "physical_waits_included": True,
        "donor_return_simulation_ticks": 0,
        "final_return_included": False,
    }


def validate_rule(rule):
    version = rule.get("schema_version") if type(rule) is dict else None
    require(version in ("ap-leaderboard-rule-v1", "ap-leaderboard-rule-v2"),
            "RULE_VERSION", "Unknown rule format.")
    deadline_rule = version == "ap-leaderboard-rule-v2"
    exact_keys(rule, DEADLINE_RULE_FIELDS if deadline_rule else RULE_FIELDS, "Rule")
    require(type(rule["rule_id"]) is str and re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,79}", rule["rule_id"]),
            "RULE_ID", "Use a stable lowercase rule identifier.")
    require(rule["difficulty"] in ("easy", "medium", "hard"), "DIFFICULTY", "Unknown difficulty.")
    require(rule["material"] in ("beads", "process_liquid", "sand"), "MATERIAL", "Unknown material.")
    require(type(rule["qualified"]) is bool and type(rule["test_only"]) is bool,
            "INVALID_SCHEMA", "Qualification flags must be booleans.")
    require(type(rule["qualification_reason"]) is str and 0 < len(rule["qualification_reason"]) <= 1000,
            "INVALID_SCHEMA", "A qualification explanation is required.")
    finite(rule["dt_s"], 0.000001, 1, "dt_s")
    count = {"easy": 1, "medium": 2, "hard": 3}[rule["difficulty"]]
    arms = ["arm_" + str(i) for i in range(count)]
    require(rule["arm_ids"] == arms, "ARM_CONTRACT", "The tier requires its ordered participating arms.")
    expected = [{"stage_id": arm + "__" + kind, "arm_id": arm, "kind": kind}
                for arm in arms for kind in ("approach", "carry")]
    require(rule["stages"] == expected, "STAGE_CONTRACT", "Each arm requires approach then carry.")
    expected_handoffs = [{"from_arm": arms[i], "to_arm": arms[i + 1],
                         "support_id": "relay_" + str(i + 1) + "_support"}
                        for i in range(count - 1)]
    require(rule["handoffs"] == expected_handoffs, "HANDOFF_CONTRACT", "Required handoff order differs.")
    require(rule["donor_return_arm_ids"] == arms[:-1], "RETURN_CONTRACT",
            "Only intermediate donors return along their own paths.")
    for name in ("task_manifest_sha256", "initialization_sha256", "physics_signature_sha256",
                 "terminal_target_sha256", "rule_digest"):
        sha(rule[name], name)
    integer(rule["sample_count"], 1, 1000000, "sample_count")
    integer(rule["minimum_final_rest_steps"], 1, 10000000, "minimum_final_rest_steps")
    if deadline_rule:
        require(rule["material"] == "beads" and rule["dt_s"] == 0.0025,
                "CLOCK_PROFILE", "This deadline clock uses the unchanged bead task and 0.0025-second dt.")
        require(canonical_bytes(rule["completion_clock"]) == canonical_bytes(completion_clock_contract()),
                "CLOCK_CONTRACT", "The fixed initial stabilization and transport clock cannot be changed.")
        integer(rule["max_completion_ticks"], rule["minimum_final_rest_steps"] + len(rule["stages"]),
                1000000000, "max_completion_ticks")
    finite(rule["max_wall_s"], 1, 86400, "max_wall_s")
    integer(rule["max_vram_bytes"], 1, 5000000000, "max_vram_bytes")
    sources = rule["evaluation_sources"]
    require(type(sources) is dict and 1 <= len(sources) <= 10000, "SOURCE_INVENTORY", "Supplied sources are required.")
    for name, expected_hash in sources.items():
        safe_relative(name)
        sha(expected_hash, name)
    versions = rule["runtime_versions"]
    require(type(versions) is dict and 1 <= len(versions) <= 30, "RUNTIME_VERSIONS", "Runtime versions are required.")
    for name, version in versions.items():
        require(type(name) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name)
                and type(version) is str and 0 < len(version) <= 100,
                "RUNTIME_VERSIONS", "Use package names and explicit version strings.")
    require(rule["rule_digest"] == digest({k: v for k, v in rule.items() if k != "rule_digest"}),
            "RULE_DIGEST_MISMATCH", "The frozen rule was changed.", 409)
    return rule


def load_rule(path):
    return validate_rule(decode_json(Path(path).read_bytes()))


def source_snapshot(rule, project_root):
    validate_rule(rule)
    root = Path(project_root).resolve(strict=True)
    result = {}
    for name, expected in rule["evaluation_sources"].items():
        path = root.joinpath(*safe_relative(name).parts).resolve()
        require(path.is_relative_to(root) and path.is_file(), "SOURCE_MISSING",
                "A required supplied evaluation file is missing: " + name, 409)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, "EVALUATION_CODE_MODIFIED",
                "Provided evaluation code or task data was changed. This result is ineligible: " + name, 409)
        result[name] = actual
    return result


def runtime_snapshot(rule):
    result = {}
    for name in rule["runtime_versions"]:
        try:
            result[name] = (f"{sys.version_info.major}.{sys.version_info.minor}"
                            if name == "python_major_minor" else importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProtocolError("RUNTIME_MISSING", "Required runtime package is unavailable: " + name, 409) from exc
    require(result == rule["runtime_versions"], "RUNTIME_CHANGED",
            "The evaluator runtime differs from this rule version.", 409)
    return result


CHECKS = {
    "collision_free", "sample_retained", "tracking_passed", "finite_state",
    "handling_passed", "supported_release", "final_rest", "original_initial_state",
    "immutable_terminal", "complete_native_lineage", "all_required_phases",
}
EVALUATION_FIELDS = {
    "schema_version", "run_id", "scope", "status", "backend", "initialization_mode",
    "rule_digest", "task_manifest_sha256", "initialization_sha256",
    "physics_signature_sha256", "terminal_target_sha256", "material",
    "input_sha256", "artifact_sha256", "completed_stage_ids", "completed_arm_ids",
    "handoffs", "donor_returns", "checks", "initial_sample_ids", "final_sample_ids",
    "final_rest_steps", "completion_ticks", "resources",
    "source_hashes_before", "source_hashes_after", "runtime_versions",
}
DEADLINE_EVALUATION_FIELDS = EVALUATION_FIELDS | {"completion_clock"}


def validate_completion_clock(rule, report):
    """Check elapsed native ticks, never a client-selected stabilization offset.

    A future qualified producer must bind these endpoints to its original
    physical lineage. This data contract does not synthesize that lineage.
    """
    clock = report["completion_clock"]
    exact_keys(clock, {"schema_version", "native_initial_step", "first_motion_start_step",
                       "final_rest_end_step", "native_end_step"}, "Completion clock")
    require(clock["schema_version"] == "ap-first-motion-final-rest-observation-v1",
            "CLOCK_VERSION", "Unknown completion clock observation.")
    for name in ("native_initial_step", "first_motion_start_step", "final_rest_end_step", "native_end_step"):
        integer(clock[name], 0, 1000000200, name)
    fixed = rule["completion_clock"]
    require(clock["native_initial_step"] == fixed["native_initial_step"]
            and clock["first_motion_start_step"] == fixed["first_motion_start_step"],
            "CLOCK_START", "Exclude exactly the supplied 200 INITIAL stabilization ticks.", 409)
    end = clock["final_rest_end_step"]
    start = clock["first_motion_start_step"]
    require(end == clock["native_end_step"] and end > start
            and report["completion_ticks"] == end - start,
            "INCONSISTENT_CLOCK", "Completion ticks must equal the final accepted native step minus the fixed motion start.", 409)
    require(report["completion_ticks"] <= rule["max_completion_ticks"],
            "TRANSPORT_DEADLINE_EXCEEDED", "The completed transport exceeds this tier's simulated-time limit.", 409)


def validate_evaluation(rule, report):
    validate_rule(rule)
    require(rule["qualified"], "RULE_NOT_QUALIFIED", rule["qualification_reason"], 409)
    return validate_evaluation_record(rule, report)


def validate_evaluation_record(rule, report):
    """Validate complete evidence data without granting scoring eligibility.

    Native qualification runs may bind disabled rules. The public submission
    path always calls validate_evaluation and retains its qualification gate.
    """
    validate_rule(rule)
    deadline_rule = rule["schema_version"] == "ap-leaderboard-rule-v2"
    exact_keys(report, DEADLINE_EVALUATION_FIELDS if deadline_rule else EVALUATION_FIELDS, "Full-mission evaluation")
    expected_version = "ap-full-mission-evaluation-v2" if deadline_rule else "ap-full-mission-evaluation-v1"
    require(report["schema_version"] == expected_version, "EVALUATION_VERSION", "Evaluation format does not match this rule version.")
    try:
        require(type(report["run_id"]) is str and str(uuid.UUID(report["run_id"])) == report["run_id"],
                "RUN_ID", "Use a canonical UUID run identifier.")
    except (ValueError, AttributeError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError("RUN_ID", "Use a canonical UUID run identifier.") from exc
    require(report["scope"] == "full_mission" and report["status"] == "PASS",
            "INCOMPLETE_MISSION", "Only a completed full-mission PASS is eligible.", 409)
    require(report["backend"] == "genesis_cuda"
            and report["initialization_mode"] == "original_task_initial",
            "INITIAL_OR_BACKEND", "Use the frozen original INITIAL and CUDA native evaluation.", 409)
    for name in ("rule_digest", "task_manifest_sha256", "initialization_sha256",
                 "physics_signature_sha256", "terminal_target_sha256", "material"):
        require(report[name] == rule[name], "RULE_INPUT_MISMATCH",
                "Evaluation does not match the frozen " + name + ".", 409)
    for name in ("input_sha256", "artifact_sha256"):
        sha(report[name], name)
    for name in ("source_hashes_before", "source_hashes_after"):
        require(report[name] == rule["evaluation_sources"], "EVALUATION_CODE_MODIFIED",
                "Modified, missing or unapproved evaluation sources invalidate the score.", 409)
    require(report["runtime_versions"] == rule["runtime_versions"], "RUNTIME_CHANGED",
            "The runtime differs from the frozen rule.", 409)
    require(report["completed_stage_ids"] == [s["stage_id"] for s in rule["stages"]]
            and report["completed_arm_ids"] == rule["arm_ids"], "INCOMPLETE_STAGES",
            "Every required arm must complete approach and carry in order.", 409)
    expected_handoffs = [dict(h, released_supported_and_resting=True,
                              same_sample_acquired=True) for h in rule["handoffs"]]
    require(canonical_bytes(report["handoffs"]) == canonical_bytes(expected_handoffs), "INCOMPLETE_HANDOFFS",
            "Every required supported handoff must preserve the same sample.", 409)
    expected_returns = [{"arm_id": arm, "policy": "kinematic_return",
                         "original_path_reversed": True, "physical_steps": 0}
                        for arm in rule["donor_return_arm_ids"]]
    require(canonical_bytes(report["donor_returns"]) == canonical_bytes(expected_returns), "RETURN_POLICY",
            "Intermediate donors must retrace their own path without sample physics.", 409)
    exact_keys(report["checks"], CHECKS, "Physical checks")
    require(all(v is True for v in report["checks"].values()), "PHYSICAL_CHECK_FAILED",
            "All original physical and terminal checks must pass.", 409)
    ids = report["initial_sample_ids"]
    require(type(ids) is list and len(ids) == rule["sample_count"]
            and all(type(v) is str and 0 < len(v) <= 120 for v in ids)
            and len(set(ids)) == len(ids) and report["final_sample_ids"] == ids,
            "SAMPLE_IDENTITY", "The complete original sample must remain present in order.", 409)
    integer(report["final_rest_steps"], rule["minimum_final_rest_steps"], 10000000, "final_rest_steps")
    integer(report["completion_ticks"], 1, 1000000000, "completion_ticks")
    require(report["completion_ticks"] >= report["final_rest_steps"] + len(rule["stages"]),
            "INCONSISTENT_CLOCK", "The completion clock cannot be shorter than final rest and required motion stages.", 409)
    if deadline_rule:
        validate_completion_clock(rule, report)
    resources = report["resources"]
    exact_keys(resources, {"wall_s", "sampled_peak_process_vram_bytes", "valid_memory_samples",
                          "unknown_memory_samples", "timed_out", "memory_limit_stop"}, "Resources")
    finite(resources["wall_s"], 0.000001, rule["max_wall_s"], "wall_s")
    integer(resources["sampled_peak_process_vram_bytes"], 1, rule["max_vram_bytes"],
            "sampled_peak_process_vram_bytes")
    integer(resources["valid_memory_samples"], 1, 10000000, "valid_memory_samples")
    integer(resources["unknown_memory_samples"], 0, 10000000, "unknown_memory_samples")
    require(resources["timed_out"] is False and resources["memory_limit_stop"] is False,
            "RESOURCE_STOP", "Timed-out or memory-stopped runs are not eligible.", 409)


def validate_submission(rule, payload):
    exact_keys(payload, {"schema_version", "rule_id", "rule_digest", "commit_id", "evaluation"}, "Submission")
    require(payload["schema_version"] == "ap-leaderboard-submission-v1", "SUBMISSION_VERSION", "Unknown submission format.")
    require(payload["rule_id"] == rule["rule_id"] and payload["rule_digest"] == rule["rule_digest"],
            "RULE_INPUT_MISMATCH", "Select the exact installed rule version.", 409)
    require(type(payload["commit_id"]) is str and re.fullmatch(r"[0-9a-f]{40}", payload["commit_id"]),
            "COMMIT_ID", "Provide the full 40-character commit ID.")
    validate_evaluation(rule, payload["evaluation"])
    report = payload["evaluation"]
    return {
        "submission_id": report["run_id"], "rule_id": rule["rule_id"],
        "rule_digest": rule["rule_digest"], "difficulty": rule["difficulty"],
        "material": rule["material"], "commit_id": payload["commit_id"],
        "completion_ticks": report["completion_ticks"],
        "simulation_seconds": report["completion_ticks"] * rule["dt_s"],
        "receipt_sha256": digest(payload), "evaluation_mode": "student_run_source_matched",
    }


def build_submission(rule, report, commit_id):
    payload = {"schema_version": "ap-leaderboard-submission-v1",
               "rule_id": rule["rule_id"], "rule_digest": rule["rule_digest"],
               "commit_id": commit_id, "evaluation": report}
    validate_submission(rule, payload)
    return payload
