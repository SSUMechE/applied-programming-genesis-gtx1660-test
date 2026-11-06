"""Verify and package an existing qualified full-mission evaluation.

This command DOES NOT RUN PHYSICS. The current production tier rules remain
unqualified because no complete tier-native producer has been qualified. A
first-carry review cannot become a full-mission result through this packager.
Edited supplied evaluation code receives no score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from leaderboard.contracts import (
    ProtocolError, build_submission, canonical_bytes, decode_json, load_rule,
    require, runtime_snapshot, source_snapshot, validate_evaluation, validate_rule,
)

REPORT_NAME = "FULL_MISSION_EVALUATION.json"
INPUT_NAME = "EVALUATED_INPUT.json"
RESULT_NAME = "result.json"
MAX_REPORT_BYTES = 1_048_576
MAX_ARTIFACT_BYTES = 67_108_864
NATIVE_RESULT_FIELDS = {
    "schema_version", "status", "full_mission_passed", "run_id", "rule_digest",
    "input_sha256", "completion_ticks", "source_bindings_unchanged", "failure",
    "cleanup_errors", "completed_stage_ids", "completed_arm_ids", "handoffs",
    "donor_returns", "checks", "initial_sample_ids", "final_sample_ids",
    "final_rest_steps", "resources", "backend", "initialization_mode",
}
NATIVE_BOUND_FIELDS = {
    "run_id", "rule_digest", "input_sha256", "completion_ticks",
    "completed_stage_ids", "completed_arm_ids", "handoffs", "donor_returns",
    "checks", "initial_sample_ids", "final_sample_ids", "final_rest_steps",
    "resources", "backend", "initialization_mode",
}


def validate_native_result(report, result):
    """Reject contradictory artifacts without synthesizing native evidence.

    This is a future qualified producer envelope. No existing first-carry or
    legacy result is converted into this schema by the supplied packager.
    """
    deadline_report = report.get("schema_version") == "ap-full-mission-evaluation-v2"
    expected_fields = NATIVE_RESULT_FIELDS | ({"completion_clock"} if deadline_report else set())
    expected_version = "ap-full-mission-native-result-v2" if deadline_report else "ap-full-mission-native-result-v1"
    require(type(result) is dict and set(result) == expected_fields
        and result.get("schema_version") == expected_version,
        "NATIVE_RESULT_SCHEMA", "result.json is not a complete typed native full-mission envelope.", 409)
    require(result["status"] == "FULL_MISSION_PASS" and result["full_mission_passed"] is True
        and result["source_bindings_unchanged"] is True and result["failure"] is None
        and type(result["cleanup_errors"]) is list and result["cleanup_errors"] == [],
        "NATIVE_RESULT_CONTRADICTION", "The bound native result does not show a clean complete full-mission PASS.", 409)
    bound_fields = NATIVE_BOUND_FIELDS | ({"completion_clock"} if deadline_report else set())
    require(all(canonical_bytes(result[name]) == canonical_bytes(report[name]) for name in bound_fields),
        "NATIVE_RESULT_CONTRADICTION", "Native identity, phases, resources or final state contradict the full-mission report.", 409)


def _bytes(path, limit):
    require(path.is_file(), "REVIEW_FILE_MISSING", "Required evaluation file is missing: " + path.name, 409)
    with path.open("rb") as stream:
        value = stream.read(limit+1)
    require(len(value) <= limit, "REVIEW_FILE_TOO_LARGE", "Evaluation file exceeds its supplied limit: " + path.name)
    return value


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _inside(directory, name):
    path = (directory / name).resolve()
    require(path.is_relative_to(directory), "REVIEW_PATH_ESCAPE", "Evaluation files must stay inside the review directory.")
    return path


def _first_carry(value):
    return (type(value) is dict and (
        value.get("scope") in ("one_carry", "one-carry")
        or value.get("one_carry_transport_accepted") is True
        or str(value.get("status", "")).startswith("ONE_CARRY_")
        or value.get("schema_version") == "student-one-carry-review-supervisor-v1"))


def _review_report_path(directory):
    report = _inside(directory, REPORT_NAME)
    if report.is_file():
        return report
    # These are explicitly recognized existing formats, not full-tier adapters.
    for name in ("REVIEW_SUPERVISOR.json", RESULT_NAME):
        path = _inside(directory, name)
        if path.is_file():
            value = decode_json(_bytes(path, MAX_ARTIFACT_BYTES))
            if _first_carry(value):
                raise ProtocolError("FIRST_CARRY_NOT_FULL_MISSION",
                    "This review covers one grasp-ready carry only. It does not prove the original-INITIAL full tier mission.", 409)
    raise ProtocolError("FULL_MISSION_REPORT_UNAVAILABLE",
        "A qualified full-tier producer must supply FULL_MISSION_EVALUATION.json. Existing legacy/partial results cannot be relabelled.", 409)


def verify_review(rule, review_dir, *, project_root=ROOT):
    """Return the unchanged report and local audit after all current checks.

No source snapshots or physical flags are filled into a report that lacks them.
The producer must have recorded its own complete before/after evidence.
"""
    validate_rule(rule)
    directory = Path(review_dir).resolve(strict=True)
    require(directory.is_dir(), "REVIEW_DIRECTORY", "Use a review output directory.")
    require(not (directory / "EXPORT_REVOKED.json").exists(), "EXPORT_REVOKED",
            "The producer revoked this export after final execution/resource checks.", 409)
    report_path = _review_report_path(directory)
    require(rule["qualified"], "RULE_NOT_QUALIFIED", rule["qualification_reason"], 409)
    before = source_snapshot(rule, project_root)
    versions_before = runtime_snapshot(rule)
    raw_report = _bytes(report_path, MAX_REPORT_BYTES)
    report = decode_json(raw_report)
    if _first_carry(report):
        raise ProtocolError("FIRST_CARRY_NOT_FULL_MISSION", "A first-carry report is not a complete tier mission.", 409)
    validate_evaluation(rule, report)
    require(report["source_hashes_before"] == before,
        "EVALUATION_CODE_MODIFIED", "The producer's initial sources differ from the current supplied files.", 409)
    require(report["runtime_versions"] == versions_before, "RUNTIME_CHANGED", "The recorded runtime differs from this environment.", 409)
    input_path = _inside(directory, INPUT_NAME)
    result_path = _inside(directory, RESULT_NAME)
    raw_input = _bytes(input_path, MAX_ARTIFACT_BYTES)
    raw_result = _bytes(result_path, MAX_ARTIFACT_BYTES)
    # Parse the data too. Nonfinite values, duplicate keys and executable formats
    # cannot hide behind a matching byte digest.
    decode_json(raw_input)
    result = decode_json(raw_result)
    require(type(result) is dict, "NATIVE_RESULT_SCHEMA", "result.json must be a JSON object.")
    if _first_carry(result):
        raise ProtocolError("FIRST_CARRY_NOT_FULL_MISSION", "The bound native result still describes one carry only.", 409)
    validate_native_result(report, result)
    require(_sha(raw_input) == report["input_sha256"], "EVALUATED_INPUT_CHANGED", "EVALUATED_INPUT.json differs from the recorded input.", 409)
    require(_sha(raw_result) == report["artifact_sha256"], "NATIVE_ARTIFACT_CHANGED", "result.json differs from the recorded native artifact.", 409)
    after = source_snapshot(rule, project_root)
    versions_after = runtime_snapshot(rule)
    require(before == after == report["source_hashes_after"], "EVALUATION_CODE_MODIFIED", "Supplied evaluation files changed during packaging.", 409)
    require(versions_before == versions_after, "RUNTIME_CHANGED", "Runtime versions changed during packaging.", 409)
    for path, original, limit in ((report_path, raw_report, MAX_REPORT_BYTES),
            (input_path, raw_input, MAX_ARTIFACT_BYTES), (result_path, raw_result, MAX_ARTIFACT_BYTES)):
        require(_sha(_bytes(path, limit)) == _sha(original), "REVIEW_ARTIFACT_CHANGED", "An evaluation artifact changed during packaging.", 409)
    audit = {"schema_version": "ap-full-mission-package-audit-v1",
        "status": "VERIFIED_EXISTING_FULL_REPORT", "rule_id": rule["rule_id"],
        "rule_digest": rule["rule_digest"], "test_only_rule": rule["test_only"],
        "review_directory": str(directory), "source_hashes_before": before,
        "source_hashes_after": after, "runtime_versions_before": versions_before,
        "runtime_versions_after": versions_after,
        "report": {"path": str(report_path), "sha256": _sha(raw_report)},
        "evaluated_input": {"path": str(input_path), "sha256": _sha(raw_input)},
        "native_result": {"path": str(result_path), "sha256": _sha(raw_result)},
        "native_physics_launched": False,
        "scope": "Student-run source-matched existing-result packaging. Not remote attestation or a new simulation."}
    return report, audit


def package_review(rule, review_dir, commit_id, *, project_root=ROOT):
    report, audit = verify_review(rule, review_dir, project_root=project_root)
    payload = build_submission(rule, report, commit_id)
    # The root protocol's wire size must fit the service's published1MiB cap.
    require(len(canonical_bytes(payload)) <= 1_048_576, "SUBMISSION_TOO_LARGE", "Submission exceeds the1MiB service limit.")
    return payload, audit


def evaluate(rule_path, review_dir, commit_id, output, *, project_root=ROOT):
    rule = load_rule(rule_path)
    payload, audit = package_review(rule, review_dir, commit_id, project_root=project_root)
    target = Path(output).resolve()
    audit_target = target.with_name(target.stem + ".audit.json")
    require(not target.is_relative_to(Path(review_dir).resolve()), "IMMUTABLE_REVIEW_OUTPUT",
        "Write the submission outside the reviewed run directory.")
    require(not target.exists() and not audit_target.exists(), "OUTPUT_EXISTS", "Use a fresh submission and audit output path.", 409)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Both files are generated by supplied code. No extra student report is
    # requested, and the submission remains exactly the protocol's five fields.
    with audit_target.open("xb") as stream:
        stream.write(canonical_bytes(audit)+b"\n")
    with target.open("xb") as stream:
        stream.write(canonical_bytes(payload)+b"\n")
    return {"status": "SUBMISSION_PACKAGED", "submission": str(target),
        "submission_file_sha256": _sha(target.read_bytes()), "audit": str(audit_target),
        "rule_id": rule["rule_id"], "native_physics_launched": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--commit", required=True, help="Full40-character repository commit ID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.rule, args.review_dir, args.commit, args.output, project_root=args.project_root)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except ProtocolError as error:
        print(json.dumps({"status": "EVALUATION_BLOCKED", "code": error.code, "message": str(error),
            "native_physics_launched": False}, ensure_ascii=False), file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "EVALUATION_BLOCKED", "code": "REVIEW_IO_OR_DATA_ERROR", "message": str(error),
            "native_physics_launched": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
