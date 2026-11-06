"""Supplied fixed benchmark paths, freshly validated on this computer.

This is a hardware/replay example, not a path-search performance benchmark.
The CPU exporter retains its original complete geometric and timing checks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from physics.provided_manipulation import _pose
from transport_challenge.contracts import ContractError, digest, point

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "example" / "tier_paths.json"
SCHEMA = "supplied-fixed-tier-benchmarks-v1"
POSITION_TOLERANCE_M = .00001
ORIENTATION_TOLERANCE_RAD = .00005
_recipe = None
_stage = None
planning_receipts = []


def require(condition, message):
    if not condition:
        raise ContractError(message)


def bind_fixed_tier_benchmark(difficulty, original_world, original_scenario,
                              design, world, scenario):
    """Bind exact source inputs before any supplied endpoint is selected."""
    global _recipe, _stage
    data = json.loads(DATA.read_bytes())
    require(data.get("schema_version") == SCHEMA, "Unknown fixed tier benchmark schema")
    require(data.get("data_sha256") == digest({k: v for k, v in data.items() if k != "data_sha256"}),
            "Fixed tier benchmark data changed")
    recipe = data.get("tiers", {}).get(difficulty)
    require(isinstance(recipe, dict), "No supplied benchmark for this difficulty")
    actual = dict(original_world_sha256=digest(original_world),
                  original_initialization_sha256=digest(original_scenario),
                  design_sha256=digest(design), world_sha256=digest(world),
                  scenario_sha256=digest(scenario))
    require(all(recipe.get(key) == value for key, value in actual.items()),
            "Fixed tier benchmark requires its exact world, original INITIAL and tier design")
    speed = recipe.get("joint_speed_scale")
    require(type(speed) in (int, float) and math.isfinite(speed) and 0 < speed <= 1
            and type(design.get("joint_speed_scale")) in (int, float)
            and design["joint_speed_scale"] == speed,
            "Fixed tier benchmark requires its exact declared tier timing input")
    require(len(recipe.get("stages", [])) == 2 * {"easy": 1, "medium": 2, "hard": 3}[difficulty],
            "Fixed tier benchmark stage count differs")
    _recipe, _stage = copy.deepcopy(recipe), None
    planning_receipts.clear()
    return benchmark_receipt()


def benchmark_source_files():
    require(_recipe is not None, "Bind the fixed benchmark before source admission")
    return (DATA,)


def benchmark_receipt():
    require(_recipe is not None, "No bound fixed benchmark")
    return dict(schema_version="supplied-fixed-tier-benchmark-binding-v1",
                difficulty=_recipe["difficulty"], data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                recipe_sha256=digest(_recipe), design_sha256=_recipe["design_sha256"],
                joint_speed_scale=_recipe["joint_speed_scale"],
                path_generated=False, runtime_ik_calls=0,
                original_cpu_geometry_checks_required=True, physical_transport_verified=False,
                cost_scope="Fresh local validation/export cost. Excludes original path generation.")


def _rotation(q):
    # Same wxyz -> SciPy matrix conversion as the original CPU oracle.
    values = np.asarray(q, dtype=float)
    return Rotation.from_quat(values[[1, 2, 3, 0]]).as_matrix()


def validate_stage_endpoint(stage_id, kinematics, oracle, start_q, target, *, wall_check):
    """Validate declared endpoint using original FK residual caps and state oracle."""
    global _stage
    require(_recipe is not None, "Bind the fixed tier benchmark first")
    matches = [s for s in _recipe["stages"] if s["stage_id"] == stage_id]
    require(len(matches) == 1, "Unknown or duplicated fixed benchmark stage")
    stage = matches[0]
    path = tuple(point(q, 7) for q in stage["waypoints"])
    require(path[0] == point(start_q, 7), "Fixed benchmark stage starts at a different command")
    bounds = tuple(point(pair, 2) for pair in kinematics.joint_bounds_rad)
    workspace = tuple(point(pair, 2) for pair in kinematics.workspace_bounds_m)
    require(len(bounds) == 7 and len(workspace) == 3
            and all(lo < hi for lo, hi in bounds + workspace), "Invalid native/workspace bounds")
    q = path[-1]
    require(all(lo <= value <= hi for value, (lo, hi) in zip(q, bounds)),
            "Fixed endpoint leaves the original native joint bounds")
    target = _pose(target)
    require(all(lo <= value <= hi for value, (lo, hi) in zip(target.position_m, workspace)),
            "Fixed endpoint target leaves the original workspace")
    wall_check()
    actual = kinematics.fk(q)
    require(all(lo <= value <= hi for value, (lo, hi) in zip(actual.position_m, workspace)),
            "Fixed endpoint FK leaves the original workspace")
    # Exact residual expressions/caps of CpuOracle.ik; no new IK or relaxed gate.
    position_error = float(np.linalg.norm(np.asarray(target.position_m) - actual.position_m))
    rotation_error = Rotation.from_matrix(_rotation(target.quaternion_wxyz)
                                         @ _rotation(actual.quaternion_wxyz).T).as_rotvec()
    orientation_error = float(np.linalg.norm(rotation_error))
    require(position_error <= POSITION_TOLERANCE_M and orientation_error <= ORIENTATION_TOLERANCE_RAD,
            "Fixed endpoint exceeds original CPU FK residual limits: "
            + repr(dict(position_error_m=position_error, orientation_error_rad=orientation_error)))
    valid = oracle.state_valid(q)
    wall_check()
    require(type(valid) is bool and valid, "Fixed endpoint rejected by original state/collision oracle")
    _stage = copy.deepcopy(stage)
    receipt = dict(schema_version="supplied-fixed-tier-endpoint-validation-v1", status="ENDPOINT_FOUND",
                   stage_id=stage_id, q_rad=list(q), endpoint_hand_pose=target.to_record(),
                   position_error_m=position_error, orientation_error_rad=orientation_error,
                   position_tolerance_m=POSITION_TOLERANCE_M,
                   orientation_tolerance_rad=ORIENTATION_TOLERANCE_RAD,
                   original_state_valid=True, CPU_FK_verified=True, CPU_IK_calls=0,
                   native_ik_performed=False, path_generated=False,
                   physical_grasp_verified=False, physical_transport_verified=False)
    receipt["receipt_sha256"] = digest(receipt)
    return receipt


def plan_path(request, oracle, *, kinematics=None, context=None):
    require(_stage is not None, "Validate this supplied fixed endpoint before requesting its path")
    path = tuple(point(q, 7) for q in _stage["waypoints"])
    require(point(request.start, 7) == path[0] and point(request.goal, 7) == path[-1],
            "Fixed benchmark path identity or endpoints changed")
    planning_receipts.append(dict(stage_id=_stage["stage_id"], path_generated=False,
                                  returned_fixed_waypoints=len(path), fresh_full_validation_required=True))
    return path


def verify_stage_trajectory(stage_id, trajectory):
    """The unchanged declared-speed analytic parameterizer must reproduce the supplied q/T."""
    require(_stage is not None and _stage["stage_id"] == stage_id,
            "Fixed benchmark stage identity changed")
    require(trajectory["waypoints"] == _stage["waypoints"]
            and trajectory["edge_durations_s"] == _stage["edge_durations_s"]
            and trajectory["limits"] == _stage["limits"],
            "Regenerated fixed benchmark q, declared analytic times or native limits changed")
    return True
