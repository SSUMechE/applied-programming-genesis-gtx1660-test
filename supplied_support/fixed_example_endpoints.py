"""Validate the supplied fixed benchmark endpoints without solving new IK branches.

This declares a fresh source-bound INITIAL and checks nominal CPU geometry.
It never restores a native state or certifies a physical grasp or transport.
"""
from __future__ import annotations

import math

from physics.provided_manipulation import _pose
from student.fixed_example import WAYPOINTS
from transport_challenge.contracts import ContractError, digest, point
from transport_challenge.joint_kinematics import ToolPose


WORLD_SHA256 = "a1abe6fd35a8f25efe19190d97350de427dc0150f20668bcb653fe4f1b4ccd27"
SOURCE_SCENARIO_SHA256 = "e38cbf30172f16d862e4f1cfcda8b9436583bb2b738f3328b821f478a64a5e48"
DESIGN_SHA256 = "b75fdb3160f892e939021c57a3a18cb6196cc0d5c74c2b23c11ac0a2b0d53b9f"
# These are the existing effective CpuOracle.ik residual caps, not new tolerances.
POSITION_TOLERANCE_M = 0.00001
ORIENTATION_TOLERANCE_RAD = 0.00005


def bind_fixed_example_endpoints(world, scenario, design_record):
    if (digest(world) != WORLD_SHA256 or digest(scenario) != SOURCE_SCENARIO_SHA256
            or digest(design_record) != DESIGN_SHA256):
        raise ContractError("The supplied fixed example requires its exact world, INITIAL layout and design")
    if not isinstance(WAYPOINTS, tuple) or len(WAYPOINTS) != 5:
        raise ContractError("The supplied fixed example requires its five declared waypoints")
    path = tuple(point(q, 7) for q in WAYPOINTS)
    return path[0], path[-1]


def validate_fixed_endpoint(kinematics, oracle, q, target, *, wall_check=None):
    """Check the declared q using FK and the unchanged original collision oracle."""
    if wall_check is not None and not callable(wall_check):
        raise ContractError("wall_check must be callable")
    if not callable(getattr(kinematics, "fk", None)) or not callable(getattr(oracle, "state_valid", None)):
        raise ContractError("Use the supplied bounded FK and original state oracle")
    q = point(q, 7)
    bounds = tuple(point(pair, 2) for pair in kinematics.joint_bounds_rad)
    workspace = tuple(point(pair, 2) for pair in kinematics.workspace_bounds_m)
    if (len(bounds) != 7 or len(workspace) != 3
            or any(lo >= hi or not math.isfinite(hi-lo) for lo, hi in bounds + workspace)):
        raise ContractError("Fixed endpoint validation needs the original native and workspace bounds")
    if any(not lo <= value <= hi for value, (lo, hi) in zip(q, bounds)):
        raise ContractError("The supplied fixed endpoint leaves native joint bounds")
    target = _pose(target)
    if any(not lo <= value <= hi for value, (lo, hi) in zip(target.position_m, workspace)):
        raise ContractError("Fixed endpoint target leaves the supplied workspace")
    if wall_check is not None:
        wall_check()
    actual = kinematics.fk(q)
    actual = ToolPose(actual.position_m, actual.quaternion_wxyz)
    if wall_check is not None:
        wall_check()
    if any(not lo <= value <= hi for value, (lo, hi) in zip(actual.position_m, workspace)):
        raise ContractError("Fixed endpoint FK leaves the supplied workspace")
    position_error = math.dist(actual.position_m, target.position_m)
    dot = math.fsum(a*b for a, b in zip(actual.quaternion_wxyz, target.quaternion_wxyz))
    cosine = abs(dot) / (math.hypot(*actual.quaternion_wxyz) * math.hypot(*target.quaternion_wxyz))
    orientation_error = 2*math.acos(min(1., cosine))
    if position_error > POSITION_TOLERANCE_M or orientation_error > ORIENTATION_TOLERANCE_RAD:
        raise ContractError("The supplied fixed endpoint exceeds the original FK residual limits: "
                            + str({"position_error_m": position_error,
                                   "orientation_error_rad": orientation_error}))
    valid = oracle.state_valid(q)
    if wall_check is not None:
        wall_check()
    if type(valid) is not bool:
        raise ContractError("The original endpoint state oracle must return a Boolean")
    if not valid:
        raise ContractError("The supplied fixed endpoint is rejected by the original collision/state oracle")
    result = dict(schema_version="supplied-fixed-endpoint-validation-v1", status="ENDPOINT_FOUND",
                  q_rad=list(q), endpoint_hand_pose=target.to_record(),
                  actual_fk_hand_pose=dict(frame="world", position_m=list(actual.position_m),
                                           quaternion_wxyz=list(actual.quaternion_wxyz)),
                  position_error_m=position_error, orientation_error_rad=orientation_error,
                  position_tolerance_m=POSITION_TOLERANCE_M,
                  orientation_tolerance_rad=ORIENTATION_TOLERANCE_RAD,
                  original_state_valid=True, CPU_FK_verified=True,
                  CPU_IK_calls=0, native_ik_performed=False, path_generated=False,
                  physical_grasp_verified=False, physical_transport_verified=False)
    result["receipt_sha256"] = digest(result)
    return result
