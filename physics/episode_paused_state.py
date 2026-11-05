"""Supplied pause identity checks. No native proof, reads or state writes.

The caller supplies actual observations and trusted completed/rest receipts.
This module neither observes a scene nor verifies that the caller paused it.
"""
from __future__ import annotations

import copy
import math

from transport_challenge.contracts import ContractError, digest


def _json(value):
    """Reject non-JSON/nonfinite values without coercing native observations."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ContractError("Nonfinite observation or binding")
        return
    if type(value) is list:
        for item in value:
            _json(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():
            _json(item)
        return
    raise ContractError("Use detached JSON values with string dictionary keys")


def _fields(value, fields, label):
    if type(value) is not dict or set(value) != set(fields):
        raise ContractError(label + " requires the declared fields")


def _number(value, label, *, positive=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ContractError(label + " must be finite numeric")
    if positive and value <= 0:
        raise ContractError(label + " must be positive")
    return value


def _index(value, label):
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise ContractError(label + " must be a nonnegative integer")
    return value


def _name(value):
    if type(value) is not str or not value.strip():
        raise ContractError("Expected a nonempty explicit identity")
    return value


def _hash(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError("Expected lowercase SHA256")
    return value


def _vector(value, size, label):
    if type(value) is not list or len(value) != size:
        raise ContractError(label + " has wrong dimension")
    return [_number(v, label) for v in value]


def _pose(value):
    _fields(value, ("frame", "position_m", "quaternion_wxyz"), "World pose")
    if value["frame"] != "world":
        raise ContractError("Explicit world-frame pose required")
    quaternion = _vector(value["quaternion_wxyz"], 4, "Quaternion")
    if not any(quaternion):
        raise ContractError("Zero quaternion")
    # Preserve native getter bits. This identity check does not normalize or
    # certify geometric accuracy of the caller's quaternion.
    return dict(frame="world", position_m=_vector(value["position_m"], 3, "Position"),
                quaternion_wxyz=quaternion)


def _bindings(value):
    _json(value)
    _fields(value, ("schema_version", "scene_token", "solver_token", "dt_s",
                   "world_sha256", "configuration_sha256", "source_manifest_sha256",
                   "geometry_sha256", "reader_layout_sha256", "arms", "body", "materials"),
            "Pause bindings")
    if value["schema_version"] != "episode-pause-bindings-v1":
        raise ContractError("Unknown pause bindings")
    _name(value["scene_token"]); _name(value["solver_token"])
    _number(value["dt_s"], "dt_s", positive=True)
    for key in ("world_sha256", "configuration_sha256", "source_manifest_sha256",
                "geometry_sha256", "reader_layout_sha256"):
        _hash(value[key])
    arms = value["arms"]
    if type(arms) is not dict or len(arms) not in (2, 3):
        raise ContractError("Bind every one of the two or three Panda arms")
    entities = []
    names = []
    for name, arm in arms.items():
        names.append(_name(name))
        _fields(arm, ("entity_idx", "model_sha256"), "Arm identity")
        entities.append(_index(arm["entity_idx"], "Arm entity"))
        _hash(arm["model_sha256"])
    body = value["body"]
    _fields(body, ("id", "entity_idx", "model_sha256"), "Body identity")
    names.append(_name(body["id"]))
    entities.append(_index(body["entity_idx"], "Body entity"))
    _hash(body["model_sha256"])
    materials = value["materials"]
    if type(materials) is not list or not materials:
        raise ContractError("Explicit persistent supply required")
    logical_ids = []
    for item in materials:
        _fields(item, ("id", "entity", "entity_idx", "mass_kg", "radius_m"), "Material identity")
        logical_ids.append(_name(item["id"])); names.append(_name(item["entity"]))
        entities.append(_index(item["entity_idx"], "Material entity"))
        _number(item["mass_kg"], "Material mass", positive=True)
        _number(item["radius_m"], "Material radius", positive=True)
    if (len(entities) != len(set(entities)) or len(names) != len(set(names))
            or len(logical_ids) != len(set(logical_ids)) or set(logical_ids) & set(names)):
        raise ContractError("Aliased body, arm or material identity")
    return copy.deepcopy(value)


def project_paused_state(observation, *, bindings):
    """Return detached native state, exact bindings and their canonical digest.

    Finite qdd/tracking/effort/tilt/containment fields may be present but are not
    native-state identity. All q/qd and world pose/velocity values stay exact.
    """
    _json(observation)
    b = _bindings(bindings)
    if type(observation) is not dict:
        raise ContractError("Observation must be an object")
    required = ("simulation_s", "native_scene_step", "arms", "body_world",
                "body_linear_velocity_m_s", "body_angular_velocity_rad_s", "materials")
    if not set(required).issubset(observation):
        raise ContractError("Missing native observation fields")
    step = _index(observation["native_scene_step"], "Native scene step")
    now = _number(observation["simulation_s"], "Simulation time")
    if now < 0 or abs(now - step * b["dt_s"]) > 1e-9:
        raise ContractError("Native clock is off the bound dt grid")
    if type(observation["arms"]) is not dict or set(observation["arms"]) != set(b["arms"]):
        raise ContractError("Observed arm identities differ")
    arms = {}
    for name, arm in observation["arms"].items():
        if type(arm) is not dict or not {"q_all_dofs", "qd_all_dofs", "hand_world", "finger_world"}.issubset(arm):
            raise ContractError("Incomplete native arm observation")
        if "entity_idx" in arm and (type(arm["entity_idx"]) is not int or arm["entity_idx"] != b["arms"][name]["entity_idx"]):
            raise ContractError("Observed arm entity differs")
        fingers = arm["finger_world"]
        _fields(fingers, ("left_finger", "right_finger"), "Finger poses")
        arms[name] = dict(q_all_dofs=_vector(arm["q_all_dofs"], 9, "Seven joints and two fingers"),
                          qd_all_dofs=_vector(arm["qd_all_dofs"], 9, "Nine native velocities"),
                          hand_world=_pose(arm["hand_world"]),
                          finger_world={k: _pose(v) for k, v in fingers.items()})
    observed_materials = observation["materials"]
    if type(observed_materials) is not list or len(observed_materials) != len(b["materials"]):
        raise ContractError("Persistent supply count differs")
    materials = []
    for item, bound in zip(observed_materials, b["materials"]):
        if type(item) is not dict or not {"id", "entity_idx", "mass_kg", "position_world_m", "velocity_world_m_s"}.issubset(item):
            raise ContractError("Incomplete material observation")
        _name(item["id"]); _index(item["entity_idx"], "Observed material entity")
        _number(item["mass_kg"], "Observed mass", positive=True)
        if any(item[k] != bound[k] for k in ("id", "entity_idx", "mass_kg")):
            raise ContractError("Material order, identity or mass differs")
        if any(k in item and item[k] != bound[k] for k in ("entity", "radius_m")):
            raise ContractError("Optional observed material identity differs")
        materials.append({**copy.deepcopy(bound),
                          "position_world_m": _vector(item["position_world_m"], 3, "Material position"),
                          "velocity_world_m_s": _vector(item["velocity_world_m_s"], 3, "Material velocity")})
    state = dict(simulation_s=now, native_scene_step=step, arms=arms,
                 body_world=_pose(observation["body_world"]),
                 body_linear_velocity_m_s=_vector(observation["body_linear_velocity_m_s"], 3, "Body linear velocity"),
                 body_angular_velocity_rad_s=_vector(observation["body_angular_velocity_rad_s"], 3, "Body angular velocity"),
                 materials=materials)
    record = dict(schema_version="episode-paused-native-state-v1", state=state, bindings=b)
    return {**record, "sha256": digest(record)}


def _context(projection, dispatcher_record, boundary_receipt, committed_hash):
    _json(dispatcher_record); _json(boundary_receipt)
    d, r = dispatcher_record, boundary_receipt
    if (type(d) is not dict or d.get("schema_version") != "episode-continuation-dispatch-v1"
            or not {"status", "current_segment_complete", "pending_step", "global_step",
                    "simulation_s", "last_observation_sha256", "current_segment_sha256"}.issubset(d)
            or d.get("status") != "SEGMENT_COMPLETED" or d.get("current_segment_complete") is not True
            or d.get("pending_step") is not None):
        raise ContractError("Require a fully committed segment with no pending command")
    state = projection["state"]; names = sorted(state["arms"])
    _number(d["simulation_s"], "Dispatcher time")
    if (type(d.get("global_step")) is not int or d["global_step"] != state["native_scene_step"]
            or d.get("simulation_s") != state["simulation_s"]
            or d.get("last_observation_sha256") != _hash(committed_hash)):
        raise ContractError("Dispatcher differs from the committed observation")
    _hash(d.get("current_segment_sha256"))
    _fields(r, ("schema_version", "previous_segment_sha256", "step", "simulation_s",
                "observation_sha256", "verified_arm_ids", "actual_state_verified",
                "measured_rest_verified", "source_bindings_verified"), "Trusted boundary receipt")
    _number(r["simulation_s"], "Trusted receipt time")
    if (r["schema_version"] != "episode-continuation-rest-boundary-v1"
            or r["previous_segment_sha256"] != d["current_segment_sha256"]
            or type(r["step"]) is not int or r["step"] != state["native_scene_step"]
            or r["simulation_s"] != state["simulation_s"]
            or r["observation_sha256"] != committed_hash
            or type(r["verified_arm_ids"]) is not list
            or any(type(v) is not str for v in r["verified_arm_ids"])
            or sorted(r["verified_arm_ids"]) != names
            or any(r[k] is not True for k in ("actual_state_verified", "measured_rest_verified", "source_bindings_verified"))):
        raise ContractError("Trusted receipt does not bind the exact completed/rest/source boundary")


def begin_pause(completed_observation, *, bindings, dispatcher_record, boundary_receipt):
    """Bind the exact observation committed by the dispatcher, plus projection."""
    projection = project_paused_state(completed_observation, bindings=bindings)
    committed_hash = digest(completed_observation)
    _context(projection, dispatcher_record, boundary_receipt, committed_hash)
    record = dict(schema_version="episode-paused-state-receipt-v1", projection=projection,
                  committed_observation_sha256=committed_hash,
                  dispatcher_record_sha256=digest(dispatcher_record),
                  boundary_receipt=copy.deepcopy(boundary_receipt))
    return {**record, "sha256": digest(record)}


def verify_pause(receipt, after_observation, *, bindings, dispatcher_record, boundary_receipt):
    """Reject any changed native value, identity, binding, clock or receipt.

    An accepted comparison is conditional on caller-supplied measurements and
    trusted assertions. It is not a native isolation, rest or safety proof.
    """
    _json(receipt)
    _fields(receipt, ("schema_version", "projection", "committed_observation_sha256",
                      "dispatcher_record_sha256", "boundary_receipt", "sha256"), "Pause receipt")
    if receipt["schema_version"] != "episode-paused-state-receipt-v1":
        raise ContractError("Unknown pause receipt")
    payload = {k: v for k, v in receipt.items() if k != "sha256"}
    if digest(payload) != _hash(receipt["sha256"]):
        raise ContractError("Pause receipt changed")
    before = receipt["projection"]
    _fields(before, ("schema_version", "state", "bindings", "sha256"), "Native projection")
    if (before["schema_version"] != "episode-paused-native-state-v1"
            or digest({k: v for k, v in before.items() if k != "sha256"}) != _hash(before["sha256"])):
        raise ContractError("Stored projection changed")
    after = project_paused_state(after_observation, bindings=bindings)
    _context(after, dispatcher_record, boundary_receipt, receipt["committed_observation_sha256"])
    if digest(dispatcher_record) != receipt["dispatcher_record_sha256"] or digest(boundary_receipt) != digest(receipt["boundary_receipt"]):
        raise ContractError("Dispatcher lineage or trusted boundary changed")
    if after["sha256"] != before["sha256"]:
        raise ContractError("Paused native state, identities or provided bindings changed")
    return dict(schema_version="episode-paused-state-comparison-v1", passed=True,
                projection_sha256=after["sha256"], pause_receipt_sha256=receipt["sha256"],
                committed_observation_sha256=receipt["committed_observation_sha256"],
                extension_boundary_receipt=copy.deepcopy(boundary_receipt),
                native_pause_proved=False, native_calls_performed=False,
                measured_rest_proved_by_this_module=False,
                scope="Exact supplied-state comparison conditional on trusted caller observations and receipts.")
