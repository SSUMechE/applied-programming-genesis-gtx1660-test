"""Narrow, source-bound contact witnesses for an observed Panda vessel grasp.

This module never steps, controls or positions a body. A classified witness is
geometric evidence only, never a solved force or proof of a load-bearing grip.
Callers must supply poses from the same contact-detection instant. Unknown
pairs remain prohibited. The query adapter additionally disables pruning and
checks pair/witness parity. Live execution must disclose its own coverage.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import copy
import hashlib
from importlib import metadata
import inspect
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from physics.carried_body import HeldBodySpec
from physics.gravity_feedforward import rotation_matrix_wxyz
from transport_challenge.contracts import ContractError


MAX_PENETRATION_M = .001
# Chosen contents-only numeric policy, not a guaranteed MPR error bound.
# Native MPR stores a signed portal estimate. Never clamp that raw value or
# count a negative estimate as a pad/handle or bottom/support witness.
CONTENTS_MINIMUM_SIGNED_DEPTH_M = -.000001
NUMERIC_TOLERANCE_M = .000020
MINIMUM_NORMAL_COSINE = .95
PANDA_XML = "assets/xml/franka_emika_panda/panda.xml"
PINNED_SOURCES = {
    PANDA_XML: "e502e07fcef6219024be02dfb9cd65d5c194b4767adf381ca51f10172ff8a32d",
    "engine/entities/rigid_entity/rigid_entity.py": "b0f86f9c9ebdf62b9df2e5aadc0719c28e06cd52e0ce1bef452b1a269d9b2092",
    "engine/entities/rigid_entity/rigid_geom.py": "399e0e0ec7d63a9a965c781f6946aed67109051ac8ccd666a47092cfb6a10a14",
    "engine/solvers/rigid/rigid_solver.py": "e076854af17262323041544ff692d366402a700357053618c8d89fed1e86b19f",
    "engine/solvers/rigid/collider/collider.py": "496e2f56acdfb7bd20822c993ba12d275824335759b49cbc965347a64c833cc0",
    "engine/solvers/rigid/collider/broadphase.py": "923e6ebdd3612233ea751d74c2f532ecdbe305c392a97782a13b9b66af8c739f",
    "engine/solvers/rigid/collider/contact.py": "cd11feed70d9badfa9e56327e71cd49bcf9fe280bd8ed774b890ef6a1226023c",
    "options/solvers.py": "2f799beaf1a5acb7aa55edcc063187f3e6a384ce1ea85092c2ee72f4e202e536",
}


def contents_contact_witness_valid(depth, normal_error, owners):
    """Bound a native signed estimate for the two existing contents pair kinds.

    Ownership must be copied from the caller's built native geometry registry.
    This pure check cannot authenticate fabricated registry records or prove a
    load-bearing contact. All other pair types retain their nonnegative guard.
    """
    if (type(depth) not in (int, float) or not math.isfinite(depth)
            or type(normal_error) not in (int, float) or not math.isfinite(normal_error)
            or not isinstance(owners, (list, tuple)) or len(owners) != 2
            or any(not isinstance(owner, dict) or type(owner.get('kind')) is not str for owner in owners)):
        return False
    kinds = {owner.get('kind') for owner in owners}
    return (kinds in ({'bead'}, {'bead', 'tray'})
            and CONTENTS_MINIMUM_SIGNED_DEPTH_M <= depth <= MAX_PENETRATION_M
            and 0. <= normal_error <= .001)


def _contains_bool(item):
    if isinstance(item, (bool, np.bool_)):
        return True
    return isinstance(item, (list, tuple)) and any(_contains_bool(x) for x in item)


def _numeric(value, shape, label):
    if _contains_bool(value):
        raise ContractError(f"{label} cannot contain booleans")
    raw = value.detach().cpu().numpy() if hasattr(value, "detach") else value
    result = np.asarray(raw)
    if result.shape != shape or result.dtype.kind not in "iuf" or not np.isfinite(result).all():
        raise ContractError(f"{label} requires finite numeric shape {shape}")
    return result.astype(float, copy=True)


def verify_contact_sources():
    """Read installed metadata and exact source bytes without importing Genesis."""
    distribution = metadata.distribution("genesis-world")
    if distribution.version != "1.3.2":
        raise ContractError("Unreviewed Genesis version for contact witnesses")
    root = Path(distribution.locate_file("genesis")).resolve()
    observed = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in PINNED_SOURCES}
    if observed != PINNED_SOURCES:
        raise ContractError("Installed contact or Panda sources differ from the reviewed versions")
    return root, observed


@dataclass(frozen=True)
class SupportedHandleContactPolicy:
    """Explicit opt-in, bound to one immutable measured hand/body relation."""
    held_body: HeldBodySpec

    def __post_init__(self):
        if not isinstance(self.held_body, HeldBodySpec):
            raise ContractError("Contact policy requires the observed HeldBodySpec")

    def validate_binding(self, held_body):
        if (not isinstance(held_body, HeldBodySpec)
                or held_body.binding_sha256 != self.held_body.binding_sha256
                or held_body.model.geometry_sha256 != self.held_body.model.geometry_sha256):
            raise ContractError("Contact policy is bound to a different observed body/model")

    def to_record(self):
        return {"name": "observed_panda_pad_handle_and_named_bottom_support_v1",
                "held_body_binding_sha256": self.held_body.binding_sha256,
                "model_geometry_sha256": self.held_body.model.geometry_sha256,
                "max_penetration_m": MAX_PENETRATION_M,
                "numeric_tolerance_m": NUMERIC_TOLERANCE_M,
                "minimum_normal_cosine": MINIMUM_NORMAL_COSINE,
                "loaded_state_requires_bilateral_witnesses_in_both_queries": True,
                "unknown_contact_policy": "reject", "load_bearing_grasp_proved": False}


def read_panda_pad_primitives(xml_path):
    """Parse ten actual XML boxes. Fixed model hash avoids implicit defaults."""
    path = Path(xml_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINNED_SOURCES[PANDA_XML]:
        raise ContractError("Unreviewed Panda XML")
    root = ET.parse(path).getroot()
    records = []
    for link in ("left_finger", "right_finger"):
        body = root.find(f".//body[@name='{link}']")
        for index in range(1, 6):
            name = f"fingertip_pad_collision_{index}"
            actual = body.findall(f"geom[@class='{name}']")
            default = root.find(f".//default[@class='{name}']/geom")
            if len(actual) != 1 or dict(actual[0].attrib) != {"class": name} or default.get("type") != "box":
                raise ContractError("Native pad XML ownership or primitive changed")
            records.append({"link": link,
                "size_m": [2*float(x) for x in default.get("size").split()],
                "center_link_m": [float(x) for x in default.get("pos").split()],
                "quaternion_link_wxyz": [1., 0., 0., 0.]})
    return records


def match_native_pads(arm, primitive_records, box_type):
    """Match actual pad geometry, local rotation and ownership, not whole fingers."""
    if len(primitive_records) != 10:
        raise ContractError("Expected five native boxes on each finger")
    result = []
    for expected in primitive_records:
        size = _numeric(expected["size_m"], (3,), "pad size")
        center = _numeric(expected["center_link_m"], (3,), "pad center")
        quat = _numeric(expected["quaternion_link_wxyz"], (4,), "pad rotation")
        if not np.allclose(np.abs(quat), [1., 0., 0., 0.], atol=1e-7, rtol=0.):
            raise ContractError("Pad face requires the verified identity local rotation")
        matches = []
        for geom in arm.geoms:
            if geom.type != box_type or geom.link.name != expected["link"]:
                continue
            data = _numeric(geom.data, (7,), "native pad data")
            if (np.allclose(data[:3], size, rtol=1e-6, atol=1e-7) and np.all(data[3:] == 0.)
                    and np.allclose(geom.init_pos, center, rtol=1e-6, atol=1e-7)):
                matches.append(geom)
        if len(matches) != 1:
            raise ContractError("Pad primitive has no unique native match")
        geom = matches[0]
        built = _numeric(geom.init_quat, (4,), "native pad local quaternion")
        if (geom.entity is not arm or geom.contype != 1 or geom.conaffinity != 1
                or not np.allclose(np.abs(built), [1., 0., 0., 0.], rtol=0., atol=1e-7)):
            raise ContractError("Pad ownership, masks or local quaternion changed")
        result.append({**copy.deepcopy(expected), "geom_id": int(geom.idx),
                       "link_id": int(geom.link.idx), "built_quaternion_link_wxyz": built.tolist()})
    if len({x["geom_id"] for x in result}) != 10:
        raise ContractError("Duplicate native pad identity")
    return result


def validate_support_stations(stations):
    if not isinstance(stations, (list, tuple)):
        raise ContractError("support_stations must be a sequence")
    result = []
    for station in stations:
        if not isinstance(station, dict) or set(station) != {"id", "size_m", "center_world_m"}:
            raise ContractError("Station requires exactly id, size_m and center_world_m")
        name = station["id"]
        size = _numeric(station["size_m"], (3,), "station size")
        center = _numeric(station["center_world_m"], (3,), "station center")
        if not isinstance(name, str) or not name or np.any(size <= 0.):
            raise ContractError("Station identity and positive dimensions are required")
        result.append({"id": name, "size_m": size.tolist(), "center_world_m": center.tolist()})
    if len({x["id"] for x in result}) != len(result):
        raise ContractError("Station identities must be unique")
    return tuple(result)


def validate_contact_snapshot(raw):
    """Copy exact current witnesses. Forces are deliberately unused."""
    if not isinstance(raw, dict) or not {"geom_a", "geom_b", "position", "normal", "penetration"} <= set(raw):
        raise ContractError("Contact snapshot omitted witness fields")
    def ids(value):
        if _contains_bool(value):
            raise ContractError("Contact geometry IDs cannot contain booleans")
        return np.empty(0, dtype=int) if isinstance(value, (list, tuple)) and not value else np.asarray(value)
    a, b = ids(raw["geom_a"]), ids(raw["geom_b"])
    if (a.ndim != 1 or b.shape != a.shape or a.dtype.kind not in "iu" or b.dtype.kind not in "iu"
            or np.any(a < 0) or np.any(b < 0) or np.any(a == b)):
        raise ContractError("Contact geometry IDs must be distinct nonnegative integer vectors")
    n = len(a)
    def vectors(value):
        return np.empty((0, 3), dtype=float) if n == 0 and isinstance(value, (list, tuple)) and not value else value
    return {"geom_a": a.astype(int, copy=True), "geom_b": b.astype(int, copy=True),
            "position": _numeric(vectors(raw["position"]), (n, 3), "contact positions"),
            "normal": _numeric(vectors(raw["normal"]), (n, 3), "contact normals"),
            "penetration": _numeric(raw["penetration"], (n,), "contact depths")}


def validate_pair_array(value):
    if _contains_bool(value):
        raise ContractError("Public geometry IDs cannot contain booleans")
    pairs = np.asarray(value)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise ContractError("Malformed public query pairs")
    return pairs.copy()


def query_witness_snapshot(collider, public_pairs, queried_ids, known_ids):
    """Immediately copy after detect_collision, then verify pair MULTISETS.

    No row-order assumptions, new detection, stepping or solved-force inference.
    The caller verifies source, unbatched runtime and pruning-disabled contract.
    """
    copied = copy.deepcopy(collider.get_contacts(as_tensor=True, to_torch=False))
    raw = validate_contact_snapshot(copied)
    pairs = validate_pair_array(public_pairs)
    all_ids = set(map(int, raw["geom_a"])) | set(map(int, raw["geom_b"]))
    if not all_ids <= set(known_ids):
        raise ContractError("Contact witnesses refer to unknown geometry")
    mask = np.array([bool({int(a), int(b)} & set(queried_ids))
                     for a, b in zip(raw["geom_a"], raw["geom_b"])], dtype=bool)
    filtered = {key: values[mask].copy() for key, values in raw.items()}
    expected = Counter(tuple(sorted(map(int, pair))) for pair in pairs)
    observed = Counter(tuple(sorted((int(a), int(b))))
                       for a, b in zip(filtered["geom_a"], filtered["geom_b"]))
    if expected != observed:
        raise ContractError("Public pair and same-query witness multisets disagree")
    return filtered


def verify_query_witness_runtime(solver, source_root):
    root = Path(source_root)
    for obj, module, name, path in (
            (solver, "genesis.engine.solvers.rigid.rigid_solver", "RigidSolver", "engine/solvers/rigid/rigid_solver.py"),
            (solver.collider, "genesis.engine.solvers.rigid.collider.collider", "Collider", "engine/solvers/rigid/collider/collider.py")):
        cls = type(obj)
        if (cls.__module__ != module or cls.__name__ != name
                or Path(inspect.getsourcefile(cls)).resolve() != root/path):
            raise ContractError("Unreviewed collision runtime class")
    if (solver.n_envs != 0 or solver._use_hibernation or solver._requires_grad
            or solver._options.contact_pruning_tolerance is not None
            or float(solver.collider._collider_info.contact_pruning_tolerance[None]) != 0.):
        raise ContractError("Same-query witnesses require unbatched forward queries without contact pruning")
    return {"contact_pruning_tolerance": None, "native_pruning_tolerance": 0.,
            "query_only_configuration": True, "forces_used_for_allowance": False,
            "public_pair_witness_check": "canonical multiset, immediate copy after each detection"}


def classify_supported_contacts(raw, *, components_by_geom, pads_by_geom, body_pose,
                                finger_poses, stations_by_geom, phase="hold"):
    """Return (rows, bilateral_pad_contact), denying every unknown contact.

    ``settle`` allows named bottom/support witnesses only. ``close``, ``hold``,
    ``carry``, ``place`` and ``release`` additionally allow verified pad/handle
    witnesses. Component/pad/station records must first be matched to the actual
    built geometry. Caller owns any separate contents-contact classification.
    """
    phases = ("settle", "close", "hold", "carry", "place", "release")
    if phase not in phases:
        raise ContractError("Unsupported contact phase")
    raw = validate_contact_snapshot(raw)
    position = _numeric(body_pose["position_m"], (3,), "body position")
    rotation = rotation_matrix_wxyz(body_pose["quaternion_wxyz"])
    def local(point, pose):
        return rotation_matrix_wxyz(pose["quaternion_wxyz"]).T @ (
            point-_numeric(pose["position_m"], (3,), "finger position"))
    def inside(point, center, size):
        return bool(np.all(np.abs(point-np.asarray(center)) <= np.asarray(size)/2+NUMERIC_TOLERANCE_M))
    rows, bilateral = [], set()
    for index, (a, b, point, normal, depth) in enumerate(zip(
            raw["geom_a"], raw["geom_b"], raw["position"], raw["normal"], raw["penetration"])):
        a, b = int(a), int(b)
        category, detail = "UNALLOWED_CONTACT", {}
        normal_error = abs(float(np.linalg.norm(normal))-1.)
        valid = 0. <= depth <= MAX_PENETRATION_M and normal_error <= .001
        surface_tolerance = NUMERIC_TOLERANCE_M+max(0., float(depth))
        candidates = [(a, b), (b, a)]
        for tray_id, other_id in candidates:
            component = components_by_geom.get(tray_id)
            if component is None:
                continue
            body_local, body_normal = rotation.T@(point-position), rotation.T@normal
            detail.update(tray_component=component["component_id"], contact_body_local_m=body_local.tolist())
            station = stations_by_geom.get(other_id)
            pad = pads_by_geom.get(other_id)
            if station is not None:
                support_local = point-np.asarray(station["center_world_m"])
                bottom = component["center_body_m"][2]-component["size_m"][2]/2
                if (component["component_id"] == "solid_0" and valid
                        and inside(body_local, component["center_body_m"], component["size_m"])
                        and inside(support_local, [0, 0, 0], station["size_m"])
                        and abs(body_local[2]-bottom) <= surface_tolerance
                        and abs(support_local[2]-station["size_m"][2]/2) <= surface_tolerance
                        and abs(body_normal[2]) >= MINIMUM_NORMAL_COSINE
                        and abs(normal[2]) >= MINIMUM_NORMAL_COSINE):
                    category = "named_tray_bottom_support_top"
                detail["support_station_id"] = station["id"]
            elif pad is not None:
                finger_pose = finger_poses[pad["link"]]
                finger_local = local(point, finger_pose)
                finger_normal = rotation_matrix_wxyz(finger_pose["quaternion_wxyz"]).T@normal
                side_distance = abs(abs(body_local[1]-component["center_body_m"][1])-component["size_m"][1]/2)
                inward = pad["center_link_m"][1]-pad["size_m"][1]/2
                face_distance = abs(finger_local[1]-inward)
                if (phase != "settle" and component["component_id"] == "solid_5" and valid
                        and side_distance <= surface_tolerance and face_distance <= surface_tolerance
                        and abs(body_normal[1]) >= MINIMUM_NORMAL_COSINE
                        and abs(finger_normal[1]) >= MINIMUM_NORMAL_COSINE
                        and inside(body_local, component["center_body_m"], component["size_m"])
                        and inside(finger_local, pad["center_link_m"], pad["size_m"])):
                    category = "exact_finger_pad_handle"
                    bilateral.add(pad["link"])
                detail.update(contact_finger_local_m=finger_local.tolist(),
                              pad_inward_face_distance_m=float(face_distance),
                              handle_side_face_distance_m=float(side_distance))
            break
        rows.append({"raw_row": index, "geom_ids": [a, b], "category": category,
                     "position_world_m": point.tolist(), "normal_world": normal.tolist(),
                     "penetration_m": float(depth), "normal_unit_error": normal_error, **detail})
    return rows, bilateral == {"left_finger", "right_finger"}
