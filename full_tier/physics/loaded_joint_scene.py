"""Isolated full-primitive held-vessel collision queries, never a live grasp.

The measured body pose must name the authored URDF 'payload' frame used by
VesselModel, not an automatically aligned solver COM frame. The query-only
URDF uses align=False to retain that exact frame. Only this isolated tray is
positioned from FK times the observed hand/body relation. No scene.step(),
runtime weld, collision exemption or intermediate route is provided.

All reported arm/tray contacts reject by default. An explicit policy bound to
observed hand/body evidence can allow only exact native pad/handle and named
bottom/support witnesses. This geometric query never proves a friction grasp.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from physics.carried_body import HeldBodySpec, WorldPose, default_contact_policy, write_carried_vessel_urdf
from physics.joint_scene import JointScene, array
from physics.gravity_feedforward import rotation_matrix_wxyz
from physics.handle_contacts import (
    SupportedHandleContactPolicy, classify_supported_contacts,
    match_native_pads, read_panda_pad_primitives, validate_support_stations,
    verify_contact_sources, verify_query_witness_runtime, query_witness_snapshot, validate_pair_array, PANDA_XML,
)
from transport_challenge.contracts import ContractError


GEOMETRY_ABSOLUTE_TOLERANCE_M = 1e-7
GEOMETRY_RELATIVE_TOLERANCE = 1e-6
POSE_TOLERANCE = 1e-5
SUPPORT_CONTACT_MASK = 0xFFFF  # Native Primitive default, unlike URDF collision defaults.


def _finite_array(value, shape, label):
    result = array(value)
    if result.shape != shape or not np.isfinite(result).all():
        raise ContractError(f"{label} must be finite with shape {shape}")
    return result


def _same_quaternion(a, b, tolerance=POSE_TOLERANCE):
    a, b = np.asarray(a), np.asarray(b)
    return min(float(np.linalg.norm(a-b)), float(np.linalg.norm(a+b))) <= tolerance


def validate_asset_components(asset, model):
    """Bind generated URDF solid names and local boxes to the immutable model."""
    if hashlib.sha256(asset.path.read_bytes()).hexdigest() != asset.asset_sha256:
        raise ContractError("Vessel asset changed after creation")
    document = ET.parse(asset.path).getroot()
    links = document.findall("link")
    if len(links) != 1 or links[0].get("name") != "payload" or document.findall("joint"):
        raise ContractError("Held vessel requires exactly one authored payload link")
    collisions = links[0].findall("collision")
    if len(collisions) != len(model.component_boxes):
        raise ContractError("Vessel collision count differs from model")
    records = []
    for index, (collision, box) in enumerate(zip(collisions, model.component_boxes)):
        origin, primitive = collision.find("origin"), collision.find("geometry/box")
        if collision.get("name") != f"solid_{index}" or origin is None or primitive is None:
            raise ContractError("Vessel collision components must be named native boxes")
        size = np.array([float(v) for v in primitive.get("size", "").split()])
        center = np.array([float(v) for v in origin.get("xyz", "").split()])
        rpy = np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
        if (size.shape != (3,) or center.shape != (3,) or rpy.shape != (3,)
                or not np.array_equal(size, box.size_m) or not np.array_equal(center, box.center_body_m)
                or not np.array_equal(rpy, np.zeros(3))):
            raise ContractError("URDF primitive differs from supplied exact geometry")
        records.append({"component_id": f"solid_{index}", "size_m": list(box.size_m),
                        "center_body_m": list(box.center_body_m), "quaternion_body_wxyz": [1.,0.,0.,0.]})
    return records


def validate_built_components(body, model, asset_records, box_type):
    """One-to-one geometric matching avoids assuming engine geom order/names."""
    geoms = tuple(body.geoms)
    if body.n_links != 1 or len(geoms) != len(asset_records):
        raise ContractError("Built tray must retain every primitive on one link")
    masses = _finite_array(body.get_links_inertial_mass(), (1,), "built vessel mass")
    if not np.isclose(masses[0], model.empty_mass_kg, rtol=1e-6, atol=1e-9):
        raise ContractError("Built vessel mass differs from fixed-density model")
    remaining = set(range(len(asset_records)))
    result = []
    geom_ids = set()
    for geom in geoms:
        if (geom.entity is not body or geom.link.name != "payload" or geom.type != box_type
                or geom.idx in geom_ids or geom.contype != 1 or geom.conaffinity != 1):
            raise ContractError("Unexpected vessel ownership, primitive type or contact masks")
        geom_ids.add(geom.idx)
        # Genesis RigidGeom pads native primitive data to seven coordinates.
        packed = _finite_array(geom.data, (7,), "built primitive data")
        if np.any(packed[3:] != 0.):
            raise ContractError("Native BOX has unexpected reserved primitive data")
        size = packed[:3]
        center = _finite_array(geom.init_pos, (3,), "built primitive origin")
        quat = _finite_array(geom.init_quat, (4,), "built primitive quaternion")
        matches = [i for i in remaining
                   if np.allclose(size, asset_records[i]["size_m"],
                                  rtol=GEOMETRY_RELATIVE_TOLERANCE, atol=GEOMETRY_ABSOLUTE_TOLERANCE_M)
                   and np.allclose(center, asset_records[i]["center_body_m"],
                                   rtol=GEOMETRY_RELATIVE_TOLERANCE, atol=GEOMETRY_ABSOLUTE_TOLERANCE_M)
                   and _same_quaternion(quat, [1.,0.,0.,0.], 1e-6)]
        if len(matches) != 1:
            raise ContractError("Built vessel geometry is missing, duplicated or reframed")
        index = matches[0]
        remaining.remove(index)
        result.append({**copy.deepcopy(asset_records[index]), "geom_id": int(geom.idx),
                       "entity": body.name, "link": geom.link.name, "engine_type": "BOX",
                       "built_size_m": size.tolist(), "built_center_body_m": center.tolist(),
                       "engine_primitive_data": packed.tolist(),
                       "contype": int(geom.contype), "conaffinity": int(geom.conaffinity)})
    if remaining:
        raise ContractError("Some vessel components were not matched")
    return sorted(result, key=lambda record: int(record["component_id"].split("_")[1]))


def validate_built_support(entity, station, box_type):
    """Exact native box with explicit representation evidence on every failure."""
    if entity.n_links != 1 or len(entity.geoms) != 1:
        raise ContractError("Support must remain one exact native box")
    geom = entity.geoms[0]
    data = _finite_array(geom.data, (7,), "support primitive data")
    position = _finite_array(entity.get_pos(relative=False), (3,), "support world position")
    quaternion = _finite_array(entity.get_quat(relative=False), (4,), "support world quaternion")
    local_position = _finite_array(geom.init_pos, (3,), "support local position")
    local_quaternion = _finite_array(geom.init_quat, (4,), "support local quaternion")
    observed = {"size_m": data[:3].tolist(), "reserved_primitive_data": data[3:].tolist(),
                "center_world_m": position.tolist(), "quaternion_world_wxyz": quaternion.tolist(),
                "geom_center_local_m": local_position.tolist(), "geom_quaternion_local_wxyz": local_quaternion.tolist(),
                "contype": int(geom.contype), "conaffinity": int(geom.conaffinity),
                "geom_id": int(geom.idx), "is_native_box": geom.type == box_type,
                "entity_ownership_matches": geom.entity is entity}
    checks = {"entity_ownership": geom.entity is entity, "native_box": geom.type == box_type,
              "contype": geom.contype == SUPPORT_CONTACT_MASK, "conaffinity": geom.conaffinity == SUPPORT_CONTACT_MASK,
              "reserved_data_zero": bool(np.all(data[3:] == 0.)),
              "size_matches": bool(np.allclose(data[:3], station["size_m"], rtol=1e-6, atol=1e-7)),
              "world_center_matches": bool(np.allclose(position, station["center_world_m"], rtol=0., atol=1e-7)),
              "world_orientation_identity": _same_quaternion(quaternion, [1.,0.,0.,0.], 1e-7),
              "local_center_zero": bool(np.allclose(local_position, [0.,0.,0.], rtol=0., atol=1e-7)),
              "local_orientation_identity": _same_quaternion(local_quaternion, [1.,0.,0.,0.], 1e-7)}
    record = {**copy.deepcopy(station), "geom_id": int(geom.idx), "observed_native_geometry": observed,
              "native_representation_checks": checks, "expected_contact_mask": SUPPORT_CONTACT_MASK,
              "geometry_tolerance_m": 1e-7, "size_relative_tolerance": 1e-6,
              "clearance_inflation_m": 0.}
    if not all(checks.values()):
        raise ContractError("Built support differs from exact authored box: "+json.dumps(record, allow_nan=False))
    return record


class LoadedJointScene(JointScene):
    """Seven arm variables plus separately observed, native-bounded fingers.

    asset_directory receives the exact instructor vessel URDF. It must belong
    to this isolated query run, never a running physical scene's asset folder.
    held_body is immutable evidence, not proof that friction maintains the pose.
    """

    def __init__(self, case, held_body, asset_directory, base=(0., 0., 0.), *,
                 render=False, debug_visuals=False, contact_policy=None,
                 support_stations=(), maximum_body_tilt_deg=None):
        if not isinstance(held_body, HeldBodySpec):
            raise ContractError("Loaded queries require an observed HeldBodySpec")
        if held_body.object_id in {"arm_0", "floor", *(item["id"] for item in case["obstacles"])}:
            raise ContractError("Held-body identity must be distinct from the arm and map")
        self.held_body = held_body
        if contact_policy is not None:
            if not isinstance(contact_policy, SupportedHandleContactPolicy):
                raise ContractError("Unsupported loaded contact policy")
            contact_policy.validate_binding(held_body)
        self.contact_policy = contact_policy
        self.support_stations = validate_support_stations(support_stations)
        occupied = {held_body.object_id, "arm_0", "floor", *(item["id"] for item in case["obstacles"])}
        if any(item["id"] in occupied for item in self.support_stations):
            raise ContractError("Support identities must be distinct from existing bodies")
        if (maximum_body_tilt_deg is not None and
                (isinstance(maximum_body_tilt_deg, (bool, np.bool_))
                 or not isinstance(maximum_body_tilt_deg, (int, float))
                 or not math.isfinite(maximum_body_tilt_deg) or not 0. <= maximum_body_tilt_deg <= 180.)):
            raise ContractError("Optional body tilt bound must be finite in [0, 180] degrees")
        self.maximum_body_tilt_deg = maximum_body_tilt_deg
        self._source_root, self._source_hashes = (verify_contact_sources() if contact_policy else (None, {}))
        self._station_entities = []
        self._stations_by_geom = {}
        self._pads_by_geom = {}
        self._witness_contract = None
        self._pair_coverage = None
        self._allowed_witness_count = 0
        self._rejected_witness_count = 0
        self.asset = write_carried_vessel_urdf(Path(asset_directory), held_body.model)
        self._asset_components = validate_asset_components(self.asset, held_body.model)
        self._tray_pose_updates = 0
        self._body_query_counts = {"arm": 0, "tray": 0}
        self._complete_body_queries = 0
        self._pair_class_counts = {}
        self._last_body_pose = None
        self._last_detection = None
        super().__init__(copy.deepcopy(case), base, render=render, debug_visuals=debug_visuals)

    def _before_build(self):
        if self.contact_policy is not None:
            # Explicitly authorized QUERY-ONLY setting. Live physics is untouched.
            self.scene.rigid_solver._options.contact_pruning_tolerance = None
        for station in self.support_stations:
            # Exact geometry, never tracking-clearance inflation. Movable only
            # to retain arm-base/station pairs in this scene, which never steps.
            self._station_entities.append(self.scene.add_entity(self.gs.morphs.Box(
                size=station["size_m"], pos=station["center_world_m"], fixed=False,
                contype=SUPPORT_CONTACT_MASK, conaffinity=SUPPORT_CONTACT_MASK), name=station["id"]))
        # The authored payload frame is preserved explicitly, unlike auto-align.
        self.tray = self.scene.add_entity(self.gs.morphs.URDF(
            file=str(self.asset.path), pos=(0.,0.,0.), quat=(1.,0.,0.,0.),
            fixed=False, merge_fixed_links=True, align=False),
            material=self.gs.materials.Rigid(friction=1., needs_coup=True, coup_friction=.4),
            name=self.held_body.object_id)
        if self.tray is self.arm:
            raise ContractError("The isolated arm and tray must be separate entities")

    def _query_finger_positions(self):
        low, high = (array(values) for values in self.arm.get_dofs_limit())
        if low.shape != (9,) or high.shape != (9,) or not np.isfinite([low, high]).all():
            raise ContractError("Observed fingers require the built nine-DOF native limits")
        fingers = self.held_body.finger_dof_positions_m
        if any(not low[index] <= value <= high[index] for index, value in zip((7,8), fingers)):
            raise ContractError("Observed finger coordinates violate native limits. No clipping is allowed")
        return fingers

    def _after_arm_q_placement(self):
        hand = WorldPose(
            tuple(_finite_array(self.hand.get_pos(relative=False), (3,), "query hand world position")),
            tuple(_finite_array(self.hand.get_quat(relative=False), (4,), "query hand world quaternion")))
        body = self.held_body.body_pose_at(hand)
        self.tray.set_pos(body.position_m, relative=False, zero_velocity=True)
        self.tray.set_quat(body.quaternion_wxyz, relative=False, zero_velocity=True)
        self._tray_pose_updates += 1
        actual = WorldPose(
            tuple(_finite_array(self.tray.get_pos(relative=False), (3,), "query tray world position")),
            tuple(_finite_array(self.tray.get_quat(relative=False), (4,), "query tray world quaternion")))
        if (np.linalg.norm(np.array(actual.position_m)-body.position_m) > POSE_TOLERANCE
                or not _same_quaternion(actual.quaternion_wxyz, body.quaternion_wxyz)):
            raise ContractError("Isolated query tray did not realize the measured-frame composition")
        self._last_body_pose = {"hand_world": hand.to_record(), "body_world": actual.to_record(),
                                "query_pose_update_index": self._tray_pose_updates}

    def current_body_pose(self):
        """Copy the last FK-composed query pose, not a live physical measurement."""
        return copy.deepcopy(self._last_body_pose["body_world"])

    def _additional_state_valid(self):
        if self.maximum_body_tilt_deg is None:
            return True
        z_axis = rotation_matrix_wxyz(self._last_body_pose["body_world"]["quaternion_wxyz"])[:, 2]
        tilt = math.degrees(math.acos(float(np.clip(z_axis[2], -1., 1.))))
        if tilt > self.maximum_body_tilt_deg:
            return self._reject("body_tilt_constraint", body_tilt_deg=tilt,
                                maximum_body_tilt_deg=self.maximum_body_tilt_deg)
        return True

    def state_valid(self, q):
        valid = super().state_valid(q)
        if valid and self.contact_policy is not None:
            snapshots = self._last_detection["same_query_witnesses_by_body"]
            if not all(snapshots[label]["bilateral_pad_handle_witnesses"] for label in ("arm", "tray")):
                return self._reject("missing_bilateral_handle_contact",
                    bilateral_by_query={label: snapshots[label]["bilateral_pad_handle_witnesses"]
                                        for label in ("arm", "tray")})
        return valid

    def _after_build(self):
        self._component_ledger = validate_built_components(
            self.tray, self.held_body.model, self._asset_components, self.gs.GEOM_TYPE.BOX)
        self._arm_geom_ids = frozenset(int(geom.idx) for geom in self.arm.geoms)
        self._tray_geom_ids = frozenset(record["geom_id"] for record in self._component_ledger)
        if not self._arm_geom_ids or self._arm_geom_ids & self._tray_geom_ids:
            raise ContractError("Arm and tray geometry sets must be nonempty and disjoint")
        for record in self._component_ledger:
            self.geometry_names[record["geom_id"]]["component_id"] = record["component_id"]
        for station, entity in zip(self.support_stations, self._station_entities):
            record = validate_built_support(entity, station, self.gs.GEOM_TYPE.BOX)
            self._stations_by_geom[record["geom_id"]] = record
        if self.contact_policy is not None:
            self._witness_contract = verify_query_witness_runtime(self.scene.rigid_solver, self._source_root)
            pads = match_native_pads(self.arm, read_panda_pad_primitives(self._source_root/PANDA_XML), self.gs.GEOM_TYPE.BOX)
            self._pads_by_geom = {record["geom_id"]: record for record in pads}
            from physics.self_collision_audit import built_self_pair_ledger
            self._pair_coverage = built_self_pair_ledger(self.scene.rigid_solver, [self.arm])
            if self._pair_coverage["robots"]["arm_0"]["unexplained_excluded_pair_count"]:
                raise ContractError("Native same-arm pair filtering has unexplained exclusions")
            enabled = {tuple(sorted(map(int, pair))) for pair in self.scene.rigid_solver.collider._valid_collision_pairs}
            environment = set(self.geometry_names)-self._arm_geom_ids-self._tray_geom_ids
            obstacle_names = {item["id"] for item in self.case["obstacles"]}
            obstacle_ids = {geom_id for geom_id, names in self.geometry_names.items() if names["entity"] in obstacle_names}
            groups = {"arm_tray": [(a,b) for a in self._arm_geom_ids for b in self._tray_geom_ids],
                      "tray_environment": [(a,b) for a in self._tray_geom_ids for b in environment],
                      "arm_map_obstacle": [(a,b) for a in self._arm_geom_ids for b in obstacle_ids],
                      "arm_station": [(a,b) for a in self._arm_geom_ids for b in self._stations_by_geom]}
            self._pair_coverage["required_cross_body_pairs"] = {}
            for name, pairs in groups.items():
                rows = [{"geom_ids": sorted((a,b)), "enabled": tuple(sorted((a,b))) in enabled} for a,b in pairs]
                self._pair_coverage["required_cross_body_pairs"][name] = rows
                if not all(row["enabled"] for row in rows):
                    raise ContractError(f"Native filter omitted a required {name} collision pair")
            self._pair_coverage["arm_floor_pairs"] = [
                {"geom_ids": sorted((int(geom.idx), floor_id)),
                 "enabled": tuple(sorted((int(geom.idx), floor_id))) in enabled,
                 "arm_link_fixed_relative_to_world": bool(geom.link.is_fixed),
                 "scope": "native fixed-base/floor exclusions remain disclosed, not a contact allowance"}
                for geom in self.arm.geoms for floor_id, names in self.geometry_names.items() if names["entity"] == "floor"]

    def _current_finger_poses(self):
        result = {}
        for name in ("left_finger", "right_finger"):
            link = self.arm.get_link(name)
            result[name] = WorldPose(tuple(array(link.get_pos(relative=False))),
                                     tuple(array(link.get_quat(relative=False)))).to_record()
        return result

    def _query_collision_pairs(self):
        union = set()
        counts = {}
        forbidden, witness_rows = set(), {}
        policy = getattr(self, "contact_policy", None)
        finger_poses = self._current_finger_poses() if policy else None
        # Both public queries recompute at the same unchanged q/body pose.
        for label, entity, ids in (("arm", self.arm, self._arm_geom_ids),
                                   ("tray", self.tray, self._tray_geom_ids)):
            self._body_query_counts[label] += 1
            self.collision_queries += 1
            pairs = entity.detect_collision()
            # Must be the very next contact-buffer operation after detection.
            raw = (query_witness_snapshot(self.scene.rigid_solver.collider, pairs, ids, self.geometry_names)
                   if policy else None)
            self.scene.rigid_solver.check_errno()
            pairs = validate_pair_array(pairs)
            if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
                raise ContractError("Public collision query returned malformed pair IDs")
            counts[label] = len(pairs)
            for a, b in pairs:
                a, b = int(a), int(b)
                if a == b or a not in self.geometry_names or b not in self.geometry_names or not ({a,b} & ids):
                    raise ContractError("Collision pair lacks valid queried-body ownership")
                union.add(tuple(sorted((a,b))))
            if policy:
                rows, bilateral = classify_supported_contacts(raw,
                    components_by_geom={row["geom_id"]: row for row in self._component_ledger},
                    pads_by_geom=self._pads_by_geom, body_pose=self.current_body_pose(),
                    finger_poses=finger_poses, stations_by_geom=self._stations_by_geom, phase="carry")
                witness_rows[label] = {"rows": rows, "bilateral_pad_handle_witnesses": bilateral,
                    "pair_witness_multiset_agrees": True,
                    "finger_poses_world": copy.deepcopy(finger_poses)}
                for row in rows:
                    if row["category"] == "UNALLOWED_CONTACT":
                        forbidden.add(tuple(sorted(row["geom_ids"])))
                        self._rejected_witness_count += 1
                    else:
                        self._allowed_witness_count += 1
        self._complete_body_queries += 1
        classes = {}
        for a, b in union:
            owners = {"arm" if value in self._arm_geom_ids else "tray" if value in self._tray_geom_ids
                      else "environment" for value in (a,b)}
            name = "_".join(sorted(owners))
            classes[name] = classes.get(name, 0)+1
            self._pair_class_counts[name] = self._pair_class_counts.get(name, 0)+1
        self._last_detection = {**copy.deepcopy(self._last_body_pose),
            "detection_index": self._complete_body_queries,
            "raw_pair_counts_by_body": counts, "unique_pair_count": len(union),
            "pair_classes": classes, "all_reported_pairs_rejected": policy is None,
            "contact_point_allowance_applied": policy is not None,
            "forbidden_unique_pair_count": len(forbidden) if policy else len(union),
            "same_query_witnesses_by_body": witness_rows}
        return np.array(sorted(forbidden if policy else union), dtype=int).reshape((-1,2))

    def evidence(self):
        record = copy.deepcopy(super().evidence())
        record.update({
            "loaded_payload_checked": self._complete_body_queries > 0,
            "loaded_query_scope": "isolated exact vessel geometry only, no contents or physical grasp",
            "held_body": self.held_body.to_record(),
            "body_frame": "authored_URDF_payload_frame_in_world",
            "query_urdf_align": False,
            "asset_path": str(self.asset.path), "asset_sha256": self.asset.asset_sha256,
            "model_geometry_sha256": self.asset.model_geometry_sha256,
            "primitive_geometry_ledger": copy.deepcopy(self._component_ledger),
            "query_body_geometry_ids": {"arm": sorted(self._arm_geom_ids), "tray": sorted(self._tray_geom_ids),
                "environment": sorted(set(self.geometry_names)-self._arm_geom_ids-self._tray_geom_ids)},
            "observed_finger_dof_positions_m": list(self.fingers),
            "isolated_tray_pose_updates": self._tray_pose_updates,
            "body_collision_query_calls": dict(self._body_query_counts),
            "completed_dual_body_collision_queries": self._complete_body_queries,
            "cumulative_unique_pair_classes": dict(self._pair_class_counts),
            "last_loaded_detection": copy.deepcopy(self._last_detection),
            "contact_policy": self.contact_policy.to_record() if self.contact_policy else default_contact_policy(),
            "same_query_contact_allowance_implemented": self.contact_policy is not None,
            "support_stations_exact_geometry": copy.deepcopy(self.support_stations),
            "support_geometry_ledger": copy.deepcopy(self._stations_by_geom),
            "native_pad_geometry_ledger": copy.deepcopy(self._pads_by_geom),
            "contact_source_sha256": dict(self._source_hashes),
            "same_query_witness_runtime_contract": copy.deepcopy(self._witness_contract),
            "native_pair_coverage_ledger": copy.deepcopy(self._pair_coverage),
            "cumulative_allowed_witnesses_by_query": self._allowed_witness_count,
            "cumulative_rejected_witnesses_by_query": self._rejected_witness_count,
            "maximum_body_tilt_deg": self.maximum_body_tilt_deg,
            "body_tilt_constraint_role": "optional disclosed design constraint, all seven joints remain variables",
            "collision_filter_coverage": "engine enabled pairs only, adjacent arm links remain excluded",
            "collision_filter_completeness_independently_approved": False,
            "physical_grasp_verified": False, "physical_pick_carry_place_completed": False,
            "runtime_weld_or_payload_pose_command": False,
        })
        return record

