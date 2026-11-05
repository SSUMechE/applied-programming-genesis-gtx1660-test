"""Full-source Panda/map/vessel geometry on CPU, without Genesis or stepping.

MuJoCo supplies FK, Jacobians and explicit geom distances. A vectorized AABB
broadphase selects the complete declared pair table independently of MuJoCo's
parent/weld collision filters. This is nominal geometric planning, not a native
grasp receipt, dynamics, material retention or continuous-collision certificate.
"""
from __future__ import annotations

import copy
import hashlib
from importlib import metadata
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from physics.carried_body import WorldPose, _multiply, _rotate
from physics.handle_contacts import classify_supported_contacts, contents_contact_witness_valid
from transport_challenge.contracts import ContractError, digest, point
from transport_challenge.joint_kinematics import ToolPose
from transport_challenge.query_diagnostics import diagnostic_record
from transport_challenge.vessel import build_container_geometry


def _numbers(values):
    return " ".join(format(float(v), ".17g") for v in values)


def _rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def _quat(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return q[[3, 0, 1, 2]]


def _pose(value):
    if isinstance(value, WorldPose):
        return value
    return WorldPose(value["position_m"], value["quaternion_wxyz"])


class CpuGeometry:
    def __init__(self, world, scenario):
        self.world, self.scenario = copy.deepcopy(world), copy.deepcopy(scenario)
        if world["case_record"] != scenario["case"]:
            raise ContractError("CPU world and scenario cases differ")
        self.case = copy.deepcopy(scenario["case"])
        self.clearance = max(.004, float(self.case["planning"].get("tracking_clearance_m", 0.)))
        self.resolution = min(.005, float(self.case["planning"]["edge_max_joint_delta_rad"]))
        self.asset_root = Path(metadata.distribution("genesis-world").locate_file("genesis"))
        self.panda_xml = self.asset_root / "assets/xml/franka_emika_panda/panda.xml"
        if hashlib.sha256(self.panda_xml.read_bytes()).hexdigest() != world["robot_model_sha256"]:
            raise ContractError("Installed Panda source differs from the task")
        self.source_pins = {str(self.panda_xml): hashlib.sha256(self.panda_xml.read_bytes()).hexdigest()}
        self.owners_by_name = {}
        self.components_by_name = {}
        self.stations_by_name = {}
        self.body_name_to_joint = {}
        self.compiled_started = time.perf_counter()
        self.model = mujoco.MjModel.from_xml_string(self._xml())
        self.data = mujoco.MjData(self.model)
        self.build_wall_s = time.perf_counter() - self.compiled_started
        self.arm_addresses, self.arm_dofs, self.hand_ids = {}, {}, {}
        for robot in scenario["robots"]:
            name = robot["id"]
            joint_ids = [self.model.joint(name + "__" + j).id for j in [*self.case["arm_joint_names"], "finger_joint1", "finger_joint2"]]
            self.arm_addresses[name] = np.asarray([self.model.jnt_qposadr[j] for j in joint_ids])
            self.arm_dofs[name] = np.asarray([self.model.jnt_dofadr[j] for j in joint_ids[:7]])
            self.hand_ids[name] = self.model.body(name + "__hand").id
        self.bounds = tuple(tuple(float(np.float32(v)) for v in row) for row in
                            self.model.jnt_range[[self.model.joint("arm_0__" + j).id for j in self.case["arm_joint_names"]]])
        self.geom_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(self.model.ngeom)]
        self.owners = {i: self.owners_by_name[name] for i, name in enumerate(self.geom_names)}
        self.components = {self.model.geom(name).id: value for name, value in self.components_by_name.items()}
        self.stations = {self.model.geom(name).id: value for name, value in self.stations_by_name.items()}
        self.required_pairs = self._pairs()
        self.initial_q_by_arm = {r["id"]: [*map(float, np.asarray(r["q_rad"], dtype=np.float32)),
            *map(float, np.asarray(r["finger_q_m"], dtype=np.float32))] for r in scenario["robots"]}
        initial_body = _pose(scenario["body_pose"])
        inverse = _rotation(initial_body.quaternion_wxyz).T
        self.content_local = {item["id"]: (inverse @ (np.asarray(item["center_world_m"]) - initial_body.position_m)).tolist()
                              for item in scenario["contents"]}
        self.configure("arm_0", self.initial_q_by_arm, initial_body)

    def _xml(self):
        source = ET.parse(self.panda_xml).getroot()
        root = ET.Element("mujoco", model="cpu_nominal_three_arm_transport")
        ET.SubElement(root, "compiler", angle="radian", autolimits="true")
        ET.SubElement(root, "option", gravity="0 0 0")
        defaults = ET.SubElement(root, "default")
        assets = ET.SubElement(root, "asset")
        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "geom", name="floor_geom", type="plane", size="10 10 .1", contype="1", conaffinity="1")
        self.owners_by_name["floor_geom"] = {"kind": "floor", "entity": "floor", "link": "floor"}
        referenced = set()
        for robot in self.scenario["robots"]:
            prefix = robot["id"] + "__"
            default = copy.deepcopy(source.find("default/default"))
            for node in default.iter():
                if "class" in node.attrib:
                    node.set("class", prefix + node.get("class"))
            defaults.append(default)
            body = copy.deepcopy(source.find("worldbody/body"))
            b = robot["base_placement"]
            body.set("pos", _numbers([b["x_m"], b["y_m"], robot.get("base_z_m", self.case["base_z_m"])]))
            body.set("quat", _numbers([math.cos(b["yaw_rad"]/2), 0, 0, math.sin(b["yaw_rad"]/2)]))
            ordinal = 0
            for link in body.iter("body"):
                original_link = link.get("name")
                for geom in list(link.findall("geom")):
                    if geom.get("class") == "visual":
                        link.remove(geom)
                        continue
                    name = prefix + "geom_" + str(ordinal)
                    geom.set("name", name)
                    self.owners_by_name[name] = {"kind": "arm", "entity": robot["id"], "link": original_link}
                    ordinal += 1
                    if "material" in geom.attrib:
                        del geom.attrib["material"]
                    if "mesh" in geom.attrib:
                        referenced.add(geom.get("mesh"))
                        geom.set("mesh", prefix + geom.get("mesh"))
                for joint in link.findall("joint"):
                    joint.set("name", prefix + joint.get("name"))
                link.set("name", prefix + original_link)
                if "childclass" in link.attrib:
                    link.set("childclass", prefix + link.get("childclass"))
            if ordinal != 23:
                raise ContractError("Expected all23 source collision geometries per Panda")
            for node in body.iter():
                if "class" in node.attrib and not node.get("class").startswith(prefix):
                    node.set("class", prefix + node.get("class"))
            worldbody.append(body)
            for mesh in source.findall("asset/mesh"):
                name = mesh.get("name", Path(mesh.get("file")).stem)
                if name not in referenced:
                    continue
                asset = copy.deepcopy(mesh)
                path = self.panda_xml.parent / "assets" / mesh.get("file")
                asset.set("name", prefix + name)
                asset.set("file", str(path))
                assets.append(asset)
                self.source_pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        embedded = self.case.get("embedded_support_components", {})
        support_by_component = {component: next(s for s in self.scenario["support_stations"] if s["id"] == station)
                                for station, component in embedded.items()}
        occupied = set()
        for robot in self.scenario["robots"]:
            base = robot["base_placement"]
            candidates = [o["id"] for o in self.case["obstacles"] if o["lower_m"][0] <= base["x_m"] <= o["upper_m"][0]
                and o["lower_m"][1] <= base["y_m"] <= o["upper_m"][1]
                and abs(robot.get("base_z_m", self.case["base_z_m"]) - o["upper_m"][2] - .001) <= 1e-7]
            if len(candidates) > 1:
                raise ContractError("CPU occupied mounting plate is ambiguous")
            occupied.update(candidates)
        self.nominal_mount_ids = sorted(occupied)
        group_by_component = {name: group["id"] for group in self.case.get("compound_obstacle_groups", []) for name in group["component_ids"]}
        groups = {}
        self.obstacle_specs = {}
        for obstacle in self.case["obstacles"]:
            component = obstacle["id"]
            group = group_by_component.get(component, component)
            if group not in groups:
                groups[group] = self._free_body(worldbody, "map_" + group, [0, 0, 0])
            clearance = 0. if component in occupied or component in support_by_component else self.clearance
            low = np.asarray(obstacle["lower_m"]) - clearance
            low[2] = max(0., low[2])
            high = np.asarray(obstacle["upper_m"]) + clearance
            center, size = (low + high)/2, high-low
            name = "map_geom_" + component
            ET.SubElement(groups[group], "geom", name=name, type="box", size=_numbers(size/2), pos=_numbers(center), density="1000")
            self.obstacle_specs[name] = {"id": component, "nominal_lower_m": obstacle["lower_m"], "query_lower_m": low.tolist(),
                                        "size_m": size.tolist(), "center_world_m": center.tolist()}
            if component in support_by_component:
                self.stations_by_name[name] = copy.deepcopy(support_by_component[component])
                self.owners_by_name[name] = {"kind": "support", "entity": group, "link": group, "station_id": support_by_component[component]["id"]}
            else:
                self.owners_by_name[name] = {"kind": "obstacle", "entity": group, "link": group, "component_id": component}
        for station in self.scenario["support_stations"]:
            if station["id"] in embedded:
                continue
            body = self._free_body(worldbody, "support_" + station["id"], station["center_world_m"])
            name = "support_geom_" + station["id"]
            ET.SubElement(body, "geom", name=name, type="box", size=_numbers(np.asarray(station["size_m"])/2))
            self.stations_by_name[name] = copy.deepcopy(station)
            self.owners_by_name[name] = {"kind": "support", "entity": station["id"], "link": station["id"]}
        body_pose = _pose(self.scenario["body_pose"])
        vessel = self._free_body(worldbody, "vessel", body_pose.position_m, body_pose.quaternion_wxyz)
        self.vessel_geometry = build_container_geometry(self.scenario["container"])
        for index, (size, center) in enumerate(self.vessel_geometry["component_boxes"]):
            name = "vessel_solid_" + str(index)
            ET.SubElement(vessel, "geom", name=name, type="box", size=_numbers(np.asarray(size)/2), pos=_numbers(center))
            self.owners_by_name[name] = {"kind": "tray", "entity": self.scenario["body_id"], "link": "payload"}
            self.components_by_name[name] = {"component_id": "solid_" + str(index), "size_m": list(size), "center_body_m": list(center)}
        for item in self.scenario["contents"]:
            body = self._free_body(worldbody, "content_" + item["id"], item["center_world_m"])
            name = "content_geom_" + item["id"]
            ET.SubElement(body, "geom", name=name, type="sphere", size=_numbers([item["radius_m"]]))
            self.owners_by_name[name] = {"kind": "bead", "entity": item["id"], "link": item["id"]}
        return ET.tostring(root, encoding="unicode")

    def _free_body(self, parent, name, position, quaternion=(1, 0, 0, 0)):
        body = ET.SubElement(parent, "body", name=name, pos=_numbers(position), quat=_numbers(quaternion))
        joint = name + "__free"
        ET.SubElement(body, "freejoint", name=joint)
        self.body_name_to_joint[name] = joint
        return body

    def _pairs(self):
        model = self.model
        pairs, excluded = [], {}
        def adjacent(a, b):
            low, high = sorted((a, b))
            while int(model.body_parentid[high]) != 0:
                if int(model.body_parentid[high]) == low:
                    return True
                if model.body_jntnum[high] != 0:
                    break
                high = int(model.body_parentid[high])
            return False
        for a in range(model.ngeom):
            for b in range(a+1, model.ngeom):
                oa, ob = self.owners[a], self.owners[b]
                ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
                reason = None
                if ba == bb:
                    reason = "same_rigid_link"
                elif oa["kind"] == ob["kind"] == "arm" and oa["entity"] == ob["entity"] and adjacent(ba, bb):
                    reason = "native_parent_or_fixed_chain"
                if reason:
                    excluded[reason] = excluded.get(reason, 0) + 1
                else:
                    pairs.append((a, b))
        self.pair_policy = {"all_unordered_pairs": model.ngeom*(model.ngeom-1)//2,
            "required_pairs": len(pairs), "excluded_pairs_by_reason": excluded,
            "required_pair_table_sha256": digest(pairs),
            "coverage": "All non-same-link/non-native-adjacent pairs, including fixed roots, peers and map. Explicit geomDistance after AABB broadphase. No MuJoCo collision filter exemptions."}
        return np.asarray(pairs, dtype=np.int32)

    def _put_body(self, name, pose):
        address = self.model.jnt_qposadr[self.model.joint(self.body_name_to_joint[name]).id]
        self.data.qpos[address:address+7] = [*pose.position_m, *pose.quaternion_wxyz]

    def configure(self, active_arm_id, q_by_arm, body_pose, holder_arm_id=None, hand_to_body=None):
        if active_arm_id not in self.arm_addresses or holder_arm_id not in (None, active_arm_id):
            raise ContractError("CPU query requires one named active arm and optional same-arm nominal holder")
        self.active_arm_id = active_arm_id
        self.generation = getattr(self, "generation", 0) + 1
        self.holder_arm_id = holder_arm_id
        self.nominal_hand_to_body = copy.deepcopy(hand_to_body)
        if holder_arm_id is not None and hand_to_body is None:
            raise ContractError("Nominal carrying requires an explicit hand-to-body transform")
        for arm, address in self.arm_addresses.items():
            values = q_by_arm[arm]
            if isinstance(values, dict):
                values = [*values["q_rad"], *values["finger_q_m"]]
            elif len(values) == 7:
                values = [*values, *self.initial_q_by_arm[arm][7:]]
            self.data.qpos[address] = point(values, 9)
        self.body_pose = _pose(body_pose)
        self._put_body("vessel", self.body_pose)
        for item in self.scenario["contents"]:
            center = self.body_pose.transform_point(self.content_local[item["id"]])
            self._put_body("content_" + item["id"], WorldPose(center, (1, 0, 0, 0)))
        mujoco.mj_kinematics(self.model, self.data)
        return CpuOracle(self)

    def _aabbs(self):
        rotation = self.data.geom_xmat.reshape((-1, 3, 3))
        local = self.model.geom_aabb
        center = self.data.geom_xpos + np.einsum("nij,nj->ni", rotation, local[:, :3])
        half = np.einsum("nij,nj->ni", np.abs(rotation), local[:, 3:])
        lower, upper = center-half, center+half
        floor = self.model.geom("floor_geom").id
        lower[floor], upper[floor] = (-np.inf, -np.inf, -np.inf), (np.inf, np.inf, 0.)
        return lower, upper

    def evidence(self):
        return {"schema_version": "cpu-full-source-nominal-geometry-v1", "backend": "mujoco " + mujoco.__version__,
            "geometry_count": self.model.ngeom, "per_arm_collision_geometries": 23,
            "vessel_box_count": len(self.components), "content_sphere_count": len(self.scenario["contents"]),
            "planning_tracking_clearance_m": self.clearance, "edge_max_joint_delta_rad": self.resolution,
            "support_contact_full_bottom_footprint_margin_m": self.clearance,
            "support_footprint_scope": "Additional conservative CPU planning rule only: on vessel/support contact all four bottom corners must lie inside that named support with the existing tracking-clearance inset. Native geometry and contact predicates are unchanged.",
            "nominal_mount_component_ids": self.nominal_mount_ids, "pair_policy": self.pair_policy,
            "asset_source_pins": copy.deepcopy(self.source_pins), "build_wall_s": self.build_wall_s,
            "scope": "Full source collision meshes and exact map/vessel primitives. CPU nominal FK/collision only. No Genesis import, stepping, forces or future actual native grasp claim.",
            "mesh_backend_limit": "MuJoCo and Genesis hull/narrowphase numerical representations require separate actual native saved-path validation. Source mesh hulls are not TCP proxies."}


class CpuOracle:
    def __init__(self, geometry):
        self.geometry = geometry
        self.bounds = geometry.bounds
        self.states_checked = self.edges_checked = self.collision_queries = 0
        self.fk_calls = self.ik_calls = 0
        self.last_failure = None
        self.query_wall_s = 0.
        self.generation = geometry.generation
        self.static_pairs_validated = False
        moving = {i for i, owner in geometry.owners.items() if owner["kind"] == "arm" and owner["entity"] == geometry.active_arm_id
                  or geometry.holder_arm_id is not None and owner["kind"] in ("tray", "bead")}
        self.dynamic_pair_mask = np.asarray([int(a) in moving or int(b) in moving for a, b in geometry.required_pairs])
        self.static_qpos_mask = np.ones(geometry.model.nq, dtype=bool)
        self.static_qpos_mask[geometry.arm_addresses[geometry.active_arm_id][:7]] = False
        if geometry.holder_arm_id is not None:
            for name in ("vessel", *("content_" + item["id"] for item in geometry.scenario["contents"])):
                address = geometry.model.jnt_qposadr[geometry.model.joint(geometry.body_name_to_joint[name]).id]
                self.static_qpos_mask[address:address+7] = False
        self.static_qpos = geometry.data.qpos[self.static_qpos_mask].copy()

    def _require_snapshot(self):
        if self.geometry.generation != self.generation:
            raise ContractError("CPU oracle snapshot was superseded by configure")
        if not np.array_equal(self.geometry.data.qpos[self.static_qpos_mask], self.static_qpos):
            raise ContractError("A CPU peer, fixed fixture, finger or unloaded vessel snapshot changed")

    def _set(self, q):
        engine = self.geometry
        self._require_snapshot()
        engine.data.qpos[engine.arm_addresses[engine.active_arm_id][:7]] = q
        mujoco.mj_kinematics(engine.model, engine.data)
        if engine.holder_arm_id is not None:
            hand = self.fk(q)
            transform = engine.nominal_hand_to_body
            body_position = np.asarray(hand.position_m) + _rotate(hand.quaternion_wxyz, transform["translation_hand_m"])
            body_quat = _multiply(hand.quaternion_wxyz, transform["quaternion_hand_body_wxyz"])
            engine.body_pose = WorldPose(body_position.tolist(), body_quat)
            engine._put_body("vessel", engine.body_pose)
            for item in engine.scenario["contents"]:
                center = engine.body_pose.transform_point(engine.content_local[item["id"]])
                engine._put_body("content_" + item["id"], WorldPose(center, (1, 0, 0, 0)))
            mujoco.mj_kinematics(engine.model, engine.data)

    def fk(self, q):
        self._require_snapshot()
        self.fk_calls += 1
        engine = self.geometry
        engine.data.qpos[engine.arm_addresses[engine.active_arm_id][:7]] = point(q, 7)
        mujoco.mj_kinematics(engine.model, engine.data)
        body = engine.hand_ids[engine.active_arm_id]
        return ToolPose(engine.data.xpos[body].tolist(), _quat(engine.data.xmat[body].reshape(3, 3)).tolist())

    def ik(self, position_m, quaternion_wxyz, seed_q, *, position_tolerance_m=.003, orientation_tolerance_rad=.02):
        self._require_snapshot()
        self.ik_calls += 1
        engine = self.geometry
        target_rotation = _rotation(quaternion_wxyz)
        q = np.asarray(point(seed_q, 7), dtype=float).copy()
        lower, upper = np.asarray(self.bounds).T
        if np.any(q < lower) or np.any(q > upper):
            return None
        position_tolerance_m = min(float(position_tolerance_m), .00001)
        orientation_tolerance_rad = min(float(orientation_tolerance_rad), .00005)
        jp, jr = np.zeros((3, engine.model.nv)), np.zeros((3, engine.model.nv))
        for _ in range(100):
            actual = self.fk(q.tolist())
            position_error = np.asarray(position_m) - actual.position_m
            rotation_error = Rotation.from_matrix(target_rotation @ _rotation(actual.quaternion_wxyz).T).as_rotvec()
            if np.linalg.norm(position_error) <= position_tolerance_m and np.linalg.norm(rotation_error) <= orientation_tolerance_rad:
                return tuple(q)
            mujoco.mj_comPos(engine.model, engine.data)
            mujoco.mj_jacBody(engine.model, engine.data, jp, jr, engine.hand_ids[engine.active_arm_id])
            jacobian = np.vstack((jp[:, engine.arm_dofs[engine.active_arm_id]], jr[:, engine.arm_dofs[engine.active_arm_id]]))
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6)*1e-5, np.r_[position_error, rotation_error])
            norm = np.linalg.norm(delta)
            if norm > .25:
                delta *= .25/norm
            q = np.minimum(upper, np.maximum(lower, q+delta))
        return None

    def _reject(self, reason, **details):
        self.last_failure = {"reason": reason, **details}
        return False

    def _support_bottom_footprint(self, station):
        engine = self.geometry
        bottom = next(value for value in engine.components.values() if value["component_id"] == "solid_0")
        center, size = bottom["center_body_m"], bottom["size_m"]
        corners = np.asarray([engine.body_pose.transform_point([center[0]+x*size[0]/2,
            center[1]+y*size[1]/2, center[2]-size[2]/2]) for x in (-1, 1) for y in (-1, 1)])
        support_center, support_size = np.asarray(station["center_world_m"]), np.asarray(station["size_m"])
        lower, upper = support_center-support_size/2, support_center+support_size/2
        margins = np.minimum(corners[:, :2]-lower[:2], upper[:2]-corners[:, :2])
        minimum = float(margins.min())
        return {"passed": minimum >= engine.clearance-1e-12, "support_id": station["id"],
            "minimum_xy_margin_m": minimum, "required_xy_margin_m": engine.clearance,
            "bottom_corners_world_m": corners.tolist(), "scope": "CPU conservative contact planning only, not a changed native physical acceptance rule."}

    def state_valid(self, q):
        started = time.perf_counter()
        self.states_checked += 1
        try:
            q = point(q, 7)
            if any(not lo <= x <= hi for x, (lo, hi) in zip(q, self.bounds)):
                return self._reject("joint_limit")
            engine = self.geometry
            self._set(q)
            hand = engine.data.xpos[engine.hand_ids[engine.active_arm_id]]
            workspace = np.asarray(engine.case["tool_workspace_m"]) + [engine.clearance, -engine.clearance]
            if np.any(hand < workspace[:, 0]) or np.any(hand > workspace[:, 1]):
                return self._reject("tool_workspace")
            for obstacle in engine.case["obstacles"]:
                nearest = np.maximum(obstacle["lower_m"], np.minimum(hand, obstacle["upper_m"]))
                if np.linalg.norm(hand-nearest) <= engine.case.get("tool_reference_radius_m", 0.) + engine.clearance:
                    return self._reject("tool_reference_obstacle", obstacle=obstacle["id"])
            if engine.holder_arm_id is not None and _rotation(engine.body_pose.quaternion_wxyz)[2, 2] < math.cos(math.radians(7.5)):
                return self._reject("body_tilt_constraint")
            lower, upper = engine._aabbs()
            if engine.holder_arm_id is None:
                fingers = [i for i, owner in engine.owners.items() if owner["kind"] == "arm" and owner["entity"] == engine.active_arm_id
                           and owner["link"] in ("left_finger", "right_finger")]
                for finger in fingers:
                    for vessel in engine.components:
                        gap = np.maximum(np.maximum(lower[finger]-upper[vessel], lower[vessel]-upper[finger]), 0.)
                        if np.linalg.norm(gap) < .003:
                            return self._reject("open_finger_body_aabb_clearance")
            pairs = engine.required_pairs
            if self.static_pairs_validated:
                pairs = pairs[self.dynamic_pair_mask]
            active = np.all(lower[pairs[:, 0]] <= upper[pairs[:, 1]], axis=1) & np.all(lower[pairs[:, 1]] <= upper[pairs[:, 0]], axis=1)
            contacts = None
            fromto = np.zeros(6)
            for a, b in pairs[active]:
                self.collision_queries += 1
                distance = mujoco.mj_geomDistance(engine.model, engine.data, int(a), int(b), 1., fromto)
                if distance > 0.:
                    continue
                oa, ob = engine.owners[int(a)], engine.owners[int(b)]
                kinds = {oa["kind"], ob["kind"]}
                if kinds == {"tray", "support"}:
                    support = int(a) if oa["kind"] == "support" else int(b)
                    footprint = self._support_bottom_footprint(engine.stations[support])
                    if not footprint["passed"]:
                        return self._reject("named_support_full_bottom_footprint_clearance", footprint=footprint,
                                            pair=[int(a), int(b)], distance_m=float(distance))
                if "arm" in kinds or kinds not in ({"tray", "support"}, {"bead"}, {"bead", "tray"}, {"floor", "support"}, {"floor", "obstacle"}):
                    return self._reject("full_model_collision", pair=[int(a), int(b)], owners=[oa, ob], distance_m=float(distance))
                if contacts is None:
                    mujoco.mj_collision(engine.model, engine.data)
                    contacts = engine.data.contact
                witnesses = [{"dist": float(c.dist), "pos": np.asarray(c.pos).copy(), "normal": np.asarray(c.frame[:3]).copy()}
                             for c in contacts if {int(c.geom1), int(c.geom2)} == {int(a), int(b)}]
                if not witnesses and distance == 0.:
                    # MuJoCo omits exactly tangent contacts from data.contact.
                    # Its explicit distance query still returns the touching
                    # surface point. Known planes/spheres provide an analytic
                    # CPU normal, not a claimed native contact observation.
                    contact_point = (fromto[:3] + fromto[3:])/2
                    normal = np.array([0., 0., 1.])
                    if kinds == {"tray", "support"}:
                        vessel = int(a) if oa["kind"] == "tray" else int(b)
                        support = int(b) if vessel == int(a) else int(a)
                        component = engine.components[vessel]
                        station = engine.stations[support]
                        bottom = np.asarray(component["center_body_m"]) - np.asarray(component["size_m"])/2
                        corners = [engine.body_pose.transform_point([bottom[0]+x*component["size_m"][0],
                            bottom[1]+y*component["size_m"][1], bottom[2]]) for x in (0, 1) for y in (0, 1)]
                        top = station["center_world_m"][2] + station["size_m"][2]/2
                        if component["component_id"] != "solid_0" or min(corner[2] for corner in corners) < top-1e-12:
                            return self._reject("zero_depth_box_contact_has_no_vertical_separation_proof", pair=[int(a), int(b)])
                    if "bead" in kinds:
                        sphere = int(a) if oa["kind"] == "bead" else int(b)
                        direction = engine.data.geom_xpos[sphere] - contact_point
                        length = np.linalg.norm(direction)
                        if length > 0.:
                            normal = direction/length
                    witnesses = [{"dist": 0., "pos": contact_point, "normal": normal}]
                if not witnesses:
                    return self._reject("missing_cpu_same_query_contact_witness", pair=[int(a), int(b)], distance_m=float(distance))
                for contact in witnesses:
                    depth = -contact["dist"]
                    normal = contact["normal"]
                    if kinds in ({"bead"}, {"bead", "tray"}) and contents_contact_witness_valid(depth, float(abs(np.linalg.norm(normal)-1.)), [oa, ob]):
                        continue
                    if kinds == {"floor", "support"} and 0 <= depth <= .001 and abs(normal[2]) >= .95:
                        continue
                    if kinds == {"floor", "obstacle"}:
                        obstacle_index = int(a) if oa["kind"] == "obstacle" else int(b)
                        spec = engine.obstacle_specs[engine.geom_names[obstacle_index]]
                        if spec["nominal_lower_m"][2] == spec["query_lower_m"][2] == 0. and 0 <= depth <= .001 and abs(normal[2]) >= .95 and abs(contact["pos"][2]) <= .000020+depth:
                            continue
                    raw = {"geom_a": [int(a)], "geom_b": [int(b)], "position": [contact["pos"].tolist()],
                           "normal": [normal.tolist()], "penetration": [depth]}
                    rows, _ = classify_supported_contacts(raw, components_by_geom=engine.components, pads_by_geom={},
                        body_pose=engine.body_pose.to_record(), finger_poses={}, stations_by_geom=engine.stations, phase="settle")
                    if rows[0]["category"] != "named_tray_bottom_support_top":
                        return self._reject("unallowed_cpu_contact", witness=rows[0])
            self.last_failure = None
            self.static_pairs_validated = True
            return True
        except (ContractError, ValueError) as error:
            return self._reject("invalid_joint_state", message=str(error))
        finally:
            self.query_wall_s += time.perf_counter() - started

    def edge_valid(self, a, b):
        self.edges_checked += 1
        try:
            a, b = point(a, 7), point(b, 7)
        except ContractError:
            return self._reject("invalid_joint_edge")
        intervals = max(1, math.ceil(max(abs(x-y) for x, y in zip(a, b))/self.geometry.resolution))
        return all(self.state_valid(a if i == 0 else b if i == intervals else [x+(y-x)*i/intervals for x, y in zip(a, b)])
                   for i in range(intervals+1))

    def last_query_diagnostic(self):
        record = diagnostic_record("accepted" if self.last_failure is None else "rejected",
                                   reason=None if self.last_failure is None else self.last_failure["reason"])
        record["details"] = copy.deepcopy(self.last_failure)
        return record

    def evidence(self):
        return {"states_checked": self.states_checked, "edges_checked": self.edges_checked, "explicit_geom_distance_calls": self.collision_queries,
            "fk_calls": self.fk_calls, "ik_calls": self.ik_calls, "query_wall_s": self.query_wall_s,
            "last_failure": copy.deepcopy(self.last_failure), "nominal_only": True, "physical_grasp_verified": False}
