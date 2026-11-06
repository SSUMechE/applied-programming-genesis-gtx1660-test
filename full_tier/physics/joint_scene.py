"""Provided full-model configuration queries in an isolated planning scene.

The actual Genesis collision geometry is queried without advancing physics.
No workspace waypoint, intermediate joint posture or substitute route is made.
This first adapter checks a bare arm. A loaded-material mission must add its
carried geometry and account for dynamic tracking before it can be accepted.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np


def array(value):
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value, dtype=float)


def source_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_map(case, base):
    def vector(values, count, label):
        if (not isinstance(values, (list, tuple)) or len(values) != count
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values)):
            raise ValueError(f"{label} must contain {count} finite numeric values")
        return values

    def intervals(values, count, label):
        if not isinstance(values, (list, tuple)) or len(values) != count:
            raise ValueError(f"{label} must contain {count} intervals")
        for pair in values:
            lo, hi = vector(pair, 2, label)
            if not lo < hi:
                raise ValueError(f"Invalid {label} interval")

    if case.get("schema_version") != "joint-map-v1":
        raise ValueError("Expected joint-map-v1")
    if case.get("robot_model") != "xml/franka_emika_panda/panda.xml":
        raise ValueError("This adapter has been implemented only for the declared Panda model")
    vector(base, 3, "base x_m, y_m, yaw_rad")
    if case.get("arm_joint_names") != [f"joint{i}" for i in range(1, 8)] or case.get("tool_link") != "hand":
        raise ValueError("Joint and tool names must match the supported model")
    base_z, radius = vector([case["base_z_m"], case.get("tool_reference_radius_m", 0.)], 2, "base height and tool radius")
    if base_z < 0 or radius < 0:
        raise ValueError("Base height and tool reference radius must be nonnegative")
    if not isinstance(case["installation_zones"], list) or not case["installation_zones"]:
        raise ValueError("Provide at least one bounded installation zone")
    for zone in case["installation_zones"]:
        intervals([zone["x_m"], zone["y_m"], zone["yaw_rad"]], 3, "installation zone")
    if not any(all(lo <= x <= hi for x, (lo, hi) in zip(base, (z["x_m"], z["y_m"], z["yaw_rad"])))
               for z in case["installation_zones"]):
        raise ValueError("Robot installation leaves all permitted zones")
    intervals(case["tool_workspace_m"], 3, "tool workspace")
    if not isinstance(case["obstacles"], list) or not case["obstacles"]:
        raise ValueError("Obstacle map must contain physical obstacles")
    obstacle_ids = []
    for obstacle in case["obstacles"]:
        vector(obstacle["lower_m"], 3, "obstacle lower XYZ")
        vector(obstacle["upper_m"], 3, "obstacle upper XYZ")
        if any(not a < b
               for a, b in zip(obstacle["lower_m"], obstacle["upper_m"])):
            raise ValueError("Invalid obstacle bounds")
        if not isinstance(obstacle["id"], str) or not obstacle["id"]:
            raise ValueError("Obstacle needs an identity")
        obstacle_ids.append(obstacle["id"])
    if len(set(obstacle_ids)) != len(obstacle_ids):
        raise ValueError("Obstacle identities must be unique")
    for name in ("start", "goal"):
        pose = case[f"{name}_tool_pose"]
        xyz = vector(pose["position_m"], 3, "endpoint XYZ")
        quaternion = vector(pose["quaternion_wxyz"], 4, "endpoint quaternion")
        if not math.isclose(sum(v*v for v in quaternion), 1., abs_tol=1e-6):
            raise ValueError("Endpoint quaternion must be normalized")
        if any(not lo <= v <= hi for v, (lo, hi) in zip(xyz, case["tool_workspace_m"])):
            raise ValueError("Endpoint lies outside declared tool workspace")
    planning = case["planning"]
    resolution, = vector([planning["edge_max_joint_delta_rad"]], 1, "edge resolution")
    clearance, = vector([planning.get("tracking_clearance_m", 0.)], 1, "planning tracking clearance")
    if clearance < 0 or any(2*clearance >= hi-lo for lo,hi in case["tool_workspace_m"]):
        raise ValueError("Tracking clearance must be nonnegative and leave a nonempty tool workspace")
    if resolution <= 0 or type(planning["max_queries"]) is not int or planning["max_queries"] <= 0 or type(planning["seed"]) is not int:
        raise ValueError("Planning resolution, budget and seed are invalid")
    for name in ("velocity_rad_s", "acceleration_rad_s2"):
        if any(v <= 0 for v in vector(case["motion_limits"][name], 7, name)):
            raise ValueError("Motion derivative limits must be positive")
    return tuple(map(float, base))


class JointScene:
    """Own a query-only Genesis scene. Never use this object as a live episode."""

    def __init__(self, case, base=(0., 0., 0.), *, render=False, debug_visuals=False):
        import genesis as gs
        self.gs = gs
        self.case = case
        self.base = validate_map(case, base)
        self.clearance = case["planning"].get("tracking_clearance_m", 0.)
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=.01, substeps=2),
            rigid_options=gs.options.RigidOptions(enable_self_collision=True,
                enable_neutral_collision=True, enable_adjacent_collision=False),
            vis_options=gs.options.VisOptions(ambient_light=(.5, .5, .5)),
            renderer=gs.renderers.Rasterizer(), show_viewer=False)
        self.scene.add_entity(gs.morphs.Plane(), name="floor")
        x, y, yaw = self.base
        self.arm = self.scene.add_entity(gs.morphs.MJCF(file=case["robot_model"],
            pos=(x, y, case["base_z_m"]), euler=(0., 0., math.degrees(yaw))), name="arm_0")
        self.obstacles = []
        for index, item in enumerate(case["obstacles"]):
            # Planning-only obstacle inflation reserves nominal tracking room.
            # Execution retains the declared physical map dimensions.
            low = np.array(item["lower_m"])-self.clearance
            high = np.array(item["upper_m"])+self.clearance
            # Query obstacles are movable only to retain fixed robot-base pairs.
            # This scene never steps, so their poses remain exactly the map poses.
            self.obstacles.append(self.scene.add_entity(gs.morphs.Box(size=high-low, pos=(low+high)/2, fixed=False),
                surface=gs.surfaces.Default(color=((.38,.48,.65),(.56,.41,.31),(.35,.55,.40))[index % 3]), name=item["id"]))
        self.camera = self.scene.add_camera(res=(960, 720), pos=(1.9, -1.6, 1.65),
            lookat=(.35, 0., .43), fov=42, GUI=False, debug=debug_visuals) if render else None
        self._before_build()
        build_started = time.monotonic()
        self.scene.build()
        self.build_wall_s = time.monotonic()-build_started
        self.hand = self.arm.get_link(case["tool_link"])
        self.dofs = [index for name in case["arm_joint_names"] for index in self.arm.get_joint(name).dofs_idx_local]
        self.q_indices = [index for name in case["arm_joint_names"] for index in self.arm.get_joint(name).qs_idx_local]
        if len(self.dofs) != 7 or self.dofs != list(range(7)) or self.q_indices != self.dofs or self.arm.n_qs != 9:
            raise ValueError("Installed robot joint mapping differs from the explicitly supported model")
        lower, upper = (array(v).reshape(-1)[:7] for v in self.arm.get_dofs_limit())
        self.bounds = tuple(zip(lower.tolist(), upper.tolist()))
        self.hand_workspace = np.array(case["tool_workspace_m"])+np.array([self.clearance,-self.clearance])
        self.fingers = self._query_finger_positions()
        self.qhome = (0., -.785, 0., -2.356, 0., 1.571, .785)
        self.arm.set_qpos([*self.qhome, *self.fingers])
        self._after_arm_q_placement()
        self.states_checked = 0
        self.edges_checked = 0
        self.collision_queries = 0
        self.failure_reasons = {}
        self.last_failure = None
        self.query_wall_s = 0.
        self.geometry_names = {g.idx: {"entity": g.entity.name, "link": g.link.name} for g in self.scene.rigid_solver.geoms}
        self._after_build()

    def _before_build(self):
        """Protected extension point for additional isolated query bodies."""

    def _after_build(self):
        """Protected extension point for built query-geometry verification."""

    def _query_finger_positions(self):
        return (.04, .04)

    def _after_arm_q_placement(self):
        """Protected extension point. Bare-arm queries need no extra action."""

    def _query_collision_pairs(self):
        pairs = self.arm.detect_collision()
        self.scene.rigid_solver.check_errno()
        self.collision_queries += 1
        return pairs

    def _additional_state_valid(self):
        """Protected opt-in constraints after placement. Bare behavior is unchanged."""
        return True

    def _reject(self, reason, **detail):
        self.failure_reasons[reason] = self.failure_reasons.get(reason, 0)+1
        self.last_failure = {"reason": reason, **detail}
        return False

    def last_query_diagnostic(self):
        """Detached retained failure data. Does not read or alter native state."""
        from transport_challenge.query_diagnostics import native_failure_diagnostic
        return native_failure_diagnostic(self.last_failure, self.geometry_names,
            getattr(self, "_last_detection", None), geometry={
                "scope": "Planning query geometry, not a physical collision measurement",
                "tracking_clearance_m": self.clearance,
                "edge_max_joint_delta_rad": self.case["planning"]["edge_max_joint_delta_rad"],
                "obstacle_geometry": "Planning-expanded boxes with the adapter's declared floor intersection. Nominal map geometry is separate."})

    def state_valid(self, q):
        started = time.monotonic()
        self.states_checked += 1
        try:
            if len(q) != len(self.bounds) or any(isinstance(v, (bool, np.bool_)) or not math.isfinite(float(v)) for v in q):
                return self._reject("invalid_joint_state")
            if any(not low <= v <= high for v, (low, high) in zip(q, self.bounds)):
                return self._reject("joint_limit")
            self.arm.set_qpos([*q, *self.fingers])
            self._after_arm_q_placement()
            xyz = array(self.hand.get_pos(relative=False)).reshape(3)
            if np.any(xyz < self.hand_workspace[:, 0]) or np.any(xyz > self.hand_workspace[:, 1]):
                return self._reject("tool_workspace", xyz_m=xyz.tolist())
            # The explicitly declared tool reference sphere is an additional
            # conservative task constraint. Full robot geometry is still checked.
            radius = self.case.get("tool_reference_radius_m", 0.)+self.clearance
            for obstacle in self.case["obstacles"]:
                nearest = np.maximum(obstacle["lower_m"], np.minimum(xyz, obstacle["upper_m"]))
                if float(np.linalg.norm(xyz-nearest)) <= radius:
                    return self._reject("tool_reference_obstacle", obstacle=obstacle["id"], xyz_m=xyz.tolist())
            if not self._additional_state_valid():
                return False
            pairs = self._query_collision_pairs()
            if len(pairs):
                return self._reject("full_model_collision", pairs=[{"geom_ids": [int(a), int(b)],
                    "a": self.geometry_names[int(a)], "b": self.geometry_names[int(b)]} for a, b in pairs])
            self.last_failure = None
            return True
        finally:
            self.query_wall_s += time.monotonic()-started

    def edge_valid(self, a, b):
        self.edges_checked += 1
        if len(a) != len(self.bounds) or len(b) != len(self.bounds):
            return self._reject("invalid_joint_edge")
        if any(not math.isfinite(float(v)) for v in (*a, *b)):
            return self._reject("invalid_joint_edge")
        resolution = self.case["planning"]["edge_max_joint_delta_rad"]
        count = max(1, math.ceil(max(abs(x-y) for x, y in zip(a, b))/resolution))
        return all(self.state_valid(tuple(x+(y-x)*i/count for x, y in zip(a, b))) for i in range(count+1))

    def endpoint_candidates(self, pose, *, seed=71, attempts=24):
        """IK creates endpoint configurations only. No path or intermediate pose."""
        rng = np.random.default_rng(seed)
        candidates, rejected = [], []
        lower, upper = np.array(self.bounds).T
        for attempt in range(attempts):
            initial = self.qhome if attempt == 0 else rng.uniform(lower, upper)
            solved = self.arm.inverse_kinematics(link=self.hand, pos=pose["position_m"],
                quat=pose["quaternion_wxyz"], init_qpos=np.array([*initial, *self.fingers]),
                respect_joint_limit=True, max_samples=1, max_solver_iters=80)
            q = array(solved).reshape(-1)[:7]
            self.arm.set_qpos([*q, *self.fingers])
            self._after_arm_q_placement()
            xyz, quat = array(self.hand.get_pos(relative=False)).reshape(3), array(self.hand.get_quat(relative=False)).reshape(4)
            pe = float(np.linalg.norm(xyz-np.array(pose["position_m"])))
            re = 2*math.acos(min(1., abs(float(np.dot(quat, pose["quaternion_wxyz"])))))
            if pe >= .003 or re >= .02:
                valid = self._reject("endpoint_ik_residual", position_error_m=pe, rotation_error_rad=re)
            else:
                valid = self.state_valid(q)
            record = {"q_rad": q.tolist(), "position_error_m": pe, "rotation_error_rad": re,
                      "valid": bool(valid), "failure": self.last_failure if not valid else None}
            if valid and not any(np.linalg.norm(q-np.array(item["q_rad"])) < .05 for item in candidates):
                candidates.append(record)
            elif not valid:
                rejected.append(record)
        return candidates, rejected

    def fk(self, q):
        """Provided kinematics only. This does not validate collision or a path."""
        if len(q) != 7 or any(not math.isfinite(float(v)) or not lo <= v <= hi
                for v,(lo,hi) in zip(q,self.bounds)):
            raise ValueError("FK requires seven finite native-bounded joint angles")
        self.arm.set_qpos([*q,*self.fingers])
        # Keep this supplied FK method usable by existing lightweight adapters.
        after_placement = getattr(self, "_after_arm_q_placement", None)
        if after_placement is not None:
            after_placement()
        return {"position_m":tuple(array(self.hand.get_pos(relative=False)).reshape(3)),
                "quaternion_wxyz":tuple(array(self.hand.get_quat(relative=False)).reshape(4))}

    def ik(self, position_m, quaternion_wxyz, seed_q, *, position_tolerance_m=None,
           orientation_tolerance_rad=None):
        """Solve one caller-chosen pose from a caller-chosen seven-joint seed.

        BoundedKinematics validates inputs and independently checks residuals.
        No sampled pose, route, collision exemption or fallback is supplied here.
        """
        precision = {}
        for key, value in (("pos_tol", position_tolerance_m), ("rot_tol", orientation_tolerance_rad)):
            if value is not None:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Explicit IK tolerances must be finite positive values")
                precision[key] = float(value)
        solved = self.arm.inverse_kinematics(link=self.hand, pos=position_m,
            quat=quaternion_wxyz, init_qpos=np.array([*seed_q,*self.fingers]),
            respect_joint_limit=True,max_samples=1,max_solver_iters=80,**precision)
        return tuple(array(solved).reshape(-1)[:7])

    def evidence(self):
        return {"base_pose_xy_yaw": self.base, "joint_bounds_rad": self.bounds,
            "arm_dofs_local": self.dofs, "finger_dofs_separate": [7, 8],
            "arm_q_indices_local": self.q_indices,
            "derivative_limits_source": "instructor chosen task constraints, not native XML limits",
            "additional_tool_reference_radius_m": self.case.get("tool_reference_radius_m", 0.),
            "planning_tracking_clearance_m":self.clearance,
            "clearance_scope":"Planning obstacles expanded in XYZ and tool workspace inset. Physical execution uses nominal map. No tracking-robustness proof.",
            "states_checked": self.states_checked, "edges_checked": self.edges_checked,
            "collision_queries": self.collision_queries, "failure_reasons": self.failure_reasons,
            "build_wall_s": self.build_wall_s, "query_wall_s": self.query_wall_s,
            "physics_steps": 0, "collision_geometry": "actual installed Genesis Panda and map geometry",
            "edge_sampling_max_joint_delta_rad": self.case["planning"]["edge_max_joint_delta_rad"],
            "continuous_collision_proof": False, "loaded_payload_checked": False,
            "planning_scene_only": True,
            "query_obstacles_movable_without_steps": True,
            "query_obstacle_reason": "Fixed-fixed pairs are filtered by Genesis. This retains fixed-base obstacle queries."}


def initialize_genesis(seed=71):
    versions = {n: importlib.metadata.version(n) for n in ("torch", "genesis-world")}
    if versions != {"torch": "2.14.0+cu126", "genesis-world": "1.3.2"}:
        raise RuntimeError("Existing environment version mismatch. No installation attempted.")
    import genesis as gs
    gs.init(backend=gs.cuda, precision="32", logging_level="warning", seed=seed)
    return gs, versions
