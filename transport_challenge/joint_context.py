"""Pure immutable map/design context for the opt-in joint-project hook.

All positions and AABBs are world-frame metres. Base yaw and joint angles are
radians. Joint velocity/acceleration limits are rad/s and rad/s². Map boxes are
nominal geometry, not a whole-robot or continuous collision certificate.

The joint planner can explicitly deliver this context to its student hook.
Physical outer design evaluation remains separate and incomplete. This module
supplies no route, intermediate pose or graph.
Legacy arm_bases_m XYZ and Cartesian speed_profile inputs are not accepted or
reinterpreted here. Native position bounds come from the caller's JointLimits;
this pure module cannot verify them against a built robot. Derivative limits
are the instructor's task constraints, not asserted hardware capabilities.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math

from .contracts import ContractError, digest, identifier, number, point
from .joint_kinematics import ToolPose
from .joint_planning import JointLimits


def _interval(value):
    low, high = point(value, 2)
    if low >= high or not math.isfinite(high-low):
        raise ContractError("Context intervals must have finite positive spans")
    return low, high


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractError(f"{label} must be a nonempty explicit string")
    return value


def _pose(value):
    if isinstance(value, ToolPose):
        return value
    if not isinstance(value, Mapping) or set(value) not in (
            {"position_m", "quaternion_wxyz"}, {"frame", "position_m", "quaternion_wxyz"}):
        raise ContractError("Endpoint requires world position_m and quaternion_wxyz")
    if "frame" in value and value["frame"] != "world":
        raise ContractError("Endpoint poses must be in the world frame")
    return ToolPose(value["position_m"], value["quaternion_wxyz"])


def _reject_solution_data(value):
    # Ordinary model file strings remain data. No generic nested metadata is
    # copied into the context, and explicit solution containers are rejected.
    forbidden = {"waypoints", "joint_waypoints", "pose_nodes", "nodes", "edges", "graph",
                 "route", "path", "trajectory", "intermediate_states", "intermediate_poses"}
    if isinstance(value, Mapping):
        if forbidden.intersection(value):
            raise ContractError("A task context cannot contain intermediate states, a graph or a route")
        for child in value.values():
            _reject_solution_data(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_solution_data(child)


@dataclass(frozen=True, kw_only=True)
class JointBasePlacement:
    """Explicit variable X/Y/yaw. Z is separately fixed by the task case."""
    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self):
        for name in ("x_m", "y_m", "yaw_rad"):
            object.__setattr__(self, name, number(getattr(self, name), name))

    def to_record(self):
        return {"x_m": self.x_m, "y_m": self.y_m, "yaw_rad": self.yaw_rad}


@dataclass(frozen=True)
class JointInstallationZone:
    x_m: tuple
    y_m: tuple
    yaw_rad: tuple

    def __post_init__(self):
        for name in ("x_m", "y_m", "yaw_rad"):
            object.__setattr__(self, name, _interval(getattr(self, name)))

    def contains(self, placement):
        if not isinstance(placement, JointBasePlacement):
            raise ContractError("Use an explicitly named JointBasePlacement, not an XYZ list")
        return all(low <= getattr(placement, name) <= high for name, (low, high) in
                   ((name, getattr(self, name)) for name in ("x_m", "y_m", "yaw_rad")))

    def to_record(self):
        return {name: list(getattr(self, name)) for name in ("x_m", "y_m", "yaw_rad")}


@dataclass(frozen=True)
class JointMapObstacle:
    obstacle_id: str
    lower_m: tuple
    upper_m: tuple

    def __post_init__(self):
        identifier(self.obstacle_id, "obstacle ID")
        lower, upper = point(self.lower_m, 3), point(self.upper_m, 3)
        if any(a >= b or not math.isfinite(b-a) for a, b in zip(lower, upper)):
            raise ContractError("Nominal obstacle AABB must have three finite positive extents")
        object.__setattr__(self, "lower_m", lower)
        object.__setattr__(self, "upper_m", upper)

    def to_record(self):
        return {"id": self.obstacle_id, "lower_m": list(self.lower_m), "upper_m": list(self.upper_m)}


@dataclass(frozen=True)
class JointTaskContext:
    case_id: str
    case_sha256: str
    robot_model: str
    joint_names: tuple
    tool_link: str
    base_placement: JointBasePlacement
    fixed_base_z_m: float
    installation_zones: tuple
    workspace_bounds_m: tuple
    nominal_obstacles: tuple
    start_tool_pose: ToolPose
    goal_tool_pose: ToolPose
    joint_limits: JointLimits
    derivative_limits_source: str
    query_tracking_clearance_m: float
    edge_max_joint_delta_rad: float
    start_endpoint_source: str = "case"
    goal_endpoint_source: str = "case"

    def __post_init__(self):
        identifier(self.case_id, "case ID")
        if (not isinstance(self.case_sha256, str) or len(self.case_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.case_sha256)):
            raise ContractError("Context requires a canonical case SHA256")
        _text(self.robot_model, "robot model")
        identifier(self.tool_link, "tool link")
        _text(self.derivative_limits_source, "derivative limit source")
        if not isinstance(self.joint_limits, JointLimits):
            raise ContractError("Use the supplied immutable JointLimits")
        if not isinstance(self.joint_names, (list, tuple)):
            raise ContractError("Joint order must be a list or tuple of names, never a string")
        names = tuple(self.joint_names)
        for name in names:
            identifier(name, "joint name")
        if len(names) != self.joint_limits.dimension or len(set(names)) != len(names):
            raise ContractError("Ordered joint names must be unique and match the native-bound dimension")
        if not isinstance(self.base_placement, JointBasePlacement):
            raise ContractError("Base must be explicitly named X/Y/yaw, never legacy XYZ")
        height = number(self.fixed_base_z_m, "fixed base height", minimum=0.)
        zones = tuple(self.installation_zones)
        if (not zones or any(not isinstance(z, JointInstallationZone) for z in zones)
                or not any(z.contains(self.base_placement) for z in zones)):
            raise ContractError("Chosen base must lie in an explicitly provided installation zone")
        workspace = tuple(_interval(pair) for pair in self.workspace_bounds_m)
        if len(workspace) != 3:
            raise ContractError("World tool workspace needs three XYZ intervals")
        obstacles = tuple(self.nominal_obstacles)
        if not obstacles or any(not isinstance(o, JointMapObstacle) for o in obstacles):
            raise ContractError("Context requires nominal physical obstacle AABBs")
        if len({o.obstacle_id for o in obstacles}) != len(obstacles):
            raise ContractError("Nominal obstacle IDs must be unique")
        for pose in (self.start_tool_pose, self.goal_tool_pose):
            if not isinstance(pose, ToolPose) or any(not lo <= x <= hi for x, (lo, hi) in zip(pose.position_m, workspace)):
                raise ContractError("World endpoint must lie in the nominal task workspace")
        if self.start_endpoint_source not in ("case", "explicit_world_pose") or self.goal_endpoint_source not in ("case", "explicit_world_pose"):
            raise ContractError("Endpoint origin must be explicit")
        clearance = number(self.query_tracking_clearance_m, "query clearance", minimum=0.)
        resolution = number(self.edge_max_joint_delta_rad, "edge sampling resolution", minimum=0.)
        if resolution == 0. or any(2*clearance >= hi-lo for lo, hi in workspace):
            raise ContractError("Positive edge resolution and a nonempty clearance-inset workspace are required")
        for name, value in (("joint_names", names), ("fixed_base_z_m", height), ("installation_zones", zones),
                            ("workspace_bounds_m", workspace), ("nominal_obstacles", obstacles),
                            ("query_tracking_clearance_m", clearance), ("edge_max_joint_delta_rad", resolution)):
            object.__setattr__(self, name, value)

    @classmethod
    def from_case(cls, case, *, base_placement, limits, start_tool_pose=None, goal_tool_pose=None):
        """Read declared geometry and explicit endpoints, without IK or route creation.

        Optional endpoint overrides are world ToolPose values or named records.
        They can describe an observed start or an explicit diagnostic goal. No
        observed grasp, native-model match or physical feasibility is inferred.
        """
        if not isinstance(case, dict) or case.get("schema_version") != "joint-map-v1":
            raise ContractError("Use an explicit joint-map-v1 case, not a legacy transport design")
        try:
            checksum = digest(case)  # Also rejects nonfinite/non-JSON/cyclic case data.
            _reject_solution_data(case)
            if not isinstance(limits, JointLimits):
                raise ContractError("Native positions and task derivative limits require JointLimits")
            motion = case["motion_limits"]
            if (point(motion["velocity_rad_s"], limits.dimension) != limits.velocity_rad_s
                    or point(motion["acceleration_rad_s2"], limits.dimension) != limits.acceleration_rad_s2):
                raise ContractError("Provided derivative limits must exactly match this case")
            return cls(case_id=case["case_id"], case_sha256=checksum, robot_model=case["robot_model"],
                joint_names=case["arm_joint_names"], tool_link=case["tool_link"],
                base_placement=base_placement, fixed_base_z_m=case["base_z_m"],
                installation_zones=tuple(JointInstallationZone(**zone) for zone in case["installation_zones"]),
                workspace_bounds_m=case["tool_workspace_m"],
                nominal_obstacles=tuple(JointMapObstacle(o["id"], o["lower_m"], o["upper_m"]) for o in case["obstacles"]),
                start_tool_pose=_pose(case["start_tool_pose"] if start_tool_pose is None else start_tool_pose),
                goal_tool_pose=_pose(case["goal_tool_pose"] if goal_tool_pose is None else goal_tool_pose),
                joint_limits=limits, derivative_limits_source=motion["source"],
                query_tracking_clearance_m=case["planning"].get("tracking_clearance_m", 0.),
                edge_max_joint_delta_rad=case["planning"]["edge_max_joint_delta_rad"],
                start_endpoint_source="case" if start_tool_pose is None else "explicit_world_pose",
                goal_endpoint_source="case" if goal_tool_pose is None else "explicit_world_pose")
        except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"Invalid joint context data: {exc}") from exc

    @property
    def base_position_world_m(self):
        return self.base_placement.x_m, self.base_placement.y_m, self.fixed_base_z_m

    def to_record(self):
        def pose(value):
            return {"frame": "world", "position_m": list(value.position_m), "quaternion_wxyz": list(value.quaternion_wxyz)}
        return {"schema_version": "joint-task-context-v1", "case_id": self.case_id, "case_sha256": self.case_sha256,
            "robot_model": self.robot_model, "joint_names": list(self.joint_names), "tool_link": self.tool_link,
            "base_placement": self.base_placement.to_record(), "fixed_base_z_m": self.fixed_base_z_m,
            "installation_zones": [z.to_record() for z in self.installation_zones],
            "workspace_bounds_m": [list(pair) for pair in self.workspace_bounds_m],
            "nominal_obstacles": [o.to_record() for o in self.nominal_obstacles],
            "start_tool_pose": pose(self.start_tool_pose), "goal_tool_pose": pose(self.goal_tool_pose),
            "start_endpoint_source": self.start_endpoint_source, "goal_endpoint_source": self.goal_endpoint_source,
            "joint_limits": self.joint_limits.to_record(), "derivative_limits_source": self.derivative_limits_source,
            "query_tracking_clearance_m": self.query_tracking_clearance_m,
            "edge_max_joint_delta_rad": self.edge_max_joint_delta_rad,
            "geometry_scope": "nominal world AABBs; native meshes, inflation and all edge queries remain owned by the supplied oracle",
            "integration_status": "IMMUTABLE_CONTEXT_PHYSICAL_EVALUATOR_NOT_CERTIFIED",
            "physical_feasibility_verified": False}
