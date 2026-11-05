"""Endpoint-only carry RRT and supplied bounded-yaw representation.

This development adapter does not plan empty approach, grasp, lift, placement or
return motions. Robot-link/IK feasibility still requires the physical runner.
"""
import copy
from dataclasses import asdict
import math

from .contracts import ContractError, RRTRequest, digest, number, point, validate_path_document
from .design import SPEED_PROFILES
from .geometry import (Box, CartesianOracle, ToolEnvelopeOracle, PayloadPoseOracle,
                       distance, path_length, quaternion_angle)
from .planning import _source_fingerprint


class CarryStateSpace:
    """Instructor-owned encoding, metric units and quaternion conversion."""
    def __init__(self, specification, workspace):
        self.specification = copy.deepcopy(specification)
        self.workspace = Box(*(point(v) for v in workspace))
        self.kind = specification.get("kind")
        self.bounds = tuple(zip(self.workspace.lower, self.workspace.upper))
        if self.kind == "xyz":
            if set(specification) != {"kind"}:
                raise ContractError("XYZ state space has no student-adjustable metric fields")
            self.dimension = 3
        elif self.kind == "xyz_scaled_yaw":
            if set(specification) != {"kind", "yaw_scale_m_per_rad", "yaw_bounds_rad"}:
                raise ContractError("Provide the fixed bounded-yaw state-space contract")
            self.scale = number(specification["yaw_scale_m_per_rad"], "yaw scale", minimum=0)
            lo, hi = point(specification["yaw_bounds_rad"], 2)
            if self.scale <= 0 or not -math.pi/2 <= lo < hi <= math.pi/2:
                raise ContractError("Bounded yaw must remain within [-pi/2,+pi/2] with positive scale")
            self.bounds += ((self.scale*lo,self.scale*hi),)
            self.dimension = 4
        else:
            raise ContractError("Unsupported free carry state space")

    def endpoint(self, transfer, name):
        xyz = point(transfer[name])
        return xyz if self.dimension == 3 else xyz+(self.scale*number(transfer[name+"_yaw_rad"],"endpoint yaw"),)

    def pose(self, state):
        state = point(state,self.dimension)
        if any(not lo <= v <= hi for v,(lo,hi) in zip(state,self.bounds)):
            raise ContractError("Carry state leaves instructor bounds")
        yaw = state[3]/self.scale if self.dimension == 4 else 0
        return {"xyz_m":list(state[:3]), "quaternion_wxyz":[0.0,math.cos(yaw/2),math.sin(yaw/2),0.0]}


class _CarryOracle:
    def __init__(self, case, space):
        self.space = space
        boxes = [Box(*(point(v) for v in box)) for box in case["obstacles_m"]]
        arguments = (space.workspace,boxes)
        if space.dimension == 4:
            self.oracle = PayloadPoseOracle(*arguments,case["payload_half_size_m"],case["payload_center_in_hand_m"],
                case["proxy_clearance_m"],case.get("physical_constraints"))
        elif "payload_envelope_relative_to_hand_m" in case:
            self.oracle = ToolEnvelopeOracle(*arguments,Box(*(point(v) for v in case["payload_envelope_relative_to_hand_m"])),
                case["proxy_clearance_m"],case.get("physical_constraints"))
        else:
            self.oracle = CartesianOracle(*arguments,case["proxy_clearance_m"],case.get("physical_constraints"))

    def state_valid(self, state):
        try:
            pose = self.space.pose(state)
        except ContractError:
            return False
        return self.oracle.state_valid(pose if self.space.dimension == 4 else pose["xyz_m"])

    def edge_valid(self, first, second):
        if not self.state_valid(first) or not self.state_valid(second):
            return False
        return self.oracle.edge_valid(self.space.pose(first),self.space.pose(second)) if self.space.dimension == 4 else self.oracle.edge_valid(first,second)


def _checked_path(path, request, oracle, *, raw=False):
    if path is None:
        raise ContractError("RRT found no route within the fixed sampling budget")
    if not isinstance(path,(tuple,list)) or not 2 <= len(path) <= (request.max_iterations+2 if raw else 200):
        raise ContractError("Carry route needs at least two states within the supplied path-size limit")
    path = tuple(point(p,len(request.bounds)) for p in path)
    if path[0] != request.start or path[-1] != request.goal:
        raise ContractError("Carry planner/refinement must preserve exact endpoint states")
    if not all(oracle.state_valid(p) for p in path) or not all(oracle.edge_valid(a,b) for a,b in zip(path,path[1:])):
        raise ContractError("Carry planner/refinement returned a task-invalid or colliding proxy edge")
    if raw and any(distance(a,b)>request.step_size+1e-10 for a,b in zip(path,path[1:])):
        raise ContractError("Raw RRT exceeds the supplied extension step limit")
    return path


def free_route_document(case, strategy, *, speed_profile="careful"):
    """Actually call student RRT once per carry using only start/goal and oracle."""
    mode = case.get("planning_mode")
    if mode not in ("endpoint_rrt","continuous_yaw_rrt") or case.get("motion_scope") != "carry_only_development":
        raise ContractError("Use the explicit endpoint-only carry development contract")
    if any(key in case for key in ("pose_nodes","candidate_carry_heights_m","deposit_segment","deposit_segments")):
        raise ContractError("Free-design case must not prescribe an intermediate path or deposit route")
    space = CarryStateSpace(case["rrt_state_space"],case["workspace_m"])
    if (mode == "continuous_yaw_rrt") != (space.dimension == 4):
        raise ContractError("Planning mode and supplied state space disagree")
    oracle = _CarryOracle(case,space)
    budget = case["rrt_budget"]
    attempts = budget["attempts_per_transfer"]
    transfers = case["transfers"]
    ids = [t["id"] for t in transfers]
    if (type(attempts) is not int or attempts <= 0 or len(set(ids)) != len(ids)
            or budget["total_sampling_attempts"] != attempts*len(transfers)
            or set(budget["seeds_by_segment"]) != set(ids)):
        raise ContractError("Instructor RRT budget must bind all transfers and fixed seeds")
    options = case.get("selected_rrt_options",case["rrt_defaults"])
    if set(options) != {"step_size_m_equiv","goal_bias"}:
        raise ContractError("RRT options cannot replace teacher budget, metric or seeds")
    profiles = case.get("selected_segment_speed_profiles",{})
    if speed_profile not in SPEED_PROFILES or not set(profiles)<=set(ids):
        raise ContractError("Unknown speed profile or transfer-specific speed target")
    segments,calls = [],[]
    refine = getattr(strategy,"refine_path",None)
    for transfer in transfers:
        if any(key in transfer for key in ("pose_nodes","extra_nodes","nodes","graph","waypoints","quaternion_waypoints")):
            raise ContractError("Free-design transfer must expose endpoints without intermediate route states")
        start,goal = space.endpoint(transfer,"start"),space.endpoint(transfer,"goal")
        request = RRTRequest(start,goal,space.bounds,options["step_size_m_equiv"],attempts,
                             budget["seeds_by_segment"][transfer["id"]],options["goal_bias"])
        raw = _checked_path(strategy.plan_rrt(request,oracle),request,oracle,raw=True)
        path = _checked_path(refine(raw,oracle.edge_valid) if refine is not None else raw,request,oracle)
        poses = [space.pose(p) for p in path]
        profile = profiles.get(transfer["id"],speed_profile)
        if profile not in SPEED_PROFILES:
            raise ContractError("Use a supplied speed profile")
        # The provided executor allocates XYZ distance + .10 m/rad of rotation.
        # RRT's Euclidean encoded metric is distinct from this timing convention.
        timing_length = sum(distance(a["xyz_m"],b["xyz_m"])+.10*quaternion_angle(a["quaternion_wxyz"],b["quaternion_wxyz"])
                            for a,b in zip(poses,poses[1:]))
        duration = max(1.0,timing_length/SPEED_PROFILES[profile]["nominal_path_progress_speed_m_s"])
        if duration > 30:
            raise ContractError("Carry exceeds supplied 30-s segment execution limit. Improve the route or choose a permitted speed.")
        segment = {"id":transfer["id"],"robot_id":transfer["robot_id"],"object_id":transfer["object_id"],
                   "phase":"transfer","entry_phase":transfer.get("entry_phase","after_lift"),
                   "waypoints":[p["xyz_m"] for p in poses],"duration_s":duration,"speed_profile":profile}
        if space.dimension == 4:
            segment["quaternion_waypoints"] = [p["quaternion_wxyz"] for p in poses]
        segments.append(segment)
        calls.append({"segment_id":transfer["id"],"request":asdict(request),
            "raw_rrt_states":[list(p) for p in raw],"raw_rrt_states_sha256":digest(raw),
            "refined_states":[list(p) for p in path],"refined_states_sha256":digest(path),
            "exported_waypoints":copy.deepcopy(segment["waypoints"]),
            "exported_quaternion_waypoints":copy.deepcopy(segment.get("quaternion_waypoints")),
            "direct_edge_proxy_valid":oracle.edge_valid(start,goal),"refinement_hook_called":refine is not None})
    document = {"schema_version":"transport-path-v1","coordinate_frame":"world_xyz_m",
        "case_id":case["case_id"],"scenario_content_sha256":case.get("source_scenario_content_sha256",digest(case)),
        "algorithm":"endpoint_rrt","motion_scope":"carry_only_development","segments":segments,
        "speed_profile":speed_profile,"rrt_provenance":{"schema_version":"free-rrt-export-provenance-v1",
            "state_space":space.specification,"workspace_m":case["workspace_m"],"budget":copy.deepcopy(budget),
            "plan_rrt_source_sha256":_source_fingerprint(strategy.plan_rrt),
            "refine_path_source_sha256":_source_fingerprint(refine) if refine is not None else None,"calls":calls},
        "validation_scope":"Endpoint-only student RRT for loaded carry. Supplied empty approach/grasp/lift/lower/retreat/return remain. Payload proxy excludes full robot links and IK. Bases are physically bound, not part of this proxy oracle."}
    if "design_content_sha256" in case:
        document["design_content_sha256"] = case["design_content_sha256"]
    return validate_path_document(document,require_rrt_provenance=True)


def validate_free_provenance(document):
    """Supplied consistency boundary. Not proof of algorithm or physical success."""
    provenance = document["rrt_provenance"]
    if document.get("algorithm") != "endpoint_rrt" or document.get("motion_scope") != "carry_only_development":
        raise ContractError("Free RRT provenance requires the explicit loaded-carry scope")
    space = CarryStateSpace(provenance["state_space"],provenance["workspace_m"])
    calls,segments = provenance.get("calls"),document["segments"]
    if not isinstance(calls,list) or len(calls)!=len(segments) or not calls:
        raise ContractError("Free RRT provenance must cover every carry exactly once")
    budget=provenance["budget"]
    if budget["total_sampling_attempts"]!=budget["attempts_per_transfer"]*len(segments):
        raise ContractError("Free RRT total sampling budget is inconsistent")
    for call,segment in zip(calls,segments):
        if call.get("segment_id")!=segment["id"]:
            raise ContractError("Free RRT trace and executed carry identity/order differ")
        try:
            request=RRTRequest(**call["request"])
        except (TypeError,KeyError) as error:
            raise ContractError("Invalid free RRT request record") from error
        if (tuple(tuple(v) for v in request.bounds)!=space.bounds or request.max_iterations!=budget["attempts_per_transfer"]
                or request.seed!=budget["seeds_by_segment"][segment["id"]]):
            raise ContractError("Free RRT request changed teacher bounds, seed or attempt budget")
        for key in ("raw_rrt_states","refined_states"):
            values=call.get(key)
            if not isinstance(values,(list,tuple)) or len(values)<2:
                raise ContractError("Free RRT requires nontrivial raw and refined state paths")
            checked=[point(p,space.dimension) for p in values]
            if checked[0]!=tuple(request.start) or checked[-1]!=tuple(request.goal):
                raise ContractError("Free RRT state endpoints do not match request")
            for value in checked:
                space.pose(value)
            if call.get(key+"_sha256")!=digest(values):
                raise ContractError("Free RRT state content hash mismatch")
            if key=="raw_rrt_states" and any(distance(a,b)>request.step_size+1e-10 for a,b in zip(checked,checked[1:])):
                raise ContractError("Raw free RRT path exceeds its step limit")
        poses=[space.pose(p) for p in call["refined_states"]]
        xyz=[p["xyz_m"] for p in poses]
        quats=[p["quaternion_wxyz"] for p in poses] if space.dimension==4 else None
        if (segment["waypoints"]!=xyz or call.get("exported_waypoints")!=xyz
                or segment.get("quaternion_waypoints")!=quats or call.get("exported_quaternion_waypoints")!=quats):
            raise ContractError("Free RRT encoded states do not map exactly to executed XYZ/quaternions")
