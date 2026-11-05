"""Provided graph construction and honest student-route export, without physics."""

import copy
from dataclasses import asdict
import hashlib
import inspect

from .contracts import ContractError, RRTRequest, digest, number, point, validate_path_document
from .design import SPEED_PROFILES
from .geometry import (Box, CartesianOracle, ToolEnvelopeOracle, PayloadPoseOracle, FixedOrientationPoseOracle,
                       distance, path_length, quaternion_angle)


def build_visibility_graph(nodes, oracle):
    """Only creates candidate edges. The selected route comes from student code."""
    return {name: {other: distance(position, target)
                   for other, target in nodes.items()
                   if other != name and oracle.edge_valid(position, target)}
            for name, position in nodes.items()}


def _source_fingerprint(function):
    """Audit the actual called function. This is not algorithm certification."""
    try:
        source = inspect.getsource(function).encode("utf-8")
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source).hexdigest()


def _checked_xyz_path(path, start, goal, oracle, *, step_size=None):
    if path is None:
        raise ContractError("RRT found no route within the supplied sampling budget")
    if not isinstance(path, (list, tuple)) or not path:
        raise ContractError("Planner must return a nonempty path or None")
    path = tuple(point(value) for value in path)
    if path[0] != tuple(start) or path[-1] != tuple(goal):
        raise ContractError("Planner must preserve exact segment endpoints")
    if not all(oracle.state_valid(value) for value in path) or not all(
            oracle.edge_valid(a, b) for a, b in zip(path, path[1:])):
        raise ContractError("Planner returned a proxy-colliding or task-invalid edge")
    if step_size is not None and any(distance(a, b) > step_size + 1e-10 for a, b in zip(path, path[1:])):
        raise ContractError("Raw RRT extensions and final goal connection must respect step_size")
    return path


def _rrt_interval(case, strategy, oracle, start, goal, seed, segment_id):
    request = RRTRequest(tuple(start), tuple(goal), tuple(zip(*case["workspace_m"])),
                         0.07, 1500, seed)
    raw = _checked_xyz_path(strategy.plan_rrt(request, oracle), start, goal, oracle,
                            step_size=request.step_size)
    shortened = _checked_xyz_path(strategy.shorten_path(raw, oracle.edge_valid), start, goal, oracle)
    trace = {"segment_id": segment_id, "request": asdict(request),
             "direct_edge_proxy_valid": oracle.edge_valid(start, goal),
             "raw_rrt_waypoints": [list(p) for p in raw], "raw_rrt_path_sha256": digest(raw),
             "exported_waypoints": [list(p) for p in shortened], "exported_path_sha256": digest(shortened)}
    return shortened, trace


def _rrt_provenance(strategy, calls):
    return {"schema_version": "rrt-export-provenance-v1", "calls": calls,
            "plan_rrt_source_sha256": _source_fingerprint(strategy.plan_rrt),
            "shorten_path_source_sha256": _source_fingerprint(strategy.shorten_path),
            "scope": "Actual planner calls and raw-to-export linkage. Does not certify the algorithm implementation, physical execution or full robot collision safety."}


def _same_orientation(first, second):
    # q and -q encode the same orientation. Component comparison avoids the
    # acos amplification of roundoff in an identical non-cardinal quaternion.
    return min(distance(first, second), distance(first, tuple(-v for v in second))) <= 1e-8


def route_document(case, strategy, *, algorithm="astar", speed_profile="careful", bend_x_m=0.49):
    """Export tool-centre points the physical adapter must actually execute.

    Candidate waypoints are not a certified full-body robot path. No physical
    quantity, measured throughput or mission-success flag is produced here.
    """
    workspace = Box(tuple(case["workspace_m"][0]), tuple(case["workspace_m"][1]))
    obstacles = tuple(Box(tuple(b[0]), tuple(b[1])) for b in case["obstacles_m"])
    if "payload_envelope_relative_to_hand_m" in case:
        envelope = Box(*(tuple(v) for v in case["payload_envelope_relative_to_hand_m"]))
        oracle = ToolEnvelopeOracle(workspace, obstacles, envelope, case["proxy_clearance_m"], case.get("physical_constraints"))
    else:
        oracle = CartesianOracle(workspace, obstacles, case["proxy_clearance_m"], case.get("physical_constraints"))
    number(bend_x_m, "bend x", minimum=workspace.lower[0], maximum=workspace.upper[0])
    speeds = {name: profile["nominal_path_progress_speed_m_s"] for name, profile in SPEED_PROFILES.items()}
    if speed_profile not in speeds:
        raise ContractError("Choose a supplied speed profile")
    segments, total_length, rrt_calls = [], 0.0, []
    for index, transfer in enumerate(case["transfers"]):
        start, goal = tuple(transfer["start"]), tuple(transfer["goal"])
        if algorithm == "astar":
            nodes = {"start": start, "goal": goal}
            for i, y in enumerate((start[1], (start[1] + goal[1]) / 2, goal[1])):
                nodes[f"bend_{i}"] = (bend_x_m, y, start[2])
            for i, location in enumerate(transfer.get("extra_nodes", [])):
                nodes[f"extra_{i}"] = tuple(location)
            for level in case.get("candidate_carry_heights_m", []):
                for i, position in enumerate((start, goal)):
                    nodes[f"high_{level}_{i}"] = (position[0], position[1], level)
                for i, y in enumerate((start[1], (start[1] + goal[1]) / 2, goal[1])):
                    nodes[f"high_bend_{level}_{i}"] = (bend_x_m, y, level)
            graph = build_visibility_graph(nodes, oracle)
            ids = strategy.plan_graph(graph, "start", "goal", lambda n: distance(nodes[n], goal))
            path = None if ids is None else tuple(nodes[n] for n in ids)
        elif algorithm == "rrt":
            path, trace = _rrt_interval(case, strategy, oracle, start, goal, case.get("seed", 17) + index,
                                        transfer.get("id", f"transfer_{index}"))
            trace["executed_waypoint_index_range"] = [0, len(path)-1]
            rrt_calls.append(trace)
        else:
            raise ContractError("Algorithm must be astar or rrt")
        if path is None:
            raise ContractError(f"No proxy-valid route for transfer {index}")
        if tuple(path[0]) != start or tuple(path[-1]) != goal:
            raise ContractError("Planner must preserve exact segment endpoints")
        if not all(oracle.edge_valid(a, b) for a, b in zip(path, path[1:])):
            raise ContractError("Planner returned a proxy-colliding edge")
        if algorithm == "astar":
            path = strategy.shorten_path(path, oracle.edge_valid)
        if tuple(path[0]) != start or tuple(path[-1]) != goal or not all(
                oracle.edge_valid(a, b) for a, b in zip(path, path[1:])):
            raise ContractError("Shortcut returned an invalid path")
        length = path_length(path)
        total_length += length
        segments.append({"id": transfer.get("id", f"transfer_{index}"), "robot_id": transfer["robot_id"],
                         "phase": "transfer", "object_id": transfer.get("object_id", "payload"),
                         "waypoints": [list(p) for p in path],
                         "duration_s": max(0.1, length / speeds[speed_profile])})
    document = {"schema_version": "transport-path-v1", "coordinate_frame": "world_xyz_m",
                "case_id": case["case_id"], "scenario_content_sha256": case.get("source_scenario_content_sha256", digest(case)), "algorithm": algorithm,
                "speed_profile": speed_profile, "segments": segments,
                "proxy_path_length_m": total_length,
                "validation_scope": "Cartesian proxy only. Genesis execution required."}
    if algorithm == "rrt":
        document["rrt_provenance"] = _rrt_provenance(strategy, rrt_calls)
    if "design_content_sha256" in case:
        document["design_content_sha256"] = case["design_content_sha256"]
    if "deposit_segment" in case:
        document["deposit_segment"] = copy.deepcopy(case["deposit_segment"])
    if "deposit_segments" in case:
        document["deposit_segments"] = copy.deepcopy(case["deposit_segments"])
    return validate_path_document(document, require_rrt_provenance=algorithm == "rrt")


def joint_graph_document(case, strategy):
    """Select from supplied, checked IK configurations and full-edge checks.

    The instructor physical adapter produces the graph. Each node contains seven
    joint angles and represents an orientation as well as a position. Students
    select the weighted route. No unchecked edge is manufactured here.
    """
    if case.get("coordinate_frame") != "joint_rad":
        raise ContractError("Joint graph case must use joint_rad")
    if case.get("edge_validation_status") != "actual_genesis_checked":
        raise ContractError("Joint edges require actual instructor physical validation")
    segments = []
    for task in case["transfers"]:
        nodes, graph = task["nodes"], task["graph"]
        ids = strategy.plan_graph(graph, task["start_node"], task["goal_node"], lambda _: 0)
        if ids is None:
            raise ContractError("No joint route through supplied verified edges")
        if len(ids) < 2:
            raise ContractError("A physical transfer must contain a nontrivial joint route")
        segments.append({"id": task["id"], "robot_id": task["robot_id"],
                         "phase": "transfer", "object_id": task["object_id"],
                         "joint_waypoints": [nodes[node] for node in ids],
                         "duration_s": task["duration_s"]})
    return validate_path_document({"schema_version": "transport-path-v1", "coordinate_frame": "joint_rad",
                                   "case_id": case["case_id"], "scenario_content_sha256": digest(case),
                                   "segments": segments,
                                   "validation_scope": "Supplied joint graph checks. Replay still required."})


def pose_graph_document(case, strategy, *, speed_profile="careful", algorithm="astar"):
    """Select orientation transitions, then optionally plan fixed-pose XYZ carries.

    The RRT option actually calls the student's existing three-dimensional RRT.
    Each maximal fixed-quaternion interval is replanned between its selected
    endpoints. Orientation-changing graph edges remain intact and explicit.
    There is no RRT-to-graph fallback, and no new student quaternion metric.
    """
    if algorithm not in ("astar", "rrt"):
        raise ContractError("Algorithm must be astar or rrt")
    oracle = PayloadPoseOracle(Box(*(tuple(v) for v in case["workspace_m"])),
                               [Box(*(tuple(v) for v in box)) for box in case["obstacles_m"]],
                               case["payload_half_size_m"], case["payload_center_in_hand_m"],
                               case["proxy_clearance_m"], case.get("physical_constraints"))
    if speed_profile not in SPEED_PROFILES:
        raise ContractError("Choose a supplied speed profile")
    linear_speed = SPEED_PROFILES[speed_profile]["nominal_path_progress_speed_m_s"]
    rotation_weight = case["rotation_cost_m_per_rad"]
    segments, rrt_calls = [], []
    for transfer in case["transfers"]:
        nodes = transfer["pose_nodes"]
        def cost(a, b):
            return distance(a["xyz_m"], b["xyz_m"]) + rotation_weight * quaternion_angle(a["quaternion_wxyz"], b["quaternion_wxyz"])
        graph = {name: {other: cost(pose, destination) for other, destination in nodes.items()
                        if name != other and oracle.edge_valid(pose, destination)}
                 for name, pose in nodes.items()}
        route = strategy.plan_graph(graph, transfer["start_node"], transfer["goal_node"], lambda _: 0)
        if route is None:
            raise ContractError(f"No supplied pose-graph route for {transfer['id']}")
        if (not route or route[0] != transfer["start_node"] or route[-1] != transfer["goal_node"]
                or any(name not in nodes for name in route)
                or any(b not in graph[a] for a, b in zip(route, route[1:]))):
            raise ContractError("Pose graph search must return a valid endpoint-preserving route")
        selected = [copy.deepcopy(nodes[name]) for name in route]
        poses, intervals = [selected[0]], []
        cursor = 0
        while cursor < len(selected)-1:
            last = cursor
            if algorithm == "rrt":
                while last+1 < len(selected) and _same_orientation(
                        selected[cursor]["quaternion_wxyz"], selected[last+1]["quaternion_wxyz"]):
                    last += 1
            first_index = len(poses)-1
            if last > cursor and distance(selected[cursor]["xyz_m"], selected[last]["xyz_m"]) > 1e-12:
                fixed_q = selected[cursor]["quaternion_wxyz"]
                xyz_oracle = FixedOrientationPoseOracle(oracle, fixed_q)
                xyz_path, trace = _rrt_interval(case, strategy, xyz_oracle,
                    selected[cursor]["xyz_m"], selected[last]["xyz_m"],
                    case.get("seed", 17)+len(rrt_calls), transfer["id"])
                poses.extend({"xyz_m": list(p), "quaternion_wxyz": list(fixed_q)} for p in xyz_path[1:])
                poses[-1] = copy.deepcopy(selected[last])
                trace["fixed_quaternion_wxyz"] = list(fixed_q)
                trace["selected_node_ids"] = list(route[cursor:last+1])
                trace["executed_waypoint_index_range"] = [first_index, len(poses)-1]
                rrt_calls.append(trace)
                source = "student_rrt_xyz_then_shortening"
            else:
                last = cursor+1
                poses.append(copy.deepcopy(selected[last]))
                source = "student_graph_selected_supplied_pose_edge"
            intervals.append({"source": source, "selected_node_ids": list(route[cursor:last+1]),
                              "executed_waypoint_index_range": [first_index, len(poses)-1]})
            cursor = last
        if not all(oracle.edge_valid(a, b) for a, b in zip(poses, poses[1:])):
            raise ContractError("Composite pose route contains an invalid swept-payload edge")
        segments.append({"id": transfer["id"], "robot_id": transfer["robot_id"],
                         "object_id": transfer["object_id"], "phase": "transfer",
                         "entry_phase": transfer.get("entry_phase", "after_lift"),
                         "waypoints": [p["xyz_m"] for p in poses],
                         "quaternion_waypoints": [p["quaternion_wxyz"] for p in poses],
                         "selected_node_ids": list(route),
                         "planning_intervals": intervals,
                         "duration_s": max(0.1, sum(cost(a,b) for a,b in zip(poses,poses[1:])) / linear_speed)})
    if algorithm == "rrt" and not rrt_calls:
        raise ContractError("Requested RRT, but the selected pose route has no nontrivial fixed-orientation XYZ carry interval")
    document = {"schema_version": "transport-path-v1", "coordinate_frame": "world_xyz_m",
                "case_id": case["case_id"], "scenario_content_sha256": case.get("source_scenario_content_sha256", digest(case)),
                "speed_profile": speed_profile, "segments": segments,
                "validation_scope": "Conservative swept payload box only. Actual robot/shelf replay required."}
    document["algorithm"] = "pose_graph_rrt_xyz" if algorithm == "rrt" else "pose_graph_astar"
    if algorithm == "rrt":
        document["rrt_provenance"] = _rrt_provenance(strategy, rrt_calls)
    if "design_content_sha256" in case:
        document["design_content_sha256"] = case["design_content_sha256"]
    return validate_path_document(document, require_rrt_provenance=algorithm == "rrt")
