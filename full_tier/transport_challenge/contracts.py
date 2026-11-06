"""Provided immutable objects, finite-number guards and route-file boundary."""

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping
from types import MappingProxyType


class ContractError(ValueError):
    """A supplied input or execution record violates the published contract."""


def number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ContractError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ContractError(f"{name} must be at most {maximum}")
    return result


def point(value, dimension=3):
    if not isinstance(value, (list, tuple)) or len(value) != dimension:
        raise ContractError(f"Expected {dimension} coordinates")
    return tuple(number(x, "coordinate") for x in value)


def identifier(value, name="identifier"):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty string")
    return value


def load_json(path):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ContractError(f"Duplicate JSON key: {key}")
            obj[key] = value
        return obj
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle, object_pairs_hook=unique,
                         parse_constant=lambda v: (_ for _ in ()).throw(
                             ContractError(f"Non-finite JSON number: {v}")))


def save_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_team(difficulty, members):
    allowed = {"easy": (1,), "medium": (1, 2), "hard": (2, 3)}
    if difficulty not in allowed:
        raise ContractError("Difficulty must be easy, medium or hard")
    if type(members) is not int or members not in allowed[difficulty]:
        raise ContractError(f"{difficulty} permits team sizes {allowed[difficulty]}")


@dataclass(frozen=True)
class Candidate:
    """Student choice: corridor x in metres, supplied profile, number of arms."""
    bend_x_m: float
    speed_profile: str
    arm_count: int


@dataclass(frozen=True)
class Evaluation:
    """One supplied evaluation. Utility is higher-is-better, feasible is a bool.

    Metric provenance is recorded outside this small selection interface.
    A CPU synthetic test Evaluation is never a physical measurement.
    """
    candidate: Candidate
    feasible: bool
    utility: float


@dataclass(frozen=True)
class RRTRequest:
    start: tuple[float, ...]
    goal: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]
    step_size: float
    max_iterations: int
    seed: int
    goal_bias: float = 0.15

    def __post_init__(self):
        dim = len(self.bounds)
        if dim < 1:
            raise ContractError("RRT needs at least one dimension")
        point(self.start, dim)
        point(self.goal, dim)
        for index, limits in enumerate(self.bounds):
            lo, hi = point(limits, 2)
            if lo >= hi or not lo <= self.start[index] <= hi or not lo <= self.goal[index] <= hi:
                raise ContractError("RRT endpoints and ordered bounds must agree")
        if number(self.step_size, "step_size", minimum=0) == 0:
            raise ContractError("RRT step size must be positive")
        if type(self.max_iterations) is not int or self.max_iterations < 0:
            raise ContractError("RRT iteration budget must be a nonnegative integer")
        if type(self.seed) is not int:
            raise ContractError("RRT seed must be an integer")
        number(self.goal_bias, "goal_bias", minimum=0, maximum=1)


@dataclass(frozen=True)
class PlanningRequest:
    """Algorithm-neutral endpoint query, with one shared validity-query budget."""
    start: tuple[float, ...]
    goal: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]
    max_queries: int
    seed: int
    options: Mapping = field(default_factory=dict)

    def __post_init__(self):
        bounds=tuple(point(pair,2) for pair in self.bounds)
        if not bounds:
            raise ContractError("Planning needs at least one coordinate")
        start,goal=point(self.start,len(bounds)),point(self.goal,len(bounds))
        if any(lo>=hi or not lo<=start[i]<=hi or not lo<=goal[i]<=hi for i,(lo,hi) in enumerate(bounds)):
            raise ContractError("Planning endpoints must lie within ordered bounds")
        if type(self.max_queries) is not int or self.max_queries<0 or type(self.seed) is not int:
            raise ContractError("Planning query budget must be a nonnegative integer and seed an integer")
        if not isinstance(self.options,Mapping):
            raise ContractError("Planner options must be a mapping")
        copied={}
        for key,value in self.options.items():
            identifier(key,"planner option")
            if isinstance(value,str):
                identifier(value,"planner option value")
                copied[key]=value
            else:
                copied[key]=number(value,key)
        object.__setattr__(self,"start",start)
        object.__setattr__(self,"goal",goal)
        object.__setattr__(self,"bounds",bounds)
        object.__setattr__(self,"options",MappingProxyType(copied))

    def to_record(self):
        return {"start":list(self.start),"goal":list(self.goal),"bounds":[list(pair) for pair in self.bounds],
                "max_queries":self.max_queries,"seed":self.seed,"options":dict(self.options)}


class PlanningBudgetExceeded(ContractError):
    """No further student-facing validity query is allowed for this carry."""


@dataclass(frozen=True)
class ValidityOracle:
    """Provided state and whole-edge checks in the request's coordinate space."""
    state_valid: Callable
    edge_valid: Callable


def validate_graph(graph: Mapping, start, goal):
    if start not in graph or goal not in graph:
        raise ContractError("Start and goal must be graph nodes")
    for node, neighbors in graph.items():
        identifier(node, "graph node")
        for neighbor, cost in neighbors.items():
            if neighbor not in graph:
                raise ContractError("Every neighbor must have a node entry")
            number(cost, "edge cost in case-defined units", minimum=0)


def _validate_rrt_provenance(document):
    provenance = document["rrt_provenance"]
    if isinstance(provenance, dict) and provenance.get("schema_version") == "free-rrt-export-provenance-v1":
        from .free_planning import validate_free_provenance
        return validate_free_provenance(document)
    if not isinstance(provenance, dict) or provenance.get("schema_version") != "rrt-export-provenance-v1":
        raise ContractError("Unknown RRT export provenance schema")
    calls = provenance.get("calls")
    if not isinstance(calls, list) or not calls:
        raise ContractError("RRT provenance requires nonempty recorded planner calls")
    if document.get("coordinate_frame") != "world_xyz_m" or document.get("algorithm") not in ("rrt", "pose_graph_rrt_xyz"):
        raise ContractError("RRT provenance requires a supported XYZ RRT route")
    segments = {s["id"]: s for s in document["segments"]}
    ranges = {name: [] for name in segments}
    for call in calls:
        if not isinstance(call, dict) or call.get("segment_id") not in segments:
            raise ContractError("RRT call must name an exported transfer segment")
        segment = segments[call["segment_id"]]
        span = call.get("executed_waypoint_index_range")
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or any(type(v) is not int for v in span)
                or not 0 <= span[0] < span[1] < len(segment["waypoints"])):
            raise ContractError("RRT call needs a nontrivial executed waypoint index range")
        ranges[segment["id"]].append(tuple(span))
        try:
            request = RRTRequest(**call["request"])
        except (TypeError, KeyError) as error:
            raise ContractError("RRT provenance request does not match RRTRequest") from error
        if len(request.start) != 3:
            raise ContractError("Exported carry RRT must use three-dimensional XYZ")
        raw = call.get("raw_rrt_waypoints")
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            raise ContractError("RRT provenance requires a nontrivial raw path")
        checked_raw = tuple(point(p) for p in raw)
        if checked_raw[0] != tuple(request.start) or checked_raw[-1] != tuple(request.goal):
            raise ContractError("RRT raw path and request endpoints differ")
        if any(not lo <= value <= hi for p in checked_raw for value, (lo, hi) in zip(p, request.bounds)):
            raise ContractError("Raw RRT path leaves its request bounds")
        if any(math.dist(a, b) > request.step_size+1e-10 for a, b in zip(checked_raw, checked_raw[1:])):
            raise ContractError("Raw RRT path exceeds the extension step limit")
        exported = segment["waypoints"][span[0]:span[1]+1]
        if (call.get("exported_waypoints") != exported
                or tuple(exported[0]) != tuple(request.start) or tuple(exported[-1]) != tuple(request.goal)):
            raise ContractError("RRT exported path does not equal the exact executed segment slice")
        if call.get("raw_rrt_path_sha256") != digest(raw) or call.get("exported_path_sha256") != digest(exported):
            raise ContractError("RRT path content hash mismatch")
        if type(call.get("direct_edge_proxy_valid")) is not bool:
            raise ContractError("RRT trace must state whether its direct proxy edge was valid")
        if "fixed_quaternion_wxyz" in call:
            q = point(call["fixed_quaternion_wxyz"], 4)
            if not math.isclose(sum(v*v for v in q), 1, abs_tol=1e-6) or "quaternion_waypoints" not in segment:
                raise ContractError("Fixed-orientation RRT needs matching normalized segment quaternions")
            for actual in segment["quaternion_waypoints"][span[0]:span[1]+1]:
                if min(math.dist(q, actual), math.dist(tuple(-v for v in q), actual)) > 1e-8:
                    raise ContractError("RRT interval changes its fixed orientation")
        elif document["algorithm"] == "pose_graph_rrt_xyz":
            raise ContractError("Composite RRT call must identify its fixed quaternion")
    for spans in ranges.values():
        ordered = sorted(spans)
        if len(set(ordered)) != len(ordered) or any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise ContractError("RRT provenance contains duplicate or overlapping intervals")
    if document["algorithm"] == "rrt":
        covered = list(segments)
        if provenance.get("coverage") == "regenerated_remaining_segments_only":
            prefix = provenance.get("preserved_prefix_segment_ids")
            if not isinstance(prefix, list) or prefix != covered[:len(prefix)]:
                raise ContractError("Replanned RRT provenance must preserve an exact prefix")
            events = document.get("replanning_events")
            if (not isinstance(events, list) or not events or not isinstance(events[-1], dict)
                    or events[-1].get("event") != "blocked_edge" or events[-1].get("decision") != "replan"
                    or events[-1].get("segment_index") != len(prefix)
                    or events[-1].get("preserved_completed_segment_ids") != prefix
                    or not isinstance(events[-1].get("archived_pre_event_rrt_provenance"), dict)):
                raise ContractError("RRT prefix exemption requires the matching recorded replanning event and archived provenance")
            if prefix:
                # Bind the unchanged prefix to its last matching pre-event raw
                # trace, including across earlier legitimate replanning events.
                # This is consistency checking, not authentication of student data.
                archived_calls = {}
                for event in events:
                    archived = event.get("archived_pre_event_rrt_provenance", {})
                    if not isinstance(archived, dict):
                        continue
                    for call in archived.get("calls", []):
                        if (isinstance(call, dict) and call.get("segment_id") in prefix
                                and call.get("exported_waypoints") == segments[call["segment_id"]]["waypoints"]):
                            archived_calls[call["segment_id"]] = call
                if set(archived_calls) != set(prefix):
                    raise ContractError("Archived RRT traces must bind every preserved executed prefix segment")
                _validate_rrt_provenance({"coordinate_frame":"world_xyz_m", "algorithm":"rrt",
                    "segments":[segments[name] for name in prefix],
                    "rrt_provenance":{"schema_version":"rrt-export-provenance-v1",
                                      "calls":[archived_calls[name] for name in prefix]}})
            covered = covered[len(prefix):]
            if provenance.get("covered_segment_ids") != covered:
                raise ContractError("Replanned RRT provenance must cover all remaining segments")
        for name, segment in segments.items():
            expected = [(0, len(segment["waypoints"])-1)] if name in covered else []
            if ranges[name] != expected:
                raise ContractError("Cartesian RRT traces must cover every exported carry segment")
    else:
        for name, segment in segments.items():
            expected = [tuple(item["executed_waypoint_index_range"])
                for item in segment.get("planning_intervals", [])
                if item.get("source") == "student_rrt_xyz_then_shortening"]
            if ranges[name] != expected:
                raise ContractError("Composite RRT traces and declared planning intervals differ")


def validate_path_document(document, *, require_rrt_provenance=False, require_planner_provenance=False):
    """Check format only. IK, full-body collision and dynamics remain runtime work.

    world_xyz_m points locate the Panda hand/tool origin, NOT payload centres.
    joint_rad contains seven arm angles, excluding the gripper fingers.
    A duration is simulation seconds for the entire segment, including its edges.
    Present RRT traces must bind to the exported path. New RRT exporters require
    them. Historical untraced files remain usable as diagnostics by default,
    without establishing RRT use or satisfying the revised project requirement.
    """
    if document.get("schema_version") != "transport-path-v1":
        raise ContractError("Expected transport-path-v1")
    frame = document.get("coordinate_frame")
    if frame not in ("world_xyz_m", "joint_rad"):
        raise ContractError("Unknown coordinate frame")
    segments = document.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ContractError("Provide at least one segment")
    seen = set()
    key, dimension = ("waypoints", 3) if frame == "world_xyz_m" else ("joint_waypoints", 7)
    other_key = "joint_waypoints" if dimension == 3 else "waypoints"
    checked_segments = list(segments)
    if "deposit_segment" in document:
        if frame != "world_xyz_m" or not isinstance(document["deposit_segment"], dict):
            raise ContractError("deposit_segment requires a world-pose motion object")
        checked_segments.append(document["deposit_segment"])
    if "deposit_segments" in document:
        if "deposit_segment" in document or frame != "world_xyz_m" or not isinstance(document["deposit_segments"], list):
            raise ContractError("Use one world-pose deposit_segments list, without deposit_segment")
        objects = {segment["object_id"] for segment in segments}
        deposited = []
        for segment in document["deposit_segments"]:
            if not isinstance(segment, dict) or segment.get("phase") != "deposit" or segment.get("object_id") not in objects:
                raise ContractError("Each batch deposit must name a supplied object and deposit phase")
            deposited.append(segment["object_id"])
        if len(set(deposited)) != len(deposited):
            raise ContractError("A batch object has duplicate deposit instructions")
        checked_segments.extend(document["deposit_segments"])
    for segment in checked_segments:
        sid = identifier(segment.get("id"), "segment id")
        if sid in seen:
            raise ContractError("Segment ids must be unique")
        seen.add(sid)
        if segment.get("robot_id") not in ("arm_0", "arm_1", "arm_2"):
            raise ContractError("Unknown robot_id")
        if segment.get("phase") not in ("approach", "transfer", "retreat", "deposit"):
            raise ContractError("Unknown motion phase")
        if segment.get("entry_phase", "after_lift") not in ("after_lift", "after_grasp"):
            raise ContractError("Unknown transfer entry phase")
        identifier(segment.get("object_id"), "object_id")
        if other_key in segment:
            raise ContractError("Do not mix joint and world coordinates")
        path = segment.get(key)
        if not isinstance(path, list) or len(path) < 2:
            raise ContractError("A segment needs at least two waypoints")
        for waypoint in path:
            point(waypoint, dimension)
        if "quaternion_waypoints" in segment:
            orientations = segment["quaternion_waypoints"]
            if frame != "world_xyz_m" or not isinstance(orientations, list) or len(orientations) != len(path):
                raise ContractError("World hand quaternion_waypoints must match every XYZ waypoint")
            for orientation in orientations:
                q = point(orientation, 4)
                if not math.isclose(sum(x*x for x in q), 1.0, abs_tol=1e-6):
                    raise ContractError("Hand orientations must be normalized [w,x,y,z]")
        if "duration_s" in segment and number(segment["duration_s"], "duration_s", minimum=0) == 0:
            raise ContractError("Segment duration must be positive")
        if "hold_s" in segment:
            number(segment["hold_s"], "hold_s", minimum=0)
    if require_rrt_provenance and "rrt_provenance" not in document:
        raise ContractError("This operation requires recorded RRT export provenance")
    if "rrt_provenance" in document:
        _validate_rrt_provenance(document)
    if require_planner_provenance and "planner_provenance" not in document:
        raise ContractError("This operation requires endpoint planner provenance")
    if "planner_provenance" in document:
        from .endpoint_planning import validate_planner_provenance
        validate_planner_provenance(document)
    return document
