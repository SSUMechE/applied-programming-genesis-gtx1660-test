"""Supplied n-joint planning boundary and quintic trajectory primitives.

This module does not import a simulator. The caller supplies the collision
oracle and describes its coverage. Whole-edge acceptance means that oracle's
check passed, not that continuous collision freedom or execution was proved.
Native bounded joint angles are radians. No wrapping or clipping is performed.
"""
from dataclasses import dataclass
import hashlib
import inspect
import json
import math

from .contracts import ContractError, PlanningBudgetExceeded, PlanningRequest, digest, number, point
from .endpoint_planning import QueryBudgetOracle
from .query_diagnostics import completed_query_diagnostic, diagnostic_record


QUINTIC_PEAK_SPEED = 15.0 / 8.0
QUINTIC_PEAK_ACCELERATION = 10.0 / math.sqrt(3.0)


@dataclass(frozen=True)
class JointLimits:
    position_rad: tuple
    velocity_rad_s: tuple
    acceleration_rad_s2: tuple

    def __post_init__(self):
        bounds = tuple(point(pair, 2) for pair in self.position_rad)
        if not bounds or any(lo >= hi or not math.isfinite(hi-lo) for lo, hi in bounds):
            raise ContractError("Native joint ranges must be finite, nonempty and ordered")
        velocity = point(self.velocity_rad_s, len(bounds))
        acceleration = point(self.acceleration_rad_s2, len(bounds))
        if any(value <= 0 for value in velocity + acceleration):
            raise ContractError("Every joint velocity and acceleration bound must be positive")
        object.__setattr__(self, "position_rad", bounds)
        object.__setattr__(self, "velocity_rad_s", velocity)
        object.__setattr__(self, "acceleration_rad_s2", acceleration)

    @property
    def dimension(self):
        return len(self.position_rad)

    def to_record(self):
        return {"position_rad": [list(pair) for pair in self.position_rad],
                "velocity_rad_s": list(self.velocity_rad_s),
                "acceleration_rad_s2": list(self.acceleration_rad_s2)}


def make_joint_request(start_q, goal_q, limits, *, max_queries, seed, options=None):
    """Only endpoints, native bounds and a neutral validity-query budget."""
    return PlanningRequest(start_q, goal_q, limits.position_rad, max_queries,
                           seed, {} if options is None else options)


def _configuration(q, limits):
    q = point(q, limits.dimension)
    if any(not lo <= value <= hi for value, (lo, hi) in zip(q, limits.position_rad)):
        raise ContractError("Joint configuration is outside native bounds")
    return q


class _BoundedOracle:
    """Reject invalid native coordinates before calling the physical oracle."""
    def __init__(self, oracle, limits):
        if not callable(getattr(oracle, "state_valid", None)) or not callable(getattr(oracle, "edge_valid", None)):
            raise ContractError("Oracle must provide callable state_valid and edge_valid")
        self.oracle, self.limits = oracle, limits
        self._last_diagnostic = diagnostic_record()

    def last_query_diagnostic(self):
        import copy
        return copy.deepcopy(self._last_diagnostic)

    def state_valid(self, q):
        try:
            q = _configuration(q, self.limits)
        except ContractError as error:
            self._last_diagnostic = diagnostic_record("rejected", reason="invalid_native_joint_state", query_kind="state")
            self._last_diagnostic["details"] = {"message": str(error), "stage": "supplied_native_bounds"}
            return False
        accepted = bool(self.oracle.state_valid(q))
        self._last_diagnostic = completed_query_diagnostic(self.oracle, "state", accepted)
        return accepted

    def edge_valid(self, start, goal):
        try:
            start, goal = _configuration(start, self.limits), _configuration(goal, self.limits)
        except ContractError as error:
            self._last_diagnostic = diagnostic_record("rejected", reason="invalid_native_joint_edge", query_kind="edge")
            self._last_diagnostic["details"] = {"message": str(error), "stage": "supplied_native_bounds"}
            return False
        # Linear interpolation remains within these convex interval bounds.
        accepted = bool(self.oracle.edge_valid(start, goal))
        self._last_diagnostic = completed_query_diagnostic(self.oracle, "edge", accepted)
        return accepted


def validate_joint_path(path, request, limits, oracle, *, max_states=10000):
    """Validate all states and edges outside the student query budget.

    A one-state path is permitted exactly when start==goal. Consecutive repeated
    states are rejected rather than silently deleted. No path is manufactured.
    """
    if request.bounds != limits.position_rad:
        raise ContractError("Request bounds must equal the native joint limits")
    if type(max_states) is not int or max_states < 1:
        raise ContractError("max_states must be a positive integer")
    if not isinstance(path, (tuple, list)) or not 1 <= len(path) <= max_states:
        raise ContractError("Planner must return a nonempty bounded joint path")
    states = tuple(_configuration(q, limits) for q in path)
    if states[0] != request.start or states[-1] != request.goal:
        raise ContractError("Joint path must preserve exact requested endpoints")
    if any(a == b for a, b in zip(states, states[1:])):
        raise ContractError("Consecutive repeated configurations are not motion edges")
    checked = _BoundedOracle(oracle, limits)
    if not all(checked.state_valid(q) for q in states):
        raise ContractError("Joint path contains an oracle-invalid state")
    if not all(checked.edge_valid(a, b) for a, b in zip(states, states[1:])):
        raise ContractError("Joint path contains an oracle-invalid whole edge")
    return states


def minimum_edge_duration_s(start, goal, limits):
    """Conservative finite duration for this straight-q quintic edge.

    max(s')=15/8 and max(|s''|)=10/sqrt(3) for normalized quintic time.
    One outward floating-point rounding step makes the returned limit slightly
    conservative. A repeated configuration has no motion and returns zero.
    """
    start, goal = _configuration(start, limits), _configuration(goal, limits)
    candidates = []
    for a, b, velocity, acceleration in zip(start, goal, limits.velocity_rad_s, limits.acceleration_rad_s2):
        displacement = abs(b-a)
        if displacement:
            candidates.extend((QUINTIC_PEAK_SPEED * displacement / velocity,
                               math.sqrt(QUINTIC_PEAK_ACCELERATION * displacement / acceleration)))
    result = max(candidates, default=0.0)
    if not math.isfinite(result) or (start != goal and result == 0):
        raise ContractError("Joint duration cannot be represented safely")
    result = math.nextafter(result, math.inf) if result else 0.0
    if not math.isfinite(result):
        raise ContractError("Joint duration overflows finite time")
    return result


@dataclass(frozen=True)
class JointSample:
    q_rad: tuple
    qd_rad_s: tuple
    qdd_rad_s2: tuple


@dataclass(frozen=True)
class JointTrajectory:
    """C2 rest-to-rest at each original joint waypoint, with exact endpoints."""
    waypoints: tuple
    edge_durations_s: tuple
    limits: JointLimits

    def __post_init__(self):
        if not isinstance(self.waypoints, (tuple, list)) or not self.waypoints:
            raise ContractError("Trajectory needs at least one configuration")
        waypoints = tuple(_configuration(q, self.limits) for q in self.waypoints)
        durations = point(self.edge_durations_s, len(waypoints)-1)
        for a, b, duration in zip(waypoints, waypoints[1:], durations):
            minimum = minimum_edge_duration_s(a, b, self.limits)
            if minimum == 0 or duration < minimum:
                raise ContractError("Each motion edge needs a duration at or above its certified minimum")
        elapsed = 0.0
        for duration in durations:
            next_time = elapsed + duration
            if not math.isfinite(next_time) or next_time <= elapsed:
                raise ContractError("Cumulative trajectory times must remain finite and strictly increasing")
            elapsed = next_time
        object.__setattr__(self, "waypoints", waypoints)
        object.__setattr__(self, "edge_durations_s", durations)

    @property
    def duration_s(self):
        return sum(self.edge_durations_s)

    @property
    def waypoint_times_s(self):
        times = [0.0]
        for duration in self.edge_durations_s:
            times.append(times[-1] + duration)
        return tuple(times)

    def sample(self, time_s):
        """Evaluate q/qd/qdd at 0<=t<=duration. Out-of-range time rejects."""
        time_s = number(time_s, "trajectory time", minimum=0, maximum=self.duration_s)
        zeros = (0.0,) * self.limits.dimension
        if time_s == self.duration_s:
            return JointSample(self.waypoints[-1], zeros, zeros)
        elapsed = 0.0
        for a, b, duration in zip(self.waypoints, self.waypoints[1:], self.edge_durations_s):
            end = elapsed + duration
            if time_s <= end:
                if time_s == elapsed:
                    return JointSample(a, zeros, zeros)
                if time_s == end:
                    return JointSample(b, zeros, zeros)
                u = (time_s-elapsed) / duration
                # Evaluate from the nearer endpoint. Direct evaluation near
                # u=1 suffers cancellation and can overshoot a native limit
                # by a few ulps. Symmetry avoids that without clipping q.
                v = u if u <= .5 else 1.0-u
                s = v*v*v*(10.0 + v*(-15.0 + 6.0*v))
                ds = 30.0*u*u*(1.0-u)*(1.0-u) / duration
                dds = 60.0*u*(1.0-u)*(1.0-2.0*u) / duration / duration
                # Monotonic s keeps each commanded q on the checked geometric edge.
                q = (tuple(first + (last-first)*s for first, last in zip(a, b)) if u <= .5
                     else tuple(last + (first-last)*s for first, last in zip(a, b)))
                qd = tuple((last-first)*ds for first, last in zip(a, b))
                qdd = tuple((last-first)*dds for first, last in zip(a, b))
                if not all(math.isfinite(value) for value in q+qd+qdd):
                    raise ContractError("Trajectory sample is not finite")
                return JointSample(q, qd, qdd)
            elapsed = end
        raise ContractError("Trajectory clock could not be resolved")

    def to_record(self):
        return {"interpolation": "joint_quintic_rest_to_rest_v1", "coordinate_frame": "joint_rad",
                "waypoints": [list(q) for q in self.waypoints],
                "edge_durations_s": list(self.edge_durations_s),
                "waypoint_times_s": list(self.waypoint_times_s), "duration_s": self.duration_s,
                "limits": self.limits.to_record()}


def parameterize_joint_path(path, limits, *, desired_edge_durations_s=None, speed_scale=1.0):
    """Use explicit edge durations OR slow down the analytic minimum.

    speed_scale is in (0,1]. It divides minimum durations and consequently
    scales speed by scale and acceleration by scale squared. Explicit durations
    are never stretched silently. Combining them with a nonunit scale rejects.
    """
    speed_scale = number(speed_scale, "speed_scale", minimum=0, maximum=1)
    if speed_scale == 0:
        raise ContractError("speed_scale must be positive")
    if not isinstance(path, (tuple, list)) or not path:
        raise ContractError("Time parameterization needs a nonempty path")
    path = tuple(_configuration(q, limits) for q in path)
    if desired_edge_durations_s is not None:
        if speed_scale != 1.0:
            raise ContractError("Choose explicit durations or speed_scale, not both")
        durations = point(desired_edge_durations_s, len(path)-1)
    else:
        durations = tuple(minimum_edge_duration_s(a, b, limits)/speed_scale for a, b in zip(path, path[1:]))
    return JointTrajectory(path, durations, limits)


@dataclass(frozen=True)
class JointPlan:
    trajectory: JointTrajectory
    _evidence_json: str

    @property
    def evidence(self):
        """Return a fresh copy, so caller annotations cannot rewrite this plan."""
        return json.loads(self._evidence_json)

    def to_record(self):
        return {"schema_version": "joint-plan-v1", "trajectory": self.trajectory.to_record(),
                "evidence": self.evidence}


def plan_joint_motion(request, oracle, strategy, limits, *, desired_edge_durations_s=None,
                      speed_scale=1.0, oracle_description="caller_supplied_coverage_not_certified",
                      kinematics=None, context=None, edge_timing_hook=None):
    """Call actual plan_path, validate all edges, then allocate safe timing.

    Collision resolution/pair coverage is owned and reported by the caller's
    oracle. Final verification is explicitly outside the student's query meter.
    This produces commands, not an actual tracking/transport success result.
    Context and a timing hook are explicit opt-ins. Legacy calls receive no
    new keyword. No signature fallback or replacement planner is attempted.
    """
    if not isinstance(request, PlanningRequest) or request.bounds != limits.position_rad:
        raise ContractError("Use a PlanningRequest with the complete native joint limits")
    if request.max_queries == 0:
        raise ContractError("No planning validity queries are available")
    if not isinstance(oracle_description, str) or not oracle_description.strip():
        raise ContractError("Describe the supplied oracle's actual coverage")
    planner = getattr(strategy, "plan_path", None)
    if not callable(planner):
        raise ContractError("Strategy must implement plan_path(request,oracle)")
    if context is not None:
        from .joint_context import JointTaskContext
        if not isinstance(context, JointTaskContext) or context.joint_limits != limits:
            raise ContractError("Context must contain these exact native and task joint limits")
    if edge_timing_hook is not None:
        if context is None or not callable(edge_timing_hook) or desired_edge_durations_s is not None:
            raise ContractError("A callable timing hook needs context and cannot combine with explicit durations")
    metered = QueryBudgetOracle(_BoundedOracle(oracle, limits), request.max_queries)
    if kinematics is not None:
        from .joint_kinematics import BoundedKinematics
        if not isinstance(kinematics, BoundedKinematics) or kinematics.joint_bounds_rad != limits.position_rad:
            raise ContractError("Optional kinematics must be a bounded facade with these native joint limits")
    try:
        support = {}
        if kinematics is not None:
            support["kinematics"] = kinematics
        if context is not None:
            support["context"] = context
        returned = planner(request, metered, **support)
    except PlanningBudgetExceeded as error:
        raise ContractError("Joint planner exhausted its query budget without a returned path") from error
    states = validate_joint_path(returned, request, limits, oracle)
    timing_hook_returned_durations = False
    if edge_timing_hook is not None:
        desired_edge_durations_s = edge_timing_hook(states, limits, context=context)
        timing_hook_returned_durations = desired_edge_durations_s is not None
    trajectory = parameterize_joint_path(states, limits,
                desired_edge_durations_s=desired_edge_durations_s, speed_scale=speed_scale)
    try:
        source_hash = hashlib.sha256(inspect.getsource(planner).encode("utf-8")).hexdigest()
    except (TypeError, OSError):
        source_hash = None
    evidence = {"request": request.to_record(), "dimension": limits.dimension,
                "native_joint_limits": limits.to_record(), "plan_path_source_sha256": source_hash,
                "returned_path_sha256": digest(states), "trajectory_sha256": digest(trajectory.to_record()),
                "query_meter": metered.record(), "final_validation_outside_student_budget": True,
                "final_validation_state_queries": len(states), "final_validation_edge_queries": len(states)-1,
                "oracle_description": oracle_description,
                "continuous_collision_certificate": False, "physical_execution_performed": False,
                "timing_policy": "explicit_edge_durations" if desired_edge_durations_s is not None else "scaled_analytic_minimum",
                "speed_scale": speed_scale,
                "desired_edge_durations_s": list(desired_edge_durations_s) if desired_edge_durations_s is not None else None}
    if kinematics is not None:
        evidence["kinematics"] = kinematics.record()
    if context is not None:
        evidence["joint_task_context"] = context.to_record()
        evidence["context_delivery"] = "explicit plan_path keyword"
    if edge_timing_hook is not None:
        evidence["edge_timing_hook"] = {"called": True, "returned_durations": timing_hook_returned_durations,
            "fallback_when_none": "selected speed_scale on the same verified student path"}
    return JointPlan(trajectory, json.dumps(evidence, sort_keys=True, allow_nan=False))
