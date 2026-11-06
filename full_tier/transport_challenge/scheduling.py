"""Provided live dispatch boundary. All movement and waiting remain physical.

Candidates are unfinished stages whose predecessor has actually completed and
whose placement station is unoccupied in the current observation. The student
chooses among these candidates. This module never predicts a completed job.
"""

from dataclasses import dataclass
from types import MappingProxyType
from .contracts import ContractError, digest, number, point
from .geometry import distance, path_length
from .relay import build_transfer_schedule


DEFAULT_OBJECTIVE = {"quantity_weight": 1.0, "path_length_weight": 0.0, "carry_time_weight": 0.0}


def validate_objective(objective):
    if not isinstance(objective, dict) or set(objective) != set(DEFAULT_OBJECTIVE):
        raise ContractError("Objective needs quantity_weight, path_length_weight and carry_time_weight")
    result = {key: number(value, key, minimum=0) for key, value in objective.items()}
    if not any(result.values()):
        raise ContractError("At least one objective weight must be positive")
    return result


@dataclass(frozen=True)
class DispatchCandidate:
    segment_id: str
    object_id: str
    robot_id: str
    stage_index: int
    is_final: bool
    available_quantity_estimate: float | None
    path_length_m: float
    planned_carry_duration_s: float


@dataclass(frozen=True)
class DispatchState:
    sim_time_s: float
    deadline_s: float | None
    completed_segment_ids: frozenset[str]
    completed_jobs: frozenset[str]
    pending_segment_ids: tuple[str, ...]
    blocked_placement_segment_ids: tuple[str, ...]


@dataclass(frozen=True)
class DispatchChoice:
    segment_id: str | None
    wait_s: float = 0.0


class DispatchController:
    """Provide immutable observations and validate one real student decision.

    policy_module implements choose_dispatch(state, candidates, objective).
    candidates is a tuple of DispatchCandidate. Objective weights have units of
    utility per material quantity, per metre and per nominal carry second.
    available_quantity_estimate comes from the runtime's measured material map.
    None means unavailable, never measured zero. Carrier transfer overhead is
    excluded from planned_carry_duration_s and must not be mistaken for the full
    measured stage time.

    choose accepts exact observed simulation time, physically completed segment
    IDs, and current root XYZ of EVERY persistent object. A returned wait must
    be executed by advancing the existing scene. No timeline is fast-forwarded.
    This initial scheduler executes one stage at a time, so it does not claim
    parallel execution or a third-arm throughput advantage.
    """
    def __init__(self, scenario, route, design, policy_module):
        if route.get("scenario_content_sha256") != digest(scenario) or route.get("design_content_sha256") != digest(design):
            raise ContractError("Dispatch scenario/design must match the actual planned-route hashes")
        self.stages = build_transfer_schedule(scenario, route, len(design["arm_bases_m"]), design)
        self.by_id = {stage.segment_id: stage for stage in self.stages}
        self.segments = {segment["id"]: segment for segment in route["segments"]}
        self.object_ids = frozenset(stage.object_id for stage in self.stages)
        self.deadline = scenario.get("simulation_deadline_s")
        if self.deadline is not None:
            self.deadline = number(self.deadline, "deadline", minimum=0)
        self.objective = MappingProxyType(validate_objective(design.get("objective", dict(DEFAULT_OBJECTIVE))))
        self.policy = policy_module
        self.station_clearance = number(scenario.get("placement_occupancy_distance_m", .13), "station clearance", minimum=0)
        self.previous_by_id = {}
        last = {}
        for stage in self.stages:
            self.previous_by_id[stage.segment_id] = last.get(stage.object_id)
            last[stage.object_id] = stage.segment_id
        self.audit = []
        self._last_time = None
        self._last_completed = frozenset()
        self._required_wait_end = 0.0

    def choose(self, observed_sim_time_s, completed_segment_ids, observed_object_positions_m,
               available_quantity_by_object=None):
        now = number(observed_sim_time_s, "observed time", minimum=self._last_time or 0)
        if now < self._required_wait_end-1e-10:
            raise ContractError("The requested physical wait has not elapsed in the observed scene")
        completed = frozenset(completed_segment_ids)
        if not completed <= set(self.by_id) or not self._last_completed <= completed:
            raise ContractError("Dispatch completion IDs cannot be invented or forgotten")
        for sid in completed:
            previous = self.previous_by_id[sid]
            if previous is not None and previous not in completed:
                raise ContractError("A relay stage cannot complete before its physical predecessor")
        if set(observed_object_positions_m) != self.object_ids:
            raise ContractError("Dispatch must observe every persistent object without cloning/deletion")
        positions = {oid: point(value) for oid, value in observed_object_positions_m.items()}
        quantities = available_quantity_by_object or {}
        if not set(quantities) <= self.object_ids:
            raise ContractError("Quantity estimates contain an unknown object")
        quantities = {oid: number(value, "observed available quantity", minimum=0) for oid, value in quantities.items()}
        pending = tuple(s.segment_id for s in self.stages if s.segment_id not in completed)
        finished = frozenset(s.object_id for s in self.stages if s.is_final and s.segment_id in completed)
        if not pending or self.deadline is not None and now >= self.deadline:
            self._last_time, self._last_completed = now, completed
            return DispatchChoice(None, 0)
        candidates, blocked = [], []
        for sid in pending:
            previous = self.previous_by_id[sid]
            if previous is not None and previous not in completed:
                continue
            stage = self.by_id[sid]
            if distance(positions[stage.object_id], stage.pickup_m) > .04:
                raise ContractError("A ready job is not at its observed physical pickup station")
            if any(oid != stage.object_id and distance(position, stage.placement_m) < self.station_clearance
                   for oid, position in positions.items()):
                blocked.append(sid)
                continue
            segment = self.segments[sid]
            if "waypoints" not in segment:
                raise ContractError("Current dispatch estimates require world-pose carry paths")
            candidates.append(DispatchCandidate(sid, stage.object_id, stage.robot_id, stage.stage_index,
                                                stage.is_final, quantities.get(stage.object_id),
                                                path_length(segment["waypoints"]), segment["duration_s"]))
        state = DispatchState(now, self.deadline, completed, finished, pending, tuple(blocked))
        decision = self.policy.choose_dispatch(state, tuple(candidates), self.objective)
        if not isinstance(decision, DispatchChoice):
            raise ContractError("Student dispatch must return a DispatchChoice")
        wait = number(decision.wait_s, "physical wait", minimum=0, maximum=5)
        if decision.segment_id is not None and decision.segment_id not in {c.segment_id for c in candidates}:
            raise ContractError("Student dispatch selected an unavailable stage")
        if decision.segment_id is None and wait == 0:
            raise ContractError("Pending work needs a ready stage or a positive physical wait")
        if self.deadline is not None and now+wait > self.deadline:
            raise ContractError("Requested wait would extend beyond the common deadline")
        self.audit.append({"observed_sim_time_s": now, "completed_segment_ids": sorted(completed),
                           "completed_jobs": sorted(finished), "ready_segment_ids": [c.segment_id for c in candidates],
                           "blocked_placement_segment_ids": blocked, "selected_segment_id": decision.segment_id,
                           "requested_wait_s": wait, "objective": dict(self.objective),
                           "completion_was_inferred": False})
        self._last_time, self._last_completed = now, completed
        self._required_wait_end = now+wait
        return decision
