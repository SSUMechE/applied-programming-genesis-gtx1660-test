"""Provided persistent-object relay contract for the physical serial executor.

An explicit stage describes a physical pickup, the supplied/student carry
segment, and a physical placement. This adapter validates the order. It does
not move or copy objects, assign predicted success, or claim parallel speedup.
"""

from dataclasses import dataclass
import copy
from .contracts import ContractError, identifier, point, validate_path_document
from .geometry import distance


@dataclass(frozen=True)
class RelayStage:
    object_id: str
    segment_id: str
    robot_id: str
    pickup_m: tuple[float, float, float]
    placement_m: tuple[float, float, float]
    stage_index: int
    is_final: bool

    @property
    def arm_index(self):
        return int(self.robot_id.split("_")[1])


def physical_job_stages(job):
    """Return declared stages, or the original supplied two-arm relay shape."""
    if "stages" in job:
        return copy.deepcopy(job["stages"])
    return [
        {"robot_id": "arm_0", "pickup_m": job["source_m"], "placement_m": job["handoff_m"]},
        {"robot_id": "arm_1", "pickup_m": job["handoff_m"], "placement_m": job["target_m"]},
    ]


def resolve_stage_arm(scenario, design, segment_id, default_arm, arm_count):
    """Bind an explicit student allocation only within instructor-allowed arms."""
    assignments = {} if design is None else design.get("arm_assignments", {})
    if segment_id not in assignments:
        return default_arm
    chosen = assignments[segment_id]
    allowed = scenario.get("allowed_stage_arms", {}).get(segment_id, [])
    if chosen not in allowed or chosen not in {f"arm_{i}" for i in range(arm_count)}:
        raise ContractError("Student arm assignment is outside the supplied stage choices")
    return chosen


def build_transfer_schedule(scenario, route, arm_count, design=None):
    """Return an immutable serial stage sequence for the actual physical runner.

    Explicit physical_jobs[*].stages has 2 or 3 dictionaries. Each contains
    robot_id, segment_id, pickup_m and placement_m. Positions refer to the
    persistent payload's root, not the robot hand. Adjacent stages must use
    different arms and the same physical handoff position. The first pickup
    matches source_m. The final placement matches target_m.

    All declared jobs occur once, in the flattened route order. A design changes
    job order and the first handoff through the existing handoff_m field. For a
    three-stage chain, the second handoff remains instructor supplied. Every
    active arm must participate. Starting three arms and leaving one idle does
    not count as a three-arm relay trial.

    This contract covers serial execution. Resource reservations and concurrent
    motion require a separate runtime implementation and measured comparison.
    """
    validate_path_document(route)
    if type(arm_count) is not int or arm_count not in (2, 3):
        raise ContractError("A relay runtime must instantiate exactly two or three arms")
    jobs = copy.deepcopy(scenario.get("physical_jobs", []))
    if not jobs:
        raise ContractError("Explicit physical_jobs are required for a relay schedule")
    by_id = {}
    for job in jobs:
        object_id = identifier(job["object_id"])
        if object_id in by_id:
            raise ContractError("Duplicate physical object ID")
        by_id[object_id] = job
    if design is not None:
        if len(design["arm_bases_m"]) != arm_count:
            raise ContractError("Design base count must match the instantiated relay arms")
        order = design["job_order"]
        if len(order) != len(by_id) or set(order) != set(by_id):
            raise ContractError("Design job order must retain every persistent object exactly once")
        jobs = [by_id[object_id] for object_id in order]
    else:
        route_order = list(dict.fromkeys(s["object_id"] for s in route["segments"]))
        if set(route_order) != set(by_id):
            raise ContractError("Physical jobs and route object identities differ")
        jobs = [by_id[object_id] for object_id in route_order]
    schedule, offset = [], 0
    for job in jobs:
        object_id = job["object_id"]
        stages = physical_job_stages(job)
        if not isinstance(stages, list) or len(stages) not in (2, 3):
            raise ContractError("Each physical relay requires two or three declared stages")
        if len(stages) != arm_count:
            raise ContractError("The current serial trial requires one stage per participating arm")
        if design is not None:
            stages[0]["placement_m"] = list(design["handoff_m"])
            stages[1]["pickup_m"] = list(design["handoff_m"])
        previous = None
        for index, stage in enumerate(stages):
            if offset >= len(route["segments"]):
                raise ContractError("The route is missing a physical relay stage")
            segment = route["segments"][offset]
            offset += 1
            arm = resolve_stage_arm(scenario, design, segment["id"], stage["robot_id"], arm_count)
            if arm not in tuple(f"arm_{i}" for i in range(arm_count)):
                raise ContractError("Stage uses an arm that is not instantiated")
            pickup, placement = point(stage["pickup_m"]), point(stage["placement_m"])
            if distance(pickup, placement) <= 1e-9:
                raise ContractError("A relay stage must physically transport the object")
            if previous and (distance(previous.placement_m, pickup) > 1e-9 or previous.robot_id == arm):
                raise ContractError("Handoff must retain the same position and change the active arm")
            if index == 0 and distance(pickup, point(job["source_m"])) > 1e-9:
                raise ContractError("First pickup differs from the finite-source position")
            if index == len(stages)-1 and distance(placement, point(job["target_m"])) > 1e-9:
                raise ContractError("Final placement differs from the declared destination")
            if segment["object_id"] != object_id or segment["robot_id"] != arm or segment["phase"] != "transfer":
                raise ContractError("Route ordering must exactly match the physical relay stages")
            if "segment_id" in stage and segment["id"] != stage["segment_id"]:
                raise ContractError("A physical stage refers to a different route segment")
            if route["coordinate_frame"] == "world_xyz_m":
                first, last = segment["waypoints"][0], segment["waypoints"][-1]
                # Carry height can vary. Pickup/placement hand alignment cannot.
                if distance(first[:2], pickup[:2]) > 1e-9 or distance(last[:2], placement[:2]) > 1e-9:
                    raise ContractError("Supplied carry endpoints disagree with physical pickup/placement XY")
            previous = RelayStage(object_id, segment["id"], arm, pickup, placement,
                                  index, index == len(stages)-1)
            schedule.append(previous)
    if offset != len(route["segments"]):
        raise ContractError("Route includes undeclared extra relay stages")
    if {s.robot_id for s in schedule} != {f"arm_{i}" for i in range(arm_count)}:
        raise ContractError("Every active arm must participate in the serial relay")
    return tuple(schedule)
