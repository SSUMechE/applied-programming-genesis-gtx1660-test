"""Provided bounded design checks. Physics binding is a separate release gate."""

import copy
from pathlib import Path
from .contracts import ContractError, digest, number, point
from .geometry import Box, distance
from .vessel import CONTAINER_DENSITY_KG_M3, build_container_geometry


SPEED_PROFILES = {
    "careful": {"nominal_path_progress_speed_m_s": 0.08},
    "normal": {"nominal_path_progress_speed_m_s": 0.14},
    "fast": {"nominal_path_progress_speed_m_s": 0.20},
}
# These values set nominal path length / segment duration. The provided physical
# joint interpolation and tracking dynamics can have a higher instantaneous
# hand speed. A student's optional maximum-speed constraint uses measured speed.


def design_bounds_path(case, project_root=None):
    """Resolve the instructor case's versioned bounds inside this project."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    root = root.resolve()
    path = (root / case.get("design_bounds_file", "cases/design_bounds.json")).resolve()
    if not path.is_relative_to(root):
        raise ContractError("Instructor bounds must be inside the project")
    return path


def validate_design(design, bounds):
    """Validate only allowed fields against instructor-owned case bounds.

    Layout changes are accepted structurally here. Actual reachability, collision,
    capacity and material retention must be checked by the provided physics run.
    No changeable gravity, solver parameter, material property or robot limit is
    accepted. A valid design object is not a physically successful design.
    """
    allowed = {"arm_bases_m", "handoff_m", "container", "speed_profile", "job_order"}
    optional = {"objective", "arm_assignments", "rrt_options", "planner_options", "segment_speed_profiles"}
    if not allowed <= set(design) or not set(design) <= allowed | optional:
        raise ContractError(f"Design requires {sorted(allowed)}, optional {sorted(optional)}")
    arms = design["arm_bases_m"]
    if not isinstance(arms, list) or len(arms) not in bounds["allowed_arm_counts"]:
        raise ContractError("Unsupported number of arms")
    if len(arms) > len(bounds["arm_base_regions_m"]):
        raise ContractError("Missing instructor installation region")
    positions = [point(value) for value in arms]
    assignments = design.get("arm_assignments", {})
    if not isinstance(assignments, dict) or any(not isinstance(sid, str) or not sid or arm not in {f"arm_{i}" for i in range(len(arms))}
                                               for sid, arm in assignments.items()):
        raise ContractError("Arm assignments must map segment IDs to instantiated arm IDs")
    if "objective" in design:
        from .scheduling import validate_objective
        validate_objective(design["objective"])
    if "rrt_options" in design:
        limits = bounds.get("rrt_options_bounds")
        options = design["rrt_options"]
        if (not isinstance(limits, dict) or not isinstance(options, dict)
                or set(options) != {"step_size_m_equiv", "goal_bias"}):
            raise ContractError("Use only the instructor-bounded RRT step size and goal bias")
        for name, value in options.items():
            number(value, name, minimum=limits[name][0], maximum=limits[name][1])
    if "planner_options" in design:
        limits=bounds.get("planner_options_bounds")
        options=design["planner_options"]
        if not isinstance(limits,dict) or not isinstance(options,dict) or not set(options)<=set(limits):
            raise ContractError("Use only declared optional planner settings, without changing shared budgets or states")
        for name,value in options.items():
            number(value,name,minimum=limits[name][0],maximum=limits[name][1])
    profiles = design.get("segment_speed_profiles", {})
    if "segment_speed_profiles" in design and not ("rrt_options_bounds" in bounds or "planner_options_bounds" in bounds):
        raise ContractError("Segment speed choices require the new free-design case contract")
    if (not isinstance(profiles, dict) or any(not isinstance(sid,str) or not sid
            or profile not in SPEED_PROFILES for sid,profile in profiles.items())):
        raise ContractError("Segment speeds must map transfer IDs to supplied speed profiles")
    for position, region in zip(positions, bounds["arm_base_regions_m"]):
        if not Box(tuple(region[0]), tuple(region[1])).contains(position):
            raise ContractError("An arm base is outside its allowed installation region")
    for i, a in enumerate(positions):
        if any(distance(a, b) < bounds["minimum_base_separation_m"] for b in positions[i + 1:]):
            raise ContractError("Arm bases overlap the minimum installation separation")
    handoff_region = bounds["handoff_region_m"]
    if not Box(tuple(handoff_region[0]), tuple(handoff_region[1])).contains(point(design["handoff_m"])):
        raise ContractError("Handoff table is outside its allowed region")
    container = design["container"]
    fields = {"outer_size_m", "wall_m", "open_top", "partitions", "handle_width_m"}
    mass_field = "empty_mass_kg" if "empty_mass_kg" in container else "mass_limit_kg"
    fields.add(mass_field)
    if "geometry_model" in container:
        fields.add("geometry_model")
    if set(container) != fields or container["open_top"] is not True:
        raise ContractError("Use an open-top container with the supplied design fields")
    dimensions = point(container["outer_size_m"])
    maximum = bounds["container_max_size_m"]
    minimum = bounds["container_min_size_m"]
    if not all(lo <= size <= hi for lo, size, hi in zip(minimum, dimensions, maximum)):
        raise ContractError("Container outer dimensions exceed the allowed bounds")
    wall = number(container["wall_m"], "wall_m", minimum=bounds["wall_min_m"], maximum=bounds["wall_max_m"])
    if any(size <= 2 * wall for size in dimensions):
        raise ContractError("Container has no positive internal space")
    number(container[mass_field], mass_field, minimum=bounds["empty_mass_min_kg"], maximum=bounds["empty_mass_max_kg"])
    if type(container["partitions"]) is not int or not 0 <= container["partitions"] <= bounds["partitions_max"]:
        raise ContractError("Partition count exceeds allowed bounds")
    number(container["handle_width_m"], "handle_width_m", minimum=bounds["handle_min_m"], maximum=bounds["handle_max_m"])
    if design["speed_profile"] not in SPEED_PROFILES:
        raise ContractError("Unknown supplied speed profile")
    order = design["job_order"]
    if not isinstance(order, list) or len(set(order)) != len(order) or set(order) != set(bounds["job_ids"]):
        raise ContractError("Job order must contain each supplied job exactly once")
    geometry = build_container_geometry(container)
    capacity = number(geometry["capacity_m3"], "capacity_m3", maximum=bounds["capacity_max_m3"])
    return {"design_valid": True, "capacity_m3": capacity,
            "computed_empty_mass_kg": geometry["empty_mass_kg"],
            "material_model_legacy": geometry["material_model_legacy"],
            "material_policy_compatible": geometry["material_policy_compatible"],
            "physical_validation": "NOT RUN", "runtime_design_binding": "REQUIRED"}


def apply_design_to_case(case, design, bounds):
    """Bind the same design to route endpoints, object order and payload envelope.

    The physical runner receives the original case plus the SAME design file.
    The source case and source design digests remain in the route document.
    Two-arm legacy cases and explicit two/three-stage serial relays are bound
    here. A three-arm design must supply actual stages for every participating
    arm. It is never silently mapped to a two-arm schedule.
    """
    fixed = copy.deepcopy(bounds)
    objects = tuple(dict.fromkeys(task.get("object_id", "payload") for task in case["transfers"]))
    fixed["job_ids"] = list(objects)
    validate_design(design, fixed)
    if design["container"]["partitions"] and case["material"] != "balls":
        raise ContractError("This experimental partition supports the fixed-ball load only")
    required_model = case.get("required_container_geometry_model")
    if required_model and design["container"].get("geometry_model") != required_model:
        raise ContractError("Design geometry_model does not match the declared physical case")
    arm_count = len(design["arm_bases_m"])
    if arm_count == 3 and not all(len(job.get("stages", [])) == 3 for job in case.get("physical_jobs", [{}])):
        raise ContractError("A three-arm design requires an explicit three-stage physical relay")
    changed = copy.deepcopy(case)
    changed["source_scenario_content_sha256"] = digest(case)
    changed["design_content_sha256"] = digest(design)
    changed["selected_speed_profile"] = design["speed_profile"]
    if "rrt_options" in design:
        changed["selected_rrt_options"] = copy.deepcopy(design["rrt_options"])
    if "planner_options" in design:
        changed["selected_planner_options"] = copy.deepcopy(design["planner_options"])
    changed["selected_segment_speed_profiles"] = copy.deepcopy(design.get("segment_speed_profiles", {}))
    # The hand remains at the same vertical clearance over the table.
    old_handoff = case.get("physical_jobs", [{}])[0].get("handoff_m", [0.65, 0.35, 0.02])
    handoff = design["handoff_m"]
    physical_jobs = {job["object_id"]: job for job in changed.get("physical_jobs", [])}
    ordered = []
    for object_id in design["job_order"]:
        pair = [copy.deepcopy(task) for task in case["transfers"] if task.get("object_id", "payload") == object_id]
        if len(pair) != arm_count:
            raise ContractError("Current serial jobs need exactly one transfer segment per participating arm")
        job = physical_jobs.get(object_id, {})
        if "stages" in job:
            stages = job["stages"]
            if len(stages) != len(pair):
                raise ContractError("Explicit stage count does not match the carry segments")
            original_stages = copy.deepcopy(stages)
            stages[0]["placement_m"] = list(handoff)
            stages[1]["pickup_m"] = list(handoff)
            for task, stage, original_stage in zip(pair, stages, original_stages):
                if "pose_nodes" in task:
                    raise ContractError("Explicit serial stage binding currently requires XYZ transfer endpoints")
                if task["id"] != stage["segment_id"] or task["robot_id"] != stage["robot_id"]:
                    raise ContractError("Explicit physical stage and route transfer identity differ")
                for endpoint, root_endpoint in (("start", "pickup_m"), ("goal", "placement_m")):
                    height = task[endpoint][2] - original_stage[root_endpoint][2]
                    task[endpoint] = [stage[root_endpoint][0], stage[root_endpoint][1], stage[root_endpoint][2]+height]
            ordered.extend(pair)
            continue
        for task in pair:
            if "pose_nodes" in task:
                for name, pose in task["pose_nodes"].items():
                    if name in ("handoff", "above_handoff"):
                        old = pose["xyz_m"]
                        pose["xyz_m"] = [handoff[0], handoff[1], handoff[2] + old[2] - old_handoff[2]]
                continue
            if task["robot_id"] == "arm_0":
                old = task["goal"]
                task["goal"] = [handoff[0], handoff[1], handoff[2] + old[2] - old_handoff[2]]
            elif task["robot_id"] == "arm_1":
                old = task["start"]
                task["start"] = [handoff[0], handoff[1], handoff[2] + old[2] - old_handoff[2]]
            else:
                raise ContractError("Current two-arm job uses an unsupported arm")
        ordered.extend(pair)
    changed["transfers"] = ordered
    from .relay import resolve_stage_arm
    transfer_ids = {task.get("id", f"transfer_{i}") for i, task in enumerate(ordered)}
    if not set(design.get("arm_assignments", {})) <= transfer_ids:
        raise ContractError("Arm assignment names an unknown segment")
    if not set(design.get("segment_speed_profiles", {})) <= transfer_ids:
        raise ContractError("Segment speed choice names an unknown transfer")
    for i, task in enumerate(ordered):
        sid = task.get("id", f"transfer_{i}")
        task["robot_id"] = resolve_stage_arm(case, design, sid, task["robot_id"], arm_count)
    if "physical_jobs" in changed:
        by_id = {job["object_id"]: job for job in changed["physical_jobs"]}
        changed["physical_jobs"] = [by_id[oid] for oid in design["job_order"]]
        for job in changed["physical_jobs"]:
            job["handoff_m"] = list(handoff)
            for stage in job.get("stages", []):
                stage["robot_id"] = resolve_stage_arm(case, design, stage["segment_id"], stage["robot_id"], arm_count)
    if case["material"] in ("balls", "water", "sand"):
        geometry = build_container_geometry(design["container"])
        changed["payload_envelope_relative_to_hand_m"] = geometry["payload_envelope_relative_to_hand_m"]
        changed["container_material_policy_compatible"] = geometry["material_policy_compatible"]
    return changed
