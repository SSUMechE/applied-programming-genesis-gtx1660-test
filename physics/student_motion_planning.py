"""Provided endpoint-only boundary to the actual student's seven-joint planner.

The caller owns Genesis initialization and a paused, unchanged physical scene.
This module builds and closes an isolated query scene. It never steps, resets
or controls the live scene, imports a reference planner, or supplies route nodes.
The caller must authenticate the observation, source files and physical rest or
grasp gate. Detached hashes cannot establish those facts by themselves.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import traceback
import time

import numpy as np

from physics.carried_body import VesselModel, WorldPose
from physics.fixed_world_task import FixedSupportStation
from physics.provided_query_clearance import install_open_finger_margin
from transport_challenge.contracts import ContractError, digest, identifier, number, point
from transport_challenge.joint_context import JointBasePlacement, JointTaskContext
from transport_challenge.joint_kinematics import BoundedKinematics, ToolPose
from transport_challenge.joint_planning import JointLimits, make_joint_request, plan_joint_motion


def _save(path, value):
    from physics.run_joint_path import _json_safe
    with Path(path).open("x", encoding="utf8") as stream:
        stream.write(json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def _pose(value):
    if not isinstance(value, dict) or value.get("frame") != "world":
        raise ContractError("Require an explicit measured world pose")
    return WorldPose(value["position_m"], value["quaternion_wxyz"])


def effective_query_case(request):
    """Derive conservative rigid-payload geometry without changing the live map.

    The original nominal obstacle list is retained exactly. The existing
    query constructor expands non-support obstacles and insets the workspace.
    Embedded support plates keep their nominal geometry for real placement.
    """
    from physics.episode_runtime import empty_payload_mode
    nominal=request['scenario']['case']
    case=copy.deepcopy(nominal)
    empty=empty_payload_mode(request['world'],request['scenario'])
    mount_proofs=[]
    if empty or request['world'].get('rigid_runtime_profile')=='rigid16_bead_tc2h_v1':
        case['planning']['tracking_clearance_m']=max(float(case['planning'].get('tracking_clearance_m',0.)),.004)
        case['planning']['edge_max_joint_delta_rad']=min(float(case['planning']['edge_max_joint_delta_rad']),.005)
        for robot in request['scenario']['robots']:
            base=robot['base_placement'];z=float(robot.get('base_z_m',nominal.get('base_z_m',0.)))
            candidates=[item for item in nominal['obstacles']
                if item['lower_m'][0]<=base['x_m']<=item['upper_m'][0]
                and item['lower_m'][1]<=base['y_m']<=item['upper_m'][1]
                and abs(z-item['upper_m'][2]-.001)<=1e-7]
            if len(candidates)>1:
                raise ContractError('Occupied robot mounting surface is ambiguous')
            if candidates:
                mount_proofs.append(dict(arm_id=robot['id'],component_id=candidates[0]['id'],
                    base_world_m=[base['x_m'],base['y_m'],z],support_top_m=candidates[0]['upper_m'][2],
                    nominal_base_to_surface_gap_m=z-candidates[0]['upper_m'][2]))
        case['planning']['tracking_clearance_exempt_component_ids']=sorted({r['component_id'] for r in mount_proofs})
    receipt=dict(schema_version='student-motion-query-geometry-policy-v1',payload_mode='empty' if empty else 'original',
        nominal_case_sha256=digest(nominal),effective_query_case_sha256=digest(case),
        nominal_obstacles_sha256=digest(nominal['obstacles']),
        tracking_clearance_m=case['planning'].get('tracking_clearance_m',0.),
        edge_max_joint_delta_rad=case['planning']['edge_max_joint_delta_rad'],
        occupied_mount_inflation_exemptions=mount_proofs,
        occupied_mount_exemption_scope='Only geometrically derived components directly under a verified fixed robot base with the existing1mm gap. Nominal collision geometry remains active.',
        original_obstacle_coordinates_preserved=case['obstacles']==nominal['obstacles'],
        physical_map_modified=False,physical_tolerances_modified=False,
        scope='Isolated planning geometry only. Native execution retains the nominal map and original physical contact/tracking/rest checks.')
    return case,receipt


def _request(value, *, saved_clearance=False):
    if type(saved_clearance) is not bool:
        raise ContractError('Saved clearance validation selection must be Boolean')
    required = {"schema_version", "stage_id", "kind", "world", "scenario", "observation",
                "command_start_q_rad", "active_arm_id", "goal", "planning"}
    if (not isinstance(value, dict) or not required <= set(value)
            or set(value) - required - {"held_body", "placement_goal", "probe_scope"}
            or value["schema_version"] != "student-motion-request-v1"):
        raise ContractError("Require the explicit student-motion-request-v1 fields")
    digest(value)  # Require detached, finite JSON inputs, not simulator objects.
    request = copy.deepcopy(value)
    identifier(request["stage_id"], "student motion stage")
    if request["kind"] not in ("approach", "carry", "retreat"):
        raise ContractError("Student motion kind must be approach, carry or retreat")
    if saved_clearance and request['kind']!='retreat':
        raise ContractError('Saved clearance selection is restricted to a declared retreat')
    probe = "probe_scope" in request
    if probe and (request["probe_scope"] != "three_arm_short_motion_unit" or request["kind"] != "retreat"):
        raise ContractError("Only the explicit three-arm retreat unit probe scope is supported")
    world, scenario, observation = (request[k] for k in ("world", "scenario", "observation"))
    from physics.episode_runtime import empty_payload_mode
    empty = empty_payload_mode(world, scenario)
    if empty and (observation.get('materials') != [] or 'particle_material' in observation):
        raise ContractError('Explicit empty motion requires no observed rigid or particle contents')
    if (scenario["case"] != world["case_record"]
            or scenario["provenance"]["world_sha256"] != digest(world)
            or scenario["body_id"] != world["object_id"]):
        raise ContractError("Student motion world, scenario and object identity differ")
    names = [r["id"] for r in scenario["robots"]]
    from physics.single_active_boundary import contract_from_observation
    single = contract_from_observation(observation, scenario)
    measured_names = {single['active_arm_id']} if single is not None else set(names)
    if (len(names) != 3 or len(set(names)) != 3 or set(observation["arms"]) != measured_names
            or request["active_arm_id"] not in measured_names):
        raise ContractError("Require all three current arms and one named active arm")
    if (type(observation["native_scene_step"]) is not int or observation["native_scene_step"] < 0
            or type(observation["simulation_s"]) not in (int, float)
            or not math.isfinite(observation["simulation_s"]) or observation["simulation_s"] < 0
            or abs(observation["simulation_s"] - observation["native_scene_step"] * .0025) > 1e-9):
        raise ContractError("Measured student motion observation clock is invalid")
    request["command_start_q_rad"] = list(point(request["command_start_q_rad"], 7))
    for arm in observation["arms"].values():
        point(arm["q_all_dofs"], 9)
        _pose(arm["hand_world"])
    _pose(observation["body_world"])
    if 'particle_material_request' in world:
        from physics.particle_student_completion import particle_context
        particle_context(world, scenario, observation)
    planning = request["planning"]
    keys = {"seed", "max_queries", "speed_scale", "endpoint_attempts", "options",
            "use_edge_timing_hook", "query_body_tilt_limit_deg"}
    if not isinstance(planning, dict) or set(planning) - keys or not {"seed", "max_queries", "speed_scale"} <= set(planning):
        raise ContractError("Require explicit seed, query budget and joint speed scale")
    for key, low, high in (("seed", 0, 2**31 - 1), ("max_queries", 1, 100000)):
        if type(planning[key]) is not int or not low <= planning[key] <= high:
            raise ContractError("Invalid student motion " + key)
    planning.setdefault("endpoint_attempts", 16)
    planning.setdefault("options", {})
    planning.setdefault("use_edge_timing_hook", False)
    planning.setdefault("query_body_tilt_limit_deg", 7.5)
    if type(planning["endpoint_attempts"]) is not int or not 1 <= planning["endpoint_attempts"] <= 128:
        raise ContractError("Endpoint attempts must be1..128")
    if type(planning["use_edge_timing_hook"]) is not bool:
        raise ContractError("Timing hook selection must be Boolean")
    if not isinstance(planning["options"], dict):
        raise ContractError("Student planner options must be a mapping")
    scale = number(planning["speed_scale"], "joint speed scale", minimum=0., maximum=1.)
    if scale == 0.:
        raise ContractError("Joint speed scale must be positive")
    planning["speed_scale"] = scale
    tilt = number(planning["query_body_tilt_limit_deg"], "query body tilt", minimum=0., maximum=180. if empty else 8.)
    if tilt == 0.:
        raise ContractError("Query body tilt must remain positive and within the selected payload limit")
    goal = request["goal"]
    if request["kind"] == "retreat":
        expected_kind = "clear_handoff" if saved_clearance else "declared_probe_endpoint" if probe else "return_initial"
        if saved_clearance and (probe or not request['stage_id'].endswith('__clear_handoff')):
            raise ContractError('Saved clearance requires its explicit computed stage identity')
        if not isinstance(goal, dict) or set(goal) != {"kind", "q_rad"} or goal["kind"] != expected_kind:
            raise ContractError("Retreat requires the declared policy endpoint, never an old manual parking target")
        point(goal["q_rad"], 7)
    else:
        if (not isinstance(goal, dict) or set(goal) not in (
                {"kind", "position_m", "quaternion_wxyz"},
                {"kind", "frame", "position_m", "quaternion_wxyz"})
                or goal["kind"] != "world_hand_pose" or goal.get("frame", "world") != "world"):
            raise ContractError("Approach/carry goal is only a world hand endpoint")
        WorldPose(goal["position_m"], goal["quaternion_wxyz"])
    if request["kind"] == "carry":
        if not isinstance(request.get("held_body"), dict):
            raise ContractError("Carry requires the serialized actual current HeldBodySpec")
    elif request.get("held_body") is not None or request.get("placement_goal") is not None:
        raise ContractError("An unloaded motion cannot claim a held body or placement")
    return request


def _query_scene(request, output):
    from physics.episode_joint_scene import EpisodeJointScene
    from physics.episode_loaded_joint_scene import EpisodeLoadedJointScene
    from physics.episode_held_phase_inputs import observed_held_spec
    from physics.handle_contacts import SupportedHandleContactPolicy
    world, scenario, obs = (request[k] for k in ("world", "scenario", "observation"))
    from physics.episode_runtime import empty_payload_mode
    empty = empty_payload_mode(world, scenario)
    model = VesselModel.from_container(scenario["container"])
    if model.geometry_sha256 != world["vessel_model"]["geometry_sha256"]:
        raise ContractError("Current vessel model differs from the bound world")
    robots = copy.deepcopy(scenario["robots"])
    from physics.single_active_boundary import robot_coordinates, contract_from_observation
    single = contract_from_observation(obs, scenario)
    for robot in robots:
        values = robot_coordinates(obs, robot['id'])
        robot.update(q_rad=values[:7], finger_q_m=values[7:])
    if empty:
        if obs.get('materials') != [] or 'particle_material' in obs:
            raise ContractError('Empty query may not hide an observed content inventory')
        contents = ()
    elif 'particle_material_request' in world:
        from physics.particle_student_completion import particle_context
        particle_context(world, scenario, obs)
        # Free particles remain exclusively in the actual native scene. The
        # isolated robot/vessel/map query creates no rigid content surrogate.
        contents = ()
    else:
        materials, original = obs["materials"], world["initial_contents"]
        if (len(materials) != len(original) or len(original) != 4
                or any(m["id"] != old["id"] or m["mass_kg"] != old["mass_kg"]
                       or m["inside_carrier"] is not True for m, old in zip(materials, original))):
            raise ContractError("Current persistent material IDs, masses or retention differ")
        contents = [{"id": m["id"], "radius_m": old["radius_m"],
                     "center_world_m": list(point(m["position_world_m"], 3))}
                    for m, old in zip(materials, original)]
    observed = {"schema_version": ("episode-single-active-mixed-query-initialization-v1" if single is not None
                                  else "episode-observed-initialization-v1"),
        "observation": copy.deepcopy(obs), "observation_sha256": digest(obs),
        "source_bindings_sha256": digest(request), "robots_sha256": digest(robots)}
    query_case,_=effective_query_case(request)
    kwargs = dict(case=query_case, active_arm_id=request["active_arm_id"],
        robots=robots, asset_directory=output / "assets", render=False,
        support_stations=[FixedSupportStation(s["id"], s["size_m"], s["center_world_m"])
                          for s in scenario["support_stations"]],
        contents=contents, observed_snapshot=observed)
    if request["kind"] == "carry":
        spec = observed_held_spec(world, scenario, obs, request["held_body"],
            active_arm_id=request["active_arm_id"], repeated_three_arm_mode=True)
        if spec.grasp_model=='provided_ideal':
            query = EpisodeLoadedJointScene(**kwargs, held_body=spec,
                attachment_binding=obs['provided_grasp'],
                maximum_body_tilt_deg=request["planning"]["query_body_tilt_limit_deg"])
        else:
            query = EpisodeLoadedJointScene(**kwargs, held_body=spec,
                contact_policy=SupportedHandleContactPolicy(spec),
                maximum_body_tilt_deg=request["planning"]["query_body_tilt_limit_deg"])
    else:
        spec = None
        query = EpisodeJointScene(**kwargs, model=model, body_pose=_pose(obs["body_world"]),
                                  body_id=scenario["body_id"])
    return query, spec


class _Oracle:
    def __init__(self, query, guard):
        self.query, self.guard = query, guard

    def state_valid(self, q):
        self.guard(); result = self.query.state_valid(q); self.guard()
        return result

    def edge_valid(self, a, b):
        self.guard(); result = self.query.edge_valid(a, b); self.guard()
        return result

    def last_query_diagnostic(self):
        return copy.deepcopy(self.query.last_query_diagnostic())


def _terminal(request, query, spec, oracle, guard):
    from physics.provided_manipulation import solve_endpoint, solve_placement_endpoint
    options = request["planning"]; start = request["command_start_q_rad"]
    if request["kind"] == "retreat":
        if request.get("probe_scope") == "three_arm_short_motion_unit":
            goal = point(request["goal"]["q_rad"], 7)
            if any(not lo <= value <= hi for value, (lo, hi) in zip(goal, query.bounds)):
                raise ContractError("Declared unit probe endpoint exceeds native bounds")
            if not oracle.state_valid(goal):
                raise ContractError("Declared unit probe endpoint is invalid")
            receipt = {"status": "ENDPOINT_FOUND", "q_rad": list(goal), "endpoint_hand_pose": None,
                       "kind": "declared_probe_endpoint", "probe_scope": request["probe_scope"],
                       "scope": "Explicit unit-test terminal endpoint only. Not a normal mission parking target or supplied intermediate route."}
            receipt["receipt_sha256"] = digest(receipt)
            return receipt
        initial = next(r for r in request["scenario"]["robots"] if r["id"] == request["active_arm_id"])["q_rad"]
        native_lower = query.arm.get_dofs_limit()[0]
        dtype = np.asarray(native_lower.detach().cpu().numpy()
                           if hasattr(native_lower, "detach") else native_lower).dtype
        native_initial = np.asarray(initial, dtype=dtype).astype(float).tolist()
        if list(request["goal"]["q_rad"]) not in (list(initial), native_initial):
            raise ContractError("Retreat goal is not this layout's exact native initial command")
        if not oracle.state_valid(native_initial):
            raise ContractError("Declared return-to-initial endpoint is invalid")
        receipt = {"status": "ENDPOINT_FOUND", "q_rad": native_initial,
                   "endpoint_hand_pose": None, "kind": "return_initial",
                   "initial_representation": "Single native-dtype conversion of the exact declared initialization endpoint. No clipping, wrapping or alternate parking target."}
        receipt["receipt_sha256"] = digest(receipt)
        return receipt
    if request.get("placement_goal") is not None:
        receipt = solve_placement_endpoint(query, request["world"], request["scenario"], spec,
            request["placement_goal"]["support_id"], start,
            attempts=options["endpoint_attempts"], seed=options["seed"], wall_check=guard)
        if receipt["placement_goal"] != request["placement_goal"]:
            raise ContractError("Provided placement goal changed from the current measured grasp")
    else:
        kin = BoundedKinematics(query, query.bounds, request["scenario"]["case"]["tool_workspace_m"],
            max_fk_calls=options["endpoint_attempts"], max_ik_calls=options["endpoint_attempts"], wall_check=guard)
        target = WorldPose(request["goal"]["position_m"], request["goal"]["quaternion_wxyz"])
        receipt = solve_endpoint(kin, oracle, start, (target,), attempts=options["endpoint_attempts"],
                                 seed=options["seed"], wall_check=guard)
        receipt["endpoint_kinematics"] = kin.record()
        receipt["receipt_sha256"] = digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    if receipt["status"] != "ENDPOINT_FOUND":
        raise ContractError("No valid provided terminal endpoint. No replacement route is supplied")
    target = request["goal"]
    if receipt["endpoint_hand_pose"] != WorldPose(target["position_m"], target["quaternion_wxyz"]).to_record():
        raise ContractError("Resolved terminal hand pose differs from this request")
    return receipt


def _durable_validation(query, trajectory, case, output, guard):
    from physics.run_joint_path import _json_safe
    path = output / "path_validation.jsonl"; count = 0
    single = len(trajectory.waypoints) == 1
    def samples():
        if single:
            # Two actual fresh reads satisfy the existing durable receipt's
            # minimum. They add no route node, edge, motion or physical step.
            for index in range(2):
                yield None, index, 0, list(trajectory.waypoints[0])
        else:
            resolution = case["planning"]["edge_max_joint_delta_rad"]
            for edge, (a, b) in enumerate(zip(trajectory.waypoints, trajectory.waypoints[1:])):
                intervals = max(1, math.ceil(max(abs(x-y) for x, y in zip(a, b)) / resolution))
                for index in range(intervals+1):
                    q = list(a) if index == 0 else list(b) if index == intervals else [x+(y-x)*index/intervals for x, y in zip(a, b)]
                    yield edge, index, intervals, q
    with path.open("x", encoding="utf8") as stream:
        for edge, index, intervals, q in samples():
            guard(); valid = query.state_valid(q); guard()
            witness = copy.deepcopy(getattr(query, "_last_detection", None))
            if valid and not isinstance(witness, dict):
                raise ContractError("Accepted path sample lacks native same-query witnesses")
            row = {"edge": edge, "sample": index, "intervals": intervals, "q_rad": q,
                "valid": bool(valid), "diagnostic": query.last_query_diagnostic(),
                "same_query": witness, "single_state_repeat_query": single}
            stream.write(json.dumps(_json_safe(row), allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush(); count += 1
            if not valid:
                raise ContractError("Fresh durable whole-path sample rejected")
    guard()
    return {"passed": True, "sample_count": count, "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "single_state_path": single, "continuous_collision_certificate": False}


def plan_student_motion(request, strategy, *, output, source_check, wall_check,fixture_readback='scalar'):
    """Return a source-bound zero-step response or raise after query cleanup.

    request uses student-motion-request-v1. Planning supplies seed, max_queries
    and speed_scale, with optional endpoint_attempts/options/timing-hook/tilt.
    The student's actual plan_path sees only the two terminal q values, native
    bounds and immutable context. Initial q is the preceding command. The exact
    measured q and sampled measured-to-command edge are checked separately.
    Caller-owned initialization and a paused physical grasp/rest gate are
    prerequisites. This function does not authenticate a fabricated request.
    """
    if not callable(source_check) or not callable(wall_check):
        raise ContractError("Source and wall checks must be callable")
    raw_request = copy.deepcopy(request)
    raw_digest = digest(raw_request); request = _request(raw_request)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in ("student_motion_request.json", "student_motion_result.json", "endpoint_receipt.json", "path_validation.jsonl", "assets")):
        raise FileExistsError("Student motion output must be fresh")
    query = None; failure = None; response = None; cleanup = []
    timing = {}; attempt_started = time.perf_counter()
    result = {"schema_version": "student-motion-attempt-v1", "stage_id": request["stage_id"],
        "kind": request["kind"], "request_sha256": raw_digest, "status": "STUDENT_MOTION_FAILED",
        "physical_steps": 0, "full_mission_passed": False, "student_release_approved": False}
    if "probe_scope" in request:
        result["probe_scope"] = request["probe_scope"]
    def guard():
        wall_check()
        if source_check() is not True:
            raise ContractError("Student motion sources changed")
    try:
        guard(); _save(output/"student_motion_request.json", {
            "request": raw_request, "request_sha256": raw_digest,
            "resolved_planning": copy.deepcopy(request["planning"])})
        stage_started = time.perf_counter()
        query, spec = _query_scene(request, output)
        timing['query_scene_construction_s'] = time.perf_counter() - stage_started
        query_case,query_policy=effective_query_case(request)
        result['query_geometry_policy']=query_policy
        if fixture_readback not in ('scalar','batched'):raise ContractError('Unknown fixture readback selection')
        if fixture_readback=='batched':
            result['fixture_readback']=query.enable_batched_fixture_readback();guard()
        install_open_finger_margin(query)
        guard()
        oracle = _Oracle(query, guard); active = request["active_arm_id"]
        observed = point(request["observation"]["arms"][active]["q_all_dofs"][:7], 7)
        start = point(request["command_start_q_rad"], 7)
        tolerance = request["world"]["physics_settings"]["execution_limits"]["final_joint_error_rad"]
        if max(abs(x-y) for x, y in zip(observed, start)) > tolerance:
            raise ContractError("Measured start differs from prior command beyond the existing rest-boundary limit")
        observed_ok = oracle.state_valid(observed); command_ok = oracle.state_valid(start)
        bridge = oracle.edge_valid(observed, start) if observed_ok and command_ok else False
        result.update(observed_start_valid=bool(observed_ok), command_start_valid=bool(command_ok),
                      observed_to_command_edge_valid=bool(bridge))
        if not observed_ok or not command_ok or not bridge:
            raise ContractError("Measured start, command start or their sampled connection is invalid")
        stage_started = time.perf_counter()
        receipt = _terminal(request, query, spec, oracle, guard)
        timing['terminal_ik_and_state_validation_s'] = time.perf_counter() - stage_started
        _save(output/"endpoint_receipt.json", receipt)
        goal = point(receipt["q_rad"], 7); options = request["planning"]; case = query_case
        limits = JointLimits(query.bounds, case["motion_limits"]["velocity_rad_s"], case["motion_limits"]["acceleration_rad_s2"])
        recipe = next(r for r in request["scenario"]["robots"] if r["id"] == active)
        context = JointTaskContext.from_case(case, base_placement=JointBasePlacement(**recipe["base_placement"]),
            limits=limits, start_tool_pose=query.fk(start), goal_tool_pose=query.fk(goal))
        context = replace(context, fixed_base_z_m=recipe.get("base_z_m", case["base_z_m"]))
        kin = BoundedKinematics(query, query.bounds, case["tool_workspace_m"], max_fk_calls=50000,
                                max_ik_calls=10000, wall_check=guard)
        hook = None
        if options["use_edge_timing_hook"]:
            hook = getattr(strategy, "choose_edge_durations_s", None)
            if not callable(hook):
                raise ContractError("Timing opt-in requires the actual student timing hook")
        planning_request = make_joint_request(start, goal, limits, max_queries=options["max_queries"],
            seed=options["seed"], options=options["options"])
        guard()
        stage_started = time.perf_counter()
        plan = plan_joint_motion(planning_request, oracle, strategy, limits,
            speed_scale=options["speed_scale"], kinematics=kin, context=context, edge_timing_hook=hook,
            oracle_description=("Isolated current three-arm, vessel and map geometry with actual measured peers/holder. No free particle collision query or rigid particle proxies. Particle safety and retention require actual native physics."
                if 'particle_material_request' in request['world'] else
                "Isolated complete current three-arm snapshot, actual measured peers and material geometry. Loaded mode uses the current measured holder/body relation. Discrete checks, no dynamic retention proof."))
        timing['planner_and_parameterization_and_final_edge_validation_s'] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        guard(); validation = _durable_validation(query, plan.trajectory, case, output, guard)
        timing['durable_path_collision_validation_s'] = time.perf_counter() - stage_started
        if request.get("placement_goal") is not None:
            from physics.episode_transport_inputs import assess_relay_seating_endpoint
            assessment = assess_relay_seating_endpoint(request["placement_goal"], spec,
                plan.trajectory.waypoints[-1], copy.deepcopy(query._last_detection), state_valid=True)
            result["final_seating_assessment"] = assessment
            if assessment.get("passed") is not True:
                raise ContractError("Fresh exact terminal seating witnesses rejected")
        if query.scene.t != 0 or float(query.scene.cur_t) != 0.:
            raise ContractError("Student query scene advanced physics")
        function = getattr(strategy, "plan_path")
        source_file = inspect.getsourcefile(function)
        result["student_strategy_source"] = {"path": source_file,
            "sha256": None if source_file is None else hashlib.sha256(Path(source_file).read_bytes()).hexdigest()}
        response = {"schema_version": "student-motion-response-v1", "request_sha256": raw_digest,
            "stage_id": request["stage_id"], "kind": request["kind"], "phase": request["kind"],
            "active_arm_id": active, "plan": plan.to_record(), "endpoint_hand_pose": receipt["endpoint_hand_pose"],
            "source_bindings_verified": True, "physical_steps": 0, "observed_start_valid": True,
            "command_start_valid": True, "observed_to_command_edge_valid": True,
            "durable_path_validation": validation, "cleanup_errors": [],
            "provided_adapter_plan_path_invocations": 1}
        response['measured_host_timings_s'] = copy.deepcopy(timing)
        response['query_geometry_policy']=copy.deepcopy(query_policy)
        if "probe_scope" in request:
            response["probe_scope"] = request["probe_scope"]
        if 'particle_material_request' in request['world']:
            response.update(contents_collision_queried=False,
                contents_query_scope='No free particles in the isolated robot/vessel/map query. Actual native particle safety remains caller-owned.')
        result.update(status="STUDENT_MOTION_DISCRETE_PATH_FOUND", response=response)
    except Exception as exc:
        failure = exc
        result.update(failure={"type": type(exc).__name__, "message": str(exc)}, traceback=traceback.format_exc())
    finally:
        if query is not None:
            try: result["query_evidence"] = query.evidence()
            except Exception as exc: cleanup.append("query evidence: " + repr(exc))
            try: query.close()
            except Exception as exc: cleanup.append("query close: " + repr(exc))
        try:
            if source_check() is not True:
                cleanup.append("sources changed by query completion")
        except Exception as exc:
            cleanup.append("source check: " + repr(exc))
        result["cleanup_errors"] = cleanup
        timing['whole_attempt_including_query_cleanup_s'] = time.perf_counter() - attempt_started
        result['measured_host_timings_s'] = timing
        result['timing_scope'] = 'Observed host wall time. Whole attempt contains the named subintervals. Planner interval includes strategy search/shortening and common final edge validation. No GPU-kernel-only claim.'
        if cleanup:
            result["status"] = "STUDENT_MOTION_FAILED"
            result.pop("response", None)
        _save(output/"student_motion_result.json", result)
    if failure is not None:
        raise ContractError("Student motion planning failed: " + str(failure)) from failure
    if cleanup:
        raise ContractError("Student motion cleanup/source verification failed: " + repr(cleanup))
    return response
