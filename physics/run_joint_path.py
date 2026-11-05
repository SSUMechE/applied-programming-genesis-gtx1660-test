"""Actual bare-arm joint trajectory execution in the map, with recorded commands.

No IK or supplied transit poses are used during execution. This checkpoint
does not carry a payload and cannot establish material or full-mission success.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from physics.joint_scene import array, initialize_genesis, validate_map
from transport_challenge.contracts import digest
from transport_challenge.joint_planning import JointLimits, JointTrajectory


def _dof_vector(value, label):
    values = array(value)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a finite unbatched vector of all nine model DOFs")
    return values


def _rest_window(rows, *, ending_s, duration_s=.1, speed_limit=.001, acceleration_limit=.05, dt_s=.005):
    """Measured rest evidence, distinct from elapsed fixed hold time."""
    if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Rest-window dt_s must be a positive finite number")
    selected = [row for row in rows if ending_s-duration_s+1e-10 < row["simulation_s"] <= ending_s+1e-10]
    expected = round(duration_s/dt_s)
    peak_speed = max((max(abs(v) for v in row["measured_qd_rad_s"]) for row in selected),default=None)
    peak_acc = max((max(abs(v) for v in row["measured_qdd_fd_rad_s2"]) for row in selected),default=None)
    return {"ending_s":ending_s,"duration_s":duration_s,"observed_rows":len(selected),
        "expected_rows":expected,"speed_limit_rad_s":speed_limit,"acceleration_limit_rad_s2":acceleration_limit,
        "peak_speed_rad_s":peak_speed,"peak_acceleration_fd_rad_s2":peak_acc,
        "passed":len(selected)==expected and peak_speed<=speed_limit and peak_acc<=acceleration_limit}


def _json_safe(value):
    """Keep invalid observations explicit without breaking the failure record."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


_JSON_DIRECT_ATOMIC_TYPES = (str, int, float, bool, type(None))


def _json_can_encode_unchanged(value):
    """Prove that strict JSON needs no key/type conversion for this one tree.

    Nonfinite floats are still rejected by the strict encoder. A caller must
    use the original conversion on encoding failure to retain failure markers.
    No native state or cross-call result is cached.
    """
    return _json_builtin_tree(value, {})


def _json_builtin_tree(value, seen):
    kind = type(value)
    if kind in _JSON_DIRECT_ATOMIC_TYPES:
        return True
    if kind not in (dict, tuple, list):
        return False
    identity = id(value)
    cached = seen.get(identity)
    if cached is not None:
        return cached
    seen[identity] = False  # An active cycle is ineligible, never a new success.
    if kind is dict:
        for key, item in value.items():
            if type(key) is not str or not _json_builtin_tree(item, seen):
                return False
    else:
        for item in value:
            if not _json_builtin_tree(item, seen):
                return False
    seen[identity] = True
    return True


def _cleanup_runtime(state):
    """Attempt every cleanup independently. Never replace an earlier failure."""
    started = time.monotonic()
    errors = []
    operations = []
    if state.get("trace_stream") is not None:
        operations.append(("trace_close", state["trace_stream"], "close"))
    if state.get("recording_attempted") and state.get("camera") is not None:
        operations.append(("video_stop", state["camera"], "stop_recording"))
    gs = state.get("gs")
    if gs is None and state.get("initialization_attempted"):
        gs = sys.modules.get("genesis")
    if gs is not None:
        operations.append(("genesis_destroy", gs, "destroy"))
    attempted = []
    for operation, target, method in operations:
        attempted.append(operation)
        try:
            getattr(target, method)()
        except Exception as error:
            errors.append({"operation": operation, "error_type": type(error).__name__, "error": str(error)})
    return {"attempted": attempted, "errors": errors, "all_attempted_cleanup_completed": not errors,
            "wall_time_s": time.monotonic()-started}


def _partial_evidence(state, error):
    rows = state.get("rows", [])
    result = {"status": "EXECUTION_ERROR", "material_mission_passed": False,
        "error": f"{type(error).__name__}: {error}", "error_type": type(error).__name__,
        "traceback": traceback.format_exc(), "failure_phase": state.get("phase", "validation"),
        "partial_runtime_evidence": True, "all_mission_phases_executed": False,
        "simulation_time_s": state.get("simulation_time_s", 0.),
        "physics_steps": state.get("completed_physics_steps", 0),
        "physics_steps_scope": "Completed scene.step calls only. An interrupted call may have partially advanced.",
        "scene_step_interrupted": state.get("scene_step_in_progress", False),
        "observation_rows": len(rows), "trace_rows_flushed": state.get("trace_rows_flushed", 0),
        "trace_path": "joint_commands_and_observations.jsonl" if state.get("trace_stream") is not None else None,
        "last_recorded_observation": rows[-1] if rows else None,
        "last_unvalidated_observation": state.get("last_unvalidated_observation"),
        "violations": state.get("violations", []), "first_execution_failure": state.get("first_execution_failure"),
        "maximum_tracking_error_rad": max((row["tracking_error_rad"] for row in rows), default=None),
        "continuous_collision_certificate": False, "student_release_approved": False,
        "post_initialization_robot_teleports": 0, "intermediate_ik_calls": 0,
        "controller_command": "control_dofs_position_velocity(q, qd)", "recording_fps": 20}
    if "derivative_findings" in state:
        result["measured_derivative_limits_acceptance"] = "FAIL_OR_INCOMPLETE"
        result["observed_derivative_violating_intervals"] = len(state["derivative_findings"])
        result["first_derivative_finding"] = next(iter(state["derivative_findings"]),None)
    for key in ("versions", "case", "base", "plan_file_sha256", "trajectory_sha256", "planned_motion_time_s",
                "build_wall_s", "native_pair_ledger", "initial_tool_position_error_m", "initial_tool_rotation_error_rad",
                "controller", "controller_gains", "initial_observation", "gravity_feedforward_model"):
        if key in state:
            result[key] = state[key]
    if state.get("run_started") is not None:
        result["execution_wall_s"] = time.monotonic()-state["run_started"]
    if state.get("first_execution_failure") is not None:
        result["status"] = "BARE_ARM_EXECUTION_FAIL"
        result["secondary_execution_error"] = result["error"]
    return _json_safe(result)


def run(args):
    state = {"phase": "validation", "completed_physics_steps": 0, "trace_rows_flushed": 0}
    primary = None
    try:
        result = _run_with_capture(args, state)
    except Exception as error:
        primary = error
        result = _partial_evidence(state, error)
    finally:
        cleanup = _cleanup_runtime(state)
    result["cleanup"] = cleanup
    result["execution_wall_scope"] = "Control loop or failure interval, excluding trace/video/Genesis cleanup"
    if cleanup["errors"] and result["status"] == "BARE_ARM_TRACKING_AND_SAMPLED_CONTACT_PASS":
        result["unaccepted_runtime_status"] = result["status"]
        result["status"] = "EXECUTION_CLEANUP_ERROR"
    if primary is not None:
        # Preserve run()'s exception contract while giving main() the partial
        # evidence. Cleanup exceptions cannot replace this original exception.
        primary.runtime_evidence = result
        raise primary.with_traceback(primary.__traceback__)
    return result


def _run_with_capture(args, state):
    source = json.loads(args.plan.read_bytes())
    if source.get("status") != "FULL_JOINT_DISCRETE_PATH_FOUND":
        raise ValueError("Execution requires a successfully validated joint plan")
    record = source["plan"]
    if digest(record["trajectory"]) != record["evidence"]["trajectory_sha256"]:
        raise ValueError("Recorded joint trajectory digest mismatch")
    raw = record["trajectory"]
    limits = JointLimits(**raw["limits"])
    trajectory = JointTrajectory(raw["waypoints"], raw["edge_durations_s"], limits)
    case, base = source["case"], validate_map(source["case"], source["base"])
    if digest(trajectory.to_record()) != digest(raw):
        raise ValueError("Noncanonical or inconsistent trajectory timing record")
    if (limits.velocity_rad_s != tuple(case["motion_limits"]["velocity_rad_s"])
            or limits.acceleration_rad_s2 != tuple(case["motion_limits"]["acceleration_rad_s2"])):
        raise ValueError("Trajectory derivative limits differ from the instructor case")
    request = record["evidence"]["request"]
    if (tuple(request["start"]) != trajectory.waypoints[0]
            or tuple(request["goal"]) != trajectory.waypoints[-1]
            or digest(trajectory.waypoints) != record["evidence"]["returned_path_sha256"]):
        raise ValueError("Trajectory does not match its planned endpoint request")
    state.update(case=case, base=base, plan_file_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        trajectory_sha256=digest(raw), planned_motion_time_s=trajectory.duration_s,
        phase="initialize_genesis", initialization_attempted=True)
    gs, versions = initialize_genesis(case["planning"]["seed"])
    state.update(gs=gs, versions=versions, phase="construct_scene")
    dt = .005
    gravity_world = (0.,0.,-9.81)
    controller = getattr(args,"controller","pd")
    if controller not in ("pd","gravity_pd"):
        raise ValueError("Unknown supplied controller")
    state["controller"] = controller
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=1,gravity=gravity_world),
        rigid_options=gs.options.RigidOptions(enable_self_collision=True,
            enable_neutral_collision=True, enable_adjacent_collision=False),
        vis_options=gs.options.VisOptions(ambient_light=(.5,.5,.5)),
        renderer=gs.renderers.Rasterizer(), show_viewer=False)
    scene.add_entity(gs.morphs.Plane(), name="floor")
    arm = scene.add_entity(gs.morphs.MJCF(file=case["robot_model"],
        pos=(base[0],base[1],case["base_z_m"]), euler=(0,0,math.degrees(base[2])),
        requires_jac_and_IK=True), name="arm_0")
    torque_sensor = scene.add_sensor(gs.sensors.JointTorque(entity_idx=arm.idx,
        dofs_idx_local=list(range(9)),delay=0.,noise=0.,jitter=0.))
    for index, obstacle in enumerate(case["obstacles"]):
        low, high = np.array(obstacle["lower_m"]), np.array(obstacle["upper_m"])
        scene.add_entity(gs.morphs.Box(size=high-low, pos=(low+high)/2, fixed=True), name=obstacle["id"],
            surface=gs.surfaces.Default(color=((.38,.48,.65),(.56,.41,.31),(.35,.55,.40))[index%3]))
    camera = scene.add_camera(res=(960,720), pos=(1.9,-1.6,1.65), lookat=(.35,0,.43), fov=42, GUI=False)
    state.update(camera=camera, phase="build_scene")
    build_started = time.monotonic()
    scene.build()
    build_wall = time.monotonic()-build_started
    state.update(build_wall_s=build_wall, phase="initialize_robot")
    native_limits = tuple(zip(*(array(v).reshape(-1)[:7].tolist() for v in arm.get_dofs_limit())))
    if native_limits != limits.position_rad:
        raise ValueError("Executed robot's native limits differ from planned model")
    joints = [index for name in case["arm_joint_names"] for index in arm.get_joint(name).dofs_idx_local]
    if joints != list(range(7)):
        raise ValueError("Robot joint-index mapping mismatch")
    # Initial reset is before the physical episode. There are no later teleports.
    arm.set_qpos([*trajectory.waypoints[0],.04,.04])
    # Teacher controller calibration: the preceding unchanged-path run reached
    # the goal without contact but settled 8.87 mm below its target under gravity.
    # Increase stiffness, retaining gravity, native efforts and pose tolerances.
    kp = np.array([9000,9000,7000,7000,4000,4000,4000,2000,2000])
    kv = np.array([450,450,350,350,200,200,200,100,100])*np.array([math.sqrt(2)]*7+[1.,1.])
    arm.set_dofs_kp(kp)
    arm.set_dofs_kv(kv)
    effort = np.array([87,87,87,87,12,12,12,100,100])
    arm.set_dofs_force_range(-effort, effort)
    if arm.gravity_compensation != 0.:
        raise ValueError("Robot material must retain physical gravity")
    state["controller_gains"] = {"kp":kp.tolist(),"kv":kv.tolist(),"effort_limits":effort.tolist()}
    gravity_helper = None
    if controller == "gravity_pd":
        from physics.gravity_feedforward import GravityFeedforward
        gravity_helper = GravityFeedforward(arm,gravity_world)
        state["gravity_feedforward_model"] = gravity_helper.model_evidence.to_record()
    geom_names = {g.idx:{"entity":g.entity.name,"link":g.link.name} for g in scene.rigid_solver.geoms}
    from physics.self_collision_audit import built_self_pair_ledger
    ledger = built_self_pair_ledger(scene.rigid_solver,[arm])
    state["native_pair_ledger"] = ledger
    hand = arm.get_link(case["tool_link"])
    initial_xyz = array(hand.get_pos(relative=False)).reshape(3)
    initial_quat = array(hand.get_quat(relative=False)).reshape(4)
    initial_position_error = float(np.linalg.norm(initial_xyz-np.array(case["start_tool_pose"]["position_m"])))
    initial_rotation_error = 2*math.acos(min(1.,abs(float(np.dot(initial_quat,case["start_tool_pose"]["quaternion_wxyz"])))))
    state.update(initial_tool_position_error_m=initial_position_error, initial_tool_rotation_error_rad=initial_rotation_error)
    if (not np.isfinite(initial_xyz).all() or not np.isfinite(initial_quat).all()
            or initial_position_error > .003 or initial_rotation_error > .02):
        raise ValueError("Initial joint state does not realize the declared map start pose")
    initial_q = _dof_vector(arm.get_qpos(),"initial q")
    initial_qd = _dof_vector(arm.get_dofs_velocity(),"initial qd")
    initial_observation = {"simulation_s":0.,"measured_q_all_dofs":initial_q.tolist(),
        "measured_qd_all_dofs":initial_qd.tolist(),"tool_xyz_m":initial_xyz.tolist(),
        "tool_quaternion_wxyz":initial_quat.tolist(),"coordinate_frame":"world"}
    state["initial_observation"] = initial_observation
    (args.output/"initial_observation.json").write_text(json.dumps(initial_observation,indent=2,allow_nan=False),encoding="utf-8")
    workspace = np.array(case["tool_workspace_m"])
    radius = case.get("tool_reference_radius_m",0.)
    rows, violations = [], []
    state.update(rows=rows, violations=violations, phase="open_trace")
    trace_stream = (args.output/"joint_commands_and_observations.jsonl").open("w",encoding="utf-8")
    state.update(trace_stream=trace_stream, phase="start_recording", recording_attempted=True)
    # Genesis 1.3.2 streams frames during scene.step. Filename/fps belong to
    # start_recording, while stop_recording takes no arguments.
    camera.start_recording(save_to_filename=str(args.output/"joint_motion.mp4"), fps=20)
    sim_time = 0.
    run_started = time.monotonic()
    state.update(run_started=run_started, phase="execute_trajectory")
    trajectory_clock = 0.
    deadline = run_started+args.wall_limit_s
    settling_s = .5
    final_hold_s = .5
    derivative_findings = []
    derivative_tolerance = {"speed_rad_s":1e-6,"acceleration_rad_s2":1e-5}
    # The native controller works in float32. Check its reported clamped effort
    # against independently reconstructed arithmetic with a declared tolerance.
    effort_identity_tolerance = .005
    state["derivative_findings"] = derivative_findings
    steps = math.ceil((settling_s+trajectory.duration_s+final_hold_s)/dt)
    for index in range(steps):
        if time.monotonic() > deadline:
            raise TimeoutError("Joint physical execution exceeded its wall-time limit")
        trajectory_clock = max(0.,min(trajectory.duration_s,(index+1)*dt-settling_s))
        sample = trajectory.sample(trajectory_clock)
        pre_q = _dof_vector(arm.get_qpos(),"pre-step q")
        pre_qd = _dof_vector(arm.get_dofs_velocity(),"pre-step qd")
        tau_g = gravity_helper.torques() if gravity_helper is not None else np.zeros(9)
        tau_g = _dof_vector(tau_g,"gravity compensation effort")
        if gravity_helper is not None:
            arm.set_dofs_act_bias(tau_g,-kp,-kv)
        command_q = np.array([*sample.q_rad,.04,.04])
        command_qd = np.array([*sample.qd_rad_s,0.,0.])
        # Use the same planned q and qd in the supplied PD controller. A
        # zero-velocity damping target creates avoidable lag during motion.
        arm.control_dofs_position_velocity(command_q,command_qd)
        requested_effort = tau_g+kp*(command_q-pre_q)+kv*(command_qd-pre_qd)
        clipped_effort = _dof_vector(arm.get_dofs_control_force(),"pre-step controller effort")
        saturation = np.abs(requested_effort)>effort
        effort_identity_error = float(np.max(np.abs(clipped_effort-np.clip(requested_effort,-effort,effort))))
        if np.any(np.abs(clipped_effort)>effort+1e-5) or effort_identity_error>effort_identity_tolerance:
            raise ValueError("Pre-step native controller effort does not reconcile with the bounded command")
        state["scene_step_in_progress"] = True
        scene.step()
        state.update(scene_step_in_progress=False, completed_physics_steps=index+1,
                     simulation_time_s=(index+1)*dt)
        scene.rigid_solver.check_errno()
        sim_time = (index+1)*dt
        actual_all = _dof_vector(arm.get_qpos(),"post-step q")
        speed_all = _dof_vector(arm.get_dofs_velocity(),"post-step qd")
        actual, speed = actual_all[:7],speed_all[:7]
        acceleration = (speed-pre_qd[:7])/dt
        gearbox_effort = _dof_vector(torque_sensor.read_ground_truth(),"post-step gearbox output effort")
        xyz = array(hand.get_pos(relative=False)).reshape(3)
        measured_quat = array(hand.get_quat(relative=False)).reshape(4)
        tracking = float(np.max(np.abs(actual-np.array(sample.q_rad))))
        if not np.isfinite(acceleration).all() or not np.isfinite(xyz).all() or not np.isfinite(measured_quat).all() or not math.isfinite(tracking):
            state["last_unvalidated_observation"] = {"step": index+1, "simulation_s": sim_time,
                "command_q_rad": list(sample.q_rad), "measured_q_rad": actual.tolist(),
                "measured_qd_rad_s": speed.tolist(), "tool_xyz_m": xyz.tolist(), "tracking_error_rad": tracking}
            raise FloatingPointError("Nonfinite measured robot state. Execution is invalid.")
        contact = arm.get_contacts(exclude_self_contact=False)
        pairs = list(zip(array(contact["geom_a"]).reshape(-1).astype(int), array(contact["geom_b"]).reshape(-1).astype(int)))
        failures = []
        derivative_exceeded = {"speed_joints":[j+1 for j,(v,b) in enumerate(zip(speed,limits.velocity_rad_s)) if abs(v)>b+derivative_tolerance["speed_rad_s"]],
            "acceleration_joints":[j+1 for j,(a,b) in enumerate(zip(acceleration,limits.acceleration_rad_s2)) if abs(a)>b+derivative_tolerance["acceleration_rad_s2"]]}
        if derivative_exceeded["speed_joints"] or derivative_exceeded["acceleration_joints"]:
            derivative_findings.append({"step":index+1,"simulation_s":sim_time,**derivative_exceeded})
        if pairs:
            failures.append({"kind":"observed_robot_contact", "pairs":[{"geom_ids":[int(a),int(b)],
                "a":geom_names[a],"b":geom_names[b]} for a,b in pairs]})
        if tracking > .03:
            failures.append({"kind":"joint_tracking", "maximum_error_rad":tracking, "limit_rad":.03})
        if any(v < lo-1e-5 or v > hi+1e-5 for v,(lo,hi) in zip(actual,limits.position_rad)):
            failures.append({"kind":"measured_native_joint_limit"})
        if np.any(xyz < workspace[:,0]) or np.any(xyz > workspace[:,1]):
            failures.append({"kind":"measured_tool_workspace"})
        for obstacle in case["obstacles"]:
            nearest = np.maximum(obstacle["lower_m"],np.minimum(xyz,obstacle["upper_m"]))
            if float(np.linalg.norm(xyz-nearest)) <= radius:
                failures.append({"kind":"measured_tool_reference_clearance","obstacle":obstacle["id"]})
        rows.append({"step":index+1,"simulation_s":sim_time,"trajectory_s":trajectory_clock,
            "command_q_rad":list(sample.q_rad),"command_qd_rad_s":list(sample.qd_rad_s),
            "command_qdd_rad_s2":list(sample.qdd_rad_s2),"measured_q_rad":actual.tolist(),
            "measured_qd_rad_s":speed.tolist(),"measured_qdd_fd_rad_s2":acceleration.tolist(),
            "pre_q_all_dofs":pre_q.tolist(),"pre_qd_all_dofs":pre_qd.tolist(),
            "post_q_all_dofs":actual_all.tolist(),"post_qd_all_dofs":speed_all.tolist(),
            "gravity_feedforward_effort":tau_g.tolist(),"requested_unclipped_effort":requested_effort.tolist(),
            "pre_step_controller_effort":clipped_effort.tolist(),"post_step_gearbox_effort":gearbox_effort.tolist(),
            "effort_saturated":saturation.tolist(),"effort_identity_error":effort_identity_error,
            "derivative_exceeded":derivative_exceeded,
            "tool_xyz_m":xyz.tolist(),"tool_quaternion_wxyz":measured_quat.tolist(),"tracking_error_rad":tracking,
            "observed_robot_contact_count":len(pairs),"failures":failures})
        if failures:
            violations.extend(failures)
            state.setdefault("first_execution_failure", {"step": index+1, "simulation_s": sim_time,
                "failures": failures})
        trace_stream.write(json.dumps(rows[-1],separators=(",",":"),allow_nan=False)+"\n")
        trace_stream.flush()
        state["trace_rows_flushed"] = len(rows)
        if index%200 == 0:
            (args.output/"progress.json").write_text(json.dumps({"status":"EXECUTING_BARE_ARM",
                "simulation_s":sim_time,"trajectory_duration_s":trajectory.duration_s,"steps":index+1}),encoding="utf-8")
        if failures:
            break
    state["phase"] = "final_observation_summary"
    elapsed = time.monotonic()-run_started
    final_error = float(np.max(np.abs(actual-np.array(trajectory.waypoints[-1]))))
    if not violations and final_error > .01:
        violations.append({"kind":"final_joint_error","error_rad":final_error,"limit_rad":.01})
    final_quat = array(hand.get_quat(relative=False)).reshape(4)
    final_position_error = float(np.linalg.norm(xyz-np.array(case["goal_tool_pose"]["position_m"])))
    final_rotation_error = (2*math.acos(min(1.,abs(float(np.dot(final_quat,case["goal_tool_pose"]["quaternion_wxyz"])))))
                            if np.isfinite(final_quat).all() else None)
    if final_position_error > .006 or final_rotation_error is None or final_rotation_error > .04:
        violations.append({"kind":"final_tool_goal_pose", "position_error_m":final_position_error,
            "rotation_error_rad":final_rotation_error,"limits":{"position_m":.006,"rotation_rad":.04}})
    np_rows = np.array([row["measured_qd_rad_s"] for row in rows])
    peak_speed = np.max(np.abs(np_rows),axis=0).tolist() if rows else None
    peak_acceleration = np.max(np.abs([row["measured_qdd_fd_rad_s2"] for row in rows]),axis=0).tolist() if rows else None
    initial_rest = _rest_window(rows,ending_s=settling_s)
    final_rest = _rest_window(rows,ending_s=sim_time)
    if derivative_findings:
        violations.append({"kind":"measured_joint_derivative_limits","violating_intervals":len(derivative_findings),
            "first":derivative_findings[0],"peak_speed_rad_s":peak_speed,"peak_acceleration_fd_rad_s2":peak_acceleration})
    if not initial_rest["passed"] or not final_rest["passed"]:
        violations.append({"kind":"measured_rest_window","initial":initial_rest,"final":final_rest})
    return {"status":"BARE_ARM_TRACKING_AND_SAMPLED_CONTACT_PASS" if not violations else "BARE_ARM_EXECUTION_FAIL",
        "schema_version":"bare-arm-execution-with-effort-and-derivatives-v2",
        "material_mission_passed":False,"all_mission_phases_executed":False,"versions":versions,
        "case":case,"base":base,"plan_file_sha256":hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "trajectory_sha256":digest(raw),"simulation_time_s":sim_time,"planned_motion_time_s":trajectory.duration_s,
        "build_wall_s":build_wall,"execution_wall_s":elapsed,"physics_steps":len(rows),"substeps_per_step":1,
        "contact_observation_scope":"every completed single-substep physics step, using engine contact buffer",
        "continuous_collision_certificate":False,"native_pair_ledger":ledger,
        "post_initialization_robot_teleports":0,"intermediate_ik_calls":0,
        "controller_command":"control_dofs_position_velocity(q, qd)",
        "controller":controller,"gravity_world_m_s2":gravity_world,"material_gravity_compensation":0.,
        "controller_gains":{"kp":kp.tolist(),"kv":kv.tolist(),"effort_limits":effort.tolist()},
        "gravity_feedforward_model":state.get("gravity_feedforward_model"),
        "gravity_feedforward_last_query":gravity_helper.last_evidence.to_record() if gravity_helper is not None else None,
        "effort_units":{"arm_dofs_0_through_6":"Nm","finger_dofs_7_and_8":"N"},
        "effort_observation_scope":"Controller effort before step is the combined clamped motor command. JointTorque after step is modeled gearbox output, not the motor limit quantity.",
        "effort_identity_tolerance":effort_identity_tolerance,
        "maximum_effort_identity_error":max(row["effort_identity_error"] for row in rows),
        "saturated_step_count":sum(any(row["effort_saturated"]) for row in rows),
        "recording_fps":20,
        "maximum_tracking_error_rad":max((r["tracking_error_rad"] for r in rows),default=None),
        "initial_tool_position_error_m":initial_position_error,"initial_tool_rotation_error_rad":initial_rotation_error,
        "initial_observation":initial_observation,"final_tool_quaternion_wxyz":final_quat.tolist(),
        "initial_rest_window":initial_rest,"final_rest_window":final_rest,
        "final_tool_position_error_m":final_position_error,"final_tool_rotation_error_rad":final_rotation_error,
        "final_joint_error_rad":final_error,"measured_peak_joint_speed_rad_s":peak_speed,
        "measured_peak_joint_acceleration_fd_rad_s2":peak_acceleration,
        "measured_derivative_limits_acceptance":"PASS" if not derivative_findings and len(rows)==steps else "FAIL_OR_INCOMPLETE",
        "derivative_numeric_tolerance":derivative_tolerance,
        "derivative_observation_scope":"Each single integration interval, including t=0 to first step. Acceleration is interval-average finite difference, not a continuous instantaneous bound.",
        "violations":violations,
        "first_execution_failure":state.get("first_execution_failure"),
        "trace_rows_flushed":state["trace_rows_flushed"],
        "student_release_approved":False,"laboratory_gtx1660":"NOT_RUN"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--wall-limit-s",type=float,default=600.)
    parser.add_argument("--controller",choices=("pd","gravity_pd"),default="pd")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    # Source snapshots precede all scene construction. Runs are sequential with
    # source edits and hash equivalence is checked again on exit.
    files = [Path(__file__),ROOT/"physics/joint_scene.py",ROOT/"physics/self_collision_audit.py",
        ROOT/"physics/gravity_feedforward.py",
        ROOT/"transport_challenge/joint_planning.py",ROOT/"transport_challenge/contracts.py",
        ROOT/"transport_challenge/endpoint_planning.py"]
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        target = args.output/"inputs"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copyfile(args.plan,args.output/"inputs/planning_result.json")
    args.plan = args.output/"inputs/planning_result.json"
    started = time.monotonic()
    try:
        result = run(args)
    except Exception as error:
        result = getattr(error, "runtime_evidence", None)
        if result is None:
            result = {"status":"EXECUTION_ERROR","material_mission_passed":False,
                "error":f"{type(error).__name__}: {error}","traceback":traceback.format_exc()}
    result["source_hashes"] = hashes
    result["source_hashes_still_match"] = all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    if not result["source_hashes_still_match"]:
        result["unaccepted_runtime_status"] = result["status"]
        result["status"] = "EXECUTION_SOURCE_CHANGED_NOT_ACCEPTED"
    result["total_wall_s"] = time.monotonic()-started
    result = _json_safe(result)
    (args.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k in ("status","error","simulation_time_s","total_wall_s","maximum_tracking_error_rad","violations")},indent=2))
    return 0 if result["status"] == "BARE_ARM_TRACKING_AND_SAMPLED_CONTACT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
