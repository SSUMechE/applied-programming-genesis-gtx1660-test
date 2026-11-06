"""Execute an external seven-joint program with the provided native grasp.

The real scene retains all three arms, the vessel, gravity and map collisions.
Only contents are absent. Geometry validation never plans or repairs a path.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time

from transport_challenge.contracts import ContractError, digest, point
from transport_challenge.joint_planning import JointTrajectory
from transport_challenge.empty_container_program import resolve_empty_program


def _require(condition, message):
    if not condition:
        raise ContractError(message)


def _guard(state):
    state['wall_check']()
    _require(state['source_check']() is True, 'External path sources changed')


def _validate_geometry(driver, dispatcher, state, trajectory):
    """Check the exact user states/edges in an isolated, zero-step full scene."""
    from physics.episode_loaded_joint_scene import EpisodeLoadedJointScene
    from physics.fixed_world_task import FixedSupportStation
    from physics.provided_query_clearance import install_open_finger_margin
    from physics.student_motion_planning import _Oracle, _durable_validation
    from physics.automatic_runtime_binding import physical_observation

    folder=Path(state['output'])/'empty_path_geometry'
    folder.mkdir(exist_ok=False)
    obs=copy.deepcopy(state['observation'])
    before=copy.deepcopy(state['observe']())
    dispatch_before=copy.deepcopy(dispatcher.record())
    _require(physical_observation(before)==physical_observation(obs)
        and before.get('provided_grasp')==obs.get('provided_grasp'),
        'Geometry check does not start at the current measured state')
    robots=copy.deepcopy(driver.scenario['robots'])
    for robot in robots:
        values=list(point(obs['arms'][robot['id']]['q_all_dofs'],9))
        robot.update(q_rad=values[:7],finger_q_m=values[7:])
    binding=dict(program_sha256=digest(driver.empty_program),trajectory_sha256=digest(trajectory.to_record()),
        world_sha256=digest(driver.world),scenario_sha256=digest(driver.scenario))
    snapshot=dict(schema_version='episode-observed-initialization-v1',observation=obs,
        observation_sha256=digest(obs),source_bindings_sha256=digest(binding),robots_sha256=digest(robots))
    query=None
    evidence={}
    started=time.monotonic()
    try:
        _guard(state)
        query=EpisodeLoadedJointScene(copy.deepcopy(driver.scenario['case']),
            active_arm_id=driver.empty_program['arm_id'],robots=robots,held_body=driver.held,
            attachment_binding=obs['provided_grasp'],asset_directory=folder/'assets',
            support_stations=[FixedSupportStation(s['id'],s['size_m'],s['center_world_m'])
                for s in driver.scenario['support_stations']],
            contents=(),observed_snapshot=snapshot,maximum_body_tilt_deg=None,render=False)
        if driver.fixture_readback=='batched':
            evidence['fixture_readback']=query.enable_batched_fixture_readback()
        install_open_finger_margin(query)
        oracle=_Oracle(query,lambda:_guard(state))
        arm=driver.empty_program['arm_id']
        measured=point(obs['arms'][arm]['q_all_dofs'][:7],7)
        start=trajectory.waypoints[0]
        tolerance=state['settings']['execution_limits']['final_joint_error_rad']
        _require(max(abs(a-b) for a,b in zip(measured,start))<=tolerance,
            'Measured start exceeds the original command-boundary tolerance')
        _require(oracle.state_valid(measured) and oracle.state_valid(start)
            and oracle.edge_valid(measured,start),
            'Measured-to-command start connection is invalid: '+str(query.last_query_diagnostic()))
        for index,(a,b) in enumerate(zip(trajectory.waypoints,trajectory.waypoints[1:])):
            _require(oracle.state_valid(b) and oracle.edge_valid(a,b),
                'External path edge '+str(index)+' collides: '+str(query.last_query_diagnostic()))
        evidence['durable_path_validation']=_durable_validation(query,trajectory,driver.scenario['case'],folder,lambda:_guard(state))
        _require(query.scene.t==0 and float(query.scene.cur_t)==0.,'Geometry query advanced physics')
        evidence.update(passed=True,physical_steps=0,planner_invocations=0,
            measured_to_command_edge_valid=True,contents_count=0,
            continuous_collision_certificate=False,query_evidence=query.evidence())
    finally:
        if query is not None:
            query.close()
        _guard(state)
        _require(state['observe']()==before and dispatcher.record()==dispatch_before,
            'Geometry validation changed live physics, clock or Dispatcher')
        evidence.update(wall_s=time.monotonic()-started,live_state_and_clock_unchanged=True,dispatcher_unchanged=True)
        driver._save(state,'empty_path_geometry_receipt.json',evidence)
    return evidence


def advance(driver, dispatcher, state):
    from physics.automatic_manipulation import pick, rested, validate_observation, held_spec
    from physics.episode_continuation_protocol import rest_receipt
    _guard(state)
    _require(dispatcher.record().get('current_segment_complete') is True,
        'An incomplete or truncated external path cannot complete')
    arm=driver.empty_program['arm_id']
    if driver.empty_phase=='initial_settle':
        op=dict(stage_id='provided_empty_grasp',kind='approach',active_arm_id=arm,support_id=driver.route[0])
        gate=pick(driver,state,op)
        driver.gates.append(dict(**op,gate=gate))
        driver.empty_start_observation=copy.deepcopy(state['observation'])
        driver._capture('empty_grasp_start',state,dict(payload_mode='empty',provided_grasp=True))
        obs=state['observation']
        commands=dispatcher.record()['last_command_by_arm']
        resolved=resolve_empty_program(driver.empty_program,commands[arm]['q_rad'],driver.limits_by_arm[arm])
        driver.empty_resolution=resolved
        trajectory=JointTrajectory(resolved['waypoints_rad'],resolved['edge_durations_s'],driver.limits_by_arm[arm])
        geometry=_validate_geometry(driver,dispatcher,state,trajectory)
        actions=[dict(action_id='empty_external_joint_path',arm_id=arm,role='carry',kind='trajectory',
            trajectory=trajectory.to_record()),
            dict(action_id='empty_external_end_hold',arm_id=arm,role='carry',kind='hold',duration_s=resolved['end_hold_s'])]
        plan=driver._plan(obs,commands,actions,'empty_external_program',external_grid=True)
        _require(plan.actions[0].steps==resolved['motion_steps']
            and plan.actions[1].steps==resolved['end_hold_steps'], 'External timing grid changed')
        boundary=rest_receipt(dispatcher.record(),obs,list(driver.arm_ids))
        dispatcher.extend(plan,model_sha256_by_arm={n:driver.world['robot_model_sha256'] for n in driver.arm_ids},boundary=boundary)
        driver.empty_motion_start_step=obs['native_scene_step']
        driver.current=copy.deepcopy(driver.queue[0])
        driver.empty_phase='executing_external_path'
        driver.plans.append(dict(stage_id='empty_external_joint_path',active_arm_id=arm,
            trajectory_sha256=digest(trajectory.to_record()),program_sha256=digest(driver.empty_program),
            duration_s=trajectory.duration_s,waypoint_count=len(trajectory.waypoints),
            input_timing_preserved=True,planner_invocations=0,geometry=geometry))
        driver._save(state,'resolved_empty_program.json',resolved)
        driver._save(state,'empty_external_program__episode.json',plan.to_record())
        return True
    _require(driver.empty_phase=='executing_external_path', 'Unexpected empty path phase')
    obs=state['observation']
    arms,body,material=rested(driver,state)
    validate_observation(driver,obs,arm)
    resolved=driver.empty_resolution
    expected_steps=resolved['motion_steps']+resolved['end_hold_steps']
    _require(obs['native_scene_step']-driver.empty_motion_start_step==expected_steps,
        'Actual external motion/hold step count differs from input')
    commands=dispatcher.record()['last_command_by_arm']
    _require(commands[arm]['q_rad']==resolved['waypoints_rad'][-1]
        and driver.provided_native_target_rows==obs['native_scene_step'],
        'Final command or per-step native target evidence differs')
    error=max(abs(a-b) for a,b in zip(obs['arms'][arm]['q_all_dofs'][:7],resolved['waypoints_rad'][-1]))
    _require(error<=state['settings']['execution_limits']['final_joint_error_rad'], 'Final native tracking error exceeds original bound')
    start=driver.empty_start_observation
    moved=math.dist(start['body_world']['position_m'],obs['body_world']['position_m'])
    joint_move=max(abs(a-b) for a,b in zip(start['arms'][arm]['q_all_dofs'][:7],obs['arms'][arm]['q_all_dofs'][:7]))
    _require(joint_move>1e-5, 'External input produced no measured joint movement')
    gate=dict(schema_version='empty-native-joint-path-completion-v1',passed=True,active_arm_id=arm,
        native_scene_step=obs['native_scene_step'],simulation_s=obs['simulation_s'],
        expected_motion_and_hold_steps=expected_steps,actual_motion_and_hold_steps=expected_steps,
        input_timing_preserved=True,final_joint_error_rad=error,measured_body_displacement_m=moved,
        measured_body_height_change_m=obs['body_world']['position_m'][2]-start['body_world']['position_m'][2],
        measured_max_joint_displacement_rad=joint_move,all_arms_rest=arms,body_rest=body,
        empty_payload=material,automatic_holder=driver.automatic_grasp.holder,
        final_observation_sha256=digest(obs),native_target_rows=driver.provided_native_target_rows)
    driver.empty_completion=gate
    driver.gates.append(dict(stage_id='empty_external_joint_path',kind='carry',active_arm_id=arm,gate=gate))
    driver.completed.append(dict(stage_id='empty_external_joint_path',native_scene_step=obs['native_scene_step'],simulation_s=obs['simulation_s']))
    driver.held=held_spec(driver,obs,arm)
    driver.held_observation_sha256=digest(obs)
    driver.done=True
    driver.index=1
    driver.empty_phase='held_end'
    driver._capture('empty_path_end',state,gate)
    driver._save(state,'empty_path_completion.json',gate)
    return False


def record(driver):
    return dict(schema_version='empty-container-external-path-driver-v1',mode=driver.mode,
        initialized=driver.initialized,operation_sequence_complete=driver.done,
        phase=driver.empty_phase,payload_mode='empty',material_count=0,material_rest='NOT_APPLICABLE',
        program=copy.deepcopy(driver.empty_program),resolved_program=copy.deepcopy(driver.empty_resolution),
        planned_motions=copy.deepcopy(driver.plans),provided_gates=copy.deepcopy(driver.gates),
        completed=copy.deepcopy(driver.completed),automatic_events=copy.deepcopy(driver.automatic_events),
        image_captures=copy.deepcopy(driver.image_captures),native_target_rows=driver.provided_native_target_rows,
        current_held_body=None if driver.held is None else driver.held.to_record(),
        planner_invocations=0,reference_planner_fallback=False,supplied_intermediate_postures=False,
        motion_source='external_joint_waypoints',student_release_approved=False)


def final_result(driver, observation):
    from physics.automatic_manipulation import validate_observation
    _require(driver.done and driver.empty_phase=='held_end' and driver.held is not None
        and driver.held_observation_sha256==digest(observation)
        and driver.empty_completion['final_observation_sha256']==digest(observation),
        'External empty path did not reach its verified held end')
    validate_observation(driver,observation,driver.empty_program['arm_id'])
    return dict(status='EMPTY_CONTAINER_JOINT_PATH_SAMPLED_PASS',final_observation=observation,
        student_execution=record(driver),empty_path_completion=copy.deepcopy(driver.empty_completion),
        payload_mode='empty',material_count=0,material_rest='NOT_APPLICABLE',
        full_mission_passed=False,material_delivery_accepted=False,physical_handoff_verified=False,
        student_release_approved=False,
        scope='Actual external seven-joint trajectory and exact edge times with a provided native grasp in the three-arm map. End held. No contents, handoff or full delivery claim.')
