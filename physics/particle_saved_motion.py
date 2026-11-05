"""Recheck an unchanged particle carry with CPU geometry at its actual pause.

This backend creates no Genesis query scene. It checks robot/map/container
geometry using the real paused arm coordinates and measured held transform.
Particle collisions, retention and dynamics remain the physical worker's work.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time
import traceback

from physics.saved_motion_planning import file_sha, pose_error, require
from transport_challenge.contracts import ContractError, digest, point
from transport_challenge.joint_planning import JointLimits, JointTrajectory, make_joint_request, validate_joint_path

BACKEND = 'cpu_geometry_actual_boundary'
CONTENTS_SCOPE = (
    'No particle collision surrogate or native query scene. Actual native particle '
    'inventory, retained quantity, deformation, contact and rest checks remain '
    'required throughout physical execution.')


def _edge_samples(a, b, resolution):
    a, b = point(a, 7), point(b, 7)
    intervals = max(1, math.ceil(max(abs(x-y) for x, y in zip(a, b))/resolution))
    for index in range(intervals+1):
        q = list(a) if index == 0 else list(b) if index == intervals else [
            x+(y-x)*index/intervals for x, y in zip(a, b)]
        yield index, intervals, q


def _samples(trajectory, resolution):
    for edge, (a, b) in enumerate(zip(trajectory.waypoints, trajectory.waypoints[1:])):
        for index, intervals, q in _edge_samples(a, b, resolution):
            yield edge, index, intervals, q


def _actual_geometry(request, spec):
    from physics.carried_body import WorldPose
    from physics.one_carry_planning import native_precision, unbound_particle_inputs
    from physics.plan_student_transport import native_geometry_precision
    from supplied_support.cpu_geometry import CpuGeometry
    world, scenario = unbound_particle_inputs(request['world'], request['scenario'])
    snapshot = copy.deepcopy(scenario)
    commands = {}
    for robot in snapshot['robots']:
        q9 = list(point(request['observation']['arms'][robot['id']]['q_all_dofs'], 9))
        robot.update(q_rad=q9[:7], finger_q_m=q9[7:])
        commands[robot['id']] = {'q_rad': q9[:7], 'finger_q_m': q9[7:]}
    snapshot['body_pose'] = copy.deepcopy(request['observation']['body_world'])
    require(snapshot['contents'] == [] and world['initial_contents'] == [],
            'Particle CPU recheck may not create rigid content substitutes')
    geometry = native_geometry_precision(CpuGeometry(world, snapshot), snapshot, native_precision(world))
    body = request['observation']['body_world']
    cpu = geometry.configure(request['active_arm_id'], commands,
        WorldPose(body['position_m'], body['quaternion_wxyz']),
        holder_arm_id=request['active_arm_id'], hand_to_body=spec.hand_to_body.to_record())
    return geometry, cpu


def _endpoint(geometry, cpu, request, stage, spec, oracle):
    """CPU nominal FK plus unchanged physical-target geometric predicates."""
    import mujoco
    from physics.carried_body import WorldPose
    from physics.episode_transport_inputs import (
        _bottom_corners, VERIFY_POSITION_TOLERANCE_M, VERIFY_ORIENTATION_TOLERANCE_RAD)
    from physics.handle_contacts import MAX_PENETRATION_M, classify_supported_contacts
    goal = tuple(stage['trajectory']['waypoints'][-1])
    require(oracle.state_valid(goal), 'Saved particle endpoint fails actual-boundary CPU geometry')
    tool = cpu.fk(goal)
    hand = WorldPose(tool.position_m, tool.quaternion_wxyz).to_record()
    nominal_error = pose_error(hand, stage['nominal_goal_hand_world'])
    require(nominal_error['position_m'] <= VERIFY_POSITION_TOLERANCE_M
            and nominal_error['orientation_rad'] <= VERIFY_ORIENTATION_TOLERANCE_RAD,
            'Saved particle endpoint differs from its nominal hand FK target')
    placement = request['placement_goal']
    nominal = placement.get('seating', {}).get('nominal_goal', placement)
    require(stage['destination_support_id'] == nominal['support_id']
            and stage['nominal_goal_body_world'] == nominal['body_world'],
            'Saved particle destination support or canonical body target changed')
    actual_body = spec.body_pose_at(WorldPose(hand['position_m'], hand['quaternion_wxyz']))
    error = pose_error(actual_body.to_record(), nominal['body_world'])
    execution = request['world']['physics_settings']['execution_limits']
    require(error['position_m'] <= execution['final_body_position_error_m']
            and error['orientation_rad'] <= execution['final_body_orientation_error_rad'],
            'Saved particle path with the actual grasp misses original body target limits')
    support = nominal['support']; center = support['center_world_m']; size = support['size_m']
    top = center[2]+size[2]/2
    corners = _bottom_corners(spec.model, actual_body)
    footprint = all(abs(c[i]-center[i]) <= size[i]/2 for c in corners for i in (0, 1))
    gaps = [c[2]-top for c in corners]
    require(footprint and all(-MAX_PENETRATION_M < gap < 0. for gap in gaps),
            'Actual-grasp particle endpoint misses original support footprint/contact-depth limits')
    # state_valid has already applied the unchanged contact classifier and the
    # additional conservative CPU footprint rule. Collect this endpoint's CPU
    # named-support rows without calling them native contact observations.
    mujoco.mj_collision(geometry.model, geometry.data)
    rows = []
    for contact in geometry.data.contact:
        a, b = int(contact.geom1), int(contact.geom2)
        if not ({a, b} & set(geometry.components)) or not ({a, b} & set(geometry.stations)):
            continue
        raw = {'geom_a': [a], 'geom_b': [b], 'position': [contact.pos.tolist()],
               'normal': [contact.frame[:3].tolist()], 'penetration': [float(-contact.dist)]}
        classified, _ = classify_supported_contacts(raw, components_by_geom=geometry.components,
            pads_by_geom={}, body_pose=actual_body.to_record(), finger_poses={},
            stations_by_geom=geometry.stations, phase='settle')
        rows.extend(row for row in classified if row.get('category') == 'named_tray_bottom_support_top'
                    and row.get('support_station_id') == nominal['support_id']
                    and row.get('tray_component') == 'solid_0')
    require(bool(rows), 'Particle CPU endpoint lacks named bottom/support CPU contact geometry')
    return {'schema_version': 'saved-particle-cpu-nominal-endpoint-v1', 'passed': True,
        'validation_backend': BACKEND, 'native_query_certificate': False, 'fresh_ik_calls': 0,
        'frozen_endpoint_q_rad': list(goal), 'cpu_hand_world': hand,
        'frozen_nominal_target_fk_error': nominal_error,
        'frozen_nominal_target_fk_limits': {'position_m': VERIFY_POSITION_TOLERANCE_M,
                                         'orientation_rad': VERIFY_ORIENTATION_TOLERANCE_RAD},
        'predicted_actual_body_world': actual_body.to_record(),
        'canonical_goal_body_world': nominal['body_world'], 'actual_grasp_body_target_error': error,
        'unchanged_physical_body_limits': {'position_m': execution['final_body_position_error_m'],
                                          'orientation_rad': execution['final_body_orientation_error_rad']},
        'bottom_footprint_inside_support': footprint, 'bottom_signed_gaps_m': gaps,
        'cpu_named_bottom_support_witnesses': rows,
        'old_actual_grasp_seating_ik_certificate_claimed': False,
        'physical_support_rest_release_required': True, 'endpoint_hand_pose': stage['nominal_goal_hand_world']}


def validate_particle_saved_motion_cpu(request, saved_plan, index, *, previous_commands,
        output, source_check, wall_check, fixture_readback='scalar'):
    from physics import student_motion_planning as motion
    from physics.episode_held_phase_inputs import observed_held_spec
    from physics.one_carry_planning import SavedOneCarryPlan
    require(isinstance(saved_plan, SavedOneCarryPlan), 'Require an owned saved one-carry plan')
    record = saved_plan.record()
    require(record['payload'] in ('process_liquid', 'sand')
            and record.get('particle_recheck_mode') == BACKEND,
            'Particle CPU recheck must be selected by the sealed material plan')
    require(callable(source_check) and callable(wall_check), 'Live source and wall guards are required')
    require(fixture_readback in ('scalar', 'batched'), 'Unknown supplied fixture readback option')
    request = motion._request(copy.deepcopy(request))
    stage = saved_plan.stage(index); active = request['active_arm_id']
    require(index == 0 and (request['stage_id'], request['kind'], active) == saved_plan.stage_signatures[0],
            'Require this exact one-carry operation')
    saved_plan.verify_bound_inputs(request['world'], request['scenario'])
    require(stage['expected_peer_commands_by_arm'] == {k:v for k,v in previous_commands.items() if k != active},
            'Actual previous peer commands differ from the frozen carry')
    require(stage['trajectory']['waypoints'][0] == previous_commands[active]['q_rad'] == request['command_start_q_rad'],
            'Saved particle path must start at the exact previous command')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    require(not any((output/name).exists() for name in (
        'saved_motion_request.json', 'saved_motion_result.json', 'path_validation.jsonl',
        'saved_endpoint_receipt.json', 'cpu_geometry_evidence.json')), 'Particle CPU validation output must be fresh')
    started = time.perf_counter(); failure = response = cpu = geometry = None
    result = {'schema_version': 'saved-motion-attempt-v1', 'status': 'SAVED_MOTION_VALIDATION_FAILED',
        'validation_backend': BACKEND, 'stage_id': request['stage_id'], 'request_sha256': digest(request),
        'saved_bundle_sha256': saved_plan.sha256, 'saved_stage_sha256': digest(stage),
        'physical_steps': 0, 'fresh_ik_calls': 0, 'runtime_rrt_calls': 0,
        'provided_adapter_plan_path_invocations': 0, 'contents_collision_queried': False,
        'contents_query_scope': CONTENTS_SCOPE, 'native_query_scene_created': False,
        'measured_host_timings_s': {}, 'cleanup_errors': [], 'full_mission_passed': False,
        'student_release_approved': False}
    def save(name, value):
        with (output/name).open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
    def guard():
        wall_check(); require(source_check() is True, 'Particle CPU validation sources changed')
    def timed(name, action):
        start = time.perf_counter()
        try: return action()
        finally: result['measured_host_timings_s'][name] = time.perf_counter()-start
    class Guarded:
        def state_valid(self, q):
            guard(); valid = cpu.state_valid(q); guard(); return valid
        def edge_valid(self, a, b):
            # Guard every internal sample, preserving the existing resolution.
            return all(self.state_valid(q) for _, _, q in _edge_samples(a, b, geometry.resolution))
        def last_query_diagnostic(self): return cpu.last_query_diagnostic()
    try:
        guard(); timed('saved_cpu_source_boundary_s', saved_plan.verify_sources)
        save('saved_motion_request.json', {'request': request, 'saved_stage': stage,
                                        'saved_bundle_sha256': saved_plan.sha256})
        spec = observed_held_spec(request['world'], request['scenario'], request['observation'],
            request['held_body'], active_arm_id=active, repeated_three_arm_mode=True)
        require(spec.grasp_model == 'provided_ideal', 'Actual supported native attachment is required')
        actual_body_error = pose_error(request['observation']['body_world'], stage['nominal_body_world'])
        execution = request['world']['physics_settings']['execution_limits']
        peer_errors = {name:max(abs(a-b) for a,b in zip(
            request['observation']['arms'][name]['q_all_dofs'][:7], command['q_rad']))
            for name, command in stage['expected_peer_commands_by_arm'].items()}
        require(actual_body_error['position_m'] <= execution['final_body_position_error_m']
                and actual_body_error['orientation_rad'] <= execution['final_body_orientation_error_rad'],
                'Actual particle vessel differs from the saved nominal boundary beyond original limits')
        require(all(error <= execution['final_joint_error_rad'] for error in peer_errors.values()),
                'Actual particle peers exceed the original resting command tolerance')
        result['nominal_actual_boundary'] = {'body_pose_error': actual_body_error,
            'peer_measured_command_maximum_joint_error_rad': peer_errors,
            'peer_geometry_source': 'All actual paused observed nine-DOF coordinates. No ideal-peer substitution.'}
        geometry, cpu = timed('cpu_actual_geometry_construction_s', lambda: _actual_geometry(request, spec))
        limits = JointLimits(cpu.bounds, request['scenario']['case']['motion_limits']['velocity_rad_s'],
                             request['scenario']['case']['motion_limits']['acceleration_rad_s2'])
        tr = stage['trajectory']; require(tr['limits'] == limits.to_record(), 'Saved particle native limits changed')
        trajectory = JointTrajectory(tr['waypoints'], tr['edge_durations_s'], limits)
        require(trajectory.to_record() == tr, 'Saved particle trajectory or timing changed')
        oracle = Guarded(); observed = point(request['observation']['arms'][active]['q_all_dofs'][:7], 7)
        start = point(request['command_start_q_rad'], 7)
        require(max(abs(a-b) for a,b in zip(observed, start)) <= execution['final_joint_error_rad'],
                'Actual particle arm start exceeds the original command-boundary tolerance')
        def bridge():
            require(oracle.state_valid(observed) and oracle.state_valid(start) and oracle.edge_valid(observed, start),
                    'Actual particle observed/command boundary or sampled bridge is invalid')
        timed('observed_command_bridge_validation_s', bridge)
        endpoint = timed('saved_endpoint_cpu_validation_s', lambda: _endpoint(geometry, cpu, request, stage, spec, oracle))
        save('saved_endpoint_receipt.json', endpoint)
        planning = request['planning']
        planned = make_joint_request(start, trajectory.waypoints[-1], limits,
            max_queries=planning['max_queries'], seed=planning['seed'], options=planning['options'])
        timed('common_final_state_edge_validation_s', lambda: validate_joint_path(trajectory.waypoints, planned, limits, oracle))
        def durable():
            count = 0
            with (output/'path_validation.jsonl').open('x', encoding='utf-8') as stream:
                for edge, sample, intervals, q in _samples(trajectory, geometry.resolution):
                    valid = oracle.state_valid(q)
                    stream.write(json.dumps({'edge':edge, 'sample':sample, 'intervals':intervals,
                        'q_rad':q, 'valid':bool(valid), 'diagnostic':cpu.last_query_diagnostic(),
                        'validation_backend':BACKEND, 'native_query_certificate':False}, allow_nan=False)+'\n')
                    stream.flush(); count += 1
                    require(valid, 'Fresh durable particle CPU path sample rejected')
            guard()
            return {'passed':True, 'sample_count':count, 'trace_sha256':file_sha(output/'path_validation.jsonl'),
                'validation_backend':BACKEND, 'continuous_collision_certificate':False,
                'contents_collision_queried':False, 'native_query_scene_created':False}
        validation = timed('durable_path_collision_validation_s', durable)
        offline = saved_plan.offline_planner
        evidence = {'trajectory_sha256':digest(tr), 'returned_path_sha256':digest(tr['waypoints']),
            'plan_path_source_sha256':offline['plan_path_source_sha256'], 'dimension':7,
            'motion_source':'saved_cpu_plan', 'offline_planner':offline,
            'saved_bundle_sha256':saved_plan.sha256, 'runtime_plan_path_invocations':0,
            'runtime_ik_calls':0, 'runtime_rrt_calls':0, 'physical_execution_performed':False,
            'offline_planning':stage['offline_planning'], 'validation_backend':BACKEND}
        response = {'schema_version':'saved-motion-response-v1', 'request_sha256':digest(request),
            'stage_id':request['stage_id'], 'kind':request['kind'], 'active_arm_id':active,
            'plan':{'schema_version':'joint-plan-v1', 'trajectory':tr, 'evidence':evidence},
            'endpoint_hand_pose':endpoint['endpoint_hand_pose'], 'source_bindings_verified':True,
            'physical_steps':0, 'observed_start_valid':True, 'command_start_valid':True,
            'observed_to_command_edge_valid':True, 'durable_path_validation':validation,
            'provided_adapter_plan_path_invocations':0, 'runtime_ik_calls':0, 'runtime_rrt_calls':0,
            'saved_bundle_sha256':saved_plan.sha256, 'saved_stage_sha256':digest(stage),
            'cleanup_errors':[], 'query_geometry_policy':{'validation_backend':BACKEND,
                'tracking_clearance_m':geometry.clearance, 'edge_max_joint_delta_rad':geometry.resolution,
                'native_query_scene_created':False}, 'endpoint_assessment':endpoint,
            'validation_backend':BACKEND, 'contents_collision_queried':False,
            'contents_query_scope':CONTENTS_SCOPE}
        result.update(status='SAVED_MOTION_DISCRETE_PATH_VALIDATED', response=response)
    except Exception as exc:
        failure = exc
        result.update(failure={'type':type(exc).__name__, 'message':str(exc)}, traceback=traceback.format_exc())
    finally:
        if geometry is not None:
            try: save('cpu_geometry_evidence.json', {'geometry':geometry.evidence(), 'queries':cpu.evidence(),
                'validation_backend':BACKEND, 'contents_collision_queried':False, 'contents_query_scope':CONTENTS_SCOPE})
            except Exception as exc: result['cleanup_errors'].append('CPU evidence: '+repr(exc))
        try:
            timed('saved_cpu_final_source_boundary_s', saved_plan.verify_sources); guard()
        except Exception as exc: result['cleanup_errors'].append('source verification: '+repr(exc))
        if result['cleanup_errors']:
            result['status'] = 'SAVED_MOTION_VALIDATION_FAILED'; result.pop('response', None)
        result['measured_host_timings_s']['whole_attempt_including_cleanup_s'] = time.perf_counter()-started
        if response is not None: response['measured_host_timings_s'] = copy.deepcopy(result['measured_host_timings_s'])
        save('saved_motion_result.json', result)
    if failure is not None:
        raise ContractError('Particle CPU saved-path validation failed: '+str(failure)) from failure
    require(not result['cleanup_errors'], 'Particle CPU final source/evidence checks failed')
    return response
