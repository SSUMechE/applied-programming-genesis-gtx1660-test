"""Pure current-recipient grasp bindings for an isolated final-carry query.

Trusted runtime receipts bind the actual supported-grasp/rest assessment to
this observation. This module neither measures contacts nor proves a pause,
load-bearing grasp, physical carry or delivery. It never creates a scene.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from physics.carried_body import (VesselModel, WorldPose, HeldBodySpec, _multiply, _inverse,
    validate_provided_grasp_binding)
from physics.episode_phase_inputs import _check_fixed, _station
from physics.episode_paused_state import project_paused_state, _fields, _json, _vector, _pose, _number
from physics.episode_joint_scene import validate_episode_snapshot, native_snapshot_command
from physics.episode_transport_inputs import _seating_from_nominal, _three_arm_layout, selected_support_goal
from physics.episode_transport_assessment import _angle
from physics.handle_contacts import SupportedHandleContactPolicy
from physics.sampled_mission import _FIXED
from transport_challenge.contracts import ContractError, digest


ACTIVE_ARM = 'arm_1'
# Existing episode_runtime.PROTOCOL criterion. No new physical threshold.
FINGER_REST_SPEED_M_S = .001
BOUNDARY_FIELDS = {'schema_version', 'step', 'simulation_s', 'observation_sha256',
    'fixed_state_sha256', 'scenario_sha256', 'source_bindings_sha256',
    'supported_grasp_gate_sha256', 'held_body_sha256', 'previous_commands',
    'active_arm_id', 'support_id', 'endpoint_hand_pose', 'actual_state_verified',
    'measured_rest_verified', 'source_bindings_verified', 'fixed_state_verified', 'grasp_state_verified'}


def observed_held_spec(world, scenario, observation, held_body, *, active_arm_id=None,repeated_three_arm_mode=False):
    """Rebuild from raw current arm_1 observation and require the exact record.

    ``held_body`` may be a serialized record or an immutable HeldBodySpec.
    Reconstructing from raw observations avoids re-normalizing its serialized
    quaternions. No donor transform or already-normalized restoration is used.
    """
    for item in (world, scenario, observation): _json(item)
    if type(repeated_three_arm_mode) is not bool:raise ContractError('Mode must be explicit boolean')
    active=ACTIVE_ARM
    if repeated_three_arm_mode:
        names,_=_three_arm_layout(world,scenario)
        from physics.single_active_boundary import contract_from_observation
        single=contract_from_observation(observation,scenario)
        measured={single['active_arm_id']} if single is not None else set(names)
        if active_arm_id not in measured or set(observation['arms'])!=measured:
            raise ContractError('Explicit current holder and all three measured arms required')
        active=active_arm_id
    elif active_arm_id not in (None,ACTIVE_ARM):
        raise ContractError('Legacy holder is arm_1')
    record = held_body.to_record() if type(held_body) is HeldBodySpec else held_body
    _json(record)
    if (scenario['case'] != world['case_record'] or scenario['provenance']['world_sha256'] != digest(world)
            or scenario['container'] != world['vessel_model']['container'] or active not in observation['arms']):
        raise ContractError('Held observation requires the exact world/scenario and recipient')
    model = VesselModel.from_container(scenario['container'])
    if model.geometry_sha256 != world['vessel_model']['geometry_sha256']:
        raise ContractError('Held model differs from fixed world')
    hand = _pose(observation['arms'][active]['hand_world']); body = _pose(observation['body_world'])
    q = _vector(observation['arms'][active]['q_all_dofs'], 9, 'Observed recipient q')
    step = observation['native_scene_step']
    if type(step) is not int or step < 0: raise ContractError('Invalid current native step')
    grasp={}
    if record.get('grasp_model')=='provided_ideal':
        grasp=dict(grasp_model='provided_ideal',attachment_receipt_sha256=record.get('attachment_receipt_sha256'))
    elif ('grasp_model' in record or 'attachment_receipt_sha256' in record
            or observation.get('provided_grasp') is not None):
        raise ContractError('Automatic attachment requires its explicit provided ideal HeldBodySpec')
    spec = HeldBodySpec(world['object_id'], model, WorldPose(hand['position_m'], hand['quaternion_wxyz']),
        WorldPose(body['position_m'], body['quaternion_wxyz']), tuple(q[7:]),
        _number(observation['simulation_s'], 'Current time'), f'episode_step_{step}',**grasp)
    if spec.grasp_model=='provided_ideal':
        validate_provided_grasp_binding(spec,observation.get('provided_grasp'),
            active_arm_id=active,observation=observation)
    if spec.to_record() != record: raise ContractError('Held relation is not the exact current recipient observation')
    return spec


def _gate_check(gate, observation, boundary, settings, dt, *, active_arm_id=ACTIVE_ARM):
    """Check recorded assessor values. Raw contact/rest truth stays caller-owned."""
    execution = settings['execution_limits']; scenario = settings['scenario']; names = set(observation['arms'])
    if gate.get('passed') is not True or gate.get('source_support_observed') is not True:
        raise ContractError('Current supported grasp did not pass')
    for field, threshold in (('bilateral_window_s', scenario['bilateral_window_s']),
                             ('stable_contents_window_s', scenario['contained_stability_s']),
                             ('source_support_window_s', scenario['bilateral_window_s'])):
        value = _number(gate[field], field)
        if not 0 <= value <= observation['simulation_s']+1e-9 or value < threshold-1e-9:
            raise ContractError('Incomplete supported-grasp observation window')
    duration = execution['rest_window_s']; expected = round(duration/dt)
    for rest in (gate['all_arms_rest'], gate['body_rest']):
        if (rest.get('passed') is not True or rest.get('window_s') != duration
                or type(rest.get('expected_rows')) is not int or type(rest.get('observed_rows')) is not int
                or rest['expected_rows'] != expected or rest['observed_rows'] != expected
                or abs(expected*dt-duration) > 1e-9):
            raise ContractError('Current complete measured rest window required')
    if set(gate['all_arms_rest']['arms']) != names: raise ContractError('Rest gate omits an arm')
    for arm in gate['all_arms_rest']['arms'].values():
        if arm.get('passed') is not True or arm.get('finger_speed_limit_m_s') != FINGER_REST_SPEED_M_S:
            raise ContractError('Arm/finger rest criterion differs')
        for key, limit in (('peak_speed_rad_s', execution['rest_speed_rad_s']),
                           ('peak_acceleration_fd_rad_s2', execution['rest_acceleration_fd_rad_s2']),
                           ('peak_finger_speed_m_s', FINGER_REST_SPEED_M_S)):
            if not 0 <= _number(arm[key], key) <= limit: raise ContractError('Recorded arm/finger rest failed')
    body = gate['body_rest']
    for key, limit_key in (('body_linear_velocity_m_s','maximum_body_linear_speed_m_s'),
                           ('body_angular_velocity_rad_s','maximum_body_angular_speed_rad_s')):
        if body['limits'].get(limit_key) != _FIXED[limit_key] or not 0 <= _number(body['peaks'][key], key) <= _FIXED[limit_key]:
            raise ContractError('Recorded body rest differs from existing prototype policy')
    hand = observation['arms'][active_arm_id]['hand_world']; endpoint = _pose(boundary['endpoint_hand_pose'])
    position = math.dist(hand['position_m'], endpoint['position_m'])
    angle = _angle(hand['quaternion_wxyz'], endpoint['quaternion_wxyz'])
    if (gate['hand_position_error_m'] != position or gate['hand_orientation_error_rad'] != angle
            or position > execution['final_hand_position_error_m'] or angle > execution['final_hand_orientation_error_rad']):
        raise ContractError('Supported-grasp endpoint errors differ from current observation')


def build_held_phase_input(world, scenario, observation, *, active_arm_id, bindings,
                           boundary, supported_grasp_gate, held_body, fixed_state,
                           repeated_three_arm_mode=False,support_id=None):
    """Return detached recipient_carry input, never a route or native verdict."""
    for item in (world, scenario, observation, bindings, boundary, supported_grasp_gate, fixed_state): _json(item)
    if type(repeated_three_arm_mode) is not bool:raise ContractError('Mode must be explicit boolean')
    if repeated_three_arm_mode:
        names,stations=_three_arm_layout(world,scenario)
        if active_arm_id not in names or support_id not in stations:
            raise ContractError('Explicit measured holder and named grasp support required')
    elif active_arm_id != ACTIVE_ARM or support_id is not None:
        raise ContractError('This phase is explicit recipient_carry/arm_1')
    spec = observed_held_spec(world, scenario, observation, held_body,
        active_arm_id=active_arm_id,repeated_three_arm_mode=repeated_three_arm_mode)
    if bindings['world_sha256'] != digest(world): raise ContractError('Pause binding world changed')
    projection = project_paused_state(observation, bindings=bindings)
    _fields(boundary, BOUNDARY_FIELDS, 'Trusted held boundary')
    if (boundary['schema_version'] != 'episode-held-boundary-v1' or boundary['active_arm_id'] != active_arm_id
            or boundary['support_id'] != (support_id if repeated_three_arm_mode else scenario['support_stations'][1]['id'])
            or type(boundary['step']) is not int or boundary['step'] != observation['native_scene_step']
            or type(boundary['simulation_s']) not in (int,float) or boundary['simulation_s'] != observation['simulation_s']):
        raise ContractError('Held boundary has stale clock, holder or grasp support')
    for key, value in (('observation_sha256',observation), ('fixed_state_sha256',fixed_state),
                       ('scenario_sha256',scenario), ('source_bindings_sha256',bindings),
                       ('supported_grasp_gate_sha256',supported_grasp_gate), ('held_body_sha256',spec.to_record())):
        if boundary[key] != digest(value): raise ContractError('Held boundary hash differs: '+key)
    for key in ('actual_state_verified','measured_rest_verified','source_bindings_verified','fixed_state_verified','grasp_state_verified'):
        if boundary[key] is not True: raise ContractError('Trusted runtime did not verify '+key)
    _gate_check(supported_grasp_gate, observation, boundary, world['physics_settings'], bindings['dt_s'],active_arm_id=active_arm_id)
    model, robots, stations, limits = _check_fixed(world, scenario, observation, bindings, fixed_state,
        repeated_three_arm_mode=repeated_three_arm_mode)
    material = projection['state']['materials']; original = world['initial_contents']
    if (len(original) != 4 or len(material) != 4
            or any(a['id'] != b['id'] or a['mass_kg'] != b['mass_kg'] or a['radius_m'] != b['radius_m'] for a,b in zip(original,material))
            or any(m.get('inside_carrier') is not True for m in observation['materials'])
            or any(math.hypot(*m['velocity_world_m_s']) > world['physics_settings']['scenario']['stable_speed_m_s'] for m in material)):
        raise ContractError('Current persistent material identities, masses or retention differ')
    contents = [{'id':m['id'],'radius_m':m['radius_m'],'center_world_m':m['position_world_m']} for m in material]
    commands = boundary['previous_commands']; names = set(observation['arms'])
    if type(commands) is not dict or set(commands) != names: raise ContractError('Previous commands for every arm required')
    differences = {}; tolerance = world['physics_settings']['closure_match_tolerances']
    for name, command in commands.items():
        _fields(command, {'q_rad','finger_q_m'}, 'Previous command')
        values = _vector(command['q_rad'],7,'Seven-q command')+_vector(command['finger_q_m'],2,'Finger command')
        if any(not lo <= v <= hi for v,lo,hi in zip(values,limits[name]['native_lower'],limits[name]['native_upper'])):
            raise ContractError('Previous command exceeds strict native bounds')
        measured = observation['arms'][name]['q_all_dofs']
        arm_error = max(abs(x-y) for x,y in zip(values[:7],measured[:7]))
        finger_error = max(abs(x-y) for x,y in zip(values[7:],measured[7:]))
        if arm_error > tolerance['joint_rad'] or (name != active_arm_id and finger_error > tolerance['finger_m']):
            raise ContractError('Measured state differs from continuous arm/peer-finger command')
        differences[name] = {'maximum_joint_difference_rad':arm_error,'maximum_finger_difference_m':finger_error,
            'active_finger_position_match_required':name != active_arm_id}
    observed = {'schema_version':'episode-observed-initialization-v1','observation':copy.deepcopy(observation),
        'observation_sha256':digest(observation),'source_bindings_sha256':digest(bindings),'robots_sha256':digest(robots)}
    for robot in robots:
        native = limits[robot['id']]; dtype = np.dtype(native['native_dtype'])
        native_snapshot_command(robot,np.asarray(native['native_lower'],dtype=dtype),
            np.asarray(native['native_upper'],dtype=dtype),observed_snapshot=observed)
    snapshot = validate_episode_snapshot(world['case_record'],active_arm_id,robots,model,spec.measured_body_world,
        world['object_id'],[_station(s) for s in stations],contents,observed_snapshot=observed)
    record = {'schema_version':'episode-held-phase-input-v1','phase':'recipient_carry','active_arm_id':active_arm_id,
        'world':copy.deepcopy(world),'scenario':copy.deepcopy(scenario),'world_sha256':digest(world),'scenario_sha256':digest(scenario),
        'observation':copy.deepcopy(observation),'observation_sha256':digest(observation),'fixed_state':copy.deepcopy(fixed_state),
        'source_bindings':copy.deepcopy(bindings),'boundary':copy.deepcopy(boundary),'supported_grasp_gate':copy.deepcopy(supported_grasp_gate),
        'held_body':spec.to_record(),'held_binding_sha256':spec.binding_sha256,'native_projection':projection,
        'query_snapshot':snapshot,'observed_snapshot':observed,'previous_commands':copy.deepcopy(commands),'native_limits':limits,
        'persistent_materials':copy.deepcopy(material),'original_mass_kg':math.fsum(m['mass_kg'] for m in original),
        'observed_start_q_rad':copy.deepcopy(observation['arms'][active_arm_id]['q_all_dofs'][:7]),
        'command_start_q_rad':copy.deepcopy(commands[active_arm_id]['q_rad']),'observed_command_differences':differences,
        'query_scope':'Measured held relation and body-local sphere proxies in an isolated query only. No dynamic retention proof.',
        'command_continuity_scope':'Measured q and preceding commanded q are separate. Caller checks both and their sampled bridge. No executed bridge is inserted.',
        'active_finger_scope':'Exact observed handle-constrained fingers define query geometry. Difference from .015 command is reported, not corrected.',
        'native_grasp_proved_by_this_module':False,'load_bearing_verified':False,'endpoint_supplied':False,
        'route_generated':False,'physical_handoff_verified':False,'final_delivery_verified':False}
    if repeated_three_arm_mode:record.update(repeated_three_arm_mode=True,support_id=support_id)
    return copy.deepcopy({**record,'sha256':digest(record)})


def _validated_record(record):
    _json(record)
    if (record.get('schema_version') != 'episode-held-phase-input-v1'
            or record.get('sha256') != digest({k:v for k,v in record.items() if k != 'sha256'})):
        raise ContractError('Held phase record changed')
    mode=record.get('repeated_three_arm_mode',False)
    kwargs={'repeated_three_arm_mode':True,'support_id':record['support_id']} if mode is True else {}
    if type(mode) is not bool:raise ContractError('Mode must be explicit boolean')
    rebuilt = build_held_phase_input(record['world'],record['scenario'],record['observation'],
        active_arm_id=record['active_arm_id'],bindings=record['source_bindings'],boundary=record['boundary'],
        supported_grasp_gate=record['supported_grasp_gate'],held_body=record['held_body'],fixed_state=record['fixed_state'],**kwargs)
    if rebuilt != record: raise ContractError('Held phase no longer reconstructs exactly')
    return rebuilt


def held_spec(record):
    r = _validated_record(record)
    return observed_held_spec(r['world'],r['scenario'],r['observation'],r['held_body'],
        active_arm_id=r['active_arm_id'],repeated_three_arm_mode=r.get('repeated_three_arm_mode',False))


def query_arguments(record):
    """Detached EpisodeLoadedJointScene kwargs. Caller adds assets/tilt/render."""
    r = _validated_record(record); snap = r['query_snapshot']
    spec = observed_held_spec(r['world'],r['scenario'],r['observation'],r['held_body'],
        active_arm_id=r['active_arm_id'],repeated_three_arm_mode=r.get('repeated_three_arm_mode',False))
    return {'case':copy.deepcopy(snap['case']),'active_arm_id':r['active_arm_id'],
        'robots':copy.deepcopy(snap['robots']),'held_body':spec,'contact_policy':SupportedHandleContactPolicy(spec),
        'support_stations':[_station(s) for s in snap['support_stations']],
        'contents':copy.deepcopy(snap['contents']),'observed_snapshot':copy.deepcopy(r['observed_snapshot'])}


def selected_goal(world, scenario, spec, endpoint, *, repeated_three_arm_mode=False,selected_body_pose=None):
    """Only the official fixed final body/support, composed with this grasp."""
    for item in (world,scenario,endpoint): _json(item)
    _fields(endpoint, {'kind','support_id','command_seating_depth_m'}, 'Final placement endpoint')
    if type(repeated_three_arm_mode) is not bool:raise ContractError('Mode must be explicit boolean')
    if repeated_three_arm_mode:
        _,stations=_three_arm_layout(world,scenario)
        final=endpoint['support_id']==world['goal_support']['id']
        if (endpoint['support_id'] not in stations
                or endpoint['kind']!=('fixed_goal_placement' if final else 'fixed_support_placement')):
            raise ContractError('Placement kind must identify its exact fixed support')
        return selected_support_goal(world,scenario,spec,support_id=endpoint['support_id'],
            selected_body_pose=selected_body_pose,command_seating_depth_m=endpoint['command_seating_depth_m'])
    if selected_body_pose is not None:raise ContractError('Selected body pose requires three-arm mode')
    if (type(spec) is not HeldBodySpec or endpoint['kind'] != 'fixed_goal_placement'
            or endpoint['support_id'] != world['goal_support']['id']
            or scenario['case'] != world['case_record'] or scenario['provenance']['world_sha256'] != digest(world)
            or len(scenario['support_stations']) != 3 or scenario['support_stations'][-1] != world['goal_support']
            or spec.object_id != world['object_id'] or spec.model.geometry_sha256 != world['vessel_model']['geometry_sha256']):
        raise ContractError('Final placement must use the fixed world body and goal support')
    _station(world['goal_support']); raw = _pose(world['goal_body_world'])
    body = WorldPose(raw['position_m'],raw['quaternion_wxyz'])
    orientation = WorldPose((0.,0.,0.),_multiply(body.quaternion_wxyz,_inverse(spec.hand_to_body.quaternion_hand_body_wxyz)))
    offset = orientation.transform_point(spec.hand_to_body.translation_hand_m)
    hand = WorldPose(tuple(p-o for p,o in zip(body.position_m,offset)),orientation.quaternion_wxyz)
    reconstructed = spec.body_pose_at(hand); position_error = math.dist(body.position_m,reconstructed.position_m)
    quaternion_error = min(max(abs(x-y) for x,y in zip(body.quaternion_wxyz,reconstructed.quaternion_wxyz)),
                           max(abs(x+y) for x,y in zip(body.quaternion_wxyz,reconstructed.quaternion_wxyz)))
    if position_error > 256*math.ulp(max(1.,*map(abs,body.position_m))) or quaternion_error > 128*math.ulp(1.):
        raise ContractError('Final hand/body inverse composition did not round-trip')
    nominal = {'support_id':world['goal_support']['id'],'support':copy.deepcopy(world['goal_support']),
        'body_world':copy.deepcopy(raw),'hand_world':hand.to_record(),'reconstructed_body_world':reconstructed.to_record(),
        'roundtrip_position_error_m':position_error,'roundtrip_quaternion_component_error':quaternion_error,
        'held_binding_sha256':spec.binding_sha256,
        'selection_scope':'Official fixed goal body/support with current recipient relation. Endpoint only, no route or placement verdict.'}
    goal = _seating_from_nominal(nominal,spec,endpoint['command_seating_depth_m'])
    if any(not lo <= x <= hi for x,(lo,hi) in zip(goal['hand_world']['position_m'],scenario['case']['tool_workspace_m'])):
        raise ContractError('Final hand endpoint outside fixed workspace')
    return goal
