"""Supplied detached continuation contracts. No path generation or native calls.

The trusted runtime owns observation, process, collision and rest evidence.
These checks cannot authenticate a fabricated child result.
"""
from __future__ import annotations

import copy
import math
import re

from transport_challenge.contracts import ContractError, digest
from transport_challenge.episode_plan import EpisodePlan
from transport_challenge.joint_planning import JointTrajectory


def _phase_contract(item, index=None, *, repeated_three_arm_mode=False):
    if type(repeated_three_arm_mode) is not bool:
        raise ContractError('Repeated three-arm mode must be an explicit Boolean')
    sequence = (('donor_retreat', 'arm_0'), ('recipient_approach', 'arm_1'),
                ('recipient_carry', 'arm_1'))
    fields = {'phase', 'active_arm_id', 'endpoint'}
    if repeated_three_arm_mode:
        fields |= {'stage_id', 'donor_arm_id', 'recipient_arm_id', 'support_id', 'destination_support_id'}
    if not isinstance(item, dict) or set(item) != fields:
        raise ContractError('Continuation phase fields must be explicit and exact')
    pair = (item['phase'], item['active_arm_id'])
    if repeated_three_arm_mode:
        if (type(item['stage_id']) is not str or not re.fullmatch(r'[A-Za-z0-9_-]+', item['stage_id'])
                or (item['donor_arm_id'], item['recipient_arm_id']) not in (('arm_0', 'arm_1'), ('arm_1', 'arm_2'))
                or any(type(item[k]) is not str or not item[k] or item[k].strip()!=item[k]
                       for k in ('support_id', 'destination_support_id'))
                or item['support_id']==item['destination_support_id']):
            raise ContractError('Require a unique safe stage ID and an explicit sequential donor/recipient/support pair')
        sequence = (('donor_retreat', item['donor_arm_id']),
                    ('recipient_approach', item['recipient_arm_id']), ('recipient_carry', item['recipient_arm_id']))
        if index is not None and (not 0<=index<6 or item['donor_arm_id']!=f'arm_{index//3}'):
            raise ContractError('Repeated handoffs must progress arm_0 to arm_1 to arm_2')
    slot = index % 3 if repeated_three_arm_mode and index is not None else index
    if pair not in sequence or (slot is not None and pair != sequence[slot]):
        raise ContractError('Use donor retreat, optional recipient approach, then optional recipient carry')
    endpoint = item['endpoint']
    from transport_challenge.contracts import point
    from physics.carried_body import WorldPose
    if pair[0] == 'donor_retreat':
        if not isinstance(endpoint, dict) or set(endpoint) != {'kind', 'q_rad'} or endpoint['kind'] != 'joint_configuration':
            raise ContractError('Donor parking needs an explicit seven-joint goal, with no hidden home fallback')
        point(endpoint['q_rad'], 7)
    elif pair[0] == 'recipient_approach':
        if (not isinstance(endpoint, dict) or set(endpoint) != {'kind', 'position_body_m', 'quaternion_body_wxyz'}
                or endpoint['kind'] != 'body_relative_hand_pose'):
            raise ContractError('Recipient grasp needs an explicit body-relative hand endpoint')
        WorldPose(endpoint['position_body_m'], endpoint['quaternion_body_wxyz'])
    else:
        if (not isinstance(endpoint, dict) or set(endpoint) != {'kind', 'support_id', 'command_seating_depth_m'}
                or endpoint['kind'] not in (('fixed_goal_placement','fixed_support_placement')
                                           if repeated_three_arm_mode else ('fixed_goal_placement',))
                or type(endpoint['support_id']) is not str or not endpoint['support_id']
                or endpoint['support_id'].strip() != endpoint['support_id']
                or type(endpoint['command_seating_depth_m']) not in (int, float)
                or endpoint['command_seating_depth_m'] not in (0., .0003)):
            raise ContractError('Recipient carry needs the named fixed goal and explicit 0 or .0003 m seating')
        if repeated_three_arm_mode and endpoint['support_id']!=item['destination_support_id']:
            raise ContractError('Carry endpoint must name this stage destination support')


def validate_repeated_stage(item, world, scenario, index=None):
    """Bind one explicit stage to the fixed three-arm/two-relay prototype."""
    _phase_contract(item,index,repeated_three_arm_mode=True)
    if (not isinstance(world,dict) or not isinstance(scenario,dict)
            or [r['id'] for r in scenario['robots']]!=['arm_0','arm_1','arm_2']
            or scenario['active_arm_id']!='arm_0'
            or not isinstance(world.get('relay_supports'),list) or len(world['relay_supports'])!=2):
        raise ContractError('Repeated mode needs the exact three named arms and two fixed relays')
    support_ids=[s['id'] for s in scenario['support_stations']]
    route=[world['source_support']['id'],*[s['id'] for s in world['relay_supports']],world['goal_support']['id']]
    if len(set(route))!=4 or not set(route)<=set(support_ids):
        raise ContractError('Fixed source, both relays and goal must be distinct known supports')
    transfer=0 if item['donor_arm_id']=='arm_0' else 1
    if (item['support_id']!=route[transfer+1] or item['destination_support_id']!=route[transfer+2]
            or (item['phase']=='recipient_carry' and item['endpoint']['kind']!=
                ('fixed_support_placement' if transfer==0 else 'fixed_goal_placement'))):
        raise ContractError('Repeated stages must bind source to relay1 to relay2 to the fixed goal')
    return copy.deepcopy(item)


def validate_options(phases, planner, global_deadline_s, *, initial_deadline_s, dt_s,
                     repeated_three_arm_mode=False, world=None, scenario=None, initial_support_id=None):
    if type(repeated_three_arm_mode) is not bool:
        raise ContractError('Repeated three-arm mode must be an explicit Boolean')
    if phases is None and planner is None and global_deadline_s is None:
        if repeated_three_arm_mode:raise ContractError('Repeated mode requires its complete declared continuation')
        return None
    count_ok=isinstance(phases,list) and (len(phases)==6 if repeated_three_arm_mode else 1<=len(phases)<=3)
    if not callable(planner) or not count_ok:
        raise ContractError('Continuation requires a supplied planner and one to three explicit phase requests')
    for index, item in enumerate(phases):
        if repeated_three_arm_mode:validate_repeated_stage(item,world,scenario,index)
        else:_phase_contract(item, index)
    if repeated_three_arm_mode:
        if (len({p['stage_id'] for p in phases})!=6
                or initial_support_id!=world['relay_supports'][0]['id']):
            raise ContractError('Unique repeated stages must follow the actual relay1 release prefix')
    if (type(global_deadline_s) not in (int, float) or not math.isfinite(global_deadline_s)
            or global_deadline_s <= initial_deadline_s
            or abs(global_deadline_s / dt_s - round(global_deadline_s / dt_s)) > 1e-8):
        raise ContractError('Continuation needs one explicit larger global deadline on the dt grid')
    return copy.deepcopy(phases)


def resolved_hand_endpoint(phase_request, observation, *, world=None, scenario=None, held_body=None,
                           repeated_three_arm_mode=False):
    """Resolve only declared endpoints. Carry uses this boundary's measured grasp."""
    if repeated_three_arm_mode:validate_repeated_stage(phase_request,world,scenario)
    else:_phase_contract(phase_request)
    if phase_request['phase'] == 'donor_retreat':
        return None
    if phase_request['phase'] == 'recipient_carry':
        from physics.episode_held_phase_inputs import observed_held_spec, selected_goal
        if not all(isinstance(value, dict) for value in (world, scenario, held_body)):
            raise ContractError('Recipient carry requires its fixed world and actual held-body observation')
        kwargs={'active_arm_id':phase_request['active_arm_id'],'repeated_three_arm_mode':True} if repeated_three_arm_mode else {}
        spec = observed_held_spec(world, scenario, observation, held_body,**kwargs)
        goal_kwargs={'repeated_three_arm_mode':True} if repeated_three_arm_mode else {}
        return selected_goal(world, scenario, spec, phase_request['endpoint'],**goal_kwargs)['hand_world']
    from physics.carried_body import WorldPose, _multiply
    body = observation['body_world']; relative = phase_request['endpoint']
    world = WorldPose(body['position_m'], body['quaternion_wxyz'])
    return WorldPose(world.transform_point(relative['position_body_m']),
                     _multiply(world.quaternion_wxyz, relative['quaternion_body_wxyz'])).to_record()


def rest_receipt(dispatch, observation, arm_ids):
    if (dispatch.get('status') != 'SEGMENT_COMPLETED' or dispatch.get('pending_step') is not None
            or dispatch.get('last_observation_sha256') != digest(observation)
            or dispatch.get('global_step') != observation['native_scene_step']
            or dispatch.get('simulation_s') != observation['simulation_s']):
        raise ContractError('A fully committed current segment is required')
    return dict(schema_version='episode-continuation-rest-boundary-v1',
        previous_segment_sha256=dispatch['current_segment_sha256'], step=dispatch['global_step'],
        simulation_s=dispatch['simulation_s'], observation_sha256=digest(observation),
        verified_arm_ids=list(arm_ids), actual_state_verified=True,
        measured_rest_verified=True, source_bindings_verified=True)


def checked_segment(request, response, *, previous_plan, limits_by_arm, protocol, transport_protocol=None):
    """Wrap only the supplied child trajectory and explicit settle/grasp actions."""
    required = {'schema_version', 'request_sha256', 'phase', 'active_arm_id', 'plan',
                'endpoint_hand_pose', 'source_bindings_verified', 'physical_steps',
                'observed_start_valid', 'command_start_valid', 'observed_to_command_edge_valid',
                'cleanup_errors', 'durable_path_validation'}
    if not isinstance(response, dict) or set(response) != required:
        raise ContractError('Continuation child response requires the exact declared fields')
    repeated=request.get('repeated_three_arm_mode',False)
    if type(repeated) is not bool:raise ContractError('Repeated request flag must be Boolean')
    if repeated:validate_repeated_stage(request['phase_request'],request['world'],request['scenario'])
    else:_phase_contract(request['phase_request'])
    phase = request['phase_request']['phase']; active = request['phase_request']['active_arm_id']
    if (response['schema_version'] != 'episode-continuation-query-response-v1'
            or response['request_sha256'] != digest(request) or response['phase'] != phase
            or response['active_arm_id'] != active or type(response['physical_steps']) is not int
            or response['physical_steps'] != 0 or response['cleanup_errors'] != []):
        raise ContractError('Continuation child identity, clock or cleanup failed')
    for key in ('source_bindings_verified', 'observed_start_valid', 'command_start_valid',
                'observed_to_command_edge_valid'):
        if response[key] is not True:
            raise ContractError('Missing continuation evidence: ' + key)
    validation = response['durable_path_validation']
    if (not isinstance(validation, dict) or validation.get('passed') is not True
            or type(validation.get('sample_count')) is not int or validation['sample_count'] < 2
            or type(validation.get('trace_sha256')) is not str or len(validation['trace_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in validation['trace_sha256'])):
        raise ContractError('Fresh durable complete-path validation is required')
    saved = response['plan']; raw = saved['trajectory']; limits = limits_by_arm[active]
    trajectory = JointTrajectory(raw['waypoints'], raw['edge_durations_s'], limits)
    command = request['dispatcher']['last_command_by_arm'][active]
    if (trajectory.to_record() != raw or digest(raw) != saved['evidence']['trajectory_sha256']
            or list(trajectory.waypoints[0]) != command['q_rad']):
        raise ContractError('Continuation trajectory, limits or prior commanded endpoint changed')
    if phase in ('recipient_approach', 'recipient_carry'):
        from physics.carried_body import WorldPose
        endpoint = response['endpoint_hand_pose']
        if not isinstance(endpoint, dict) or endpoint.get('frame') != 'world':
            raise ContractError('Recipient needs an explicit checked world-frame hand endpoint')
        WorldPose(endpoint['position_m'], endpoint['quaternion_wxyz'])
        if phase == 'recipient_carry':
            gate = request.get('supported_grasp_gate')
            if (not isinstance(gate, dict) or gate.get('passed') is not True
                    or any(not isinstance(gate.get(key), dict) or gate[key].get('passed') is not True
                           for key in ('all_arms_rest', 'body_rest'))):
                raise ContractError('Recipient carry requires actual supported grasp and measured rest')
            expected = resolved_hand_endpoint(request['phase_request'], request['observation'],
                world=request.get('world'), scenario=request.get('scenario'), held_body=request.get('held_body'),
                repeated_three_arm_mode=repeated)
        else:
            kwargs={'world':request['world'],'scenario':request['scenario'],'repeated_three_arm_mode':True} if repeated else {}
            expected = resolved_hand_endpoint(request['phase_request'], request['observation'],**kwargs)
        if endpoint != expected:
            raise ContractError('Recipient endpoint changed from the declared measured-body goal')
    elif response['endpoint_hand_pose'] is not None:
        raise ContractError('A joint parking endpoint has no implicit hand target')
    elif list(trajectory.waypoints[-1]) != request['phase_request']['endpoint']['q_rad']:
        raise ContractError('Donor path changed the explicitly selected parking goal')
    if (saved['evidence']['request']['start'] != command['q_rad']
            or saved['evidence']['request']['goal'] != list(trajectory.waypoints[-1])):
        raise ContractError('Planned endpoints differ from the complete returned trajectory')
    record = previous_plan.to_record()
    record.update(episode_id=phase, initial_state_sha256=digest(request['observation']))
    for arm in record['arms']:
        prior = request['dispatcher']['last_command_by_arm'][arm['arm_id']]
        arm.update(initial_q_rad=prior['q_rad'], initial_finger_m=prior['finger_q_m'])
    actions = [dict(action_id=phase+'_motion', arm_id=active,
                    role={'donor_retreat':'retreat', 'recipient_approach':'recipient_approach',
                          'recipient_carry':'carry'}[phase],
                    kind='trajectory', trajectory=raw)]
    if phase == 'donor_retreat':
        actions.append(dict(action_id='donor_retreated_settle', arm_id=active,
                            role='released_settle', kind='hold', duration_s=1.))
    elif phase == 'recipient_approach':
        for suffix, role, kind, duration in (
                ('settle', 'approach_settle', 'hold', protocol['approach_settle_s']),
                ('close', 'close', 'close', protocol['close_s']),
                ('hold', 'final_hold', 'hold', protocol['final_hold_s'])):
            action = dict(action_id='recipient_'+suffix, arm_id=active, role=role, kind=kind, duration_s=duration)
            if kind == 'close': action['target_finger_m'] = [protocol['closed_finger_target_m']]*2
            actions.append(action)
    else:
        keys = ('placement_hold_s', 'release_s', 'released_settle_s')
        if (not isinstance(transport_protocol, dict) or any(
                type(transport_protocol.get(key)) not in (int, float)
                or not math.isfinite(transport_protocol[key]) or transport_protocol[key] <= 0
                for key in keys)):
            raise ContractError('Recipient carry requires explicit existing placement/release/settle durations')
        native_bounds = next(a.finger_bounds_m for a in previous_plan.arms if a.arm_id == active)
        open_target = [min(bound[1] for bound in native_bounds)]*2
        for suffix, role, kind, key in (
                ('place', 'place', 'hold', keys[0]),
                ('release', 'release', 'open', keys[1]),
                ('released_settle', 'released_settle', 'hold', keys[2])):
            action = dict(action_id='recipient_'+suffix, arm_id=active, role=role, kind=kind,
                          duration_s=transport_protocol[key])
            if kind == 'open': action['target_finger_m'] = open_target
            actions.append(action)
    if repeated:
        stage=request['phase_request']['stage_id']
        record['episode_id']=stage
        for action in actions:action['action_id']=stage+'__'+action['action_id']
    record['actions'] = actions
    dt = record['dt_s']
    record['deadline_s'] = (math.ceil(trajectory.duration_s/dt)
        + sum(round(a['duration_s']/dt) for a in actions if a['kind'] != 'trajectory'))*dt
    return EpisodePlan.from_record(record, limits_by_arm=limits_by_arm)
