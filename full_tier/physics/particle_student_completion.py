"""Explicit particle contracts for the existing student driver.

Detached predicates do not authenticate native truth. The runtime owns the
reader, every physical interval and the original source/INITIAL binding.
Empty rigid query contents mean particles were not collision queried.
"""
from __future__ import annotations

import copy
import math

from physics.carried_body import WorldPose
from physics.particle_episode_state import validate_particle_packet, window_rest, REST_LIMITS
from transport_challenge.contracts import ContractError, digest, point


def particle_context(world, scenario, observation, contract=None):
    """Check the opt-in, original native contract and full-N compact packet."""
    from physics.particle_episode_bridge import validate_request, bind_native_contract
    validate_request(world, scenario)
    native = world.get('native_particle_contract')
    if type(native) is not dict or (contract is not None and native != contract):
        raise ContractError('Original bound native particle contract required')
    rebound_world, rebound_scenario = bind_native_contract(world, scenario, native)
    if rebound_world != world or rebound_scenario != scenario or 'materials' in observation:
        raise ContractError('Particle observation cannot substitute an empty or rigid material list')
    validate_particle_packet(observation['particle_material'], native, observation, require_retained=True)
    if native['geometry_sha256'] != world['vessel_model']['geometry_sha256']:
        raise ContractError('Particle reader geometry differs from the actual vessel')
    return copy.deepcopy(native)


def checked_bulk_rest(rows, observation, contract):
    result = window_rest(rows, observation['simulation_s'], contract)
    if result['passed'] is not True:
        raise ContractError('Original full-inventory bulk rest failed: '+' | '.join(result['failures']))
    return result


def _recorded_bulk_rest(gate, ending_s):
    bulk=gate.get('particle_bulk_rest',{})
    if (bulk.get('schema_version')!='particle-episode-window-rest-v1'
            or bulk.get('passed') is not True or bulk.get('limits')!=REST_LIMITS
            or bulk.get('window_s')!=.5 or bulk.get('dt_s')!=.0025
            or type(bulk.get('expected_rows')) is not int or bulk['expected_rows']!=201
            or type(bulk.get('observed_rows')) is not int or bulk['observed_rows']!=201
            or bulk.get('ending_s')!=ending_s or bulk.get('failures')!=[]
            or any(not 0<=bulk['metrics'][key]<=limit for key,limit in REST_LIMITS.items() if key!='window_s')):
        raise ContractError('Original complete particle bulk-rest receipt missing or failed')
    return bulk


def source_lift_gate(state, *, arm, model, source_support, start_body, contract):
    """The private 30mm endpoint lift uses existing mission tolerances only."""
    from physics.episode_runtime import measured_rest, body_rest
    from physics.episode_transport_inputs import _bottom_corners
    from physics.episode_transport_assessment import _angle
    observation = state['observation']; execution = state['settings']['execution_limits']
    validate_particle_packet(observation['particle_material'], contract, observation, require_retained=True)
    rest = measured_rest(state['recent_rows'], observation['simulation_s'], execution, tuple(observation['arms']))
    body = body_rest(state['recent_rows'], observation['simulation_s'], execution['rest_window_s'], state['rest_policy'])
    bulk = window_rest(state['recent_rows'], observation['simulation_s'], contract)
    actual = WorldPose(observation['body_world']['position_m'], observation['body_world']['quaternion_wxyz'])
    target = list(point(start_body['position_m'], 3)); target[2] += .03
    error = math.dist(actual.position_m, target)
    orientation = _angle(actual.quaternion_wxyz, start_body['quaternion_wxyz'])
    top = source_support['center_world_m'][2]+source_support['size_m'][2]/2
    clearance = min(p[2] for p in _bottom_corners(model, actual))-top
    current = state['unsupported_bilateral_s']; maximum = state['maximum_unsupported_bilateral_by_arm_s'][arm]
    passed = bool(current >= .5 and maximum >= .5 and clearance >= .015
        and error <= execution['final_body_position_error_m']
        and orientation <= execution['final_body_orientation_error_rad']
        and rest['passed'] and body['passed'] and bulk['passed'])
    return dict(schema_version='particle-source-lift-probe-gate-v1', passed=passed,
        requested_lift_m=.03, target_body_position_m=target,
        measured_lift_m=actual.position_m[2]-start_body['position_m'][2], body_position_error_m=error,
        body_orientation_error_rad=orientation, minimum_bottom_clearance_m=clearance,
        unsupported_bilateral_s=current, load_bearing_interval_s=maximum,
        all_arms_rest=rest, body_rest=body, particle_bulk_rest=bulk,
        particle_material_contract_sha256=digest(contract), physical_handoff_verified=False,
        full_mission_passed=False, material_delivery_accepted=False,
        scope='One source approach and endpoint-planned 30mm unsupported lift in the retained scene. No release, relay, handoff or full mission.')


def assess_particle_student_single_batch(execution,observation,*,route,initial_supply,
                                      final_goal,currently_held,execution_limits):
    """Assemble one current task's completion from trusted measured evidence.

    This pure predicate does not authenticate fabricated records. The runtime
    owns each native interval, source binding and grasp/place predicate before
    this function is called. It cannot approve other materials, throughput,
    continuous collision safety, laboratory hardware, or student release.
    """
    from physics.student_episode_driver import SINGLE_BATCH_STAGES
    failures=[];carry_supports=[];load_by_arm={}
    try:
        if (execution.get('mode')!='mission' or execution.get('initialized') is not True
                or execution.get('operation_sequence_complete') is not True
                or execution.get('current_held_body') is not None or currently_held is not None
                or execution.get('all_requested_student_transport_motion_from_selected_strategy',
                    execution.get('all_motion_from_selected_strategy')) is not True
                or execution.get('reference_planner_fallback') is not False
                or execution.get('supplied_intermediate_postures') is not False):
            raise ContractError('Not a completed released student mission')
        if (not isinstance(route,(list,tuple)) or len(route)!=4
                or any(type(v) is not str or not v for v in route) or len(set(route))!=4
                or execution.get('support_route')!=list(route)
                or execution.get('current_support_id')!=route[-1]):
            raise ContractError('Source, two relays and final support are not bound')
        completed=execution.get('completed',[]);plans=execution.get('planned_motions',[])
        gates=execution.get('provided_gates',[]);pauses=execution.get('planning_pauses',[])
        if any(len(rows)!=8 for rows in (completed,plans,gates,pauses)):
            raise ContractError('Completion requires eight operations, plans, gates and pauses')
        signatures=[(r.get('stage_id'),r.get('kind'),r.get('active_arm_id')) for r in completed]
        if tuple(signatures)!=SINGLE_BATCH_STAGES:
            raise ContractError('Current eight-stage order or arm identities differ')
        for rows in (plans,gates):
            if [(r.get('stage_id'),r.get('kind'),r.get('active_arm_id')) for r in rows]!=signatures:
                raise ContractError('Plan or gate identity differs from completed stages')
        if [r.get('stage_id') for r in pauses]!=[s[0] for s in SINGLE_BATCH_STAGES]:
            raise ContractError('Planning pause identities differ')
        def valid_sha(v):
            return type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)
        for plan,pause in zip(plans,pauses):
            if (plan.get('supplied_intermediate_postures') is not False
                    or not all(valid_sha(plan.get(k)) for k in
                               ('request_sha256','trajectory_sha256','plan_path_source_sha256'))
                    or not all(pause.get(k) is True for k in
                               ('native_state_and_clock_unchanged','dispatcher_unchanged','source_bindings_verified'))):
                raise ContractError('A source-bound student plan or unchanged planning pause is missing')
        if len({r['plan_path_source_sha256'] for r in plans})!=1:
            raise ContractError('Selected strategy source changes between motions')
        contract=initial_supply
        if execution.get('particle_material_contract_sha256')!=digest(contract):
            raise ContractError('Execution does not retain the original particle contract')
        validate_particle_packet(observation['particle_material'],contract,observation,require_retained=True)
        if 'materials' in observation:raise ContractError('Explicit particle observation must not contain rigid material rows')
        previous_step=0;previous_time=0.;last_gate=None
        for index,(done,entry) in enumerate(zip(completed,gates)):
            if (type(done.get('native_scene_step')) is not int or done['native_scene_step']<=previous_step
                    or not math.isfinite(done['simulation_s']) or done['simulation_s']<=previous_time):
                raise ContractError('Completion clocks are not strictly increasing')
            previous_step=done['native_scene_step'];previous_time=done['simulation_s']
            gate=entry.get('gate',{})
            if gate.get('passed') is not True:
                raise ContractError('A provided stage gate did not pass')
            if done['kind'] in ('approach','carry'):
                _recorded_bulk_rest(gate,done['simulation_s'])
            arm=done['active_arm_id'];arm_index=int(arm[-1]);kind=done['kind']
            expected_support=route[arm_index+1] if kind in ('carry','retreat') else route[arm_index]
            if entry.get('support_id')!=expected_support:
                raise ContractError('A stage is bound to the wrong physical support')
            if kind=='approach':
                if (gate.get('source_support_observed') is not True
                        or gate.get('all_arms_rest',{}).get('passed') is not True
                        or gate.get('body_rest',{}).get('passed') is not True
                        or any(gate.get(k,0.)<=0 for k in
                               ('bilateral_window_s','stable_contents_window_s','source_support_window_s'))):
                    raise ContractError('Recipient/source supported grasp evidence is incomplete')
            if kind=='carry':
                if (gate.get('schema_version')!='sampled-released-particle-relay-gate-v1'
                        or gate.get('support_id')!=expected_support or gate.get('active_arm_id')!=arm
                        or gate.get('failures')!=[] or gate.get('particle_material_contract_sha256')!=digest(contract)
                        or gate.get('all_arms_rest',{}).get('passed') is not True
                        or gate.get('body_rest',{}).get('passed') is not True
                        or type(gate.get('expected_rows')) is not int or gate['expected_rows']<1
                        or gate.get('observed_rows')!=gate['expected_rows']
                        or not math.isfinite(gate['window_s']) or gate['window_s']<=0
                        or not math.isfinite(gate['dt_s']) or gate['dt_s']<=0
                        or abs(gate['expected_rows']*gate['dt_s']-gate['window_s'])>1e-9
                        or abs(gate['ending_s']-done['simulation_s'])>1e-9
                        or not math.isfinite(gate.get('load_bearing_interval_s',-1.))
                        or gate['load_bearing_interval_s']<gate['window_s']):
                    raise ContractError('A released carrying stage lacks stable support, retention or load bearing')
                carry_supports.append(expected_support);load_by_arm[arm]=gate['load_bearing_interval_s'];last_gate=gate
        if carry_supports!=list(route[1:]) or len(load_by_arm)!=3:
            raise ContractError('Both relay placements and the final placement are required')
        if (observation.get('native_scene_step')!=previous_step
                or abs(observation['simulation_s']-previous_time)>1e-9
                or abs(previous_step*last_gate['dt_s']-previous_time)>1e-9):
            raise ContractError('Final observation does not belong to the final completed release')
        _recorded_bulk_rest(last_gate,observation['simulation_s'])
        for pause in pauses:
            comparison=pause.get('particle_pause_comparison',{})
            if (comparison.get('passed') is not True or comparison.get('schema_version')!='particle-episode-paused-state-comparison-v1'
                    or not all(valid_sha(comparison.get(k)) for k in
                        ('projection_sha256','pause_receipt_sha256','committed_observation_sha256'))
                    or comparison['committed_observation_sha256']!=pause.get('measured_observation_sha256')
                    or comparison.get('extension_boundary_receipt',{}).get('observation_sha256')!=pause.get('measured_observation_sha256')):
                raise ContractError('Exact full-native particle planning pause evidence missing')
        arms=observation['arms']
        if set(arms)!={'arm_0','arm_1','arm_2'}:
            raise ContractError('Final named arm observations are incomplete')
        for arm in arms.values():
            point(arm['q_all_dofs'],9);v=point(arm['qd_all_dofs'],9)
            if (max(map(abs,v[:7]))>execution_limits['rest_speed_rad_s']
                    or max(map(abs,v[7:]))>.001):
                raise ContractError('Final arm/finger rest failed')
        body=observation['body_world'];nominal=final_goal.get('seating',{}).get('nominal_goal',final_goal)
        if final_goal.get('support_id')!=route[-1] or nominal.get('support_id')!=route[-1]:
            raise ContractError('Final body goal is not the fixed destination support')
        position=math.dist(point(body['position_m'],3),point(nominal['body_world']['position_m'],3))
        actual_q=point(body['quaternion_wxyz'],4);wanted_q=point(nominal['body_world']['quaternion_wxyz'],4)
        norms=math.hypot(*actual_q)*math.hypot(*wanted_q)
        if norms<=0:raise ContractError('Final orientation is undefined')
        angle=2*math.acos(min(1.,abs(sum(a*b for a,b in zip(actual_q,wanted_q)))/norms))
        body_limits=last_gate['body_rest']['limits']
        if (position>last_gate['limits']['body_position_m'] or angle>last_gate['limits']['body_orientation_rad']
                or math.hypot(*point(observation['body_linear_velocity_m_s'],3))>body_limits['maximum_body_linear_speed_m_s']
                or math.hypot(*point(observation['body_angular_velocity_rad_s'],3))>body_limits['maximum_body_angular_speed_rad_s']):
            raise ContractError('Final destination/body rest failed')
    except (ContractError,KeyError,TypeError,ValueError,OverflowError) as exc:
        failures.append(str(exc))
    result={'schema_version':'current-student-particle-single-batch-completion-v1','passed':not failures,
        'completion_scope':'current_student_three_arm_particle_single_batch_v1','failures':failures,
        'expected_stage_ids':[s[0] for s in SINGLE_BATCH_STAGES],'carry_support_ids':carry_supports,
        'load_bearing_interval_by_arm_s':load_by_arm,'physical_handoff_count':2 if not failures else 0,
        'final_observation_sha256':None if failures else digest(observation),'student_release_approved':False,
        'scope':'One current endpoint-planned three-arm particle batch. Trusted runtime must separately verify every native interval and unchanged sources. No repeated series, hardware, throughput or release approval.'}
    return result

