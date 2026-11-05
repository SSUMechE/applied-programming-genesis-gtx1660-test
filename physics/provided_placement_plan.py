"""Instructor-owned, bounded Z seating after an unchanged student carry path.

Fresh measured grasp composition and an isolated zero-step native query are
used. No student planner, cached route, live pose setter, weld or contact
exemption is used. A validated local plan is not physical placement success.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from physics.carried_body import HeldBodySpec, WorldPose
from transport_challenge.contracts import ContractError, digest, point
from transport_challenge.joint_planning import parameterize_joint_path


MAX_ATTEMPTS=2
MAX_TOTAL_TRANSLATION_M=.001
MAX_ROTATION_RAD=.005
MAX_JOINT_CORRECTION_RAD=.02
MIN_TRANSLATION_M=.000001
HOLDER_WINDOW_ROWS=40
LOCAL_MAX_Q_DELTA_RAD=.001
MIN_LOCAL_INTERVALS=8
MAX_PLANNED_XY_DRIFT_M=.0001


def _angle(a,b):
    a,b=point(a,4),point(b,4)
    return 2*math.acos(min(1.,abs(math.fsum(x*y for x,y in zip(a,b)))/(math.hypot(*a)*math.hypot(*b))))


def pose_displacements(hand,body,anchor):
    """Maximum-displacement quantities. These are not accumulated arc lengths."""
    return {name+suffix:value for name,pose in (('hand',hand),('body',body))
        for suffix,value in (('_translation_m',math.dist(point(pose['position_m'],3),point(anchor[name+'_world']['position_m'],3))),
            ('_rotation_rad',_angle(pose['quaternion_wxyz'],anchor[name+'_world']['quaternion_wxyz'])))}


def displacement_failures(values):
    return [key for key,value in values.items() if value>(MAX_ROTATION_RAD if key.endswith('_rotation_rad') else MAX_TOTAL_TRANSLATION_M)]


def validate_dense_envelope(path,*,anchor,local_origin,remaining_translation_m,goal,arm):
    """Re-use the durable same-query native poses at every fresh q sample."""
    maxima={key:0. for key in pose_displacements(local_origin['hand_world'],local_origin['body_world'],anchor)}
    local_max=dict(maxima);xy_max={'hand':0.,'body':0.};count=0
    low,high=sorted((local_origin['body_world']['position_m'][2],goal['body_world']['position_m'][2]))
    for line in Path(path).open(encoding='utf-8'):
        row=json.loads(line);witness=row['same_query'];poses=witness['same_query_proxy_poses']
        if (row.get('valid') is not True or witness.get('forbidden_pairs')!=[]
                or witness.get('bilateral_by_arm',{}).get(arm) is not True):
            raise ContractError('Dense provided placement lacks its same-query qualified grasp witness')
        hand,body=poses['hand_world'],poses['body_world']
        values=pose_displacements(hand,body,anchor);local=pose_displacements(hand,body,local_origin)
        if displacement_failures(values):raise ContractError('Dense planned hand/body motion exceeds the first-correction anchor envelope')
        if max(local['hand_translation_m'],local['body_translation_m'])>remaining_translation_m:
            raise ContractError('Dense planned motion exceeds the remaining commanded correction budget')
        for name,pose in (('hand',hand),('body',body)):
            drift=math.dist(pose['position_m'][:2],local_origin[name+'_world']['position_m'][:2]);xy_max[name]=max(xy_max[name],drift)
            if drift>MAX_PLANNED_XY_DRIFT_M:raise ContractError('Dense local Z placement path has excessive planned XY drift')
        if not low-.00005<=body['position_m'][2]<=high+.00005:
            raise ContractError('Dense local Z placement path overshoots its terminal interval')
        for key in maxima:maxima[key]=max(maxima[key],values[key]);local_max[key]=max(local_max[key],local[key])
        count+=1
    if count<MIN_LOCAL_INTERVALS+1:raise ContractError('Dense provided placement trace is incomplete')
    return {'passed':True,'sample_count':count,'maximum_displacement_from_first_anchor':maxima,
        'maximum_displacement_from_current_boundary':local_max,'maximum_planned_xy_drift_m':xy_max,
        'maximum_planned_xy_drift_limit_m':MAX_PLANNED_XY_DRIFT_M,
        'scope':'Fresh sampled FK/native proxy envelope. Not an arc-length or continuous-time certificate.'}


def refresh_measured_held(world,scenario,model,observation,recent_rows,previous_held,initial_supply,arm):
    """Require one retained native body, continuous qualified holder and four IDs.

    This is a current held-frame observation, not a new supported acquisition
    or a reset of the native handling guard. The caller owns native truth.
    """
    if 'particle_material_request' in world:
        from physics.particle_student_completion import particle_context
        from physics.particle_episode_assessment import refresh_particle_measured_held
        particle_context(world,scenario,observation,initial_supply)
        return refresh_particle_measured_held(world,scenario,model,observation,recent_rows,
            previous_held,initial_supply,arm)
    from physics.episode_held_phase_inputs import observed_held_spec
    from physics.episode_runtime import empty_payload_mode
    empty = empty_payload_mode(world, scenario)
    if (type(previous_held) is not HeldBodySpec or previous_held.object_id!=world['object_id']
            or previous_held.model.geometry_sha256!=model.geometry_sha256
            or set(observation['arms'])!={'arm_0','arm_1','arm_2'} or arm not in observation['arms']):
        raise ContractError('Measured seating requires the same currently held body and exact three arms')
    supply={m['id']:(m['mass_kg'],m['entity_idx']) for m in initial_supply}
    automatic_empty=empty and previous_held.grasp_model=='provided_ideal'
    if empty:
        if initial_supply!=[] or observation.get('materials')!=[] or 'particle_material' in observation:
            raise ContractError('Empty held refresh requires no original or observed content inventory')
    elif len(supply)!=4 or len(initial_supply)!=4:raise ContractError('Keep the original four native material IDs')
    selected=list(recent_rows)[-HOLDER_WINDOW_ROWS:]
    end=observation['native_scene_step']
    if (len(selected)!=HOLDER_WINDOW_ROWS or selected[-1]['native_scene_step']!=end
            or abs(observation['simulation_s']-end*.0025)>1e-9):
        raise ContractError('Current held-frame refresh requires a complete native interval window')
    for index,row in enumerate(selected,end-HOLDER_WINDOW_ROWS+1):
        interval=row.get('handling_interval',{})
        if (row.get('step')!=index or row.get('native_scene_step')!=index
                or abs(row['simulation_s']-index*.0025)>1e-9 or row.get('post_processing_complete') is not True
                or row.get('failures')!=[] or row.get('coverage',{}).get('contacts_complete') is not True
                or interval.get('passed') is not True or interval.get('qualified_holder_id')!=arm
                or interval.get('first_failure') is not None
                or (not automatic_empty and row.get('bilateral_by_arm',{}).get(arm) is not True)):
            raise ContractError('Measured seating cannot inherit a missing, switched or unqualified holder')
        if automatic_empty:
            binding=row.get('provided_grasp')
            if (type(binding) is not dict or binding.get('holder_arm_id')!=arm
                    or binding.get('model')!='provided_ideal'
                    or digest(binding.get('attach_receipt'))!=previous_held.attachment_receipt_sha256
                    or interval.get('automatic_grasp_post',{}).get('holder_arm_id')!=arm):
                raise ContractError('Empty held refresh lacks the actual continuous automatic attachment')
        materials=row['materials']
        if (len(materials)!=(0 if empty else 4) or {m['id']:(m['mass_kg'],m['entity_idx']) for m in materials}!=supply
                or any(m.get('inside_carrier') is not True for m in materials)):
            raise ContractError('Measured seating cannot replace, refill or lose a native material')
    actual={m['id']:(m['mass_kg'],m['entity_idx']) for m in observation['materials']}
    if len(observation['materials'])!=(0 if empty else 4) or actual!=supply or any(m.get('inside_carrier') is not True for m in observation['materials']):
        raise ContractError('Current native material identity or retention differs')
    hand=observation['arms'][arm]['hand_world'];body=observation['body_world']
    if math.dist(point(hand['position_m'],3),point(body['position_m'],3))>.20:
        raise ContractError('Current held root leaves the existing native grasp reach bound')
    spec=HeldBodySpec(world['object_id'],model,WorldPose(hand['position_m'],hand['quaternion_wxyz']),
        WorldPose(body['position_m'],body['quaternion_wxyz']),tuple(point(observation['arms'][arm]['q_all_dofs'],9)[7:]),
        observation['simulation_s'],f"episode_step_{end}",
        **({'grasp_model':'provided_ideal','attachment_receipt_sha256':previous_held.attachment_receipt_sha256} if automatic_empty else {}))
    return observed_held_spec(world,scenario,observation,spec.to_record(),active_arm_id=arm,repeated_three_arm_mode=True)


def measured_z_goal(world,scenario,spec,observation,original_goal,*,used_translation_m,attempt,used_rotation_rad=0.):
    """Preserve measured XY within the original goal tolerance, upright at Z."""
    from physics.provided_manipulation import placement_endpoint
    if (type(attempt) is not int or not 1<=attempt<=MAX_ATTEMPTS
            or type(used_translation_m) not in (int,float) or not math.isfinite(used_translation_m)
            or not 0<=used_translation_m<MAX_TOTAL_TRANSLATION_M
            or type(used_rotation_rad) not in (int,float) or not math.isfinite(used_rotation_rad)
            or not 0<=used_rotation_rad<MAX_ROTATION_RAD):
        raise ContractError('Provided seating attempt or cumulative translation limit exhausted')
    nominal=original_goal.get('seating',{}).get('nominal_goal',original_goal)
    body=observation['body_world'];target=nominal['body_world']
    execution=world['physics_settings']['execution_limits']
    xy=point(body['position_m'],3)[:2]
    if (math.dist(body['position_m'],target['position_m'])>execution['final_body_position_error_m']
            or _angle(body['quaternion_wxyz'],target['quaternion_wxyz'])>execution['final_body_orientation_error_rad']):
        raise ContractError('Provided place may not repair a missing student destination')
    selected=WorldPose((*xy,target['position_m'][2]),target['quaternion_wxyz'])
    goal=placement_endpoint(world,scenario,spec,original_goal['support_id'],selected_body_pose=selected)
    hand=spec.measured_hand_world.to_record()
    distance=math.dist(goal['hand_world']['position_m'],hand['position_m'])
    rotation=_angle(goal['hand_world']['quaternion_wxyz'],hand['quaternion_wxyz'])
    if (not MIN_TRANSLATION_M<=distance<=MAX_TOTAL_TRANSLATION_M-used_translation_m
            or rotation+used_rotation_rad>MAX_ROTATION_RAD):
        raise ContractError('No necessary bounded sub-mm Z seating correction is available')
    return goal,{'requested_translation_m':distance,'requested_rotation_rad':rotation,
        'used_translation_before_m':used_translation_m,'maximum_total_translation_m':MAX_TOTAL_TRANSLATION_M,
        'maximum_rotation_rad':MAX_ROTATION_RAD,'attempt':attempt,'maximum_attempts':MAX_ATTEMPTS,
        'used_rotation_before_rad':used_rotation_rad,
        'body_xy_preserved_world_m':list(xy),'original_nominal_goal_sha256':digest(nominal)}


def plan_measured_seating(*,world,scenario,model,observation,recent_rows,previous_held,initial_supply,
        arm,command_start_q,original_goal,limits,speed_scale,attempt,used_translation_m,output,
        source_check,wall_check,fixture_readback='batched',endpoint_attempts=8,seed=71,used_rotation_rad=0.,anchor=None):
    """Plan just one local teacher place segment. Always write a new receipt."""
    from physics.provided_manipulation import solve_placement_endpoint
    from physics.student_motion_planning import _query_scene,_durable_validation
    path=Path(output);path.mkdir(parents=True,exist_ok=False)
    query=None;failure=None;trajectory=None
    receipt={'schema_version':'provided-measured-placement-plan-v1','status':'NOT_VALIDATED',
        'instructor_owned_place_only':True,'student_route_replaced':False,'supplied_student_intermediate_postures':False,
        'observation_sha256':digest(observation),'native_scene_step':observation['native_scene_step'],
        'simulation_s':observation['simulation_s'],'active_arm_id':arm,'support_id':original_goal['support_id'],
        'world_sha256':digest(world),'original_goal_sha256':digest(original_goal),'attempt':attempt,
        'physical_steps':0,'physical_execution_verified':False,'cleanup_errors':[],
        'provider_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def guard():
        wall_check()
        if source_check() is not True:raise ContractError('Sources changed during provided local seating')
    try:
        guard()
        identity_path=path.parent/'handling_enforcement_initial.json'
        initial_guard=json.loads(identity_path.read_text(encoding='utf-8'))['guard']
        identity=initial_guard['body_identity']
        if (identity.get('id')!=world['object_id'] or identity.get('model_sha256')!=model.geometry_sha256
                or type(identity.get('entity_idx')) is not int or identity['entity_idx']<0):
            raise ContractError('Provided seating requires the unchanged native body identity')
        receipt.update(native_body_identity=copy.deepcopy(identity),
            initial_handling_record_sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest())
        spec=refresh_measured_held(world,scenario,model,observation,recent_rows,previous_held,initial_supply,arm)
        local_origin={'hand_world':copy.deepcopy(observation['arms'][arm]['hand_world']),
            'body_world':copy.deepcopy(observation['body_world'])}
        anchor=copy.deepcopy(local_origin if anchor is None else anchor)
        if displacement_failures(pose_displacements(local_origin['hand_world'],local_origin['body_world'],anchor)):
            raise ContractError('Current placement boundary already exceeds the first-correction envelope')
        goal,bounds=measured_z_goal(world,scenario,spec,observation,original_goal,
            used_translation_m=used_translation_m,attempt=attempt,used_rotation_rad=used_rotation_rad)
        receipt.update(bounds,held_body=spec.to_record(),held_body_sha256=spec.binding_sha256,placement_goal=goal,
            first_correction_anchor=anchor,current_boundary_pose=local_origin)
        request={'schema_version':'student-motion-request-v1','stage_id':f'{arm}__provided_seat_{attempt}',
            'kind':'carry','world':copy.deepcopy(world),'scenario':copy.deepcopy(scenario),
            'observation':copy.deepcopy(observation),'active_arm_id':arm,'command_start_q_rad':list(point(command_start_q,7)),
            'goal':{'kind':'world_hand_pose','position_m':goal['hand_world']['position_m'],
                'quaternion_wxyz':goal['hand_world']['quaternion_wxyz']},'held_body':spec.to_record(),
            'planning':{'seed':seed,'max_queries':1,'speed_scale':speed_scale,'endpoint_attempts':endpoint_attempts,
                'options':{},'use_edge_timing_hook':False,'query_body_tilt_limit_deg':7.5}}
        receipt['query_request_sha256']=digest(request)
        (path/'provided_placement_query_request.json').write_text(json.dumps({'request':request,
            'request_sha256':digest(request),'instructor_owned_place_only':True},indent=2,allow_nan=False)+'\n',encoding='utf-8')
        query,_=_query_scene(request,path)
        if fixture_readback=='batched':receipt['fixture_readback']=query.enable_batched_fixture_readback()
        elif fixture_readback!='scalar':raise ContractError('Unknown provided query readback')
        guard();observed=tuple(point(observation['arms'][arm]['q_all_dofs'][:7],7));start=tuple(point(command_start_q,7))
        if max(abs(a-b) for a,b in zip(observed,start))>world['physics_settings']['execution_limits']['final_joint_error_rad']:
            raise ContractError('Current native q does not match the preceding command boundary')
        checks={'observed_start_valid':query.state_valid(observed),'command_start_valid':query.state_valid(start)}
        checks['observed_to_command_edge_valid']=all(checks.values()) and query.edge_valid(observed,start)
        if not all(checks.values()):raise ContractError('Provided seating start/command bridge rejected')
        endpoint=solve_placement_endpoint(query,world,scenario,spec,original_goal['support_id'],start,
            attempts=endpoint_attempts,seed=seed,wall_check=guard,
            selected_body_pose=WorldPose((*bounds['body_xy_preserved_world_m'],
                original_goal.get('seating',{}).get('nominal_goal',original_goal)['body_world']['position_m'][2]),
                original_goal.get('seating',{}).get('nominal_goal',original_goal)['body_world']['quaternion_wxyz']))
        if endpoint['status']!='ENDPOINT_FOUND' or endpoint['placement_goal']!=goal:
            raise ContractError('No fresh exact native terminal seating endpoint')
        end=tuple(point(endpoint['q_rad'],7))
        if max(abs(a-b) for a,b in zip(start,end))>MAX_JOINT_CORRECTION_RAD:
            raise ContractError('Provided local seating cannot change IK branch or repair a global route')
        command_hand=query.fk(start);end_hand=query.fk(end)
        commanded_distance=math.dist(command_hand['position_m'],end_hand['position_m'])
        commanded_rotation=_angle(command_hand['quaternion_wxyz'],end_hand['quaternion_wxyz'])
        used=max(bounds['requested_translation_m'],commanded_distance)
        rotation_used=max(bounds['requested_rotation_rad'],commanded_rotation)
        if (not MIN_TRANSLATION_M<=commanded_distance or used+used_translation_m>MAX_TOTAL_TRANSLATION_M
                or rotation_used+used_rotation_rad>MAX_ROTATION_RAD):
            raise ContractError('Actual commanded local correction exceeds the cumulative motion bound')
        guard();checks['terminal_state_valid']=query.state_valid(end);guard()
        checks['local_edge_valid']=checks['terminal_state_valid'] and query.edge_valid(start,end);guard()
        if not checks['local_edge_valid']:raise ContractError('Fresh full-model local seating edge rejected')
        trajectory=parameterize_joint_path((start,end),limits,speed_scale=speed_scale)
        local_case=copy.deepcopy(scenario['case'])
        resolution=min(LOCAL_MAX_Q_DELTA_RAD,max(abs(a-b) for a,b in zip(start,end))/MIN_LOCAL_INTERVALS)
        if resolution<=0:raise ContractError('Provided placement has no nonzero bounded joint correction')
        local_case['planning']['edge_max_joint_delta_rad']=resolution
        validation=_durable_validation(query,trajectory,local_case,path,guard)
        if validation.get('passed') is not True:raise ContractError('Fresh durable provided seating validation failed')
        validation.update(edge_max_joint_delta_rad=resolution,provided_local_dense=True)
        dense=validate_dense_envelope(path/'path_validation.jsonl',anchor=anchor,local_origin=local_origin,
            remaining_translation_m=MAX_TOTAL_TRANSLATION_M-used_translation_m,goal=goal,arm=arm)
        used=max(used,dense['maximum_displacement_from_current_boundary']['hand_translation_m'],
            dense['maximum_displacement_from_current_boundary']['body_translation_m'])
        rotation_used=max(rotation_used,dense['maximum_displacement_from_current_boundary']['hand_rotation_rad'],
            dense['maximum_displacement_from_current_boundary']['body_rotation_rad'])
        if used+used_translation_m>MAX_TOTAL_TRANSLATION_M or rotation_used+used_rotation_rad>MAX_ROTATION_RAD:
            raise ContractError('Dense provided correction exceeds its cumulative commanded budget')
        if query.scene.t!=0 or float(query.scene.cur_t)!=0.:raise ContractError('Provided query advanced native physics')
        receipt.update(status='PROVIDED_LOCAL_PLACEMENT_PLAN_VALIDATED',checks=checks,endpoint=endpoint,
            trajectory=trajectory.to_record(),trajectory_sha256=digest(trajectory.to_record()),
            durable_path_validation=validation,dense_task_space_envelope=dense,commanded_translation_m=commanded_distance,
            charged_translation_m=used,cumulative_translation_m=used_translation_m+used,
            commanded_rotation_rad=commanded_rotation,charged_rotation_rad=rotation_used,
            cumulative_rotation_rad=used_rotation_rad+rotation_used,student_planner_invocations=0,
            budget_semantics='Cumulative commanded correction magnitudes and first-anchor maximum displacements are separate. Neither is accumulated trajectory arc length.',
            cached_paths_loaded=False,reference_planner_fallback=False)
    except Exception as exc:
        failure=exc;receipt.update(status='PROVIDED_LOCAL_PLACEMENT_PLAN_FAILED',failure={'type':type(exc).__name__,'message':str(exc)})
    finally:
        if query is not None:
            try:receipt['query_evidence']=query.evidence()
            except Exception as exc:receipt['cleanup_errors'].append('query evidence: '+repr(exc))
            try:query.close()
            except Exception as exc:receipt['cleanup_errors'].append('query close: '+repr(exc))
        try:guard()
        except Exception as exc:receipt['cleanup_errors'].append('final source/wall check: '+repr(exc))
        if receipt['cleanup_errors']:receipt['status']='PROVIDED_LOCAL_PLACEMENT_PLAN_FAILED'
        (path/'provided_placement_plan.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if failure is not None or receipt['cleanup_errors']:
        raise ContractError('Provided local seating failed closed: '+json.dumps(receipt.get('failure',receipt['cleanup_errors']))) from failure
    return trajectory,receipt
