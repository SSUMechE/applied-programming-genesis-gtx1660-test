"""Explicit ideal-grasp driver branch. Trusted runtime owns native authenticity.

Friction acquisition/retention uses the explicit native provided grasp model.
After actual supported detach, the existing supplied open action clears fingers.
Native sources, actual seven-joint paths, all-ID material quality/rest, sampled
contact coverage, named support and actual released-state checks remain required.
"""
from __future__ import annotations
import copy
import math
from pathlib import Path
from transport_challenge.contracts import ContractError,digest,point
from physics.carried_body import HeldBodySpec,WorldPose
from physics.provided_manipulation import assess_place_ready

HELPER_SHA='7458d4f339da20d91d819f981cc8e30c3b018dd5cce375e9d61f32eb55e3c291'
RELEASED_EXTRA_HOLD_S=.5
RELEASED_EXTRA_LIMIT_S=5.


def require(value,message):
    if not value:raise ContractError(message)


def policy():
    return dict(schema_version='provided-ideal-manipulation-policy-v1',grasp_model='provided_ideal',
        acquisition='Actual supplied endpoint then native hand/cup weld without pose/velocity reset.',
        release='Actual named support, arm/body rest and all-ID quality, same-clock detach, supplied open, released hold and strict final material rest.',
        released_material_settle=dict(hold_s=RELEASED_EXTRA_HOLD_S,maximum_extra_s=RELEASED_EXTRA_LIMIT_S,all_simulation_time_counted=True,final_rest_limits_unchanged=True),
        fingers='Preserve commands except the owned post-detach place_actions open to the common native maximum. No close or preload.',
        native_collision_rule='Native weld excludes entire hand-link/cup-link pair only. Other links, fingers, world and materials retain collisions.',
        student_intermediate_postures=False,legacy_friction_holder_certification=False)


def validate_observation(driver,obs,arm=None):
    helper=driver.automatic_grasp;require(helper is not None,'Actual automatic grasp helper not bound')
    holder=helper.holder;binding=obs.get('provided_grasp')
    if holder is None:
        require(binding is None and driver.automatic_attachment_receipt is None,'Released observation retains an automatic holder')
        return None
    require(holder in driver.arm_ids and (arm is None or holder==arm),'Actual automatic holder differs')
    snapshot=helper.snapshot(holder);receipt=driver.automatic_attachment_receipt
    expected=dict(schema_version='provided-automatic-grasp-binding-v1',model='provided_ideal',holder_arm_id=holder,
        helper_source_sha256=HELPER_SHA,attach_receipt=receipt,current_snapshot=snapshot)
    require(binding==expected and receipt is not None,'Current trusted automatic binding differs from native helper')
    require(snapshot['native_clock'][:2]==[obs['native_scene_step'],obs['simulation_s']]
        and snapshot['body_world']==obs['body_world'] and snapshot['hand_world']==obs['arms'][holder]['hand_world']
        and snapshot['body_linear_velocity_m_s']==obs['body_linear_velocity_m_s']
        and snapshot['body_angular_velocity_rad_s']==obs['body_angular_velocity_rad_s'],'Automatic native pose/clock/velocity differs from observation')
    require(snapshot['inventory']==receipt['after']['inventory'],'Actual automatic weld inventory changed')
    return snapshot


def transition_observation(driver,state,previous,receipt):
    """Reobserve an actual native equality transaction. Never rewrite a trace row."""
    callback=state.get('record_provided_grasp_event')
    require(callable(callback),'Actual native boundary transaction callback missing')
    fresh=callback(receipt);a=copy.deepcopy(previous);b=copy.deepcopy(fresh)
    a.pop('provided_grasp',None);b.pop('provided_grasp',None)
    require(a==b and state['source_check']() is True,'Automatic transaction changed physical state, clock or sources')
    state['observation']=fresh
    validate_observation(driver,fresh)
    return fresh


def empty_material_receipt(driver,state):
    """Authenticate absence only. There is no material RMS/COM/rest to pass."""
    from physics.episode_runtime import empty_payload_mode
    obs=state['observation']
    require(empty_payload_mode(driver.world,driver.scenario) and not driver.particle_mode
        and driver.initial_supply==[] and obs.get('materials')==[] and 'particle_material' not in obs,
        'Explicit empty payload has unexpected material entities or packet')
    step=obs['native_scene_step'];now=obs['simulation_s']
    require(type(step) is int and step>=0 and type(now) in (int,float) and math.isfinite(now)
        and abs(now-step*.0025)<=1e-9 and state['source_check']() is True,
        'Empty payload receipt requires current native clock and unchanged sources')
    return dict(schema_version='empty-payload-presence-v1',payload_mode='empty',
        status='NOT_APPLICABLE',material_count=0,material_rest='NOT_APPLICABLE',
        quality_passed=True,rest_assessed=False,native_scene_step=step,simulation_s=now,
        observation_sha256=digest(obs),source_bindings_verified=True)


def _recorded_empty_presence(receipt,step,now):
    """Validate absence, never convert absence into a material-rest PASS."""
    require(type(receipt) is dict and receipt.get('schema_version')=='empty-payload-presence-v1'
        and receipt.get('payload_mode')=='empty' and receipt.get('status')=='NOT_APPLICABLE'
        and type(receipt.get('material_count')) is int and receipt['material_count']==0
        and receipt.get('material_rest')=='NOT_APPLICABLE' and receipt.get('rest_assessed') is False
        and receipt.get('source_bindings_verified') is True and receipt.get('quality_passed') is True
        and receipt.get('native_scene_step')==step and receipt.get('simulation_s')==now,
        'Explicit current empty payload absence receipt missing')
    value=receipt.get('observation_sha256')
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),
        'Empty payload observation source hash missing')


def bulk_rest(driver,state):
    obs=state['observation']
    if driver.world.get('payload_mode')=='empty':return empty_material_receipt(driver,state)
    if driver.particle_mode:
        from physics.particle_student_completion import particle_context,checked_bulk_rest
        particle_context(driver.world,driver.scenario,obs,driver.initial_supply)
        return checked_bulk_rest(state['recent_rows'],obs,driver.initial_supply)
    from physics.episode_transport_assessment import bead_material_rest_gate
    result=bead_material_rest_gate(state['recent_rows'],obs['simulation_s'],driver.initial_supply,driver.model)
    require(result['passed'],'Original all-ID bead bulk rest failed: '+str(result))
    return result


def material_quality(driver,state):
    """Full current native inventory/retention quality, without a rest verdict."""
    obs=state['observation']
    if driver.world.get('payload_mode')=='empty':return empty_material_receipt(driver,state)
    if driver.particle_mode:
        from physics.particle_student_completion import particle_context
        particle_context(driver.world,driver.scenario,obs,driver.initial_supply)
    else:
        from physics.episode_transport_assessment import _materials,_bead_material_sample
        row=state['recent_rows'][-1]
        require(row['native_scene_step']==obs['native_scene_step'] and row['simulation_s']==obs['simulation_s']
            and row['materials']==obs['materials'] and row['body_world']==obs['body_world'],'Quality row is not this actual endpoint')
        _bead_material_sample(row,_materials(driver.initial_supply),driver.model.geometry_record())
    return dict(schema_version='provided-automatic-material-quality-v1',quality_passed=True,retained=True,
        native_scene_step=obs['native_scene_step'],simulation_s=obs['simulation_s'],observation_sha256=digest(obs),rest_assessed=False)


def material_rest_status(driver,state):
    """Distinguish invalid data from a valid, still moving material window."""
    obs=state['observation'];material_quality(driver,state)
    if driver.world.get('payload_mode')=='empty':return empty_material_receipt(driver,state)
    if driver.particle_mode:
        from physics.particle_episode_state import validate_particle_packet,window_rest
        rows=[r for r in state['recent_rows'] if obs['simulation_s']-.5-1e-10<=r['simulation_s']<=obs['simulation_s']+1e-10]
        require(len(rows)==201,'Incomplete original particle rest window is not a settling retry')
        for row in rows:validate_particle_packet(row['particle_material'],driver.initial_supply,row,require_retained=True)
        result=window_rest(rows,obs['simulation_s'],driver.initial_supply)
    else:
        from physics.episode_transport_assessment import bead_material_rest_gate
        result=bead_material_rest_gate(state['recent_rows'],obs['simulation_s'],driver.initial_supply,driver.model)
        require(result['quality_passed'] is True,'Invalid bead data cannot request a settling hold')
    return result


def rested(driver,state,*,material_rest_required=True):
    from physics.episode_runtime import measured_rest,body_rest
    obs=state['observation'];execution=state['settings']['execution_limits']
    arms=measured_rest(state['recent_rows'],obs['simulation_s'],execution,driver.arm_ids)
    body=body_rest(state['recent_rows'],obs['simulation_s'],execution['rest_window_s'],state['rest_policy'])
    material=bulk_rest(driver,state) if material_rest_required else material_quality(driver,state)
    require(arms['passed'] and body['passed'] and state['source_check']() is True,'Actual automatic boundary lacks rest or unchanged sources')
    return arms,body,material


def held_spec(driver,obs,arm):
    validate_observation(driver,obs,arm)
    h,b=obs['arms'][arm]['hand_world'],obs['body_world']
    return HeldBodySpec(driver.world['object_id'],driver.model,WorldPose(h['position_m'],h['quaternion_wxyz']),
        WorldPose(b['position_m'],b['quaternion_wxyz']),tuple(point(obs['arms'][arm]['q_all_dofs'],9)[7:]),
        obs['simulation_s'],f"episode_step_{obs['native_scene_step']}",grasp_model='provided_ideal',
        attachment_receipt_sha256=digest(driver.automatic_attachment_receipt))


def pick(driver,state,op):
    obs=state['observation'];arm=op['active_arm_id'];arms,body,material=rested(driver,state)
    support=op['support_id'];goal=dict(support_id=support,body_world=driver.scenario['supported_body_poses'][support])
    ready=assess_place_ready(obs,state['recent_rows'],state['settings'],state['rest_policy'],goal)
    require(ready['passed'],'Automatic acquisition lacks actual named support/body goal')
    stable=0. if state['stable_since'] is None else obs['simulation_s']-state['stable_since']
    supported=0. if state['support_since'] is None else obs['simulation_s']-state['support_since']
    empty=driver.world.get('payload_mode')=='empty'
    require((empty or stable>=state['settings']['scenario']['contained_stability_s']-1e-9)
        and supported>=state['settings']['scenario']['bilateral_window_s']-1e-9,'Original material/support stability duration incomplete')
    require(driver.automatic_grasp.holder is None and driver.held is None,'Automatic acquisition already held')
    receipt=driver.automatic_grasp.attach(arm);driver.automatic_attachment_receipt=copy.deepcopy(receipt)
    fresh=transition_observation(driver,state,obs,receipt);driver.held=held_spec(driver,fresh,arm);driver.held_observation_sha256=digest(fresh)
    event=dict(event='attach',stage_id=op['stage_id'],native_scene_step=fresh['native_scene_step'],simulation_s=fresh['simulation_s'],receipt=receipt)
    driver.automatic_events.append(event)
    driver._save(state,'provided_automatic_grasp_events.json',driver.automatic_events)
    gate=dict(schema_version='provided-automatic-supported-pick-gate-v1',passed=True,grasp_model='provided_ideal',
        all_arms_rest=arms,body_rest=body,source_support_observed=True,support_id=support,support_ready=ready,
        automatic_attachment_receipt_sha256=digest(receipt),actual_attach_receipt=receipt,observation_sha256=digest(fresh),
        stable_contents_window_s=None if empty else stable,source_support_window_s=supported,
        **({'empty_payload':material,'material_window_required':False} if empty else
            {'particle_bulk_rest':material} if driver.particle_mode else {'bead_bulk_rest':material}))
    driver._save(state,op['stage_id']+'__provided_pick.json',dict(gate=gate,held_body=driver.held.to_record(),observation_sha256=digest(fresh)))
    return gate


def lift(driver,state,op):
    from physics.episode_transport_inputs import _bottom_corners
    from physics.episode_transport_assessment import _angle
    obs=state['observation'];arm=op['active_arm_id'];arms,body,material=rested(driver,state)
    validate_observation(driver,obs,arm);target=copy.deepcopy(driver.probe_lift_start_body)
    height=.055 if op.get('bead_lift_unit') else .03 if op.get('particle_source_lift') else .005;target['position_m'][2]+=height
    pose=WorldPose(obs['body_world']['position_m'],obs['body_world']['quaternion_wxyz'])
    support=next(s for s in driver.scenario['support_stations'] if s['id']==op['support_id']);top=support['center_world_m'][2]+support['size_m'][2]/2
    clearance=min(p[2]-top for p in _bottom_corners(driver.model,pose));execution=state['settings']['execution_limits']
    load=state['maximum_unsupported_provided_grasp_by_arm_s'][arm];current=state['unsupported_provided_grasp_s']
    position=math.dist(pose.position_m,target['position_m']);angle=_angle(pose.quaternion_wxyz,target['quaternion_wxyz'])
    passed=bool(current>=.5 and load>=.5 and clearance>=(.015 if height>=.03 else .003)
        and position<=execution['final_body_position_error_m'] and angle<=execution['final_body_orientation_error_rad'])
    if height==.005:passed=bool(passed and abs(pose.position_m[2]-driver.probe_lift_start_body['position_m'][2]-.005)<=.0005)
    if op.get('bead_lift_unit'):
        require(driver.mode=='bead_lift_unit' and not driver.particle_mode
            and driver.world.get('rigid_runtime_profile')=='rigid16_bead_tc2h_v1','Private bead lift opt-in differs')
        passed=bool(passed and pose.position_m[2]-driver.probe_lift_start_body['position_m'][2]>=.05)
    gate=dict(schema_version='provided-automatic-source-lift-probe-gate-v1',passed=passed,grasp_model='provided_ideal',
        requested_lift_m=height,measured_lift_m=pose.position_m[2]-driver.probe_lift_start_body['position_m'][2],
        target_body_position_m=target['position_m'],body_position_error_m=position,body_orientation_error_rad=angle,
        minimum_bottom_clearance_m=clearance,unsupported_provided_grasp_s=current,load_bearing_interval_s=load,
        all_arms_rest=arms,body_rest=body,automatic_attachment_receipt_sha256=digest(driver.automatic_attachment_receipt),
        **({'particle_bulk_rest':material,'particle_material_contract_sha256':digest(driver.initial_supply)} if driver.particle_mode else {'bead_bulk_rest':material}))
    require(passed,'Automatic native source lift did not satisfy original displacement/rest/load bounds')
    driver.held=held_spec(driver,obs,arm);driver.held_observation_sha256=digest(obs)
    driver._save(state,op['stage_id']+'__provided_lift.json',dict(gate=gate,held_body=driver.held.to_record(),observation_sha256=digest(obs)))
    return gate


def _post_detach_release_actions(arm,finger_bounds,commands,*,prefix):
    """Keep the supplied physical wait when its native open command is exact.

    This compares preceding commands, never a nearby measured finger pose.
    Only an unchanged open becomes hold. Its ID, role and duration stay fixed.
    """
    from physics.provided_manipulation import place_actions
    actions=place_actions(arm,finger_bounds,prefix=prefix)[1:]
    if point(commands[arm]['finger_q_m'],2)==point(actions[0]['target_finger_m'],2):
        actions[0]={k:copy.deepcopy(v) for k,v in actions[0].items() if k!='target_finger_m'}
        actions[0]['kind']='hold'
    return actions


def advance_place(driver,dispatcher,state):
    if driver.placement_phase in ('automatic_released_tail','automatic_material_wait'):
        return advance_released_wait(driver,dispatcher,state)
    if driver.placement_phase!='automatic_ready':return False
    from physics.episode_runtime import TRANSPORT_PROTOCOL
    from physics.episode_continuation_protocol import rest_receipt
    op=driver.current;arm=op['active_arm_id'];obs=state['observation'];arms,body,material=rested(driver,state,material_rest_required=False)
    snapshot=validate_observation(driver,obs,arm)
    ready=assess_place_ready(obs,state['recent_rows'],state['settings'],state['rest_policy'],driver.goal)
    require(ready['passed'],'Automatic nominal placement lacks actual support/body rest; no bilateral seating fallback')
    evidence=dict(ready=ready,all_arms_rest=arms,material_quality=material,final_material_rest_not_yet_assessed=True,observation_sha256=digest(obs),native_snapshot=snapshot)
    def support_ready(actual):
        require(actual==snapshot and state['source_check']() is True,'Support callback differs from actual same-clock native snapshot')
        return dict(passed=True,native_clock=actual['native_clock'],body_entity_idx=actual['body_entity_idx'],
            support_id=driver.goal['support_id'],evidence_sha256=digest(evidence))
    receipt=driver.automatic_grasp.release(arm,support_ready=support_ready);driver.automatic_attachment_receipt=None
    fresh=transition_observation(driver,state,obs,receipt);driver.held=None
    event=dict(event='release',stage_id=op['stage_id'],native_scene_step=fresh['native_scene_step'],simulation_s=fresh['simulation_s'],receipt=receipt,support_gate=evidence)
    driver.automatic_events.append(event);driver._save(state,'provided_automatic_grasp_events.json',driver.automatic_events)
    commands=dispatcher.record()['last_command_by_arm']
    actions=_post_detach_release_actions(arm,driver.finger_bounds[arm],commands,
        prefix=op['stage_id']+'__automatic_provided_place')
    plan=driver._plan(fresh,commands,actions,actions[0]['action_id']);boundary=rest_receipt(dispatcher.record(),fresh,list(driver.arm_ids))
    driver._automatic_release_open=dict(stage_id=op['stage_id'],arm_id=arm,action=copy.deepcopy(actions[0]),
        plan_sha256=plan.sha256,commands=copy.deepcopy(commands),start_step=fresh['native_scene_step'],
        release_receipt_sha256=digest(receipt),authorized=False)
    dispatcher.extend(plan,model_sha256_by_arm={n:driver.world['robot_model_sha256'] for n in driver.arm_ids},boundary=boundary)
    driver.placement_phase='automatic_released_tail';driver._save(state,plan.episode_id+'__episode.json',plan.to_record())
    return True


def check_boundary(driver,request,state):
    """Authorize only the supplied, owned post-detach open within this carry."""
    validate_observation(driver,state['observation'])
    kind=request['action_kind']
    require(kind not in ('close','supported_preload'),'Automatic manipulation cannot close or preload')
    own=getattr(driver,'_automatic_release_open',None)
    unchanged_hold=(kind=='hold' and (request.get('action_role')=='release'
        or (type(own) is dict and request.get('action_id')==own['action']['action_id'])))
    if kind=='open' or unchanged_hold:
        op=driver.current;obs=state['observation']
        require(type(own) is dict and own['authorized'] is False and op is not None and op['kind']=='carry'
            and not op.get('probe_lift') and own['stage_id']==op['stage_id']
            and driver.placement_phase=='automatic_released_tail' and driver.held is None
            and driver.automatic_grasp.holder is None,'Open is not the owned actual released carry action')
        require(request['action_id']==own['action']['action_id'] and kind==own['action']['kind'] and request['action_role']=='release'
            and request['active_arm_id']==own['arm_id']==op['active_arm_id']
            and request['plan_sha256']==own['plan_sha256'] and request['at_step']==own['start_step']==obs['native_scene_step']
            and request['measured_state_sha256']==digest(obs) and request['measured_rest_required'] is (kind=='open'),
            'Open action/plan/actor/current boundary differs from the supplied owner')
        expected={r['arm_id']:dict(q_rad=r['q_rad'],finger_q_m=r['finger_q_m']) for r in request['expected_arm_commands']}
        require(len(request['expected_arm_commands'])==len(driver.arm_ids) and expected==own['commands']
            and state['source_check']() is True,'Supplied open changed seven-q/peer command anchors or sources')
        supplied=_post_detach_release_actions(own['arm_id'],driver.finger_bounds[own['arm_id']],
            own['commands'],prefix=op['stage_id']+'__automatic_provided_place')[0]
        require(supplied==own['action'],'Owned open differs from the unchanged provided native-bounds action')
        release=next((e for e in reversed(driver.automatic_events) if e['event']=='release' and e['stage_id']==op['stage_id']),None)
        require(release is not None and digest(release['receipt'])==own['release_receipt_sha256'],'Owned native detach receipt differs')
        require(assess_place_ready(obs,state['recent_rows'],state['settings'],state['rest_policy'],driver.goal)['passed'],
            'Supplied open lacks current actual named support/body readiness')
        if unchanged_hold:
            arms,body,_=rested(driver,state,material_rest_required=False)
            require(arms['passed'] and body['passed'],'Unchanged-open hold still requires actual original arm/body rest')
        material_quality(driver,state);own['authorized']=True
        driver._save(state,op['stage_id']+'__supplied_release_open.json',own)
    return {'provided_policy':policy()['schema_version'],'action_role':request['action_role'],
        'owned_post_detach_open':kind=='open' or unchanged_hold}


def observe_owned_open(driver,state):
    row=state['recent_rows'][-1];command=row['command']
    own=getattr(driver,'_automatic_release_open',None)
    unchanged_hold=(command['kind']=='hold' and (command.get('action_role')=='release'
        or (type(own) is dict and command.get('action_id')==own['action']['action_id'])))
    if command['kind']!='open' and not unchanged_hold:return
    require(type(own) is dict and own['authorized'] is True and command['kind']==own['action']['kind'] and command['action_id']==own['action']['action_id']
        and command['active_arm_id']==own['arm_id'] and command['action_role']=='release'
        and driver.held is None and driver.automatic_grasp.holder is None,'Native open interval lacks its owned released action')
    require(len(command['arms'])==len(driver.arm_ids) and {c['arm_id'] for c in command['arms']}==set(driver.arm_ids),'Open must preserve all arm commands')
    for c in command['arms']:
        anchor=own['commands'][c['arm_id']]
        require(c['q_rad']==anchor['q_rad'] and not any(c['qd_rad_s']) and not any(c['qdd_rad_s2']),
            'Supplied opening changed a seven-q command')
        if c['arm_id']!=own['arm_id']:
            require(c['finger_q_m']==anchor['finger_q_m'] and not any(c['finger_qd_m_s']),'Supplied opening changed peer fingers')
        elif unchanged_hold:
            require(c['finger_q_m']==anchor['finger_q_m'] and not any(c['finger_qd_m_s']),
                'Unchanged-open hold must preserve the exact active finger command')


def released_finger_targets(driver,state):
    """Use the current real native command targets, including earlier opened arms."""
    obs=state['observation'];receipt=state['provided_native_control_targets']
    require(type(receipt) is dict and receipt.get('step')==obs['native_scene_step']
        and type(receipt.get('simulation_s')) in (int,float) and math.isfinite(receipt['simulation_s'])
        and abs(receipt['simulation_s']-obs['simulation_s'])<=1e-9 and receipt.get('exact_native_dtype_target_match') is True
        and receipt.get('schedule_phase')=='hold:released_settle' and set(receipt.get('by_arm',{}))==set(driver.arm_ids),
        'Released fingers need this committed native target receipt')
    result={}
    for arm,record in receipt['by_arm'].items():
        actual=point(record['native_ctrl_pos'],9);scheduled=point(record['scheduled_ctrl_pos'],9)
        velocity=point(record['native_ctrl_vel'],9)
        require(actual==scheduled and velocity==point(record['scheduled_ctrl_vel'],9)
            and not any(velocity) and record['native_ctrl_mode']==[0]*9,'Released native POSITION target/velocity/mode differs')
        result[arm]=actual[7:]
    return result


def released_gate(driver,state,op,*,material_rest_required=True):
    """Original complete released window, with actual no-weld and constant fingers.

    Supplied opening precedes this original released window. No observation,
    classified contact, material packet or legacy assessment result is rewritten.
    """
    from physics.episode_runtime import measured_rest,body_rest,DT,TRANSPORT_PROTOCOL,PROTOCOL
    from physics.episode_transport_assessment import _coverage,_materials,_angle
    from physics.handle_contacts import contents_contact_witness_valid
    obs=state['observation'];now=obs['simulation_s'];window=TRANSPORT_PROTOCOL['acceptance_window_s'];end=round(now/DT);count=round(window/DT)
    rows=[r for r in state['recent_rows'] if now-window+1e-10<r['simulation_s']<=now+1e-10]
    require(len(rows)==count and window==.5,'Incomplete original automatic released window')
    require(driver.held is None and driver.automatic_grasp.holder is None,'Actual automatic cup not released')
    nominal=driver.goal.get('seating',{}).get('nominal_goal',driver.goal);execution=state['settings']['execution_limits'];policy_ids=set()
    empty=driver.world.get('payload_mode')=='empty'
    if empty:empty_material_receipt(driver,state)
    supply=None if driver.particle_mode or empty else _materials(driver.initial_supply)
    finger_targets=released_finger_targets(driver,state)
    for row,step in zip(rows,range(end-count+1,end+1)):
        require(row['step']==row['native_scene_step']==step and abs(row['simulation_s']-step*DT)<=1e-9
            and row['post_processing_complete'] is True and row['failures']==[] and row['phase']=='released_settle','Failed/incomplete/nonconsecutive automatic released interval')
        pre=row['pre'];require(pre['native_scene_step']==step-1 and abs(pre['simulation_s']-(step-1)*DT)<=1e-9,'Wrong automatic interval pre-state')
        contacts=row['classified_contacts'];require(type(contacts) is list,'Complete classified contacts required')
        policy_ids.add(driver._particle_runtime[1](row,step,DT,contacts) if driver.particle_mode else
            _coverage(row,step,DT,contacts,required_support_id=nominal['support_id']))
        supported=False
        for i,c in enumerate(contacts):
            require(c['raw_row']==i and c['geom_ids']==[row['raw_contacts']['geom_a'][i],row['raw_contacts']['geom_b'][i]],'Raw/classified contact order changed')
            require(c['category'] in (('named_tray_bottom_support_top','named_support_floor') if empty else
                ('named_tray_bottom_support_top','contents_contact','named_support_floor')),'Unallowed released contact')
            valid=contents_contact_witness_valid(c['penetration_m'],c['normal_unit_error'],c.get('owners')) if c['category']=='contents_contact' and c['penetration_m']<0 else (math.isfinite(c['penetration_m']) and 0<=c['penetration_m']<=.001 and math.isfinite(c['normal_unit_error']) and 0<=c['normal_unit_error']<=.001)
            require(valid,'Original contact accuracy guard failed')
            supported |= c['category']=='named_tray_bottom_support_top' and c.get('support_station_id')==nominal['support_id']
        require(supported,'Named actual support missing from released interval')
        for value in (pre,row):
            require(value.get('provided_grasp') is None and set(value['arms'])==set(driver.arm_ids),'Released interval has attachment or missing arms')
            for arm,a in value['arms'].items():
                q=point(a['q_all_dofs'],9);v=point(a['qd_all_dofs'],9)
                require(max(map(abs,v[:7]))<=execution['rest_speed_rad_s'] and max(map(abs,v[7:]))<=PROTOCOL['finger_rest_speed_m_s'],'Actual released arm/finger not at rest')
                require(max(abs(q[7+i]-finger_targets[arm][i]) for i in range(2))<=state['settings']['closure_match_tolerances']['finger_m'],'Automatic current finger command/readback differs')
            actual=value['body_world'];require(math.dist(point(actual['position_m'],3),point(nominal['body_world']['position_m'],3))<=execution['final_body_position_error_m']
                and _angle(point(actual['quaternion_wxyz'],4),point(nominal['body_world']['quaternion_wxyz'],4))<=execution['final_body_orientation_error_rad'],'Actual released body outside goal')
            require(math.hypot(*point(value['body_linear_velocity_m_s'],3))<=state['rest_policy']['maximum_body_linear_speed_m_s']
                and math.hypot(*point(value['body_angular_velocity_rad_s'],3))<=state['rest_policy']['maximum_body_angular_speed_rad_s'],'Actual released body not at rest')
            if empty:
                require(value.get('materials')==[] and 'particle_material' not in value,'Empty released interval acquired contents')
            elif driver.particle_mode:
                from physics.particle_episode_state import validate_particle_packet,REST_LIMITS
                packet=validate_particle_packet(value['particle_material'],driver.initial_supply,value,require_retained=True)
                if material_rest_required:
                    require(packet['bulk']['relative_RMS_m_s']<=REST_LIMITS['relative_RMS_max_m_s'] and packet['bulk']['relative_q95_m_s']<=REST_LIMITS['relative_q95_max_m_s'],'Particle interval rest failed')
            else:
                current=_materials(value['materials']);require(set(current)==set(supply),'Original bead IDs changed')
                for name,m in current.items():
                    speed=math.hypot(*point(m['velocity_world_m_s'],3))
                    require((m['mass_kg'],m['entity_idx'])==(supply[name]['mass_kg'],supply[name]['entity_idx']) and m['inside_carrier'] is True
                        and math.isfinite(speed) and (not material_rest_required or speed<=state['settings']['scenario']['stable_speed_m_s']),'Original bead mass/entity/retention/rest changed')
                    point(m['position_world_m'],3)
    require(len(policy_ids)==1 and state['source_check']() is True,'Released contact/source policy changed')
    arms=measured_rest(rows,now,{**execution,'rest_window_s':window},driver.arm_ids);body=body_rest(rows,now,window,state['rest_policy']);material=bulk_rest(driver,state) if material_rest_required else material_quality(driver,state)
    load=state['maximum_unsupported_provided_grasp_by_arm_s'][op['active_arm_id']]
    require(arms['passed'] and body['passed'] and (driver.mode not in ('mission','one_carry','first_handoff') or load>=window),'Released rest/actual provided load interval failed')
    return dict(schema_version='provided-automatic-released-relay-gate-v1' if material_rest_required else 'provided-automatic-released-quality-gate-v1',passed=True,failures=[],grasp_model='provided_ideal',
        ending_s=now,window_s=window,dt_s=DT,expected_rows=count,observed_rows=len(rows),support_id=nominal['support_id'],active_arm_id=op['active_arm_id'],
        all_arms_rest=arms,body_rest=body,load_bearing_interval_s=load,coverage_policy_sha256=next(iter(policy_ids)),
        limits=dict(body_position_m=execution['final_body_position_error_m'],body_orientation_rad=execution['final_body_orientation_error_rad'],content_speed_m_s=state['settings']['scenario']['stable_speed_m_s']),
        actual_release_receipt_sha256=digest(next(e['receipt'] for e in reversed(driver.automatic_events) if e['event']=='release' and e['stage_id']==op['stage_id'])),
        **({'empty_payload':material,'material_window_required':False} if empty else
            ({'particle_bulk_rest':material,'particle_material_contract_sha256':digest(driver.initial_supply)} if driver.particle_mode else {'bead_bulk_rest':material,'persistent_material_ids':sorted(supply)}) if material_rest_required else {'material_quality':material,'final_rest_not_assessed':True}))


def observe_released_wait(driver,state):
    """Every extra interval retains the released cup and identical native commands."""
    if driver.placement_phase!='automatic_material_wait':return
    wait=driver._automatic_material_wait;obs=state['observation'];row=state['recent_rows'][-1]
    require(driver.current is not None and wait['stage_id']==driver.current['stage_id'] and obs['native_scene_step']==wait['last_step']+1
        and row['native_scene_step']==obs['native_scene_step'] and row['phase']=='released_settle','Extra released hold skipped an actual native interval')
    validate_observation(driver,obs);require(driver.held is None and driver.automatic_grasp.holder is None,'Extra released hold acquired a holder')
    command=row['command'];require(command['kind']=='hold' and command['action_role']=='released_settle'
        and {c['arm_id'] for c in command['arms']}==set(driver.arm_ids) and len(command['arms'])==len(driver.arm_ids),'Extra rest must be an all-arm constant hold')
    for c in command['arms']:
        anchor=wait['commands'][c['arm_id']]
        require(c['q_rad']==anchor['q_rad'] and c['finger_q_m']==anchor['finger_q_m']
            and not any(c['qd_rad_s']) and not any(c['qdd_rad_s2']) and not any(c['finger_qd_m_s']),'Extra rest changed seven-q/fingers or issued motion')
    rested(driver,state,material_rest_required=False)
    require(assess_place_ready(obs,state['recent_rows'],state['settings'],state['rest_policy'],driver.goal)['passed'],'Extra rest lost actual support/body goal')
    wait['last_step']=obs['native_scene_step']


def advance_released_wait(driver,dispatcher,state):
    from physics.episode_continuation_protocol import rest_receipt
    obs=state['observation'];now=obs['simulation_s'];op=driver.current
    require(op is not None and op['kind']=='carry' and not op.get('probe_lift'),'Extra settling needs an actual completed placement')
    state['wall_check']();require(now<=driver.simulation_deadline_s+1e-9,'Extra settling cannot exceed the simulation deadline')
    hard=released_gate(driver,state,op,material_rest_required=False);bulk=material_rest_status(driver,state)
    empty=driver.world.get('payload_mode')=='empty'
    if empty:_recorded_empty_presence(bulk,obs['native_scene_step'],now)
    accepted=empty or bulk['passed'] is True
    wait=getattr(driver,'_automatic_material_wait',None);elapsed=0. if wait is None else now-wait['start_s']
    event=dict(stage_id=op['stage_id'],native_scene_step=obs['native_scene_step'],simulation_s=now,
        original_released_quality_gate=hard,material_bulk_rest=bulk,additional_simulation_s=elapsed,
        extra_hold_s=0.,accepted=accepted,timeout=False,observation_sha256=digest(obs))
    events=getattr(driver,'automatic_settle_events',[]);events.append(event);driver.automatic_settle_events=events
    driver._save(state,'provided_automatic_material_settle.json',events)
    if accepted:return False
    if elapsed+RELEASED_EXTRA_HOLD_S>RELEASED_EXTRA_LIMIT_S+1e-9 or now+RELEASED_EXTRA_HOLD_S>driver.simulation_deadline_s+1e-9:
        event['timeout']=True;driver._save(state,'provided_automatic_material_settle.json',events)
        raise ContractError('Released material did not reach unchanged final rest within5s/remaining deadline')
    commands=dispatcher.record()['last_command_by_arm']
    if wait is None:
        wait=dict(stage_id=op['stage_id'],start_s=now,start_step=obs['native_scene_step'],last_step=obs['native_scene_step'],segments=0,commands=copy.deepcopy(commands))
        driver._automatic_material_wait=wait
    require(wait['stage_id']==op['stage_id'] and wait['commands']==commands and wait['last_step']==obs['native_scene_step'],'Extra released hold changed its command boundary or skipped guards')
    wait['segments']+=1
    action=dict(action_id=op['stage_id']+f'__automatic_material_wait_{wait["segments"]:02d}',arm_id=op['active_arm_id'],role='released_settle',kind='hold',duration_s=RELEASED_EXTRA_HOLD_S)
    plan=driver._plan(obs,commands,[action],action['action_id']);boundary=rest_receipt(dispatcher.record(),obs,list(driver.arm_ids))
    dispatcher.extend(plan,model_sha256_by_arm={n:driver.world['robot_model_sha256'] for n in driver.arm_ids},boundary=boundary)
    driver.placement_phase='automatic_material_wait';event.update(extra_hold_s=RELEASED_EXTRA_HOLD_S,episode_sha256=digest(plan.to_record()),commands_sha256=digest(commands))
    driver._save(state,plan.episode_id+'__episode.json',plan.to_record());driver._save(state,'provided_automatic_material_settle.json',events)
    return True


def finish_current(driver,state):
    if driver.current is None:return
    op=driver.current;arm=op['active_arm_id'];kind=op['kind']
    if kind=='approach':gate=pick(driver,state,op)
    elif kind=='carry' and op.get('probe_lift'):gate=lift(driver,state,op)
    elif kind=='carry':gate=released_gate(driver,state,op)
    else:
        from physics.episode_runtime import measured_rest
        gate=measured_rest(state['recent_rows'],state['observation']['simulation_s'],state['settings']['execution_limits'],driver.arm_ids)
        require(gate['passed'],'Student return endpoint has not settled')
    obs=state['observation'];driver._save(state,op['stage_id']+'__automatic_gate.json',gate)
    if kind=='approach' and driver.mode in ('first_handoff','mission') and arm=='arm_1':
        driver._capture('recipient_regrasp',state,dict(support_id=op['support_id'],gate_passed=True,
            held_body=driver.held.to_record(),held_binding_sha256=driver.held.binding_sha256,
            automatic_attachment_receipt_sha256=gate['automatic_attachment_receipt_sha256'],
            grasp_policy='automatic',grasp_model='provided_ideal',scope='Actual recipient native attachment after the automatic support/rest gate.'))
    elif kind=='carry' and not op.get('probe_lift') and driver.mode in ('first_handoff','mission'):
        checkpoint='relay_release' if driver.mode=='first_handoff' else 'mission_'+arm+('_goal_release' if arm=='arm_2' else '_relay_release')
        driver._capture(checkpoint,state,dict(support_id=driver.goal['support_id'],gate_passed=True,
            load_bearing_interval_s=gate['load_bearing_interval_s'],placement_goal=driver.goal,
            actual_release_receipt_sha256=gate['actual_release_receipt_sha256'],
            grasp_policy='automatic',grasp_model='provided_ideal',scope='Actual released placement after the automatic support/material-rest gate.'))
    if driver.mode in ('placement_probe','bead_lift_unit') and kind=='carry' and getattr(driver,'image_sink',None) is not None:
        label='unit_lifted' if op.get('probe_lift') else 'unit_released'
        driver._capture(label,state,dict(gate_passed=True,support_id=op['support_id'] if op.get('probe_lift') else driver.goal['support_id'],
            actual_gate=copy.deepcopy(gate),grasp_policy='automatic',grasp_model='provided_ideal',
            scope='Actual placement-unit observation after the unchanged lift or final released-rest gate.'))
    driver.gates.append(dict(stage_id=op['stage_id'],kind=kind,active_arm_id=arm,support_id=driver.goal['support_id'] if kind=='carry' and not op.get('probe_lift') else op['support_id'],gate=gate))
    driver.completed.append(dict(stage_id=op['stage_id'],kind=kind,active_arm_id=arm,native_scene_step=obs['native_scene_step'],simulation_s=obs['simulation_s']))
    driver.current=None;driver.placement_phase=None;driver._bead_settle_wait=None;driver._automatic_material_wait=None;driver._automatic_release_open=None


def record(driver,record):
    record.update(schema_version='student-three-arm-automatic-driver-v1',grasp_policy='automatic',grasp_model='provided_ideal',
        provided_manipulation_policy=policy(),provided_control=dict(native_target_rows=driver.provided_native_target_rows,
            finger_commands_preserved_except_owned_release_open=True,owned_release_open='Existing place_actions open after actual supported native detach.',
            friction_close_or_preload=False),automatic_grasp_events=copy.deepcopy(driver.automatic_events))
    record.update(provided_placement_policy='automatic_nominal_support_then_native_release',legacy_measured_seating_executed=False)
    record.pop('legacy_all_motion_scope',None);record.pop('provided_placement_actual_maximum_displacements',None)
    record['provided_automatic_material_settle']=dict(hold_s=RELEASED_EXTRA_HOLD_S,maximum_extra_s=RELEASED_EXTRA_LIMIT_S,
        events=copy.deepcopy(getattr(driver,'automatic_settle_events',[])),all_extra_simulation_time_counted=True,final_rest_limits_unchanged=True)
    return record


def final_result(driver,observation):
    require(driver.done and len(driver.completed)==len(driver.queue),'Automatic student sequence incomplete')
    require(driver.provided_native_target_rows==observation['native_scene_step'],'Actual native target rows incomplete')
    if driver.mode=='one_carry':
        from physics.one_carry_planning import final_result as one_carry_result
        return one_carry_result(driver,observation)
    execution=driver.record();completion=None;passed=False;empty=driver.world.get('payload_mode')=='empty'
    if driver.mode=='mission':
        completion=assess_single_batch(driver,observation,execution);require(completion['passed'],'Automatic three-arm completion failed: '+str(completion));passed=True
    elif driver.mode in ('first_handoff','source_lift_probe'):
        arm='arm_1' if driver.mode=='first_handoff' else 'arm_0'
        validate_observation(driver,observation,arm)
        require(driver.held is not None and driver.held_observation_sha256==digest(observation),'Exact current automatic held state missing')
        if driver.mode=='first_handoff':
            require(driver.completed[-1]['active_arm_id']=='arm_1' and driver.support_id==driver.route[1],'Recipient automatic grasp belongs to wrong relay')
    else:
        validate_observation(driver,observation)
        require(driver.held is None and driver.automatic_grasp.holder is None,'Completed automatic placement is still held')
    require(all(e['gate']['passed'] is True for e in driver.gates),'Automatic stage gate incomplete')
    if driver.mode=='bead_lift_unit':
        # Existing source placement IK and student's checked inverse/RRT policy.
        # An exact reverse return path is deliberately not claimed.
        require(len(driver.completed)==len(driver.plans)==len(driver.gates)==len(driver.pauses)==3
            and [(r['stage_id'],r['kind'],r['active_arm_id']) for r in driver.completed]==[
                ('arm_0__approach','approach','arm_0'),('arm_0__probe_lift','carry','arm_0'),('arm_0__carry','carry','arm_0')],
            'Private bead lift needs the three completed actual operations')
        lift_gate=driver.gates[1]['gate']
        require(lift_gate['passed'] is True and lift_gate['requested_lift_m']==.055
            and lift_gate['measured_lift_m']>=.05 and lift_gate['minimum_bottom_clearance_m']>=.015
            and lift_gate['unsupported_provided_grasp_s']>=.5 and lift_gate['load_bearing_interval_s']>=.5
            and driver.gates[-1]['support_id']==driver.route[0], 'Actual private bead lift/return evidence incomplete')
    if driver.mode=='source_lift_probe':
        require(driver.particle_mode and len(driver.gates)==len(driver.plans)==len(driver.pauses)==2 and driver.gates[-1]['gate']['schema_version']=='provided-automatic-source-lift-probe-gate-v1'
            and driver.gates[-1]['gate'].get('particle_material_contract_sha256')==digest(driver.initial_supply)
            and all(p.get('particle_pause_comparison',{}).get('passed') is True for p in driver.pauses),'Automatic source lift needs both real planned/paused operations')
    status={'mission':'STUDENT_THREE_ARM_SINGLE_BATCH_SAMPLED_PASS','pick_place_unit':'PROVIDED_PICK_PLACE_UNIT_SAMPLED_PASS',
        'bead_lift_unit':'PROVIDED_BEAD_LIFT_UNIT_SAMPLED_PASS',
        'first_handoff':'STUDENT_FIRST_HANDOFF_SAMPLED_PASS','placement_probe':'PROVIDED_PLACEMENT_PROBE_SAMPLED_PASS','source_lift_probe':'PARTICLE_SOURCE_LIFT_PROBE_SAMPLED_PASS'}[driver.mode]
    result=dict(status=status,grasp_policy='automatic',grasp_model='provided_ideal',final_observation=observation,student_execution=execution,
        post_place_policy=driver.post_place_policy,post_place_resets=copy.deepcopy(driver.post_place_resets),
        physical_motion_count=len(driver.completed),zero_time_robot_reset_count=len(driver.post_place_resets),
        post_place_returns=copy.deepcopy(driver.post_place_returns),kinematic_robot_return_count=len(driver.post_place_returns),
        material_delivery_accepted=passed,physical_handoff_verified=passed or driver.mode=='first_handoff',physical_handoff_count=2 if passed else 1 if driver.mode=='first_handoff' else 0,
        full_mission_passed=passed,current_student_single_batch_mission_passed=passed,mission_completion_receipt=completion,
        completion_scope='current_student_three_arm_automatic_single_batch_v1' if passed else None,student_release_approved=False,
        legacy_friction_holder_certified=False,scope='Actual explicit provided ideal native-grasp policy. No friction/bilateral retention stamp; no repeated-model or student-release approval.')
    if driver.post_place_policy=='instant_reset':
        result['scope']='Six actual transport motions with two explicitly authorized zero-time donor-robot resets after supported native release. Cup and bead physics remain continuous. No physical robot return or swept reset path is claimed.'
    elif driver.post_place_policy=='kinematic_return':
        result['scope']='Six actual transport motions and two explicitly nonphysical reverse replays of each donor\'s saved carry then approach. Display time is separate from native time. Cup and bead physics remain continuous. No dynamics or swept collision claim is made for the return.'
    if passed and empty:
        result.update(status='EMPTY_THREE_ARM_RRT_MISSION_SAMPLED_PASS',empty_container_mission_passed=True,
            completion_scope='current_student_three_arm_empty_automatic_mission_v1',payload_mode='empty',
            material_delivery_accepted=False,material_count=0,material_rest='NOT_APPLICABLE',
            scope='One actual three-arm empty-container mission with three supported releases and two recipient reacquisitions. Paths came from the selected endpoint planner. No material delivery or student-release approval.')
    elif passed:
        result.update(delivered_count=driver.initial_supply['particle_count'] if driver.particle_mode else len(driver.initial_supply),
            delivered_mass_kg=driver.initial_supply['initial_total_mass_kg'] if driver.particle_mode else math.fsum(m['mass_kg'] for m in driver.initial_supply))
        if driver.particle_mode:result.update(status='STUDENT_THREE_ARM_PARTICLE_SINGLE_BATCH_SAMPLED_PASS',
            delivered_native_particle_ids_sha256=observation['particle_material']['native_particle_ids_sha256'])
        else:result['delivered_material_ids']=sorted(m['id'] for m in observation['materials'])
    if driver.mode=='bead_lift_unit':
        result.update(bead_lift_unit_verified=True,load_bearing_grasp_verified=True,unit_fixture=True,
            source_support_id=driver.route[0],exact_reverse_path_claimed=False,
            scope='Private near-grasp INITIAL, actual >=50mm bead16 rise and existing0.6s hold, then source return and released rest. No relay, recipient or full mission approval.')
    if driver.mode=='source_lift_probe':result.update(source_lift_probe_verified=True,unit_fixture=True)
    if driver.mode=='placement_probe':result.update(placement_probe_verified=True,unit_fixture=True)
    return result


def assess_single_batch(driver,observation,execution):
    """Explicit automatic completion. No legacy friction schema is synthesized."""
    from physics.saved_motion_planning import stages_for_policy
    expected_stages=stages_for_policy(driver.post_place_policy)
    failures=[];empty=driver.world.get('payload_mode')=='empty'
    try:
        if empty:
            from physics.episode_runtime import empty_payload_mode
            require(empty_payload_mode(driver.world,driver.scenario) and not driver.particle_mode
                and driver.initial_supply==[] and observation.get('materials')==[]
                and 'particle_material' not in observation,'Empty completion has unexpected contents')
        require(driver.held is None and driver.automatic_grasp.holder is None and observation.get('provided_grasp') is None,'Final automatic cup not released')
        require(execution['schema_version']=='student-three-arm-automatic-driver-v1' and execution['operation_sequence_complete'] is True
            and execution['reference_planner_fallback'] is False and execution['supplied_intermediate_postures'] is False
            and execution['all_requested_student_transport_motion_from_selected_strategy'] is True,'Not a completed source-bound automatic mission')
        require(len(driver.route)==4 and len(set(driver.route))==4 and driver.support_id==driver.route[-1],'Wrong automatic support route')
        require(all(len(rows)==len(expected_stages) for rows in (driver.completed,driver.plans,driver.gates,driver.pauses)),'Automatic completion needs every physical motion, plan, gate and pause in the selected policy')
        signatures=lambda rows:[(r['stage_id'],r['kind'],r['active_arm_id']) for r in rows]
        require(tuple(signatures(driver.completed))==expected_stages and signatures(driver.plans)==signatures(driver.completed)==signatures(driver.gates),'Automatic operation identities/order differ')
        require([p['stage_id'] for p in driver.pauses]==[s[0] for s in expected_stages],'Planning pause identities differ')
        require(execution.get('post_place_policy','return_home')==driver.post_place_policy,'Recorded post-placement policy differs')
        resets=getattr(driver,'post_place_resets',[])
        require(execution.get('post_place_resets',[])==resets,'Recorded native robot-reset transactions differ')
        if driver.post_place_policy=='instant_reset':
            from physics.provided_robot_reset import validate_robot_reset_receipt
            require(len(resets)==2,'The six-motion policy requires two actual zero-time donor resets')
            for reset,arm,carry_index in zip(resets,('arm_0','arm_1'),(1,3)):
                stage_id=arm+'__carry'
                validate_robot_reset_receipt(reset,expected_command=driver.initial_commands[arm],
                    expected_arm_id=arm,expected_stage_id=stage_id)
                done=driver.completed[carry_index]
                require(reset['native_scene_step']==done['native_scene_step']
                    and reset['simulation_s']==done['simulation_s']
                    and reset['released_gate_sha256']==digest(driver.gates[carry_index]['gate'])
                    and reset['after_observation_sha256']==driver.pauses[carry_index+1]['measured_observation_sha256'],
                    'Robot reset is not bound to this actual completed release and following recipient observation')
        else:
            require(resets==[],'A physical-return policy cannot borrow a robot reset')
        returns=getattr(driver,'post_place_returns',[])
        require(execution.get('post_place_returns',[])==returns,'Recorded kinematic return transactions differ')
        if driver.post_place_policy=='kinematic_return':
            from physics.provided_robot_return import validate_kinematic_return_receipt
            require(len(returns)==2,'The kinematic policy requires two complete nonphysical returns')
            for returned,arm,carry_index in zip(returns,('arm_0','arm_1'),(1,3)):
                stage_id=arm+'__carry'
                schedule=driver.saved_plan.kinematic_return_after(stage_id)
                validate_kinematic_return_receipt(returned,expected_schedule=schedule,
                    expected_command=driver.initial_commands[arm],expected_arm_id=arm,expected_stage_id=stage_id)
                done=driver.completed[carry_index]
                require(returned['native_scene_step']==done['native_scene_step']
                    and returned['simulation_s']==done['simulation_s']
                    and returned['released_gate_sha256']==digest(driver.gates[carry_index]['gate'])
                    and returned['after_observation_sha256']==driver.pauses[carry_index+1]['measured_observation_sha256'],
                    'Kinematic return is not bound to this completed release and following recipient observation')
        else:
            require(returns==[],'Another post-placement policy cannot borrow kinematic return receipts')
        require(len({p['plan_path_source_sha256'] for p in driver.plans})==1,'Student strategy source changed')
        previous_step=0;previous_time=0.;load={};supports=[]
        for done,entry,plan,pause in zip(driver.completed,driver.gates,driver.plans,driver.pauses):
            require(type(done['native_scene_step']) is int and done['native_scene_step']>previous_step
                and type(done['simulation_s']) in (int,float) and math.isfinite(done['simulation_s']) and done['simulation_s']>previous_time,'Completion clocks not increasing')
            previous_step,previous_time=done['native_scene_step'],done['simulation_s']
            require(all(type(plan.get(k)) is str and len(plan[k])==64 and all(c in '0123456789abcdef' for c in plan[k]) for k in ('request_sha256','trajectory_sha256','plan_path_source_sha256'))
                and plan['supplied_intermediate_postures'] is False,'Source-bound student plan missing')
            require(all(pause.get(k) is True for k in ('native_state_and_clock_unchanged','dispatcher_unchanged','source_bindings_verified')),'Unchanged planning pause missing')
            if driver.particle_mode:
                comparison=pause.get('particle_pause_comparison',{})
                require(comparison.get('passed') is True and comparison.get('schema_version')=='particle-episode-paused-state-comparison-v1'
                    and comparison.get('committed_observation_sha256')==pause.get('measured_observation_sha256')
                    and comparison.get('extension_boundary_receipt',{}).get('observation_sha256')==pause.get('measured_observation_sha256'),'Actual complete particle pause missing')
                require(all(type(comparison.get(k)) is str and len(comparison[k])==64 and all(c in '0123456789abcdef' for c in comparison[k])
                    for k in ('projection_sha256','pause_receipt_sha256','committed_observation_sha256')),'Particle pause source/digest receipt missing')
            gate=entry['gate'];kind=done['kind'];arm=done['active_arm_id'];expected=driver.route[int(arm[-1])+(1 if kind in ('carry','retreat') else 0)]
            require(entry['support_id']==expected and gate['passed'] is True,'Automatic support/stage gate mismatch')
            if kind=='approach':
                require(gate['schema_version']=='provided-automatic-supported-pick-gate-v1' and gate['actual_attach_receipt']['event']=='attach'
                    and digest(gate['actual_attach_receipt'])==gate['automatic_attachment_receipt_sha256'] and gate['source_support_observed'] is True
                    and gate['source_support_window_s']>0,'Real automatic attachment/support receipt missing')
                if empty:
                    require(gate['stable_contents_window_s'] is None and gate.get('material_window_required') is False
                        and gate['source_support_window_s']>=driver.world['physics_settings']['scenario']['bilateral_window_s']-1e-9,
                        'Empty acquisition must preserve the physical source-support window')
                else:require(gate['stable_contents_window_s']>0,'Original material stability receipt missing')
            if kind=='carry':
                require(gate['schema_version']=='provided-automatic-released-relay-gate-v1' and gate['failures']==[] and gate['support_id']==expected
                    and gate['active_arm_id']==arm and gate['expected_rows']==gate['observed_rows']==200 and gate['ending_s']==done['simulation_s']
                    and gate['window_s']==.5 and gate['dt_s']==.0025 and gate['load_bearing_interval_s']>=.5,'Actual automatic released support window incomplete')
                supports.append(expected);load[arm]=gate['load_bearing_interval_s']
            if kind in ('approach','carry'):
                require(gate['all_arms_rest']['passed'] is True and gate['body_rest']['passed'] is True,'Actual arm/body rest missing')
                if empty:
                    _recorded_empty_presence(gate.get('empty_payload'),done['native_scene_step'],done['simulation_s'])
                else:
                    bulk=gate['particle_bulk_rest' if driver.particle_mode else 'bead_bulk_rest']
                    require(bulk['passed'] is True and bulk['expected_rows']==bulk['observed_rows']==201 and bulk['ending_s']==done['simulation_s'],'Original full-inventory material rest missing')
                if not empty and driver.particle_mode:
                    from physics.particle_student_completion import _recorded_bulk_rest
                    _recorded_bulk_rest(gate,done['simulation_s'])
                elif not empty:
                    from physics.episode_transport_assessment import BEAD_MATERIAL_REST_LIMITS
                    require(bulk['schema_version']=='bead-material-window-rest-v1' and bulk['quality_passed'] is True
                        and bulk['limits']==BEAD_MATERIAL_REST_LIMITS and bulk['quality_failures']==bulk['settling_failures']==[],
                        'Original bead rest limits or quality changed')
        require(supports==driver.route[1:] and len(load)==3,'Both relay releases and final destination required')
        require(observation['native_scene_step']==previous_step and observation['simulation_s']==previous_time,'Final observation not from final actual release')
        require(sum(e['event']=='attach' for e in driver.automatic_events)==sum(e['event']=='release' for e in driver.automatic_events)==3,'Three actual attach/release transactions required')
        for event in driver.automatic_events:
            receipt=event['receipt'];require(receipt['event']==event['event'] and receipt['model']=='provided-ideal-native-weld-v1','Wrong actual native receipt model')
            a,b=receipt['before'],receipt['after'];require({k:v for k,v in a.items() if k!='inventory'}=={k:v for k,v in b.items() if k!='inventory'},'Automatic transaction changed actual physical state')
        require([e['event'] for e in driver.automatic_events]==['attach','release']*3,'Automatic transaction order differs')
        if empty:
            _assess_empty_native_transactions(driver,observation)
        elif driver.particle_mode:
            from physics.particle_episode_state import validate_particle_packet
            validate_particle_packet(observation['particle_material'],driver.initial_supply,observation,require_retained=True)
            require(execution['particle_material_contract_sha256']==digest(driver.initial_supply),'Original particle contract changed')
        else:
            from physics.episode_transport_assessment import _materials
            a,b=_materials(observation['materials']),_materials(driver.initial_supply)
            require(set(a)==set(b) and all((a[k]['entity_idx'],a[k]['mass_kg'])==(b[k]['entity_idx'],b[k]['mass_kg']) and a[k]['inside_carrier'] is True for k in a),'Original final bead inventory changed')
            for m in a.values():
                point(m['position_world_m'],3)
                require(math.hypot(*point(m['velocity_world_m_s'],3))<=driver.gates[-1]['gate']['limits']['content_speed_m_s'],'Final bead rest failed')
        execution_limits=driver.world['physics_settings']['execution_limits'];last_gate=driver.gates[-1]['gate']
        require(set(observation['arms'])==set(driver.arm_ids),'Final named arms missing')
        for arm in observation['arms'].values():
            point(arm['q_all_dofs'],9);v=point(arm['qd_all_dofs'],9)
            require(max(map(abs,v[:7]))<=execution_limits['rest_speed_rad_s'] and max(map(abs,v[7:]))<=.001,'Final arm/finger rest failed')
        from physics.episode_transport_assessment import _angle
        nominal=driver.goal.get('seating',{}).get('nominal_goal',driver.goal);actual=observation['body_world'];body_limits=last_gate['body_rest']['limits']
        require(driver.goal['support_id']==driver.route[-1] and nominal['support_id']==driver.route[-1]
            and math.dist(point(actual['position_m'],3),point(nominal['body_world']['position_m'],3))<=last_gate['limits']['body_position_m']
            and _angle(point(actual['quaternion_wxyz'],4),point(nominal['body_world']['quaternion_wxyz'],4))<=last_gate['limits']['body_orientation_rad']
            and math.hypot(*point(observation['body_linear_velocity_m_s'],3))<=body_limits['maximum_body_linear_speed_m_s']
            and math.hypot(*point(observation['body_angular_velocity_rad_s'],3))<=body_limits['maximum_body_angular_speed_rad_s'],'Final destination/body rest failed')
    except (ContractError,KeyError,TypeError,ValueError,OverflowError) as exc:failures.append(str(exc))
    result=dict(schema_version='current-student-empty-automatic-mission-completion-v1' if empty else 'current-student-automatic-single-batch-completion-v1',passed=not failures,failures=failures,
        grasp_model='provided_ideal',post_place_policy=driver.post_place_policy,
        expected_stage_ids=[s[0] for s in expected_stages],physical_handoff_count=2 if not failures else 0,final_observation_sha256=None if failures else digest(observation),student_release_approved=False)
    if empty:result.update(payload_mode='empty',material_count=0,material_rest='NOT_APPLICABLE')
    return result


def _assess_empty_native_transactions(driver,observation):
    """Bind all three empty-cup acquisitions/releases to one actual native cup."""
    events=driver.automatic_events
    expected=[(event,arm,arm+suffix) for arm in driver.arm_ids
        for event,suffix in (('attach','__approach'),('release','__carry'))]
    require([(e['event'],e['receipt']['arm_id'],e['stage_id']) for e in events]==expected,
        'Empty mission needs three named arm acquisitions and three named supported releases')
    baseline=events[0]['receipt']['before']['inventory'];identity=None;hands=[];previous_step=-1
    require(baseline['count']==len(baseline['rows'])==3 and baseline['weld_pairs']==[]
        and all(r[0]==2 for r in baseline['rows']),'Empty mission baseline is not the three original Panda finger equalities')
    gates={e['stage_id']:e['gate'] for e in driver.gates}
    for event in events:
        receipt=event['receipt'];before,after=receipt['before'],receipt['after'];pair=receipt['pair']
        require(type(event['native_scene_step']) is int and event['native_scene_step']>previous_step
            and before['native_clock'][:2]==[event['native_scene_step'],event['simulation_s']],
            'Empty automatic event clock/sequence differs')
        previous_step=event['native_scene_step']
        actual=(before['body_entity_idx'],before['cup_link_idx'],before['cup_mass_kg'])
        require(type(actual[0]) is int and type(actual[1]) is int and type(actual[2]) in (int,float)
            and math.isfinite(actual[2]) and actual[2]>0 and (identity is None or actual==identity),
            'Empty mission changed the original native cup identity or mass')
        identity=actual
        require(pair==[before['hand_link_idx'],before['cup_link_idx']], 'Empty receipt has wrong hand/cup pair')
        owned={**copy.deepcopy(baseline),'count':baseline['count']+1,
            'rows':baseline['rows']+[[1,*pair]],'weld_pairs':[pair]}
        if event['event']=='attach':
            require(before['inventory']==baseline and after['inventory']==owned,
                'Empty acquisition did not add exactly its owned native weld')
            require(gates[event['stage_id']]['actual_attach_receipt']==receipt,
                'Empty acquisition gate differs from its actual native event')
            hands.append(pair[0])
        else:
            require(before['inventory']==owned and after['inventory']==baseline,
                'Empty supported release did not restore the original native inventory')
            require(gates[event['stage_id']]['actual_release_receipt_sha256']==digest(receipt),
                'Empty release gate differs from its actual native event')
    require(len(set(hands))==3,'Empty handoffs reused a hand instead of all three actual arms')
    snapshot=driver.automatic_grasp.snapshot(driver.arm_ids[-1])
    require((snapshot['body_entity_idx'],snapshot['cup_link_idx'],snapshot['cup_mass_kg'])==identity
        and snapshot['inventory']==baseline and snapshot['body_world']==observation['body_world']
        and snapshot['native_clock'][:2]==[observation['native_scene_step'],observation['simulation_s']],
        'Final released empty cup differs from the actual original native body/inventory/clock')
