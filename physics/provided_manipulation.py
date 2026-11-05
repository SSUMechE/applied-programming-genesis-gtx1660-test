"""Instructor-owned terminal targets and supported pick/place operations.

This module generates terminal targets and gripper/hold records. The separate
provided_placement_plan adapter may make a bounded measured local seating
correction after the student's unchanged carry path. It never substitutes an
approach, lift, retreat or carrying route, attachment or live pose setter.
Every physical interval and the existing contact/rest gates remain mandatory.

The standard policy is deliberately limited to the centered overhead vessel's
40 mm cube handle. Its calibrated terminal hand offset is teacher-owned. An IK
success is an endpoint candidate, never a physical grasp or path certificate.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import random

from physics.carried_body import VesselModel, WorldPose, _multiply
from transport_challenge.contracts import ContractError, digest, point


STANDARD_HANDLE_SIZE_M=(.04,.04,.04)
STANDARD_HAND_POSITION_BODY_M=(0.,0.,.1031)
STANDARD_HAND_QUATERNION_BODY_WXYZ=(0.,0.,1.,0.)
PROVIDED_PLACEMENT_HOLD_S=2.0
PROVIDED_TRANSPORT_CLOSE_M=.010
PROVIDED_SUPPORTED_TARGET_M=.015
PROVIDED_PRELOAD_RAMP_S=2.5
PROVIDED_PRELOAD_DURATION_S=3.5


def teacher_policy():
    """Return detached policy/source identity, without a physical-success flag."""
    root=Path(__file__).resolve().parents[1]
    return {'schema_version':'provided-supported-manipulation-policy-v1',
        'geometry_model':'centered_overhead_handle_v3',
        'handle_size_m':list(STANDARD_HANDLE_SIZE_M),
        'hand_position_body_m':list(STANDARD_HAND_POSITION_BODY_M),
        'hand_quaternion_body_wxyz':list(STANDARD_HAND_QUATERNION_BODY_WXYZ),
        'placement_hold_s':PROVIDED_PLACEMENT_HOLD_S,
        'transport_close_target_m':PROVIDED_TRANSPORT_CLOSE_M,
        'supported_preload':{'target_m':PROVIDED_SUPPORTED_TARGET_M,'ramp_s':PROVIDED_PRELOAD_RAMP_S,'duration_s':PROVIDED_PRELOAD_DURATION_S},
        'placement_hold_policy':'Closed 2.0s hold. Mission/first_handoff then uses the declared supported preload before the unchanged measured release gate. Bounded current-grasp Z seating remains separately logged when required. The original hold is not relabelled or lengthened.',
        'placement_correction_scope':'Instructor-owned local place operation only. Student route endpoints, returned paths and their hashes are preserved. No intermediate route answer is supplied.',
        'frame':'body','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'geometry_source_sha256':hashlib.sha256((root/'transport_challenge/vessel.py').read_bytes()).hexdigest(),
        'supplied_intermediate_configurations':False,'supplied_paths':False,
        'physical_grasp_certified_by_endpoint_generation':False}


def _pose(value):
    if type(value) is WorldPose:return value
    if (not isinstance(value,dict) or value.get('frame')!='world'
            or set(value)!={'frame','position_m','quaternion_wxyz'}):
        raise ContractError('Require an explicit world-frame terminal pose')
    return WorldPose(value['position_m'],value['quaternion_wxyz'])


def grasp_endpoint_candidates(model,body_pose):
    """Return the standard terminal grasp pose. No waypoint or joint seed leaks.

    A measured body pose is composed with the instructor's fixed handle policy.
    The caller owns source/clock binding of that pose. Unsupported handles fail
    instead of guessing a new grasp or asking students to tune finger mechanics.
    """
    if type(model) is not VesselModel:
        raise ContractError('Require the immutable actual vessel model')
    geometry=model.geometry_record()
    if (geometry.get('geometry_model')!='centered_overhead_handle_v3'
            or len(model.component_boxes)<6
            or model.component_boxes[5].size_m!=STANDARD_HANDLE_SIZE_M
            or model.component_boxes[5].center_body_m!=(0.,0.,0.)):
        raise ContractError('The provided grasp supports only the standard 40mm overhead handle')
    body=_pose(body_pose)
    return (WorldPose(body.transform_point(STANDARD_HAND_POSITION_BODY_M),
        _multiply(body.quaternion_wxyz,STANDARD_HAND_QUATERNION_BODY_WXYZ)),)


def solve_endpoint(kinematics,oracle,start_q,endpoints,*,attempts=16,seed=0,wall_check=None,endpoint_check=None):
    """Try bounded IK seeds for terminal poses and choose a valid terminal q.

    The supplied BoundedKinematics facade verifies FK residuals and native
    bounds. This helper also validates its returned seven-q and asks state_valid
    for endpoint collision validity. It never calls edge_valid or a planner.
    Only the selected terminal q is returned. Random IK guesses are not route
    states. Exhausted supplied budgets propagate to the runner without fallback.
    """
    if type(attempts) is not int or not 1<=attempts<=128 or type(seed) is not int or not 0<=seed<2**31:
        raise ContractError('Endpoint solve needs1..128attempts and a nonnegative31bit seed')
    if wall_check is not None and not callable(wall_check):raise ContractError('wall_check must be callable')
    if endpoint_check is not None and not callable(endpoint_check):raise ContractError('endpoint_check must be callable')
    if not callable(getattr(kinematics,'ik',None)) or not callable(getattr(oracle,'state_valid',None)):
        raise ContractError('Use the supplied bounded IK and state oracle')
    bounds=tuple(point(pair,2) for pair in kinematics.joint_bounds_rad)
    if len(bounds)!=7 or any(lo>=hi or not math.isfinite(hi-lo) for lo,hi in bounds):
        raise ContractError('Require seven ordered native joint bounds')
    start=point(start_q,7)
    if any(not lo<=q<=hi for q,(lo,hi) in zip(start,bounds)):
        raise ContractError('Endpoint IK start seed leaves native joint bounds')
    if not isinstance(endpoints,(tuple,list)) or not 1<=len(endpoints)<=4:
        raise ContractError('Require one to four explicit terminal pose candidates')
    poses=tuple(_pose(value) for value in endpoints)
    rng=random.Random(seed);accepted=[];ik_misses=0;state_rejections=0;endpoint_rejections=0
    for attempt in range(attempts):
        if wall_check is not None:wall_check()
        index=attempt%len(poses);target=poses[index]
        guess=start if attempt<len(poses) else tuple(rng.uniform(lo,hi) for lo,hi in bounds)
        raw=kinematics.ik(target.position_m,target.quaternion_wxyz,guess)
        if wall_check is not None:wall_check()
        if raw is None:ik_misses+=1;continue
        q=point(raw,7)
        if any(not lo<=v<=hi for v,(lo,hi) in zip(q,bounds)):
            raise ContractError('Supplied endpoint IK returned out-of-bound joint angles')
        valid=oracle.state_valid(q)
        if type(valid) is not bool:raise ContractError('Endpoint state oracle must return a Boolean')
        if not valid:state_rejections+=1;continue
        assessment=None if endpoint_check is None else copy.deepcopy(endpoint_check(q))
        if endpoint_check is not None:
            if not isinstance(assessment,dict) or type(assessment.get('passed')) is not bool:
                raise ContractError('Supplied endpoint check must return an explicit passed Boolean')
            if not assessment['passed']:endpoint_rejections+=1;continue
        accepted.append((math.dist(start,q),index,attempt,q,assessment))
    chosen=min(accepted) if accepted else None
    result={'schema_version':'provided-terminal-endpoint-solve-v1',
        'status':'ENDPOINT_FOUND' if chosen is not None else 'ENDPOINT_NOT_FOUND',
        'q_rad':None if chosen is None else list(chosen[3]),
        'endpoint_hand_pose':None if chosen is None else poses[chosen[1]].to_record(),
        'selected_candidate_index':None if chosen is None else chosen[1],
        'selected_attempt':None if chosen is None else chosen[2],
        'attempts':attempts,'seed':seed,'ik_misses':ik_misses,
        'state_rejections':state_rejections,'endpoint_rejections':endpoint_rejections,
        'selected_endpoint_assessment':None if chosen is None else chosen[4],
        'accepted_endpoint_count':len(accepted),
        'policy':teacher_policy(),'intermediate_q_provided':False,'path_generated':False,
        'edge_validated':False,'physical_grasp_verified':False}
    result['receipt_sha256']=digest(result)
    return result


def placement_endpoint(world,scenario,held_spec,support_id,*,selected_body_pose=None):
    """Resolve only a named placement endpoint using this actual measured grasp.

    Reuses the existing support footprint, inverse composition and command
    seating checks. It does not lower, move or attach the physical vessel.
    """
    from physics.episode_transport_inputs import selected_support_goal,SEATING_DEPTH_M,_seating_from_nominal
    if selected_body_pose is None:
        return selected_support_goal(world,scenario,held_spec,support_id=support_id,
            command_seating_depth_m=SEATING_DEPTH_M)
    # The shared selected_support_goal's optional pose is an EXACT identity
    # assertion, not an override. Keep that contract unchanged. This explicit
    # instructor-place branch preserves measured XY only within its original
    # goal tolerance and uses the same footprint/seating geometry checks.
    nominal=selected_support_goal(world,scenario,held_spec,support_id=support_id)
    selected=_pose(selected_body_pose);canonical=nominal['body_world']
    delta=tuple(a-b for a,b in zip(selected.position_m,canonical['position_m']))
    if (delta[2]!=0. or selected.quaternion_wxyz!=tuple(canonical['quaternion_wxyz'])
            or math.hypot(*delta[:2])>world['physics_settings']['execution_limits']['final_body_position_error_m']):
        raise ContractError('Provided local placement preserves canonical Z/orientation and bounded measured XY only')
    hand=_pose(nominal['hand_world'])
    moved=WorldPose(tuple(p+d for p,d in zip(hand.position_m,delta)),hand.quaternion_wxyz)
    reconstructed=held_spec.body_pose_at(moved)
    original=digest(nominal)
    nominal.update(body_world=selected.to_record(),hand_world=moved.to_record(),
        reconstructed_body_world=reconstructed.to_record(),
        roundtrip_position_error_m=math.dist(selected.position_m,reconstructed.position_m),
        selection_scope='Instructor local place XY preservation within the unchanged original task tolerance. No student destination override.')
    result=_seating_from_nominal(nominal,held_spec,SEATING_DEPTH_M)
    for target,bounds in ((selected.position_m,world['body_root_workspace_m']),
            (moved.position_m,scenario['case']['tool_workspace_m']),
            (result['hand_world']['position_m'],scenario['case']['tool_workspace_m'])):
        if any(not lo<=v<=hi for v,(lo,hi) in zip(target,bounds)):
            raise ContractError('Provided local placement endpoint leaves the original workspace')
    result['seating']['original_task_nominal_goal_sha256']=original
    return result


class _SeatingKinematics:
    """One precise native endpoint solve, preserving the existing teacher policy."""
    def __init__(self,query,policy):self.query,self.policy=query,policy
    def fk(self,q):return self.query.fk(q)
    def ik(self,position_m,quaternion_wxyz,seed_q):
        return self.query.ik(position_m,quaternion_wxyz,seed_q,
            position_tolerance_m=self.policy['solver_position_tolerance_m'],
            orientation_tolerance_rad=self.policy['solver_orientation_tolerance_rad'])


def solve_placement_endpoint(query,world,scenario,held_spec,support_id,start_q,*,attempts=16,seed=0,wall_check=None,
                             selected_body_pose=None):
    """Own precise seating IK and its exact same-query native witness gate.

    The isolated held query must retain the pinned _last_detection snapshot
    from state_valid(q). Its source, clock and geometry remain the runner's
    responsibility. Only the selected terminal q and seating assessment escape.
    No old reference grasp, intermediate q, Cartesian lowering path or physical
    placement verdict is substituted for the supplied student's carry path.
    """
    from physics.episode_transport_inputs import assess_relay_seating_endpoint
    from transport_challenge.joint_kinematics import BoundedKinematics
    goal=placement_endpoint(world,scenario,held_spec,support_id,selected_body_pose=selected_body_pose)
    if 'seating' not in goal:raise ContractError('Provided placement requires the explicit seating policy')
    policy=goal['seating']
    kin=BoundedKinematics(_SeatingKinematics(query,policy),query.bounds,scenario['case']['tool_workspace_m'],
        max_fk_calls=0,max_ik_calls=attempts,wall_check=wall_check,
        position_tolerance_m=policy['verification_position_tolerance_m'],
        orientation_tolerance_rad=policy['verification_orientation_tolerance_rad'])
    def check(q):
        snapshot=copy.deepcopy(getattr(query,'_last_detection',None))
        return assess_relay_seating_endpoint(goal,held_spec,q,snapshot,state_valid=True)
    result=solve_endpoint(kin,query,start_q,(goal['hand_world'],),attempts=attempts,seed=seed,
        wall_check=wall_check,endpoint_check=check)
    result.pop('receipt_sha256')
    result.update(placement_goal=goal,endpoint_kinematics=kin.record())
    result['receipt_sha256']=digest(result)
    return result


def _name(value):
    if type(value) is not str or not value or value.strip()!=value:
        raise ContractError('Require an explicit arm/action name')
    return value


def pick_actions(arm_id,*,prefix='provided_pick',transport_close=False):
    """Existing settle/close/hold records at the previously reached terminal q.

    EpisodePlan/Dispatcher hold every arm's existing seven-q command throughout
    these records. The runner must bind that command, current observation and
    supported endpoint, then call assess_pick on completed physical observations.
    """
    from physics.episode_runtime import PROTOCOL
    if type(transport_close) is not bool:raise ContractError('Explicit instructor transport policy required')
    arm=_name(arm_id);prefix=_name(prefix);actions=[]
    for suffix,role,kind,key in (('settle','approach_settle','hold','approach_settle_s'),
            ('close','close','close','close_s'),('hold','final_hold','hold','final_hold_s')):
        action={'action_id':prefix+'__'+suffix,'arm_id':arm,'role':role,'kind':kind,'duration_s':PROTOCOL[key]}
        if kind=='close':action['target_finger_m']=[PROVIDED_TRANSPORT_CLOSE_M if transport_close else PROTOCOL['closed_finger_target_m']]*2
        actions.append(action)
    return actions


def supported_preload_action(arm_id,*,prefix):
    """Staff-only supported force adjustment. No arm path/posture is inserted."""
    return dict(action_id=_name(prefix)+'__instructor_supported_preload',arm_id=_name(arm_id),
        kind='supported_preload',role='place',duration_s=PROVIDED_PRELOAD_DURATION_S,
        ramp_duration_s=PROVIDED_PRELOAD_RAMP_S,target_finger_m=[PROVIDED_SUPPORTED_TARGET_M]*2,instructor_only=True)


def native_control_target_receipt(scene,arms,command,*,plan_sha256):
    """Read the actual committed9-DOF targets without exposing a dispatcher hook."""
    import numpy as np
    if int(scene.t)!=command.step or abs(float(scene.cur_t)-command.post_time_s)>1e-9:
        raise ContractError('Read provided native targets at the just-committed clock')
    values={key:np.asarray(getattr(scene.rigid_solver.dyn_state.dofs,key).to_numpy()) for key in ('ctrl_pos','ctrl_vel','ctrl_mode')}
    if any(v.ndim!=2 or v.shape[1]!=1 for v in values.values()):
        raise ContractError('Source-bound native DofsState layout changed')
    records={}
    for c in command.arms:
        arm=arms[c.arm_id];indices=list(range(arm.dof_start,arm.dof_start+9))
        actual=values['ctrl_pos'][indices,0];velocity=values['ctrl_vel'][indices,0];mode=values['ctrl_mode'][indices,0]
        target=np.asarray(c.q_rad+c.finger_q_m,dtype=actual.dtype)
        target_v=np.asarray(c.qd_rad_s+c.finger_qd_m_s,dtype=velocity.dtype)
        if not np.array_equal(actual,target) or not np.array_equal(velocity,target_v) or not np.all(mode==0):
            raise ContractError('Actual native POSITION target/mode differs from the provided operation')
        records[c.arm_id]=dict(native_ctrl_pos=actual.tolist(),native_ctrl_vel=velocity.tolist(),native_ctrl_mode=mode.tolist(),
            scheduled_ctrl_pos=target.tolist(),scheduled_ctrl_vel=target_v.tolist())
    return dict(step=command.step,simulation_s=command.post_time_s,schedule_phase=command.kind+':'+command.action_role,
        plan_sha256=plan_sha256,by_arm=records,exact_native_dtype_target_match=True,
        source='Read-only rigid_solver.dyn_state.dofs.ctrl_pos/ctrl_vel/ctrl_mode; POSITION=0')


def place_actions(arm_id,finger_bounds_m,*,prefix='provided_place'):
    """Fixed seating hold and existing open/settle after a student carry path.

    The runner must pass assess_place_ready before the open boundary and
    assess_place_release after the completed released observation window.
    """
    from physics.episode_runtime import TRANSPORT_PROTOCOL
    arm=_name(arm_id);prefix=_name(prefix)
    if not isinstance(finger_bounds_m,(tuple,list)) or len(finger_bounds_m)!=2:
        raise ContractError('Require the two actual native finger intervals')
    bounds=tuple(point(pair,2) for pair in finger_bounds_m)
    if any(lo>hi for lo,hi in bounds):raise ContractError('Finger intervals must be ordered')
    target=min(hi for lo,hi in bounds)
    if any(not lo<=target<=hi for lo,hi in bounds):
        raise ContractError('Coupled fingers lack a common native open target')
    actions=[]
    for suffix,role,kind,key in (('place','place','hold','placement_hold_s'),
            ('release','release','open','release_s'),('settle','released_settle','hold','released_settle_s')):
        duration=PROVIDED_PLACEMENT_HOLD_S if key=='placement_hold_s' else TRANSPORT_PROTOCOL[key]
        action={'action_id':prefix+'__'+suffix,'arm_id':arm,'role':role,'kind':kind,'duration_s':duration}
        if kind=='open':action['target_finger_m']=[target]*2
        actions.append(action)
    return actions


def assess_pick(*args,**kwargs):
    """Use the unchanged actual bilateral contact/support/rest/retention gate."""
    from physics.episode_transport_assessment import supported_grasp_gate
    return supported_grasp_gate(*args,**kwargs)


def assess_place_ready(observation,recent_rows,settings,rest_policy,goal):
    """The existing runtime's exact pre-release support/body-rest predicate.

    This pure assessment authenticates no fabricated observation. The trusted
    runtime still owns all per-step source, contact, clock and content checks.
    """
    from physics.episode_runtime import rest_rows,body_rest,TRANSPORT_PROTOCOL
    from physics.episode_transport_assessment import _angle
    execution=settings['execution_limits']
    selected,expected,complete=rest_rows(recent_rows,observation['simulation_s'],TRANSPORT_PROTOCOL['acceptance_window_s'])
    supported=complete and all(any(c['category']=='named_tray_bottom_support_top'
        and c.get('support_station_id')==goal['support_id'] for c in row['classified_contacts']) for row in selected)
    nominal=goal.get('seating',{}).get('nominal_goal',goal)
    expected_body=nominal['body_world'];actual=observation['body_world']
    position=math.dist(actual['position_m'],expected_body['position_m'])
    angle=_angle(actual['quaternion_wxyz'],expected_body['quaternion_wxyz'])
    rest=body_rest(recent_rows,observation['simulation_s'],execution['rest_window_s'],rest_policy)
    result={'support_window_rows':len(selected),'expected_rows':expected,'support_present_every_interval':supported,
        'body_rest':rest,'body_goal_position_error_m':position,'body_goal_orientation_error_rad':angle}
    result['passed']=bool(supported and rest['passed'] and position<=execution['final_body_position_error_m']
        and angle<=execution['final_body_orientation_error_rad'])
    return result


def assess_place_release(*args,**kwargs):
    """Use the unchanged every-interval open/support/rest/material release gate."""
    from physics.episode_transport_assessment import released_relay_gate
    return released_relay_gate(*args,**kwargs)
