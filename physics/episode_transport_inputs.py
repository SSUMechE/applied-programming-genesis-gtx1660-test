"""Measured episode inputs and a selected relay endpoint, without scene calls.

The measured grasp describes one exact terminal observation. The external
carry starts at the preceding commanded endpoint for command continuity.
Both states and their connecting edge must be separately queried. No bridge
waypoint, runtime pose setter, path or physical-grasp verdict is created here.
"""
from __future__ import annotations
import copy
import itertools
import math

from physics.carried_body import VesselModel, WorldPose, HeldBodySpec, _multiply, _inverse
from transport_challenge.contracts import ContractError, digest, point, number


def pose(record):
    if record.get('frame')!='world':raise ContractError('An explicit world pose is required')
    return WorldPose(record['position_m'],record['quaternion_wxyz'])


def _three_arm_layout(world, scenario):
    """Validate the explicit industrial recipe without inventing arm roles."""
    from physics.compound_obstacles import validate_embedded_support_components
    from physics.fixed_world_task import FixedSupportStation
    from physics.episode_paused_state import _json
    _json(world); _json(scenario)
    names=[r['id'] for r in scenario['robots']]
    if (len(names)!=3 or any(type(n) is not str or not n.strip() for n in names) or len(set(names))!=3
            or ('arm_ids' in world and set(world['arm_ids'])!=set(names))
            or scenario['case']!=world['case_record']
            or scenario['provenance']['world_sha256']!=digest(world)):
        raise ContractError('Explicit three-arm scenario/world identity required')
    stations=scenario['support_stations']
    expected=[world['source_support'],*world['relay_supports'],world['goal_support'],*world['quality_supports']]
    if (len(world['relay_supports'])!=2 or len(world['quality_supports'])!=2
            or len(stations)!=6 or len({s['id'] for s in stations})!=6
            or stations!=expected or stations!=world['support_stations']):
        raise ContractError('All six exact source/relay/goal/quality support recipes required')
    for s in stations:
        if FixedSupportStation(s['id'],s['size_m'],s['center_world_m']).to_record()!=s:
            raise ContractError('Malformed fixed support recipe')
    validate_embedded_support_components(scenario['case'],stations)
    return names,{s['id']:s for s in stations}


def observed_transport_inputs(world,scenario,approach,grasp,*,repeated_three_arm_mode=False):
    """Validate the measured reference and detach a complete query snapshot."""
    if type(repeated_three_arm_mode) is not bool:raise ContractError('Mode must be explicit boolean')
    if repeated_three_arm_mode:_three_arm_layout(world,scenario)
    if (grasp.get('status')!='SUPPORTED_GRASP_SAMPLED_PASS'
            or grasp.get('world_sha256')!=digest(world) or grasp.get('scenario_sha256')!=digest(scenario)
            or digest(world)!=scenario['provenance']['world_sha256']
            or scenario['case']!=world['case_record'] or scenario['active_arm_id']!='arm_0'):
        raise ContractError('Require a completed grasp for this exact world/scenario')
    for key in ('source_manifest_hash_still_match','input_source_hashes_still_match','frozen_snapshot_hashes_still_match'):
        if grasp.get(key) is not True:raise ContractError('Grasp source binding is incomplete: '+key)
    if grasp.get('cleanup_errors')!=[]:raise ContractError('Grasp cleanup did not complete')
    gate=grasp['supported_grasp_gate']
    if not (gate['passed'] is True and gate['all_arms_rest']['passed'] is True and gate['body_rest']['passed'] is True):
        raise ContractError('Measured rest and supported grasp are required')
    observation=copy.deepcopy(grasp['final_observation'])
    if (observation!=grasp['last_observation'] or observation['simulation_s']!=grasp['simulation_s']
            or observation['native_scene_step']!=grasp['physical_steps']
            or observation['simulation_s']!=observation['native_scene_step']*grasp['simulation_dt_s']):
        raise ContractError('Final grasp observation/clock is inconsistent')
    if set(observation['arms'])!={r['id'] for r in scenario['robots']}:
        raise ContractError('Measured arm identities differ')
    dispatch=grasp['dispatch']
    if (dispatch['last_observation_sha256']!=digest(observation)
            or dispatch['committed_steps']!=observation['native_scene_step']
            or dispatch['simulation_s']!=observation['simulation_s']
            or dispatch['command_grid_time_s']!=observation['simulation_s']
            or dispatch['pending_step'] is not None or dispatch['command_sequence_completed'] is not True
            or dispatch['plan_sha256']!=grasp['external_episode_sha256']):
        raise ContractError('Final observation is not the committed terminal dispatch state')
    model=VesselModel.from_container(scenario['container'])
    if model.geometry_sha256!=world['vessel_model']['geometry_sha256']:
        raise ContractError('Measured container geometry differs')
    active=scenario['active_arm_id'];a=observation['arms'][active]
    spec=HeldBodySpec(world['object_id'],model,pose(a['hand_world']),pose(observation['body_world']),
        tuple(a['q_all_dofs'][7:]),observation['simulation_s'],f"episode_step_{observation['native_scene_step']}")
    if spec.to_record()!=grasp['measured_held_body'] or spec.binding_sha256!=grasp['measured_held_binding_sha256']:
        raise ContractError('Held-body record is not derived from this exact observation')
    if digest(approach['plan']['trajectory'])!=grasp['external_trajectory_sha256']:
        raise ContractError('Grasp prefix and supplied approach differ')
    start=point(approach['plan']['trajectory']['waypoints'][-1],7)
    measured_q=point(a['q_all_dofs'][:7],7)
    nominal_delta=max(abs(x-y) for x,y in zip(start,measured_q))
    if nominal_delta>world['physics_settings']['closure_match_tolerances']['joint_rad']:
        raise ContractError('Observed grasp differs from the nominal continuous command')
    robots=copy.deepcopy(scenario['robots'])
    for r in robots:
        q=point(observation['arms'][r['id']]['q_all_dofs'],9)
        r['q_rad']=list(q[:7]);r['finger_q_m']=list(q[7:])
    materials=observation['materials'];supplies=grasp['initial_supply']
    if len(materials)!=len(world['initial_contents']) or len(supplies)!=len(materials):
        raise ContractError('Complete persistent contents required')
    if (len({s['entity_idx'] for s in supplies})!=len(supplies)
            or len({s['entity'] for s in supplies})!=len(supplies)
            or any(type(s['entity_idx']) is not int or s['entity_idx']<0 for s in supplies)):
        raise ContractError('Native material entities must be distinct nonnegative identities')
    contents=[]
    for expected,initial,actual in zip(world['initial_contents'],supplies,materials):
        if (actual['id']!=expected['id'] or initial['id']!=expected['id']
                or actual['entity_idx']!=initial['entity_idx'] or actual['mass_kg']!=expected['mass_kg']
                or initial['mass_kg']!=expected['mass_kg'] or initial['radius_m']!=expected['radius_m']
                or actual['inside_carrier'] is not True):
            raise ContractError('Persistent material identity/mass/retention differs')
        point(actual['velocity_world_m_s'],3)
        contents.append({'id':expected['id'],'radius_m':expected['radius_m'],
                         'center_world_m':list(point(actual['position_world_m'],3))})
    record={'schema_version':'observed-episode-held-inputs-v1','active_arm_id':active,'robots':robots,
        'body_id':world['object_id'],'body_world':spec.measured_body_world.to_record(),
        'held_body':spec.to_record(),'held_binding_sha256':spec.binding_sha256,
        'contents':contents,'observed_materials':copy.deepcopy(materials),
        'observation':observation,'observation_sha256':digest(observation),
        'observed_start_q_rad':list(measured_q),'command_start_q_rad':list(start),
        'maximum_observed_command_start_difference_rad':nominal_delta,
        'command_start_scope':'Preserved preceding command endpoint. The measured state and connecting edge require separate validation. No runtime bridge is inserted.'}
    if repeated_three_arm_mode:
        record['repeated_three_arm_mode']=True
        record['observed_snapshot']={'schema_version':'episode-observed-initialization-v1',
            'observation':copy.deepcopy(observation),'observation_sha256':digest(observation),
            'source_bindings_sha256':digest(grasp['source_bindings']),'robots_sha256':digest(robots)}
    return spec,record


def selected_relay_goal(world,scenario,spec,*,command_seating_depth_m=0.,
                        repeated_three_arm_mode=False,support_id=None,selected_body_pose=None):
    """Translate the existing supported goal convention to the chosen relay."""
    if type(repeated_three_arm_mode) is not bool:raise ContractError('Mode must be explicit boolean')
    if repeated_three_arm_mode:
        return selected_support_goal(world,scenario,spec,support_id=support_id,
            selected_body_pose=selected_body_pose,command_seating_depth_m=command_seating_depth_m)
    if support_id is not None or selected_body_pose is not None:
        raise ContractError('Explicit support selection requires the three-arm mode')
    stations=scenario['support_stations']
    if len(stations)!=3 or stations[0]!=world['source_support'] or stations[-1]!=world['goal_support']:
        raise ContractError('Source and final support are fixed')
    relay=stations[1];region=world['relay_support_region'];center=point(relay['center_world_m'],3)
    if (relay['id']!='relay_support' or relay['size_m']!=region['support_size_m']
            or not region['center_x_m'][0]<=center[0]<=region['center_x_m'][1]
            or not region['center_y_m'][0]<=center[1]<=region['center_y_m'][1]
            or center[2]!=region['center_z_m']):
        raise ContractError('Selected relay is outside the allowed region')
    original=pose(world['goal_body_world']);support_center=world['goal_support']['center_world_m']
    body=WorldPose(tuple(p+c-s for p,c,s in zip(original.position_m,center,support_center)),original.quaternion_wxyz)
    orientation=WorldPose((0.,0.,0.),_multiply(body.quaternion_wxyz,_inverse(spec.hand_to_body.quaternion_hand_body_wxyz)))
    offset=orientation.transform_point(spec.hand_to_body.translation_hand_m)
    hand=WorldPose(tuple(p-o for p,o in zip(body.position_m,offset)),orientation.quaternion_wxyz)
    reconstructed=spec.body_pose_at(hand)
    position_error=math.dist(body.position_m,reconstructed.position_m)
    quaternion_error=min(max(abs(x-y) for x,y in zip(body.quaternion_wxyz,reconstructed.quaternion_wxyz)),
                         max(abs(x+y) for x,y in zip(body.quaternion_wxyz,reconstructed.quaternion_wxyz)))
    if position_error>256*math.ulp(max(1.,*map(abs,body.position_m))) or quaternion_error>128*math.ulp(1.):
        raise ContractError('Relay hand/body inverse composition did not round-trip')
    if any(not lo<=v<=hi for v,(lo,hi) in zip(hand.position_m,scenario['case']['tool_workspace_m'])):
        raise ContractError('Relay hand endpoint outside fixed workspace')
    nominal={'support_id':relay['id'],'support':copy.deepcopy(relay),'body_world':body.to_record(),
        'hand_world':hand.to_record(),'reconstructed_body_world':reconstructed.to_record(),
        'roundtrip_position_error_m':position_error,'roundtrip_quaternion_component_error':quaternion_error,
        'selection_scope':'Reference-selected relay support, using the existing goal body/support relative placement. Endpoint only, no route.',
        'held_binding_sha256':spec.binding_sha256}
    goal=_seating_from_nominal(nominal,spec,command_seating_depth_m)
    if any(not lo<=x<=hi for x,(lo,hi) in zip(goal['hand_world']['position_m'],scenario['case']['tool_workspace_m'])):
        raise ContractError('Command seating hand endpoint leaves the workspace')
    return goal


def selected_support_goal(world,scenario,spec,*,support_id,selected_body_pose=None,command_seating_depth_m=0.):
    """Compose an explicitly named canonical support target with the current grip.

    No intermediate states are chosen. Optional body pose is an exact identity
    check against the declared target, not a hidden placement override.
    """
    _,stations=_three_arm_layout(world,scenario)
    if (type(spec) is not HeldBodySpec or spec.object_id!=world['object_id']
            or spec.model.geometry_sha256!=world['vessel_model']['geometry_sha256']
            or support_id not in stations or support_id not in world['support_targets_by_id']):
        raise ContractError('Explicit known support and exact current held model required')
    declared=world['support_targets_by_id'][support_id]['body_world']
    if selected_body_pose is not None and selected_body_pose!=declared:
        raise ContractError('Selected body pose differs from the declared fixed support target')
    if support_id==world['goal_support']['id'] and declared!=world['goal_body_world']:
        raise ContractError('Final support target must be the official fixed goal body pose')
    body=pose(declared); support=stations[support_id]
    if any(not lo<=v<=hi for v,(lo,hi) in zip(body.position_m,world['body_root_workspace_m'])):
        raise ContractError('Selected body root outside the fixed world domain')
    corners=_bottom_corners(spec.model,body);center=support['center_world_m'];size=support['size_m']
    top=center[2]+size[2]/2
    if (any(abs(c[i]-center[i])>size[i]/2+1e-12 for c in corners for i in (0,1))
            or any(abs(c[2]-top)>1e-12 for c in corners)):
        raise ContractError('Canonical supported body target must place its bottom on the named plate')
    orientation=WorldPose((0.,0.,0.),_multiply(body.quaternion_wxyz,_inverse(spec.hand_to_body.quaternion_hand_body_wxyz)))
    offset=orientation.transform_point(spec.hand_to_body.translation_hand_m)
    hand=WorldPose(tuple(p-o for p,o in zip(body.position_m,offset)),orientation.quaternion_wxyz)
    reconstructed=spec.body_pose_at(hand)
    pe=math.dist(body.position_m,reconstructed.position_m)
    qe=min(max(abs(a-b) for a,b in zip(body.quaternion_wxyz,reconstructed.quaternion_wxyz)),
           max(abs(a+b) for a,b in zip(body.quaternion_wxyz,reconstructed.quaternion_wxyz)))
    if pe>256*math.ulp(max(1.,*map(abs,body.position_m))) or qe>128*math.ulp(1.):
        raise ContractError('Selected support endpoint inverse composition did not round-trip')
    nominal={'support_id':support_id,'support':copy.deepcopy(support),'body_world':body.to_record(),
        'hand_world':hand.to_record(),'reconstructed_body_world':reconstructed.to_record(),
        'roundtrip_position_error_m':pe,'roundtrip_quaternion_component_error':qe,
        'held_binding_sha256':spec.binding_sha256,
        'selection_scope':'Explicit canonical support/body endpoint with the current measured holder. No route or physical placement verdict.'}
    goal=_seating_from_nominal(nominal,spec,command_seating_depth_m)
    for target in (hand.position_m,goal['hand_world']['position_m']):
        if any(not lo<=v<=hi for v,(lo,hi) in zip(target,scenario['case']['tool_workspace_m'])):
            raise ContractError('Selected hand endpoint outside the fixed workspace')
    return goal


SEATING_DEPTH_M = .0003
SOLVER_POSITION_TOLERANCE_M = .000025
SOLVER_ORIENTATION_TOLERANCE_RAD = .000125
VERIFY_POSITION_TOLERANCE_M = .00005
VERIFY_ORIENTATION_TOLERANCE_RAD = .00025


def _bottom_corners(model, body):
    box = model.component_boxes[0]
    return tuple(body.transform_point((box.center_body_m[0]+sx*box.size_m[0]/2,
        box.center_body_m[1]+sy*box.size_m[1]/2,box.center_body_m[2]-box.size_m[2]/2))
        for sx,sy in itertools.product((-1.,1.),repeat=2))


def _seating_orientation_error(a,b):
    a,b=point(a,4),point(b,4)
    cosine=abs(math.fsum(x*y for x,y in zip(a,b)))/(math.hypot(*a)*math.hypot(*b))
    return 2*math.acos(min(1.,cosine))


def _seating_from_nominal(nominal,spec,depth_m):
    depth=number(depth_m,'command seating depth',minimum=0.,maximum=SEATING_DEPTH_M)
    if depth not in (0.,SEATING_DEPTH_M):
        raise ContractError('This explicit diagnostic supports only zero or 0.3mm command seating')
    if depth==0.:
        return copy.deepcopy(nominal)
    body=WorldPose(nominal['body_world']['position_m'],nominal['body_world']['quaternion_wxyz'])
    support=nominal['support'];center=point(support['center_world_m'],3);size=point(support['size_m'],3)
    top=center[2]+size[2]/2
    original_corners=_bottom_corners(spec.model,body)
    if any(abs(c[i]-center[i])>size[i]/2 for c in original_corners for i in (0,1)):
        raise ContractError('Nominal bottom footprint leaves the selected support')
    lowest_gap=min(c[2]-top for c in original_corners)
    delta_z=-depth-lowest_gap
    command=WorldPose((body.position_m[0],body.position_m[1],body.position_m[2]+delta_z),body.quaternion_wxyz)
    orientation=WorldPose((0.,0.,0.),_multiply(command.quaternion_wxyz,_inverse(spec.hand_to_body.quaternion_hand_body_wxyz)))
    offset=orientation.transform_point(spec.hand_to_body.translation_hand_m)
    hand=WorldPose(tuple(p-o for p,o in zip(command.position_m,offset)),orientation.quaternion_wxyz)
    reconstructed=spec.body_pose_at(hand)
    position_error=math.dist(command.position_m,reconstructed.position_m)
    component_error=min(max(abs(x-y) for x,y in zip(command.quaternion_wxyz,reconstructed.quaternion_wxyz)),
                        max(abs(x+y) for x,y in zip(command.quaternion_wxyz,reconstructed.quaternion_wxyz)))
    if position_error>256*math.ulp(max(1.,*map(abs,command.position_m))) or component_error>128*math.ulp(1.):
        raise ContractError('Seating endpoint inverse composition did not round-trip')
    command_corners=_bottom_corners(spec.model,command)
    gaps=[c[2]-top for c in command_corners]
    if not all(-.001<gap<-VERIFY_POSITION_TOLERANCE_M for gap in gaps):
        raise ContractError('Tilted nominal bottom cannot realize this bounded seated endpoint')
    result=copy.deepcopy(nominal)
    result.update(body_world=command.to_record(),hand_world=hand.to_record(),reconstructed_body_world=reconstructed.to_record(),
        roundtrip_position_error_m=position_error,roundtrip_quaternion_component_error=component_error,
        selection_scope='Endpoint command seating hypothesis. No pose setter or physical support approval.')
    result['seating']={'schema_version':'relay-command-seating-v1','command_depth_m':depth,
        'nominal_goal':copy.deepcopy(nominal),'nominal_goal_sha256':digest(nominal),
        'vessel_model_geometry_sha256':spec.model.geometry_sha256,'held_binding_sha256':spec.binding_sha256,
        'nominal_bottom_corners_world_m':[list(c) for c in original_corners],
        'command_bottom_corners_world_m':[list(c) for c in command_corners],
        'support_top_z_m':top,'nominal_lowest_gap_m':lowest_gap,'command_delta_z_m':delta_z,
        'command_corner_signed_gaps_m':gaps,
        'solver_position_tolerance_m':SOLVER_POSITION_TOLERANCE_M,
        'solver_orientation_tolerance_rad':SOLVER_ORIENTATION_TOLERANCE_RAD,
        'verification_position_tolerance_m':VERIFY_POSITION_TOLERANCE_M,
        'verification_orientation_tolerance_rad':VERIFY_ORIENTATION_TOLERANCE_RAD,
        'physical_scope':'Compliant joint-control endpoint only. The unchanged native 1mm contact guard and measured 200-interval support gate remain mandatory.'}
    return result


def assess_relay_seating_endpoint(goal,spec,endpoint_q,snapshot,*,state_valid):
    """Check copied same-query endpoint evidence. It cannot grant an allowance."""
    if 'seating' not in goal:
        return {'required':False,'passed':True,'scope':'Legacy endpoint policy unchanged'}
    policy=goal['seating']; q=point(endpoint_q,7)
    try:
        if type(state_valid) is not bool or state_valid is not True:
            raise ContractError('The full native state oracle did not accept the endpoint')
        if snapshot['pair_witness_multiset_agrees'] is not True:
            raise ContractError('Endpoint witnesses are not bound to a complete same-query buffer')
        proxy=snapshot['same_query_proxy_poses']
        if tuple(point(proxy['active_q_at_composition'],9)[:7])!=q:
            raise ContractError('Endpoint witnesses describe a different native q')
        hand=proxy['hand_world'];body=proxy['body_world']
        position=math.dist(point(hand['position_m']),point(goal['hand_world']['position_m']))
        angle=_seating_orientation_error(hand['quaternion_wxyz'],goal['hand_world']['quaternion_wxyz'])
        corners=_bottom_corners(spec.model,WorldPose(body['position_m'],body['quaternion_wxyz']))
        corner_error=max(math.dist(a,b) for a,b in zip(corners,policy['command_bottom_corners_world_m']))
        gaps=[c[2]-policy['support_top_z_m'] for c in corners]
        rows=snapshot['rows']
        support=[r for r in rows if r.get('category')=='named_tray_bottom_support_top'
                 and r.get('support_station_id')==goal['support_id'] and r.get('tray_component')=='solid_0'
                 and not isinstance(r.get('penetration_m'),bool) and isinstance(r.get('penetration_m'),(int,float))
                 and math.isfinite(r['penetration_m']) and 0.<=r['penetration_m']<=.001]
        passed=(position<=VERIFY_POSITION_TOLERANCE_M and angle<=VERIFY_ORIENTATION_TOLERANCE_RAD
                and corner_error<=VERIFY_POSITION_TOLERANCE_M and all(-.001<gap<0. for gap in gaps)
                and bool(support) and not any(r.get('category')=='UNALLOWED_CONTACT' for r in rows))
        return {'required':True,'passed':passed,'hand_position_error_m':position,'hand_orientation_error_rad':angle,
            'maximum_bottom_corner_command_error_m':corner_error,'actual_query_bottom_signed_gaps_m':gaps,
            'named_relay_support_witness_count':len(support),'same_query_sha256':digest(snapshot),
            'endpoint_q_rad':list(q),'scope':'Native query endpoint only. No live penetration, support duration or released-relay approval.'}
    except (KeyError,TypeError,ValueError,ZeroDivisionError) as error:
        return {'required':True,'passed':False,'failure':type(error).__name__+': '+str(error)}
