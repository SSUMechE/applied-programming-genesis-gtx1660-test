"""Persistent two-arm external approach, grasp and optional relay transport.

No IK, route insertion, post-initialization rigid-body pose setter or refill.
The explicit automatic student policy uses a provided native attachment.
The imported trajectory is an offline geometric query, not live replanning.
Native imports occur only in run_episode. The default ends at supported grasp.
An explicit external carry may continue to checked relay release. The optional
supplied continuation planner adds donor retreat, recipient supported grasp
and an optional measured recipient carry to the fixed goal in that same scene.
One sampled bead delivery does not establish repeated throughput or release.
Optional detail stills temporarily move only the rendering camera, then restore it.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from physics.carried_body import VesselModel, WorldPose, HeldBodySpec
from physics.contact_accuracy import contact_accuracy_request, validate_request
from physics.episode_joint_scene import (validate_episode_snapshot, episode_base_position,
                                         validate_native_base_position)
from physics.fixed_world_task import FixedSupportStation
from transport_challenge.contracts import ContractError, digest
from transport_challenge.episode_plan import EpisodePlan, EpisodeDispatcher
from transport_challenge.joint_planning import JointLimits, JointTrajectory


def _encode_owned_step_log(value, *, direct=False):
    """Strict full-row encoding with the original invalid-value fallback."""
    try:
        return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    except (TypeError,ValueError):
        if not direct:
            raise
        from physics.run_joint_path import _json_safe
        return json.dumps(_json_safe(value),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'


DT=.0025
RIGID16_PROFILE='rigid16_bead_tc2h_v1'
EMPTY_RIGID_PROFILE='empty_rigid_default_v1'
DETAIL_CAMERA_SOURCE_SHA256='d0191a2bcde29c8e8b996e2844f2e5ee76cc0942231a45dcdcace808ecbf3814'
DETAIL_IMAGE_CHECKPOINTS=('first_relay_release','second_relay_release','final_goal_release')
SCENE_PRESENTATION_SOURCES={
    'utils/urdf.py':'b7684e074d813d56ab99f5d8c9ca46e396a37afc72f78589485d94c8363b69f0',
    'options/vis.py':'cb2033895af3f92d10dccb8c8cdf519f250f61a09754b2e683786051ca5f8cc0',
    'options/surfaces.py':'3ffb310e9c21a5d3808a355015f07e39e77084d166455a6ea344268cbcf75bfb',
    'options/morphs.py':'bedaf986729d70507eb724f8ba04c1f8ed15d95e695ad345bf3ccdb53513f4c7',
    'engine/scene.py':'d4b6b2dd11d6745147daaa3fc47e2f172d832c9bfc8c6b68576bb1d4415fce3c'}
EQUALITY_LAYOUT_SOURCES={
    'engine/solvers/rigid/abd/misc.py':'bb3d2c80cf68c3fe89536b1b798c5913f62390ef42f58747021b63cdbf9f8ca2',
    'engine/solvers/rigid/rigid_solver.py':'e076854af17262323041544ff692d366402a700357053618c8d89fed1e86b19f',
    'engine/entities/rigid_entity/rigid_entity.py':'b0f86f9c9ebdf62b9df2e5aadc0719c28e06cd52e0ce1bef452b1a269d9b2092',
    'engine/entities/rigid_entity/rigid_equality.py':'2a2143af7d644dd2917947fffbadd6933bd384d2dc93ccdf0f5ae197f9e2b369'}
PROTOCOL={'initial_settle_s':.5,'approach_settle_s':.5,'close_s':2.5,'final_hold_s':1.,
          'closed_finger_target_m':.015,'finger_rest_speed_m_s':.001,
          'finger_rest_scope':'New instructor prototype criterion, metres/second. Not an inherited joint limit or released rubric.',
          'scope':'Instructor partial-stage reference. Every interval counts.'}
TRANSPORT_PROTOCOL={'placement_hold_s':.5,'release_s':1.,'released_settle_s':1.,
    'acceptance_window_s':.5,'scope':'Explicit relay-stage reference durations, reused from the earlier released-stage test. Not a calibrated competition deadline.'}


def _capture_release_detail(camera,output,observation,result,*,checkpoint,support_id):
    """Optional raw media only. Return False only if camera restoration failed.

    The caller invokes this after the existing release gate. It supplies the
    actual post-step observation, not a reconstructed state or command pose.
    No rendered image establishes contact, content count or physical acceptance.
    """
    media=result.get('detail_images')
    if (not media or not media['enabled'] or media['disabled'] or camera is None
            or checkpoint not in DETAIL_IMAGE_CHECKPOINTS
            or len(media['captures'])>=3
            or any(row['checkpoint']==checkpoint for row in media['captures'])):
        return True
    entry={'checkpoint':checkpoint,'support_id':support_id,'image_saved':False,
           'camera_restored':None,'native_scene_step':observation['native_scene_step'],
           'simulation_s':observation['simulation_s']}
    media['captures'].append(entry)
    saved=None
    def values(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        value=np.asarray(value).copy()
        if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('Camera/root vector must have three finite values')
        return value
    def error(stage,exc):
        result['detail_image_errors'].append({'checkpoint':checkpoint,'stage':stage,
            'type':type(exc).__name__,'message':str(exc)})
    try:
        saved={key:values(getattr(camera,'get_'+key)()) for key in ('pos','lookat','up')}
        body=values(observation['body_world']['position_m'])
        target=body+np.array((0.,0.,.02))
        position=body+np.array((.25,-.30,.70))
        entry.update(body_root_m=body.tolist(),camera_before={k:v.tolist() for k,v in saved.items()},
                     camera_requested={'pos':position.tolist(),'lookat':target.tolist(),'up':[0.,0.,1.]})
        camera.set_pose(pos=position,lookat=target,up=(0.,0.,1.))
        from PIL import Image
        filename='detail_'+checkpoint+'.png'
        Image.fromarray(camera.render(force_render=True)[0]).save(Path(output)/filename)
        entry.update(file=filename,image_saved=True)
    except Exception as exc:
        error('capture',exc)
    finally:
        if saved is not None:
            try:
                camera.set_pose(**saved)
                restored={key:values(getattr(camera,'get_'+key)()) for key in saved}
                differences={key:float(np.max(np.abs(restored[key]-saved[key]))) for key in saved}
                entry['camera_restore_max_abs_difference']=differences
                # Camera-only float32 reconstruction check, never a physics tolerance.
                if max(differences.values())>1e-6:raise ValueError('Camera public pose did not restore within float32 precision')
                entry['camera_restored']=True
            except Exception as exc:
                entry['camera_restored']=False
                media.update(disabled=True,subsequent_camera_images_unavailable=True)
                error('restore',exc)
    return entry['camera_restored'] is not False


def _rgb(value):
    if type(value) is not str or len(value)!=7 or value[0]!='#':raise ContractError('Use an explicit RGB hex color')
    try:return tuple(int(value[i:i+2],16)/255. for i in (1,3,5))+(1.,)
    except ValueError as exc:raise ContractError('Malformed RGB hex color') from exc


def scene_support_roles(world,scenario):
    """Resolve ordered supports, with optional quality stations after the goal."""
    stations=scenario['support_stations']
    qualities=world.get('quality_supports',[])
    if ('quality_supports' in world and (not isinstance(qualities,list) or len(qualities)!=2
            or 'relay_supports' not in world)):
        raise ContractError('Quality stations require exactly two named supports after the three-arm goal')
    if not isinstance(stations,list) or len(stations) not in (3,4,6):
        raise ContractError('Require source, one or two relays, goal and optional quality supports')
    names=[s['id'] for s in stations]
    if len(set(names))!=len(names):raise ContractError('Physical support names must be unique')
    goal_index=len(stations)-len(qualities)-1
    if stations[0]!=world['source_support'] or stations[goal_index]!=world['goal_support']:
        raise ContractError('Named source or goal support changed')
    if 'relay_supports' in world:
        relays=world['relay_supports']
        if (len(scenario['robots'])!=3 or not isinstance(relays,list) or len(relays)!=2
                or stations!=[world['source_support'],*relays,world['goal_support'],*qualities]):
            raise ContractError('Three-arm world must bind both selected relays, goal and declared quality supports')
    elif len(stations)!=3 or len(scenario['robots'])!=2:
        raise ContractError('Additional arms/supports need the explicit relay-support contract')
    return ['source',*['relay']*(goal_index-1),'goal',*['quality']*len(qualities)]


def runtime_scene_options(world,scenario):
    """Validate opt-in nominal assemblies and appearance without changing physics."""
    from physics.compound_obstacles import validate_compound_obstacle_groups,validate_embedded_support_components
    from transport_challenge.contracts import point
    groups=validate_compound_obstacle_groups(scenario['case'])
    reserved={'floor',scenario['body_id'],*[r['id'] for r in scenario['robots']],
              *[s['id'] for s in scenario['support_stations']],*[f'tray_0_ball_{i}' for i in range(len(scenario['contents']))]}
    if any(g['id'] in reserved for g in groups):raise ContractError('Assembly aliases a physical entity')
    validate_embedded_support_components(scenario['case'],scenario['support_stations'])
    if 'visual_contract' not in world:
        from physics.episode_joint_scene import validate_query_visual
        q=validate_query_visual(scenario['case'],scenario['support_stations'])
        if q is None:return groups,None
        return groups,{'appearance_source':'case.query_visual','component_colors':q['component_colors'],
            'support_colors':q.get('support_colors',{}),'floor_color':q['floor_color'],
            'object_color':q.get('vessel_color'),'camera_proposals':[q['camera']],
            'hidden_component_ids':q.get('hidden_component_ids',[]),'floor_patches_enabled':False}
    scene_support_roles(world,scenario)
    v=copy.deepcopy(world['visual_contract'])
    if not isinstance(v,dict):raise ContractError('Visual contract must be a mapping')
    if (v.get('floor_patches_are_collision_geometry') is not False or v.get('floor_color')!='#ffffff'
            or v.get('mount_zone_color')!='#969696'):
        raise ContractError('Visual floor must remain white with noncollision gray base zones')
    for k in ('floor_color','mount_zone_color','source_color','relay_color','goal_color','object_color'):_rgb(v[k])
    if 'quality_color' in v:_rgb(v['quality_color'])
    ids={o['id'] for o in scenario['case']['obstacles']}
    if not isinstance(v.get('component_colors'),dict) or set(v['component_colors'])!=ids:
        raise ContractError('Every nominal obstacle component needs exactly one visual color')
    for c in v['component_colors'].values():_rgb(c)
    if 'component_roles' in v or 'obstacle_role_palette' in v:
        if set(v.get('component_roles',{}))!=ids:raise ContractError('Incomplete obstacle semantic roles')
        palette=v.get('obstacle_role_palette',{})
        if any(role not in palette or palette[role]!=v['component_colors'][name] for name,role in v['component_roles'].items()):
            raise ContractError('Component color disagrees with its semantic role')
    zones=world['installation_zones_by_arm']
    if len(zones)!=len(scenario['robots']) or {z['arm_id'] for z in zones}!={r['id'] for r in scenario['robots']}:
        raise ContractError('Base-zone displays must cover every exact arm once')
    paint_ids={'display_source','display_goal','display_relay_region',*['display_base_'+z['arm_id'] for z in zones]}
    if paint_ids & (reserved | ids | {g['id'] for g in groups}):raise ContractError('Visual patch aliases a physical entity')
    regions=world['relay_regions'] if 'relay_supports' in world else [world['relay_support_region']]
    for bounds in [z[k] for z in zones for k in ('x_m','y_m')]+[r[k] for r in regions for k in ('center_x_m','center_y_m')]:
        lo,hi=point(bounds,2)
        if lo>=hi:raise ContractError('Display zones require ordered metre bounds')
    proposals=v.get('camera_proposals',[])
    if proposals:
        c=proposals[0];pos=point(c['position_m'],3);look=point(c['lookat_m'],3);fov=c['fov_deg']
        if pos==look or type(fov) not in (int,float) or not math.isfinite(fov) or not 1<=fov<179:
            raise ContractError('Invalid visual camera proposal')
    if 'mount_regions' in world or 'relay_regions' in world:
        from physics.episode_presentation import shared_region_specs,board_line_specs,framed_camera_options
        display=shared_region_specs(world,v)+board_line_specs(v)
        if {p['id'] for p in display} & (reserved | ids | {g['id'] for g in groups}):
            raise ContractError('Shared-region display aliases a physical entity')
        if not proposals:raise ContractError('Shared-region display needs the explicit reviewed camera')
        framed_camera_options(proposals[0])
        if 'particle_material_request' in world:
            from physics.particle_episode_bridge import validate_request
            validate_request(world,scenario)
        elif empty_payload_mode(world,scenario):
            pass
        elif len(world['initial_contents'])!=4 or any(c.get('shape')!='sphere' for c in world['initial_contents']):
            raise ContractError('Current live material palette requires the actual four-sphere supply')
    return groups,v


def add_nominal_obstacles(gs,scene,case,directory,groups,visual,*,headless_visuals=False):
    """Physical fixed geometry. No planning inflation or pose/path operation."""
    from physics.compound_obstacles import write_compound_obstacle_asset
    obstacles=[];assets=[];by_id={o['id']:o for o in case['obstacles']}
    if not groups:
        for o in case['obstacles']:
            lo,hi=np.asarray(o['lower_m']),np.asarray(o['upper_m'])
            appearance={} if visual is None else {'surface':gs.surfaces.Default(color=_rgb(visual['component_colors'][o['id']]))}
            hidden={} if visual is None or 'hidden_component_ids' not in visual else {'visualization':o['id'] not in visual['hidden_component_ids']}
            if headless_visuals:hidden={'visualization':False}
            obstacles.append(scene.add_entity(gs.morphs.Box(size=hi-lo,pos=(lo+hi)/2,fixed=True,**hidden),name=o['id'],**appearance))
        return obstacles,assets
    for group in groups:
        specs=[]
        for name in group['component_ids']:
            o=by_id[name];lo,hi=np.asarray(o['lower_m']),np.asarray(o['upper_m'])
            specs.append({'id':name,'size_m':(hi-lo).tolist(),'center_world_m':((lo+hi)/2).tolist(),
                          'nominal_lower_m':lo.tolist(),'nominal_upper_m':hi.tolist(),'planning_clearance_m':0.})
        colors=None if visual is None else {n:visual['component_colors'][n] for n in group['component_ids']}
        hidden={} if visual is None or 'hidden_component_ids' not in visual else {
            'hidden_component_ids':[n for n in group['component_ids'] if n in visual['hidden_component_ids']]}
        asset=write_compound_obstacle_asset(directory,group,specs,component_colors=colors,**hidden);assets.append(asset)
        obstacles.append(scene.add_entity(gs.morphs.URDF(file=asset['path'],pos=asset['center_world_m'],fixed=True,
            align=False,merge_fixed_links=True,prioritize_urdf_material=visual is not None,
            **({'visualization':False} if headless_visuals else {})),name=group['id']))
    return obstacles,assets


def nominal_obstacle_ledger(gs,obstacles,case,assets):
    from physics.compound_obstacles import validate_built_compound
    from physics.loaded_joint_scene import validate_built_support
    rows=[]
    if assets:
        if len(obstacles)!=len(assets):raise ContractError('Missing native compound assembly')
        for entity,asset in zip(obstacles,assets):rows.extend(validate_built_compound(entity,asset,gs.GEOM_TYPE.BOX,fixed=True))
        if len(rows)!=len(case['obstacles']) or {r['component_id'] for r in rows}!={o['id'] for o in case['obstacles']}:
            raise ContractError('Native compound inventory differs from nominal map')
    else:
        if len(obstacles)!=len(case['obstacles']):raise ContractError('Missing native obstacle')
        for entity,o in zip(obstacles,case['obstacles']):
            lo,hi=np.asarray(o['lower_m']),np.asarray(o['upper_m'])
            rows.append(validate_built_support(entity,{'id':o['id'],'size_m':(hi-lo).tolist(),'center_world_m':((lo+hi)/2).tolist()},gs.GEOM_TYPE.BOX))
    return rows


def runtime_support_registry(gs,supports,stations,obstacles,assets,case,*,native_components=None):
    """Authenticate standalone boxes and exact embedded plate geometry together."""
    from physics.compound_obstacles import validate_embedded_support_components,embedded_support_station
    from physics.loaded_joint_scene import validate_built_support
    embedded=validate_embedded_support_components(case,stations)
    standalone=[s for s in stations if s['id'] not in embedded]
    if len(supports)!=len(standalone):raise ContractError('Standalone support inventory changed')
    records={};entities={}
    for entity,station in zip(supports,standalone):
        checked=validate_built_support(entity,station,gs.GEOM_TYPE.BOX)
        records[checked['geom_id']]=checked;entities[station['id']]=entity
    if embedded:
        rows=nominal_obstacle_ledger(gs,obstacles,case,assets) if native_components is None else native_components
        by_component={r['component_id']:r for r in rows};by_entity={e.name:e for e in obstacles}
        if len(by_component)!=len(rows) or len(by_entity)!=len(obstacles):raise ContractError('Compound component/entity identity repeated')
        for station in stations:
            if station['id'] in embedded:
                component=embedded[station['id']]
                if component not in by_component:raise ContractError('Embedded native support component missing')
                checked=embedded_support_station(by_component[component],station)
                if checked['geom_id'] in records or checked['entity'] not in by_entity:
                    raise ContractError('Embedded support geometry/entity binding changed')
                records[checked['geom_id']]=checked;entities[station['id']]=by_entity[checked['entity']]
    return records,entities


def unique_scene_entities(entities):
    """One configuration/equality/material visit per actual native object."""
    result=[];seen=set()
    for entity in entities:
        if id(entity) not in seen:result.append(entity);seen.add(id(entity))
    return result


def add_visual_floor_patches(gs,scene,world,visual):
    if visual is None or visual.get('floor_patches_enabled') is False:return [],[]
    if 'mount_regions' in world or 'relay_regions' in world:
        from physics.episode_presentation import shared_region_specs,board_line_specs
        specs=shared_region_specs(world,visual)+board_line_specs(visual)
        entities=[scene.add_entity(gs.morphs.Box(size=s['size_m'],pos=s['center_world_m'],fixed=True,collision=False,visualization=True),
            surface=gs.surfaces.Default(color=_rgb(s['color'])),name=s['id']) for s in specs]
        if any(e.n_geoms!=0 or len(e.geoms)!=0 for e in entities):raise ContractError('Display patch unexpectedly adds collision geometry')
        return entities,specs
    specs=[]
    def patch(name,x,y,color):
        specs.append({'id':name,'size_m':[x[1]-x[0],y[1]-y[0],.0002],
            'center_world_m':[(x[0]+x[1])/2,(y[0]+y[1])/2,.0002],'color':color})
    for z in world['installation_zones_by_arm']:patch('display_base_'+z['arm_id'],z['x_m'],z['y_m'],visual['mount_zone_color'])
    for role in ('source','goal'):
        s=world[role+'_support'];x,y,_=s['center_world_m'];sx,sy,_=s['size_m']
        patch('display_'+role,[x-sx/2,x+sx/2],[y-sy/2,y+sy/2],visual[role+'_color'])
    region=world['relay_support_region'];patch('display_relay_region',region['center_x_m'],region['center_y_m'],visual['relay_color'])
    entities=[scene.add_entity(gs.morphs.Box(size=s['size_m'],pos=s['center_world_m'],fixed=True,collision=False,visualization=True),
        surface=gs.surfaces.Default(color=_rgb(s['color'])),name=s['id']) for s in specs]
    if any(e.n_geoms!=0 or len(e.geoms)!=0 for e in entities):raise ContractError('Display patch unexpectedly adds collision geometry')
    return entities,specs


def limits_from_record(record):
    return JointLimits(record['position_rad'],record['velocity_rad_s'],record['acceleration_rad_s2'])


def empty_payload_mode(world,scenario=None):
    """Explicit zero-content task. Never infer empty from a missing material."""
    mode=world.get('payload_mode')
    if mode is None:
        if world.get('rigid_runtime_profile')==EMPTY_RIGID_PROFILE:
            raise ContractError('Empty physical profile requires explicit payload_mode empty')
        return False
    if mode!='empty':
        raise ContractError('Unknown explicit payload mode')
    if (world.get('rigid_runtime_profile')!=EMPTY_RIGID_PROFILE
            or 'particle_material_request' in world
            or world.get('initial_contents')!=[]
            or world.get('initial_content_world_centers_m')!=[]
            or scenario is not None and scenario.get('contents')!=[]):
        raise ContractError('Empty payload requires its exact profile and no rigid or particle contents')
    return True


def empty_runtime_source_paths():
    """Actual reused cup-contact helper must be in the caller's source closure."""
    path=Path(__file__).resolve().parents[1]/'qa/all_material_physics_20261031/liquid_integration_branch/canonical_liquid_scene.py'
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest()}


def rigid_runtime_profile(world):
    """Explicit bead-only model selection. Absence preserves substeps1/100."""
    if empty_payload_mode(world):return None  # Explicit default1 profile, never rigid16.
    profile=world.get('rigid_runtime_profile')
    if profile is not None and (profile!=RIGID16_PROFILE or 'particle_material_request' in world):
        raise ContractError('Only explicit rigid16_bead_tc2h_v1 without native particles is supported')
    return profile


def rigid16_duration_counters(timeline, *, phase, active_arm_id, support_id, state):
    """Consume authenticated actual h witnesses chronologically, not last*DT."""
    if type(timeline) is not list or len(timeline)!=16:
        raise ContractError('Rigid16 counters require every actual internal interval')
    result=copy.deepcopy(state);previous=None
    for item in timeline:
        if (item['dt_s']!=DT/16 or item['contacts_allowed'] is not True
                or (previous is not None and item['start_s']!=previous)):
            raise ContractError('Incomplete or disallowed rigid16 duration timeline')
        previous=item['end_s'];h=item['dt_s'];start=item['start_s'];supports=item['support_ids']
        bilateral=item['bilateral_by_arm'].get(active_arm_id) is True
        supported=support_id in supports
        result['support_since']=(start if result['support_since'] is None else result['support_since']) if supported else None
        good=phase=='hold' and bilateral
        result['bilateral_since']=(start if result['bilateral_since'] is None else result['bilateral_since']) if good else None
        unsupported=phase=='carry' and bilateral and not supports
        result['unsupported_bilateral_s']=result['unsupported_bilateral_s']+h if unsupported else 0.
        result['maximum_unsupported_bilateral_s']=max(result['maximum_unsupported_bilateral_s'],result['unsupported_bilateral_s'])
        result['maximum_unsupported_bilateral_by_arm_s'][active_arm_id]=max(
            result['maximum_unsupported_bilateral_by_arm_s'][active_arm_id],result['unsupported_bilateral_s'])
        holder=item['automatic_holder_arm_id']
        unheld_support=holder is not None and phase=='carry' and not supports
        result['unsupported_provided_grasp_s']=result['unsupported_provided_grasp_s']+h if unheld_support else 0.
        if holder is not None:
            if holder not in result['maximum_unsupported_provided_grasp_by_arm_s']:
                raise ContractError('Unknown actual internal provided holder')
            result['maximum_unsupported_provided_grasp_by_arm_s'][holder]=max(
                result['maximum_unsupported_provided_grasp_by_arm_s'][holder],result['unsupported_provided_grasp_s'])
    return result


def validate_inputs(world, scenario, approach, reference, *, student_limits=None):
    """Bind the supplied world, external trajectory and prior measured settings."""
    rigid_runtime_profile(world)
    if (digest(world)!=scenario['provenance']['world_sha256'] or world['case_record']!=scenario['case']
            or world['object_id']!=scenario['body_id'] or world['source_body_world']!=scenario['body_pose']
            or world['vessel_model']['container']!=scenario['container']):
        raise ContractError('World/scenario identity changed')
    scene_support_roles(world,scenario)
    runtime_scene_options(world,scenario)
    arm_ids=[r['id'] for r in scenario['robots']]
    if len(arm_ids) not in (2,3) or len(set(arm_ids))!=len(arm_ids) or scenario['active_arm_id'] not in arm_ids:
        raise ContractError('Select one source actor from two or three persistent named arms')
    goal_index=scene_support_roles(world,scenario).index('goal')
    if (scenario['support_stations'][0]!=world['source_support'] or scenario['support_stations'][goal_index]!=world['goal_support']):
        raise ContractError('Fixed source/goal support changed')
    if 'particle_material_request' in world:
        from physics.particle_episode_bridge import validate_request
        validate_request(world,scenario)
    elif empty_payload_mode(world,scenario):
        pass
    elif (len(world['initial_contents'])!=4 or len(scenario['contents'])!=4
            or [c['id'] for c in world['initial_contents']]!=['tray_0:'+str(i) for i in range(4)]):
        raise ContractError('Need four fixed persistent material identities')
    for expected,actual,center in zip(world['initial_contents'],scenario['contents'],world['initial_content_world_centers_m']):
        if (actual!={'id':expected['id'],'radius_m':expected['radius_m'],'center_world_m':center}
                or expected['radius_m']!=.012 or not math.isfinite(expected['mass_kg']) or expected['mass_kg']<=0):
            raise ContractError('Initial supply differs from the fixed world')
    model=VesselModel.from_container(scenario['container'])
    if model.geometry_sha256!=world['vessel_model']['geometry_sha256']:raise ContractError('Vessel geometry binding changed')
    validate_episode_snapshot(scenario['case'],scenario['active_arm_id'],scenario['robots'],model,
        WorldPose(scenario['body_pose']['position_m'],scenario['body_pose']['quaternion_wxyz']),scenario['body_id'],
        tuple(FixedSupportStation(s['id'],s['size_m'],s['center_world_m']) for s in scenario['support_stations']),scenario['contents'])
    if student_limits is not None:
        if approach is not None or type(student_limits) is not JointLimits:
            raise ContractError('Live student mode requires native-model limits and no frozen approach')
        initial=next(r['q_rad'] for r in scenario['robots'] if r['id']==scenario['active_arm_id'])
        # Initialization command only. The driver calls the student planner for
        # every real motion after the first measured settling segment.
        trajectory=JointTrajectory((initial,),(),student_limits)
    else:
        trajectory=_validated_external_approach(approach,scenario)
    settings=_validated_physics_settings(world,reference)
    return model,trajectory,settings


def _validated_external_approach(approach,scenario):
    if (approach.get('status')!='EPISODE_APPROACH_DISCRETE_PATH_FOUND' or approach['scenario']!=scenario
            or approach.get('input_source_hashes_still_match') is not True
            or approach.get('frozen_snapshot_hashes_still_match') is not True or approach.get('cleanup_errors')!=[]):
        raise ContractError('Require the exact completed external query result')
    t=approach['plan']['trajectory'];limits=limits_from_record(t['limits'])
    trajectory=JointTrajectory(t['waypoints'],t['edge_durations_s'],limits)
    if trajectory.to_record()!=t or digest(t)!=approach['plan']['evidence']['trajectory_sha256']:
        raise ContractError('External trajectory record/hash changed')
    if list(trajectory.waypoints[0])!=next(r['q_rad'] for r in scenario['robots'] if r['id']==scenario['active_arm_id']):
        raise ContractError('External path starts at a different initial command')
    return trajectory


def _validated_physics_settings(world,reference):
    settings=world['physics_settings']
    if digest(settings)!=world['physics_settings_sha256']:raise ContractError('World physics settings hash changed')
    request=validate_request(settings['instructor_contact_accuracy'])
    expected={'noslip_iterations':5,'torsional_friction':True,'tray_torsional_radius_m':.001}
    if (reference.get('status')!='RELEASED_DELIVERY_SAMPLED_PASS'
            or reference['configured_grasp_physics']['requested']!=expected
            or reference['contact_accuracy_configuration']['requested']!=request
            or reference['configured_grasp_physics']['diagnostic_simulation']!={'dt_s':DT,'substeps':1,'finger_equality_timeconst_s':.01}
            or reference.get('input_source_hashes_still_match') is not True
            or reference.get('frozen_snapshot_hashes_still_match') is not True):
        raise ContractError('Explicit measured ninth-configuration provenance is missing')
    for key in ('kp','kv','effort_limits'):
        if settings['controller'][key]!=reference['controller'][key]:raise ContractError('Controller differs from measured reference')
    if settings['execution_limits']!=reference['execution_limits']:raise ContractError('Execution limits changed')
    return copy.deepcopy(settings)


def reference_body_tilt_limits(transport_plan, reference_execution_body_tilt_limit_deg=None):
    """Optional reference tracking margin, never a clamp on measured tilt.

    None preserves the historical queried-limit execution rule and record shape.
    The explicit prototype hard limit is exactly 8 degrees. Planning may be more
    conservative, and the queried limit remains bound to every child request.
    """
    hard=reference_execution_body_tilt_limit_deg
    if hard is None:return None
    if type(hard) not in (int,float) or not math.isfinite(hard) or hard!=8.:
        raise ContractError('Explicit reference execution body-tilt limit must be exactly 8 degrees')
    planning=None if transport_plan is None else transport_plan.get('query_body_tilt_limit_deg')
    if type(planning) not in (int,float) or not math.isfinite(planning) or not 0<=planning<=hard:
        raise ContractError('Planning body-tilt limit must be finite, nonnegative and no greater than the hard execution limit')
    return {'planning_limit_deg':planning,'execution_hard_limit_deg':hard,
        'planning_margin_deg':hard-planning,
        'scope':'Explicit instructor reference tracking margin. Measured tilt is compared directly without rounding or clamping. Child planning retains the queried limit.'}


def validate_transport_inputs(world,scenario,approach,physics_reference,transport_plan,grasp_reference,*,repeated_three_arm_mode=False):
    """Authenticate one measured scene and its externally generated carry."""
    model,_,settings=validate_inputs(world,scenario,approach,physics_reference)
    from physics.episode_transport_inputs import observed_transport_inputs,selected_relay_goal,assess_relay_seating_endpoint
    mode_kwargs={'repeated_three_arm_mode':True} if repeated_three_arm_mode else {}
    spec,observed=observed_transport_inputs(world,scenario,approach,grasp_reference,**mode_kwargs)
    goal_kwargs={**mode_kwargs,'support_id':world['relay_supports'][0]['id']} if repeated_three_arm_mode else {}
    goal=selected_relay_goal(world,scenario,spec,command_seating_depth_m=transport_plan.get('command_seating_depth_m',0.),**goal_kwargs)
    if (transport_plan.get('status')!='EPISODE_HELD_DISCRETE_PATH_FOUND'
            or transport_plan.get('schema_version')!='episode-held-query-result-v1'
            or transport_plan.get('world_sha256')!=digest(world) or transport_plan.get('scenario_sha256')!=digest(scenario)
            or transport_plan.get('approach_result_sha256')!=digest(approach)
            or transport_plan.get('grasp_reference_sha256')!=digest(grasp_reference)
            or transport_plan.get('observed_inputs')!=observed or transport_plan.get('goal')!=goal
            or transport_plan.get('command_start_q_rad')!=observed['command_start_q_rad']):
        raise ContractError('External held plan does not bind the exact measured episode and relay')
    for key in ('source_manifest_hash_still_match','input_source_hashes_still_match','frozen_snapshot_hashes_still_match',
                'initial_observed_state_valid','initial_command_state_valid','observed_to_command_edge_valid'):
        if transport_plan.get(key) is not True:raise ContractError('External held plan lacks required evidence: '+key)
    if transport_plan.get('cleanup_errors')!=[] or transport_plan.get('physical_steps')!=0:
        raise ContractError('Require a clean zero-step held query')
    if 'seating' in goal:
        endpoint=transport_plan['selected_endpoint']
        assessment=assess_relay_seating_endpoint(goal,spec,endpoint['q_rad'],endpoint.get('same_query_witnesses'),state_valid=endpoint.get('valid'))
        if assessment.get('passed') is not True or endpoint.get('seating_assessment')!=assessment:
            raise ContractError('Command seating lacks exact accepted same-query endpoint evidence')
    evidence=transport_plan['query_evidence']
    tilt=transport_plan.get('query_body_tilt_limit_deg')
    if type(tilt) not in (int,float) or not math.isfinite(tilt) or not 0<=tilt<=180 or tilt!=evidence.get('maximum_body_tilt_deg'):
        raise ContractError('Live tilt constraint must exactly match the finite queried constraint')
    if (evidence.get('held_carried_query_supported') is not True or evidence.get('full_scene_collision_query') is not True
            or evidence.get('held_body_binding_sha256')!=spec.binding_sha256
            or evidence['contact_policy']['permission_arm_id']!=scenario['active_arm_id']
            or evidence['contact_policy']['accepted_state_requires_active_bilateral_witnesses'] is not True):
        raise ContractError('Full peer-aware held geometry/contact evidence required')
    validation=transport_plan['durable_path_validation']
    if validation.get('passed') is not True or type(validation.get('sample_count')) is not int or validation['sample_count']<2:
        raise ContractError('Complete durable sampled-path validation required')
    t=transport_plan['plan']['trajectory'];limits=limits_from_record(t['limits'])
    carry=JointTrajectory(t['waypoints'],t['edge_durations_s'],limits)
    if (carry.to_record()!=t or digest(t)!=transport_plan['plan']['evidence']['trajectory_sha256']
            or t['limits']!=approach['plan']['trajectory']['limits']
            or list(carry.waypoints[0])!=observed['command_start_q_rad']
            or list(carry.waypoints[-1])!=transport_plan['selected_endpoint']['q_rad']):
        raise ContractError('External carry path/limits/endpoints changed')
    return {'trajectory':carry,'observed_inputs':observed,'goal':goal,'held_body':spec,
        'transport_plan_sha256':digest(transport_plan),'grasp_reference_sha256':digest(grasp_reference)}


def episode_record(scenario, trajectory, initial_observation, limits_by_arm, native_finger_bounds,transport=None):
    """Wrap only the external path in explicit holds/closure, never extra q poses."""
    arms=[];active=scenario['active_arm_id']
    zero_motion=trajectory.duration_s==0.
    if zero_motion and (len(trajectory.waypoints)!=1 or list(trajectory.waypoints[0])!=next(
            r['q_rad'] for r in scenario['robots'] if r['id']==active)):
        raise ContractError('Zero-motion external approach must preserve the exact declared initial seven-q command')
    for record in scenario['robots']:
        name=record['id'];observed=initial_observation['arms'][name]
        arms.append({'arm_id':name,'role':'source_actor' if name==active else 'parked_peer',
            'initial_q_rad':list(record['q_rad']),'initial_finger_m':observed['q_all_dofs'][7:],
            'finger_bounds_m':[list(v) for v in native_finger_bounds[name]],'joint_limits':limits_by_arm[name].to_record()})
    actions=[]
    for aid,role,kind,duration in (
            ('initial_settle','initial_settle','hold',PROTOCOL['initial_settle_s']),
            ('external_approach','approach','hold' if zero_motion else 'trajectory',.5 if zero_motion else None),
            ('approach_settle','approach_settle','hold',PROTOCOL['approach_settle_s']),
            ('supported_close','close','close',PROTOCOL['close_s']),
            ('supported_hold','final_hold','hold',PROTOCOL['final_hold_s'])):
        entry={'action_id':aid,'arm_id':active,'role':role,'kind':kind}
        if kind=='trajectory':entry['trajectory']=trajectory.to_record()
        else:entry['duration_s']=duration
        if kind=='close':entry['target_finger_m']=[PROTOCOL['closed_finger_target_m']]*2
        actions.append(entry)
    steps=math.ceil(trajectory.duration_s/DT)+sum(round(PROTOCOL[k]/DT) for k in
        ('initial_settle_s','approach_settle_s','close_s','final_hold_s'))
    if zero_motion:steps+=round(.5/DT)
    if transport is not None:
        carry=transport['trajectory']
        actions.append({'action_id':'external_carry','arm_id':active,'role':'carry','kind':'trajectory','trajectory':carry.to_record()})
        for aid,role,kind,duration in (
                ('relay_placement_hold','place','hold',TRANSPORT_PROTOCOL['placement_hold_s']),
                ('relay_release','release','open',TRANSPORT_PROTOCOL['release_s']),
                ('relay_released_settle','released_settle','hold',TRANSPORT_PROTOCOL['released_settle_s'])):
            action={'action_id':aid,'arm_id':active,'role':role,'kind':kind,'duration_s':duration}
            if kind=='open':action['target_finger_m']=[min(v[1] for v in native_finger_bounds[active])]*2
            actions.append(action)
        steps+=math.ceil(carry.duration_s/DT)+sum(round(TRANSPORT_PROTOCOL[k]/DT) for k in ('placement_hold_s','release_s','released_settle_s'))
    record={'schema_version':'serial-joint-episode-plan-v1','episode_id':'two_arm_source_approach_and_supported_grasp',
        'initial_state_sha256':digest(initial_observation),'dt_s':DT,'deadline_s':steps*DT,'arms':arms,'actions':actions}
    if transport is not None:record['episode_id']='two_arm_continuous_approach_grasp_carry_relay_release'
    if len(arms)==3:record['episode_id']=record['episode_id'].replace('two_arm','three_arm',1)
    return EpisodePlan.from_record(record,limits_by_arm=limits_by_arm)


def rest_rows(rows, ending_s, duration):
    if not math.isfinite(ending_s) or ending_s<0:raise ContractError('Invalid rest-window ending clock')
    selected=[r for r in rows if ending_s-duration+1e-10<r['simulation_s']<=ending_s+1e-10]
    expected=round(duration/DT);end=round(ending_s/DT)
    complete=len(selected)==expected
    if complete:
        for r,i in zip(selected,range(end-expected+1,end+1)):
            if (type(r['native_scene_step']) is not int or r['native_scene_step']!=i
                    or not math.isfinite(r['simulation_s']) or abs(r['simulation_s']-i*DT)>1e-9):
                raise ContractError('Rest requires consecutive actual step/time observations')
    return selected,expected,complete


def measured_rest(rows, ending_s, execution, arms):
    duration=execution['rest_window_s']
    selected,expected,complete=rest_rows(rows,ending_s,duration);details={}
    for r in selected:
        if set(r['arms'])!=set(arms):raise ContractError('Rest window lacks exact arm identities')
        for arm in r['arms'].values():
            for key,size in (('qd_all_dofs',9),('qdd_fd_rad_s2',7)):
                values=np.asarray(arm[key])
                if values.shape!=(size,) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
                    raise ContractError('Malformed measured rest vector')
    for name in arms:
        speed=max((max(map(abs,r['arms'][name]['qd_all_dofs'][:7])) for r in selected),default=None)
        finger_speed=max((max(map(abs,r['arms'][name]['qd_all_dofs'][7:])) for r in selected),default=None)
        acceleration=max((max(map(abs,r['arms'][name]['qdd_fd_rad_s2'])) for r in selected),default=None)
        details[name]={'peak_speed_rad_s':speed,'peak_acceleration_fd_rad_s2':acceleration,'peak_finger_speed_m_s':finger_speed,
            'finger_speed_limit_m_s':PROTOCOL['finger_rest_speed_m_s'],
            'passed':complete and speed<=execution['rest_speed_rad_s'] and acceleration<=execution['rest_acceleration_fd_rad_s2']
                and finger_speed<=PROTOCOL['finger_rest_speed_m_s']}
    return {'window_s':duration,'expected_rows':expected,'observed_rows':len(selected),'arms':details,
            'scope':'Seven arm joints use inherited radian limits. Both fingers use a separate new prototype metre/second criterion.',
            'passed':all(v['passed'] for v in details.values())}


def body_rest(rows, ending_s, duration, policy):
    selected,expected,complete=rest_rows(rows,ending_s,duration)
    peaks={}
    for key in ('body_linear_velocity_m_s','body_angular_velocity_rad_s'):
        values=[np.asarray(r[key]) for r in selected]
        if any(v.shape!=(3,) or v.dtype.kind not in 'fiu' or not np.isfinite(v).all() for v in values):
            raise ContractError('Malformed actual body velocity')
        peaks[key]=max((float(np.linalg.norm(v)) for v in values),default=None)
    return {'observed_rows':len(selected),'expected_rows':expected,'window_s':duration,'peaks':peaks,
        'limits':{k:policy[k] for k in ('maximum_body_linear_speed_m_s','maximum_body_angular_speed_rad_s')},
        'passed':complete and peaks['body_linear_velocity_m_s']<=policy['maximum_body_linear_speed_m_s']
            and peaks['body_angular_velocity_rad_s']<=policy['maximum_body_angular_speed_rad_s']}


def verify_boundary(request, observation, recent_rows, settings, *, source_bindings_verified):
    """Use actual last state. This cannot repair a mismatch or insert motion."""
    limits=settings['execution_limits'];differences={}
    if digest(observation)!=request['measured_state_sha256'] or observation['simulation_s']!=request['simulation_s']:
        raise ContractError('Stale measured boundary observation')
    failures=[]
    for wanted in request['expected_arm_commands']:
        name=wanted['arm_id'];actual=observation['arms'][name]['q_all_dofs']
        qerr=max(abs(a-b) for a,b in zip(actual[:7],wanted['q_rad']))
        ferr=max(abs(a-b) for a,b in zip(actual[7:],wanted['finger_q_m']))
        # A blocked closing finger does not reach its commanded .015 target.
        # The released settling hold also permits residual finger motion.
        # Its separate final assessment requires the full open/no-pad window.
        finger_required=not (request['action_role'] in ('final_hold','carry','place','release','released_settle') and name==request['active_arm_id'])
        differences[name]={'max_joint_error_rad':qerr,'max_finger_error_m':ferr,'finger_command_match_required':finger_required}
        if qerr>limits['final_joint_error_rad'] or (finger_required and ferr>settings['closure_match_tolerances']['finger_m']):
            failures.append(name)
    rest=measured_rest(recent_rows,observation['simulation_s'],limits,observation['arms'])
    detail={'action_id':request['action_id'],'deviations':differences,'rest':rest,
            'source_bindings_verified':source_bindings_verified,'query_is_offline':True}
    if failures or source_bindings_verified is not True or (request['measured_rest_required'] and not rest['passed']):
        raise ContractError('Measured action boundary rejected: '+json.dumps(detail,allow_nan=False))
    ack={k:request[k] for k in ('plan_sha256','initial_state_sha256','action_id','at_step','simulation_s','measured_state_sha256')}
    ack.update(schema_version='episode-boundary-ack-v1',verified_arm_ids=list(observation['arms']),
        actual_state_verified=True,measured_rest_verified=rest['passed'],source_bindings_verified=True)
    return ack,detail


def prepare_equalities(entities, arms, equality_type):
    from supplied_support.native_settings import _constraint_objects,_parameter_vector
    changes=[]
    for name,arm in arms.items():
        wanted=[int(arm.get_joint(n).idx) for n in ('finger_joint1','finger_joint2')]
        matches=[e for e in arm.equalities if e.type==equality_type and e.entity is arm
                 and [int(e.eq_obj1id),int(e.eq_obj2id)]==wanted]
        if len(matches)!=1 or arm.is_built:raise ContractError('Need one unbuilt named Panda finger equality per arm')
        eq=matches[0];before=_parameter_vector(eq.sol_params,None);after=before.copy();after[0]=.01
        eq.set_sol_params(after.copy())
        if not np.array_equal(_parameter_vector(eq.sol_params,None),after):raise ContractError('Unexpected equality setter result')
        changes.append({'entity':name,'global_id':int(eq.idx),'joint_ids':wanted,'joint_names':['finger_joint1','finger_joint2'],
                        'before':before.tolist(),'after':after.tolist()})
    return {'equalities':changes,'inventory':[(group,{**row,'prebuild':_parameter_vector(obj.sol_params,None).tolist()})
            for group,obj,row in _constraint_objects(entities)]}


def native_equalities(scene, entities, before, diagnostic_path=None, *, expected_substeps=1, rigid_profile=None):
    from supplied_support.native_settings import _constraint_objects,_parameter_vector
    def key(group,record):return (group,record['entity'],record['identity']['local_index'])
    old={key(g,r):r for g,r in before['inventory']};rows=[]
    if len(old)!=len(before['inventory']):raise ContractError('Duplicate prebuild semantic constraint identity')
    if (rigid_profile not in (None,RIGID16_PROFILE)
            or (rigid_profile is not None and expected_substeps!=16)
            or type(expected_substeps) is not int
            or (expected_substeps not in (1,100) and not (expected_substeps==16 and rigid_profile==RIGID16_PROFILE))
            or float(scene.sim.dt)!=DT or scene.sim.substeps!=expected_substeps):
        raise ContractError('Actual integration timestep changed')
    inventory=_constraint_objects(entities)
    if diagnostic_path is not None:
        Path(diagnostic_path).write_text(json.dumps({'scope':'Pre/post semantic identity with explicit source-pinned native equality ordinal validation.',
            'prebuild':before,'postbuild_identity':[(group,identity) for group,obj,identity in inventory]},
            ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    solver=scene.rigid_solver;equalities=list(solver.equalities)
    if len(equalities)!=solver.n_equalities or len({id(e) for e in equalities})!=len(equalities):
        raise ContractError('Native equality row sequence is incomplete or duplicated')
    native_equalities=solver.dyn_info.equalities
    native_data=np.asarray(native_equalities.eq_data.to_numpy()).copy()
    if native_data.ndim!=3 or native_data.shape[0]<len(equalities) or native_data.shape[1:]!=(1,11) or native_data.dtype.kind!='f':
        raise ContractError('Unexpected native equality data layout')
    native_ids=set();mapping=[]
    for group,obj,identity in inventory:
        semantic=key(group,identity)
        if semantic not in old:raise ContractError('Missing prebuild semantic constraint identity: '+repr(semantic))
        previous=old.pop(semantic)
        compared=('entity','identity') if group=='equalities' else tuple(identity)
        if any(previous[k]!=identity[k] for k in compared):
            raise ContractError('Constraint ownership changed during build: '+json.dumps({
                'group':group,'global_id':identity['global_id'],'previous':previous,'current':identity,
                'different_fields':[k for k in compared if previous[k]!=identity[k]]},allow_nan=False))
        native_index=identity['global_id'];mapping_row=None
        if group=='equalities':
            matches=[i for i,e in enumerate(equalities) if e is obj]
            if len(matches)!=1:raise ContractError('Equality object absent from exact native initialization order')
            native_index=matches[0]
            if native_index in native_ids:raise ContractError('Duplicate native equality row')
            native_ids.add(native_index)
            entity=next((e for e in entities if e.name==identity['entity']),None)
            if obj.entity is not entity or entity.equalities[identity['identity']['local_index']] is not obj:
                raise ContractError('Equality semantic owner does not match the actual object')
            values=[int(getattr(native_equalities,name)[native_index,0]) for name in ('eq_type','eq_obj1id','eq_obj2id')]
            expected_ids=[int(obj.type),int(obj.eq_obj1id),int(obj.eq_obj2id)]
            if values!=expected_ids:raise ContractError('Native equality type/joint IDs differ from owning Panda')
            expected_data=np.asarray(obj.eq_data,dtype=native_data.dtype)
            if expected_data.shape!=(11,) or not np.isfinite(expected_data).all() or not np.array_equal(native_data[native_index,0],expected_data):
                raise ContractError('Native equality coefficients differ from semantic record')
            public=solver.get_sol_params(eqs_idx=native_index)
            raw=public.detach().cpu().numpy() if hasattr(public,'detach') else np.asarray(public)
            if raw.shape!=(1,7):raise ContractError('Unexpected public native-row parameter shape')
            native=_parameter_vector(raw[0],None)
            mapping_row={'entity':identity['entity'],'entity_local_index':identity['identity']['local_index'],
                'prebuild_public_wrapper_idx':previous['global_id'],'built_public_wrapper_idx':identity['global_id'],
                'native_row_index':native_index,'native_type_and_joint_ids':values,
                'native_eq_data':native_data[native_index,0].tolist(),'native_data_dtype':str(native_data.dtype)}
            mapping.append(mapping_row)
        else:native=_parameter_vector(obj.sol_params,None)
        expected=np.asarray(previous['prebuild'],dtype=native.dtype)
        expected[0]=max(2*DT/expected_substeps,float(expected[0])) if previous['prebuild'][0]>np.finfo(float).eps else .01
        if not np.array_equal(native,expected):raise ContractError('Native constraint parameters differ from declared resolution')
        rows.append({'group':group,**previous,'native_row_index':native_index,'native':native.tolist(),'dtype':str(native.dtype),
                     **({'native_mapping':mapping_row} if mapping_row else {})})
    if old:raise ContractError('Incomplete native parameter inventory')
    if native_ids!=set(range(solver.n_equalities)):raise ContractError('Missing native equality rows')
    for change in before['equalities']:
        row=next(r for r in rows if r['group']=='equalities' and r['entity']==change['entity'] and r['identity']['local_index']==0)
        if row['entity']!=change['entity'] or row['native'][0]!=np.asarray(.01,dtype=row['dtype']).item():
            raise ContractError('Named Panda equality did not receive .01 seconds')
    return {'schema_version':'episode-native-constraints-v1','dt_s':DT,'substeps':expected_substeps,'overrides':before['equalities'],'rows':rows,
        'equality_native_row_mapping':mapping,'native_row_source_sha256':dict(EQUALITY_LAYOUT_SOURCES),
        'mapping_scope':'Pinned solver._init_equality_fields/kernel initialization order. Public equality wrapper IDs may repeat and are never used as native row indices.'}


def close_trace_stream(stream,result):
    """An I/O cleanup error must not hide the primary failure or skip the scene."""
    if stream is not None:
        try:stream.close()
        except Exception as exc:result['cleanup_errors'].append('trace stream: '+repr(exc))


def run_episode(gs, *, world, scenario, approach, physics_reference, output, source_bindings,
                source_check, wall_limit_s=1200., render=True,transport_plan=None,grasp_reference=None,observation_readback='scalar',
                continuation_phases=None,continuation_planner=None,global_deadline_s=None,record_video=True,
                repeated_three_arm_mode=False,detail_images=False,reference_execution_body_tilt_limit_deg=None,
                profile_max_steps=None,profile_synchronize=False,gravity_readback=None,profile_gravity_parity_every=0,
                batched_motor_control=None,profile_native_timing=None,headless_visuals=None,reuse_static_aabbs=None,
                compact_setup_logs=True,student_driver=None,particle_cpic_diagnostic=False,
                particle_world_quality_diagnostic=False,rigid_constraint_solver=None,
                rigid16_pose_readback='packed',rigid16_pose_verify_scalar=False,rigid16_observer_profile=False,
                rigid16_classifier_abba=False,automatic_grasp_readback=None,
                 automatic_grasp_verify_scalar=False,automatic_grasp_abba=False,rigid16_combined_abba=False,
                  rigid16_readback='legacy',trace_mode='full',execution_backend='legacy',
                  single_active_arm=None,fixed_peer_reference=None):
    """Execute in one owned scene. Optional images-only capture uses actual boundaries."""
    if rigid_constraint_solver is not None and rigid_constraint_solver is not gs.constraint_solver.CG:
        raise ContractError('Optional numerical screen accepts only explicit gs.constraint_solver.CG')
    rigid_profile=rigid_runtime_profile(world)
    single_active_contract=None;single_active_kwargs={};fixed_peers={};fixed_peer_descriptors={};fixed_peer_hands={};fixed_peer_fingers={}
    if single_active_arm is not None:
        if (single_active_arm!='arm_0' or fixed_peer_reference is None or execution_backend!='device'
                or getattr(student_driver,'mode',None)!='one_carry' or getattr(student_driver,'saved_plan',None) is None
                or rigid_profile!=RIGID16_PROFILE):
            raise ContractError('Single active execution currently requires explicit saved arm0 bead16 one-carry and its native geometry reference')
        fixed_peer_reference=Path(fixed_peer_reference).resolve(strict=True)
        fixed_reference_record=json.loads(fixed_peer_reference.read_bytes())
        if (fixed_reference_record.get('schema_version')!='three-dynamic-arm-initial-native-geometry-reference-v1'
                or fixed_reference_record.get('native_clock')!=[0,0.,0,0]
                or fixed_reference_record.get('native_steps')!=0
                or set(fixed_reference_record.get('arms',{}))!={'arm_0','arm_1','arm_2'}):
            raise ContractError('Fixed peers require the actual unchanged three-dynamic INITIAL reference')
    elif fixed_peer_reference is not None:
        raise ContractError('A fixed peer reference requires the explicit single active mode')
    if execution_backend not in ('legacy','device'):raise ContractError('Unknown explicit execution backend')
    if execution_backend=='device' and (getattr(student_driver,'mode',None) not in ('one_carry','tier_mission')
            or getattr(student_driver,'saved_plan',None) is None
            or getattr(student_driver,'grasp_policy',None)!='automatic'
            or rigid_profile!=RIGID16_PROFILE or render or record_video
            or observation_readback!='batched' or rigid16_readback!='native' or trace_mode!='compact'
            or profile_max_steps is not None or profile_native_timing is not None):
        raise ContractError('Device backend requires headless automatic saved bead16 one-carry/native/compact')
    if rigid16_readback not in ('legacy','consolidated','verify','native','native-verify') or trace_mode not in ('full','compact'):
        raise ContractError('Explicit rigid16 readback and trace modes required')
    if rigid16_readback!='legacy' and (rigid_profile!=RIGID16_PROFILE or observation_readback!='batched'
            or rigid16_pose_readback!='packed' or rigid16_pose_verify_scalar or rigid16_combined_abba):
        raise ContractError('Consolidated readback requires the original batched/packed rigid16 profile')
    if trace_mode=='compact' and rigid_profile!=RIGID16_PROFILE:
        raise ContractError('Compact traces are scoped to the original rigid16 episode')
    if type(rigid16_combined_abba) is not bool or (rigid16_combined_abba and
            (profile_max_steps!=160 or rigid_profile is None or automatic_grasp_readback!='packed')):
        raise ContractError('Combined readback ABBA requires the160-step rigid16 diagnostic')
    if rigid16_combined_abba:
        rigid16_classifier_abba=True;automatic_grasp_abba=True
    if rigid16_pose_readback not in ('scalar','packed') or type(rigid16_pose_verify_scalar) is not bool:
        raise ContractError('Explicit rigid16 scalar/packed readback and Boolean parity required')
    if type(rigid16_observer_profile) is not bool or (rigid16_observer_profile and profile_max_steps is None):
        raise ContractError('Observer timing is an explicit short-profile diagnostic only')
    if type(rigid16_classifier_abba) is not bool or (rigid16_classifier_abba and profile_max_steps!=160):
        raise ContractError('Classifier ABBA requires the160-step initial-settle diagnostic')
    if automatic_grasp_readback not in (None,'scalar','packed') or type(automatic_grasp_verify_scalar) is not bool:
        raise ContractError('Explicit provided-grasp readback and Boolean parity required')
    if type(automatic_grasp_abba) is not bool or (automatic_grasp_abba and
            (profile_max_steps!=160 or rigid_profile is None or (rigid16_classifier_abba and not rigid16_combined_abba) or automatic_grasp_readback!='packed')):
        raise ContractError('Provided-grasp ABBA requires an isolated160-step packed diagnostic')
    resolved_grasp_readback=('packed' if rigid_profile is not None else 'scalar') if automatic_grasp_readback is None else automatic_grasp_readback
    if resolved_grasp_readback=='packed' and rigid_profile is None:
        raise ContractError('Packed provided-grasp readback is currently scoped to rigid16')
    if automatic_grasp_verify_scalar and (resolved_grasp_readback!='packed' or profile_max_steps is None):
        raise ContractError('Provided-grasp parity requires a short packed diagnostic')
    empty_mode=empty_payload_mode(world,scenario)
    empty_video=empty_mode and record_video and render
    empty_timing=empty_mode and (empty_video or getattr(student_driver,'mode',None)=='mission')
    empty_video_costs=Counter()
    if empty_mode and (rigid_constraint_solver is not None or profile_native_timing is not None):
        raise ContractError('Empty default physical profile cannot mix a different solver/contact timing diagnostic')
    if rigid_profile is not None and (rigid_constraint_solver is not None or profile_native_timing is not None):
        raise ContractError('Rigid16 model cannot be mixed with a different numerical/contact timing diagnostic')
    setup_profile=None
    if profile_max_steps is not None or 'particle_material_request' in world or empty_timing:
        from physics.episode_profile import EpisodeSetupProfile
        setup_profile=EpisodeSetupProfile()
    def setup_mark(name):
        if setup_profile is not None:setup_profile.mark(name)
    from physics.carried_body import write_carried_vessel_urdf
    from physics.episode_joint_scene import native_snapshot_command,audit_episode_pairs
    from physics.episode_contacts import classify_episode_live_contacts
    from physics.particle_episode_contacts import classify_particle_episode_live_contacts
    from physics.gravity_feedforward import GravityFeedforward,rotation_matrix_wxyz
    from physics.handle_contacts import verify_contact_sources,read_panda_pad_primitives,match_native_pads
    from physics.joint_scene import array
    from physics.loaded_joint_scene import validate_asset_components,validate_built_components,validate_built_support
    from physics.contact_accuracy import prepare_geometry_parameters,built_evidence
    from physics.compound_obstacles import validate_embedded_support_components
    from physics.live_contact_coverage import build_live_contact_coverage,require_live_contact_coverage,live_contact_coverage_stamp
    from physics.scene_assets import interior_membership
    from physics.substep_observer import _copy_contacts
    from physics.run_joint_path import _json_safe, _json_can_encode_unchanged
    from physics.delivery_inputs import prototype_policy
    from physics.episode_transport_assessment import supported_grasp_gate,reference_agreement,released_relay_gate
    from supplied_support.native_settings import (solver_configuration_evidence,torsional_configuration_evidence,
        tray_torsional_radius_evidence,native_bead_geometry,body_tilt_deg,quaternion_error)
    setup_mark('runtime_imports')
    if observation_readback not in ('scalar','verify','batched'):
        raise ContractError('Observation readback must be scalar, verify or batched')
    if type(record_video) is not bool:raise ContractError('record_video must be an explicit Boolean')
    if type(detail_images) is not bool:raise ContractError('detail_images must be an explicit Boolean')
    if headless_visuals is None:headless_visuals=not render
    if type(headless_visuals) is not bool or headless_visuals and render:
        raise ContractError('Headless visual geometry omission requires rendering disabled')
    physical_display={'visualization':False} if headless_visuals else {}
    if reuse_static_aabbs is not None and type(reuse_static_aabbs) is not bool:
        raise ContractError('Static AABB reuse must be Boolean or automatic')
    if single_active_arm is not None:reuse_static_aabbs=False
    if type(compact_setup_logs) is not bool:raise ContractError('Setup log format selection must be Boolean')
    if type(repeated_three_arm_mode) is not bool:raise ContractError('repeated_three_arm_mode must be an explicit Boolean')
    automatic_performance_selection=gravity_readback is None or batched_motor_control is None
    native_three_arm_cuda=('particle_material_request' not in world
        and isinstance(scenario,dict) and len(scenario.get('robots',()))==3
        and getattr(gs,'cuda',None) is not None and getattr(gs,'backend',None)==gs.cuda
        and getattr(getattr(gs,'device',None),'type',None)=='cuda' and getattr(gs,'use_zerocopy',False) is True)
    if gravity_readback is None:gravity_readback='fused' if native_three_arm_cuda else 'scalar'
    if batched_motor_control is None:batched_motor_control=bool(native_three_arm_cuda and gravity_readback in ('packed','fused'))
    if single_active_arm is not None:batched_motor_control=False
    if gravity_readback not in ('scalar','packed','fused'):raise ContractError('Unknown gravity readback')
    if type(batched_motor_control) is not bool:raise ContractError('Batched motor control must be Boolean')
    if (type(profile_gravity_parity_every) is not int or not 0<=profile_gravity_parity_every<=2000
            or profile_gravity_parity_every and (profile_max_steps is None or gravity_readback not in ('packed','fused'))):
        raise ContractError('Gravity parity is only a short packed-readback diagnostic')
    profiler=None
    if profile_max_steps is not None:
        from physics.episode_profile import EpisodeStepProfile
        import quadrants as profile_quadrants
        import torch
        def profile_fence():
            profile_quadrants.sync()
            if torch.cuda.is_initialized():torch.cuda.synchronize()
        profiler=EpisodeStepProfile(profile_max_steps,synchronize=profile_synchronize,fence=profile_fence)
    elif profile_synchronize:
        raise ContractError('Profiling fences require a short profiling limit')
    observation_reader=None
    particle_request=None;particle_reader=None;particle_archive=None;particle_contact_copy=None
    particle_costs=Counter()
    if empty_mode:
        raise ContractError('This hardware package supports the fixed four-bead tiers only')
    if 'particle_material_request' in world:
        from physics import particle_episode_bridge as particle_bridge
        from physics.particle_material import ParticleMaterialReader
        from physics.particle_archive import ParticleArchive
        from physics.particle_episode_state import packet_from_reader_result,validate_particle_packet,window_rest
        from physics.particle_contact_coverage import (build_particle_contact_coverage,require_particle_contact_coverage,
            particle_contact_coverage_stamp,validate_particle_contact_coverage)
        raise ContractError('This hardware package supports the fixed four-bead tiers only')
        raise ContractError('This hardware package supports the fixed four-bead tiers only')
        raise ContractError('This hardware package supports the fixed four-bead tiers only')
        raise ContractError('This hardware package supports the fixed four-bead tiers only')
        particle_request=particle_bridge.validate_request(world,scenario)
        if student_driver is None or observation_readback!='batched':
            raise ContractError('Native particles require the explicit measured student executor')
    if (type(particle_cpic_diagnostic) is not bool or particle_cpic_diagnostic and
            (particle_request is None or particle_request['material_kind']!='sand')):
        raise ContractError('Post-outer CPIC readback is an explicit Sand diagnostic only')
    if (type(particle_world_quality_diagnostic) is not bool or particle_world_quality_diagnostic and
            (particle_request is None or particle_cpic_diagnostic or student_driver is None
                or type(getattr(student_driver,'step_limit',None)) is not int or student_driver.step_limit!=1
                or getattr(student_driver,'mode',None)!='mission'
                or not Path(output).resolve().is_relative_to(Path(__file__).resolve().parents[1]/'qa/all_material_physics_20261031'))):
        raise ContractError('World-quality diagnostic is an explicit one-step particle pipeline into a fresh QA output only')
    private_contact_fixture=False
    if student_driver is not None:
        from physics.student_episode_driver import StudentEpisodeDriver
        if type(student_driver) is not StudentEpisodeDriver or any(v is not None for v in
                (approach,transport_plan,grasp_reference,continuation_phases,continuation_planner,global_deadline_s)) or repeated_three_arm_mode:
            raise ContractError('Live student driver owns all motion planning and cannot mix frozen reference paths')
        if len(scenario['robots'])!=3 or observation_readback!='batched':
            raise ContractError('Live student execution requires three arms and the measured observation reader')
        if profile_max_steps is not None:
            if not (rigid_profile=='rigid16_bead_tc2h_v1' and type(profile_max_steps) is int
                    and 1<=profile_max_steps<=160 and student_driver.step_limit==profile_max_steps):
                raise ContractError('Student profiler is restricted to the bounded rigid16 initial settle before any planning')
        if student_driver.private_contact_profile:
            student_driver.require_private_contact_profile(world)
            if profile_native_timing is None:
                raise ContractError('Explicit private contact fixture requires its source-bound observer')
            private_contact_fixture=True
    automatic_grasp_mode=student_driver is not None and getattr(student_driver,'grasp_policy','friction')=='automatic'
    automatic_grasp=None
    automatic_boundary_events=[]
    automatic_initialization=None
    if automatic_grasp_mode:
        if private_contact_fixture:raise ContractError('Automatic grasp cannot be mixed with the legacy friction contact fixture')
        from physics.automatic_runtime_binding import prepare_automatic_scenario
        scenario,automatic_initialization=prepare_automatic_scenario(scenario)
    model,trajectory,settings=validate_inputs(world,scenario,approach,physics_reference,
        **({'student_limits':student_driver.limits} if student_driver is not None else {}))
    if (transport_plan is None)!=(grasp_reference is None):raise ContractError('Carry plan and exact grasp reference are supplied together')
    mode_kwargs={'repeated_three_arm_mode':True} if repeated_three_arm_mode else {}
    transport=None if transport_plan is None else validate_transport_inputs(world,scenario,approach,physics_reference,transport_plan,grasp_reference,**mode_kwargs)
    tilt_limits=reference_body_tilt_limits(transport_plan,reference_execution_body_tilt_limit_deg)
    execution_body_tilt_limit_deg=(None if transport_plan is None else transport_plan['query_body_tilt_limit_deg']) if tilt_limits is None else tilt_limits['execution_hard_limit_deg']
    if repeated_three_arm_mode and (continuation_phases is None or len(scenario['robots'])!=3):
        raise ContractError('Repeated mode requires three persistent arms and six explicit continuation stages')
    if continuation_phases is not None or continuation_planner is not None or global_deadline_s is not None:
        if len(scenario['robots'])!=2 and not repeated_three_arm_mode:
            raise ContractError('Three-arm repeated handoff integration is unfinished. This runner currently permits its source-grasp stage only.')
        if transport is None or observation_readback=='scalar':
            raise ContractError('Explicit continuation requires the released-relay prefix and a bound packed observation reader')
    output=Path(output);case=scenario['case'];active=scenario['active_arm_id'];execution=settings['execution_limits']
    compound_groups,visual=runtime_scene_options(world,scenario)
    embedded_supports=validate_embedded_support_components(case,scenario['support_stations'])
    scene=None;camera=None;recording=False;stream=None;compact_log=None;steps=0;pending=None;dispatcher=None;observation=None
    rigid16_observer=None;rigid16_samples=[];rigid16_transfer_reader=None
    result={'schema_version':('persistent-three-arm-approach-v1' if len(scenario['robots'])==3 else 'persistent-two-arm-approach-v1'),'status':'NOT_COMPLETED','physical_steps':0,
        'world_sha256':digest(world),'scenario_sha256':digest(scenario),'external_trajectory_sha256':digest(trajectory.to_record()),
        'source_bindings':copy.deepcopy(source_bindings),'protocol':copy.deepcopy(PROTOCOL),
        'physics_reference_scope':'Measured ninth-run settings reused, not old native IDs or old grasp observation.',
        'offline_query_replanned_live':False,'simulation_dt_s':DT,'physical_handoff_verified':False,
        'load_bearing_grasp_verified':False,'material_delivery_accepted':False,'full_mission_passed':False,
        'student_release_approved':False,'continuous_collision_certificate':False,
        'post_initialization_pose_setters':0,'runtime_welds':0,'refills':0,'intermediate_ik_calls':0,'cleanup_errors':[]}
    if automatic_grasp_mode:
        result['provided_grasp_model']='provided_ideal'
        result['automatic_grasp_initialization']=automatic_initialization
        result['legacy_grasp_protocol_not_executed']=result['protocol']
        result['protocol']={'schema_version':'provided-automatic-grasp-protocol-v1','initial_settle_s':PROTOCOL['initial_settle_s'],
            'approach_joint_settle_s':PROTOCOL['approach_settle_s'],'finger_q_m':.04,'finger_representation':'Authored0.04; native_snapshot_command casts once to actual native dtype/bounds and records exact readback. No clipping.', 'friction_close_or_preload':False,
            'attach':'Actual supplied endpoint then zero-step native weld event',
            'release':'Actual supported/rested placement then zero-step native release event'}
    if tilt_limits is not None:result['reference_body_tilt_limits']=tilt_limits
    if rigid_profile is not None:
        result['rigid_runtime_profile']=rigid_profile
        result['payload_mode']='empty' if empty_mode else 'native_particles' if particle_request is not None else 'four_beads'
        result['physical_profile']=EMPTY_RIGID_PROFILE if empty_mode else rigid_profile
        result['physics_reference_scope']='Byte-preserved old ninth-run provenance binds controller/rest only. Rigid16/TC2h changes native numerical physics and is not established by that historical PASS. All16 and fresh model/mission verification remain required.'
    if gravity_readback!='scalar':result['gravity_readback']=gravity_readback
    if automatic_performance_selection:
        result['performance_readback_selection']={'gravity':gravity_readback,'batched_motor_control':batched_motor_control,
            'scope':'Validated source-pinned native Jacobian collection and public motor batching are selected for the three-arm zero-copy CUDA backend. Other conditions retain scalar behavior. Physical checks and settings are unchanged.'}
    if profile_gravity_parity_every:
        result['profile_gravity_parity']={'interval_steps':profile_gravity_parity_every,'pairs':[],
            'scope':'Exact same-clock optimized/scalar generalized-effort and model-evidence comparison. Fused mode also compares every Jacobian entry. Not a full mission.'}
    if batched_motor_control:
        result['batched_motor_control']=True
        if profile_gravity_parity_every:result['profile_motor_parity']=[]
    if student_driver is not None:
        result.pop('external_trajectory_sha256',None)
        result.update(schema_version='student-three-arm-live-execution-v1',student_execution=student_driver.record())
    if particle_request is not None:
        result.update(schema_version='student-native-particle-three-arm-execution-v1',
            particle_material_request=copy.deepcopy(particle_request),
            particle_control_readback_scope='Automatic control uses the existing public per-arm calls. The rigid-only batched motor/fused Jacobian acceptance has not been extended to private particle couplers.',
            physics_reference_scope='Old ninth-run reference binds the controller/rest policy only. Native particle physics, 100 substeps and nine cup contacts are separately recorded. No material or transport acceptance follows from source equality.')
    if trajectory.duration_s==0. and student_driver is None:
        result['zero_motion_approach']={'external_waypoint_count':1,'external_duration_s':0.,
            'action_id':'external_approach','role':'approach','kind':'hold','duration_s':.5,
            'command_q_rad':list(trajectory.waypoints[0]),
            'scope':'Explicit initial command held for 0.5 simulation seconds. No additional joint pose or approach path is inserted.'}
    if not record_video:
        result['image_capture']={'mode':'actual_still_checkpoints','render_enabled':bool(render),
            'video_recorded':False,'periodic_video_rendering':False,'checkpoints':[]}
    if detail_images:
        result['detail_images']={'enabled':bool(render) and not record_video,'maximum_images':3,
            'captures':[],'disabled':False,'media_qa_passed':False,
            'camera_source_sha256':DETAIL_CAMERA_SOURCE_SHA256,
            'scope':'Optional actual released-stage PNGs. No physics steps or image-based acceptance. Visual quality has not been reviewed.',
            'skip_reason':None if render and not record_video else 'Requires rendering enabled and video recording disabled.'}
        result['detail_image_errors']=[]
    if transport is not None:
        result.update(schema_version='persistent-two-arm-relay-v1',transport_protocol=copy.deepcopy(TRANSPORT_PROTOCOL),
            transport_plan_sha256=transport['transport_plan_sha256'],grasp_reference_sha256=transport['grasp_reference_sha256'],
            relay_placement_verified=False,query_contents_model='Measured body-local sphere proxy. Actual contents remain free dynamic bodies.')
    if repeated_three_arm_mode:
        result.update(schema_version='persistent-three-arm-repeated-handoff-v1',repeated_three_arm_mode=True,
            repeated_mode_scope='Instructor reference: one persistent four-bead batch, two supported handoffs and fixed-goal delivery. Not full project or student release approval.')
    def save(name,value):
        formatting=({'separators':(',',':')} if compact_setup_logs and name in ('built_geometry.json','live_contact_coverage.json') else {'indent':2})
        (output/name).write_text(json.dumps(_json_safe(value),ensure_ascii=False,allow_nan=False,**formatting)+'\n',encoding='utf-8')
    if compact_setup_logs:
        result['compact_setup_logs']={'files':['built_geometry.json','live_contact_coverage.json'],
            'parsed_fields_preserved':True,'per_step_trace_format_and_flush_unchanged':True}
    def vector(value,n):
        x=np.asarray(array(value),dtype=float).copy()
        if x.shape!=(n,) or not np.isfinite(x).all():raise ContractError('Nonfinite/wrong-shaped native observation')
        return x
    def pose(entity):return {'frame':'world','position_m':vector(entity.get_pos(relative=False),3).tolist(),
                             'quaternion_wxyz':vector(entity.get_quat(relative=False),4).tolist()}
    started=time.monotonic();recent=[];counts=Counter();boundaries=[]
    setup_mark('validate_inputs_and_options')
    try:
        installed,_=verify_contact_sources()
        import hashlib
        if detail_images and result['detail_images']['enabled']:
            try:
                if hashlib.sha256((installed/'vis/camera.py').read_bytes()).hexdigest()!=DETAIL_CAMERA_SOURCE_SHA256:
                    raise ValueError('Camera source differs from the reviewed public API')
            except Exception as exc:
                result['detail_images']['disabled']=True
                result['detail_image_errors'].append({'stage':'source','type':type(exc).__name__,'message':str(exc)})
        for path,expected in EQUALITY_LAYOUT_SOURCES.items():
            if hashlib.sha256((installed/path).read_bytes()).hexdigest()!=expected:
                raise ContractError('Native equality layout source changed: '+path)
        if compound_groups or visual is not None:
            for path,expected in SCENE_PRESENTATION_SOURCES.items():
                if hashlib.sha256((installed/path).read_bytes()).hexdigest()!=expected:
                    raise ContractError('Nominal scene/visual source changed: '+path)
        if hashlib.sha256((installed/'assets'/case['robot_model']).read_bytes()).hexdigest()!=world['robot_model_sha256']:
            raise ContractError('Installed robot differs from the fixed world model')
        setup_mark('installed_source_checks')
        vis_settings={'ambient_light':(.5,.5,.5)}
        if visual is not None:vis_settings.update(shadow=False,background_color=(1.,1.,1.))
        def appearance(color):return {} if color is None else {'surface':gs.surfaces.Default(color=_rgb(color))}
        sim_arguments=(dict(sim_options=(gs.options.SimOptions(dt=DT,substeps=16,substeps_local=1,gravity=(0,0,-9.81))
            if rigid_profile is not None else gs.options.SimOptions(dt=DT,substeps=1,gravity=(0,0,-9.81))))
            if particle_request is None else particle_bridge.scene_options(gs,world))
        scene=gs.Scene(**sim_arguments,
            rigid_options=(gs.options.RigidOptions(enable_self_collision=True,enable_neutral_collision=True,
                enable_adjacent_collision=False,use_hibernation=False,noslip_iterations=5,enable_torsional_friction=True,
                enable_rolling_friction=(particle_request is None and not empty_mode),contact_pruning_tolerance=None,
                **({} if particle_request is None else particle_bridge.rigid_option_overrides(world))) if rigid_constraint_solver is None else gs.options.RigidOptions(enable_self_collision=True,enable_neutral_collision=True,
                enable_adjacent_collision=False,use_hibernation=False,noslip_iterations=5,enable_torsional_friction=True,
                enable_rolling_friction=(particle_request is None and not empty_mode),contact_pruning_tolerance=None,
                **({} if particle_request is None else particle_bridge.rigid_option_overrides(world)),constraint_solver=gs.constraint_solver.CG)),
            vis_options=gs.options.VisOptions(**vis_settings),renderer=gs.renderers.Rasterizer(),show_viewer=False)
        if particle_request is not None:particle_bridge.install_empty_scene(scene,world)
        if profile_native_timing is not None:
            if private_contact_fixture:
                import inspect
                raise ContractError('This hardware package supports the fixed four-bead tiers only')
                expected=Path(__file__).resolve().parents[1]/'qa/all_material_physics_20261031/cache_runtime_branch/closed_grip_fixture_hook_v1/contact_profile_observer.py'
                if (type(profile_native_timing) is not private_observer.ContactProfileObserver
                        or Path(private_observer.__file__).resolve()!=expected
                        or Path(inspect.getsourcefile(type(profile_native_timing))).resolve()!=expected
                        or hashlib.sha256(expected.read_bytes()).hexdigest()!='7a115b166c9e19ea2f373fea2f57f5e48c005472e882838642692b4b841d34fc'):
                    raise ContractError('Private contact observer exact class/source/hash differs')
                profile_native_timing.validate_source_binding()
            else:
                if profile_max_steps is None:raise ContractError('Native timing requires a short profile')
            profile_native_timing.install_scene(scene)
        floor=scene.add_entity(gs.morphs.Plane(**physical_display),name='floor',**appearance(None if visual is None else visual['floor_color']));arms={};sensors={}
        for record in scenario['robots']:
            b=record['base_placement'];name=record['id']
            if single_active_arm is not None and name!=single_active_arm:
                from physics.fixed_peer_arms import write_fixed_peer_asset
                q9=[float(np.float32(v)) for v in [*record['q_rad'],*record['finger_q_m']]]
                if fixed_reference_record['arms'][name]['native_q']!=q9:
                    raise ContractError('Fixed peer supplied configuration differs from the native INITIAL reference')
                descriptor=write_fixed_peer_asset(installed/'assets'/case['robot_model'],q9,output/'assets'/('fixed_'+name+'.xml'),
                    base_position=episode_base_position(record,case),base_quaternion=[math.cos(b['yaw_rad']/2),0.,0.,math.sin(b['yaw_rad']/2)])
                fixed_peer_descriptors[name]=descriptor
                fixed_peers[name]=scene.add_entity(gs.morphs.MJCF(file=descriptor['path'],pos=episode_base_position(record,case),
                    euler=(0,0,math.degrees(b['yaw_rad'])),requires_jac_and_IK=False,
                    decompose_object_error_threshold=float('inf'),decompose_robot_error_threshold=float('inf'),**physical_display),name=name)
                continue
            arm=scene.add_entity(gs.morphs.MJCF(file=case['robot_model'],pos=episode_base_position(record,case),
                euler=(0,0,math.degrees(b['yaw_rad'])),requires_jac_and_IK=True,**physical_display),name=name)
            arms[name]=arm;sensors[name]=scene.add_sensor(gs.sensors.JointTorque(entity_idx=arm.idx,dofs_idx_local=list(range(9)),delay=0.,noise=0.,jitter=0.))
        support_colors=([None]*len(scenario['support_stations']) if visual is None else
            [visual['support_colors'].get(s['id']) for s in scenario['support_stations']]
            if visual.get('appearance_source')=='case.query_visual' else
            [visual.get('quality_color',visual['object_color']) if k=='quality' else visual[k+'_color']
             for k in scene_support_roles(world,scenario)])
        supports=[scene.add_entity(gs.morphs.Box(size=s['size_m'],pos=s['center_world_m'],fixed=True,**physical_display),name=s['id'],
            **appearance(c)) for s,c in zip(scenario['support_stations'],support_colors) if s['id'] not in embedded_supports]
        obstacles,compound_assets=add_nominal_obstacles(gs,scene,case,output/'assets',compound_groups,visual,headless_visuals=headless_visuals)
        material_presentation=None;material_colors=None
        shared_display=visual is not None and ('mount_regions' in world or 'relay_regions' in world)
        if particle_request is not None:
            tray,particle_entity,particle_provider,particle_construction=particle_bridge.add_material(gs,scene,world,output/'assets/native_material')
            beads=[]
            save('native_particle_construction.json',particle_construction)
        else:
            asset=write_carried_vessel_urdf(output/'assets',model)
        if shared_display and particle_request is None:
            from physics.episode_presentation import write_material_vessel_urdf,material_palette
            asset,material_presentation=write_material_vessel_urdf(output/'assets',asset,model,'balls')
            material_colors=material_palette('balls')
        if particle_request is None:
            asset_components=validate_asset_components(asset,model)
            p=scenario['body_pose']
            tray=scene.add_entity(gs.morphs.URDF(file=str(asset.path),pos=p['position_m'],quat=p['quaternion_wxyz'],
                fixed=False,align=False,merge_fixed_links=True,**physical_display,**({'prioritize_urdf_material':True} if shared_display else {})),material=gs.materials.Rigid(friction=1.,needs_coup=True,coup_friction=.4,friction_torsional=.001),name='tray_0',
                **appearance(None if visual is None or shared_display else visual['object_color']))
            beads=[scene.add_entity(gs.morphs.Sphere(radius=c['radius_m'],pos=c['center_world_m'],**physical_display),
                material=gs.materials.Rigid(rho=600.,friction=.4),name=f'tray_0_ball_{i}',
                **appearance(None if material_colors is None else material_colors.content_hex)) for i,c in enumerate(scenario['contents'])]
        entities=unique_scene_entities([floor,*arms.values(),*fixed_peers.values(),*supports,*obstacles,tray,*beads])
        paints,paint_specs=add_visual_floor_patches(gs,scene,world,visual)
        if rigid_profile is not None:
            from physics.contact_accuracy import rigid16_contact_request,prepare_rigid16_geometry,rigid16_built_evidence
            accuracy_before=prepare_rigid16_geometry(entities,rigid16_contact_request(),floor=floor,tray=tray,beads=beads,
                box_type=gs.GEOM_TYPE.BOX,sphere_type=gs.GEOM_TYPE.SPHERE,plane_type=gs.GEOM_TYPE.PLANE)
        else:
            accuracy_before=(prepare_geometry_parameters(entities,contact_accuracy_request(),box_type=gs.GEOM_TYPE.BOX,sphere_type=gs.GEOM_TYPE.SPHERE)
                if particle_request is None and not empty_mode else prepare_cup_contacts(tray,other_entities=[e for e in entities if e is not tray]))
        equality_before=prepare_equalities(entities,arms,gs.EQUALITY_TYPE.JOINT)
        camera_options={'res':(960,720),'pos':(-1.6,-2.0,2.2),'lookat':(.35,.42,.45),'fov':50,'GUI':False}
        if visual is not None and visual.get('camera_proposals'):
            c=visual['camera_proposals'][0];camera_options.update(pos=c['position_m'],lookat=c['lookat_m'],fov=c['fov_deg'])
            if shared_display:
                from physics.episode_presentation import framed_camera_options
                camera_options.update(framed_camera_options(c))
        if render:camera=scene.add_camera(**camera_options)
        setup_mark('construct_scene_and_assets')
        build_started=time.monotonic();scene.build();result['build_wall_s']=time.monotonic()-build_started
        setup_mark('scene_build')
        result['headless_visual_geometry']={'omitted_from_physical_entities':headless_visuals,
            'render_enabled':render,'collision_geometry_count':sum(e.n_geoms for e in entities),
            'physical_entity_visual_geometry_count':sum(e.n_vgeoms for e in entities),
            'visual_only_patch_count':len(paints),
            'scope':'Public Morph.visualization=False only. Collision/mass/inertia/links/joints and visual-only patch entity identities retained. Rendering-enabled execution retains original visuals.'}
        if any(e.n_geoms!=0 or len(e.geoms)!=0 for e in paints):raise ContractError('Built display patch has collision geometry')
        accuracy=(rigid16_built_evidence(scene,entities,accuracy_before,rigid16_contact_request()) if rigid_profile is not None else
            built_evidence(scene,entities,accuracy_before,contact_accuracy_request())
            if particle_request is None and not empty_mode else native_cup_contacts(scene,tray,accuracy_before))
        constraints=native_equalities(scene,entities,equality_before,output/'constraint_identity_diagnostic.json',
            **({'expected_substeps':16,'rigid_profile':rigid_profile} if rigid_profile is not None else
                {} if particle_request is None else {'expected_substeps':100}))
        solver=solver_configuration_evidence(scene.rigid_solver,5,gs.friction_cone.pyramidal)
        torsion=torsional_configuration_evidence(scene.rigid_solver,entities,True,array,
            expected_rolling=(particle_request is None and not empty_mode))
        tray_material=tray_torsional_radius_evidence(torsion,.001)
        requested_accuracy=(rigid16_contact_request() if rigid_profile is not None else contact_accuracy_request() if particle_request is None and not empty_mode else
            {'schema_version':'empty-cup-contact-request-v1' if empty_mode else 'native-particle-cup-contact-request-v1','dt_s':DT,'substeps':1 if empty_mode else 100,
             'geometry_timeconst_s':.005,'target_entities':[tray.name],
             'selected_geometry_count':9,'native_geom_ids':[r['geom_idx'] for r in accuracy['rows']],
             'scope':'Nine actual vessel boxes. No rigid bead geometry is present.'})
        configuration={'requested':{'noslip_iterations':5,'torsional_friction':True,'tray_torsional_radius_m':.001,
            'contact_accuracy':requested_accuracy,'finger_equality_timeconst_s_by_arm':{n:.01 for n in arms}},
            'resolved_solver':solver,'torsion':torsion,'tray_material':tray_material,'constraints':constraints,'contact_accuracy':accuracy,
            'direct_contact_spin_moment':'NOT_COLLECTED','teacher_physical_setting_approved':False}
        if particle_request is None and not empty_mode:
            configuration['rolling_resistance_model']={
                'schema_version':'rigid-bead-native-rolling-candidate-v1','enabled':True,
                'coefficient_source':'Installed public Rigid material fallback: genesis.utils.geom.default_friction_rolling',
                'bead_default_friction_rolling_m':.0001,
                'actual_coefficients':'torsion.geometries public/native coefficient ledger',
                'scope':'Global eligible rigid contacts. No coefficient, damping, force or velocity setter is added.',
                'legacy_physics_equivalence':False,'physical_validation':'NOT_ESTABLISHED'}
        if empty_mode:
            configuration['physical_profile']={'name':EMPTY_RIGID_PROFILE,'payload_mode':'empty',
                'dt_s':DT,'substeps':1,'substeps_local':1,'cup_nine_TC_s':.005,
                'rolling_friction':False,'geometry_scope':'Canonical cup9 only. Every other native geometry parameter and collision filter is preserved.',
                'source_scope':'Existing empty automatic-grasp proof profile. New user-trajectory execution needs its own native evidence.'}
        if rigid_profile is not None:
            if (int(scene.sim.substeps)!=16 or int(scene.sim.substeps_local)!=1 or float(scene.sim.dt)!=DT
                    or float(scene.rigid_solver.substep_dt)!=DT/16):
                raise ContractError('Rigid16 actual constructed native clock differs')
            configuration['rigid_runtime_profile']={'name':rigid_profile,'outer_dt_s':DT,'substeps':16,
                'substeps_local':1,'native_substep_dt_s':float(scene.rigid_solver.substep_dt),
                'floor_and_four_bead_TC_s':.0003125,'cup_nine_TC_s':.005,
                'old_reference_scope':'Controller/rest provenance only; not native physics equivalence or approval.',
                'native_integrated_validation':'NOT_ESTABLISHED'}
        save('solver_configuration.json',configuration);result['configuration_sha256']=digest(configuration)
        setup_mark('verify_native_physical_settings')
        kp=np.asarray(settings['controller']['kp']);kv=np.asarray(settings['controller']['kv']);effort=np.asarray(settings['controller']['effort_limits'])
        limits_by_arm={};finger_bounds={};pads={};hands={};fingers={};gravity={};initialization={};gravity_parity={}
        for record in scenario['robots']:
            name=record['id']
            if name not in arms:continue
            arm=arms[name]
            command,representation=native_snapshot_command(record,*arm.get_dofs_limit());arm.set_qpos(command)
            if vector(arm.get_qpos(),9).tolist()!=command.tolist():raise ContractError('Initial native q readback differs')
            b=record['base_placement'];expected_root=episode_base_position(record,case)
            expected_quaternion=[math.cos(b['yaw_rad']/2),0.,0.,math.sin(b['yaw_rad']/2)]
            base_validation=validate_native_base_position(expected_root,arm.get_pos(relative=False),arm_id=name)
            actual_root=pose(arm)
            if quaternion_error(actual_root['quaternion_wxyz'],expected_quaternion)>1e-6:
                raise ContractError('Live native base differs from named installation')
            representation['native_base_world']=actual_root
            if 'base_z_m' in record:representation['base_position_validation']=base_validation
            if arm.n_dofs!=9 or [i for n in case['arm_joint_names'] for i in arm.get_joint(n).dofs_idx_local]!=list(range(7)):
                raise ContractError('Native joint order changed')
            bounds=list(zip(representation['native_lower'],representation['native_upper']))
            limits_by_arm[name]=JointLimits(bounds[:7],trajectory.limits.velocity_rad_s,trajectory.limits.acceleration_rad_s2)
            if limits_by_arm[name]!=trajectory.limits:raise ContractError('Native robot limits differ from external trajectory')
            finger_bounds[name]=bounds[7:];initialization[name]=representation
            hands[name]=arm.get_link(case['tool_link']);fingers[name]={n:arm.get_link(n) for n in ('left_finger','right_finger')}
            pads[name]={p['geom_id']:p for p in match_native_pads(arm,read_panda_pad_primitives(installed/'assets'/case['robot_model']),gs.GEOM_TYPE.BOX)}
            arm.set_dofs_kp(kp);arm.set_dofs_kv(kv);arm.set_dofs_force_range(-effort,effort)
            if arm.gravity_compensation!=0.:raise ContractError('Physical gravity must remain enabled')
            gravity[name]=(GravityFeedforward(arm,(0,0,-9.81)) if gravity_readback=='scalar'
                else GravityFeedforward(arm,(0,0,-9.81),readback=gravity_readback))
            if profile_gravity_parity_every:
                gravity_parity[name]=GravityFeedforward(arm,(0,0,-9.81))
                if gravity_parity[name].model_evidence.model_sha256!=gravity[name].model_evidence.model_sha256:
                    raise ContractError('Packed/scalar gravity model snapshots differ')
        if single_active_arm is not None:
            from physics.fixed_peer_arms import qualify_native_fixed_peers,validated_single_active_contract
            fixed_peer_ledger=qualify_native_fixed_peers(scene,arms,fixed_peers,fixed_peer_descriptors,
                active_arm_id=single_active_arm,reference_native_ledgers=fixed_reference_record['arms'])
            single_active_contract=validated_single_active_contract(scene._ap_single_active_arm_contract)
            single_active_kwargs={'single_active_contract':single_active_contract}
            from functools import partial
            supported_grasp_gate=partial(supported_grasp_gate,**single_active_kwargs)
            reference_agreement=partial(reference_agreement,**single_active_kwargs)
            released_relay_gate=partial(released_relay_gate,**single_active_kwargs)
            classify_episode_live_contacts=partial(classify_episode_live_contacts,**single_active_kwargs)
            fixed_peer_hands={name:peer.get_link(case['tool_link']) for name,peer in fixed_peers.items()}
            fixed_peer_fingers={name:{finger:peer.get_link(finger) for finger in ('left_finger','right_finger')} for name,peer in fixed_peers.items()}
            save('fixed_peer_geometry_ledger.json',fixed_peer_ledger);save('single_active_arm_contract.json',single_active_contract)
            result['single_active_arm_contract']=copy.deepcopy(single_active_contract)
            result['fixed_peer_reference']={'path':str(fixed_peer_reference),'sha256':hashlib.sha256(fixed_peer_reference.read_bytes()).hexdigest()}
        motor_batch=None
        if batched_motor_control:
            from physics.episode_motor_batch import BatchedEpisodeMotors
            motor_batch=BatchedEpisodeMotors(scene,arms,kp,kv)
        if gravity_readback=='fused':
            result['gravity_jacobian_backend']={name:helper.fused_backend_configuration() for name,helper in gravity.items()}
        setup_mark('initialize_arms_controllers_gravity')
        components={c['geom_id']:c for c in (validate_built_components(tray,model,asset_components,gs.GEOM_TYPE.BOX)
            if particle_request is None else particle_bridge.native_component_records(tray,model,particle_construction,gs.GEOM_TYPE.BOX))}
        obstacle_native=nominal_obstacle_ledger(gs,obstacles,case,compound_assets)
        measured_static_aabbs=None
        if compound_assets and reuse_static_aabbs is not False:
            from physics.episode_joint_scene import MeasuredStaticAABBs
            measured_static_aabbs=MeasuredStaticAABBs(scene,obstacle_native,box_type=gs.GEOM_TYPE.BOX)
        setup_mark('verify_tray_and_nominal_obstacles')
        station_map,support_entities=runtime_support_registry(gs,supports,scenario['support_stations'],obstacles,compound_assets,case,
            native_components=obstacle_native)
        sphere_geometry=(native_bead_geometry(scene.rigid_solver,beads,.012,gs.GEOM_TYPE.SPHERE,array) if particle_request is None else [])
        ids=[]
        for index,(bead,c) in enumerate(zip(beads,world['initial_contents'])):
            mass=float(array(bead.get_mass()).sum())
            if mass!=c['mass_kg'] or np.max(np.abs(vector(bead.get_pos(relative=False),3)-scenario['contents'][index]['center_world_m']))>1e-7:
                raise ContractError('Native initial content mass/position differs from fixed supply')
            ids.append({'id':c['id'],'entity':bead.name,'entity_idx':int(bead.idx),'mass_kg':mass,'radius_m':c['radius_m']})
        if (np.max(np.abs(np.asarray(pose(tray)['position_m'])-scenario['body_pose']['position_m']))>1e-7
                or quaternion_error(pose(tray)['quaternion_wxyz'],scenario['body_pose']['quaternion_wxyz'])>1e-6):
            raise ContractError('Native source tray pose changed')
        owners={}
        for kind,group in (('floor',[floor]),('arm',list(arms.values())),('support',supports),('obstacle',obstacles),('tray',[tray]),('bead',beads)):
            for entity in group:
                for g in entity.geoms:owners[int(g.idx)]={'kind':kind,'entity':entity.name,'link':g.link.name,'link_id':int(g.link.idx)}
        for name,entity in fixed_peers.items():
            for g in entity.geoms:
                owners[int(g.idx)]={'kind':'obstacle','entity':entity.name,'link':g.link.name,'link_id':int(g.link.idx),
                    'fixed_peer_arm_id':name,'original_robot_link_name':g.link.name,'fixed_peer_ledger_sha256':single_active_contract['fixed_peer_ledger_sha256']}
        if compound_assets:
            for part in obstacle_native:
                owner=owners[part['geom_id']]
                if owner['kind']!='obstacle' or owner['entity']!=part['entity'] or owner['link_id']!=part['link_id']:
                    raise ContractError('Native compound ownership differs from exact component ledger')
                owner.update(component_id=part['component_id'],assembly_id=part['assembly_id'])
        for gid,station in station_map.items():
            if station.get('embedded_in_rigid_assembly') is True:
                owner=owners[gid]
                if (owner['entity']!=station['assembly_id'] or owner['component_id']!=station['component_id']
                        or owner['link_id']!=station['link_id']):raise ContractError('Embedded physical support ownership changed')
                owner.update(kind='support',station_id=station['id'],geom_id=gid)
        setup_mark('verify_supports_contents_and_ownership')
        scene_evidence={}
        if compound_groups or visual is not None:
            scene_evidence={'compound_groups':copy.deepcopy(compound_groups),'compound_assets':copy.deepcopy(compound_assets),
                'nominal_geometry_only':True,'planning_clearance_applied_m':0.,'source_sha256':dict(SCENE_PRESENTATION_SOURCES),
                'visual_contract':copy.deepcopy(visual),'camera_options':camera_options,
                'shared_region_display':shared_display,'material_presentation':material_presentation,
                'material_selection_scope':('Empty vessel. No content entities are constructed.' if empty_mode else 'Four actual sphere contents use balls palette. No water/sand physics selected by color.') if shared_display else None,
                'paint_patches':[{**s,'native_entity_idx':int(e.idx),'native_collision_geom_count':int(e.n_geoms)} for e,s in zip(paints,paint_specs)],
                'scope':'Explicit nominal compound links and appearance only. Existing physical/controller/contact gates are unchanged.'}
            if embedded_supports:scene_evidence['embedded_support_components']=copy.deepcopy(embedded_supports)
            save('scene_representation.json',scene_evidence);result['scene_representation_sha256']=digest(scene_evidence)
        # Public AABBs only. This does not recompute live contacts or step.
        static_audit=(audit_episode_pairs(scene.rigid_solver,list(arms.values()),owners)
            if measured_static_aabbs is None else audit_episode_pairs(scene.rigid_solver,list(arms.values()),owners,
                measured_static_aabbs=measured_static_aabbs))
        if measured_static_aabbs is not None:result['static_aabb_readback_reuse']=measured_static_aabbs.record()
        setup_mark('static_pair_audit_and_scene_evidence')
        if particle_request is None:
            coverage=build_live_contact_coverage(scene,arms=list(arms.values()),owners=owners,readback='batched',
                **single_active_kwargs,**({'payload_mode':'empty'} if empty_mode else {} if rigid_profile is None else {'runtime_profile':rigid_profile}));require_live_contact_coverage(coverage)
        else:
            particle_profile=('cpic_cache_mpm' if particle_request.get('runtime_profile')==particle_bridge.CACHE_RUNTIME_PROFILE
                else 'sparse_mpm' if particle_request['material_kind']=='sand' else 'process_liquid')
            coverage=build_particle_contact_coverage(scene,arms=list(arms.values()),owners=owners,profile=particle_profile,readback='batched')
            require_particle_contact_coverage(coverage)
            reader_geometry,union_binding=particle_bridge.canonical_reader_geometry(model,particle_construction)
            save('native_particle_union_binding.json',union_binding)
            particle_reader=ParticleMaterialReader(scene,particle_entity,material_kind=particle_request['material_kind'],
                source_id=particle_request['source_id'],geometry=reader_geometry,
                extra_source_pins=particle_bridge.provider_source_pins(particle_request['material_kind'],
                    particle_request.get('runtime_profile','default')))
            world,scenario=particle_bridge.bind_native_contract(world,scenario,particle_reader.contract)
            save('live_world.json',world);save('live_initialization.json',scenario)
            result.update(world_sha256=digest(world),scenario_sha256=digest(scenario),native_particle_contract_sha256=particle_reader.contract_sha256)
            particle_archive=ParticleArchive(output/'particle_archive',particle_reader.contract,outer_dt_s=DT)
        setup_mark('live_contact_coverage_build')
        save('live_contact_coverage.json',coverage)
        save('built_geometry.json',{'owners':owners,'components':components,'pads_by_arm':pads,'stations_by_geom':station_map,
            'nominal_obstacles':obstacle_native,'spheres':sphere_geometry,'content_ids':ids,'initialization':initialization,
            'static_pair_audit':static_audit,'gravity_models':{n:g.model_evidence.to_record() for n,g in gravity.items()},
            **({'scene_representation':scene_evidence} if scene_evidence else {})})
        setup_mark('save_geometry_and_coverage_logs')
        def observe_scalar():
            body=pose(tray);xyz=np.asarray([vector(b.get_pos(relative=False),3) for b in beads]).reshape((-1,3));vel=np.asarray([vector(b.get_vel(),3) for b in beads]).reshape((-1,3))
            local=(xyz-body['position_m'])@rotation_matrix_wxyz(body['quaternion_wxyz'])
            inside=interior_membership(local,model.geometry_record(),.012,.001)
            return {'simulation_s':float(scene.cur_t),'native_scene_step':int(scene.t),
                'arms':{n:{'q_all_dofs':vector(a.get_qpos(),9).tolist(),'qd_all_dofs':vector(a.get_dofs_velocity(),9).tolist(),
                    'hand_world':pose(hands[n]),'finger_world':{k:pose(v) for k,v in fingers[n].items()}} for n,a in arms.items()},
                'body_world':body,'body_linear_velocity_m_s':vector(tray.get_vel(),3).tolist(),'body_angular_velocity_rad_s':vector(tray.get_ang(),3).tolist(),
                'body_up_tilt_deg':body_tilt_deg(body['quaternion_wxyz']),
                'materials':[{'id':r['id'],'entity_idx':r['entity_idx'],'mass_kg':r['mass_kg'],'mass_basis':'initial_native_immutable_entity',
                    'position_world_m':xyz[i].tolist(),'velocity_world_m_s':vel[i].tolist(),'inside_carrier':bool(inside[i])} for i,r in enumerate(ids)]}
        if observation_readback!='scalar':
            from physics.episode_observation import EpisodeObservationReader,require_equal_observations
            observation_reader=EpisodeObservationReader(scene,arms=arms,hands=hands,fingers=fingers,tray=tray,beads=beads,
                **({'payload_mode':'empty'} if empty_mode else {} if particle_request is None else {'particle_material':particle_reader}))
        if automatic_grasp_mode:
            from physics.provided_automatic_grasp import ProvidedAutomaticGrasp
            automatic_grasp=ProvidedAutomaticGrasp(scene,cup=tray,model=model,arms=arms,
                readback=resolved_grasp_readback,verify_scalar=automatic_grasp_verify_scalar)
            student_driver.bind_automatic_grasp(automatic_grasp)
            save('automatic_grasp_initialization.json',automatic_initialization)
        def automatic_observation(raw):
            if single_active_contract is not None:raw['single_active_arm_contract']=copy.deepcopy(single_active_contract)
            if automatic_grasp is not None:
                from physics.automatic_runtime_binding import automatic_observation_binding
                raw['provided_grasp']=automatic_observation_binding(automatic_grasp,student_driver,raw)
            return raw
        last_particle_endpoint=None
        def observe():
            nonlocal last_particle_endpoint
            observation_started=time.perf_counter() if particle_request is not None else None
            if observation_readback=='scalar':return automatic_observation(observe_scalar())
            expected=observe_scalar() if observation_readback=='verify' else None
            if particle_request is None:raw=observation_reader.read_batched()
            else:
                bundle=observation_reader.read_batched_particle_bundle()
                if (type(bundle) is not dict or set(bundle)!={'schema_version','body_observation','reader_result','pause_snapshot'}
                        or bundle['schema_version']!='episode-native-particle-observation-bundle-v1'):
                    raise ContractError('Owned particle observation bundle required')
                raw=bundle['body_observation'];material=bundle['reader_result'];snapshot=bundle['pause_snapshot']
            if particle_request is not None:
                raw['body_up_tilt_deg']=body_tilt_deg(raw['body_world']['quaternion_wxyz'])
                step=raw['native_scene_step']
                if last_particle_endpoint is None or step!=last_particle_endpoint['step']:
                    if step!=(0 if last_particle_endpoint is None else last_particle_endpoint['step']+1):
                        raise ContractError('Native particle archive missed an actual outer endpoint')
                    reference=particle_archive.append(material,raw)['particle_archive_reference']
                else:
                    reference=last_particle_endpoint['reference']
                raw['particle_material']=packet_from_reader_result(particle_reader.contract,material,raw,archive_ref=reference)
                raw=stamp_observation(raw,snapshot)
                identity=digest({'packet':raw['particle_material'],'pause':raw['particle_pause_state']})
                if last_particle_endpoint is not None and step==last_particle_endpoint['step'] and identity!=last_particle_endpoint['identity']:
                    raise ContractError('Native particles changed without an actual physical interval')
                last_particle_endpoint=dict(step=step,reference=reference,identity=identity)
                particle_costs['outer_observation_calls']+=1
                particle_costs['rigid_particle_read_archive_pause_stamp_host_s']+=time.perf_counter()-observation_started
                return automatic_observation(raw)
            body=raw['body_world'];xyz=np.asarray(raw.pop('material_positions_world_m'),dtype=float).reshape((-1,3))
            vel=np.asarray(raw.pop('material_velocities_world_m_s'),dtype=float).reshape((-1,3))
            local=(xyz-body['position_m'])@rotation_matrix_wxyz(body['quaternion_wxyz'])
            inside=interior_membership(local,model.geometry_record(),.012,.001)
            raw['body_up_tilt_deg']=body_tilt_deg(body['quaternion_wxyz'])
            raw['materials']=[{'id':r['id'],'entity_idx':r['entity_idx'],'mass_kg':r['mass_kg'],'mass_basis':'initial_native_immutable_entity',
                'position_world_m':xyz[i].tolist(),'velocity_world_m_s':vel[i].tolist(),'inside_carrier':bool(inside[i])} for i,r in enumerate(ids)]
            if expected is not None:
                require_equal_observations(expected,raw);observation_reader.parity_checks+=1
                return automatic_observation(expected)
            return automatic_observation(raw)
        def particle_pause_bindings(initial_observation,initial_snapshot):
            meta=observation_reader.metadata()
            layout={k:v for k,v in meta.items() if k not in ('batched_reads','scalar_reads','same_time_parity_checks')}
            common=dict(scene_token=str(id(scene)),solver_token=str(id(scene.rigid_solver)),dt_s=DT,
                world_sha256=digest(world),configuration_sha256=result['configuration_sha256'],
                source_manifest_sha256=source_bindings['source_manifest_sha256'],
                geometry_sha256=digest({'owners':owners,'components':components,'pads':pads,'stations':station_map,'obstacles':obstacle_native,'spheres':sphere_geometry}),
                reader_layout_sha256=digest(layout),
                arms={n:{'entity_idx':int(a.idx),'model_sha256':world['robot_model_sha256']} for n,a in arms.items()},
                body={'id':scenario['body_id'],'entity_idx':int(tray.idx),'model_sha256':model.geometry_sha256})
            return freeze_bindings(common,particle_reader.contract,initial_snapshot)
        if particle_request is not None:
            student_driver.bind_particle_runtime(particle_reader.contract,coverage_validator=validate_particle_contact_coverage,
                pause_snapshot_reader=lambda body:read_particle_pause_snapshot(particle_reader,body),
                pause_bindings_factory=particle_pause_bindings)
        if private_contact_fixture:
            student_driver.bind_private_contact_profile(profile_native_timing)
        initial=observe();save('initial_observation.json',initial)
        setup_mark('observation_reader_and_initial_observation')
        if initial['simulation_s']!=0. or initial['native_scene_step']!=0:raise ContractError('Live episode did not start at native time zero')
        handling_enforcer=None
        if student_driver is not None:
            from physics.episode_handling_enforcement import SampledHandlingEnforcer,authenticate_native_reach
            reach_scenario=(scenario if single_active_contract is None else {**scenario,'robots':[r for r in scenario['robots'] if r['id'] in arms]})
            native_reach=authenticate_native_reach(arms,reach_scenario,
                model_file=installed/'assets'/case['robot_model'],model_sha256=world['robot_model_sha256'],
                revolute_type=gs.JOINT_TYPE.REVOLUTE,array_value=array)
            save('native_reach_envelopes.json',native_reach);result['native_reach_envelopes']=native_reach
            handling_enforcer=SampledHandlingEnforcer(
                body_identity={'id':'tray_0','entity_idx':int(tray.idx),'model_sha256':model.geometry_sha256},
                model_geometry=model.geometry_record(),holder_ids=tuple(arms),
                support_ids=tuple(s['id'] for s in scenario['support_stations']),
                initial_observation=initial,dt_s=DT,reach_envelopes=native_reach,
                **({} if automatic_grasp is None else {'automatic_grasp':automatic_grasp}),
                **({} if rigid_profile is None else {'rigid_profile':rigid_profile}))
            save('handling_enforcement_initial.json',handling_enforcer.record())
        plan=(episode_record(scenario,trajectory,initial,limits_by_arm,finger_bounds,transport)
            if student_driver is None else student_driver.initial_plan(world,scenario,model,initial,limits_by_arm,finger_bounds))
        from physics.episode_continuation_protocol import validate_options
        option_kwargs=({'repeated_three_arm_mode':True,'world':world,'scenario':scenario,
                        'initial_support_id':transport['goal']['support_id']} if repeated_three_arm_mode else {})
        phases=validate_options(continuation_phases,continuation_planner,global_deadline_s,
            initial_deadline_s=plan.deadline_s,dt_s=DT,**option_kwargs)
        if student_driver is not None:
            from transport_challenge.episode_continuation import ContinuationDispatcher
            dispatcher=ContinuationDispatcher(plan,model_sha256_by_arm={n:world['robot_model_sha256'] for n in arms},
                global_deadline_s=student_driver.simulation_deadline_s)
        elif phases is None:
            dispatcher=EpisodeDispatcher(plan)
        else:
            from transport_challenge.episode_continuation import ContinuationDispatcher
            from physics.episode_paused_state import begin_pause,verify_pause
            from physics.episode_continuation_protocol import rest_receipt,checked_segment
            record=plan.to_record()
            for arm in record['arms']:arm['role']='manipulator'
            plan=EpisodePlan.from_record(record,limits_by_arm=limits_by_arm)
            dispatcher=ContinuationDispatcher(plan,model_sha256_by_arm={n:world['robot_model_sha256'] for n in arms},
                global_deadline_s=global_deadline_s)
            result['continuation']={'phase_requests':copy.deepcopy(phases),'global_deadline_s':global_deadline_s,
                'completed_phases':[],'planning_pauses':[],'recipient_supported_grasp_verified':False,
                'single_delivery_verified':False,
                'scope':'Continuous measured two-arm prototype. Optional fixed-goal delivery is one bead batch, not calibrated throughput or student release.'}
            if repeated_three_arm_mode:
                result['continuation'].update(completed_stages=[],released_supports=[],supported_handoffs=[],
                    scope='Continuous three-arm prototype: arm0 to relay1, arm1 to relay2, arm2 to fixed goal. Every stage uses a fresh observed boundary.')
        rest_policy=prototype_policy(dt_s=DT,goal_body_pose=world['goal_body_world'],goal_support_id=world['goal_support']['id'],
            arm_limits=trajectory.limits,finger_limits=finger_bounds[active],effort_limits=effort.tolist()).to_record()
        save('inherited_body_rest_policy.json',{'source':'physics.delivery_inputs.prototype_policy',
            'scope':'Existing ninth sampled-policy body speed bounds, used over the arm rest window only.',
            'policy':rest_policy,'sha256':digest(rest_policy)})
        save('external_episode.json',plan.to_record());save('episode_timing.json',plan.timing_record())
        result.update(external_episode_sha256=plan.sha256,initial_observation=initial,controller={**copy.deepcopy(settings['controller']),'name':'gravity_pd_external_episode'},
                      execution_limits=execution,initial_supply=ids,live_contact_policy_sha256=coverage['coverage_policy_sha256'])
        if student_driver is not None and student_driver.mode in ('one_carry','tier_mission'):
            from physics.initial_state_fingerprint import capture_initial_fingerprint
            fingerprint=capture_initial_fingerprint(dict(scene=scene,arms=arms,tray=tray,beads=beads,ids=ids,
                model=model,world=world,scenario=scenario,settings=settings,configuration=configuration,
                automatic_grasp=automatic_grasp,handling_enforcer=handling_enforcer,dispatcher=dispatcher,
                initial_observation=initial,particle_reader=particle_reader))
            save('initial_state_fingerprint.json',fingerprint)
            result['initial_state_fingerprint']={key:fingerprint[key] for key in
                ('schema_version','status','record_sha256','fresh_scene_comparison_eligible',
                 'restored_state_claimed','full_checkpoint_inventory_complete','checkpoint_reuse_qualified')}
            print(json.dumps({'initial_fingerprint_status':fingerprint['status'],
                'captured_field_count':len(fingerprint['fields']),
                'missing_required_field_count':len(fingerprint['missing_required_fields'])}),flush=True)
        stream=(output/'episode_commands_contacts_and_material.jsonl').open('w',encoding='utf-8')
        if trace_mode=='compact':
            from physics.compact_episode_log import CompactEpisodeLog
            compact_log=CompactEpisodeLog(stream,output/'episode_failure_context.json',mode='compact',prefailure_steps=32,
                full_encoder=lambda value:json.dumps(_json_safe(value),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
        if student_driver is not None and student_driver.sparse_capture_only:
            if camera is None:raise ContractError('Requested native stills need rendering enabled')
            from physics.student_checkpoint_images import SparseNativeImages
            student_driver.image_sink=SparseNativeImages(scene,camera,output,observe)
        if camera is not None and (student_driver is None or not student_driver.sparse_capture_only):
            from PIL import Image
            if empty_video:
                body=np.asarray(initial['body_world']['position_m'])
                camera.set_pose(pos=body+np.array((.28,-.30,.20)),lookat=body+np.array((0.,0.,.035)),up=(0.,0.,1.))
                result['native_video']={'file':'empty_container_path.mp4','requested_fps':20,
                    'camera_fixed':True,'camera_position_m':(body+np.array((.28,-.30,.20))).tolist(),
                    'camera_lookat_m':(body+np.array((0.,0.,.035))).tolist(),
                    'source':'Actual Genesis camera recording during scene.step. No reconstructed state or image interpolation.'}
            Image.fromarray(camera.render()[0]).save(output/'initial_native_scene.png')
            if record_video:
                camera.start_recording(save_to_filename=str(output/('empty_container_path.mp4' if empty_video else 'episode_transport.mp4' if transport is not None else 'episode_approach.mp4')),fps=20);recording=True
        observation=initial;bilateral_since=None;stable_since=None;support_since=None;last_progress=time.monotonic();rows=[]
        unsupported_bilateral_s=0.;maximum_unsupported_bilateral_s=0.;prefix_spec=None
        maximum_unsupported_bilateral_by_arm_s={name:0. for name in arms}
        unsupported_provided_grasp_s=0.
        maximum_unsupported_provided_grasp_by_arm_s={name:0. for name in arms}
        recipient_grasp_start=None;delivery_goal=None
        continuation_index=0;continuation_phase=None;phase_body_reference=scenario['body_pose']
        phase_support_id=scenario['support_stations'][0]['id'];phase_endpoint=scenario['endpoint_hand_pose']
        released_goal=None if transport is None else transport['goal']
        released_holder=active
        provided_control_targets=None
        robot_reset_events=[]
        robot_return_events=[]
        pending_robot_reset_boundary=None
        def reset_completed_robot(driver,owned_dispatcher,state,completed_operation):
            return post_place_robot_operation(driver,owned_dispatcher,state,completed_operation,kinematic=False)
        def return_completed_robot_kinematically(driver,owned_dispatcher,state,completed_operation):
            return post_place_robot_operation(driver,owned_dispatcher,state,completed_operation,kinematic=True)
        def post_place_robot_operation(driver,owned_dispatcher,state,completed_operation,*,kinematic):
            nonlocal observation,rows,provided_control_targets,pending_robot_reset_boundary
            from physics.provided_robot_reset import perform_released_robot_reset
            if (driver is not student_driver or owned_dispatcher is not dispatcher
                    or getattr(driver,'post_place_policy',None)!=('kinematic_return' if kinematic else 'instant_reset')
                    or driver.saved_plan is None or automatic_grasp is None
                    or rigid_profile!='rigid16_bead_tc2h_v1' or empty_mode or particle_request is not None
                    or driver.current is not None or driver.held is not None or automatic_grasp.holder is not None
                    or digest(state['observation'])!=digest(observation) or pending_robot_reset_boundary is not None):
                raise ContractError('Released robot operation requires the owned saved four-bead boundary')
            op=completed_operation;arm_id=op['active_arm_id'];stage_id=op['stage_id']
            expected=(driver.saved_plan.kinematic_return_after(stage_id) if kinematic else driver.saved_plan.reset_after(stage_id))
            operation_events=robot_return_events if kinematic else robot_reset_events
            if (op['kind']!='carry' or arm_id not in ('arm_0','arm_1') or expected is None
                    or expected['arm_id']!=arm_id or expected['after_stage_id']!=stage_id
                    or len(operation_events)!=('arm_0','arm_1').index(arm_id)
                    or not driver.completed or driver.completed[-1]['stage_id']!=stage_id
                    or driver.completed[-1]['native_scene_step']!=int(scene.t)
                    or not driver.gates or driver.gates[-1]['stage_id']!=stage_id):
                raise ContractError('Released robot operation differs from its completed donor and saved schedule')
            released_gate=driver.gates[-1]['gate']
            release_events=[event for event in driver.automatic_events
                if event.get('event')=='release' and event.get('stage_id')==stage_id]
            if len(release_events)!=1 or released_gate.get('passed') is not True:
                raise ContractError('Released robot operation needs the actual native release and completed support/rest gate')
            before_rest=dict(arms=measured_rest(recent,observation['simulation_s'],execution,arms),
                body=body_rest(recent,observation['simulation_s'],execution['rest_window_s'],rest_policy))
            if not before_rest['arms']['passed'] or not before_rest['body']['passed']:
                raise ContractError('Released robot operation requires actual all-arm and body rest before returning')
            def retarget(name,target):
                arm=arms[name];target=np.asarray(target,dtype=np.float32);zero=np.zeros(9,dtype=np.float32)
                tau=vector(gravity[name].torques(),9)
                arm.set_dofs_act_bias(tau,-kp,-kv)
                arm.control_dofs_position_velocity(target,zero)
                motor=vector(arm.get_dofs_control_force(),9)
                error=float(np.max(np.abs(motor-np.clip(tau,-effort,effort))))
                if np.any(np.abs(tau)>effort) or np.any(np.abs(motor)>effort+1e-5) or error>.005:
                    raise ContractError('Home gravity-PD retarget exceeds the original effort contract')
                return dict(passed=True,gravity_effort=tau.tolist(),effort_identity_error=error,
                    derivative_history='Next actual interval starts from fresh native qd0. Old physical rows are preserved.')
            fresh_contact_rows=None
            def check_contacts():
                nonlocal fresh_contact_rows
                scene.rigid_solver.detect_collision()
                raw=_copy_contacts(scene.rigid_solver.collider);current=observe()
                classified,_=classify_episode_live_contacts(raw,owners=owners,components_by_geom=components,
                    pads_by_arm=pads,stations_by_geom=station_map,body_pose=current['body_world'],
                    finger_poses_by_arm={name:value['finger_world'] for name,value in current['arms'].items()},
                    phase='released_settle',active_holder_arm_id=None)
                stamp=live_contact_coverage_stamp(scene,coverage,automatic_grasp=automatic_grasp)
                bad=[row for row in classified if row['category']=='UNALLOWED_CONTACT']
                support_id=driver.gates[-1]['support_id']
                supported=any(row['category']=='named_tray_bottom_support_top'
                    and row.get('support_station_id')==support_id for row in classified)
                if (stamp['policy_valid_now'] is not True or stamp['native_contact_count']!=len(raw['geom_a'])
                        or len(classified)!=len(raw['geom_a']) or bad or not supported):
                    raise ContractError('Fresh native home endpoint contact/support check failed')
                fresh_contact_rows=classified
                return dict(passed=True,fresh_detection=True,physical_steps=0,
                    native_contact_count=stamp['native_contact_count'],classified_contact_count=len(classified),
                    unallowed_contact_count=len(bad),support_id=support_id,named_support_contact_present=True,
                    classified_contacts=classified,policy_stamp=stamp,
                    scope='Fresh home endpoint only. No swept-path or physical-return certificate.')
            before=copy.deepcopy(observation)
            def record_setter(status):
                key=('authorized_robot_return_setter_attempts' if kinematic else 'authorized_robot_reset_setter_attempts') if status=='attempt' else 'post_initialization_pose_setters'
                result[key]=result.get(key,0)+1
                result['post_initialization_payload_pose_setters']=0
                result['post_initialization_robot_pose_setters']=result.get('post_initialization_pose_setters',0)
                result['robot_operation_setter_failure_scope']='Executed counts advance immediately after the native qpos call. Successful reset receipts require all later invariant/contact/source checks.'
            operation_kwargs=dict(scene=scene,arms=arms,cup=tray,beads=beads,identities=ids,
                helper=automatic_grasp,arm_id=arm_id,after_stage_id=stage_id,
                target_command=expected['target_command'],initial_observation=initial,before_observation=before,
                observe=observe,released_gate=released_gate,release_event=release_events[0]['receipt'],
                pre_reset_rest=before_rest,source_check=source_check,retarget_control=retarget,check_contacts=check_contacts,
                record_setter=record_setter)
            if kinematic:
                from physics.provided_robot_return import perform_released_robot_return
                after,receipt=perform_released_robot_return(return_schedule=expected,
                    transcript_path=output/(stage_id+'__kinematic_return_native.jsonl'),limits=driver.limits_by_arm[arm_id],
                    allowed_initial_tracking_error_rad=execution['final_joint_error_rad'],wall_check=student_wall_check,
                    frame_sink=getattr(driver,'kinematic_return_frame_sink',None),**operation_kwargs)
                dispatcher_receipt=dispatcher.accept_provided_kinematic_return_event(before,after,receipt,
                    source_bindings_verified=source_check() is True)
            else:
                after,receipt=perform_released_robot_reset(**operation_kwargs)
                dispatcher_receipt=dispatcher.accept_provided_robot_reset_event(before,after,receipt,
                    source_bindings_verified=source_check() is True)
            observation=after;rows=fresh_contact_rows
            dispatch=dispatcher.record()
            extension=dict(schema_version=('episode-continuation-kinematic-return-boundary-v1' if kinematic else 'episode-continuation-reset-boundary-v1'),
                previous_segment_sha256=dispatch['current_segment_sha256'],step=dispatch['global_step'],
                simulation_s=dispatch['simulation_s'],observation_sha256=digest(after),verified_arm_ids=list(arms),
                actual_state_verified=True,measured_rest_verified=True,source_bindings_verified=True,
                native_robot_reset_receipt_sha256=digest(receipt),
                rest_semantics=('Actual pre-return rest plus native home zero velocity. No physical return interval or home rest window.' if kinematic else 'Actual pre-reset rest plus fresh post-reset native zero velocity. No post-reset pose history.'))
            if kinematic:extension['native_robot_return_receipt_sha256']=extension.pop('native_robot_reset_receipt_sha256')
            pending_robot_reset_boundary=dict(verified=True,receipt_sha256=digest(receipt),event_kind='kinematic_return' if kinematic else 'instant_reset',
                extension_boundary=extension,native_scene_step=int(scene.t),observation_sha256=digest(after))
            operation_events.append(dict(receipt=receipt,dispatcher_receipt=dispatcher_receipt,
                extension_boundary=extension,physical_rows_rewritten=False))
            save('provided_robot_return_events.json' if kinematic else 'provided_robot_reset_events.json',operation_events)
            result.update(post_initialization_robot_pose_setters=result.get('post_initialization_pose_setters',0),
                post_initialization_payload_pose_setters=0,authorized_zero_time_robot_resets=len(robot_reset_events),
                authorized_kinematic_robot_returns=len(robot_return_events),
                post_initialization_pose_setter_scope='Explicit released donor-only joint operations. No payload/base/clock setters.')
            result['provided_robot_return_events' if kinematic else 'provided_robot_reset_events']=copy.deepcopy(operation_events)
            if kinematic:
                result['kinematic_return_pose_setters']=sum(event['receipt']['pose_setter_count'] for event in robot_return_events)
                result['kinematic_return_display_duration_s']=sum(event['receipt']['display_duration_s'] for event in robot_return_events)
            native=receipt['native_after']['control_targets']
            provided_control_targets=dict(step=int(scene.t),simulation_s=float(scene.cur_t),
                schedule_phase='zero_time:kinematic_return' if kinematic else 'zero_time:instant_reset',plan_sha256=dispatcher.current_segment.plan.sha256,
                exact_native_dtype_target_match=True,source='Fresh native home POSITION targets after a released robot operation, not a physical interval.',
                by_arm={name:dict(native_ctrl_pos=value['ctrl_pos'],native_ctrl_vel=value['ctrl_vel'],
                    native_ctrl_mode=value['ctrl_mode'],scheduled_ctrl_pos=value['ctrl_pos'],scheduled_ctrl_vel=value['ctrl_vel'])
                    for name,value in native.items()})
            fresh=student_state()
            return dict(state=fresh,receipt=copy.deepcopy(receipt),extension_boundary=copy.deepcopy(extension))
        def record_provided_grasp_event(event):
            nonlocal observation
            if automatic_grasp is None:raise ContractError('Native grasp event requires the provided automatic model')
            from physics.automatic_runtime_binding import physical_observation
            before=copy.deepcopy(observation)
            measured=observe()
            if physical_observation(before)!=physical_observation(measured):
                raise ContractError('A provided grasp event changed physical observations without an interval')
            after=copy.deepcopy(before);after['provided_grasp']=copy.deepcopy(measured['provided_grasp'])
            receipt=dispatcher.accept_provided_grasp_event(before,after,event,
                source_bindings_verified=source_check() is True,runtime_profile=rigid_profile)
            automatic_boundary_events.append(receipt)
            save('provided_automatic_boundary_events.json',automatic_boundary_events)
            observation=after
            return copy.deepcopy(after)
        def student_state():
            return dict(observation=observation,recent_rows=recent,settings=settings,rest_policy=rest_policy,
                contact_rows=rows,bilateral_since=bilateral_since,stable_since=stable_since,support_since=support_since,
                maximum_unsupported_bilateral_by_arm_s=maximum_unsupported_bilateral_by_arm_s,
                unsupported_bilateral_s=unsupported_bilateral_s,
                unsupported_provided_grasp_s=unsupported_provided_grasp_s,
                maximum_unsupported_provided_grasp_by_arm_s=maximum_unsupported_provided_grasp_by_arm_s,
                observe=observe,source_check=source_check,wall_check=lambda:student_wall_check(),output=output,
                provided_native_control_targets=provided_control_targets,
                **({} if automatic_grasp is None else {'record_provided_grasp_event':record_provided_grasp_event}),
                **({} if student_driver is None or getattr(student_driver,'post_place_policy',None)!='instant_reset'
                    else {'reset_completed_robot':reset_completed_robot}),
                **({} if student_driver is None or getattr(student_driver,'post_place_policy',None)!='kinematic_return'
                    else {'return_completed_robot_kinematically':return_completed_robot_kinematically}),
                **({} if pending_robot_reset_boundary is None else
                    {('kinematic_return_boundary' if pending_robot_reset_boundary['event_kind']=='kinematic_return' else 'instant_reset_boundary'):copy.deepcopy(pending_robot_reset_boundary)}))
        def student_wall_check():
            if time.monotonic()-started>wall_limit_s:raise TimeoutError('Student execution wall budget exceeded')
        def release_detail(checkpoint,support_id):
            nonlocal camera
            if detail_images and not _capture_release_detail(camera,output,observation,result,
                    checkpoint=checkpoint,support_id=support_id):
                # A failed optional camera restore must not turn later image
                # cleanup into a false physical failure. Earlier PNGs remain.
                camera=None
        def stage_context():
            return phases[continuation_index-1] if repeated_three_arm_mode and continuation_index else None
        def stage_actor():
            return stage_context()['active_arm_id'] if repeated_three_arm_mode else 'arm_1'
        def stage_save(filename,value):
            stage=stage_context()
            save(stage['stage_id']+'__'+filename if stage is not None else filename,value)
        def stage_completed():
            result['continuation']['completed_phases'].append(continuation_phase)
            if repeated_three_arm_mode:
                stage=copy.deepcopy(stage_context())
                stage.update(completed_step=int(scene.t),completed_simulation_s=observation['simulation_s'],
                             observation_sha256=digest(observation))
                result['continuation']['completed_stages'].append(stage)
                if camera is not None:
                    from PIL import Image
                    filename=stage['stage_id']+'__completed.png'
                    Image.fromarray(camera.render(force_render=True)[0]).save(output/filename)
                    if not record_video:result['image_capture']['checkpoints'].append(
                        {'file':filename,'stage_id':stage['stage_id'],'native_scene_step':int(scene.t),
                         'simulation_s':observation['simulation_s']})
        def boundary(request):
            nonlocal prefix_spec,pending_robot_reset_boundary
            if request['action_kind']=='supported_preload' and student_driver is None:
                raise ContractError('Supported preload requires the actual provided StudentEpisodeDriver context')
            ack,detail=verify_boundary(request,observation,recent,settings,source_bindings_verified=source_check())
            if pending_robot_reset_boundary is not None:
                reset=pending_robot_reset_boundary
                if (request['at_step']!=reset['native_scene_step'] or digest(observation)!=reset['observation_sha256']
                        or request['action_role']!='approach' or request['active_arm_id'] not in ('arm_1','arm_2')):
                    raise ContractError('First continuation must use the exact fresh recipient approach boundary')
                detail['kinematic_return_boundary' if reset['event_kind']=='kinematic_return' else 'instant_robot_reset_boundary']=copy.deepcopy(reset)
                detail['rest_history_scope']='Actual pre-operation physical rows plus the separate native home/zero-velocity proof. No physical return or home rest-window history.'
                pending_robot_reset_boundary=None
            if student_driver is not None:
                detail['provided_operation']=student_driver.check_boundary(request,student_state())
                boundaries.append(detail);save('boundary_checks.json',boundaries)
                ack.update(schema_version='episode-continuation-boundary-ack-v1',
                    **{k:request[k] for k in ('segment_sha256','segment_index','global_action_index')})
                return ack
            # The precomputed query describes a resting tray, not its future motion.
            # Record every actual difference. Use existing position/orientation
            # tolerances solely as this offline replay's explicit agreement gate.
            if request['action_role'] in ('approach','retreat','recipient_approach'):
                position=math.dist(observation['body_world']['position_m'],phase_body_reference['position_m'])
                angle=quaternion_error(observation['body_world']['quaternion_wxyz'],phase_body_reference['quaternion_wxyz'])
                detail['offline_body_deviation']={'position_m':position,'orientation_rad':angle}
                detail['body_rest']=body_rest(recent,observation['simulation_s'],execution['rest_window_s'],rest_policy)
                if not detail['body_rest']['passed']:raise ContractError('Actual source body has not settled: '+json.dumps(detail))
                if position>execution['final_body_position_error_m'] or angle>execution['final_body_orientation_error_rad']:
                    raise ContractError('Actual source body differs from offline geometry: '+json.dumps(detail))
            if request['action_role']=='carry':
                if continuation_phase=='recipient_carry':
                    if (recipient_grasp_start is None or recipient_grasp_start['gate']['passed'] is not True
                            or request['active_arm_id']!=stage_actor()
                            or recipient_grasp_start['observation_sha256']!=digest(observation)):
                        raise ContractError('Recipient carry must start at the exact verified live grasp observation')
                    detail['supported_grasp_gate']=copy.deepcopy(recipient_grasp_start['gate'])
                    detail['actual_recipient_grasp_binding_sha256']=recipient_grasp_start['held_binding_sha256']
                    detail['exact_recipient_grasp_observation_unchanged']=True
                else:
                    gate,prefix_spec=supported_grasp_gate(observation,recent,settings,arms,scenario,rest_policy,rows,
                        bilateral_since,stable_since,support_since,model)
                    agreement=reference_agreement(observation,grasp_reference['final_observation'],settings)
                    detail['supported_grasp_gate']=gate;detail['offline_reference_agreement']=agreement
                    save('actual_prefix_grasp.json',{'gate':gate,'observation':observation,'observation_sha256':digest(observation),
                        'held_body':prefix_spec.to_record(),'held_binding_sha256':prefix_spec.binding_sha256,'agreement':agreement})
                    if not gate['passed'] or not agreement['passed']:
                        raise ContractError('Fresh continuous grasp differs from the offline held plan: '+json.dumps(detail))
                    result['supported_grasp_verified']=True
                    result['actual_prefix_grasp_binding_sha256']=prefix_spec.binding_sha256
            if request['action_role']=='release':
                selected_goal=delivery_goal if continuation_phase=='recipient_carry' else transport['goal']
                selected,expected,complete=rest_rows(recent,observation['simulation_s'],TRANSPORT_PROTOCOL['acceptance_window_s'])
                support_ok=complete and all(any(c['category']=='named_tray_bottom_support_top'
                    and c.get('support_station_id')==selected_goal['support_id'] for c in r['classified_contacts']) for r in selected)
                placement_goal=selected_goal.get('seating',{}).get('nominal_goal',selected_goal)
                b=placement_goal['body_world'];current=observation['body_world']
                position=math.dist(current['position_m'],b['position_m']);angle=quaternion_error(current['quaternion_wxyz'],b['quaternion_wxyz'])
                rest=body_rest(recent,observation['simulation_s'],execution['rest_window_s'],rest_policy)
                placement={'support_window_rows':len(selected),'expected_rows':expected,'support_present_every_interval':support_ok,
                    'body_rest':rest,'body_goal_position_error_m':position,'body_goal_orientation_error_rad':angle}
                placement['passed']=support_ok and rest['passed'] and position<=execution['final_body_position_error_m'] and angle<=execution['final_body_orientation_error_rad']
                detail['before_release_placement']=placement
                stage_save('before_final_release.json' if continuation_phase=='recipient_carry' else 'before_relay_release.json',placement)
                if not placement['passed']:raise ContractError('Actual named support placement is not ready for release: '+json.dumps(placement))
            boundaries.append(detail);save('boundary_checks.json',boundaries)
            if camera is not None and not record_video:
                filename=f"checkpoint_{len(boundaries):03d}_{request['action_role']}.png"
                Image.fromarray(camera.render(force_render=True)[0]).save(output/filename)
                result['image_capture']['checkpoints'].append({'file':filename,'action_role':request['action_role'],
                    'native_scene_step':int(scene.t),'simulation_s':float(scene.cur_t)})
            if phases is not None:
                ack.update(schema_version='episode-continuation-boundary-ack-v1',
                    **{k:request[k] for k in ('segment_sha256','segment_index','global_action_index')})
            return ack
        def fixed_state():
            records={}
            for name,arm in arms.items():
                record=next(r for r in scenario['robots'] if r['id']==name)
                validate_native_base_position(episode_base_position(record,case),arm.get_pos(relative=False),arm_id=name)
                lower,upper=arm.get_dofs_limit()
                lower_array=np.asarray(lower.detach().cpu().numpy() if hasattr(lower,'detach') else lower)
                records[name]={'entity_idx':int(arm.idx),'native_dtype':str(lower_array.dtype),
                    'base_world':pose(arm),'native_lower':vector(lower,9).tolist(),'native_upper':vector(upper,9).tolist()}
            stations=[]
            current_supports,current_entities=runtime_support_registry(gs,supports,scenario['support_stations'],obstacles,compound_assets,case)
            current_by_name={r['id']:r for r in current_supports.values()}
            for station in scenario['support_stations']:
                record=current_by_name[station['id']];checked=record['observed_native_geometry'];entity=current_entities[station['id']]
                world_pose=({'frame':'world','position_m':checked['center_world_m'],'quaternion_wxyz':[1.,0.,0.,0.]}
                    if record.get('embedded_in_rigid_assembly') is True else pose(entity))
                stations.append({'id':station['id'],'entity_idx':int(entity.idx),'geom_id':record['geom_id'],
                    'size_m':checked['size_m'],'world_pose':world_pose})
                if repeated_three_arm_mode and record.get('embedded_in_rigid_assembly') is True:
                    stations[-1].update(embedded_in_rigid_assembly=True,
                        assembly_id=record['assembly_id'],component_id=record['component_id'],
                        validated_native_component=copy.deepcopy(record['validated_native_component']),
                        embedded_station=copy.deepcopy(record))
            return {'schema_version':'episode-unloaded-fixed-state-v1','native_scene_step':int(scene.t),
                'simulation_s':float(scene.cur_t),'arms':records,
                'body':{'id':scenario['body_id'],'entity_idx':int(tray.idx),'mass_kg':float(array(tray.get_mass()).sum()),
                    'model_geometry_sha256':model.geometry_sha256},'supports':stations}
        def pause_bindings():
            meta=observation_reader.metadata()
            layout={k:v for k,v in meta.items() if k not in ('batched_reads','scalar_reads','same_time_parity_checks')}
            return {'schema_version':'episode-pause-bindings-v1','scene_token':str(id(scene)),
                'solver_token':str(id(scene.rigid_solver)),'dt_s':DT,'world_sha256':digest(world),
                'configuration_sha256':result['configuration_sha256'],
                'source_manifest_sha256':source_bindings['source_manifest_sha256'],
                'geometry_sha256':digest({'owners':owners,'components':components,'pads':pads,'stations':station_map,
                    'obstacles':obstacle_native,'spheres':sphere_geometry}),
                'reader_layout_sha256':digest(layout),
                'arms':{n:{'entity_idx':int(a.idx),'model_sha256':world['robot_model_sha256']} for n,a in arms.items()},
                'body':{'id':scenario['body_id'],'entity_idx':int(tray.idx),'model_sha256':model.geometry_sha256},
                'materials':copy.deepcopy(ids)}
        def continuation_boundary():
            nonlocal continuation_index,continuation_phase,phase_body_reference,phase_support_id,phase_endpoint
            nonlocal bilateral_since,stable_since,support_since
            nonlocal recipient_grasp_start,delivery_goal,unsupported_bilateral_s,released_goal,released_holder
            dispatch=dispatcher.record()
            if dispatch['current_segment_complete'] is not True or dispatch['pending_step'] is not None:
                raise ContractError('Continuation stopped at a deadline or incomplete segment')
            if continuation_phase=='recipient_carry':
                placement_goal=delivery_goal.get('seating',{}).get('nominal_goal',delivery_goal)
                carrying=stage_actor()
                gate=released_relay_gate(recent,observation['simulation_s'],settings,arms,rest_policy,placement_goal,
                    finger_bounds,ids,dt=DT,window_s=TRANSPORT_PROTOCOL['acceptance_window_s'],active_arm_id=carrying)
                load_bearing=maximum_unsupported_bilateral_by_arm_s[carrying]>=TRANSPORT_PROTOCOL['acceptance_window_s']
                final_release=not repeated_three_arm_mode or placement_goal['support_id']==world['goal_support']['id']
                if final_release:
                    result['continuation'].update(final_release_gate=gate,recipient_load_bearing_grasp_verified=load_bearing)
                stage_save('final_release_gate.json' if final_release else 'relay_release_gate.json',gate)
                if not gate['passed'] or not load_bearing:
                    raise ContractError('Recipient release or load-bearing carry failed: '+json.dumps(gate))
                released_goal=delivery_goal;released_holder=carrying
                stage_completed()
                if repeated_three_arm_mode:
                    result['continuation']['released_supports'].append({'arm_id':carrying,'support_id':placement_goal['support_id'],
                        'gate':copy.deepcopy(gate),'load_bearing_verified':load_bearing,'simulation_s':observation['simulation_s']})
                if final_release:
                    if repeated_three_arm_mode and (continuation_index!=6 or len(result['continuation']['completed_stages'])!=6
                            or len(result['continuation']['supported_handoffs'])!=2):
                        raise ContractError('Fixed-goal completion requires both actual supported handoffs and every committed stage')
                    result['continuation']['single_delivery_verified']=True
                    result.update(physical_handoff_verified=True,material_delivery_accepted=True,
                        delivered_material_ids=[m['id'] for m in ids],delivered_count=len(ids),
                        delivered_mass_kg=math.fsum(m['mass_kg'] for m in ids),
                        delivery_scope='One complete persistent bead batch at the fixed goal after the stable released window. Repeated throughput and other materials remain unverified.')
                    if repeated_three_arm_mode:result['physical_handoff_count']=2
                    release_detail('final_goal_release',placement_goal['support_id'])
                    return False
                release_detail('second_relay_release',placement_goal['support_id'])
            elif continuation_phase=='recipient_approach':
                gate,spec=supported_grasp_gate(observation,recent,settings,arms,scenario,rest_policy,rows,
                    bilateral_since,stable_since,support_since,model,active_arm_id=stage_actor(),
                    support_id=phase_support_id,endpoint_hand_pose=phase_endpoint,body_id=scenario['body_id'])
                recipient_grasp_start={'gate':gate,'observation':copy.deepcopy(observation),
                    'observation_sha256':digest(observation),'held_body':spec.to_record(),
                    'held_binding_sha256':spec.binding_sha256,'endpoint_hand_pose':copy.deepcopy(phase_endpoint),
                    'support_id':phase_support_id}
                stage_save('recipient_supported_grasp.json',recipient_grasp_start)
                result['continuation'].update(recipient_supported_grasp_verified=gate['passed'],
                    recipient_supported_grasp_gate=gate,recipient_held_body=spec.to_record())
                if not gate['passed']:raise ContractError('Actual recipient supported grasp failed')
                stage_completed()
                if repeated_three_arm_mode:
                    result['continuation']['supported_handoffs'].append({'donor_arm_id':stage_context()['donor_arm_id'],
                        'recipient_arm_id':stage_actor(),'support_id':phase_support_id,'gate':copy.deepcopy(gate),
                        'observation_sha256':digest(observation),'held_body_binding_sha256':spec.binding_sha256})
            else:
                placement_goal=released_goal.get('seating',{}).get('nominal_goal',released_goal)
                gate=released_relay_gate(recent,observation['simulation_s'],settings,arms,rest_policy,placement_goal,
                    finger_bounds,ids,dt=DT,window_s=TRANSPORT_PROTOCOL['acceptance_window_s'],
                    **({'active_arm_id':released_holder} if repeated_three_arm_mode else {}))
                if not gate['passed'] or maximum_unsupported_bilateral_by_arm_s[released_holder]<TRANSPORT_PROTOCOL['acceptance_window_s']:
                    raise ContractError('Continuous relay release/rest prerequisite failed: '+json.dumps(gate))
                if continuation_phase is None:
                    result.update(relay_release_gate=gate,relay_placement_verified=True,load_bearing_grasp_verified=True)
                    save('relay_release_gate.json',gate)
                    release_detail('first_relay_release',placement_goal['support_id'])
                    if repeated_three_arm_mode:
                        result['continuation']['released_supports'].append({'arm_id':released_holder,'support_id':placement_goal['support_id'],
                            'gate':copy.deepcopy(gate),'load_bearing_verified':True,'simulation_s':observation['simulation_s']})
                else:
                    stage_completed()
            if continuation_index==len(phases):return False
            if not source_check():raise ContractError('Sources changed before continuation planning')
            native=fixed_state();bindings=pause_bindings();receipt=rest_receipt(dispatch,observation,arms)
            paused=begin_pause(observation,bindings=bindings,dispatcher_record=dispatch,boundary_receipt=receipt)
            held_effort={n:vector(a.get_dofs_control_force(),9).tolist() for n,a in arms.items()}
            request={'schema_version':'episode-continuation-request-v1','phase_request':copy.deepcopy(phases[continuation_index]),
                'world':copy.deepcopy(world),'scenario':copy.deepcopy(scenario),'observation':copy.deepcopy(observation),
                'dispatcher':dispatch,'bindings':bindings,'fixed_state':native,
                'rest_boundary':receipt,'source_bindings':copy.deepcopy(source_bindings)}
            if repeated_three_arm_mode:
                next_stage=phases[continuation_index]
                expected_support=recipient_grasp_start['support_id'] if next_stage['phase']=='recipient_carry' else placement_goal['support_id']
                if next_stage['support_id']!=expected_support:
                    raise ContractError('Next stage support differs from the freshly verified boundary')
                request['repeated_three_arm_mode']=True
            if phases[continuation_index]['phase']=='recipient_carry':
                from physics.episode_held_phase_inputs import selected_goal
                delivery_goal=selected_goal(world,scenario,spec,phases[continuation_index]['endpoint'],**mode_kwargs)
                request.update(supported_grasp_gate=copy.deepcopy(gate),held_body=spec.to_record(),
                    grasp_endpoint_hand_pose=copy.deepcopy(recipient_grasp_start['endpoint_hand_pose']),
                    grasp_support_id=recipient_grasp_start['support_id'],
                    query_body_tilt_limit_deg=transport_plan['query_body_tilt_limit_deg'])
                save(phases[continuation_index]['stage_id']+'__recipient_delivery_goal.json' if repeated_three_arm_mode else 'recipient_delivery_goal.json',delivery_goal)
            else:
                request['released_gate']=gate
            prefix=f'continuation_{continuation_index+1}'
            save(prefix+'_request.json',request);save(prefix+'_pause_before.json',paused)
            planning_started=time.monotonic()
            try:
                response=continuation_planner(copy.deepcopy(request),wall_limit_s-(planning_started-started))
            except Exception:
                failed_pause={'phase':phases[continuation_index]['phase'],'request_sha256':digest(request),
                    'planning_wall_s':time.monotonic()-planning_started,'status':'PLANNER_RAISED',
                    'parent_state_comparison_performed':False,'parent_state_invariance_on_failure_verified':False}
                result['continuation']['planning_pauses'].append(failed_pause);save(prefix+'_pause_failed.json',failed_pause)
                raise
            planning_wall=time.monotonic()-planning_started
            if time.monotonic()-started>wall_limit_s:
                failed_pause={'phase':phases[continuation_index]['phase'],'request_sha256':digest(request),
                    'planning_wall_s':planning_wall,'status':'ORIGINAL_WALL_DEADLINE_EXCEEDED',
                    'parent_state_comparison_performed':False,'parent_state_invariance_on_failure_verified':False}
                result['continuation']['planning_pauses'].append(failed_pause);save(prefix+'_pause_failed.json',failed_pause)
                raise TimeoutError('Planning exceeded original episode wall deadline')
            after=observe();fresh_native=fixed_state()
            comparison=verify_pause(paused,after,bindings=pause_bindings(),dispatcher_record=dispatcher.record(),boundary_receipt=receipt)
            if fresh_native!=native or held_effort!={n:vector(a.get_dofs_control_force(),9).tolist() for n,a in arms.items()}:
                raise ContractError('Fixed geometry/identity or held control effort changed during planning pause')
            if not source_check():raise ContractError('Sources changed during continuation planning')
            next_plan=checked_segment(request,response,previous_plan=dispatcher.current_segment.plan,
                limits_by_arm=limits_by_arm,protocol=PROTOCOL,transport_protocol=TRANSPORT_PROTOCOL)
            dispatcher.extend(next_plan,model_sha256_by_arm={n:world['robot_model_sha256'] for n in arms},boundary=receipt)
            save(prefix+'_response.json',response);save(prefix+'_episode.json',next_plan.to_record())
            detail={'phase':phases[continuation_index]['phase'],'planning_wall_s':planning_wall,
                'request_sha256':digest(request),'response_sha256':digest(response),'comparison':comparison,
                'parent_state_comparison_performed':True,
                'fixed_state_identical':True,'held_motor_effort_identical':True,'scene_clock_unchanged':True}
            detail['fixed_geometry_readback_scope']='Fresh arm bases/bounds, support poses/sizes and body mass. Obstacle identities/geometry remain bound through the built ledger and frozen sources.'
            result['continuation']['planning_pauses'].append(detail);save(prefix+'_pause_after.json',detail)
            continuation_phase=phases[continuation_index]['phase'];continuation_index+=1
            phase_body_reference=copy.deepcopy(observation['body_world'])
            phase_support_id=delivery_goal['support_id'] if continuation_phase=='recipient_carry' else placement_goal['support_id']
            phase_endpoint=response['endpoint_hand_pose'];bilateral_since=None;stable_since=None;support_since=None
            unsupported_bilateral_s=0.
            return True
        setup_mark('dispatcher_policy_and_loop_preparation')
        owned_rigid16_log=False
        if rigid_profile is not None:
            from physics.substep_observer import RigidSubstepObserver
            from physics.episode_transport_assessment import rigid16_interval_facts
            from physics.rigid_pose_pack import RigidPosePack
            rigid16_pose_reader=RigidPosePack(scene,hands=hands,fingers=fingers,tray=tray,
                arm_names=tuple(arms),mode=rigid16_pose_readback,verify_scalar=rigid16_pose_verify_scalar,
                **({} if single_active_contract is None else {'pose_hands':{**hands,**fixed_peer_hands},'pose_fingers':{**fingers,**fixed_peer_fingers}}))
            capture_rigid16_state=rigid16_pose_reader.read
            from physics.episode_contacts import BoundEpisodeContactClassifier
            rigid16_contact_classifier=BoundEpisodeContactClassifier(owners=owners,components_by_geom=components,
                pads_by_arm=pads,stations_by_geom=station_map,**single_active_kwargs)
            result['rigid16_bound_contact_policy']=rigid16_contact_classifier.metadata()
            def consume_rigid16_capture(snapshot):
                if pending is None or pending['step']!=snapshot['outer_step_ordinal']+1:
                    raise ContractError('Actual internal capture is outside this issued interval')
                detection=snapshot['detection'];state=detection['rigid_state'];raw=detection['raw_contacts']
                legacy_classify=(rigid16_classifier_abba and (pending['step']<=32
                    or ((pending['step']-33)//32) in (0,3)))
                classify_now=classify_episode_live_contacts if legacy_classify else rigid16_contact_classifier.classify
                classified,bilateral=classify_now(raw,
                    **({'owners':owners,'components_by_geom':components,'pads_by_arm':pads,'stations_by_geom':station_map} if legacy_classify else {}),
                    body_pose=state['body_world'],finger_poses_by_arm={n:{'left_finger':r['finger_world']['left'],
                        'right_finger':r['finger_world']['right']} for n,r in state['arms'].items()},
                    phase=pending['phase'],active_holder_arm_id=pending['active_contact_permission_arm'])
                stamp=live_contact_coverage_stamp(scene,coverage,internal_capture=snapshot,
                    **({} if rigid16_transfer_reader is None else {'internal_readback':rigid16_transfer_reader}),
                    **({} if automatic_grasp is None else {'automatic_grasp':automatic_grasp}))
                sample={**snapshot,'classified_contacts':classified,'bilateral_by_arm':bilateral,'policy_stamp':stamp}
                rigid16_samples.append(sample)
                if (stamp['policy_valid_now'] is not True
                        or detection['native_contact_count']!=len(raw['geom_a']) or len(classified)!=len(raw['geom_a'])):
                    raise ContractError('Actual rigid16 internal native/copied/classified coverage is incomplete')
                if any(r['category']=='UNALLOWED_CONTACT' for r in classified):
                    raise ContractError('Unallowed actual rigid16 detection contact')
            if rigid16_readback!='legacy':
                if rigid16_readback in ('native','native-verify'):
                    from physics.rigid16_native_pack import NativeRigid16TransferPack
                    rigid16_transfer_reader=NativeRigid16TransferPack(rigid16_pose_reader,verify_legacy=rigid16_readback=='native-verify')
                else:
                    from physics.rigid16_transfer_pack import Rigid16TransferPack
                    rigid16_transfer_reader=Rigid16TransferPack(rigid16_pose_reader,verify_legacy=rigid16_readback=='verify')
            rigid16_observer=RigidSubstepObserver(scene.rigid_solver,consume_rigid16_capture,enabled=True,
                detection_capture=capture_rigid16_state,profile=rigid16_observer_profile,
                fence=profile_fence if rigid16_observer_profile else None,phase_readback=rigid16_transfer_reader)
            rigid16_observer.__enter__()
            from physics.student_episode_driver import StudentEpisodeDriver
            owned_rigid16_log=(rigid_profile==RIGID16_PROFILE and particle_request is None and not empty_mode
                and len(beads)==4 and observation_readback=='batched'
                and type(observation_reader) is EpisodeObservationReader
                and type(rigid16_pose_reader) is RigidPosePack and type(student_driver) is StudentEpisodeDriver)
        if particle_request is not None:
            def copy_particle_contact_instant():
                copy_started=time.perf_counter()
                count=np.asarray(scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()).copy()
                if count.size!=1 or not np.isfinite(count).all() or float(count.reshape(-1)[0])!=int(count.reshape(-1)[0]):
                    raise ContractError('Malformed native contact count at the actual detection instant')
                copied=_json_safe(dict(raw_contacts=_copy_contacts(scene.rigid_solver.collider),rigid_state=dict(
                    body_world=pose(tray),arms={n:{'finger_world':{k:pose(v) for k,v in fingers[n].items()}} for n in arms},
                    native_contact_count=int(count.reshape(-1)[0]))))
                particle_costs['last_internal_pose_contact_copy_calls']+=1
                particle_costs['last_internal_pose_contact_copy_inside_scene_step_host_s']+=time.perf_counter()-copy_started
                return copied
            particle_contact_copy=LastInternalContactCopy(scene,copy_particle_contact_instant,profile=particle_profile)
            particle_contact_copy.__enter__()
        empty_loop_started=time.perf_counter() if empty_timing else None
        if execution_backend=='device' and student_driver.mode=='tier_mission':
            from physics.device_tier_mission import execute_device_tier_mission
            if steps!=0 or int(scene.t)!=0 or observation['native_scene_step']!=0 or not source_check():
                raise ContractError('Tier device replay must own the original native INITIAL at clock zero')
            if rigid16_observer is None:raise ContractError('Original INITIAL observer is missing')
            rigid16_observer.assert_observed_through(0)
            rigid16_observer.__exit__(None,None,None);rigid16_observer=None
            original_handling_enforcer=handling_enforcer
            result['original_initial_handling_enforcement']=handling_enforcer.record()
            handling_enforcer=None
            result.pop('handling_enforcement',None)
            context=dict(scene=scene,student_driver=student_driver,automatic_grasp=automatic_grasp,
                arms=arms,beads=beads,tray=tray,coverage=coverage,owners=owners,components=components,
                pads=pads,station_map=station_map,gravity=gravity,limits_by_arm=limits_by_arm,
                finger_bounds=finger_bounds,hands=hands,kp=kp,kv=kv,effort=effort,case=case,
                execution=execution,ids=ids,model=model,handling_enforcer=original_handling_enforcer,
                world=world,scenario=scenario,initial_observation=initial,observation=observation,
                initial_fingerprint=fingerprint,
                observe=observe,source_check=source_check,settings=settings,rest_policy=rest_policy,
                rigid16_observer=None,particle_contact_copy=None)
            deadline=time.perf_counter()+wall_limit_s-(time.monotonic()-started)
            try:
                device=execute_device_tier_mission(context,output=output/'device_tier_execution',wall_deadline=deadline)
            finally:steps=int(scene.t)
            observation=copy.deepcopy(device['actual_final_observation'])
            save('device_tier_execution_result.json',device)
            result.update(status='TIER_MISSION_DEVICE_TRANSPORT_SAMPLED_PASS',execution_backend='device',
                device_tier_execution=device,device_tier_execution_sha256=digest(device),
                full_mission_passed=True,current_student_single_batch_mission_passed=False,
                completion_scope='provided_device_tier_mission_v1',final_observation=observation,
                physical_motion_count=len(student_driver.saved_plan.stage_signatures),motion_source='saved_cpu_plan',
                legacy_dispatcher_initial_only=True,legacy_full_episode_coverage_claimed=False)
            return result
        while True:
            if time.monotonic()-started>wall_limit_s:raise TimeoutError('Physical episode wall budget exceeded')
            if automatic_grasp_abba:
                automatic_grasp.set_diagnostic_readback('scalar' if steps<32 or ((steps-32)//32) in (0,3) else 'packed')
            if rigid16_combined_abba:
                rigid16_pose_reader.mode='scalar' if steps<32 or ((steps-32)//32) in (0,3) else 'packed'
            if profiler is not None:profiler.begin_step(int(scene.t)+1)
            command=dispatcher.prepare_next(verify_boundary=boundary)

            if execution_backend=='device' and command is not None and command.action_role=='carry':
                from physics.device_one_carry import execute_device_one_carry
                if not source_check():raise ContractError('Sources changed at genuine device carry boundary')
                if rigid16_observer is None:raise ContractError('Original fresh-pick observer is missing')
                rigid16_observer.assert_observed_through(steps)
                prefix_observer={'physical_steps':steps,'observed_internal_substeps':rigid16_observer.observed_count,
                    'actual_detection_substeps':rigid16_observer.detection_count,'scope':'Original checked INITIAL/pick prefix only'}
                rigid16_observer.__exit__(None,None,None)
                rigid16_observer=None
                result['legacy_prefix_observer']=prefix_observer
                original_handling_enforcer=handling_enforcer
                result['legacy_prefix_handling_enforcement']=handling_enforcer.record()
                handling_enforcer=None
                result.pop('handling_enforcement',None)
                pending=None
                context=dict(scene=scene,dispatcher=dispatcher,student_driver=student_driver,
                    automatic_grasp=automatic_grasp,arms=arms,beads=beads,tray=tray,
                    coverage=coverage,owners=owners,components=components,pads=pads,station_map=station_map,
                    gravity=gravity,limits_by_arm=limits_by_arm,finger_bounds=finger_bounds,hands=hands,
                    kp=kp,kv=kv,effort=effort,case=case,execution=execution,ids=ids,model=model,
                    handling_enforcer=original_handling_enforcer,
                    world=world,observation=observation,observe=observe,source_check=source_check,
                    settings=settings,rest_policy=rest_policy,rigid16_observer=None,particle_contact_copy=None,
                    fixed_peers=fixed_peers,single_active_arm_contract=single_active_contract)
                deadline=time.perf_counter()+wall_limit_s-(time.monotonic()-started)
                result['execution_backend']='device'
                result['legacy_dispatcher_prefix_only']=True
                try:
                    device=execute_device_one_carry(context,command,output=output/'device_execution',wall_deadline=deadline)
                finally:
                    steps=int(scene.t)
                observation=copy.deepcopy(device['tail']['actual_final_observation'])
                save('device_execution_result.json',device)
                result.update(status='ONE_CARRY_DEVICE_TRANSPORT_SAMPLED_PASS',device_execution=device,
                    device_execution_sha256=digest(device),one_carry_transport_accepted=True,
                    physical_motion_count=1,motion_source='saved_cpu_plan',final_observation=observation,
                    full_mission_passed=False,current_student_single_batch_mission_passed=False,
                    completion_scope='provided_device_one_carry_strict_handling_v1',legacy_full_episode_coverage_claimed=False)
                return result

            if profiler is not None:profiler.mark('dispatcher_prepare')
            if command is None:
                if student_driver is not None:
                    boundary_started=time.perf_counter() if empty_timing else None
                    extended=student_driver.advance(dispatcher,student_state())
                    if empty_timing:empty_video_costs['paused_provided_boundary_and_geometry_s']+=time.perf_counter()-boundary_started
                    if extended:
                        phase_support_id=student_driver.support_id
                        phase_endpoint=student_driver.endpoint_hand_pose
                        bilateral_since=None;stable_since=None;support_since=None
                        continue
                if phases is not None and continuation_boundary():continue
                break
            if int(scene.t)!=command.step-1 or abs(float(scene.cur_t)-command.pre_time_s)>1e-9:
                raise ContractError('Native pre-step clock differs from dispatcher')
            pre=observe();control={};motor_commands={};motor_requested={}
            if profiler is not None:profiler.mark('pre_observation')
            for c in command.arms:
                name=c.arm_id;arm=arms[name];q=np.asarray(pre['arms'][name]['q_all_dofs']);qd=np.asarray(pre['arms'][name]['qd_all_dofs'])
                tau=vector(gravity[name].torques(),9)
                if profiler is not None:profiler.mark('gravity_feedforward')
                if profile_gravity_parity_every and command.step%profile_gravity_parity_every==0:
                    clock=(int(scene.t),float(scene.cur_t))
                    expected=vector(gravity_parity[name].torques(),9)
                    first,second=gravity[name].last_evidence,gravity_parity[name].last_evidence
                    calls_compatible=first.public_calls==second.public_calls
                    if gravity_readback=='fused':
                        main_calls=dict(first.public_calls);ref_calls=dict(second.public_calls)
                        reference_jacobians=ref_calls.pop('get_jacobian',0)
                        backend=first.jacobian_evaluation or {}
                        calls_compatible=(main_calls==ref_calls and backend.get('public_calls')=={}
                            and backend.get('logical_evaluations')==reference_jacobians
                            and backend.get('collector_invocations')==1)
                    detail={'step':command.step,'arm':name,'maximum_effort_difference':float(np.max(np.abs(tau-expected))),
                        'effort_exactly_equal':bool(np.array_equal(tau,expected)),
                        'mass_delta_equal':first.maximum_mass_change_kg==second.maximum_mass_change_kg,
                        'local_com_delta_equal':first.maximum_local_com_change_m==second.maximum_local_com_change_m,
                        'public_calls_equal':first.public_calls==second.public_calls,
                        'jacobian_accounting_verified':bool(calls_compatible),
                        'same_native_clock':clock==(int(scene.t),float(scene.cur_t))}
                    if gravity_readback=='fused':
                        detail['jacobian_matrix_check']=gravity[name].verify_fused_jacobians()
                    result['profile_gravity_parity']['pairs'].append(detail)
                    if not all(detail[k] for k in ('effort_exactly_equal','mass_delta_equal','local_com_delta_equal','jacobian_accounting_verified','same_native_clock')):
                        raise ContractError('Packed gravity differs from fresh scalar gravity: '+json.dumps(detail))
                    if gravity_readback=='fused' and not all(detail['jacobian_matrix_check'][k] for k in ('matrix_exactly_equal','output_not_aliased_to_scratch','same_native_clock')):
                        raise ContractError('Fused native Jacobian matrices differ: '+json.dumps(detail))
                    if profiler is not None:profiler.mark('diagnostic_gravity_parity')
                target=np.r_[c.q_rad,c.finger_q_m];target_v=np.r_[c.qd_rad_s,c.finger_qd_m_s]
                requested=tau+kp*(target-q)+kv*(target_v-qd)
                if motor_batch is None:
                    arm.set_dofs_act_bias(tau,-kp,-kv)
                    arm.control_dofs_position_velocity(target,target_v)
                    motor=vector(arm.get_dofs_control_force(),9)
                    error=float(np.max(np.abs(motor-np.clip(requested,-effort,effort))))
                    if np.any(np.abs(motor)>effort+1e-5) or error>.005:raise ContractError('Native bounded motor command mismatch')
                    control[name]={'gravity_effort':tau.tolist(),'requested_unclipped_effort':requested.tolist(),'motor_effort':motor.tolist(),
                        'effort_identity_error':error,'effort_saturated':(np.abs(requested)>effort).tolist(),
                        'gravity_evidence':gravity[name].last_evidence.to_record()}
                else:
                    motor_commands[name]={'tau':tau,'target':target,'target_v':target_v};motor_requested[name]=requested
                if profiler is not None:profiler.mark('motor_control_checks')
            if motor_batch is not None:
                motors=motor_batch.apply(motor_commands)
                for name in motor_commands:
                    tau=motor_commands[name]['tau'];requested=motor_requested[name];motor=vector(motors[name],9)
                    error=float(np.max(np.abs(motor-np.clip(requested,-effort,effort))))
                    if np.any(np.abs(motor)>effort+1e-5) or error>.005:raise ContractError('Native bounded motor command mismatch')
                    control[name]={'gravity_effort':tau.tolist(),'requested_unclipped_effort':requested.tolist(),'motor_effort':motor.tolist(),
                        'effort_identity_error':error,'effort_saturated':(np.abs(requested)>effort).tolist(),
                        'gravity_evidence':gravity[name].last_evidence.to_record()}
                if profiler is not None:profiler.mark('motor_control_checks')
                if profile_gravity_parity_every and command.step%profile_gravity_parity_every==0:
                    for name,arm in arms.items():
                        expected=vector(arm.get_dofs_control_force(),9)
                        detail={'step':command.step,'arm':name,'maximum_effort_difference':float(np.max(np.abs(expected-motors[name]))),
                            'effort_exactly_equal':bool(np.array_equal(expected,motors[name]))}
                        result['profile_motor_parity'].append(detail)
                        if not detail['effort_exactly_equal']:raise ContractError('Batched motor force differs from entity getter: '+json.dumps(detail))
                    if profiler is not None:profiler.mark('diagnostic_motor_parity')
            phase={'initial_settle':'settle','approach':'approach','retreat':'approach','recipient_approach':'approach','approach_settle':'settle','close':'close','final_hold':'hold',
                   'carry':'carry','place':'place','release':'release','released_settle':'released_settle'}[command.action_role]
            holder=command.active_arm_id if phase in ('close','hold','carry','place','release','released_settle') else None
            if continuation_phase=='donor_retreat':holder=None
            pending={'step':command.step,'command':command.to_record(),'phase':phase,'pre':pre,'control':control,'post_processing_complete':False}
            if rigid_profile is not None:
                rigid16_samples=[]
                pending.update(rigid_profile=rigid_profile,rigid_substep_samples=rigid16_samples,
                    active_contact_permission_arm=holder)
            if profiler is not None:profiler.mark('command_bookkeeping')
            if automatic_grasp is not None:handling_enforcer.capture_automatic_grasp(pre)
            physics_started=time.perf_counter() if particle_request is not None or empty_timing else None
            scene.step()
            if empty_timing:empty_video_costs['scene_step_with_native_recording_s']+=time.perf_counter()-physics_started
            if particle_request is not None:
                particle_costs['scene_step_including_last_internal_copy_host_s']+=time.perf_counter()-physics_started
            if profiler is not None:profiler.mark('physics_step')
            steps=int(scene.t);scene.rigid_solver.check_errno()
            if profiler is not None:profiler.mark('solver_error_check')
            contact_started=time.perf_counter() if particle_request is not None else None
            last_internal=None;contact_state=pre
            rigid16_facts=None
            if rigid_profile is not None:
                rigid16_observer.assert_observed_through(steps)
                legacy_comparison=(rigid16_combined_abba and (steps<=32 or ((steps-33)//32) in (0,3)))
                if handling_enforcer is None or legacy_comparison:
                    rigid16_facts=rigid16_interval_facts(rigid16_samples,step=steps,dt=DT,
                        coverage_policy_sha256=coverage['coverage_policy_sha256'])
                last_rigid=rigid16_samples[-1]
                raw=copy.deepcopy(last_rigid['detection']['raw_contacts'])
                contact_state=last_rigid['detection']['rigid_state']
                pending['contact_detection_state']=copy.deepcopy(contact_state)
                pending['contact_detection_time_s']=last_rigid['detection']['preintegration_native_clock'][3]
            elif particle_request is not None:
                last_internal=particle_contact_copy.require_completed_outer_sample(steps)
                raw=copy.deepcopy(last_internal['sample']['raw_contacts'])
                contact_state=last_internal['sample']['rigid_state']
                pending['contact_detection_state']=copy.deepcopy(contact_state)
                pending['contact_detection_time_s']=last_internal['preintegration_time_s']
            else:raw=_copy_contacts(scene.rigid_solver.collider)
            pending['raw_contacts']=raw
            if profiler is not None:profiler.mark('contact_copy')
            contact_classifier=(classify_episode_live_contacts if particle_request is None else classify_particle_episode_live_contacts)
            if rigid_profile is not None:
                rows=copy.deepcopy(last_rigid['classified_contacts']);bilateral=copy.deepcopy(last_rigid['bilateral_by_arm'])
            else:
                rows,bilateral=contact_classifier(raw,owners=owners,components_by_geom=components,pads_by_arm=pads,
                    body_pose=contact_state['body_world'],finger_poses_by_arm={n:r['finger_world'] for n,r in contact_state['arms'].items()},
                    stations_by_geom=station_map,phase=phase,active_holder_arm_id=holder,
                    **({} if particle_request is None else {'canonical_geometry':model.geometry_record()}))
            if profiler is not None:profiler.mark('contact_classification')
            automatic_stamp={} if automatic_grasp is None else {'automatic_grasp':automatic_grasp}
            stamp=(live_contact_coverage_stamp(scene,coverage,**automatic_stamp) if particle_request is None else particle_contact_coverage_stamp(scene,coverage,**automatic_stamp))
            native_count=stamp['native_contact_count']
            contact_complete=stamp['policy_valid_now'] is True and native_count==len(raw['geom_a'])==len(rows)
            if not contact_complete:raise ContractError('Native copied/classified contact coverage incomplete')
            if particle_request is not None:
                particle_costs['post_step_contact_classification_coverage_host_s']+=time.perf_counter()-contact_started
            if profiler is not None:profiler.mark('contact_coverage')
            observation=observe()
            if profiler is not None:profiler.mark('post_observation')
            if steps!=command.step or abs(observation['simulation_s']-command.post_time_s)>1e-9:
                raise ContractError('Native completed clock differs from issued interval')
            failures=[]
            handling_interval=None
            if handling_enforcer is not None:
                handling_interval=handling_enforcer.observe_interval(pre=pre,post=observation,
                    classified_contacts=rows,bilateral_by_arm=bilateral,contacts_complete=contact_complete,
                    active_contact_permission_arm=holder,
                    **({'rigid_profile':rigid_profile,'rigid_substep_samples':rigid16_samples} if rigid_profile is not None else
                        {} if particle_request is None else {'contact_detection_clock':last_internal['preintegration_native_clock']}))
                if not handling_interval['passed']:
                    failures.append({'kind':'native_interval_handling_guard',
                                     'detail':handling_interval['first_failure']})
                if rigid_profile is not None:
                    rigid16_facts=handling_interval.get('interval_facts')
                    if (not isinstance(rigid16_facts,dict)
                            or rigid16_facts.get('coverage_policy_sha256')!=coverage['coverage_policy_sha256']):
                        raise ContractError('Owned rigid16 handling facts lack the current coverage policy')
            for c in command.arms:
                name=c.arm_id;actual=observation['arms'][name];q=np.asarray(actual['q_all_dofs']);qd=np.asarray(actual['qd_all_dofs'])
                acceleration=(qd[:7]-np.asarray(pre['arms'][name]['qd_all_dofs'][:7]))/DT
                actual['qdd_fd_rad_s2']=acceleration.tolist();actual['transmitted_effort_diagnostic']=vector(sensors[name].read_ground_truth(),9).tolist()
                tracking=float(np.max(np.abs(q[:7]-c.q_rad)));actual['tracking_error_rad']=tracking
                limits=limits_by_arm[name]
                if tracking>execution['tracking_error_rad']:failures.append({'kind':'joint_tracking','arm':name,'error_rad':tracking})
                if any(not lo-1e-5<=v<=hi+1e-5 for v,(lo,hi) in zip(q,(*limits.position_rad,*finger_bounds[name]))):failures.append({'kind':'native_position_limit','arm':name})
                if np.any(np.abs(qd[:7])>np.asarray(limits.velocity_rad_s)+1e-6) or np.any(np.abs(acceleration)>np.asarray(limits.acceleration_rad_s2)+1e-5):
                    failures.append({'kind':'measured_derivative_limit','arm':name})
                if any(control[name]['effort_saturated']):failures.append({'kind':'motor_effort_saturation','arm':name})
                xyz=np.asarray(actual['hand_world']['position_m']);workspace=np.asarray(case['tool_workspace_m'])
                if np.any(xyz<workspace[:,0]) or np.any(xyz>workspace[:,1]):failures.append({'kind':'measured_tool_workspace','arm':name})
            bad=[r for r in rows if r['category']=='UNALLOWED_CONTACT']
            if bad:failures.append({'kind':'unallowed_observed_contact','contacts':bad})
            if empty_mode:
                if observation['materials']!=[]:failures.append({'kind':'unexpected_material_in_empty_payload'})
            elif particle_request is None:
                if any(not m['inside_carrier'] for m in observation['materials']):failures.append({'kind':'persistent_content_outside'})
            else:
                try:validate_particle_packet(observation['particle_material'],particle_reader.contract,observation,require_retained=True)
                except ContractError as exc:failures.append({'kind':'native_particle_material_guard','detail':str(exc)})
            tilt_bound=(student_driver.body_tilt_limit_deg if student_driver is not None else execution_body_tilt_limit_deg)
            if (transport is not None or student_driver is not None) and observation['body_up_tilt_deg']>tilt_bound:
                failures.append({'kind':'reference_body_tilt_constraint','actual_deg':observation['body_up_tilt_deg'],
                    'maximum_deg':tilt_bound})
            counted_contacts=rows if rigid_profile is None else [r for s in rigid16_samples for r in s['classified_contacts']]
            for row in counted_contacts:counts[row['category']]+=1
            now=observation['simulation_s']
            if empty_mode:
                stable_since=None  # No material bulk exists. Arm/body/support rest remains mandatory.
            elif particle_request is None:
                good_material=all(m['inside_carrier'] and np.linalg.norm(m['velocity_world_m_s'])<=settings['scenario']['stable_speed_m_s'] for m in observation['materials'])
                stable_since=(now if stable_since is None else stable_since) if good_material else None
            elif now>=.5:
                material_rest=window_rest([*recent,observation],now,particle_reader.contract)
                stable_since=(now-.5 if stable_since is None else stable_since) if material_rest['passed'] else None
            else:stable_since=None
            if rigid_profile is not None:
                counters=rigid16_duration_counters(rigid16_facts['timeline'],phase=phase,active_arm_id=command.active_arm_id,
                    support_id=phase_support_id,state=dict(bilateral_since=bilateral_since,support_since=support_since,
                        unsupported_bilateral_s=unsupported_bilateral_s,maximum_unsupported_bilateral_s=maximum_unsupported_bilateral_s,
                        maximum_unsupported_bilateral_by_arm_s=maximum_unsupported_bilateral_by_arm_s,
                        unsupported_provided_grasp_s=unsupported_provided_grasp_s,
                        maximum_unsupported_provided_grasp_by_arm_s=maximum_unsupported_provided_grasp_by_arm_s))
                bilateral_since=counters['bilateral_since'];support_since=counters['support_since']
                unsupported_bilateral_s=counters['unsupported_bilateral_s'];maximum_unsupported_bilateral_s=counters['maximum_unsupported_bilateral_s']
                maximum_unsupported_bilateral_by_arm_s=counters['maximum_unsupported_bilateral_by_arm_s']
                unsupported_provided_grasp_s=counters['unsupported_provided_grasp_s']
                maximum_unsupported_provided_grasp_by_arm_s=counters['maximum_unsupported_provided_grasp_by_arm_s']
            else:
                good_bilateral=phase=='hold' and bilateral.get(command.active_arm_id) is True
                bilateral_since=(now if bilateral_since is None else bilateral_since) if good_bilateral else None
                supported_now=any(r['category']=='named_tray_bottom_support_top' and r.get('support_station_id')==phase_support_id for r in rows)
                support_since=(now if support_since is None else support_since) if supported_now else None
                unsupported=(phase=='carry' and bilateral.get(command.active_arm_id) is True and not any(r['category']=='named_tray_bottom_support_top' for r in rows))
                unsupported_bilateral_s=unsupported_bilateral_s+DT if unsupported else 0.
                maximum_unsupported_bilateral_s=max(maximum_unsupported_bilateral_s,unsupported_bilateral_s)
                maximum_unsupported_bilateral_by_arm_s[command.active_arm_id]=max(
                    maximum_unsupported_bilateral_by_arm_s[command.active_arm_id],unsupported_bilateral_s)
                if automatic_grasp is not None:
                    actual_holder=automatic_grasp.holder
                    provided_unsupported=(actual_holder is not None and phase=='carry' and not any(r['category']=='named_tray_bottom_support_top' for r in rows))
                    unsupported_provided_grasp_s=unsupported_provided_grasp_s+DT if provided_unsupported else 0.
                    if actual_holder is not None:
                        maximum_unsupported_provided_grasp_by_arm_s[actual_holder]=max(maximum_unsupported_provided_grasp_by_arm_s[actual_holder],unsupported_provided_grasp_s)
            if profiler is not None:profiler.mark('physical_checks_and_sensors')
            row={**pending,**observation,'classified_contacts':rows,'bilateral_by_arm':bilateral,
                'active_contact_permission_arm':holder,'coverage':{'policy_stamp':stamp,'step':steps,
                'pre_time_s':pre['simulation_s'],'post_time_s':now,'native_contact_count':native_count,
                'copied_contact_count':len(raw['geom_a']),'classified_contact_count':len(rows),'contacts_complete':contact_complete},
                'failures':failures,'post_processing_complete':True}
            if particle_request is not None:
                row['coverage']['last_internal_sample']=copy.deepcopy(last_internal)
                validate_particle_contact_coverage(row,steps,DT,rows)
            if rigid_profile is not None:
                row['coverage']['rigid16_interval_facts']=copy.deepcopy(rigid16_facts)
                row['coverage']['contact_scope']='Final actual h detection in outer summary. Every actual16 detection/raw/pose/classification/stamp is in rigid_substep_samples.'
            if handling_interval is not None:row['handling_interval']=handling_interval
            if profiler is not None:profiler.mark('row_assembly')
            log_started=time.perf_counter() if particle_request is not None else None
            if compact_log is None:
                direct_log=owned_rigid16_log and _json_can_encode_unchanged(row)
                safe_row=row if direct_log else _json_safe(row)
                if profiler is not None:profiler.mark('log_json_safe')
                encoded=_encode_owned_step_log(safe_row,direct=direct_log)
                if profiler is not None:profiler.mark('log_encode')
                stream.write(encoded)
                if profiler is not None:profiler.mark('log_write')
                stream.flush()
            else:
                compact_log.write_step(row)
                if profiler is not None:profiler.mark('compact_encode_write_flush')
            pending=None
            if particle_request is not None:
                particle_costs['trace_encode_write_flush_host_s']+=time.perf_counter()-log_started
            if profiler is not None:profiler.mark('log_flush')
            recent.append(row)
            if len(recent)>round(2./DT):recent.pop(0)
            dispatcher.commit_step(step=steps,simulation_s=now,observation_sha256=digest(observation),physical_step_completed=True)
            if profiler is not None:profiler.mark('recent_and_commit_digest')
            if camera is not None and record_video and not empty_video and steps%20==0:camera.render()
            if time.monotonic()-last_progress>=10:
                save('progress.json',{'phase':command.action_role,'physical_steps':steps,'simulation_s':now,'elapsed_wall_s':time.monotonic()-started});last_progress=time.monotonic()
            if failures:raise ContractError('Observed episode failure: '+json.dumps(failures,allow_nan=False))
            if student_driver is not None:
                from physics.provided_manipulation import native_control_target_receipt
                provided_control_targets=native_control_target_receipt(scene,arms,command,plan_sha256=dispatcher.current_segment.plan.sha256)
                with (output/'provided_native_control_targets.jsonl').open('a',encoding='utf-8') as targets_stream:
                    targets_stream.write(json.dumps(provided_control_targets,separators=(',',':'),allow_nan=False)+'\n')
                student_driver.observe_step(student_state())
                if private_contact_fixture:
                    student_driver.capture_private_contact_profile(student_state(),dispatcher.record())
            if profiler is not None:profiler.end_step(command.action_role)
            if student_driver is not None and student_driver.step_limit is not None and steps>=student_driver.step_limit:
                result.update(status='STUDENT_EXECUTION_SHORT_CUTOFF',short_profile_only=True,
                    student_execution=student_driver.record(),cutoff_scope='Complete checks for committed steps only. Operation may be unfinished.')
                return result
            if profiler is not None:
                if steps>=profiler.max_steps:
                    result.update(status='SHORT_PROFILE_COMPLETED',short_profile_only=True,
                        profile_scope='Actual fresh-start prefix only. No final grasp, carry, handoff or delivery acceptance.')
                    return result
        if empty_timing:
            empty_video_costs['loop_total_s']=time.perf_counter()-empty_loop_started
            empty_video_costs['controls_checks_and_logs_s']=(empty_video_costs['loop_total_s']
                -empty_video_costs['scene_step_with_native_recording_s']
                -empty_video_costs['paused_provided_boundary_and_geometry_s'])
        final=observation
        if particle_request is None:
            final_mass=[float(array(b.get_mass()).sum()) for b in beads]
            if final_mass!=[v['mass_kg'] for v in ids]:raise ContractError('Persistent native material mass changed')
        else:
            particle_reader.verify_source_bindings()
            validate_particle_packet(final['particle_material'],particle_reader.contract,final,require_retained=True)
        if student_driver is not None:
            result.update(student_driver.final_result(final))
        elif phases is not None:
            status=(('THREE_ARM_SINGLE_DELIVERY_SAMPLED_PASS' if repeated_three_arm_mode else 'SINGLE_DELIVERY_SAMPLED_PASS') if result['continuation']['single_delivery_verified'] else
                'RECIPIENT_SUPPORTED_GRASP_SAMPLED_PASS' if result['continuation']['recipient_supported_grasp_verified'] else 'DONOR_RETREAT_SAMPLED_PASS')
            result.update(status=status,
                maximum_unsupported_bilateral_s=maximum_unsupported_bilateral_s,final_observation=final,approach_executed=True)
        elif transport is None:
            gate,spec=supported_grasp_gate(final,recent,settings,arms,scenario,rest_policy,rows,bilateral_since,stable_since,support_since,model)
            result.update(status='SUPPORTED_GRASP_SAMPLED_PASS' if gate['passed'] else 'SUPPORTED_GRASP_GATE_FAIL',
                supported_grasp_gate=gate,measured_held_body=spec.to_record(),measured_held_binding_sha256=spec.binding_sha256,
                final_observation=final,approach_executed=True,supported_grasp_verified=gate['passed'])
            save('supported_grasp_gate.json',gate)
        else:
            placement_goal=transport['goal'].get('seating',{}).get('nominal_goal',transport['goal'])
            gate=released_relay_gate(recent,final['simulation_s'],settings,arms,rest_policy,placement_goal,finger_bounds,ids,
                dt=DT,window_s=TRANSPORT_PROTOCOL['acceptance_window_s'])
            load_bearing=maximum_unsupported_bilateral_s>=TRANSPORT_PROTOCOL['acceptance_window_s']
            result.update(status='RELAY_RELEASE_SAMPLED_PASS' if gate['passed'] and load_bearing else 'RELAY_RELEASE_GATE_FAIL',
                relay_release_gate=gate,relay_placement_verified=gate['passed'],load_bearing_grasp_verified=load_bearing,
                maximum_unsupported_bilateral_s=maximum_unsupported_bilateral_s,final_observation=final,approach_executed=True)
            save('relay_release_gate.json',gate)
    except Exception as exc:
        result.update(status='EPISODE_EXECUTION_FAIL',failure={'type':type(exc).__name__,'message':str(exc)},traceback=traceback.format_exc())
        result.update(full_mission_passed=False,current_student_single_batch_mission_passed=False)
        if rigid16_transfer_reader is not None and rigid16_transfer_reader.failed:
            pending={**({} if pending is None else pending),
                'rigid16_failed_phase_packet':rigid16_transfer_reader.failure_data()}
        if compact_log is not None:
            try:compact_log.record_failure(exc,pending=pending,observation=observation,
                stage=phase if 'phase' in locals() else None)
            except Exception as nested:result['cleanup_errors'].append('compact failure context: '+repr(nested))
        if isinstance(result.get('mission_completion_receipt'),dict):
            result['mission_completion_receipt'].update(passed=False,revoked_reason='runtime_exception')
        if result.get('material_delivery_accepted'):
            result['material_delivery_accepted']=False
            if 'continuation' in result:result['continuation']['single_delivery_verified']=False
            result['physical_handoff_verified']=False
            for key in ('delivered_material_ids','delivered_count','delivered_mass_kg','physical_handoff_count'):result.pop(key,None)
        if pending is not None and stream is not None:
            try:
                stream.write(json.dumps(_json_safe({**pending,'failure_during_observation':result['failure']}),ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            except Exception as nested:result['cleanup_errors'].append('partial trace: '+repr(nested))
    finally:
        result['rigid16_transfer_pack']=({'mode':rigid16_readback,'initialized':False} if rigid16_transfer_reader is None
            else rigid16_transfer_reader.metadata())
        if 'rigid16_pose_reader' in locals():
            result['rigid16_pose_readback']=rigid16_pose_reader.metadata()
        if 'rigid16_contact_classifier' in locals():
            result['rigid16_bound_contact_policy']=rigid16_contact_classifier.metadata()
            if rigid16_classifier_abba:
                result['rigid16_classifier_abba']={'warmup_steps':32,'block_steps':32,
                    'modes':['legacy','bound','bound','legacy'],
                    'scope':('Classifier selector within the combined comparison. See rigid16_combined_abba for all toggled work.'
                        if rigid16_combined_abba else 'Same scene and unchanged physics/checks. Only static registry revalidation differs.')}
            if rigid16_combined_abba:
                result['rigid16_combined_abba']={'warmup_steps':32,'block_steps':32,
                    'modes':['legacy','optimized','optimized','legacy'],
                    'scope':'Same scene, commands, physics and safety predicates. Compares pose/grasp readback, static registry validation and duplicate interval validation together.'}
        if rigid16_observer is not None:
            try:
                try:
                    if 'failure' not in result:rigid16_observer.assert_observed_through(steps)
                finally:
                    rigid16_observer.__exit__(None,None,None)
                restored_substep='substep' not in vars(scene.rigid_solver)
                restored_constraint='_func_constraint_force' not in vars(scene.rigid_solver)
                if not restored_substep or not restored_constraint:
                    raise ContractError('Rigid16 owned native instance hooks did not restore')
                result['rigid16_contact_observer']={'observed_internal_substeps':rigid16_observer.observed_count,
                    'actual_detection_substeps':rigid16_observer.detection_count,
                    'original_constraint_calls':rigid16_observer.constraint_original_calls,
                    'last_global_ordinal':rigid16_observer.last_global_ordinal,'failed':rigid16_observer.failed,
                    'active_after_exit':rigid16_observer.active,'expected_completed_internal_substeps':steps*16,
                    'native_substep_method_restored':restored_substep,'native_constraint_method_restored':restored_constraint,
                    'all_completed_intervals_observed':rigid16_observer.observed_count==steps*16,
                    'scope':'Every actual internal detection and produced post state. Partial failing intervals earn no completion credit. No continuous-time certificate.'}
                if rigid16_observer_profile:result['rigid16_observer_timing']=rigid16_observer.timing_record()
            except Exception as exc:result['cleanup_errors'].append('rigid16 contact observer: '+repr(exc))
        if automatic_grasp is not None:
            result['automatic_grasp_readback']=automatic_grasp.readback_metadata()
            if automatic_grasp_abba:
                result['automatic_grasp_abba']={'warmup_steps':32,'block_steps':32,
                    'modes':['scalar','packed','packed','scalar'],
                    'scope':('Grasp readback selector within the combined comparison. See rigid16_combined_abba for all toggled work.'
                        if rigid16_combined_abba else 'Same scene and physics/checks. Only fresh provided-grasp readback selection differs.')}
            result['provided_automatic_grasp']={'model':'provided_ideal','holder_at_end':automatic_grasp.holder,
                'events':copy.deepcopy(student_driver.automatic_events),'boundary_events':copy.deepcopy(automatic_boundary_events),
                'scope':'Actual native provided attachment policy. Not a friction-retention certificate.'}
            result['runtime_welds']=sum(e.get('operation',e.get('event'))=='attach' for e in student_driver.automatic_events)
        if particle_request is not None:
            result['particle_host_costs']={'totals':dict(particle_costs),
                'completed_native_material_reads':None if particle_reader is None else int(particle_reader.reads),
                'read_count_scope':'Successful actual ParticleMaterialReader.read calls, including any planning pauses. Failed partial reads are excluded. No previous material endpoint is reused.',
                'clock':dict(name='perf_counter',implementation=time.get_clock_info('perf_counter').implementation,
                    resolution_s=time.get_clock_info('perf_counter').resolution),
                'scope':'Observed host wall durations with native waits where reads require them. No extra GPU timing fence. scene_step includes the two last-internal copies. Their separate duration is a nested subtotal, not additive. Observation includes rigid read, one owned fresh material/pause bundle, C/Jp, archive and pause hashes. Not kernel-only GPU timing.'}
            if particle_request.get('runtime_profile')==particle_bridge.CACHE_RUNTIME_PROFILE and particle_reader is not None:
                try:
                    particle_reader._topology()
                    result['particle_cache_runtime']={'initial_binding':dict(particle_reader._cache_initial),
                        'final':particle_reader.solver.cache_receipt(),
                        'invocation_origin':particle_reader._cache_invocation_origin,
                        'native_rigid_tolerance':float(np.asarray(scene.rigid_solver.rigid_info.tolerance.to_numpy()).item()),
                        'scope':'Actual cache lifecycle at INITIAL and final current endpoint. Initial last-cache fields may describe build warmup. No all-internal or whole-model equivalence certificate.'}
                except Exception as exc:result['cleanup_errors'].append('particle cache lifecycle: '+repr(exc))
        if particle_contact_copy is not None:
            try:
                particle_contact_copy.__exit__()
                result['last_internal_contact_observer']={'original_calls':particle_contact_copy.calls,
                    'actual_last_internal_samples':particle_contact_copy.samples,'restored':particle_contact_copy.restored,
                    'scope':'One actual last25us sample per2.5ms interval. Not continuous or all100-subinterval coverage.'}
            except Exception as exc:result['cleanup_errors'].append('particle contact observer: '+repr(exc))
        if particle_archive is not None:
            try:
                if 'failure' in result:particle_archive._fail(RuntimeError(result['failure']['message']))
                else:particle_archive.close()
                result['particle_archive_manifest']='particle_archive/manifest.json'
            except Exception as exc:result['cleanup_errors'].append('particle archive: '+repr(exc))
        if particle_cpic_diagnostic and particle_reader is not None and steps>0:
            try:
                raise ContractError('This hardware package supports the fixed four-bead tiers only')
                result['post_outer_cpic_diagnostic']=capture_post_outer_cpic(scene,particle_reader,output/'cpic_diagnostic')
            except Exception as exc:
                result.setdefault('diagnostic_errors',[]).append('post-outer CPIC: '+repr(exc))
        if particle_world_quality_diagnostic:
            world_diagnostic={'capture_completed':False,'status':'FAILED_NOT_CAPTURED',
                'native_pass_claimed':False,'ordinary_loss_or_delivery_credit_assigned':False,
                'scope':'One uncommitted post-outer endpoint diagnostic. UNKNOWN remains UNKNOWN. No material/mission gate.'}
            world_diagnostic_started=time.perf_counter()
            world_reader_reads_before=None if particle_reader is None else int(particle_reader.reads)
            try:
                if (scene is None or not scene.is_built or steps!=1 or int(scene.t)!=1
                        or particle_reader is None or observation_reader is None or not isinstance(observation,dict)
                        or not isinstance(locals().get('components'),dict) or not isinstance(locals().get('owners'),dict)
                        or not isinstance(locals().get('union_binding'),dict)):
                    raise ContractError('World-quality diagnostic requires one completed built particle endpoint and actual native geometry/union bindings')
                import hashlib
                raise ContractError('This hardware package supports the fixed four-bead tiers only')
                world_helper=Path(world_quality.__file__).resolve()
                if (world_helper!=Path(__file__).resolve().parents[1]/'qa/all_material_physics_20261031/world_quality_branch/read_other_world_particle_quality.py'
                        or hashlib.sha256(world_helper.read_bytes()).hexdigest()!='0530081c16c0bd4c205ad1439bca20b58f50f96906099052a45654f69262cecb'):
                    raise ContractError('Exact unchanged V2 world-quality helper required')
                captured=world_quality.capture_other_world_particle_quality(observation_reader,observation,
                    components_by_geom=components,owners=owners,canonical_union_binding=union_binding)
                if (type(captured) is not dict or set(captured)!={'record','arrays'}
                        or type(captured['record']) is not dict or captured['record'].get('native_pass_claimed') is not False
                        or type(captured['arrays']) is not dict or not captured['arrays']
                        or not all(type(a) is np.ndarray and a.dtype.kind in 'biuf' for a in captured['arrays'].values())):
                    raise ContractError('Malformed world-quality diagnostic result')
                world_record=captured['record'];world_arrays=captured['arrays']
                world_array_path=output/'particle_world_quality_endpoint_arrays.npz'
                world_record_path=output/'particle_world_quality_endpoint_record.json'
                if world_array_path.exists() or world_record_path.exists():raise ContractError('World-quality endpoint output already exists')
                np.savez(world_array_path,**world_arrays)
                save(world_record_path.name,world_record)
                unknown=world_record['unknown_particle_count']
                outcome=('UNKNOWN' if unknown else 'DEEP_OR_READER_QUALITY_FAILURE_OBSERVED'
                    if world_record['native_sdf_deep_count'] or not world_record['reader_quality_all_IDs_passed'] else 'KNOWN_NON_DEEP_ENDPOINT_ONLY')
                world_diagnostic.update(capture_completed=True,status='CAPTURED_DIAGNOSTIC_ONLY',outcome=outcome,
                    record_file=world_record_path.name,record_sha256=hashlib.sha256(world_record_path.read_bytes()).hexdigest(),
                    arrays_file=world_array_path.name,arrays_sha256=hashlib.sha256(world_array_path.read_bytes()).hexdigest(),
                    arrays_bytes=world_array_path.stat().st_size,native_clock=world_record['native_clock'],
                    unknown_particle_count=unknown,native_sdf_deep_count=world_record['native_sdf_deep_count'])
            except Exception as exc:
                world_diagnostic.update(status='FAILED_NOT_CAPTURED',error_type=type(exc).__name__,error=repr(exc))
                result.setdefault('diagnostic_errors',[]).append('post-outer world quality: '+repr(exc))
                try:save('particle_world_quality_diagnostic_failure.json',world_diagnostic)
                except Exception as write_error:world_diagnostic['failure_record_write_error']=repr(write_error)
            finally:
                world_diagnostic.update(host_wall_s=time.perf_counter()-world_diagnostic_started,
                    particle_reader_reads_before=world_reader_reads_before,
                    particle_reader_reads_after=None if particle_reader is None else int(particle_reader.reads),
                    cost_scope='Late optional diagnostic and fresh Reader.read are separate from the already captured particle_host_costs totals/count snapshot.')
                result['post_outer_world_quality_diagnostic']=world_diagnostic
        if student_driver is not None:
            try:result['student_execution']=student_driver.record()
            except Exception as exc:result['cleanup_errors'].append('student driver record: '+repr(exc))
        if 'handling_enforcer' in locals() and handling_enforcer is not None:
            result['handling_enforcement']=handling_enforcer.record()
            save('handling_enforcement_final.json',result['handling_enforcement'])
        setup_mark('physical_loop_and_failure_handling')
        if profiler is not None:result['step_profile']=profiler.record()
        if observation_readback!='scalar':
            try:
                result['observation_readback']={'mode':observation_readback,
                    'reader':None if observation_reader is None else observation_reader.metadata()}
            except Exception as exc:
                result['cleanup_errors'].append('observation reader metadata: '+repr(exc))
        if compact_log is not None:
            try:
                receipt=compact_log.finish()
                if 'failure' not in result and receipt.get('episode_success_eligible') is not True:
                    raise ContractError('Compact log cannot certify successful publication')
            except Exception as exc:result['cleanup_errors'].append('compact trace finalization: '+repr(exc))
            result['step_log']=compact_log.receipt()
        else:result['step_log']={'mode':'full','normal_record_schema':'original-full-step'}
        if execution_backend=='device':result['step_log']['execution_scope']='legacy_INITIAL_pick_prefix_only'
        close_trace_stream(stream,result)
        if recording:
            try:
                video_stop_started=time.perf_counter() if empty_video else None
                camera.stop_recording()
                if empty_video:empty_video_costs['video_encoder_close_s']=time.perf_counter()-video_stop_started
            except Exception as exc:result['cleanup_errors'].append('recording: '+repr(exc))
        if empty_timing:
            result['empty_video_wall_costs' if empty_video else 'empty_execution_wall_costs']={
                'native_recording_enabled':bool(empty_video),'measured_host_s':dict(empty_video_costs),
                'scope':'loop_total includes paused boundary/geometry, scene.step (including native recording only when enabled), and controls/checks/logs. These three parts sum to loop_total. Encoder close is outside the loop. GPU work is charged where waited/read, not kernel-only profiling.'}
        if camera is not None and scene is not None and scene.is_built and (student_driver is None or not student_driver.sparse_capture_only):
            try:
                from PIL import Image
                Image.fromarray(camera.render()[0]).save(output/'final_native_scene.png')
            except Exception as exc:result['cleanup_errors'].append('final image: '+repr(exc))
        final_clock=observation['simulation_s'] if observation is not None else None
        try:
            if scene is not None and scene.is_built:final_clock=float(scene.cur_t)
        except Exception as exc:result['cleanup_errors'].append('final clock: '+repr(exc))
        result.update(physical_steps=steps,physical_execution_performed=steps>0,simulation_s=final_clock,
            dispatch=dispatcher.record() if dispatcher is not None else None,contact_category_counts=dict(counts),boundary_checks=boundaries,
            wall_s=time.monotonic()-started)
        if 'maximum_unsupported_bilateral_by_arm_s' in locals():
            result['maximum_unsupported_bilateral_by_arm_s']=maximum_unsupported_bilateral_by_arm_s
        if observation is not None:result['last_observation']=observation
        setup_mark('runtime_result_and_trace_finalization')
        if scene is not None:
            try:scene.destroy()
            except Exception as exc:result['cleanup_errors'].append('scene: '+repr(exc))
        setup_mark('scene_destroy')
        if setup_profile is not None:result['setup_profile']=setup_profile.record()
        if result['cleanup_errors']:
            result['status']='EPISODE_EXECUTION_FAIL'
            result.update(full_mission_passed=False,current_student_single_batch_mission_passed=False)
            if isinstance(result.get('mission_completion_receipt'),dict):
                result['mission_completion_receipt'].update(passed=False,revoked_reason='runtime_cleanup_error')
            result['material_delivery_accepted']=False
            result['physical_handoff_verified']=False
            for key in ('delivered_material_ids','delivered_count','delivered_mass_kg','physical_handoff_count'):result.pop(key,None)
            if 'continuation' in result:result['continuation']['single_delivery_verified']=False
    return result
