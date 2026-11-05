"""Pure observed-state input for donor retreat or recipient approach.

The trusted runtime supplies current observations, fixed-entity readbacks and a
committed released/rest boundary. This module validates their identity. It does
not observe a scene, authenticate caller assertions, choose an endpoint, create
a path, attach a body, or resume a destroyed scene. Query construction remains
the caller's isolated-process responsibility.
"""
from __future__ import annotations

import copy
import math
import struct

import numpy as np

from physics.carried_body import VesselModel, WorldPose
from physics.episode_joint_scene import validate_episode_snapshot, native_snapshot_command, episode_base_position
from physics.episode_transport_inputs import _three_arm_layout
from physics.compound_obstacles import embedded_support_station, validate_embedded_support_components
from physics.episode_paused_state import project_paused_state, _fields, _json, _vector, _pose, _index, _number
from physics.fixed_world_task import FixedSupportStation
from transport_challenge.contracts import ContractError, digest


PHASE_ARMS = {'donor_retreat': 'arm_0', 'recipient_approach': 'arm_1'}
BOUNDARY_FIELDS = {'schema_version', 'step', 'simulation_s', 'observation_sha256',
    'fixed_state_sha256', 'scenario_sha256', 'source_bindings_sha256', 'released_gate_sha256',
    'previous_commands', 'actual_state_verified', 'measured_rest_verified',
    'source_bindings_verified', 'fixed_state_verified', 'released_state_verified'}


def _cast(value, dtype):
    if dtype == 'float64':return float(value)
    if dtype != 'float32':raise ContractError('Use the actual float32/float64 native dtype')
    try:return struct.unpack('f', struct.pack('f', value))[0]
    except (OverflowError,struct.error) as exc:raise ContractError('Unrepresentable native value') from exc


def _same_representation(actual, declared):
    return actual == declared or actual == _cast(declared, 'float32')


def _station(record):
    _fields(record, {'id','frame','units','axis_aligned','size_m','center_world_m'}, 'Support recipe')
    if record['frame']!='world' or record['units']!='m' or record['axis_aligned'] is not True:
        raise ContractError('Axis-aligned world support required')
    return FixedSupportStation(record['id'], record['size_m'], record['center_world_m'])


def _check_fixed(world, scenario, observation, bindings, fixed, *, repeated_three_arm_mode=False):
    if type(repeated_three_arm_mode) is not bool:raise ContractError('Mode must be explicit boolean')
    if repeated_three_arm_mode:_three_arm_layout(world,scenario)
    _fields(fixed, {'schema_version','native_scene_step','simulation_s','arms','body','supports'}, 'Fixed readback')
    if fixed['schema_version']!='episode-unloaded-fixed-state-v1':raise ContractError('Unknown fixed-state version')
    if (type(fixed['native_scene_step']) is not int or fixed['native_scene_step']!=observation['native_scene_step']
            or type(fixed['simulation_s']) not in (int,float) or fixed['simulation_s']!=observation['simulation_s']):
        raise ContractError('Fixed geometry and moving state must share the same native clock')
    model=VesselModel.from_container(scenario['container'])
    if (model.geometry_sha256!=world['vessel_model']['geometry_sha256']
            or bindings['body']['model_sha256']!=model.geometry_sha256
            or scenario['container']!=world['vessel_model']['container']):
        raise ContractError('Body model/geometry binding differs')
    body=fixed['body']
    _fields(body, {'id','entity_idx','mass_kg','model_geometry_sha256'}, 'Native body identity')
    _index(body['entity_idx'],'Body entity');_number(body['mass_kg'],'Body mass',positive=True)
    if (body['id']!=world['object_id'] or body['id']!=scenario['body_id']
            or body['id']!=bindings['body']['id'] or body['entity_idx']!=bindings['body']['entity_idx']
            or body['model_geometry_sha256']!=model.geometry_sha256
            or not _same_representation(body['mass_kg'], model.empty_mass_kg)):
        raise ContractError('Actual body ID, mass or native model differs')
    recipes=scenario['robots'];names=[r['id'] for r in recipes]
    if (len(names)!=len(set(names)) or (not repeated_three_arm_mode and set(names)!=set(world['arm_ids']))
            or set(names)!=set(bindings['arms']) or set(names)!=set(fixed['arms'])):
        raise ContractError('Complete distinct arm names required')
    robots=[];native_limits={}
    for recipe in recipes:
        name=recipe['id'];actual=fixed['arms'][name]
        _fields(actual, {'entity_idx','native_dtype','base_world','native_lower','native_upper'}, 'Native arm fixed state')
        _index(actual['entity_idx'],'Arm entity')
        if (actual['entity_idx']!=bindings['arms'][name]['entity_idx']
                or bindings['arms'][name]['model_sha256']!=world['robot_model_sha256']):
            raise ContractError('Arm entity/model identity differs')
        low=_vector(actual['native_lower'],9,'Native lower bounds');high=_vector(actual['native_upper'],9,'Native upper bounds')
        if any(lo>hi for lo,hi in zip(low,high)):raise ContractError('Native bounds are reversed')
        dtype=actual['native_dtype']
        if any(_cast(v,dtype)!=v for v in low+high):raise ContractError('Native bounds dtype differs')
        p=_pose(actual['base_world']);base=recipe['base_placement']
        target=(episode_base_position(recipe,scenario['case']) if repeated_three_arm_mode
                else [base['x_m'],base['y_m'],world['case_record']['base_z_m']])
        q=[math.cos(base['yaw_rad']/2),0.,0.,math.sin(base['yaw_rad']/2)]
        expected_p=[_cast(v,dtype) for v in target];expected_q=[_cast(v,dtype) for v in q]
        if (p['position_m']!=expected_p or p['quaternion_wxyz'] not in (expected_q,[-v for v in expected_q])):
            raise ContractError('Measured fixed base no longer matches the named installation recipe')
        # Keep the exact recipe that reconstructs the fixed native root. Raw
        # measured roots are also retained. Do not round-trip native yaw.
        values=observation['arms'][name]['q_all_dofs']
        robots.append({'id':name,'base_placement':copy.deepcopy(base),
                       'q_rad':values[:7], 'finger_q_m':values[7:]})
        if repeated_three_arm_mode and 'base_z_m' in recipe:
            robots[-1]['base_z_m']=recipe['base_z_m']
        native_limits[name]={'native_dtype':dtype,'native_lower':low,'native_upper':high}
    declared=scenario['support_stations']
    if not repeated_three_arm_mode and (declared[0]!=world['source_support'] or declared[-1]!=world['goal_support']
            or len(declared)!=3 or len({s['id'] for s in declared})!=3):
        raise ContractError('Fixed source, relay and goal support recipes required')
    actual_supports=fixed['supports']
    if type(actual_supports) is not list or [s['id'] for s in actual_supports]!=[s['id'] for s in declared]:
        raise ContractError('Missing, reordered or stale support identity')
    stations=[];entity_ids=[b['entity_idx'] for b in bindings['arms'].values()]
    entity_ids += [bindings['body']['entity_idx']]+[m['entity_idx'] for m in bindings['materials']]
    geom_ids=[]
    embedded=validate_embedded_support_components(scenario['case'],declared) if repeated_three_arm_mode else {}
    assemblies={c:g['id'] for g in scenario['case'].get('compound_obstacle_groups',[]) for c in g['component_ids']}
    native_owners={v:('dynamic',i) for i,v in enumerate(entity_ids)}
    assembly_entities={}
    for recipe,actual in zip(declared,actual_supports):
        _station(recipe)
        extra=({'validated_native_component','assembly_id','component_id','embedded_in_rigid_assembly','embedded_station'}
               if recipe['id'] in embedded else set())
        _fields(actual, {'id','entity_idx','geom_id','size_m','world_pose'}|extra, 'Measured support')
        entity_ids.append(_index(actual['entity_idx'],'Support entity'))
        geom_ids.append(_index(actual['geom_id'],'Support geometry'))
        size=_vector(actual['size_m'],3,'Support size');p=_pose(actual['world_pose'])
        if recipe['id'] in embedded:
            component=embedded[recipe['id']];assembly=assemblies[component]
            checked=embedded_support_station(actual['validated_native_component'],recipe)
            n=checked['observed_native_geometry']
            if (actual['embedded_in_rigid_assembly'] is not True or actual['component_id']!=component
                    or actual['assembly_id']!=assembly or checked!=actual['embedded_station']
                    or checked['component_id']!=component or checked['assembly_id']!=assembly
                    or checked['geom_id']!=actual['geom_id'] or n['size_m']!=size or n['center_world_m']!=p['position_m']
                    or p['quaternion_wxyz'] not in ([1.,0.,0.,0.],[-1.,0.,0.,0.])):
                raise ContractError('Embedded native support metadata differs from its exact named component')
            owner=('assembly',assembly)
            if (actual['entity_idx'] in native_owners and native_owners[actual['entity_idx']]!=owner
                    or assembly in assembly_entities and assembly_entities[assembly]!=actual['entity_idx']):
                raise ContractError('Aliased native support assembly identity')
            native_owners[actual['entity_idx']]=owner;assembly_entities[assembly]=actual['entity_idx']
            # Query asset reconstruction uses authored geometry. The measured
            # rounded metadata remains immutable in fixed_state, never replaces it.
            stations.append(copy.deepcopy(recipe))
            continue
        if (p['quaternion_wxyz'] not in ([1.,0.,0.,0.],[-1.,0.,0.,0.])
                or not all(_same_representation(v,e) for v,e in zip(size,recipe['size_m']))
                or not all(_same_representation(v,e) for v,e in zip(p['position_m'],recipe['center_world_m']))):
            raise ContractError('Measured support moved or geometry changed')
        stations.append(FixedSupportStation(actual['id'],size,p['position_m']).to_record())
        if repeated_three_arm_mode:
            if actual['entity_idx'] in native_owners:raise ContractError('Aliased native standalone support')
            native_owners[actual['entity_idx']]=('support',actual['id'])
    if (not repeated_three_arm_mode and len(entity_ids)!=len(set(entity_ids))) or len(geom_ids)!=len(set(geom_ids)):
        raise ContractError('Aliased native entities or support geometries')
    return model,robots,stations,native_limits


def build_unloaded_phase_input(world, scenario, observation, *, phase, active_arm_id,
                               bindings, boundary, released_gate, fixed_state,
                               repeated_three_arm_mode=False,donor_arm_id=None,recipient_arm_id=None,support_id=None):
    """Return a detached versioned query input, with no endpoint or path.

    All positions use metres, arm angles radians, velocities per second and
    masses kg. ``bindings`` is episode-pause-bindings-v1. ``boundary`` is a
    trusted runtime receipt described in PHASE_INPUT_API.md. This function
    checks its hashes and assertions, not the physical truth behind them.
    """
    for item in (world,scenario,observation,bindings,boundary,released_gate,fixed_state):_json(item)
    if type(repeated_three_arm_mode) is not bool:raise ContractError('Mode must be explicit boolean')
    if repeated_three_arm_mode:
        names,stations=_three_arm_layout(world,scenario)
        if (donor_arm_id not in names or recipient_arm_id not in names or donor_arm_id==recipient_arm_id
                or phase not in PHASE_ARMS or support_id not in stations
                or active_arm_id!=({'donor_retreat':donor_arm_id,'recipient_approach':recipient_arm_id}[phase])):
            raise ContractError('Explicit distinct donor/recipient, phase holder and source support required')
    elif donor_arm_id is not None or recipient_arm_id is not None or support_id is not None:
        raise ContractError('Explicit repeated handoff roles require three-arm mode')
    if not repeated_three_arm_mode and (phase not in PHASE_ARMS or active_arm_id!=PHASE_ARMS[phase]):
        raise ContractError('Explicit donor_retreat/arm_0 or recipient_approach/arm_1 required')
    if (scenario['case']!=world['case_record'] or scenario['provenance']['world_sha256']!=digest(world)
            or bindings['world_sha256']!=digest(world)):
        raise ContractError('World/scenario/binding identity differs')
    projection=project_paused_state(observation,bindings=bindings)
    _fields(boundary,BOUNDARY_FIELDS,'Trusted unloaded boundary')
    if boundary['schema_version']!='episode-unloaded-boundary-v1':raise ContractError('Unknown boundary version')
    if (type(boundary['step']) is not int or boundary['step']!=observation['native_scene_step']
            or type(boundary['simulation_s']) not in (int,float) or boundary['simulation_s']!=observation['simulation_s']):
        raise ContractError('Boundary observation is stale or on the wrong clock')
    for key,value in (('observation_sha256',observation),('fixed_state_sha256',fixed_state),
                      ('scenario_sha256',scenario),('source_bindings_sha256',bindings),('released_gate_sha256',released_gate)):
        if boundary[key]!=digest(value):raise ContractError('Boundary identity differs: '+key)
    for key in ('actual_state_verified','measured_rest_verified','source_bindings_verified','fixed_state_verified','released_state_verified'):
        if boundary[key] is not True:raise ContractError('Trusted boundary did not verify '+key)
    gate=released_gate;names=set(observation['arms'])
    if (gate.get('schema_version')!='sampled-released-relay-gate-v1' or gate.get('passed') is not True
            or gate.get('failures')!=[] or gate.get('ending_s')!=observation['simulation_s']
            or gate.get('dt_s')!=bindings['dt_s'] or gate.get('window_s')!=.5
            or type(gate.get('expected_rows')) is not int or type(gate.get('observed_rows')) is not int
            or gate['expected_rows']!=gate['observed_rows'] or gate['expected_rows']*bindings['dt_s']!=.5
            or gate.get('support_id')!=(support_id if repeated_three_arm_mode else scenario['support_stations'][1]['id'])
            or gate.get('active_arm_id')!=(donor_arm_id if repeated_three_arm_mode else 'arm_0')
            or gate.get('all_arms_rest',{}).get('passed') is not True
            or set(gate['all_arms_rest']['arms'])!=names
            or any(a.get('passed') is not True for a in gate['all_arms_rest']['arms'].values())
            or gate.get('body_rest',{}).get('passed') is not True):
        raise ContractError('A current complete released-relay/rest gate is required')
    model,robots,stations,limits=_check_fixed(world,scenario,observation,bindings,fixed_state,
        repeated_three_arm_mode=repeated_three_arm_mode)
    original=world['initial_contents'];material=projection['state']['materials']
    if (len(original)!=len(material) or gate.get('persistent_material_ids')!=[x['id'] for x in original]
            or any(a['id']!=b['id'] or a['mass_kg']!=b['mass_kg'] or a['radius_m']!=b['radius_m'] for a,b in zip(original,material))
            or any(m.get('inside_carrier') is not True for m in observation['materials'])):
        raise ContractError('Released material IDs, masses, radii or retention differ')
    contents=[{'id':m['id'],'radius_m':m['radius_m'],'center_world_m':m['position_world_m']} for m in material]
    commands=boundary['previous_commands']
    if type(commands) is not dict or set(commands)!=names:raise ContractError('Previous commands for every arm are required')
    for name,c in commands.items():
        _fields(c,{'q_rad','finger_q_m'},'Previous command')
        values=_vector(c['q_rad'],7,'Previous seven-q command')+_vector(c['finger_q_m'],2,'Previous finger command')
        if any(not lo<=q<=hi for q,lo,hi in zip(values,limits[name]['native_lower'],limits[name]['native_upper'])):
            raise ContractError('Previous command exceeds strict native bounds')
    observed={'schema_version':'episode-observed-initialization-v1','observation':copy.deepcopy(observation),
        'observation_sha256':digest(observation),'source_bindings_sha256':digest(bindings),'robots_sha256':digest(robots)}
    for robot in robots:
        native=limits[robot['id']];dtype=np.dtype(native['native_dtype'])
        native_snapshot_command(robot,np.asarray(native['native_lower'],dtype=dtype),
            np.asarray(native['native_upper'],dtype=dtype),observed_snapshot=observed)
    body=WorldPose(observation['body_world']['position_m'],observation['body_world']['quaternion_wxyz'])
    snapshot=validate_episode_snapshot(world['case_record'],active_arm_id,robots,model,body,world['object_id'],
        [_station(s) for s in stations],contents,observed_snapshot=observed)
    record={'schema_version':'episode-unloaded-phase-input-v1','phase':phase,'active_arm_id':active_arm_id,
        'world_sha256':digest(world),'scenario_sha256':digest(scenario),'container':copy.deepcopy(scenario['container']),
        'observation':copy.deepcopy(observation),'observation_sha256':digest(observation),
        'fixed_state':copy.deepcopy(fixed_state),'fixed_state_sha256':digest(fixed_state),
        'source_bindings':copy.deepcopy(bindings),'boundary':copy.deepcopy(boundary),'released_gate':copy.deepcopy(gate),
        'native_projection':projection,'query_snapshot':snapshot,'observed_snapshot':observed,
        'nominal_robot_recipes':copy.deepcopy(scenario['robots']),
        'nominal_support_recipes':copy.deepcopy(scenario['support_stations']),
        'previous_commands':copy.deepcopy(commands),'native_limits':limits,
        'persistent_materials':copy.deepcopy(material),'original_mass_kg':math.fsum(m['mass_kg'] for m in original),
        'query_scope':'Static geometry clone. Original velocities/raw poses retained as evidence, not dynamic initial conditions.',
        'command_continuity_scope':'Previous commands remain separate. Caller must validate the measured-to-command bridge; no bridge or endpoint is inserted.',
        'endpoint_supplied':False,'route_generated':False,'physical_handoff_verified':False,'new_grasp_verified':False}
    if repeated_three_arm_mode:
        record.update(repeated_three_arm_mode=True,donor_arm_id=donor_arm_id,
            recipient_arm_id=recipient_arm_id,support_id=support_id,
            world=copy.deepcopy(world),scenario=copy.deepcopy(scenario))
    return copy.deepcopy({**record,'sha256':digest(record)})


def query_arguments(record):
    """Validate a frozen phase record and return detached EpisodeJointScene kwargs.

    Caller adds only asset_directory/render and constructs its isolated scene.
    This function neither constructs a scene nor queries FK, IK or collisions.
    """
    _json(record)
    if (record.get('schema_version')!='episode-unloaded-phase-input-v1'
            or record.get('sha256')!=digest({k:v for k,v in record.items() if k!='sha256'})):
        raise ContractError('Frozen phase record changed')
    mode=record.get('repeated_three_arm_mode',False)
    if type(mode) is not bool:raise ContractError('Mode must be explicit boolean')
    if not mode and any(k in record for k in ('world','scenario','donor_arm_id','recipient_arm_id','support_id')):
        raise ContractError('Repeated phase fields cannot be interpreted as a legacy record')
    if mode:
        rebuilt=build_unloaded_phase_input(record['world'],record['scenario'],record['observation'],
            phase=record['phase'],active_arm_id=record['active_arm_id'],bindings=record['source_bindings'],
            boundary=record['boundary'],released_gate=record['released_gate'],fixed_state=record['fixed_state'],
            repeated_three_arm_mode=True,donor_arm_id=record['donor_arm_id'],
            recipient_arm_id=record['recipient_arm_id'],support_id=record['support_id'])
        if rebuilt!=record:raise ContractError('Repeated phase no longer reconstructs exactly')
    snap=record['query_snapshot'];model=VesselModel.from_container(record['container'])
    body=WorldPose(*[record['observation']['body_world'][k] for k in ('position_m','quaternion_wxyz')])
    stations=[_station(s) for s in snap['support_stations']]
    validated=validate_episode_snapshot(snap['case'],snap['active_arm_id'],snap['robots'],model,body,
        snap['body_id'],stations,snap['contents'],observed_snapshot=record['observed_snapshot'])
    if validated!=snap:raise ContractError('Phase snapshot no longer reconstructs exactly')
    return {'case':copy.deepcopy(snap['case']),'active_arm_id':snap['active_arm_id'],
        'robots':copy.deepcopy(snap['robots']),'model':model,'body_pose':body,'body_id':snap['body_id'],
        'support_stations':stations,'contents':copy.deepcopy(snap['contents']),
        'observed_snapshot':copy.deepcopy(record['observed_snapshot'])}
