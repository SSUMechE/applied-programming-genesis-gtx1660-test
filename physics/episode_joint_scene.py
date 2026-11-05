"""Isolated unloaded-arm queries against a complete persistent-episode snapshot.

Only the selected arm's seven joints change. Other arms, the authored tray,
supports and sphere contents retain their supplied world snapshot. No physics
step, live-scene mutation, held-body attachment or intermediate route occurs.
This is a discrete geometric oracle, not a physical handoff verdict.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import numpy as np

from physics.carried_body import VesselModel, WorldPose, write_carried_vessel_urdf
from physics.compound_obstacles import (validate_compound_obstacle_groups,
    write_compound_obstacle_asset, validate_built_compound, verify_compound_sources,
    validate_embedded_support_components, embedded_support_station)
from physics.fixed_world_task import FixedSupportStation
from physics.handle_contacts import (PANDA_XML, MAX_PENETRATION_M, NUMERIC_TOLERANCE_M, MINIMUM_NORMAL_COSINE,
    classify_supported_contacts, contents_contact_witness_valid, match_native_pads, read_panda_pad_primitives,
    query_witness_snapshot, verify_contact_sources, verify_query_witness_runtime)
from physics.joint_scene import JointScene, array, validate_map
from physics.loaded_joint_scene import (SUPPORT_CONTACT_MASK, validate_asset_components,
    validate_built_components, validate_built_support)
from physics.self_collision_audit import adjacent_by_installed_rule
from transport_challenge.contracts import ContractError, digest, identifier, number, point
from transport_challenge.joint_context import _reject_solution_data


def _q(value, count, label):
    if not isinstance(value, (list, tuple, np.ndarray)) or (isinstance(value, np.ndarray) and value.ndim != 1):
        raise ContractError(label+' must be a one-dimensional sequence')
    if len(value) != count or any(isinstance(v, (bool, np.bool_)) or
            not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(v) for v in value):
        raise ContractError(label+' requires finite numeric coordinates')
    return tuple(float(v) for v in value)


def fixed_fixture_quaternion_comparison(expected, actual):
    """Compare readback orientation without mistaking near-unit scale for rotation.

    Use the existing ToolPose norm contract and the caller's unchanged 1e-7
    direction tolerance. Normalized tuples are local calculations only. No
    native quaternion or authored input is modified.
    """
    expected = _q(expected, 4, 'expected fixed fixture quaternion')
    actual = _q(actual, 4, 'actual fixed fixture quaternion')
    en, an = math.hypot(*expected), math.hypot(*actual)
    if abs(en-1.) > 1e-5 or abs(an-1.) > 1e-5:
        raise ContractError('Fixed fixture quaternions must satisfy the existing near-unit norm contract')
    eu, au = tuple(v/en for v in expected), tuple(v/an for v in actual)
    return {'expected_norm': en, 'actual_norm': an,
        'raw_sign_distance': min(math.dist(actual, expected), math.dist(actual, tuple(-v for v in expected))),
        'normalized_sign_distance': min(math.dist(au, eu), math.dist(au, tuple(-v for v in eu)))}


OBSERVED_INITIAL_RESIDUAL = 1e-5  # Existing live native-position guard, not a planner allowance.


def _observed_initialization(value, robots=None):
    """Validate a trusted teacher readback identity, never authenticate a scene.

    Hashes bind detached records. The calling runtime must verify their native
    provenance. Only initial reconstruction uses this option. Candidate states,
    edges and IK seeds continue to use the unchanged strict native bounds.
    """
    if value is None:return None
    fields={'schema_version','observation','observation_sha256','source_bindings_sha256','robots_sha256'}
    if (not isinstance(value,dict) or set(value)!=fields or value['schema_version'] not in
            ('episode-observed-initialization-v1','episode-single-active-mixed-query-initialization-v1')):
        raise ContractError('Explicit observed initialization record required')
    for key in ('observation_sha256','source_bindings_sha256','robots_sha256'):
        sha=value[key]
        if type(sha) is not str or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise ContractError('Invalid observed initialization identity')
    try:json.dumps(value,allow_nan=False)
    except (ValueError,TypeError,OverflowError) as exc:raise ContractError('Malformed observed initialization') from exc
    obs=value['observation']
    if not isinstance(obs,dict) or digest(obs)!=value['observation_sha256'] or not isinstance(obs.get('arms'),dict):
        raise ContractError('Observed initialization digest/arms differ')
    if (type(obs.get('native_scene_step')) is not int or obs['native_scene_step']<0
            or type(obs.get('simulation_s')) not in (int,float) or not math.isfinite(obs['simulation_s'])
            or obs['simulation_s']<0):raise ContractError('Invalid observed initialization clock')
    for arm in obs['arms'].values():
        if not isinstance(arm,dict):raise ContractError('Missing observed arm state')
        _q(arm.get('q_all_dofs'),9,'observed native q')
    mixed=value['schema_version']=='episode-single-active-mixed-query-initialization-v1'
    from physics.single_active_boundary import contract_from_observation,robot_coordinates
    single=contract_from_observation(obs)
    if mixed != (single is not None):
        raise ContractError('Mixed query geometry requires its explicit single-active source contract')
    names={single['active_arm_id'],*single['frozen_arm_ids']} if mixed else set(obs['arms'])
    if robots is not None:
        if (digest(robots)!=value['robots_sha256'] or {r['id'] for r in robots}!=names
                or len(robots)!=len(names)):raise ContractError('Observed robot snapshot binding differs')
        for robot in robots:
            if list(robot['q_rad'])+list(robot['finger_q_m'])!=robot_coordinates(obs,robot['id']):
                raise ContractError('Snapshot substitutes a command for actual measured q')
    return obs


def validate_episode_snapshot(case, active_arm_id, robots, model, body_pose,
                              body_id, support_stations=(), contents=(), *, observed_snapshot=None):
    """Copy named inputs before importing Genesis. No native feasibility claim.

    Optional robot ``base_z_m`` is a caller-derived fixed mounting-surface
    height, not a student-controlled fourth base coordinate. Its absence keeps
    the legacy case-wide height and the original four-field robot record.
    """
    if type(model) is not VesselModel or type(body_pose) is not WorldPose:
        raise ContractError('Use immutable VesselModel and world-frame WorldPose')
    if not isinstance(robots, (list, tuple)) or len(robots) not in (2, 3):
        raise ContractError('Supply exactly two or three complete robot snapshots')
    copied = copy.deepcopy(case)
    _reject_solution_data(copied)
    observed=_observed_initialization(observed_snapshot,robots)
    records = []
    for record in robots:
        required={'id','base_placement','q_rad','finger_q_m'}
        if not isinstance(record, dict) or set(record) not in (required,required|{'base_z_m'}):
            raise ContractError('Robot snapshot needs id, named base, seven q and two fingers, with optional fixed base_z_m')
        name = identifier(record['id'])
        base = record['base_placement']
        if not isinstance(base, dict) or set(base) != {'x_m','y_m','yaw_rad'}:
            raise ContractError('Use named X/Y metres and yaw radians, never a legacy XYZ base')
        position = _q([base['x_m'],base['y_m'],base['yaw_rad']],3,'base')
        validate_map(copied, position)
        q, fingers = _q(record['q_rad'],7,'arm q'), _q(record['finger_q_m'],2,'fingers')
        residual=OBSERVED_INITIAL_RESIDUAL if observed is not None else 0.
        if any(not -residual <= v <= .04+residual for v in fingers):
            raise ContractError('Snapshot fingers must lie in the supported Panda range')
        records.append({'id':name,'base_placement':dict(zip(('x_m','y_m','yaw_rad'),position)),
                        'q_rad':list(q),'finger_q_m':list(fingers)})
        if 'base_z_m' in record:
            records[-1]['base_z_m']=number(record['base_z_m'],'fixed robot base height',minimum=0.)
    names = [v['id'] for v in records]
    if len(set(names)) != len(names) or active_arm_id not in names:
        raise ContractError('Active arm must name exactly one distinct robot')
    if not isinstance(support_stations,(list,tuple)) or any(type(v) is not FixedSupportStation for v in support_stations):
        raise ContractError('Use immutable FixedSupportStation records')
    stations = [v.to_record() for v in support_stations]
    if not isinstance(contents,(list,tuple)):
        raise ContractError('Contents must be a sequence of fixed sphere snapshots')
    spheres=[]
    for value in contents:
        if not isinstance(value,dict) or set(value) != {'id','radius_m','center_world_m'}:
            raise ContractError('Sphere snapshot needs exactly id, radius_m and center_world_m')
        radius=number(value['radius_m'],'sphere radius',minimum=0.)
        if radius <= 0.:raise ContractError('Sphere radius must be positive')
        spheres.append({'id':identifier(value['id']),'radius_m':radius,
                        'center_world_m':list(_q(value['center_world_m'],3,'sphere world center'))})
    body_id=identifier(body_id)
    compound_groups=validate_compound_obstacle_groups(copied)
    ids=['floor',body_id,*names,*(s['id'] for s in stations),
         *(s['id'] for s in spheres),*(o['id'] for o in copied['obstacles']),
         *(g['id'] for g in compound_groups)]
    if len(set(ids)) != len(ids):raise ContractError('Every episode entity needs a distinct ID')
    validate_embedded_support_components(copied,stations)
    validate_query_visual(copied,stations)
    return {'schema_version':'episode-joint-snapshot-v1','case':copied,
            'active_arm_id':active_arm_id,'robots':records,'body_id':body_id,
            'model_geometry_sha256':model.geometry_sha256,'body_world':body_pose.to_record(),
            'support_stations':stations,'contents':spheres,
            'motion_scope':'unloaded_approach_or_retreat_only'}


def episode_base_position(record, case):
    """Resolve immutable world XYZ without reinterpreting named XY/yaw."""
    base=record['base_placement']
    z=number(record.get('base_z_m',case['base_z_m']),'fixed robot base height',minimum=0.)
    return _q([base['x_m'],base['y_m'],z],3,'robot base world position')


def _visual_rgb(value):
    if type(value) is not str or len(value)!=7 or not value.startswith('#'):
        raise ContractError('Query visual colors use #RRGGBB')
    try:return tuple(int(value[i:i+2],16)/255 for i in (1,3,5))
    except ValueError as exc:raise ContractError('Query visual colors use #RRGGBB') from exc


def validate_query_visual(case, support_stations):
    """Optional appearance only. No geometry, masks or physics parameters."""
    if 'query_visual' not in case:return None
    value=case['query_visual'];required={'component_colors','camera'}
    if (not isinstance(value,dict) or not required<=set(value)
            or set(value)-required-{'support_colors','floor_color','vessel_color','hidden_component_ids'}):
        raise ContractError('Query visual requires component_colors and camera with optional support/floor/vessel colors')
    colors=value['component_colors']
    if not isinstance(colors,dict) or set(colors)!={o['id'] for o in case['obstacles']}:
        raise ContractError('Query component colors must cover exactly every obstacle component')
    for color in colors.values():_visual_rgb(color)
    if 'hidden_component_ids' in value:
        hidden=value['hidden_component_ids']
        if (not isinstance(hidden,(list,tuple)) or any(type(v) is not str for v in hidden)
                or len(set(hidden))!=len(hidden) or not set(hidden)<=set(colors)):
            raise ContractError('Hidden visual components must be distinct declared obstacle IDs')
    if 'support_colors' in value:
        if not isinstance(value['support_colors'],dict) or set(value['support_colors'])!={s['id'] for s in support_stations}:
            raise ContractError('Query support colors must cover exactly every named support')
        for color in value['support_colors'].values():_visual_rgb(color)
    for key in ('floor_color','vessel_color'):
        if key in value:_visual_rgb(value[key])
    camera=value['camera']
    if not isinstance(camera,dict) or set(camera)!={'position_m','lookat_m','fov_deg'}:
        raise ContractError('Query camera requires position_m, lookat_m and fov_deg')
    position=_q(camera['position_m'],3,'camera position');lookat=_q(camera['lookat_m'],3,'camera lookat')
    fov=number(camera['fov_deg'],'camera FOV degrees',minimum=0.)
    if not 0.<fov<180. or position==lookat:raise ContractError('Query camera needs a direction and FOV in (0,180)')
    result=copy.deepcopy(value);result.setdefault('floor_color','#FFFFFF')
    return result


def _integer_array(value):
    if hasattr(value,'_unwrap'):value=value._unwrap()
    if hasattr(value,'to_numpy'):value=value.to_numpy()
    elif hasattr(value,'detach'):value=value.detach().cpu().numpy()
    result=np.asarray(value)
    if result.dtype.kind not in 'iu':raise ContractError('Native pair metadata must be integer')
    return result.copy()


def validate_native_base_position(authored_position_m, native_position, *, arm_id):
    """Require the exact native representation of an authored fixed base.

    Read ``entity.get_pos(relative=False)`` directly, retaining its dtype.
    A binary32 coordinate can be farther than 1e-7 m from a correct binary64
    authored value at this map's scale. Comparing against its exact cast fixes
    that representation mismatch without adding physical slack or moving it.
    Quaternion validation remains the caller's unchanged separate predicate.
    """
    authored=np.asarray(_q(authored_position_m,3,'authored base world position'))
    if hasattr(native_position,'detach'):native_position=native_position.detach().cpu().numpy()
    observed=np.asarray(native_position)
    detail={'arm_id':identifier(arm_id),'authored_position_m':authored.tolist(),
            'native_dtype':str(observed.dtype),'observed_shape':list(observed.shape),
            'observed_position_m':observed.tolist()}
    if (observed.shape!=(3,) or observed.dtype not in (np.dtype('float32'),np.dtype('float64'))
            or not np.isfinite(observed).all()):
        raise ContractError('Malformed native base position: '+repr(detail))
    with np.errstate(over='ignore'):
        represented=authored.astype(observed.dtype)
    detail.update(represented_position_m=represented.tolist(),
        authored_to_native_rounding_m=(represented.astype(float)-authored).tolist(),
        observed_minus_represented_m=(observed.astype(float)-represented.astype(float)).tolist(),
        exact_native_representation=bool(np.isfinite(represented).all() and np.array_equal(observed,represented)),
        physical_position_tolerance_m=0.)
    if not detail['exact_native_representation']:
        raise ContractError('Native robot base differs from represented named installation: '+repr(detail))
    return detail


def native_snapshot_command(record, native_lower, native_upper, *, observed_snapshot=None):
    """Validate the actual native representation, without clipping or epsilon.

    Python's authored .04 and its float32 native representation are different
    binary64 numbers. A command is converted once to the actual native-bound
    dtype, then checked against those same native bounds. The seven authored
    arm values also pass strict native bounds before conversion. Fingers
    must independently remain in [0,.04]. Already measured native coordinates
    are unchanged by this conversion. Returned evidence retains both forms.
    """
    def native(values):
        if hasattr(values,'detach'):values=values.detach().cpu().numpy()
        result=np.asarray(values)
        if result.dtype not in (np.dtype('float32'),np.dtype('float64')) or result.shape!=(9,) or not np.isfinite(result).all():
            raise ContractError('Malformed actual native bounds: '+repr({'shape':result.shape,'dtype':str(result.dtype),'values':result.tolist()}))
        return result.copy()
    low,high=native(native_lower),native(native_upper)
    if low.dtype!=high.dtype or np.any(low>high):raise ContractError('Native bound dtypes/order disagree')
    authored=np.asarray(_q(record['q_rad'],7,'snapshot arm q')+_q(record['finger_q_m'],2,'snapshot fingers'),dtype=np.float64)
    observed=_observed_initialization(observed_snapshot)
    if observed is not None:
        from physics.single_active_boundary import robot_coordinates
        if authored.tolist()!=robot_coordinates(observed,record['id']):
            raise ContractError('Native snapshot is not the bound measured or qualified frozen query state')
    residual=OBSERVED_INITIAL_RESIDUAL if observed is not None else 0.
    if any(not -residual<=v<=.04+residual for v in authored[7:]):
        raise ContractError('Authored finger coordinates leave [0,.04] metres: '+repr(authored[7:].tolist()))
    with np.errstate(over='ignore',invalid='ignore'):command=authored.astype(low.dtype)
    if not np.isfinite(command).all():raise ContractError('Snapshot command is not finite in native '+str(low.dtype))
    evidence={'robot_id':record['id'],'native_dtype':str(low.dtype),'authored_command':authored.tolist(),
        'native_command':command.tolist(),'native_lower':low.tolist(),'native_upper':high.tolist(),
        'changed_representation_indices':np.flatnonzero(authored!=command.astype(np.float64)).tolist(),
        'below_authored_arm_bound_indices':np.flatnonzero(authored[:7]<low[:7]).tolist(),
        'above_authored_arm_bound_indices':np.flatnonzero(authored[:7]>high[:7]).tolist(),
        'below_native_bound_indices':np.flatnonzero(command<low).tolist(),
        'above_native_bound_indices':np.flatnonzero(command>high).tolist(),
        'conversion':'Single dtype conversion, no clipping, wrapping, epsilon or physical-bound extension'}
    if observed is not None:
        if not np.array_equal(authored,command.astype(np.float64)):
            raise ContractError('Measured q cannot be represented exactly by this native dtype')
        if np.any(authored<low.astype(np.float64)-residual) or np.any(authored>high.astype(np.float64)+residual):
            raise ContractError('Measured initialization exceeds the existing live residual bound')
        evidence['observed_initialization']={'observation_sha256':observed_snapshot['observation_sha256'],
            'source_bindings_sha256':observed_snapshot['source_bindings_sha256'],
            'coordinate_source':('measured_active_native_state' if record['id'] in observed['arms']
                                 else 'qualified_fixed_INITIAL_geometry_configuration'),
            'initial_only_residual_bound':residual,'raw_native_values_preserved':True,
            'planner_bounds_relaxed':False,'scope':'Trusted measured query initialization only. No live pose correction.'}
    elif any(evidence[k] for k in ('below_native_bound_indices','above_native_bound_indices',
                                  'below_authored_arm_bound_indices','above_authored_arm_bound_indices')):
        raise ContractError('Snapshot violates actual native joint/finger bounds: '+json.dumps(evidence,allow_nan=False))
    return command,evidence


def _aabb(value):
    result=array(value)
    if result.shape!=(2,3) or not np.isfinite(result).all() or np.any(result[0]>result[1]):
        raise ContractError('Native world AABB must contain finite ordered bounds')
    return result


class MeasuredStaticAABBs:
    """Reuse measured BOX bounds during one unchanged zero-clock setup.

    Construct immediately from validate_built_compound's actual native rows.
    No scene step, pose setter or geometry mutation may intervene before the
    audit. Full native object identity and zero clock are checked again by the
    audit. This object does not derive bounds from authored box dimensions.
    """

    __slots__ = ('_scene','_solver','_geoms','_identities','_bounds','_reused',
                 '_fresh','_audit_calls','_measurement_clock','_last_audit_clock')

    def __init__(self, scene, measured_rows, *, box_type):
        self._scene=scene;self._solver=scene.rigid_solver
        self._geoms=tuple(self._solver.geoms)
        self._identities=tuple((g,g.entity,g.link,int(g.idx),int(g.link.idx),
                                str(g.entity.name),bool(g.link.is_fixed)) for g in self._geoms)
        self._bounds=();self._reused=[];self._fresh=[];self._audit_calls=0
        self._measurement_clock=(0,0.);self._last_audit_clock=None
        self._validate(self._solver)
        if not isinstance(measured_rows,(list,tuple)):
            raise ContractError('Measured static AABBs require native component rows')
        values={}
        checks=('native_box','one_declared_link','exact_component_match',
                'world_orientation_identity','world_aabb_matches','native_masks_match')
        for row in measured_rows:
            if not isinstance(row,dict):raise ContractError('Measured static AABB row must be a record')
            idx=row.get('geom_id');link_id=row.get('link_id')
            if (isinstance(idx,(bool,np.bool_)) or not isinstance(idx,(int,np.integer))
                    or not 0<=int(idx)<len(self._geoms) or int(idx) in values
                    or isinstance(link_id,(bool,np.bool_)) or not isinstance(link_id,(int,np.integer))):
                raise ContractError('Measured static AABB has duplicate or invalid native IDs')
            idx=int(idx);geom=self._geoms[idx]
            flags=row.get('native_representation_checks')
            if not isinstance(flags,dict) or any(flags.get(name) is not True for name in checks):
                raise ContractError('Measured static AABB lacks completed native BOX checks')
            if (geom.type!=box_type or not geom.link.is_fixed or geom.entity is not geom.link.entity
                    or row.get('entity')!=geom.entity.name or int(link_id)!=int(geom.link.idx)):
                raise ContractError('Measured static AABB native BOX ownership or fixed binding changed')
            raw=np.asarray(row.get('actual_world_aabb_m'))
            if (raw.shape!=(2,3) or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all()
                    or np.any(raw[0]>raw[1])):
                raise ContractError('Measured static AABB requires finite ordered actual native bounds')
            values[idx]=tuple(tuple(float(v) for v in axis) for axis in raw)
        # Tuples detach every bound from mutable caller rows. A full-length
        # tuple keeps native IDs as exact positions without a mutable map.
        self._bounds=tuple(values.get(i) for i in range(len(self._geoms)))

    def _validate(self, solver):
        scene=self._scene
        if solver is not self._solver or scene.rigid_solver is not solver:
            raise ContractError('Measured static AABBs belong to a different native solver')
        step=scene.t;clock=scene.cur_t
        if (not scene._is_built or isinstance(step,(bool,np.bool_))
                or not isinstance(step,(int,np.integer)) or int(step)!=0
                or isinstance(clock,(bool,np.bool_)) or not isinstance(clock,(int,float,np.integer,np.floating))
                or not math.isfinite(float(clock)) or float(clock)!=0.):
            raise ContractError('Measured static AABBs require the unchanged zero-clock scene')
        geoms=tuple(solver.geoms)
        if (solver.n_envs!=0 or len(geoms)!=len(self._geoms) or solver.n_geoms!=len(geoms)
                or any(a is not b for a,b in zip(geoms,self._geoms))):
            raise ContractError('Measured static AABB full native geometry sequence changed')
        for ordinal,(geom,entity,link,idx,link_id,name,fixed) in enumerate(self._identities):
            if (isinstance(geom.idx,(bool,np.bool_)) or not isinstance(geom.idx,(int,np.integer))
                    or idx!=ordinal or int(geom.idx)!=ordinal or geom.entity is not entity or geom.link is not link
                    or int(link.idx)!=link_id or str(entity.name)!=name or bool(link.is_fixed)!=fixed):
                raise ContractError('Measured static AABB native geometry identity changed')

    def _begin_audit(self, solver):
        self._validate(solver);self._audit_calls+=1
        self._last_audit_clock=(int(self._scene.t),float(self._scene.cur_t))

    def _read(self, solver, geom):
        # Called at most once per unique geometry by the audit's local cache.
        # Recheck the clock at every use without another native readback.
        if solver is not self._solver or self._scene.t!=0 or self._scene.cur_t!=0.:
            raise ContractError('Measured static AABB binding became stale during audit')
        idx=int(geom.idx)
        if not 0<=idx<len(self._geoms) or self._geoms[idx] is not geom:
            raise ContractError('Measured static AABB geometry identity changed during audit')
        value=self._bounds[idx]
        if value is None:
            result=_aabb(geom.get_AABB());self._fresh.append(idx)
            return result
        if not geom.link.is_fixed:
            raise ContractError('Measured static AABB no longer belongs to fixed geometry')
        self._reused.append(idx)
        return np.asarray(value,dtype=float).copy()

    def record(self):
        """Detached diagnostics only. The existing pair ledger is unchanged."""
        return {'schema_version':'measured-static-aabb-reuse-v1','audit_calls':self._audit_calls,
            'measurement_clock':{'native_scene_step':self._measurement_clock[0],'simulation_s':self._measurement_clock[1]},
            'last_audit_clock':(None if self._last_audit_clock is None else {
                'native_scene_step':self._last_audit_clock[0],'simulation_s':self._last_audit_clock[1]}),
            'measured_geometry_count':sum(value is not None for value in self._bounds),
            'reused_read_count':len(self._reused),'fresh_read_count':len(self._fresh),
            'reused_geom_ids':list(self._reused),'fresh_geom_ids':list(self._fresh),
            'scope':'Actual native fixed BOX bounds reused only within an unchanged source-bound zero-clock scene. Other geometry uses the original public AABB getter.'}


def audit_episode_pairs(solver, arms, owners, *, measured_static_aabbs=None):
    """Record every native pair; never exempt a cross-arm pair by entity name.

    Fixed base/floor exclusions are checked conservatively by native world
    AABBs. An overlapping excluded pair rejects even if it might be mesh-clear.
    Exact model adjacent/same-link exclusions remain explicitly disclosed.
    """
    geoms=list(solver.geoms); n=len(geoms)
    if [int(g.idx) for g in geoms]!=list(range(n)) or set(owners)!=set(range(n)):
        raise ContractError('Geometry ownership must cover the actual contiguous native IDs')
    scene=getattr(solver,'scene',None)
    single=getattr(scene,'_ap_single_active_arm_contract',None)
    if single is not None:
        from physics.fixed_peer_arms import validate_scene_contract
        single=validate_scene_contract(scene,{arm.name:arm for arm in arms})
    if measured_static_aabbs is not None:
        if not isinstance(measured_static_aabbs,MeasuredStaticAABBs):
            raise ContractError('Measured static AABBs require the bound native snapshot')
        measured_static_aabbs._begin_audit(solver)
    valid=_integer_array(solver.collider._valid_collision_pairs)
    native=_integer_array(solver.collider._collider_info.valid_collision_pairs)
    host=_integer_array(solver.collider._collision_pair_idx)
    native_index=_integer_array(solver.collider._collider_info.collision_pair_idx)
    if valid.ndim!=2 or valid.shape[1]!=2 or not np.array_equal(valid,native):
        raise ContractError('Host/native enabled-pair arrays disagree')
    rebuilt=np.full((n,n),-1,dtype=np.int64)
    for index,(a,b) in enumerate(valid):
        if not 0<=a<b<n or rebuilt[a,b]!=-1:raise ContractError('Invalid native enabled pair')
        rebuilt[a,b]=index
    if not np.array_equal(host,rebuilt) or not np.array_equal(native_index,rebuilt):
        raise ContractError('Native/host pair matrices disagree')
    if not solver._enable_self_collision or not solver._enable_neutral_collision or solver._enable_adjacent_collision:
        raise ContractError('Unexpected Panda collision-filter configuration')
    actual=[];expected=[]
    for arm in arms:
        named={j.name:int(j.idx) for j in arm.joints if j.name in ('finger_joint1','finger_joint2')}
        if set(named)!={'finger_joint1','finger_joint2'}:raise ContractError('Missing native finger identities')
        expected.append((arm.name,2,named['finger_joint1'],named['finger_joint2']))
        actual.extend((arm.name,int(e.type),int(e.eq_obj1id),int(e.eq_obj2id)) for e in arm.equalities)
    if sorted(actual)!=sorted(expected) or len(solver.equalities)!=len(expected):
        raise ContractError('Only exact Panda finger couplings are supported. No welds')
    rows=[];aabbs={}
    def bounds(geom):
        if geom.idx not in aabbs:
            aabbs[geom.idx]=(_aabb(geom.get_AABB()) if measured_static_aabbs is None
                           else measured_static_aabbs._read(solver,geom))
        return aabbs[geom.idx]
    for a,first in enumerate(geoms):
        for b in range(a+1,n):
            second=geoms[b];enabled=bool(host[a,b]>=0);reason=None;static=None
            if not enabled:
                if first.link.idx==second.link.idx:reason='same_rigid_link'
                elif (first.entity is second.entity and owners[a]['kind']=='arm' and
                      adjacent_by_installed_rule(first.link,second.link,solver.links)):
                    reason='pinned_model_adjacent_or_fixed_chain'
                elif (single is not None and first.entity is second.entity
                      and first.entity.name in single['frozen_arm_ids']):
                    name=first.entity.name
                    proof=single['fixed_peers'][name]['actual']
                    if (int(first.entity.idx)!=proof['entity_id'] or not first.link.is_fixed or not second.link.is_fixed
                            or any(int(g.link.idx) not in proof['native_link_ids'] for g in (first,second))
                            or any(owners[k].get('kind')!='obstacle'
                                or owners[k].get('fixed_peer_arm_id')!=name
                                or owners[k].get('original_robot_link_name')!=g.link.name
                                for k,g in ((a,first),(b,second)))):
                        raise ContractError('Frozen internal exclusion lost its exact native geometry identity')
                    reason='source_qualified_fixed_peer_internal_geometry'
                    static={'fixed_peer_arm_id':name,'geometry_ledger_sha256':single['fixed_peer_ledger_sha256'],
                            'scope':'Immutable fixed compound geometry. Internal robot self-dynamics is not simulated. Cross-entity pairs are not exempted.'}
                elif first.link.is_fixed and second.link.is_fixed:
                    reason='world_fixed_native_exclusion_with_conservative_AABB_guard'
                    if owners[a]['kind']=='floor' or owners[b]['kind']=='floor':
                        other=second if owners[a]['kind']=='floor' else first
                        box=bounds(other)
                        if box[0,2]<-1e-7:
                            detail={'geom_ids':[a,b],'owners':[copy.deepcopy(owners[a]),copy.deepcopy(owners[b])],
                                'fixed_flags':[bool(first.link.is_fixed),bool(second.link.is_fixed)],
                                'other_geom_id':int(other.idx),'other_entity':str(other.entity.name),
                                'other_link_name':str(other.link.name),'other_link_id':int(other.link.idx),
                                'other_world_aabb_m':box.tolist(),'floor_z_m':0.,'numeric_m':1e-7,
                                'depth_below_floor_m':float(-box[0,2]),
                                'entity_world_position_m':array(other.entity.get_pos(relative=False)).tolist(),
                                'entity_world_quaternion_wxyz':array(other.entity.get_quat(relative=False)).tolist(),
                                'native_geom_data':array(other.data).tolist(),
                                'geom_local_position_m':array(other.init_pos).tolist(),
                                'geom_local_quaternion_wxyz':array(other.init_quat).tolist(),
                                'scope':'Conservative excluded fixed-pair AABB diagnostic. Predicate unchanged.'}
                            raise ContractError('Excluded fixed geometry penetrates floor: '+json.dumps(detail,allow_nan=False))
                        static={'other_world_aabb_m':box.tolist(),'floor_z_m':0.,'numeric_m':1e-7}
                    else:
                        aa,bb=bounds(first),bounds(second)
                        if not np.any(aa[1]<bb[0]-1e-7) and not np.any(bb[1]<aa[0]-1e-7):
                            detail={'geom_ids':[a,b],'owners':[copy.deepcopy(owners[a]),copy.deepcopy(owners[b])],
                                'world_aabb_a_m':aa.tolist(),'world_aabb_b_m':bb.tolist(),
                                'scope':'Conservative excluded fixed-pair AABB diagnostic. Predicate unchanged.'}
                            raise ContractError('Excluded fixed bodies overlap or have unverified clearance: '+json.dumps(detail,allow_nan=False))
                        static={'world_aabb_a_m':aa.tolist(),'world_aabb_b_m':bb.tolist()}
                else:raise ContractError(f'Required episode pair filtered: {a},{b}')
            rows.append({'geom_ids':[a,b],'owners':[copy.deepcopy(owners[a]),copy.deepcopy(owners[b])],
                         'enabled':enabled,'exclusion':reason,'static_check':static})
    return {'status':'RECORDED_NATIVE_PAIR_POLICY_NOT_CONTINUOUS_SAFETY','pairs':rows,
            'geometry_count':n,'enabled_pair_count':len(valid),'all_unordered_pairs':len(rows),
            'finger_equalities':[list(v) for v in actual],
            'native_pair_matrix_sha256':digest(host.tolist()),'all_required_dynamic_pairs_enabled':True,
            'single_active_arm_contract':copy.deepcopy(single),
            'static_guard':'Native world vertex AABBs, conservative rejection of ambiguous excluded pairs'}


def query_obstacle_geometry(item, clearance):
    """Inflate the planning box only in the floor's admissible half-space.

    The physical/nominal map is unchanged. The floor independently rejects
    below-ground robot geometry, so underground obstacle inflation is redundant.
    """
    low=np.asarray(_q(item['lower_m'],3,'nominal obstacle lower'))
    high=np.asarray(_q(item['upper_m'],3,'nominal obstacle upper'))
    clearance=number(clearance,'planning obstacle clearance',minimum=0.)
    if np.any(low>=high):raise ContractError('Nominal obstacle must have positive dimensions')
    inflated_low=low-clearance;query_low=inflated_low.copy();query_low[2]=max(0.,query_low[2])
    query_high=high+clearance
    if not np.isfinite([*query_low,*query_high]).all() or np.any(query_low>=query_high):
        raise ContractError('Inflated obstacle has no finite above-ground volume')
    return {'id':identifier(item['id']),'nominal_lower_m':low.tolist(),'nominal_upper_m':high.tolist(),
        'inflated_lower_before_floor_intersection_m':inflated_low.tolist(),
        'query_lower_m':query_low.tolist(),'query_upper_m':query_high.tolist(),
        'size_m':(query_high-query_low).tolist(),'center_world_m':((query_low+query_high)/2).tolist(),
        'nominal_grounded':bool(low[2]==0.),'planning_clearance_m':clearance,
        'floor_intersection_z_m':0.,'physical_map_modified':False}


def _grounded_obstacle_floor_contact(row, owners, obstacles_by_geom):
    a,b=row['geom_ids']
    obstacle_id=a if owners[a]['kind']=='obstacle' else b
    floor_id=b if obstacle_id==a else a
    obstacle=obstacles_by_geom.get(obstacle_id)
    owner=owners[obstacle_id]
    component_id=owner.get('component_id',owner.get('entity'))
    if 'component_id' in owner and (obstacle is None or
            obstacle.get('component_id')!=component_id or
            obstacle.get('assembly_id')!=owner.get('entity') or obstacle.get('entity')!=owner.get('entity')):
        return False
    if (obstacle is None or owners[floor_id].get('entity')!='floor'
            or obstacle.get('geom_id')!=obstacle_id or obstacle.get('id')!=component_id
            or obstacle.get('nominal_grounded') is not True
            or obstacle['nominal_lower_m'][2]!=0. or obstacle['query_lower_m'][2]!=0.):
        return False
    observed=obstacle['observed_native_geometry']
    if not all(obstacle['native_representation_checks'].values()):return False
    center=np.asarray(observed['center_world_m']);size=np.asarray(observed['size_m'])
    bottom=center[2]-size[2]/2
    point=np.asarray(row['position_world_m']);normal=np.asarray(row['normal_world'])
    surface_tolerance=NUMERIC_TOLERANCE_M+row['penetration_m']
    return bool(abs(bottom)<=1e-7 and abs(point[2])<=surface_tolerance
        and abs(point[2]-bottom)<=surface_tolerance
        and np.all(np.abs(point[:2]-center[:2])<=size[:2]/2+NUMERIC_TOLERANCE_M)
        and abs(normal[2])>=MINIMUM_NORMAL_COSINE)


def classify_episode_contacts(raw, *, owners, components_by_geom, body_pose, stations_by_geom,
                              obstacles_by_geom=None):
    """Unloaded means no pad/handle allowance, including at a target endpoint."""
    rows,_=classify_supported_contacts(raw,components_by_geom=components_by_geom,
        pads_by_geom={},body_pose=body_pose,finger_poses={},stations_by_geom=stations_by_geom,phase='settle')
    for row in rows:
        a,b=row['geom_ids']
        if a not in owners or b not in owners:raise ContractError('Unknown contact owner')
        oa,ob=owners[a],owners[b];kinds={oa['kind'],ob['kind']}
        row['owners']=[copy.deepcopy(oa),copy.deepcopy(ob)]
        valid=0.<=row['penetration_m']<=MAX_PENETRATION_M and row['normal_unit_error']<=.001
        if row['category']=='UNALLOWED_CONTACT':
            if contents_contact_witness_valid(row['penetration_m'],row['normal_unit_error'],(oa,ob)):
                row['category']='contents_contact'
            elif valid and kinds=={'support','floor'}:
                row['category']='named_support_floor'
            elif valid and kinds=={'obstacle','floor'} and _grounded_obstacle_floor_contact(row,owners,obstacles_by_geom or {}):
                row['category']='named_grounded_obstacle_bottom_floor'
    return rows


class _FrozenFixtureReader:
    """Fresh public-getter pack. Only topology/indices, never native values, cache.

    The owner still performs the original exact virtual snapshot comparison.
    In particular a held subclass removes its movable proxies after this read.
    """
    def __init__(self, owner):
        from physics.episode_observation import _dependencies
        self.owner=owner;self.scene=owner.scene;self.solver=self.scene.rigid_solver
        self.torch,self.qd,self.sources=_dependencies(self.scene)
        if self.solver.n_envs!=0 or len(owner.arms) not in (2,3):
            raise ContractError('Fixture pack requires one unbatched two/three-arm scene')
        self.arm_items=tuple(owner.arms.items())
        self.bodies=tuple([owner.tray,*owner.supports,*owner.contents,*owner.obstacles])
        self.pose_entities=(*self.bodies,*(a for _,a in self.arm_items))
        if owner.active_arm_id not in owner.arms or owner.arm is not owner.arms[owner.active_arm_id]:
            raise ContractError('Fixture pack active-arm identity differs')
        if any(a.n_qs!=9 for _,a in self.arm_items):
            raise ContractError('Fixture pack requires all nine native Panda coordinates')
        self.q_indices=tuple(int(a.q_start)+i for _,a in self.arm_items for i in range(9))
        self.link_indices=tuple(int(e.base_link_idx) for e in self.pose_entities)
        named={}
        for e in self.bodies:
            if e.name in named and named[e.name] is not e:
                raise ContractError('Fixture name aliases different native entities')
            named[e.name]=e
        if (len(set(self.q_indices))!=len(self.q_indices) or min(self.q_indices)<0
                or min(self.link_indices)<0):
            raise ContractError('Invalid fixture native index map')
        self.shapes=((len(self.q_indices),),(len(self.link_indices),3),(len(self.link_indices),4))
        self._signature=self._topology();self._dtype_device=None
        self.reads=0;self.parity_checks=0

    def _topology(self):
        o=self.owner
        return (id(o.scene),id(o.scene.rigid_solver),o.scene.rigid_solver.n_envs,o.active_arm_id,id(o.arm),
            tuple((n,id(a),a.idx,a.q_start,a.n_qs,a.base_link_idx) for n,a in o.arms.items()),
            tuple((id(e),e.idx,e.name,e.base_link_idx) for e in [o.tray,*o.supports,*o.contents,*o.obstacles]),
            self.q_indices,self.link_indices,self.shapes)

    def begin(self):
        from physics.episode_observation import _clock
        if self._topology()!=self._signature:raise ContractError('Fixture pack topology/index binding changed')
        return _clock(self.scene)

    def finish(self,clock):
        if self.begin()!=clock:raise ContractError('Native clock changed during fixture readback')

    def read(self):
        clock=self.begin();self.qd.sync();s=self.solver
        tensors=(s.get_qpos(qs_idx=self.q_indices),
            s.get_links_pos(self.link_indices,ref='link_origin',relative=False),
            s.get_links_quat(self.link_indices,relative=False))
        for value,shape in zip(tensors,self.shapes):
            if (not isinstance(value,self.torch.Tensor) or tuple(value.shape)!=shape
                    or value.dtype not in (self.torch.float32,self.torch.float64)):
                raise ContractError('Unexpected fixture getter shape/dtype')
        identities={(v.dtype,v.device) for v in tensors}
        if len(identities)!=1:raise ContractError('Fixture getters disagree on native dtype/device')
        identity=next(iter(identities))
        if self._dtype_device is not None and identity!=self._dtype_device:
            raise ContractError('Fixture getter dtype/device changed')
        self._dtype_device=identity
        host=array(self.torch.cat([v.reshape(-1) for v in tensors])).copy()
        if host.shape!=(sum(math.prod(s) for s in self.shapes),) or not np.isfinite(host).all():
            raise ContractError('Nonfinite/wrong-shaped packed fixture values')
        nq=len(self.q_indices);nl=len(self.link_indices)
        q=host[:nq].reshape((-1,9));positions=host[nq:nq+3*nl].reshape((nl,3))
        quaternions=host[nq+3*nl:].reshape((nl,4))
        poses=[{'frame':'world','position_m':p.tolist(),'quaternion_wxyz':v.tolist()}
               for p,v in zip(positions,quaternions)]
        self.finish(clock);self.reads+=1
        return {'peer_q':{n:q[i].tolist() for i,(n,_) in enumerate(self.arm_items) if n!=self.owner.active_arm_id},
            'active_fingers':q[[n for n,_ in self.arm_items].index(self.owner.active_arm_id),7:].tolist(),
            'bodies':{e.name:p for e,p in zip(self.bodies,poses)},
            'robot_roots':{n:poses[len(self.bodies)+i] for i,(n,_) in enumerate(self.arm_items)}}

    def metadata(self):
        return {'schema_version':'episode-frozen-fixture-pack-v1','mode':'batched',
            'q_indices':list(self.q_indices),'pose_link_indices':list(self.link_indices),
            'body_entities':[e.name for e in self.bodies],'arm_order':[n for n,_ in self.arm_items],
            'source_hashes':dict(self.sources),'public_getters_per_read':3,
            'modeled_cpu_transfers_per_read':1,'fresh_reads':self.reads,'exact_virtual_parity_checks':self.parity_checks,
            'native_dtype_device':None if self._dtype_device is None else [str(x) for x in self._dtype_device],
            'scalar_fallback':False,'cached_native_values':False}


class EpisodeJointScene(JointScene):
    """Query-only two/three-arm snapshot. Caller owns Genesis initialization.

    ``close`` destroys only this scene, never the global Genesis runtime.
    ``arm`` is the active arm. ``arms`` maps every declared ID to its entity.
    ``qhome`` is the generic Panda seed, not a supplied route or task solution.
    """
    def __init__(self, case, *, active_arm_id, robots, model, body_pose, body_id,
                 asset_directory, support_stations=(), contents=(), render=False, observed_snapshot=None):
        self.snapshot=validate_episode_snapshot(case,active_arm_id,robots,model,body_pose,
                                               body_id,support_stations,contents,observed_snapshot=observed_snapshot)
        self._observed_snapshot=copy.deepcopy(observed_snapshot)
        self.case=copy.deepcopy(self.snapshot['case']);self.model=model
        self.active_arm_id=active_arm_id;self._closed=False;self.scene=None
        self._source_root,self._source_hashes=verify_contact_sources()
        self._compound_groups=validate_compound_obstacle_groups(self.case)
        self._embedded_supports=validate_embedded_support_components(self.case,self.snapshot['support_stations'])
        self._tracking_clearance_nominal_mounts=self._validate_nominal_mount_clearance()
        self._standalone_support_stations=[s for s in self.snapshot['support_stations'] if s['id'] not in self._embedded_supports]
        self._compound_assets=[]
        self._compound_directory=Path(asset_directory)/'obstacles'
        if self._compound_groups:
            self._source_hashes.update(verify_compound_sources(self._source_root))
        self.asset=write_carried_vessel_urdf(Path(asset_directory),model)
        self._asset_components=validate_asset_components(self.asset,model)
        self._last_detection=None;self._fixed_checks=0;self._baseline=None
        self.states_checked=0;self.edges_checked=0;self.collision_queries=0
        self.failure_reasons={};self.last_failure=None;self.query_wall_s=0.
        try:self._build(render)
        except BaseException:
            self.close()
            raise

    def _validate_nominal_mount_clearance(self):
        """Keep occupied base plates nominal while other query boxes expand.

        This removes only an extra planning buffer from authenticated robot
        installation plates. Their real collision geometry/pairs stay active.
        An arbitrary obstacle cannot be named as a buffer exemption.
        """
        from transport_challenge.three_arm_design import BASE_SURFACE_CLEARANCE_M
        values=self.case['planning'].get('tracking_clearance_exempt_component_ids',[])
        if not isinstance(values,list) or any(type(v) is not str for v in values) or len(set(values))!=len(values):
            raise ContractError('Nominal mount query components must be a distinct explicit ID list')
        if not values:return frozenset()
        expected=set()
        for robot in self.snapshot['robots']:
            base=episode_base_position(robot,self.case)
            matches=[o['id'] for o in self.case['obstacles'] if o['id'].startswith('terrain_')
                and all(o['lower_m'][i]<=base[i]<=o['upper_m'][i] for i in (0,1))
                and abs(o['upper_m'][2]+BASE_SURFACE_CLEARANCE_M-base[2])<=1e-7]
            if len(matches)!=1:raise ContractError('Occupied robot base lacks its unique nominal terrain plate')
            expected.add(matches[0])
        if set(values)!=expected:raise ContractError('Only the exact occupied robot installation plates keep nominal query geometry')
        return frozenset(values)

    def _build(self, render):
        import genesis as gs
        self.gs=gs
        visual=validate_query_visual(self.case,self.snapshot['support_stations'])
        def appearance(color):return {} if color is None else {'surface':gs.surfaces.Default(color=_visual_rgb(color))}
        self.scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.0025,substeps=1),
            rigid_options=gs.options.RigidOptions(enable_self_collision=True,enable_neutral_collision=True,
                # This isolated query uses poses and contacts, never constraint forces.
                # Avoid compiling the unused force solver in Scene.build's warmup.
                enable_collision=True,disable_constraint=True,
                enable_adjacent_collision=False,use_hibernation=False,contact_pruning_tolerance=None),
            vis_options=gs.options.VisOptions(**({'shadow':False,'ambient_light':(.65,.65,.65),'background_color':(1.,1.,1.)} if visual is not None else {})),
            renderer=gs.renderers.Rasterizer(),show_viewer=False)
        self.floor=self.scene.add_entity(gs.morphs.Plane(),name='floor',
            **appearance(None if visual is None else visual['floor_color']))
        self.arms={}
        for robot in self.snapshot['robots']:
            b=robot['base_placement']
            self.arms[robot['id']]=self.scene.add_entity(gs.morphs.MJCF(file=self.case['robot_model'],
                pos=episode_base_position(robot,self.case),euler=(0.,0.,math.degrees(b['yaw_rad'])),
                requires_jac_and_IK=True),name=robot['id'])
        self.arm=self.arms[self.active_arm_id]
        self.clearance=self.case['planning'].get('tracking_clearance_m',0.)
        self.obstacles=[];self._obstacle_specs=[]
        if self._compound_groups:
            nominal=set(self._embedded_supports.values())|self._tracking_clearance_nominal_mounts
            specs={item['id']:query_obstacle_geometry(item,0. if item['id'] in nominal else self.clearance)
                   for item in self.case['obstacles']}
            self._obstacle_specs=list(specs.values())
            for group in self._compound_groups:
                colors={} if visual is None else {'component_colors':{name:visual['component_colors'][name] for name in group['component_ids']}}
                if visual is not None and 'hidden_component_ids' in visual:
                    colors['hidden_component_ids']=[name for name in group['component_ids'] if name in visual['hidden_component_ids']]
                asset=write_compound_obstacle_asset(self._compound_directory,group,
                    [specs[name] for name in group['component_ids']],**colors)
                self._compound_assets.append(asset)
                self.obstacles.append(self.scene.add_entity(gs.morphs.URDF(file=asset['path'],
                    pos=asset['center_world_m'],fixed=False,align=False,merge_fixed_links=True),name=group['id']))
        else:
            for item in self.case['obstacles']:
                spec=query_obstacle_geometry(item,0. if item['id'] in self._tracking_clearance_nominal_mounts else self.clearance);self._obstacle_specs.append(spec)
                hidden={} if visual is None or 'hidden_component_ids' not in visual else {'visualization':item['id'] not in visual['hidden_component_ids']}
                self.obstacles.append(self.scene.add_entity(gs.morphs.Box(size=spec['size_m'],pos=spec['center_world_m'],fixed=False,**hidden),name=item['id'],
                    **appearance(None if visual is None else visual['component_colors'][item['id']])))
        self.supports=[self.scene.add_entity(gs.morphs.Box(size=s['size_m'],pos=s['center_world_m'],fixed=False,
                         contype=SUPPORT_CONTACT_MASK,conaffinity=SUPPORT_CONTACT_MASK),name=s['id'],
                         **appearance(None if visual is None else visual.get('support_colors',{}).get(s['id'])))
                       for s in self._standalone_support_stations]
        pose=self.snapshot['body_world']
        self.tray=self.scene.add_entity(gs.morphs.URDF(file=str(self.asset.path),pos=pose['position_m'],
            quat=pose['quaternion_wxyz'],fixed=False,align=False,merge_fixed_links=True),
            material=gs.materials.Rigid(friction=1.),name=self.snapshot['body_id'],
            **appearance(None if visual is None else visual.get('vessel_color')))
        self.contents=[self.scene.add_entity(gs.morphs.Sphere(radius=s['radius_m'],pos=s['center_world_m']),
                       material=gs.materials.Rigid(rho=600.,friction=.4),name=s['id']) for s in self.snapshot['contents']]
        camera={'pos':(1.9,-1.6,1.65),'lookat':(.35,0,.43),'fov':42} if visual is None else {
            'pos':visual['camera']['position_m'],'lookat':visual['camera']['lookat_m'],'fov':visual['camera']['fov_deg']}
        self.camera=self.scene.add_camera(res=(960,720),GUI=False,**camera) if render else None
        started=time.monotonic();self.scene.build();self.build_wall_s=time.monotonic()-started
        self.hand=self.arm.get_link(self.case['tool_link'])
        self._pads={};self.robot_limits={};self._initialization_representations={}
        for record in self.snapshot['robots']:
            arm=self.arms[record['id']]
            dofs=[i for name in self.case['arm_joint_names'] for i in arm.get_joint(name).dofs_idx_local]
            qs=[i for name in self.case['arm_joint_names'] for i in arm.get_joint(name).qs_idx_local]
            if arm.n_qs!=9 or dofs!=list(range(7)) or qs!=dofs:raise ContractError('Unexpected native Panda joint mapping')
            q,representation=native_snapshot_command(record,*arm.get_dofs_limit(),observed_snapshot=self._observed_snapshot)
            self._initialization_representations[record['id']]=representation
            self.robot_limits[record['id']]=list(zip(representation['native_lower'],representation['native_upper']))
            arm.set_qpos(q)  # Isolated query initialization, never the live episode.
            observed_q=array(arm.get_qpos())
            if observed_q.shape!=(9,) or not np.array_equal(observed_q,q):
                raise ContractError('Native initialization readback differs from represented command: '+repr({
                    'representation':representation,'observed_q':observed_q.tolist()}))
            representation['native_readback_exact']=True
            b=record['base_placement'];root_pose=self._pose(arm)
            expected_q=np.array([math.cos(b['yaw_rad']/2),0.,0.,math.sin(b['yaw_rad']/2)])
            representation['base_position_validation']=validate_native_base_position(
                episode_base_position(record,self.case),arm.get_pos(relative=False),arm_id=record['id'])
            if min(np.linalg.norm(np.asarray(root_pose['quaternion_wxyz'])-expected_q),
                   np.linalg.norm(np.asarray(root_pose['quaternion_wxyz'])+expected_q))>1e-7:
                raise ContractError('Native robot base orientation differs from named installation: '+repr({
                    'arm_id':record['id'],'base_position_validation':representation['base_position_validation'],
                    'expected_quaternion_wxyz':expected_q.tolist(),'observed_pose':root_pose,
                    'quaternion_sign_distance_limit':1e-7}))
            self._pads[record['id']]=match_native_pads(arm,read_panda_pad_primitives(self._source_root/PANDA_XML),gs.GEOM_TYPE.BOX)
        active=next(r for r in self.snapshot['robots'] if r['id']==self.active_arm_id)
        self.base=tuple(active['base_placement'][k] for k in ('x_m','y_m','yaw_rad'))
        self.bounds=tuple(self.robot_limits[self.active_arm_id][:7])
        initial_command=self._initialization_representations[self.active_arm_id]['native_command']
        self.fingers=tuple(initial_command[7:])
        self.dofs=list(range(7));self.q_indices=list(range(7));self.qhome=(0.,-.785,0.,-2.356,0.,1.571,.785)
        self.hand_workspace=np.asarray(self.case['tool_workspace_m'])+np.array([self.clearance,-self.clearance])
        self._components=validate_built_components(self.tray,self.model,self._asset_components,gs.GEOM_TYPE.BOX)
        self._station_map={v['geom_id']:v for v in (validate_built_support(e,s,gs.GEOM_TYPE.BOX)
            for e,s in zip(self.supports,self._standalone_support_stations))}
        # Same exact native BOX/pose/mask checks as stations, but a distinct
        # registry. Only explicitly embedded plate geom IDs are named supports.
        self._obstacle_map={}
        if self._compound_groups:
            for entity,asset in zip(self.obstacles,self._compound_assets):
                for native in validate_built_compound(entity,asset,gs.GEOM_TYPE.BOX,fixed=False):
                    if native['geom_id'] in self._obstacle_map:
                        raise ContractError('Compound native geometry IDs repeat')
                    self._obstacle_map[native['geom_id']]=native
        else:
            for entity,spec in zip(self.obstacles,self._obstacle_specs):
                native=validate_built_support(entity,spec,gs.GEOM_TYPE.BOX)
                native['clearance_inflation_m']=self.clearance
                self._obstacle_map[native['geom_id']]=native
        native_components={v.get('component_id'):v for v in self._obstacle_map.values()}
        for station in self.snapshot['support_stations']:
            if station['id'] in self._embedded_supports:
                component=self._embedded_supports[station['id']]
                if component not in native_components:raise ContractError('Embedded native plate is missing')
                native=embedded_support_station(native_components[component],station)
                if native['geom_id'] in self._station_map:raise ContractError('Support geometry registered twice')
                self._station_map[native['geom_id']]=native
        self._sphere_geometry=[]
        for entity,sphere in zip(self.contents,self.snapshot['contents']):
            if len(entity.geoms)!=1:raise ContractError('Each content sphere needs exactly one native geometry')
            g=entity.geoms[0];data=array(g.data)
            if (g.type!=gs.GEOM_TYPE.SPHERE or g.entity is not entity or data.shape!=(7,) or
                not np.isfinite(data).all() or not math.isclose(data[0],sphere['radius_m'],rel_tol=1e-6,abs_tol=1e-9)
                or np.any(data[1:]!=0) or g.contype!=0xFFFF or g.conaffinity!=0xFFFF):
                raise ContractError('Native content sphere differs from snapshot')
            self._sphere_geometry.append({**copy.deepcopy(sphere),'geom_id':int(g.idx),'native_data':data.tolist()})
        self._owners={}
        for kind,entities in (('floor',[self.floor]),('arm',list(self.arms.values())),('tray',[self.tray]),
                              ('support',self.supports),('bead',self.contents),('obstacle',self.obstacles)):
            for entity in entities:
                for g in entity.geoms:self._owners[int(g.idx)]={'kind':kind,'entity':entity.name,'link':g.link.name,'link_id':int(g.link.idx)}
        if self._compound_groups:
            for gid,spec in self._obstacle_map.items():
                self._owners[gid]['component_id']=spec['component_id']
                self._owners[gid]['assembly_id']=spec['assembly_id']
        for gid,station in self._station_map.items():
            if station.get('embedded_in_rigid_assembly') is True:
                owner=self._owners[gid]
                if (owner['entity']!=station['assembly_id'] or owner['component_id']!=station['component_id']
                        or owner['link_id']!=station['link_id']):raise ContractError('Embedded native support ownership changed')
                owner.update(kind='support',station_id=station['id'],geom_id=gid)
        self.geometry_names=copy.deepcopy(self._owners)
        self._witness_contract=verify_query_witness_runtime(self.scene.rigid_solver,self._source_root)
        self._pair_coverage=audit_episode_pairs(self.scene.rigid_solver,list(self.arms.values()),self._owners)
        self._baseline=self._read_frozen_state()
        # Verify authored fixture positions before adopting native float32 readback.
        wanted=[(self.tray,pose['position_m'],pose['quaternion_wxyz'])]
        wanted += [(e,s['center_world_m'],[1.,0.,0.,0.]) for e,s in zip(self.supports,self._standalone_support_stations)]
        wanted += [(e,s['center_world_m'],[1.,0.,0.,0.]) for e,s in zip(self.contents,self.snapshot['contents'])]
        for entity,p,q in wanted:
            actual=self._pose(entity)
            orientation=fixed_fixture_quaternion_comparison(q,actual['quaternion_wxyz'])
            if (not np.allclose(actual['position_m'],p,rtol=0.,atol=1e-7) or
                orientation['normalized_sign_distance']>1e-7):
                raise ContractError('Native fixed fixture differs from authored world snapshot')
        self.initial_state_validation={'q_rad':initial_command[:7],'authored_q_rad':list(active['q_rad']),
            'valid':self.state_valid(tuple(initial_command[:7])),
            'failure':copy.deepcopy(self.last_failure),'scope':'Unloaded snapshot geometry only, not physical initial conditions'}

    @staticmethod
    def _pose(entity):
        p=array(entity.get_pos(relative=False));q=array(entity.get_quat(relative=False))
        if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite([*p,*q]).all():raise ContractError('Nonfinite native pose')
        return {'frame':'world','position_m':p.tolist(),'quaternion_wxyz':q.tolist()}

    def _read_frozen_state(self):
        reader=getattr(self,'_fixture_reader',None)
        if reader is not None:return reader.read()
        return {'peer_q':{name:array(arm.get_qpos()).copy().tolist() for name,arm in self.arms.items() if name!=self.active_arm_id},
            'active_fingers':array(self.arm.get_qpos()).copy().tolist()[7:],
            'bodies':{e.name:self._pose(e) for e in [self.tray,*self.supports,*self.contents,*self.obstacles]},
            'robot_roots':{name:self._pose(arm) for name,arm in self.arms.items()}}

    def enable_batched_fixture_readback(self):
        """Opt in only after exact scalar/packed virtual snapshot + baseline parity.

        No scene mutation, detection, physics step or predicate change occurs.
        Default instances continue to execute the original scalar getter path.
        """
        self._require_open()
        if getattr(self,'_fixture_reader',None) is not None:
            return self.verify_batched_fixture_readback()
        if self._baseline is None:raise ContractError('Fixture pack needs a built immutable baseline')
        self._fixture_reader=_FrozenFixtureReader(self)
        try:return self.verify_batched_fixture_readback()
        except BaseException:
            del self._fixture_reader
            raise

    def verify_batched_fixture_readback(self):
        """Read both paths at one clock, including subclass proxy exclusions."""
        self._require_open();reader=getattr(self,'_fixture_reader',None)
        if reader is None:raise ContractError('Batched fixture readback is not enabled')
        clock=reader.begin();self._fixture_reader=None
        try:scalar=self._read_frozen_state()
        finally:self._fixture_reader=reader
        packed=self._read_frozen_state();reader.finish(clock)
        if scalar!=packed or packed!=self._baseline:
            raise ContractError('Scalar/packed virtual fixture snapshot or immutable baseline differs')
        reader.parity_checks+=1
        return reader.metadata()

    def _after_arm_q_placement(self):
        if self._baseline is not None and self._read_frozen_state()!=self._baseline:
            raise ContractError('A peer, finger, robot base or fixed world fixture changed during isolated query')
        self._fixed_checks+=1

    def _require_open(self):
        if self._closed:raise ContractError('The isolated episode query scene is closed')

    def state_valid(self,q):
        self._require_open();self._last_detection=None
        try:q=_q(q,7,'active arm q')
        except ContractError:return self._reject('invalid_joint_state')
        return super().state_valid(q)

    def edge_valid(self,a,b):
        self._require_open()
        try:a,b=_q(a,7,'edge start'),_q(b,7,'edge goal')
        except ContractError:return self._reject('invalid_joint_edge')
        return super().edge_valid(a,b)

    def fk(self,q):
        self._require_open()
        return super().fk(_q(q,7,'FK q'))

    def ik(self,position_m,quaternion_wxyz,seed_q,*,position_tolerance_m=None,orientation_tolerance_rad=None):
        self._require_open()
        p,q,seed=_q(position_m,3,'IK position'),_q(quaternion_wxyz,4,'IK quaternion'),_q(seed_q,7,'IK seed')
        if abs(math.fsum(v*v for v in q)-1.)>1e-6 or any(not lo<=v<=hi for v,(lo,hi) in zip(seed,self.bounds)):
            raise ContractError('IK requires a unit quaternion and native-bounded seed')
        precision={}
        if position_tolerance_m is not None:precision['position_tolerance_m']=position_tolerance_m
        if orientation_tolerance_rad is not None:precision['orientation_tolerance_rad']=orientation_tolerance_rad
        try:return super().ik(p,q,seed,**precision)
        finally:self._after_arm_q_placement()

    def _query_collision_pairs(self):
        # Public global query returns every retained native pair. Copy witnesses
        # immediately, before any other detection or pose operation.
        pairs=self.scene.rigid_solver.detect_collision()
        raw=query_witness_snapshot(self.scene.rigid_solver.collider,pairs,set(self._owners),self._owners)
        self.scene.rigid_solver.check_errno();self.collision_queries+=1
        self._after_arm_q_placement()
        rows=classify_episode_contacts(raw,owners=self._owners,
            components_by_geom={v['geom_id']:v for v in self._components},body_pose=self._pose(self.tray),
            stations_by_geom=self._station_map,obstacles_by_geom=self._obstacle_map)
        forbidden=sorted({tuple(sorted(r['geom_ids'])) for r in rows if r['category']=='UNALLOWED_CONTACT'})
        self._last_detection={'public_pair_count':len(pairs),'copied_contact_count':len(rows),
            'pair_witness_multiset_agrees':True,'rows':rows,'forbidden_pairs':[list(p) for p in forbidden],
            'world_body_pose':self._pose(self.tray),'active_arm_id':self.active_arm_id,
            'cross_arm_pad_exemptions':[],'forces_used':False}
        return np.asarray(forbidden,dtype=np.int32).reshape((-1,2))

    def evidence(self):
        record=super().evidence()
        record.update(schema_version='episode-joint-query-evidence-v1',snapshot=copy.deepcopy(self.snapshot),
            snapshot_sha256=digest(self.snapshot),motion_scope='unloaded_approach_or_retreat_only',
            full_scene_collision_query=True,held_carried_query_supported=False,contents_collision_queried=bool(self.contents),
            native_pending_physical_validation=True,native_pair_coverage=copy.deepcopy(self._pair_coverage),
            native_robot_limits=copy.deepcopy(self.robot_limits),native_pad_ledgers=copy.deepcopy(self._pads),
            initialization_representation=copy.deepcopy(self._initialization_representations),
            native_vessel_components=copy.deepcopy(self._components),native_spheres=copy.deepcopy(self._sphere_geometry),
            native_station_geometry=copy.deepcopy(self._station_map),initial_state_validation=copy.deepcopy(self.initial_state_validation),
            native_query_obstacle_geometry=copy.deepcopy(self._obstacle_map),
            last_detection=copy.deepcopy(self._last_detection),frozen_fixture_readback_checks=self._fixed_checks,
            immutable_native_snapshot=copy.deepcopy(self._baseline),query_witness_contract=copy.deepcopy(self._witness_contract),
            contact_source_sha256=copy.deepcopy(self._source_hashes),body_pose_setters=0,physics_steps=0,
            route_generated=False,physical_handoff_verified=False,closed=self._closed)
        if not self.contents:
            record['contents_model_scope']='Empty content geometry. Robot, vessel and map queries only. Free-particle collision safety/retention requires actual native physics.'
        if self._compound_groups:
            record['compound_obstacles']={'declared_groups':copy.deepcopy(self._compound_groups),
                'assets':copy.deepcopy(self._compound_assets),'native_contact_mask':1,
                'internal_pair_rule':'Only actual native same-rigid-link pairs excluded. No inter-assembly contact permission.',
                'native_component_count':len(self._obstacle_map),
                'source_scope':'Source-pinned URDF primitive parser and full native pair audit. Zero-step geometry only.'}
        if self._embedded_supports:
            record['embedded_support_components']=copy.deepcopy(self._embedded_supports)
            record['support_representation_scope']='Only named plate geometry is a support. Other same-link fixture parts remain obstacles. No duplicate support Box.'
        if self._tracking_clearance_nominal_mounts:
            record['tracking_clearance_nominal_mount_component_ids']=sorted(self._tracking_clearance_nominal_mounts)
            record['nominal_mount_clearance_scope']='Exact occupied base plate boxes retain nominal geometry and all collisions. Other query obstacles use the declared extra planning clearance.'
        if getattr(self,'_fixture_reader',None) is not None:
            record['fixture_readback']=self._fixture_reader.metadata()
        return record

    def close(self):
        """Idempotently destroy only this owned scene. Never call gs.destroy."""
        if not getattr(self,'_closed',True):
            self._closed=True
            if self.scene is not None:self.scene.destroy()

