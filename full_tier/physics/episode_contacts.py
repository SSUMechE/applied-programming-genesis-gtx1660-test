"""Pure live-contact classification for one tray and two/three Panda arms.

The caller copies the complete global native contact buffer and supplies body
and finger poses from that same detection instant, normally pre-integration.
This module never reads a simulator, recomputes collision, steps or controls.
It returns one classified row per raw witness without filtering. The runtime
must retain the original raw buffer, including any force fields, and verify
native/copied/classified counts, contact clocks and live pair-policy stamps.

Geometry registries must come from the actual built model. Each arm's ten pads
are the unchanged output of match_native_pads, indexed by global geometry ID.
Namespacing is arm ID -> global geometry ID, never the shared finger-link name.
Input consistency checks cannot authenticate invented geometry metadata.
"""
from __future__ import annotations

import copy
from numbers import Integral

from physics.gravity_feedforward import rotation_matrix_wxyz
from physics.handle_contacts import (
    MAX_PENETRATION_M, classify_supported_contacts, validate_contact_snapshot,
    contents_contact_witness_valid,
)
from transport_challenge.contracts import ContractError, digest, point

_FINGERS = {'left_finger', 'right_finger'}
_KINDS = {'arm', 'tray', 'bead', 'support', 'floor', 'obstacle'}
_NO_HOLDER_PHASES = {'approach', 'retreat', 'settle'}
_PHASES = _NO_HOLDER_PHASES | {'close', 'hold', 'carry', 'place', 'release', 'released_settle'}


def _name(value, label):
    if type(value) is not str or not value or value.strip() != value:
        raise ContractError(label+' must be an explicit nonempty name')
    return value


def _gid(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ContractError('Use exact nonnegative integer geometry/link IDs')
    return int(value)


def _registry(value, label):
    if not isinstance(value, dict):
        raise ContractError(label+' must be a dictionary indexed by native geometry ID')
    if any(not isinstance(v, dict) for v in value.values()):
        raise ContractError(label+' entries must be geometry records')
    return {_gid(k): v for k, v in value.items()}


def _pose(value, label):
    if not isinstance(value, dict) or value.get('frame', 'world') != 'world':
        raise ContractError(label+' must contain a world-frame pose')
    point(value['position_m'], 3)
    rotation_matrix_wxyz(value['quaternion_wxyz'])


def validate_single_active_contact_contract(value, arm_ids, owners=None):
    """Validate explicit fixed-peer mode; native authenticity remains caller-owned."""
    if value is None:
        return None
    from physics.fixed_peer_arms import validated_single_active_contract
    try:
        proof = validated_single_active_contract(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError('Invalid single-active fixed-peer contract: '+str(exc)) from exc
    if set(arm_ids) != {proof['active_arm_id']}:
        raise ContractError('Single-active contract requires exactly its one dynamic arm')
    if owners is not None:
        for name in proof['frozen_arm_ids']:
            rows = [o for o in owners.values() if o['entity'] == name]
            actual = proof['fixed_peers'][name]['actual']
            if (len(rows) != actual['geom_count'] or any(o['kind'] != 'obstacle' for o in rows)
                    or any(o['link_id'] not in actual['native_link_ids'] for o in rows)):
                raise ContractError('Fixed-peer collision ownership is missing or changed')
    return proof


def _validate_static(owners, components, pads_by_arm, stations, *, single_active_contract=None):
    owners = _registry(owners, 'owners')
    for owner in owners.values():
        if owner.get('kind') not in _KINDS:
            raise ContractError('Unknown geometry ownership kind')
        _name(owner.get('entity'), 'owner entity')
        _name(owner.get('link'), 'owner link')
        _gid(owner.get('link_id'))
    arm_ids = {v['entity'] for v in owners.values() if v['kind'] == 'arm'}
    single = validate_single_active_contact_contract(single_active_contract, arm_ids, owners)
    if ((len(arm_ids) != 1 if single is not None else len(arm_ids) not in (2, 3)) or not isinstance(pads_by_arm, dict)
            or set(pads_by_arm) != arm_ids):
        raise ContractError('Provide exactly the declared two/three arms in both namespaced pad and pose registries')
    components = _registry(components, 'components_by_geom')
    stations = _registry(stations, 'stations_by_geom')
    if set(components) != {g for g, o in owners.items() if o['kind'] == 'tray'}:
        raise ContractError('Tray component registry must exactly cover native tray ownership')
    if not components or len({owners[g]['entity'] for g in components}) != 1:
        raise ContractError('This policy covers exactly one identified tray')
    if set(stations) != {g for g, o in owners.items() if o['kind'] == 'support'}:
        raise ContractError('Named support registry must exactly cover native support ownership')
    component_names = []
    for gid, component in components.items():
        if (_gid(component.get('geom_id')) != gid or component.get('entity') != owners[gid]['entity']):
            raise ContractError('Tray geometry ownership differs from its built component')
        component_names.append(_name(component.get('component_id'), 'component ID'))
        if any(v <= 0 for v in point(component['size_m'], 3)):
            raise ContractError('Positive component dimensions required')
        point(component['center_body_m'], 3)
    if len(set(component_names)) != len(component_names):
        raise ContractError('Tray component IDs must be unique')
    for gid, station in stations.items():
        owner=owners[gid]
        if station.get('embedded_in_rigid_assembly') is True:
            from physics.compound_obstacles import embedded_support_station
            if (station.get('id')!=owner.get('station_id') or station.get('entity')!=owner['entity']
                    or station.get('assembly_id')!=owner.get('assembly_id') or station['assembly_id']!=owner['entity']
                    or station.get('component_id')!=owner.get('component_id')
                    or _gid(station.get('geom_id'))!=gid or _gid(owner.get('geom_id'))!=gid
                    or _gid(station.get('link_id'))!=owner['link_id']):
                raise ContractError('Embedded support ownership differs from the exact native plate')
            authored={k:station[k] for k in ('id','frame','units','axis_aligned','size_m','center_world_m')}
            if embedded_support_station(station.get('validated_native_component',{}),authored)!=station:
                raise ContractError('Embedded support native component evidence changed')
        elif station.get('id') != owner['entity'] or 'station_id' in owner:
            raise ContractError('Support station name differs from its native owner')
        if any(v <= 0 for v in point(station['size_m'], 3)):
            raise ContractError('Positive support dimensions required')
        point(station['center_world_m'], 3)
    pads = {}
    seen = set()
    for arm_id in sorted(arm_ids):
        records = _registry(pads_by_arm[arm_id], 'pads for '+arm_id)
        if len(records) != 10:
            raise ContractError('Each Panda needs ten matched pads and both same-instant finger poses')
        counts = {name: 0 for name in _FINGERS}
        for gid, pad in records.items():
            owner = owners.get(gid)
            link = pad.get('link')
            if (gid in seen or _gid(pad.get('geom_id')) != gid or owner is None
                    or owner['kind'] != 'arm' or owner['entity'] != arm_id
                    or link not in _FINGERS or owner['link'] != link
                    or _gid(pad.get('link_id')) != owner['link_id']):
                raise ContractError('Pad does not belong to this exact arm, link and global geometry ID')
            seen.add(gid)
            counts[link] += 1
            if any(v <= 0 for v in point(pad['size_m'], 3)):
                raise ContractError('Positive pad dimensions required')
            point(pad['center_link_m'], 3)
            for key in ('quaternion_link_wxyz', 'built_quaternion_link_wxyz'):
                quat = point(pad[key], 4)
                if any(abs(abs(v) - wanted) > 1e-7 for v, wanted in zip(quat, (1., 0., 0., 0.))):
                    raise ContractError('Only matched identity-local-rotation Panda pads are supported')
        if counts != {'left_finger': 5, 'right_finger': 5}:
            raise ContractError('Exactly five matched pads on each named finger required')
        pads[arm_id] = records
    return owners, components, pads, stations, arm_ids


def _validate_dynamic(body_pose, poses, arm_ids):
    if not isinstance(poses, dict) or set(poses) != arm_ids:
        raise ContractError('Provide exactly the declared two/three arms in both namespaced pad and pose registries')
    for arm_id in sorted(arm_ids):
        if not isinstance(poses[arm_id], dict) or set(poses[arm_id]) != _FINGERS:
            raise ContractError('Each Panda needs ten matched pads and both same-instant finger poses')
        for finger in _FINGERS:
            _pose(poses[arm_id][finger], arm_id+'/'+finger)
    _pose(body_pose, 'tray body')


def _validate(owners, components, pads_by_arm, body_pose, poses, stations, *, single_active_contract=None):
    policy = _validate_static(owners, components, pads_by_arm, stations, single_active_contract=single_active_contract)
    _validate_dynamic(body_pose, poses, policy[4])
    return policy


def _classify_checked(raw, *, policy, body_pose, finger_poses_by_arm, phase, active_holder_arm_id):
    """Shared dynamic predicates. Only the two owning entry points call this."""
    owners, components, pads, stations, arm_ids = policy
    if active_holder_arm_id is not None:
        if type(active_holder_arm_id) is not str or active_holder_arm_id not in arm_ids:
            raise ContractError('Contact permission must name exactly one registered arm')
        if phase in _NO_HOLDER_PHASES:
            raise ContractError('Approach, retreat and settling grant no pad/handle permission')
    copied = validate_contact_snapshot(raw)
    if not (set(map(int, copied['geom_a'])) | set(map(int, copied['geom_b']))) <= set(owners):
        raise ContractError('A global contact witness has no registered native owner')
    if ('link_a' in raw) != ('link_b' in raw):
        raise ContractError('Native link witness fields must be present together')
    if 'link_a' in raw:
        for link_key, geom_key in (('link_a', 'geom_a'), ('link_b', 'geom_b')):
            if len(raw[link_key]) != len(copied[geom_key]) or any(
                    _gid(link) != owners[int(geom)]['link_id']
                    for link, geom in zip(raw[link_key], copied[geom_key])):
                raise ContractError('Native contact link IDs differ from geometry ownership')
    supplied_phase = 'settle' if phase in _NO_HOLDER_PHASES else 'release' if phase == 'released_settle' else phase
    rows, bilateral = classify_supported_contacts(copied,
        components_by_geom=components,
        pads_by_geom=pads[active_holder_arm_id] if active_holder_arm_id is not None else {},
        body_pose=body_pose,
        finger_poses=finger_poses_by_arm[active_holder_arm_id] if active_holder_arm_id is not None else {},
        stations_by_geom=stations, phase=supplied_phase)
    for row in rows:
        a, b = row['geom_ids']
        oa, ob = owners[a], owners[b]
        row['owners'] = [copy.deepcopy(oa), copy.deepcopy(ob)]
        kinds = {oa['kind'], ob['kind']}
        valid = 0. <= row['penetration_m'] <= MAX_PENETRATION_M and row['normal_unit_error'] <= .001
        if row['category'] == 'UNALLOWED_CONTACT':
            if contents_contact_witness_valid(row['penetration_m'], row['normal_unit_error'], (oa, ob)):
                row['category'] = 'contents_contact'
            elif valid and kinds == {'support', 'floor'}:
                row['category'] = 'named_support_floor'
    if len(rows) != len(copied['geom_a']):
        raise ContractError('Every retained global witness must produce one classified row')
    return rows, {arm_id: bool(bilateral and arm_id == active_holder_arm_id) for arm_id in sorted(arm_ids)}


class BoundEpisodeContactClassifier:
    """Teacher-owned static policy, with every dynamic witness checked afresh.

    Registries are deeply detached and validated once. There is no caller facts
    argument, mutable registry exposure, native-state cache or tolerance option.
    Source/built-model authenticity still belongs to the trusted live runtime.
    """
    __slots__ = ('__policy', '__binding_sha256', '__calls', '__single_active_contract_sha256')

    def __init__(self, *, owners, components_by_geom, pads_by_arm, stations_by_geom, single_active_contract=None):
        try:
            owned = copy.deepcopy((owners, components_by_geom, pads_by_arm, stations_by_geom))
            owners, components, pads, stations, arm_ids = _validate_static(*owned, single_active_contract=single_active_contract)
            self.__policy = owners, components, pads, stations, frozenset(arm_ids)
            binding = dict(owners=owners, components_by_geom=components,
                pads_by_arm=pads, stations_by_geom=stations, arm_ids=sorted(arm_ids))
            self.__single_active_contract_sha256 = None
            if single_active_contract is not None:
                binding['single_active_contract_sha256'] = single_active_contract['contract_sha256']
                self.__single_active_contract_sha256 = single_active_contract['contract_sha256']
            self.__binding_sha256 = digest(binding)
            self.__calls = 0
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError('Malformed episode contact inputs: '+str(exc)) from exc

    def classify(self, raw, *, body_pose, finger_poses_by_arm, phase, active_holder_arm_id=None):
        try:
            if phase not in _PHASES:
                raise ContractError('Unknown physical contact phase')
            _validate_dynamic(body_pose, finger_poses_by_arm, self.__policy[4])
            self.__calls += 1
            return _classify_checked(raw, policy=self.__policy, body_pose=body_pose,
                finger_poses_by_arm=finger_poses_by_arm, phase=phase,
                active_holder_arm_id=active_holder_arm_id)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError('Malformed episode contact inputs: '+str(exc)) from exc

    def metadata(self):
        owners, components, pads, stations, arm_ids = self.__policy
        result = dict(schema_version='bound-episode-contact-static-policy-v1',
            static_binding_sha256=self.__binding_sha256, static_validation_passes=1,
            arm_ids=sorted(arm_ids), geometry_count=len(owners), tray_component_count=len(components),
            support_component_count=len(stations), pad_counts={name:len(pads[name]) for name in sorted(arm_ids)},
            classification_calls=self.__calls, static_registries_deeply_owned=True,
            dynamic_pose_raw_link_depth_normal_phase_permission_checks='fresh_each_call',
            scope='Static input consistency once. Built-model/source authenticity remains the runtime responsibility. No native contact, clock, holder or success is cached.')
        if self.__single_active_contract_sha256 is not None:
            result.update(schema_version='bound-single-active-episode-contact-static-policy-v1',
                          single_active_contract_sha256=self.__single_active_contract_sha256)
        return result


def classify_episode_live_contacts(raw, *, owners, components_by_geom, pads_by_arm,
        body_pose, finger_poses_by_arm, stations_by_geom, phase, active_holder_arm_id=None,
        single_active_contract=None):
    """Return (all_rows, bilateral_by_arm) using the unchanged narrow predicates.

    owners[global_geom_id] contains kind/entity/link/link_id. pads_by_arm is
    {arm_id: {global_geom_id: matched_pad_record}}. finger_poses_by_arm is
    {arm_id: {left_finger: world_pose, right_finger: world_pose}}. Components
    and supports are the caller's validated built-geometry maps. Poses require
    position_m and quaternion_wxyz, in world coordinates.

    approach/retreat/settle require active_holder_arm_id=None. close/hold/carry/
    place/release/released_settle may explicitly grant one named arm pad/handle
    contact permission. None grants no arm permission. A named arm is not an
    ownership, load-bearing or handoff verdict. After release, delivery still
    requires actual open fingers and no handle contact in the separate assessor.

    No cross-arm contact or other-arm/tray contact is exempted. Existing exact
    bottom/support and known contents/support-floor categories remain. Depth,
    normal and geometric guards are reused without a caller tolerance override.
    Unknown/malformed ownership raises ContractError instead of dropping rows.
    """
    try:
        if phase not in _PHASES:
            raise ContractError('Unknown physical contact phase')
        policy = _validate(
            owners, components_by_geom, pads_by_arm, body_pose, finger_poses_by_arm, stations_by_geom,
            single_active_contract=single_active_contract)
        return _classify_checked(raw, policy=policy, body_pose=body_pose,
            finger_poses_by_arm=finger_poses_by_arm, phase=phase,
            active_holder_arm_id=active_holder_arm_id)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError('Malformed episode contact inputs: '+str(exc)) from exc
