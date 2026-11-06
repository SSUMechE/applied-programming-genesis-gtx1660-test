"""Supplied QA-only fixed Panda peers, retaining collision geometry.

This module imports no simulator. Joint transforms are baked into the original
body hierarchy. Native qualification requires a separately captured original
dynamic-arm ledger at the same INITIAL coordinates and base pose.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


SCHEMA = 'single-active-arm-fixed-peers-v1'
JOINT_NAMES = tuple([f'joint{i}' for i in range(1, 8)] + ['finger_joint1', 'finger_joint2'])
# Geometry representation comparison only. Transport criteria are untouched.
POSE_REPRESENTATION_ATOL = 1e-6


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _vector(text, size, default):
    value = np.asarray(default if text is None else [float(x) for x in text.split()], dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError('Invalid finite asset vector')
    return value


def _rotation(quat):
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError('Invalid body quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def _format(value):
    return ' '.join(format(float(x), '.17g') for x in value)


def _defaults(root):
    classes = {'': {}}
    def visit(node, inherited):
        current = copy.deepcopy(inherited)
        for child in node:
            if child.tag != 'default':
                current.setdefault(child.tag, {}).update(child.attrib)
        name = node.get('class', '')
        if name in classes and name:
            raise ValueError('Duplicate MJCF default class')
        classes[name] = current
        for child in node.findall('default'):
            visit(child, current)
    for node in root.findall('default'):
        visit(node, {})
    return classes


def prepare_fixed_peer_asset(source_xml, q9_native, *, base_position=(0., 0., 0.),
                             base_quaternion=(1., 0., 0., 0.)):
    """Return XML bytes and owned descriptor, without writing or importing SDK.

    Coordinates must already be the operation's native INITIAL coordinates.
    Hinge anchors and slide translations are both preserved. Unsupported MJCF
    features are rejected rather than approximated.
    """
    source = Path(source_xml).resolve(strict=True)
    raw = source.read_bytes()
    root = ET.fromstring(raw)
    if root.tag != 'mujoco' or root.findall('.//include'):
        raise ValueError('Expected self-contained MJCF')
    compiler = root.find('compiler')
    if compiler is None or compiler.get('angle', 'degree') != 'radian':
        raise ValueError('Only explicit radian MJCF is supported')
    if compiler.get('coordinate', 'local') != 'local':
        raise ValueError('Only local MJCF coordinates are supported')
    q = np.asarray(q9_native)
    if q.shape != (9,) or q.dtype not in (np.dtype('float32'), np.dtype('float64')) or not np.isfinite(q).all():
        raise ValueError('Expected actual finite native nine-coordinate array')
    positions = dict(zip(JOINT_NAMES, map(float, q)))
    defaults = _defaults(root)
    world = root.find('worldbody')
    if world is None:
        raise ValueError('Missing worldbody')
    bodies = list(world.iter('body'))
    names = [body.get('name') for body in bodies]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError('Bodies require unique stable names')
    before_geoms = [(body.get('name'), [dict(g.attrib) for g in body.findall('geom')]) for body in bodies]
    before_contact = ET.tostring(root.find('contact')) if root.find('contact') is not None else None
    seen = []
    link_poses = {}
    joint_records = []
    base_p = _vector(None, 3, base_position)
    base_q = _vector(None, 4, base_quaternion)
    if np.linalg.norm(base_q) == 0:
        raise ValueError('Base quaternion is zero')
    base_q /= np.linalg.norm(base_q)
    def bake(body, inherited_class, parent_p, parent_q):
        if any(key in body.attrib for key in ('euler', 'axisangle', 'xyaxes', 'zaxis')):
            raise ValueError('Only explicit quaternion body orientation is supported')
        p = _vector(body.get('pos'), 3, (0., 0., 0.))
        bq = _vector(body.get('quat'), 4, (1., 0., 0., 0.))
        if np.linalg.norm(bq) == 0:
            raise ValueError('Body quaternion is zero')
        bq /= np.linalg.norm(bq)
        child_class = body.get('childclass', inherited_class)
        if child_class not in defaults:
            raise ValueError('Unknown body default class')
        joints = body.findall('joint')
        if len(joints) > 1 or body.findall('freejoint'):
            raise ValueError('Only one hinge or slide joint per body is supported')
        if joints:
            joint = joints[0]
            cls = joint.get('class', child_class)
            if cls not in defaults:
                raise ValueError('Unknown joint default class')
            attrs = {**defaults[cls].get('joint', {}), **joint.attrib}
            name = attrs.get('name')
            if name not in positions or name in seen:
                raise ValueError('Unexpected or duplicate Panda joint')
            seen.append(name)
            amount = positions[name] - float(attrs.get('ref', '0'))
            axis = _vector(attrs.get('axis'), 3, (0., 0., 1.))
            if np.linalg.norm(axis) == 0:
                raise ValueError('Joint axis is zero')
            axis /= np.linalg.norm(axis)
            anchor = _vector(attrs.get('pos'), 3, (0., 0., 0.))
            kind = attrs.get('type', 'hinge')
            if 'range' in attrs:
                low, high = _vector(attrs['range'], 2, ())
                if low > high or not low <= positions[name] <= high:
                    raise ValueError('Frozen joint exceeds declared native limit')
            r = _rotation(bq)
            if kind == 'hinge':
                jq = np.r_[math.cos(amount / 2), axis * math.sin(amount / 2)]
                p = p + r @ (anchor - _rotation(jq) @ anchor)
                bq = _multiply(bq, jq)
            elif kind == 'slide':
                p = p + r @ (axis * amount)
            else:
                raise ValueError('Unsupported fixed-peer joint type')
            joint_records.append({'name': name, 'type': kind, 'axis': axis.tolist(),
                                  'anchor': anchor.tolist(), 'native_q': positions[name],
                                  'reference_q': float(attrs.get('ref', '0'))})
            body.remove(joint)
        body.set('pos', _format(p))
        body.set('quat', _format(bq))
        wp = parent_p + _rotation(parent_q) @ p
        wq = _multiply(parent_q, bq)
        link_poses[body.get('name')] = {'position': wp.tolist(), 'quaternion_wxyz': wq.tolist()}
        for child in body.findall('body'):
            bake(child, child_class, wp, wq)
    for body in world.findall('body'):
        bake(body, '', base_p, base_q)
    if set(seen) != set(JOINT_NAMES):
        raise ValueError('Expected all seven arm and both finger joints')
    removed = {}
    for tag in ('actuator', 'tendon', 'equality', 'keyframe'):
        removed[tag] = [ET.tostring(node, encoding='unicode') for node in root.findall(tag)]
        for node in root.findall(tag):
            root.remove(node)
    if root.find('sensor') is not None:
        raise ValueError('Unreviewed sensor references in fixed asset')
    meshdir = (source.parent / compiler.get('meshdir', '')).resolve()
    asset_sources = {}
    for asset in root.findall('./asset/*'):
        if asset.get('file'):
            if asset.tag != 'mesh':
                raise ValueError('Unreviewed external asset type')
            asset_file = (meshdir / asset.get('file')).resolve(strict=True)
            asset_sources[str(asset_file)] = hashlib.sha256(asset_file.read_bytes()).hexdigest()
            asset.set('file', str(asset_file))
    compiler.set('meshdir', str(meshdir))
    after_geoms = [(body.get('name'), [dict(g.attrib) for g in body.findall('geom')]) for body in bodies]
    if before_geoms != after_geoms or before_contact != (ET.tostring(root.find('contact')) if root.find('contact') is not None else None):
        raise ValueError('Collision declarations changed during baking')
    xml_bytes = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    descriptor = {'schema_version': 'fixed-peer-panda-asset-v1',
                  'source_xml': str(source), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                  'fixed_asset_sha256': hashlib.sha256(xml_bytes).hexdigest(),
                  'native_initial_q': q.tolist(), 'native_dtype': str(q.dtype),
                  'base_position': base_p.tolist(), 'base_quaternion_wxyz': base_q.tolist(),
                  'body_names': names, 'joint_transforms': joint_records,
                  'expected_world_link_poses': link_poses, 'asset_sources': asset_sources,
                  'geom_declarations_sha256': digest(before_geoms),
                  'removed_joint_dependent_sections': removed,
                  'collision_declarations_unchanged': True, 'native_geometry_verified': False,
                  'required_morph_options': {'requires_jac_and_IK': False,
                     'decompose_object_error_threshold': 'inf', 'decompose_robot_error_threshold': 'inf'}}
    return xml_bytes, descriptor


def write_fixed_peer_asset(source_xml, q9_native, output_path, **kwargs):
    xml_bytes, descriptor = prepare_fixed_peer_asset(source_xml, q9_native, **kwargs)
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(xml_bytes)
    descriptor['path'] = str(path)
    return descriptor


def _host(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite native geometry')
    return result


def _array_record(value):
    a = np.ascontiguousarray(_host(value))
    return {'dtype': str(a.dtype), 'shape': list(a.shape),
            'sha256': hashlib.sha256(a.tobytes()).hexdigest()}


def capture_native_arm_geometry(arm):
    """Boundary-only read. Vertex getters refresh derived geometry, never step."""
    links = []
    for link in arm.links:
        geoms = []
        for ordinal, geom in enumerate(link.geoms):
            geoms.append({'ordinal': ordinal, 'type': int(geom.type),
                          'contype': int(geom.contype), 'conaffinity': int(geom.conaffinity),
                          'friction': float(geom.friction),
                          'friction_torsional': float(geom.friction_torsional),
                          'friction_rolling': float(geom.friction_rolling),
                          'sol_params': _host(geom.sol_params).tolist(),
                          'data': _host(geom.data).tolist(),
                          'init_pos': _host(geom.init_pos).tolist(),
                          'init_quat': _host(geom.init_quat).tolist(),
                          'verts': _array_record(geom.init_verts),
                          'faces': _array_record(geom.init_faces),
                          'world_position': _host(geom.get_pos(relative=False)).reshape(3).tolist(),
                          'world_quaternion_wxyz': _host(geom.get_quat(relative=False)).reshape(4).tolist(),
                          'world_vertices': _host(geom.get_verts()).reshape(-1, 3).tolist()})
        links.append({'name': link.name, 'is_fixed': bool(link.is_fixed),
                      'native_link_id': int(link.idx),
                      'world_position': _host(link.get_pos(relative=False)).reshape(3).tolist(),
                      'world_quaternion_wxyz': _host(link.get_quat(relative=False)).reshape(4).tolist(),
                      'geoms': geoms})
    result = {'schema_version': 'native-arm-geometry-reference-v1', 'entity_name': arm.name,
              'entity_id': int(arm.idx), 'n_dofs': int(arm.n_dofs), 'n_qs': int(arm.n_qs), 'links': links}
    if arm.n_dofs:
        result['native_q'] = _host(arm.get_qpos()).reshape(9).tolist()
        result['native_qd'] = _host(arm.get_dofs_velocity()).reshape(9).tolist()
    result['record_sha256'] = digest(result)
    return result


def qualify_native_fixed_peers(scene, active_arms, fixed_peers, descriptors, *,
                               active_arm_id, reference_native_ledgers):
    """Qualify built peers against real unmodified dynamic-arm INITIAL geometry.

    No new transport acceptance tolerance is introduced. The separate 1 um
    representation bound accommodates baking double FK into native F32 fields.
    Geometry data, masks, materials, local meshes and ordering must match exactly.
    """
    if set(active_arms) != {active_arm_id} or set(fixed_peers) != set(descriptors) or len(fixed_peers) != 2:
        raise ValueError('Expected one active and two fixed original arms')
    if active_arms[active_arm_id].n_dofs != 9:
        raise ValueError('Active arm must retain nine actual dynamic DOFs')
    if getattr(scene, '_t', None) != 0:
        raise ValueError('Fixed-peer qualification requires native INITIAL clock zero')
    peers = {}
    for name, peer in fixed_peers.items():
        descriptor = descriptors[name]
        reference = reference_native_ledgers[name]
        if reference.get('record_sha256') != digest({k: v for k, v in reference.items() if k != 'record_sha256'}):
            raise ValueError('Native original-arm geometry reference changed')
        if reference['entity_name'] != name or reference['n_dofs'] != 9:
            raise ValueError('Reference must be original same-named dynamic arm')
        if reference.get('native_q') != descriptor['native_initial_q'] or any(reference.get('native_qd', [1])):
            raise ValueError('Original-arm reference is not the exact frozen INITIAL coordinates/rest')
        if int(peer.n_dofs) != 0 or int(peer.n_qs) != 0:
            raise ValueError('Frozen peers must have zero native DOFs and coordinates')
        if hashlib.sha256(Path(descriptor['path']).read_bytes()).hexdigest() != descriptor['fixed_asset_sha256']:
            raise ValueError('Fixed asset changed')
        actual = capture_native_arm_geometry(peer)
        current_links = {r['name']: r for r in actual['links']}
        original_links = {r['name']: r for r in reference['links']}
        if set(current_links) != set(original_links):
            raise ValueError('Fixed-peer native link inventory changed')
        max_position = max_rotation = 0.
        for link_name, row in current_links.items():
            old = original_links[link_name]
            world_keys = {'world_position', 'world_quaternion_wxyz', 'world_vertices'}
            local_geoms = [{k: v for k, v in geom.items() if k not in world_keys} for geom in row['geoms']]
            original_geoms = [{k: v for k, v in geom.items() if k not in world_keys} for geom in old['geoms']]
            if not row['is_fixed'] or local_geoms != original_geoms:
                raise ValueError('Fixed-peer native collision geometry/material/mask changed: ' + link_name)
            for current_geom, old_geom in zip(row['geoms'], old['geoms']):
                cv, ov = np.asarray(current_geom['world_vertices']), np.asarray(old_geom['world_vertices'])
                if cv.shape != ov.shape or float(np.max(np.abs(cv - ov), initial=0)) > POSE_REPRESENTATION_ATOL:
                    raise ValueError('Fixed-peer actual world collision vertices differ: ' + link_name)
            expected = descriptor['expected_world_link_poses'].get(link_name)
            # The MJCF world pseudo-link has no geometry and is outside source body names.
            if expected is None and row['geoms']:
                raise ValueError('Unexpected native collision link')
            pairs = [(old['world_position'], old['world_quaternion_wxyz'])]
            if expected is not None:
                pairs.append((expected['position'], expected['quaternion_wxyz']))
            for pos, quat in pairs:
                dp = float(np.max(np.abs(np.asarray(row['world_position']) - pos)))
                dr = float(np.max(np.abs(_rotation(row['world_quaternion_wxyz']) - _rotation(quat))))
                max_position = max(max_position, dp)
                max_rotation = max(max_rotation, dr)
                if dp > POSE_REPRESENTATION_ATOL or dr > POSE_REPRESENTATION_ATOL:
                    raise ValueError('Fixed-peer native geometry pose differs: ' + link_name)
        peers[name] = {'descriptor': descriptor, 'actual': actual,
                       'reference_record_sha256': reference['record_sha256'],
                       'max_world_position_difference_m': max_position,
                       'max_world_rotation_matrix_difference': max_rotation}
    ledger = {'schema_version': SCHEMA, 'active_arm_id': active_arm_id,
              'frozen_arm_ids': sorted(fixed_peers), 'geometry_verified': True,
              'pose_representation_atol': POSE_REPRESENTATION_ATOL,
              'transport_criteria_changed': False, 'fixed_peers': peers,
              'collision_filter_scope': 'Fixed-fixed pairs are omitted by native solver. All dynamic-fixed pairs retain original local masks. Fixed internal joints have no dynamic constraints.'}
    ledger['fixed_peer_ledger_sha256'] = digest(ledger)
    contract = {'schema_version': SCHEMA, 'active_arm_id': active_arm_id,
                'frozen_arm_ids': sorted(fixed_peers), 'geometry_verified': True,
                'fixed_peer_ledger_sha256': ledger['fixed_peer_ledger_sha256'],
                'fixed_peers': {}}
    for name, row in peers.items():
        actual = row['actual']
        contract['fixed_peers'][name] = {
            'actual': {'entity_name': name, 'entity_id': actual['entity_id'],
                       'n_dofs': actual['n_dofs'], 'n_qs': actual['n_qs'],
                       'link_count': len(actual['links']),
                       'geom_count': sum(len(link['geoms']) for link in actual['links']),
                       'native_link_ids': [link['native_link_id'] for link in actual['links']],
                       'all_links_fixed': all(link['is_fixed'] for link in actual['links'])},
            'source_xml_sha256': row['descriptor']['source_sha256'],
            'fixed_asset_sha256': row['descriptor']['fixed_asset_sha256'],
            'source_assets_sha256': digest(row['descriptor']['asset_sources']),
            'native_geometry_record_sha256': actual['record_sha256'],
            'reference_record_sha256': row['reference_record_sha256'],
            'native_initial_q': row['descriptor']['native_initial_q']}
    contract['contract_sha256'] = digest(contract)
    validated_single_active_contract(contract)
    scene._ap_single_active_arm_contract = contract
    scene._ap_fixed_peer_geometry_ledger = ledger
    scene._ap_single_active_arm_identity = (id(scene), id(active_arms[active_arm_id]),
                                         tuple((name, id(fixed_peers[name])) for name in sorted(fixed_peers)))
    return ledger


def validated_single_active_contract(value):
    if not isinstance(value, dict) or value.get('schema_version') != SCHEMA or value.get('geometry_verified') is not True:
        raise ValueError('Missing native fixed-peer geometry qualification')
    active = value.get('active_arm_id')
    frozen = value.get('frozen_arm_ids')
    if active not in ('arm_0', 'arm_1', 'arm_2') or frozen != sorted(set(('arm_0', 'arm_1', 'arm_2')) - {active}):
        raise ValueError('Wrong active/frozen original arm identities')
    if value.get('contract_sha256') != digest({k: v for k, v in value.items() if k != 'contract_sha256'}):
        raise ValueError('Fixed-peer geometry qualification changed')
    if not isinstance(value.get('fixed_peer_ledger_sha256'), str) or len(value['fixed_peer_ledger_sha256']) != 64:
        raise ValueError('Missing full native geometry ledger binding')
    if set(value.get('fixed_peers', {})) != set(frozen):
        raise ValueError('Missing qualified peer geometry')
    for name in frozen:
        actual = value['fixed_peers'][name]['actual']
        if (actual['entity_name'] != name or actual['n_dofs'] != 0 or actual['n_qs'] != 0
                or actual['all_links_fixed'] is not True or actual['geom_count'] <= 0
                or actual['link_count'] <= 0 or len(set(actual['native_link_ids'])) != actual['link_count']):
            raise ValueError('Missing native fixed zero-DOF proof')
    return copy.deepcopy(value)


def validate_scene_contract(scene, arms):
    contract = validated_single_active_contract(getattr(scene, '_ap_single_active_arm_contract', None))
    if set(arms) != {contract['active_arm_id']} or arms[contract['active_arm_id']].n_dofs != 9:
        raise ValueError('Qualified active-arm identity changed')
    identity = getattr(scene, '_ap_single_active_arm_identity', None)
    if identity is None or identity[:2] != (id(scene), id(arms[contract['active_arm_id']])):
        raise ValueError('Qualified scene/active object changed')
    ledger = getattr(scene, '_ap_fixed_peer_geometry_ledger', None)
    if (not isinstance(ledger, dict) or ledger.get('fixed_peer_ledger_sha256') != contract['fixed_peer_ledger_sha256']
            or ledger['fixed_peer_ledger_sha256'] != digest({k: v for k, v in ledger.items() if k != 'fixed_peer_ledger_sha256'})):
        raise ValueError('Full native fixed-peer geometry ledger changed')
    by_name = {entity.name: entity for entity in scene.rigid_solver.entities}
    for name, original_id in identity[2]:
        if name not in by_name or id(by_name[name]) != original_id or by_name[name].n_dofs != 0:
            raise ValueError('Qualified fixed peer changed')
    return copy.deepcopy(contract)
