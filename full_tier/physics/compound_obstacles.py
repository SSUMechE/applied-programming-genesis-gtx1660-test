"""Exact multi-box rigid obstacle assets and native geometry validation.

Declared assembly membership removes only native same-link self pairs. It does
not exempt environment contacts. Query users supply floor-clipped inflated
component specs and fixed=False. Physical nominal users must supply nominal
specs and fixed=True. This helper neither steps nor changes a built entity.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from physics.joint_scene import array
from transport_challenge.contracts import ContractError, identifier

URDF_NATIVE_MASK = 1
COMPOUND_SOURCE_PINS = {
    'utils/urdf.py': 'b7684e074d813d56ab99f5d8c9ca46e396a37afc72f78589485d94c8363b69f0',
}


def verify_compound_sources(root):
    for name, expected in COMPOUND_SOURCE_PINS.items():
        if hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() != expected:
            raise ContractError('Reviewed compound URDF parser changed: '+name)
    return dict(COMPOUND_SOURCE_PINS)


def validate_compound_obstacle_groups(case):
    """Return detached groups, or () when the legacy optional field is absent."""
    if 'compound_obstacle_groups' not in case:
        return ()
    groups = case['compound_obstacle_groups']
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ContractError('Explicit compound groups must be a nonempty sequence')
    obstacles = case['obstacles']
    ids = [identifier(o['id']) for o in obstacles]
    if len(set(ids)) != len(ids):
        raise ContractError('Obstacle component IDs must be unique')
    result = []
    for group in groups:
        if not isinstance(group, dict) or set(group) != {'id', 'component_ids'}:
            raise ContractError('Compound group requires only id and component_ids')
        gid = identifier(group['id'])
        members = group['component_ids']
        if not isinstance(members, (list, tuple)) or not members:
            raise ContractError('Each rigid assembly requires a nonempty component sequence')
        result.append({'id': gid, 'component_ids': [identifier(v) for v in members]})
    gids = [g['id'] for g in result]
    members = [v for g in result for v in g['component_ids']]
    if len(set(gids)) != len(gids) or set(gids) & set(ids):
        raise ContractError('Assembly IDs must be distinct from each other and component IDs')
    if len(members) != len(ids) or set(members) != set(ids) or len(set(members)) != len(members):
        raise ContractError('Compound groups must partition every obstacle component exactly once')
    return tuple(result)


def validate_embedded_support_components(case, stations):
    """Bind logical support plates to exact nominal boxes in declared assemblies."""
    if 'embedded_support_components' not in case:return {}
    mapping=case['embedded_support_components']
    if not isinstance(mapping,dict) or not mapping:
        raise ContractError('Embedded support mapping must be a nonempty ID mapping')
    groups=validate_compound_obstacle_groups(case)
    if not groups:raise ContractError('Embedded supports require declared rigid assemblies')
    by_station={identifier(s['id']):s for s in stations}
    if len(by_station)!=len(stations):raise ContractError('Support IDs must be distinct')
    by_component={o['id']:o for o in case['obstacles']}
    result={identifier(k):identifier(v) for k,v in mapping.items()}
    if (not set(result)<=set(by_station) or not set(result.values())<=set(by_component)
            or len(set(result.values()))!=len(result)):
        raise ContractError('Each embedded station needs one distinct declared component')
    for name,component in result.items():
        o=by_component[component];s=by_station[name]
        lo=_vector(o['lower_m'],3,'nominal plate lower');hi=_vector(o['upper_m'],3,'nominal plate upper')
        if (np.any(lo>=hi) or not np.array_equal(hi-lo,_vector(s['size_m'],3,'station size'))
                or not np.array_equal((lo+hi)/2,_vector(s['center_world_m'],3,'station center'))):
            raise ContractError('Embedded support must exactly equal its nominal component box')
    return result


def embedded_support_station(native_component, station):
    """Project one already native-validated compound BOX into the support registry.

    The native entity/link remain the assembly. Only this precise component is
    a named support. Its existing native AABB representation bound is retained,
    while the contact classifier keeps its unchanged physical surface guards.
    """
    n=native_component
    expected_checks={'native_box','one_declared_link','exact_component_match','world_orientation_identity','world_aabb_matches','native_masks_match'}
    if (n.get('id')!=n.get('component_id') or n.get('entity')!=n.get('assembly_id')
            or set(n.get('native_representation_checks',{}))!=expected_checks
            or not all(v is True for v in n['native_representation_checks'].values())
            or n.get('clearance_inflation_m')!=0. or n.get('planning_clearance_m',0.)!=0.
            or n.get('size_m')!=station['size_m'] or n.get('center_world_m')!=station['center_world_m']):
        raise ContractError('Embedded support needs the exact validated uninflated native plate')
    observed=n['observed_native_geometry']
    if observed.get('contype')!=URDF_NATIVE_MASK or observed.get('conaffinity')!=URDF_NATIVE_MASK:
        raise ContractError('Embedded plate native masks changed')
    size=_vector(observed['size_m'],3,'native plate size')
    center=_vector(observed['center_world_m'],3,'native plate center')
    aabb=np.asarray(n['actual_world_aabb_m'],dtype=float)
    expected=np.asarray(station['center_world_m'])+np.array([-1.,1.])[:,None]*np.asarray(station['size_m'])/2
    bound=float(8*np.finfo(np.float32).eps*max(1.,float(np.max(np.abs(expected)))))
    if (aabb.shape!=(2,3) or not np.isfinite(aabb).all()
            or n.get('world_aabb_float_roundoff_bound_m')!=bound
            or np.max(np.abs(aabb-expected))>bound
            or not np.allclose(size,station['size_m'],rtol=1e-6,atol=1e-7)
            or np.max(np.abs(center-np.asarray(station['center_world_m'])))>bound):
        raise ContractError('Embedded native plate size/world bounds changed')
    return {**copy.deepcopy(station),'geom_id':int(n['geom_id']),'link_id':int(n['link_id']),
        'entity':n['entity'],'assembly_id':n['assembly_id'],'component_id':n['component_id'],
        'embedded_in_rigid_assembly':True,'observed_native_geometry':copy.deepcopy(observed),
        'native_representation_checks':copy.deepcopy(n['native_representation_checks']),
        'actual_world_aabb_m':aabb.tolist(),'world_aabb_float_roundoff_bound_m':bound,
        'expected_contact_mask':URDF_NATIVE_MASK,'clearance_inflation_m':0.,
        'validated_native_component':copy.deepcopy(n)}


def _vector(value, size, label):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    def has_bool(v):
        return isinstance(v, (bool, np.bool_)) or isinstance(v, (list, tuple)) and any(has_bool(x) for x in v)
    if has_bool(value):
        raise ContractError(label+' cannot contain boolean coordinates')
    a = np.asarray(value)
    if a.shape != (size,) or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ContractError(label+' must contain finite numeric coordinates')
    return a.astype(float, copy=True)


def _text(values):
    return ' '.join(format(float(v), '.17g') for v in values)


def write_compound_obstacle_asset(directory, group, component_specs, *, component_colors=None, hidden_component_ids=None):
    """Write one authored-root URDF link containing each exact BOX once.

    component_specs use id/size_m/center_world_m plus optional evidence fields.
    They are already nominal or planning-inflated by the caller. Unit-density
    additive box inertia is only a valid query inertial placeholder. No dynamic
    movable-obstacle or calibrated union-mass contract is provided.
    """
    if not isinstance(group, dict) or set(group) != {'id', 'component_ids'}:
        raise ContractError('Use validated explicit assembly membership')
    gid = identifier(group['id'])
    if not isinstance(component_specs, (list, tuple)) or not component_specs:
        raise ContractError('Compound asset needs exact component geometry')
    specs = copy.deepcopy(component_specs)
    ids = [identifier(v['id']) for v in specs]
    validate_compound_obstacle_groups({'obstacles': [{'id': i} for i in ids],
                                       'compound_obstacle_groups': [group]})
    if ids != list(group['component_ids']):
        raise ContractError('Asset component order must equal declared membership')
    sizes = [_vector(s['size_m'], 3, 'component size') for s in specs]
    centers = [_vector(s['center_world_m'], 3, 'component center') for s in specs]
    if any(np.any(s <= 0.) for s in sizes):
        raise ContractError('Positive component sizes required')
    colors = component_colors
    if colors is not None and (not isinstance(colors, dict) or set(colors) != set(ids)):
        raise ContractError('Optional visual colors must cover exactly these components')
    hidden=() if hidden_component_ids is None else hidden_component_ids
    if (not isinstance(hidden,(list,tuple)) or any(type(v) is not str for v in hidden)
            or len(set(hidden))!=len(hidden) or not set(hidden)<=set(ids)):
        raise ContractError('Hidden visual IDs must be unique members of this assembly')
    lower = np.min([p-s/2 for p, s in zip(centers, sizes)], axis=0)
    upper = np.max([p+s/2 for p, s in zip(centers, sizes)], axis=0)
    origin = (lower+upper)/2
    masses = np.asarray([np.prod(s) for s in sizes])
    mass = float(masses.sum())
    com = sum(m*p for m, p in zip(masses, centers))/mass-origin
    inertia = np.zeros((3, 3))
    for m, p, s in zip(masses, centers, sizes):
        d = p-origin-com
        inertia += np.diag(m*(np.sum(s*s)-s*s)/12)+m*(np.dot(d, d)*np.eye(3)-np.outer(d, d))
    if not np.isfinite([*origin, mass, *com, *inertia.ravel()]).all() or mass <= 0.:
        raise ContractError('Compound geometry exceeds finite asset representation')
    robot = ET.Element('robot', name=gid)
    link_name = 'assembly'
    link = ET.SubElement(robot, 'link', name=link_name)
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'origin', xyz=_text(com), rpy='0 0 0')
    ET.SubElement(inertial, 'mass', value=format(mass, '.17g'))
    ET.SubElement(inertial, 'inertia', **{name: format(float(inertia[i, j]), '.17g')
        for name, i, j in [('ixx',0,0),('ixy',0,1),('ixz',0,2),('iyy',1,1),('iyz',1,2),('izz',2,2)]})
    components = []
    for spec, size, center in zip(specs, sizes, centers):
        local = center-origin
        for kind in ('collision', 'visual'):
            if kind=='visual' and spec['id'] in hidden:continue
            part = ET.SubElement(link, kind, name=spec['id'])
            ET.SubElement(part, 'origin', xyz=_text(local), rpy='0 0 0')
            ET.SubElement(ET.SubElement(part, 'geometry'), 'box', size=_text(size))
            if kind == 'visual' and colors is not None:
                value = colors[spec['id']]
                if type(value) is not str or len(value) != 7 or not value.startswith('#'):
                    raise ContractError('Visual colors use RGB hex')
                try: rgb = [int(value[i:i+2],16)/255 for i in (1,3,5)]
                except ValueError as exc: raise ContractError('Visual colors use RGB hex') from exc
                ET.SubElement(ET.SubElement(part, 'material', name=spec['id']+'_material'), 'color', rgba=_text([*rgb,1.]))
        components.append({**spec, 'component_id': spec['id'], 'assembly_id': gid,
                           'center_link_m': local.tolist()})
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    # An external display ID must never become a filesystem traversal.
    path = directory/('compound_'+hashlib.sha256(gid.encode('utf-8')).hexdigest()[:20]+'.urdf')
    data = ET.tostring(robot, encoding='utf-8', xml_declaration=True)+b'\n'
    if path.exists() and path.read_bytes() != data:
        raise ContractError('Refusing to overwrite a different compound asset')
    path.write_bytes(data)
    record={'id': gid, 'path': str(path.resolve()), 'asset_sha256': hashlib.sha256(data).hexdigest(),
            'link_name': link_name, 'center_world_m': origin.tolist(), 'components': components,
            'expected_native_contact_mask': URDF_NATIVE_MASK,
            'representation': 'Exact BOX union on one authored native rigid link',
            'inertial_scope': 'Unit-density additive-box placeholder. Query-only movable or world-fixed nominal use.'}
    if hidden_component_ids is not None:record['hidden_visual_component_ids']=list(hidden)
    return record


def validate_built_compound(entity, asset, box_type, *, fixed=False):
    """Match native parts independently of parser geom ordering and names."""
    if type(fixed) is not bool:
        raise ContractError('Explicit fixed/query mode required')
    if hashlib.sha256(Path(asset['path']).read_bytes()).hexdigest() != asset['asset_sha256']:
        raise ContractError('Compound asset bytes changed')
    parts = asset['components']
    if (entity.name != asset['id'] or entity.n_links != 1 or len(entity.geoms) != len(parts)
            or len({g.link.idx for g in entity.geoms}) != 1):
        raise ContractError('Compound must retain every exact component on one named native entity/link')
    position = _vector(entity.get_pos(relative=False),3,'assembly position')
    quat = _vector(entity.get_quat(relative=False),4,'assembly orientation')
    if not np.allclose(position, asset['center_world_m'], rtol=0., atol=1e-7) or min(
            np.linalg.norm(quat-[1.,0.,0.,0.]),np.linalg.norm(quat+[1.,0.,0.,0.])) > 1e-7:
        raise ContractError('Compound authored root pose changed')
    remaining = set(range(len(parts))); result=[]; seen=set()
    for g in entity.geoms:
        if (g.entity is not entity or g.link.name != asset['link_name'] or bool(g.link.is_fixed) != fixed
                or g.type != box_type or g.contype != URDF_NATIVE_MASK or g.conaffinity != URDF_NATIVE_MASK
                or g.idx in seen):
            raise ContractError('Compound native type, owner, fixed state or masks changed')
        seen.add(g.idx)
        packed = _vector(g.data,7,'native BOX data')
        center = _vector(g.init_pos,3,'native BOX origin')
        local_q = _vector(g.init_quat,4,'native BOX orientation')
        if np.any(packed[3:] != 0.) or min(np.linalg.norm(local_q-[1.,0.,0.,0.]),np.linalg.norm(local_q+[1.,0.,0.,0.])) > 1e-7:
            raise ContractError('Compound BOX data or local orientation changed')
        matches = [i for i in remaining if np.allclose(packed[:3],parts[i]['size_m'],rtol=1e-6,atol=1e-7)
                   and np.allclose(center,parts[i]['center_link_m'],rtol=0.,atol=1e-7)]
        if len(matches) != 1:
            raise ContractError('Compound component missing, duplicated, reframed or ambiguous')
        i = matches[0]; remaining.remove(i); spec=parts[i]
        actual = array(g.get_AABB())
        expected = np.asarray(spec['center_world_m'])+np.array([-1.,1.])[:,None]*np.asarray(spec['size_m'])/2
        bound = float(8*np.finfo(np.float32).eps*max(1.,float(np.max(np.abs(expected)))))
        if actual.shape != (2,3) or not np.isfinite(actual).all() or np.max(np.abs(actual-expected)) > bound:
            raise ContractError('Compound native world BOX bounds changed')
        world_center=position+center
        checks={'native_box':True,'one_declared_link':True,'exact_component_match':True,
                'world_orientation_identity':True,'world_aabb_matches':True,'native_masks_match':True}
        result.append({**copy.deepcopy(spec),'entity':entity.name,'geom_id':int(g.idx),'link_id':int(g.link.idx),
            'observed_native_geometry':{'size_m':packed[:3].tolist(),'center_world_m':world_center.tolist(),
                'local_center_m':center.tolist(),'contype':int(g.contype),'conaffinity':int(g.conaffinity)},
            'native_representation_checks':checks,'actual_world_aabb_m':actual.tolist(),
            'world_aabb_float_roundoff_bound_m':bound,'clearance_inflation_m':spec.get('planning_clearance_m',0.)})
    if remaining:
        raise ContractError('Compound components incomplete')
    return result
