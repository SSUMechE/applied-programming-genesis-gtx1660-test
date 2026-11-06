"""Explicit instructor contact-accuracy experiment. No stepping or pose changes.

Only thirteen named geometry time constants change. This is an uncalibrated
joint change of contact response and retained-contact visibility, not proof of
continuous safety, physical convergence or a new student grading policy.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import struct

import numpy as np

from transport_challenge.contracts import ContractError, digest

TARGET_ENTITIES = ('tray_0', *(f'tray_0_ball_{i}' for i in range(4)))
SOURCE_PINS = {
    'engine/entities/rigid_entity/rigid_geom.py': '399e0e0ec7d63a9a965c781f6946aed67109051ac8ccd666a47092cfb6a10a14',
    'engine/solvers/rigid/rigid_solver.py': 'e076854af17262323041544ff692d366402a700357053618c8d89fed1e86b19f',
    'engine/solvers/rigid/collider/collider.py': '496e2f56acdfb7bd20822c993ba12d275824335759b49cbc965347a64c833cc0',
    'engine/solvers/rigid/collider/contact.py': 'cd11feed70d9badfa9e56327e71cd49bcf9fe280bd8ed774b890ef6a1226023c',
    'options/solvers.py': '2f799beaf1a5acb7aa55edcc063187f3e6a384ce1ea85092c2ee72f4e202e536',
}


def contact_accuracy_request():
    return {'schema_version': 'contact-accuracy-request-v1', 'dt_s': .0025,
            'geometry_timeconst_s': .005, 'contact_pruning_tolerance': None,
            'target_entities': list(TARGET_ENTITIES)}


def validate_request(request):
    if not isinstance(request, dict) or digest(request) != digest(contact_accuracy_request()):
        raise ContractError('Only the explicit thirteen-geometry .005s/unpruned experiment is supported')
    return copy.deepcopy(request)


def rigid_options(request):
    if request is None:
        return {}
    validate_request(request)
    return {'contact_pruning_tolerance': None}


def verify_sources(installed):
    for relative, expected in SOURCE_PINS.items():
        if hashlib.sha256((Path(installed)/relative).read_bytes()).hexdigest() != expected:
            raise ContractError('Contact accuracy installed source changed: '+relative)
    return dict(SOURCE_PINS)


def _vector(value):
    raw = value.detach().cpu().numpy() if hasattr(value, 'detach') else value
    if isinstance(raw, (tuple, list)) and any(isinstance(v, (bool, np.bool_)) for v in raw):
        raise ContractError('Constraint parameters cannot contain booleans')
    a = np.asarray(raw)
    if a.shape != (7,) or a.dtype.kind != 'f' or not np.isfinite(a).all():
        raise ContractError('Need a copied finite floating seven-parameter vector')
    return a.copy()


def _inventory(entities):
    rows = []
    if len({e.name for e in entities}) != len(entities):
        raise ContractError('Geometry entities must have unique names')
    for entity in entities:
        for local, geom in enumerate(entity.geoms):
            rows.append((geom, {'global_id': int(geom.idx), 'entity': entity.name,
                'identity': {'local_index': local, 'link': geom.link.name, 'type': int(geom.type)}}))
    if len({r['global_id'] for _, r in rows}) != len(rows):
        raise ContractError('Geometry IDs must be unique')
    return rows


def prepare_geometry_parameters(entities, request, *, box_type, sphere_type):
    """Call before build, before the existing finger-equality preparation."""
    if request is None:
        return None
    request = validate_request(request)
    if any(e.is_built for e in entities):
        raise ContractError('Contact accuracy requires unbuilt geometry')
    inventory = _inventory(entities)
    selected = [(g, r) for g, r in inventory if r['entity'] in TARGET_ENTITIES]
    if ([r['identity']['local_index'] for _, r in selected if r['entity'] == 'tray_0'] != list(range(9))
            or any(r['identity']['link'] != 'payload' for _, r in selected if r['entity'] == 'tray_0')
            or any(g.type != box_type for g, r in selected if r['entity'] == 'tray_0')
            or any(len([r for _, r in selected if r['entity'] == name]) != 1 for name in TARGET_ENTITIES[1:])
            or any(g.type != sphere_type for g, r in selected if r['entity'] != 'tray_0') or len(selected) != 13):
        raise ContractError('Require exactly nine named tray boxes and four separately owned spheres')
    before = [(g, r, _vector(g.sol_params)) for g, r in inventory]
    rows = []
    for geom, identity, original in before:
        target = identity['entity'] in TARGET_ENTITIES
        expected = original.copy()
        if target:
            expected[0] = .005
            geom.set_sol_params(expected.copy())
        actual = _vector(geom.sol_params)
        if not np.array_equal(actual, expected):
            raise ContractError('Geometry setter changed an unintended parameter')
        rows.append({**identity, 'selected': target, 'before_prebuild': original.tolist(),
                     'after_prebuild': actual.tolist()})
    # Detect setters that accidentally changed a different geometry by aliasing.
    for geom, identity in inventory:
        row = next(r for r in rows if r['global_id'] == identity['global_id'])
        if _vector(geom.sol_params).tolist() != row['after_prebuild']:
            raise ContractError('Geometry parameter storage aliases another geometry')
    return {'requested': request, 'geometries': rows}


def built_evidence(scene, entities, before, request):
    if request is None:
        if before is not None:
            raise ContractError('Unexpected contact-accuracy preparation')
        return None
    request = validate_request(request)
    if not isinstance(before, dict) or before.get('requested') != request:
        raise ContractError('Prebuild contact-accuracy request changed')
    solver = scene.rigid_solver
    pruning = float(solver.collider._collider_info.contact_pruning_tolerance[None])
    if (float(scene.sim.dt) != .0025 or int(scene.sim.substeps) != 1
            or solver._options.contact_pruning_tolerance is not None or pruning != 0.
            or solver._use_hibernation or solver._requires_grad):
        raise ContractError('Actual contact-accuracy timestep/pruning/branch changed')
    old = {r['global_id']: r for r in before['geometries']}
    rows = []
    for geom, identity in _inventory(entities):
        r = old.pop(identity['global_id'])
        if any(r[k] != v for k, v in identity.items()):
            raise ContractError('Native geometry ownership changed during build')
        native = _vector(geom.sol_params)
        rows.append({**r, 'native_sol_params': native.tolist(), 'dtype': str(native.dtype)})
    if old:
        raise ContractError('Missing native geometry evidence')
    evidence = {'schema_version': 'contact-accuracy-native-v1', 'requested': request,
        'dt_s': float(scene.sim.dt), 'substeps': int(scene.sim.substeps),
        'resolved_contact_pruning_tolerance': solver._options.contact_pruning_tolerance,
        'native_contact_pruning_tolerance': pruning, 'native_geometry_count': int(solver.n_geoms),
        'geometries': rows, 'source_sha256': dict(SOURCE_PINS),
        'scope': 'Measured configured values only. Coupled response/visibility change, not a calibrated one-factor experiment.'}
    validate_evidence(evidence, request)
    return evidence


def validate_evidence(evidence, request):
    request = validate_request(request)
    if (not isinstance(evidence, dict) or evidence.get('schema_version') != 'contact-accuracy-native-v1'
            or evidence.get('requested') != request or evidence.get('dt_s') != .0025
            or evidence.get('substeps') != 1 or evidence.get('resolved_contact_pruning_tolerance', 'missing') is not None
            or type(evidence.get('native_contact_pruning_tolerance')) not in (int, float)
            or evidence['native_contact_pruning_tolerance'] != 0. or evidence.get('source_sha256') != SOURCE_PINS):
        raise ContractError('Malformed configured contact-accuracy evidence')
    rows = evidence['geometries']; count = evidence['native_geometry_count']
    if (type(count) is not int or count <= 0 or len(rows) != count
            or any(type(r['global_id']) is not int for r in rows)
            or sorted(r['global_id'] for r in rows) != list(range(count))):
        raise ContractError('Contact-accuracy geometry inventory is incomplete')
    selected = []
    signatures = []
    for r in rows:
        if type(r['selected']) is not bool or r['selected'] != (r['entity'] in TARGET_ENTITIES):
            raise ContractError('Wrong contact-accuracy selection')
        if not isinstance(r['entity'], str) or not r['entity'] or r['dtype'] not in ('float32', 'float64'):
            raise ContractError('Missing geometry identity or native dtype')
        before, after, native = (_vector(r[k]) for k in ('before_prebuild', 'after_prebuild', 'native_sol_params'))
        expected = before.copy()
        if r['selected']:
            expected[0] = .005
            selected.append(r)
        if not np.array_equal(after, expected):
            raise ContractError('A parameter other than selected timeconstant changed')
        cast = lambda x: struct.unpack('f', struct.pack('f', float(x)))[0] if r['dtype'] == 'float32' else float(x)
        # The pinned sanitizer resolves unspecified time constants to .01.
        # The authored vectors here already satisfy its other six bounds.
        if (not 0 < after[1] or not .0001 <= after[2] <= .9999 or not .0001 <= after[3] <= .9999
                or after[4] < 0 or not .0001 <= after[5] <= .9999 or after[6] < 1
                or np.finfo(np.float64).eps < after[0] < .005):
            raise ContractError('Unsupported prebuild constraint sanitization')
        expected_native = [cast(.01 if after[0] <= np.finfo(np.float64).eps else max(.005, after[0])), *map(cast, after[1:])]
        if native.tolist() != expected_native:
            raise ContractError('Native geometry parameters differ from pinned exact dtype resolution')
        signatures.append({k: copy.deepcopy(v) for k, v in r.items() if k != 'global_id'})
    if (len(selected) != 13
            or sorted(r['identity']['local_index'] for r in selected if r['entity'] == 'tray_0') != list(range(9))
            or any(r['identity']['link'] != 'payload' for r in selected if r['entity'] == 'tray_0')
            or any(r['identity']['type'] != 5 for r in selected if r['entity'] == 'tray_0')
            or any(len([r for r in selected if r['entity'] == name]) != 1 for name in TARGET_ENTITIES[1:])
            or any(r['identity']['type'] != 1 for r in selected if r['entity'] != 'tray_0')):
        raise ContractError('Native target primitive identities are incomplete')
    return sorted(signatures, key=lambda r: (r['entity'], digest(r)))


# Separate opt-in model. The original13-geometry functions above stay unchanged.
RIGID16_PROFILE = 'rigid16_bead_tc2h_v1'
RIGID16_SOURCE_PINS = {**SOURCE_PINS,
    '__init__.py': 'e70073958649ab52a99205b93cca332b1f058d1adc6881aaaab5fe0776e75441'}


def rigid16_contact_request():
    return {'schema_version':'rigid16-bead-contact-request-v1','profile_name':RIGID16_PROFILE,
            'dt_s':.0025,'substeps':16,'substeps_local':1,'floor_and_sphere_timeconst_s':.0003125,
            'tray_timeconst_s':.005,'contact_pruning_tolerance':None}


def validate_rigid16_request(request):
    if type(request) is not dict or digest(request)!=digest(rigid16_contact_request()):
        raise ContractError('Only the explicit rigid16 bead/floor2h profile is supported')
    return copy.deepcopy(request)


def _rigid16_expected_native(value, dtype, min_timeconst):
    """Replay the pinned forward sanitizer using its native cast order."""
    a=np.asarray(value,dtype=dtype).copy()
    eps=max(1e-12,np.finfo(a.dtype).eps)  # Canonical gs.init default eps/precision.
    if a.shape!=(7,) or not np.isfinite(a).all() or a[1]<eps:
        raise ContractError('Unsupported native solver-parameter input')
    if a[0]<eps:a[0]=max(.01,min_timeconst)
    a[0]=max(a[0],min_timeconst)
    a[[2,3,5]]=np.clip(a[[2,3,5]],.0001,.9999)
    a[4]=max(a[4],0.)
    a[6]=max(a[6],1.)
    return a


def _rigid16_inventory(entities):
    rows=_inventory(entities)
    by_name={e.name:e for e in entities}
    for geom,row in rows:
        row['native_entity_idx']=int(by_name[row['entity']].idx)
        row['native_link_id']=int(geom.link.idx)
    if sorted(r['global_id'] for _,r in rows)!=list(range(len(rows))):
        raise ContractError('Need the complete contiguous native geometry inventory')
    return rows


def prepare_rigid16_geometry(entities, request, *, floor, tray, beads, box_type, sphere_type, plane_type):
    """Select actual bound objects amid any complete three-arm geometry count."""
    request=validate_rigid16_request(request)
    if (any(e.is_built for e in entities) or len({id(e) for e in entities})!=len(entities)
            or len(beads)!=4 or len({id(e) for e in [floor,tray,*beads]})!=6
            or any(not any(e is selected for e in entities) for selected in [floor,tray,*beads])
            or floor.name!='floor' or tray.name!=TARGET_ENTITIES[0]
            or [e.name for e in beads]!=list(TARGET_ENTITIES[1:])):
        raise ContractError('Rigid16 requires the actual unbuilt floor/cup/original4bead objects')
    if (len(floor.geoms)!=1 or floor.geoms[0].type!=plane_type
            or len(tray.geoms)!=9 or any(g.type!=box_type or g.link.name!='payload' for g in tray.geoms)
            or any(len(e.geoms)!=1 or e.geoms[0].type!=sphere_type for e in beads)):
        raise ContractError('Rigid16 target primitive ownership differs')
    role_by_geom={id(g):'floor' for g in floor.geoms}
    role_by_geom.update({id(g):'tray' for g in tray.geoms})
    role_by_geom.update({id(g):'bead' for e in beads for g in e.geoms})
    inventory=_rigid16_inventory(entities)
    originals=[(g,r,_vector(g.sol_params)) for g,r in inventory]
    rows=[]
    for geom,identity,original in originals:
        role=role_by_geom.get(id(geom),'untouched');expected=original.copy()
        if role!='untouched':
            expected[0]=.005 if role=='tray' else .0003125
            geom.set_sol_params(expected.copy())
        rows.append({**identity,'contact_role':role,'selected':role!='untouched',
                     'before_prebuild':original.tolist(),'after_prebuild':expected.tolist()})
    for geom,identity in inventory:
        row=next(r for r in rows if r['global_id']==identity['global_id'])
        if not np.array_equal(_vector(geom.sol_params),np.asarray(row['after_prebuild'])):
            raise ContractError('Rigid16 setter changed/aliased an unrelated parameter')
    return {'requested':request,'geometries':rows,
            'selection':{'floor':[int(g.idx) for g in floor.geoms],
                         'tray':[int(g.idx) for g in tray.geoms],
                         'beads':[int(e.geoms[0].idx) for e in beads]}}


def rigid16_built_evidence(scene, entities, before, request):
    request=validate_rigid16_request(request)
    solver=scene.rigid_solver
    pruning=float(solver.collider._collider_info.contact_pruning_tolerance[None])
    if (type(before) is not dict or before.get('requested')!=request
            or float(scene.sim.dt)!=.0025 or scene.sim.substeps!=16 or scene.sim.substeps_local!=1
            or solver._options.contact_pruning_tolerance is not None or pruning!=0.
            or solver._options.constraint_timeconst!=.01 or solver._requires_grad or solver._use_hibernation
            or any(not e.is_built for e in entities)):
        raise ContractError('Actual rigid16 contact profile/branch differs')
    installed=Path(inspect_source_root(solver))
    for relative,expected in RIGID16_SOURCE_PINS.items():
        if hashlib.sha256((installed/relative).read_bytes()).hexdigest()!=expected:
            raise ContractError('Rigid16 sanitizer/native source changed: '+relative)
    old={r['global_id']:r for r in before['geometries']};rows=[]
    native_links=np.asarray(solver.dyn_info.geoms.link_idx.to_numpy())
    native_types=np.asarray(solver.dyn_info.geoms.type.to_numpy())
    if native_links.shape!=(solver.n_geoms,) or native_types.shape!=(solver.n_geoms,):
        raise ContractError('Native rigid16 geometry identity layout differs')
    for geom,identity in _rigid16_inventory(entities):
        row=old.pop(identity['global_id'])
        if (any(row[k]!=v for k,v in identity.items()) or native_links[geom.idx]!=identity['native_link_id']
                or native_types[geom.idx]!=identity['identity']['type']):
            raise ContractError('Rigid16 native geometry ownership changed during build')
        native=_vector(geom.sol_params)
        rows.append({**row,'native_sol_params':native.tolist(),'dtype':str(native.dtype)})
    if old or len(rows)!=solver.n_geoms:raise ContractError('Rigid16 native geometry inventory incomplete')
    evidence={'schema_version':'rigid16-bead-contact-native-v1','requested':request,'dt_s':.0025,
              'substeps':16,'substeps_local':1,'native_geometry_count':int(solver.n_geoms),
              'resolved_contact_pruning_tolerance':None,'native_pruning_tolerance':pruning,
              'native_default_timeconst_s':.01,'selection':copy.deepcopy(before['selection']),
              'geometries':rows,'source_sha256':dict(RIGID16_SOURCE_PINS),
              'scope':'Configured native values under a separate rigid16 model, not the old13geometry contract or physical acceptance.'}
    validate_rigid16_evidence(evidence,request)
    return evidence


def inspect_source_root(solver):
    """Read class provenance without importing a native runtime for CPU checks."""
    import inspect
    path=Path(inspect.getsourcefile(type(solver))).resolve()
    if type(solver).__module__!='genesis.engine.solvers.rigid.rigid_solver' or type(solver).__name__!='RigidSolver':
        raise ContractError('Exact native rigid solver class required')
    return path.parents[3]


def validate_rigid16_evidence(evidence, request):
    request=validate_rigid16_request(request)
    if (type(evidence) is not dict or evidence.get('schema_version')!='rigid16-bead-contact-native-v1'
            or evidence.get('requested')!=request or evidence.get('dt_s')!=.0025
            or evidence.get('substeps')!=16 or evidence.get('substeps_local')!=1
            or evidence.get('native_pruning_tolerance')!=0. or evidence.get('resolved_contact_pruning_tolerance','missing') is not None
            or evidence.get('native_default_timeconst_s')!=.01 or evidence.get('source_sha256')!=RIGID16_SOURCE_PINS):
        raise ContractError('Malformed rigid16 configured evidence')
    rows=evidence['geometries'];count=evidence['native_geometry_count'];selection=evidence['selection']
    if (type(count) is not int or count<14 or len(rows)!=count
            or sorted(r['global_id'] for r in rows)!=list(range(count))
            or set(selection)!=set(('floor','tray','beads'))
            or [len(selection[k]) for k in ('floor','tray','beads')]!=[1,9,4]
            or len(set(sum(selection.values(),[])))!=14):
        raise ContractError('Rigid16 complete/selected native geometry inventory differs')
    signatures=[]
    for row in rows:
        idx=row['global_id'];role=('floor' if idx in selection['floor'] else 'tray' if idx in selection['tray']
                                else 'bead' if idx in selection['beads'] else 'untouched')
        if row['contact_role']!=role or row['selected'] is not (role!='untouched') or row['dtype'] not in ('float32','float64'):
            raise ContractError('Wrong rigid16 selection or native dtype')
        before,after,native=(_vector(row[k]) for k in ('before_prebuild','after_prebuild','native_sol_params'))
        expected=before.copy()
        if role!='untouched':expected[0]=.005 if role=='tray' else .0003125
        resolved=_rigid16_expected_native(after,row['dtype'],.0003125)
        baseline=_rigid16_expected_native(before,row['dtype'],.005)
        if (not np.array_equal(after,expected) or not np.array_equal(native,resolved)
                or not np.array_equal(native[1:],baseline[1:])
                or (role=='untouched' and not np.array_equal(native,baseline))):
            raise ContractError('Rigid16 changed other6 or an unrelated native parameter')
        expected_type={'floor':0,'tray':5,'bead':1}.get(role)
        if expected_type is not None and row['identity']['type']!=expected_type:
            raise ContractError('Rigid16 selected primitive differs')
        if role=='tray' and (row['entity']!='tray_0' or row['identity']['link']!='payload'):
            raise ContractError('Rigid16 selected cup identity differs')
        if role=='floor' and row['entity']!='floor':raise ContractError('Rigid16 selected floor identity differs')
        signatures.append({k:copy.deepcopy(v) for k,v in row.items() if k!='global_id'})
    selected_beads=[r for r in rows if r['contact_role']=='bead']
    if (sorted(r['entity'] for r in selected_beads)!=list(TARGET_ENTITIES[1:])
            or len({r['native_entity_idx'] for r in selected_beads})!=4
            or sorted(r['identity']['local_index'] for r in rows if r['contact_role']=='tray')!=list(range(9))):
        raise ContractError('Rigid16 original bead/cup identities incomplete')
    return sorted(signatures,key=lambda r:(r['entity'],digest(r)))
