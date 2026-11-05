"""Explicit material bindings for the existing student three-arm executor.

No native runtime is imported until scene construction. These experimental
providers are instructor-owned. A source match is not physical acceptance.
Rigid bead inputs keep their existing executor and contracts.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

from transport_challenge.contracts import ContractError, digest

PROJECT = Path(__file__).resolve().parents[1]
QA = PROJECT/'qa/all_material_physics_20261031'
SCHEMA = 'episode-native-particle-request-v1'
BOUNDS = ((-1., -1.3, -.2), (4.3, 3.2, 2.1))
LIQUID_PROVIDER_SHA = '36d228f91b291bc8205a3fa987c58c08fa7685471f6d0ba489eab85db6b3a933'
SPARSE_PROVIDER_SHA = '90bba1570f08dae3a54f7a42bfbd578f5d43bdf7071ab1303bb77f0fd92e8106'
CACHE_PROVIDER_SHA = 'ffa5e4f971185f845ea308b43601713b2e4be9d24c28045aecba7cd2b63f8b96'
CACHE_RUNTIME_PROFILE = 'cpic_cache_64_tol12'


def require(value, message):
    if not value: raise ContractError(message)


def request(kind, precision='32', runtime_profile='default'):
    require(kind in ('process_liquid', 'sand'), 'Explicit native process_liquid or sand required')
    require(precision in ('32', '64') and (precision == '32' or kind == 'sand'),
            'Optional precision64 is supported only for explicit Sand')
    require(runtime_profile == 'default' or (runtime_profile == CACHE_RUNTIME_PROFILE
            and kind == 'sand' and precision == '64'), 'Cache profile requires explicit Sand64')
    native = dict(rho=1000., mu=1e-6, gamma=0., sampler='regular', exponent=7.,
                  stiffness=50000., use_visual_raycasting=False) if kind == 'process_liquid' else dict(
                  E=1e6, nu=.2, rho=1000., friction_angle=45., sampler='regular',
                  lam=277777.7777777778, mu=416666.6666666667, use_visual_raycasting=False)
    result = dict(schema_version=SCHEMA, material_kind=kind, particle_count=2916,
        particle_size_m=.004, dt_s=.0025, substeps=100, substeps_local=1,
        requires_grad=False, source_id=kind+'_source', initial_settle_s=2.2,
        lower_bound_m=list(BOUNDS[0]), upper_bound_m=list(BOUNDS[1]),
        provider_sha256=LIQUID_PROVIDER_SHA if kind == 'process_liquid' else SPARSE_PROVIDER_SHA,
        material_settings=dict(rho=1000., mu=1e-6, gamma=0., sampler='regular') if kind == 'process_liquid'
        else dict(E=1e6, nu=.2, rho=1000., friction_angle=45., sampler='regular'),
        expected_native_material_settings=native,
        expected_native_solver_class='CupDensityMapSPHSolver' if kind == 'process_liquid' else 'ScenePrivateSparseMPM',
        expected_mass_kg=.149299201744725 if kind == 'process_liquid' else .18662400886416433,
        rest=dict(window_s=.5, RMS_max_m_s=.03, q95_max_m_s=.1, COM_excursion_max_m=.0005))
    if precision == '64':
        result.update(precision='64', expected_mass_kg=.18662400000000004)
    if runtime_profile == CACHE_RUNTIME_PROFILE:
        result.update(runtime_profile=runtime_profile, provider_sha256=CACHE_PROVIDER_SHA,
            base_provider_sha256=SPARSE_PROVIDER_SHA, rigid_solver_tolerance=1e-12,
            expected_native_solver_class='ScenePrivateCPICLatticeCacheMPM')
    return result


def validate_request(world, scenario=None):
    r = world.get('particle_material_request')
    require(type(r) is dict and r == request(r.get('material_kind'), r.get('precision', '32'),
            r.get('runtime_profile', 'default')), 'Unapproved native particle request')
    require(world.get('initial_contents') == [] and world.get('initial_content_world_centers_m') == [],
            'Native particles cannot be declared as four rigid beads')
    if scenario is not None:
        require(scenario.get('contents') == [] and scenario.get('particle_material_request') == r
                and scenario['provenance']['world_sha256'] == digest(world),
                'Particle scenario, world and rigid query contents differ')
    return copy.deepcopy(r)


def particle_inputs(world, scenario, kind, precision='32', runtime_profile='default'):
    """Convert a selected geometric layout, never a saved physical state."""
    require(scenario['provenance']['world_sha256'] == digest(world)
            and scenario['case'] == world['case_record'] and len(scenario['robots']) == 3,
            'Use the exact selected three-arm layout')
    require(world['source_body_world']['position_m'] == [-.1, 0., .65]
            and world['source_body_world']['quaternion_wxyz'] == [1., 0., 0., 0.],
            'Current particle initial recipe is bound to the canonical source')
    w, s = copy.deepcopy(world), copy.deepcopy(scenario)
    w['particle_template_world_sha256'] = digest(world)
    w['particle_material_request'] = request(kind, precision, runtime_profile)
    w['initial_contents'] = []; w['initial_content_world_centers_m'] = []
    s['contents'] = []; s['particle_material_request'] = copy.deepcopy(w['particle_material_request'])
    s['provenance']['world_sha256'] = digest(w)
    validate_request(w, s)
    return w, s


def bind_native_contract(world, scenario, contract):
    r = validate_request(world, scenario)
    require(contract['material_kind'] == r['material_kind'] and contract['source_id'] == r['source_id']
            and contract['particle_count'] == 2916 and contract['native_particle_ids'] == list(range(2916))
            and contract['native_material_settings'] == r['expected_native_material_settings']
            and contract['native_solver_class'] == r['expected_native_solver_class']
            and abs(contract['initial_total_mass_kg']-r['expected_mass_kg']) <= 1e-14,
            'Actual native inventory/material settings differ from the fixed request')
    w, s = copy.deepcopy(world), copy.deepcopy(scenario)
    w['native_particle_contract'] = copy.deepcopy(contract)
    s['provenance']['world_sha256'] = digest(w)
    validate_request(w, s)
    return w, s


def scene_options(gs, world):
    r = validate_request(world)
    options = dict(sim_options=gs.options.SimOptions(dt=.0025, substeps=100,
        substeps_local=1, requires_grad=False, gravity=(0., 0., -9.81)))
    if r['material_kind'] == 'process_liquid':
        options['sph_options'] = gs.options.SPHOptions(particle_size=.004, pressure_solver='WCSPH',
            lower_bound=BOUNDS[0], upper_bound=BOUNDS[1], hash_grid_res=None, hash_grid_cell_size=.008)
    else:
        options['mpm_options'] = gs.options.MPMOptions(particle_size=.004, grid_density=256., enable_CPIC=True,
            lower_bound=BOUNDS[0], upper_bound=BOUNDS[1])
    return options


def install_empty_scene(scene, world):
    r = validate_request(world)
    if r['material_kind'] == 'process_liquid': return None
    if r.get('runtime_profile') == CACHE_RUNTIME_PROFILE:
        raise ContractError('This public fixed example supports four rigid beads only')
        require(hashlib.sha256(Path(cache.__file__).read_bytes()).hexdigest() == CACHE_PROVIDER_SHA,
                'Frozen cache provider changed')
        return cache.install_preentity(scene, opt_in=cache.OPT_IN)[1]
    raise ContractError('This public fixed example supports four rigid beads only')
    require(hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest() == SPARSE_PROVIDER_SHA,
            'Frozen sparse provider changed')
    return provider.install_preentity(scene, opt_in=provider.OPT_IN)[1]


def rigid_option_overrides(world):
    r = validate_request(world)
    return {'tolerance':1e-12} if r.get('runtime_profile') == CACHE_RUNTIME_PROFILE else {}


def add_material(gs, scene, world, directory):
    r = validate_request(world)
    if r['material_kind'] == 'process_liquid':
        raise ContractError('This public fixed example supports four rigid beads only')
        cup, contents, provider, record = factory.add_canonical_liquid(gs, scene, directory)
        return cup, contents, provider, record
    from physics.scene_assets import write_tray_urdf
    directory = Path(directory)
    asset, mass, geometry = write_tray_urdf(directory, world['vessel_model']['container'])
    p = world['source_body_world']
    cup = scene.add_entity(gs.morphs.URDF(file=str(asset), fixed=False, align=False, merge_fixed_links=False,
        pos=p['position_m'], quat=p['quaternion_wxyz']), material=gs.materials.Rigid(friction=1., needs_coup=True,
        coup_friction=.4, friction_torsional=.001), name='tray_0')
    center, half = np.asarray(geometry['interior_center_local_m']), np.asarray(geometry['interior_half_extents_m'])
    source = np.asarray(p['position_m'])+np.array([center[0], center[1], center[2]-half[2]+.020])
    contents = scene.add_entity(gs.morphs.Box(size=(.072, .072, .036), pos=tuple(source)),
        material=gs.materials.MPM.Sand(**r['material_settings']), name='sand')
    raise ContractError('This public fixed example supports four rigid beads only')
    if r.get('runtime_profile') == CACHE_RUNTIME_PROFILE:
        raise ContractError('This public fixed example supports four rigid beads only')
    record = dict(schema='canonical-sand-prebuild-candidate-v1', geometry=geometry,
        empty_mass_kg=mass, provider_sha256=r['provider_sha256'], original_source_pos_m=source.tolist(),
        pose_or_material_state_replayed=False, robot_transport_approved=False)
    return cup, contents, provider, record


def provider_source_pins(kind, runtime_profile='default'):
    require(kind in ('process_liquid', 'sand'), 'Unknown material')
    require(runtime_profile == 'default' or (runtime_profile == CACHE_RUNTIME_PROFILE and kind == 'sand'),
            'Unknown material provider profile')
    path = (QA/'liquid_integration_branch/private_density_map_canonical_candidate.py' if kind == 'process_liquid'
            else QA/'performance_branch/private_sparse_mpm_candidate.py')
    expected = LIQUID_PROVIDER_SHA if kind == 'process_liquid' else SPARSE_PROVIDER_SHA
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Native material provider changed')
    pins = {str(path.resolve()): expected}
    if kind == 'process_liquid':
        raise ContractError('This public fixed example supports four rigid beads only')
        pins.update(canonical_liquid_scene.source_bindings())
    else:
        from importlib.metadata import distribution
        from physics.particle_material import SPARSE_MPM_NATIVE_PINS
        native = Path(distribution('genesis-world').locate_file('genesis'))
        for relative, sha in SPARSE_MPM_NATIVE_PINS.items():
            source = (native/relative).resolve()
            require(hashlib.sha256(source.read_bytes()).hexdigest() == sha,
                    'Sparse native source changed: '+relative)
            pins[str(source)] = sha
        if runtime_profile == CACHE_RUNTIME_PROFILE:
            raise ContractError('This public fixed example supports four rigid beads only')
            cache_path = Path(cache.__file__).resolve()
            require(hashlib.sha256(cache_path.read_bytes()).hexdigest() == CACHE_PROVIDER_SHA,
                    'Frozen cache provider changed')
            pins[str(cache_path)] = CACHE_PROVIDER_SHA
            pins.update(cache.source_contract()['native_source_hashes'])
    return pins


def native_component_records(cup, model, construction, box_type):
    """Match actual nine-box decomposition while preserving the fixed inertia.

    Liquid walls overlap at the corners. Their solid union is the original
    vessel union, but their individual native boxes differ. Never label them
    as the original disjoint primitives or sum their masses twice.
    """
    from physics.loaded_joint_scene import validate_built_components
    boxes = construction['geometry']['component_boxes']
    require(len(boxes) == 9, 'Native particle cup must have all nine components')
    records = [dict(component_id='solid_'+str(i), size_m=list(size),
        center_body_m=list(center), quaternion_body_wxyz=[1., 0., 0., 0.])
        for i, (size, center) in enumerate(boxes)]
    return validate_built_components(cup, model, records, box_type)


def canonical_reader_geometry(model, construction):
    """Use the canonical model after checking the actual native solid union.

    Factory metadata has extra descriptive fields and liquid's wall boxes
    overlap. Neither difference is a different physical vessel. Compare every
    cell of the combined axis-plane partition before using its canonical union
    for cavity/depth classification. Native primitives are separately matched
    by native_component_records. Plane rounding is1pm, well below the existing
    native primitive match tolerance1e-7m. This is not a new contact tolerance.
    """
    original=model.geometry_record();actual=construction['geometry']
    for key in ('interior_center_local_m','interior_half_extents_m','interior_regions',
                'interior_membership_rule','capacity_m3','partitions','empty_mass_kg',
                'center_of_mass_local_m','inertia_at_com_kg_m2'):
        require(digest(actual[key])==digest(original[key]),'Native particle cavity/inertia differs: '+key)
    a=np.asarray(actual['component_boxes'],float);b=np.asarray(original['component_boxes'],float)
    require(a.shape==b.shape==(9,2,3) and np.isfinite(a).all() and np.isfinite(b).all()
            and np.all(a[:,0]>0) and np.all(b[:,0]>0),'Require both actual/canonical nine-box solids')
    bounds=np.concatenate((a[:,1]-a[:,0]/2,a[:,1]+a[:,0]/2,b[:,1]-b[:,0]/2,b[:,1]+b[:,0]/2))
    rounded=np.round(bounds,12)
    axes=[np.unique(rounded[:,i]) for i in range(3)]
    centers=[(v[:-1]+v[1:])/2 for v in axes]
    cells=np.stack(np.meshgrid(*centers,indexing='ij'),axis=-1).reshape(-1,3)
    def inside(boxes):return np.any(np.all(np.abs(cells[:,None,:]-boxes[None,:,1,:])<=boxes[None,:,0,:]/2,axis=2),axis=1)
    require(np.array_equal(inside(a),inside(b)),'Actual and canonical vessel solid unions differ')
    receipt=dict(schema='native-particle-canonical-union-binding-v1',
        actual_geometry_sha256=digest(actual),canonical_geometry_sha256=model.geometry_sha256,
        actual_boxes_sha256=digest(actual['component_boxes']),canonical_boxes_sha256=digest(original['component_boxes']),
        partition_cells=len(cells),plane_rounding_m=1e-12,
        maximum_plane_rounding_error_m=float(np.max(np.abs(bounds-rounded))),
        all_partition_cell_memberships_equal=True,cavity_and_inertia_equal=True,
        scope='Same axis-aligned solid union and fixed cavity/inertia. Native component identities stay separately recorded, not relabeled.')
    return original,receipt
