"""Compact, pure observations of an entire native particle inventory.

The caller authenticates ParticleMaterialReader and its live source bindings,
reads the actual body and particles at one clock, and archives that same full-N
result. This module checks supplied data, not native provenance. Put its packet
under observation['particle_material']. Never substitute rigid sphere rows.
An archive reference may still be pending. It is not a durability certificate.
"""
from __future__ import annotations

import copy
import hashlib
import math
from functools import wraps

import numpy as np

from transport_challenge.contracts import ContractError, digest


SCHEMA = 'particle-episode-packet-v1'
DT_S = .0025
REST_LIMITS = {'window_s': .5, 'relative_RMS_max_m_s': .03,
               'relative_q95_max_m_s': .1, 'COM_excursion_max_m': .0005}
PACKET_FIELDS = {'schema_version', 'material_kind', 'source_id', 'native_entity_idx', 'particle_count',
    'native_particle_ids_sha256', 'particle_masses_kg_sha256', 'id_mass_sha256', 'mass_native_sha256',
    'initial_mass_kg', 'native_scene_step', 'simulation_s', 'native_clock', 'source_contract',
    'body_state_sha256', 'arrays_sha256', 'counts', 'mass', 'quality', 'maximum_solid_union_depth_m',
    'bulk', 'archiveRef', 'outside_carrier_is_verified_loss', 'delivered_credit_assigned',
    'source_authentication', 'sha256'}


def _require(condition, message):
    if not condition:
        raise ContractError(message)


def _contract_errors(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ContractError:
            raise
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ContractError('Invalid particle episode data: '+str(exc)) from exc
    return checked


def _number(value, label, minimum=None):
    _require(type(value) in (int, float) and math.isfinite(value)
             and (minimum is None or value >= minimum), label+' must be finite numeric data')
    return float(value)


def _index(value, label):
    _require(type(value) is int and 0 <= value <= np.iinfo(np.int64).max, 'Invalid '+label)
    return value


def _hash(value):
    _require(type(value) is str and len(value) == 64
             and all(c in '0123456789abcdef' for c in value), 'Missing SHA256')
    return value


def _vector(value, size, label):
    _require(type(value) in (list, tuple) and len(value) == size, 'Invalid '+label)
    return np.asarray([_number(v, label) for v in value], np.float64)


def _array_hash(value):
    a = np.ascontiguousarray(value)
    return digest(dict(dtype=a.dtype.str, shape=list(a.shape),
                       data_sha256=hashlib.sha256(a.tobytes()).hexdigest()))


def _clock(value):
    _require(type(value) is list and len(value) == 4, 'Missing actual native clock')
    _index(value[0], 'native step'); _number(value[1], 'native time', 0.)
    _index(value[2], 'global substep'); _index(value[3], 'local substep')
    return value


def _body(observation):
    _require(type(observation) is dict, 'Actual same-clock body observation required')
    pose = observation['body_world']
    _require(type(pose) is dict and pose.get('frame') == 'world', 'Body pose must be world frame')
    p = _vector(pose['position_m'], 3, 'body position')
    q = _vector(pose['quaternion_wxyz'], 4, 'body quaternion')
    _require(abs(np.linalg.norm(q)-1.) <= 1e-6, 'Actual body quaternion must be normalized')
    w, x, y, z = q / np.linalg.norm(q)
    rotation = np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    linear = _vector(observation['body_linear_velocity_m_s'], 3, 'body linear velocity')
    angular = _vector(observation['body_angular_velocity_rad_s'], 3, 'body angular velocity')
    return p, rotation, linear, angular


def _contract(contract):
    digest(contract)
    _require(type(contract) is dict and contract.get('schema_version') ==
             'native-particle-material-mass-contract-v1', 'Original reader contract required')
    _require(contract['material_kind'] in ('sand', 'process_liquid'), 'Explicit particle material kind required')
    for key in ('source_id', 'native_entity_name', 'native_solver_class'):
        _require(type(contract[key]) is str and bool(contract[key]), 'Missing '+key)
    n = _index(contract['particle_count'], 'original count')
    start = _index(contract['native_particle_start'], 'particle start')
    _index(contract['native_entity_idx'], 'native entity')
    _require(n > 0 and type(contract['native_particle_ids']) is list
             and len(contract['native_particle_ids']) == n
             and all(type(v) is int and v == start+i for i, v in enumerate(contract['native_particle_ids'])),
             'Original full native-order IDs required')
    masses = contract['particle_masses_kg']
    _require(type(masses) is list and len(masses) == n, 'Original per-ID masses required')
    weights = np.asarray([_number(v, 'particle mass', 0.) for v in masses], np.float64)
    _require(np.all(weights > 0) and contract['initial_total_mass_kg'] == math.fsum(weights),
             'Original mass/count contract differs')
    scale = 1000. if contract['material_kind'] == 'sand' else 1.
    _require(contract['native_mass_scale'] == scale
             and contract['particle_reset_or_inactive_omission_permitted'] is False,
             'Original SI scale and no-omission contract required')
    for key in ('actual_native_mass_sha256', 'geometry_sha256', 'native_material_settings_sha256'):
        _hash(contract[key])
    _require(digest(contract['native_material_settings']) == contract['native_material_settings_sha256'],
             'Original material settings differ')
    pins = contract['source_hashes']
    _require(type(pins) is dict and bool(pins), 'Reader/native source bindings required')
    for path, sha in pins.items():
        _require(type(path) is str and bool(path), 'Missing source path'); _hash(sha)
    low = _vector(contract['effective_lower_bound_m'], 3, 'effective lower bound')
    high = _vector(contract['effective_upper_bound_m'], 3, 'effective upper bound')
    native_low = _vector(contract['native_lower_bound_m'], 3, 'native lower bound')
    native_high = _vector(contract['native_upper_bound_m'], 3, 'native upper bound')
    padding = _number(contract['native_boundary_padding_m'], 'native padding', 0.)
    _require(np.all(low < high) and np.array_equal(low, native_low+padding)
             and np.array_equal(high, native_high-padding), 'Actual effective native domain differs')
    _number(contract['domain_warning_margin_m'], 'domain warning margin', 0.)
    _clock(contract['initial_native_clock'])
    return n, weights


def _inventory(contract, weights):
    return dict(material_kind=contract['material_kind'], source_id=contract['source_id'],
        native_entity_idx=contract['native_entity_idx'], particle_count=contract['particle_count'],
        native_particle_ids_sha256=_array_hash(np.asarray(contract['native_particle_ids'], np.int64)),
        particle_masses_kg_sha256=_array_hash(weights),
        id_mass_sha256=digest(dict(source_id=contract['source_id'], native_entity_idx=contract['native_entity_idx'],
            native_ids=_array_hash(np.asarray(contract['native_particle_ids'], np.int64)),
            masses_kg=_array_hash(weights))),
        mass_native_sha256=contract['actual_native_mass_sha256'],
        initial_mass_kg=contract['initial_total_mass_kg'])


def _source_contract(contract):
    return dict(schema_version=contract['schema_version'], sha256=digest(contract),
        source_hashes_sha256=digest(contract['source_hashes']), geometry_sha256=contract['geometry_sha256'],
        native_material_settings_sha256=contract['native_material_settings_sha256'])


def _archive(reference, contract, step):
    digest(reference)
    _require(type(reference) is dict and reference.get('schema_version') == 'native-particle-archive-v1'
             and reference.get('particle_material_contract_sha256') == digest(contract)
             and type(reference.get('full_native_order_count')) is int
             and reference['full_native_order_count'] == contract['particle_count']
             and type(reference.get('native_scene_step')) is int and reference['native_scene_step'] == step
             and type(reference.get('row_index')) is int and reference['row_index'] >= 0
             and type(reference.get('chunk_file')) is str and bool(reference['chunk_file'])
             and reference.get('durability') in ('PENDING_UNTIL_HASHED_MANIFEST_COMMIT',
                 'COMMITTED_IN_HASHED_MANIFEST', 'COMMITTED_ONLY_WHEN_LISTED_IN_MANIFEST'),
             'Same-clock full-N ParticleArchive reference required')


@_contract_errors
def packet_from_reader_result(contract, reader_result, body_observation, *, archive_ref):
    """Validate all original arrays and return one bounded JSON packet.

    No native methods run here. Call reader.verify_source_bindings(), then read
    and archive the same result, before attaching this packet and committing an
    episode observation. Omission of any inactive/outside ID is an error.
    """
    n, weights = _contract(contract)
    _require(type(reader_result) is dict and type(reader_result.get('arrays')) is dict
             and type(reader_result.get('summary')) is dict, 'Full reader arrays and summary required')
    a, s = reader_result['arrays'], reader_result['summary']
    expected = {'schema_version': 'native-particle-material-observation-v1',
        'particle_material_contract_sha256': digest(contract), 'material_kind': contract['material_kind'],
        'source_id': contract['source_id'], 'native_entity_idx': contract['native_entity_idx'],
        'persistent_id_count': n, 'initial_mass_kg': math.fsum(weights), 'native_mass_kg': math.fsum(weights),
        'mass_native_sha256': contract['actual_native_mass_sha256'],
        'identity_and_mass_unchanged': True, 'native_clock_unchanged': True,
        'outside_carrier_is_verified_loss': False, 'delivered_credit_assigned': False}
    _require(all(type(s.get(k)) is type(v) and s[k] == v for k, v in expected.items()),
             'Reader summary differs from original inventory/contract')
    clock = _clock(s['native_clock']); step, now = clock[:2]
    _number(s['simulation_s'], 'reader time', 0.); _number(body_observation['simulation_s'], 'body time', 0.)
    _require(type(s['native_scene_step']) is int and s['native_scene_step'] == step
             and s['simulation_s'] == now and type(body_observation['native_scene_step']) is int
             and body_observation['native_scene_step'] == step and body_observation['simulation_s'] == now
             and abs(now-step*DT_S) <= 1e-9,
             'Reader and actual body clock differ')
    p, rotation, linear, angular = _body(body_observation)
    shapes = {'pos': (n, 3), 'vel': (n, 3), 'local_xyz': (n, 3), 'mass_kg': (n,),
        'mass_native': (n,), 'native_particle_id': (n,), 'solid_union_depth_m': (n,)}
    booleans = ('active', 'inside_carrier', 'inside_solid_union', 'deep_solid',
                'outside_effective_domain', 'domain_warning')
    shapes.update({key: (n,) for key in booleans})
    if contract['material_kind'] == 'sand':
        shapes.update(F=(n, 3, 3), detF=(n,), material_idx=(n,), free=(n,))
    else:
        shapes['rho0'] = (n,)
    for key, shape in shapes.items():
        _require(type(a.get(key)) is np.ndarray and a[key].shape == shape, 'Full-N array shape differs: '+key)
        kind = 'b' if key in booleans+('free',) else 'iu' if key in ('native_particle_id', 'material_idx') else 'f'
        _require(a[key].dtype.kind in kind and (key == 'F' or key == 'detF' or np.isfinite(a[key]).all()),
                 'Native array dtype/nonfinite data: '+key)
    _require(a['native_particle_id'].dtype == np.dtype('int64') and a['mass_kg'].dtype == np.dtype('float64')
             and np.array_equal(a['native_particle_id'], np.asarray(contract['native_particle_ids'], np.int64))
             and np.array_equal(a['mass_kg'], weights)
             and _array_hash(a['mass_native']) == contract['actual_native_mass_sha256']
             and np.array_equal(a['mass_native'].astype(float)/contract['native_mass_scale'], weights),
             'Native ID/order or immutable mass changed')
    local = (a['pos'].astype(float)-p) @ rotation
    _require(np.array_equal(a['local_xyz'], local), 'Reader local coordinates differ from actual body pose')
    low = np.asarray(contract['effective_lower_bound_m']); high = np.asarray(contract['effective_upper_bound_m'])
    margin = contract['domain_warning_margin_m']
    _require(np.array_equal(a['outside_effective_domain'], np.any((a['pos'] < low)|(a['pos'] > high), axis=1))
             and np.array_equal(a['domain_warning'], np.any((a['pos'] < low+margin)|(a['pos'] > high-margin), axis=1)),
             'Actual native domain masks differ')
    _require(np.all(a['solid_union_depth_m'] >= 0)
             and np.array_equal(a['deep_solid'], a['solid_union_depth_m'] > .001)
             and not np.any((a['solid_union_depth_m'] > 0) & ~a['inside_solid_union']),
             'Solid depth/masks disagree')
    counts = {key+'_count': int(a[key].sum()) for key in
              ('active', 'inside_carrier', 'deep_solid', 'outside_effective_domain', 'domain_warning')}
    counts['retained_count'] = counts.pop('inside_carrier_count')
    counts['inactive_count'] = n-counts['active_count']; counts['outside_carrier_count'] = n-counts['retained_count']
    mass = dict(native_mass_kg=math.fsum(weights),
        inactive_mass_kg=float(weights[~a['active']].sum()),
        retained_mass_kg=float(weights[a['inside_carrier']].sum()),
        outside_carrier_mass_kg=float(weights[~a['inside_carrier']].sum()))
    depth = float(a['solid_union_depth_m'].max(initial=0.))
    quality = {'MPM_F_finite': None, 'MPM_positive_detF': None, 'MPM_minimum_detF': None}
    if contract['material_kind'] == 'sand':
        finite = bool(np.isfinite(a['F']).all() and np.isfinite(a['detF']).all())
        if finite:
            _require(np.array_equal(np.linalg.det(a['F'].astype(float)), a['detF']), 'Reader F determinant differs')
        quality.update(MPM_F_finite=finite, MPM_positive_detF=bool(finite and np.all(a['detF'] > 0)),
            MPM_minimum_detF=float(a['detF'].min()) if finite else None)
    finite = quality['MPM_F_finite'] is not False
    passed = bool(counts['inactive_count'] == counts['deep_solid_count'] ==
        counts['outside_effective_domain_count'] == 0 and finite and quality['MPM_positive_detF'] is not False)
    derived = dict(**counts, **mass, **quality, maximum_solid_union_depth_m=depth,
                   finite_arrays=finite, material_quality_passed=passed)
    _require(all(type(s.get(k)) is type(v) and s[k] == v for k, v in derived.items()),
             'Reader aggregate does not match all-N arrays')
    relative = a['vel'].astype(float)-linear-np.cross(np.broadcast_to(angular, (n, 3)), a['pos'].astype(float)-p)
    speed = np.linalg.norm(relative, axis=1)
    bulk = dict(relative_RMS_m_s=float(np.sqrt(np.sum(weights*speed**2)/weights.sum())),
        relative_q95_m_s=float(np.quantile(speed, .95)),
        COM_body_m=(np.sum(local*weights[:, None], axis=0)/weights.sum()).tolist(),
        metric_inventory='ALL_ORIGINAL_IDS_INCLUDING_INACTIVE_AND_OUTSIDE',
        velocity_definition='v_particle-v_body-omega_body_cross_(x_particle-x_body)')
    _archive(archive_ref, contract, step)
    packet = dict(schema_version=SCHEMA, **_inventory(contract, weights), native_scene_step=step,
        simulation_s=now, native_clock=copy.deepcopy(clock), source_contract=_source_contract(contract),
        body_state_sha256=digest({key: body_observation[key] for key in ('body_world',
            'body_linear_velocity_m_s', 'body_angular_velocity_rad_s')}),
        arrays_sha256=digest({key: _array_hash(value) for key, value in a.items()}),
        counts=counts, mass=mass, quality=dict(**quality, finite_arrays=finite, material_quality_passed=passed),
        maximum_solid_union_depth_m=depth, bulk=bulk, archiveRef=copy.deepcopy(archive_ref),
        outside_carrier_is_verified_loss=False, delivered_credit_assigned=False,
        source_authentication='CALLER_OWNS_NATIVE_READER_AND_LIVE_SOURCE_VERIFICATION')
    return {**packet, 'sha256': digest(packet)}


@_contract_errors
def validate_particle_packet(packet, contract, observation=None, *, require_retained=False):
    """Check a compact packet against the original contract, not caller truth."""
    _require(type(require_retained) is bool, 'Retention mode must be explicit Boolean')
    return _validate_particle_packet_with_context(packet, _packet_validation_context(contract),
        observation, require_retained=require_retained)


@_contract_errors
def _packet_validation_context(contract):
    """Derive only immutable expectations, never acceptance of any packet.

    No context is stored globally or attached to a caller's row. The public
    validator creates one per call. A rest window creates one from its own
    deep contract snapshot and still checks every packet/body/clock/hash.
    """
    n, weights = _contract(contract)
    expected = dict(**_inventory(contract, weights), source_contract=_source_contract(contract))
    return n, math.fsum(weights), expected


@_contract_errors
def _window_validation_context(contract):
    return _packet_validation_context(copy.deepcopy(contract))


def _archive_with_context(reference, contract_sha256, particle_count, step):
    digest(reference)
    _require(type(reference) is dict and reference.get('schema_version') == 'native-particle-archive-v1'
             and reference.get('particle_material_contract_sha256') == contract_sha256
             and type(reference.get('full_native_order_count')) is int
             and reference['full_native_order_count'] == particle_count
             and type(reference.get('native_scene_step')) is int and reference['native_scene_step'] == step
             and type(reference.get('row_index')) is int and reference['row_index'] >= 0
             and type(reference.get('chunk_file')) is str and bool(reference['chunk_file'])
             and reference.get('durability') in ('PENDING_UNTIL_HASHED_MANIFEST_COMMIT',
                 'COMMITTED_IN_HASHED_MANIFEST', 'COMMITTED_ONLY_WHEN_LISTED_IN_MANIFEST'),
             'Same-clock full-N ParticleArchive reference required')


@_contract_errors
def _validate_particle_packet_with_context(packet, context, observation=None, *, require_retained=False):
    _require(type(require_retained) is bool, 'Retention mode must be explicit Boolean')
    n, total_mass, expected = context
    _require(type(packet) is dict and packet.get('schema_version') == SCHEMA
             and set(packet) == PACKET_FIELDS
             and packet.get('sha256') == digest({k: v for k, v in packet.items() if k != 'sha256'}),
             'Particle packet schema/hash changed')
    _require(all(type(packet.get(k)) is type(v) and packet[k] == v for k, v in expected.items()),
             'Particle packet differs from original inventory/source contract')
    clock = _clock(packet['native_clock']); step, now = clock[:2]
    _number(packet['simulation_s'], 'packet time', 0.)
    _require(type(packet['native_scene_step']) is int and packet['native_scene_step'] == step
             and packet['simulation_s'] == now and abs(now-step*DT_S) <= 1e-9, 'Particle packet clock differs')
    if observation is not None:
        _body(observation)
        _number(observation['simulation_s'], 'actual body time', 0.)
        _require(type(observation['native_scene_step']) is int and observation['native_scene_step'] == step
                 and observation['simulation_s'] == now and packet['body_state_sha256'] == digest({key:
                 observation[key] for key in ('body_world', 'body_linear_velocity_m_s', 'body_angular_velocity_rad_s')}),
                 'Particle packet is not from this actual body/clock')
    _archive_with_context(packet['archiveRef'], expected['source_contract']['sha256'], n, step)
    for key in ('arrays_sha256', 'body_state_sha256'): _hash(packet[key])
    counts = packet['counts']; mass = packet['mass']
    _require(set(counts) == {'active_count', 'inactive_count', 'retained_count', 'outside_carrier_count',
                'deep_solid_count', 'outside_effective_domain_count', 'domain_warning_count'}
             and set(mass) == {'native_mass_kg', 'inactive_mass_kg', 'retained_mass_kg', 'outside_carrier_mass_kg'},
             'Typed particle count/mass fields differ')
    for key in ('active_count', 'inactive_count', 'retained_count', 'outside_carrier_count',
                'deep_solid_count', 'outside_effective_domain_count', 'domain_warning_count'):
        _require(_index(counts[key], key) <= n, 'Particle count exceeds original inventory')
    _require(counts['active_count']+counts['inactive_count'] == n
             and counts['retained_count']+counts['outside_carrier_count'] == n,
             'Particle count partition differs')
    for key in ('native_mass_kg', 'inactive_mass_kg', 'retained_mass_kg', 'outside_carrier_mass_kg'):
        # Reader mask sums use NumPy reduction, original total uses math.fsum.
        # This is the original 1e-10 kg ledger check, not a physical-loss allowance.
        _require(_number(mass[key], key, 0.) <= total_mass+1e-10, 'Particle mass exceeds original inventory')
    # Counts cannot establish a per-ID mass. These identities follow the all-N reader validation.
    _require(mass['native_mass_kg'] == total_mass
             and math.isclose(mass['retained_mass_kg']+mass['outside_carrier_mass_kg'], total_mass,
                              rel_tol=0., abs_tol=1e-10), 'Original particle mass is not conserved')
    for key in ('relative_RMS_m_s', 'relative_q95_m_s'): _number(packet['bulk'][key], key, 0.)
    _require(set(packet['bulk']) == {'relative_RMS_m_s', 'relative_q95_m_s', 'COM_body_m',
                 'metric_inventory', 'velocity_definition'}, 'Typed particle bulk fields differ')
    _vector(packet['bulk']['COM_body_m'], 3, 'body-frame COM')
    _require(packet['bulk']['metric_inventory'] == 'ALL_ORIGINAL_IDS_INCLUDING_INACTIVE_AND_OUTSIDE'
             and packet['bulk']['velocity_definition'] == 'v_particle-v_body-omega_body_cross_(x_particle-x_body)'
             and packet['outside_carrier_is_verified_loss'] is False
             and packet['delivered_credit_assigned'] is False
             and packet['source_authentication'] == 'CALLER_OWNS_NATIVE_READER_AND_LIVE_SOURCE_VERIFICATION',
             'Particle packet cannot assign loss/delivery or authenticate its own native truth')
    _number(packet['maximum_solid_union_depth_m'], 'solid union depth', 0.)
    quality = packet['quality']
    _require(set(quality) == {'MPM_F_finite', 'MPM_positive_detF', 'MPM_minimum_detF',
                 'finite_arrays', 'material_quality_passed'}, 'Typed particle quality fields differ')
    for key in ('finite_arrays', 'material_quality_passed'):
        _require(type(quality[key]) is bool, 'Explicit particle quality boolean required')
    if expected['material_kind'] == 'sand':
        for key in ('MPM_F_finite', 'MPM_positive_detF'):
            _require(type(quality[key]) is bool, 'Explicit MPM quality required')
        if quality['MPM_F_finite']:
            minimum = _number(quality['MPM_minimum_detF'], 'minimum detF')
            _require(quality['MPM_positive_detF'] == (minimum > 0), 'MPM detF quality differs')
        else:
            _require(quality['MPM_minimum_detF'] is None and quality['MPM_positive_detF'] is False,
                     'Nonfinite MPM quality cannot assign a positive determinant')
        _require(quality['finite_arrays'] is quality['MPM_F_finite'], 'MPM finite-array quality differs')
    else:
        _require(all(quality[key] is None for key in ('MPM_F_finite', 'MPM_positive_detF', 'MPM_minimum_detF')),
                 'SPH packet cannot substitute MPM state')
    _require((counts['deep_solid_count'] > 0) == (packet['maximum_solid_union_depth_m'] > .001),
             'Particle maximum depth/count disagree')
    expected_quality = bool(counts['inactive_count'] == counts['deep_solid_count'] ==
        counts['outside_effective_domain_count'] == 0 and quality['finite_arrays']
        and quality['MPM_positive_detF'] is not False)
    _require(quality['material_quality_passed'] is expected_quality, 'Particle aggregate quality is inconsistent')
    if require_retained:
        _require(counts['active_count'] == counts['retained_count'] == n
                 and counts['inactive_count'] == counts['outside_carrier_count'] == counts['deep_solid_count'] ==
                 counts['outside_effective_domain_count'] == counts['domain_warning_count'] == 0
                 and quality['finite_arrays'] is True and quality['material_quality_passed'] is True
                 and quality['MPM_positive_detF'] is not False and packet['maximum_solid_union_depth_m'] <= .001
                 and mass['inactive_mass_kg'] == mass['outside_carrier_mass_kg'] == 0.
                 and math.isclose(mass['retained_mass_kg'], total_mass, rel_tol=0., abs_tol=1e-10),
                 'Original particles must all remain active/retained with valid depth/domain/deformation')
    return packet


def window_rest(rows, ending_s, contract, *, dt_s=DT_S):
    """Assess the fixed .5-second inclusive window, including both endpoints.

    At .0025s this needs 201 distinct consecutive native observations. Global
    substeps must advance by one constant positive stride. Local ring indices
    are retained but may wrap. No coefficient-of-variation/repeat verdict is
    inferred from a single window.
    """
    _require(dt_s == DT_S, 'Particle rest uses the fixed actual outer timestep')
    _number(ending_s, 'rest ending time', 0.)
    end = round(ending_s/dt_s); intervals = round(REST_LIMITS['window_s']/dt_s)
    _require(abs(ending_s-end*dt_s) <= 1e-9 and end >= intervals, 'Rest ending clock is not a complete step grid')
    selected = []; failures = []
    try:
        for row in rows:
            now = _number(row['simulation_s'], 'row time', 0.)
            if ending_s-REST_LIMITS['window_s']-1e-10 <= now <= ending_s+1e-10:
                selected.append(row)
        _require(len(selected) == intervals+1, 'Incomplete particle rest window')
        context = _window_validation_context(contract)
        packets = []
        for row, step in zip(selected, range(end-intervals, end+1)):
            _require(type(row['native_scene_step']) is int and row['native_scene_step'] == step
                     and abs(row['simulation_s']-step*dt_s) <= 1e-9, 'Nonconsecutive native particle rest clock')
            packets.append(_validate_particle_packet_with_context(row['particle_material'], context, row, require_retained=True))
        strides = [b['native_clock'][2]-a['native_clock'][2] for a, b in zip(packets, packets[1:])]
        _require(len(set(strides)) == 1 and strides[0] > 0, 'Native substeps are missing/nonconsecutive')
        com = np.asarray([p['bulk']['COM_body_m'] for p in packets])
        metrics = dict(relative_RMS_max_m_s=max(p['bulk']['relative_RMS_m_s'] for p in packets),
            relative_q95_max_m_s=max(p['bulk']['relative_q95_m_s'] for p in packets),
            COM_excursion_max_m=float(np.linalg.norm(com-com[0], axis=1).max()))
        for key, value in metrics.items():
            if value > REST_LIMITS[key]: failures.append(key+' exceeds the fixed rest criterion')
    except (ContractError, KeyError, TypeError, ValueError, OverflowError) as exc:
        metrics = None; failures.append(str(exc))
    return dict(schema_version='particle-episode-window-rest-v1', passed=not failures,
        ending_s=ending_s, window_s=REST_LIMITS['window_s'], dt_s=dt_s,
        expected_rows=intervals+1, observed_rows=len(selected), limits=copy.deepcopy(REST_LIMITS),
        metrics=metrics, failures=failures, repeatability_verified=False,
        source_authentication='CALLER_OWNS_NATIVE_READER_AND_LIVE_SOURCE_VERIFICATION')
