"""Read-only, source-pinned SPH/MPM material inventory for one native entity.

Importing this module does not import or initialize a native runtime. The caller
owns the live scene and authenticates the same-clock body observation. No setter,
solver replacement, collision query, stepping or particle reset occurs here.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np

from physics.scene_assets import interior_membership
from transport_challenge.contracts import ContractError


NATIVE_SOURCE_PINS = {
    'engine/entities/particle_entity.py': '9c2c5d2a0f213e57fa297012a8884c0ff072fd4ef8dbd8d2d926f589497e6178',
    'engine/entities/sph_entity.py': 'f19fb86ade0217c80370c5dc44873723b8005e7dee48eeb3a37b7e9df167794d',
    'engine/entities/mpm_entity.py': 'da59d325d33d3d96263f9c1029061038e6827bf0b4f5d6ab7f9fc896f01d78f2',
    'engine/solvers/sph_solver.py': '7e06ddbdafcc3f607798b4dba16da653a859bc0b66fbbc1ea9cde1f17db69849',
    'engine/solvers/mpm_solver.py': '3f4e29b668a77c10e276f499e9d40259001bf61ae448a65061874b1efd5ccd5f',
    'engine/materials/SPH/liquid.py': 'b836d7a50d00ea6f847f541744092cf08bcb8274a00535e3e57bcf32e1e20f5c',
    'engine/materials/SPH/base.py': 'b6b3b077b5085afe08be8eb125eb10d87ec880e196f55524863bd4aa9e10fec2',
    'engine/materials/MPM/sand.py': 'fc286b1d1f7731d7559cb60b042c282dad9d9c8249f05cf897a6117306a8ecf8',
    'engine/materials/MPM/base.py': '9098959d12e5353842977c895e52473e2aaf0600cd7ac9c1c91743d468d9af89',
    'engine/couplers/legacy_coupler.py': '29b5778655bca346d74a1f54dff07a5188db22705ea6b384b09c35f202483b9b',
    'options/solvers.py': '2f799beaf1a5acb7aa55edcc063187f3e6a384ce1ea85092c2ee72f4e202e536',
}
GETTERS = ('get_particles_pos', 'get_particles_vel', 'get_particles_active', 'get_mass')
SPARSE_MPM_SOURCE_SHA256 = '90bba1570f08dae3a54f7a42bfbd578f5d43bdf7071ab1303bb77f0fd92e8106'
SPARSE_MPM_SOURCE_PATH = (Path(__file__).resolve().parents[1] /
    'qa/all_material_physics_20261031/performance_branch/private_sparse_mpm_candidate.py')
SPARSE_MPM_NATIVE_PINS = {
    'engine/solvers/mpm_solver.py': NATIVE_SOURCE_PINS['engine/solvers/mpm_solver.py'],
    'engine/couplers/legacy_coupler.py': NATIVE_SOURCE_PINS['engine/couplers/legacy_coupler.py'],
    'options/solvers.py': NATIVE_SOURCE_PINS['options/solvers.py'],
    'engine/scene.py': 'd4b6b2dd11d6745147daaa3fc47e2f172d832c9bfc8c6b68576bb1d4415fce3c',
    'engine/simulator.py': 'b385b363dfb56512ae189d3ad86fd9bd75acc7209e2cf4b9c934ec76e98073da',
}
CPIC_CACHE_SOURCE_PATH = (Path(__file__).resolve().parents[1] /
    'qa/all_material_physics_20261031/cpic_cache_branch/private_cpic_lattice_cache_candidate.py')
CPIC_CACHE_SOURCE_SHA256 = 'ffa5e4f971185f845ea308b43601713b2e4be9d24c28045aecba7cd2b63f8b96'
CPIC_CACHE_NATIVE_PINS = dict(SPARSE_MPM_NATIVE_PINS, **{
    'engine/solvers/rigid/rigid_solver.py':'e076854af17262323041544ff692d366402a700357053618c8d89fed1e86b19f',
    'utils/sdf.py':'2fdb6fd047d2acb09ec0c8ecab720d0909d9a59b73af47be643a4985a3e08068'})
CPIC_CACHE_METHODS = ('init_grid_fields','substep_pre_coupling','p2g','_cpic_build_cache','outer_safety','cache_receipt')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _array_hash(value):
    a = np.ascontiguousarray(value)
    return _digest({'dtype': a.dtype.str, 'shape': list(a.shape),
                    'data_sha256': hashlib.sha256(a.tobytes()).hexdigest()})


def _require(condition, message):
    if not condition:
        raise ContractError(message)


def _integer(value):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)) and value >= 0,
             'Expected nonnegative native integer')
    return int(value)


def _host(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _sync_native():
    import quadrants as qd
    qd.sync()


def _clock(scene):
    step = _integer(scene.t); now = float(scene.cur_t)
    _require(math.isfinite(now) and now >= 0., 'Invalid native material observation clock')
    return (step, now, _integer(scene.sim.cur_substep_global), _integer(scene.sim.cur_substep_local))


def _verify_sparse_mpm_sources(entity, solver_cls, native_base, extra, root):
    """One exact private implementation, with explicit additional source binding.

    This branch authenticates the existing real MPMEntity. It creates no entity,
    imports no solver runtime and changes no public-getter/read/ledger behavior.
    """
    path = str(Path(inspect.getsourcefile(solver_cls)).resolve())
    _require(solver_cls.__name__ == 'ScenePrivateSparseMPM'
             and solver_cls.__qualname__ == 'install_preentity.<locals>.ScenePrivateSparseMPM'
             and solver_cls.__bases__ == (native_base,)
             and Path(path) == SPARSE_MPM_SOURCE_PATH.resolve(), 'Unreviewed private sparse MPM class/source')
    explicit = {str(Path(p).resolve()): value for p, value in extra.items()}
    _require(explicit.get(path) == SPARSE_MPM_SOURCE_SHA256
             and hashlib.sha256(Path(path).read_bytes()).hexdigest() == SPARSE_MPM_SOURCE_SHA256,
             'Exact sparse MPM factory must be explicitly source-bound')
    for relative, expected in SPARSE_MPM_NATIVE_PINS.items():
        _require(explicit.get(str((root / relative).resolve())) == expected,
                 'Sparse MPM requires its complete explicit native source closure: ' + relative)
    solver = entity.solver; scene = entity.scene; sim = scene.sim; coupler = sim.coupler
    cls = type(coupler)
    bases = [c for c in cls.__mro__ if c.__name__ == 'LegacyCoupler'
             and c.__module__ == 'genesis.engine.couplers.legacy_coupler']
    _require(len(bases) == 1 and cls.__bases__ == (bases[0],)
             and cls.__name__ == 'ScenePrivateSparseLegacyCoupler'
             and cls.__qualname__ == 'install_preentity.<locals>.ScenePrivateSparseLegacyCoupler'
             and Path(inspect.getsourcefile(cls)).resolve() == Path(path),
             'Sparse MPM requires the same source-bound private coupler')
    _require(solver.sim is sim and sim.mpm_solver is solver and coupler.sim is sim
             and coupler.mpm_solver is solver and solver._coupler is coupler
             and scene.rigid_solver is sim.rigid_solver and coupler.rigid_solver is sim.rigid_solver
             and sum(s is solver for s in sim._solvers) == 1
             and len(sim._active_solvers) == 2
             and {id(s) for s in sim._active_solvers} == {id(solver), id(sim.rigid_solver)},
             'Private sparse same-scene solver/coupler aliases or active inventory differ')
    _require(sim.requires_grad is False and sim._B == 1 and sim.substeps_local == 1
             and sim.substeps == 100 and abs(sim.dt - .0025) <= 1e-15
             and solver._constraints_initialized is False
             and sim.cur_substep_local == 0 and sim.cur_substep_global == scene.t * sim.substeps,
             'Private sparse material reader requires the fixed forward outer-endpoint phase')
    _require('outer_safety' not in vars(solver), 'Private sparse overflow method instance override')
    safety = solver.outer_safety()
    _require(type(safety) is dict and safety.get('sticky_overflow') == 0
             and safety.get('dirty_count') == 0 and safety.get('dirty_capacity') == 27 * solver.n_particles,
             'Sparse overflow or dirty lifecycle is not clean at the outer endpoint')


def _cpic_cache_endpoint(solver, scene):
    """Actual cache lifecycle and native numerical settings, no cached physics."""
    _require(not any(name in vars(solver) for name in CPIC_CACHE_METHODS),
             'CPIC cache method instance override')
    receipt = solver.cache_receipt()
    glob = _integer(scene.sim.cur_substep_global)
    _require(type(receipt) is dict and receipt['cache_sticky_overflow'] == 0
             and receipt['sticky_overflow'] == 0 and receipt['dirty_count'] == 0
             and receipt['cache_capacity'] == receipt['dirty_capacity'] == 27*solver.n_particles
             and 0 <= receipt['cache_current_nodes'] <= receipt['cache_capacity']
             and receipt['actual_geometry_count'] == scene.rigid_solver.n_geoms
             and receipt['normal_field_shape'] == [receipt['cache_capacity'], scene.rigid_solver.n_geoms]
             and receipt['normal_scalar_bytes'] == 8 and receipt['current_global_substep'] == glob,
             'CPIC cache capacity, precision, overflow or current phase differs')
    _require(float(scene.rigid_options.tolerance) == 1e-12
             and float(np.asarray(scene.rigid_solver.rigid_info.tolerance.to_numpy()).item()) == 1e-12,
             'Explicit cache profile requires actual native rigid tolerance1e-12')
    if glob:
        _require(receipt['last_cache_global_substep'] == glob-1
                 and receipt['last_cache_local_f'] == 0,
                 'CPIC cache was not rebuilt in the last actual internal interval')
    return receipt


def _verify_cpic_cache_sources(entity, solver_cls, native_base, extra, root):
    """Exact FFa ->90b ->native class chain and the unchanged90b coupler."""
    _require(len(solver_cls.__bases__) == 1, 'CPIC cache needs one exact sparse base')
    sparse_base = solver_cls.__bases__[0]
    path = CPIC_CACHE_SOURCE_PATH.resolve()
    defining = sys.modules.get(solver_cls.__module__)
    _require(solver_cls.__name__ == 'ScenePrivateCPICLatticeCacheMPM'
             and solver_cls.__qualname__ == 'install_preentity.<locals>.ScenePrivateCPICLatticeCacheMPM'
             and sparse_base.__bases__ == (native_base,)
             and defining is not None and Path(defining.__file__).resolve() == path
             and Path(inspect.getsourcefile(solver_cls)).resolve() == path,
             'Unreviewed private CPIC cache class/source')
    explicit = {str(Path(p).resolve()): h for p,h in extra.items()}
    _require(explicit.get(str(path)) == CPIC_CACHE_SOURCE_SHA256
             and hashlib.sha256(path.read_bytes()).hexdigest() == CPIC_CACHE_SOURCE_SHA256,
             'Exact CPIC cache provider must be explicitly source-bound')
    for relative, expected in CPIC_CACHE_NATIVE_PINS.items():
        _require(explicit.get(str((root/relative).resolve())) == expected,
                 'CPIC cache requires complete explicit native closure: '+relative)
    for name in CPIC_CACHE_METHODS:
        method = inspect.unwrap(getattr(solver_cls,name))
        _require(Path(inspect.getsourcefile(method)).resolve() == path
                 and method.__qualname__ == solver_cls.__qualname__+'.'+name,
                 'CPIC cache method definition changed: '+name)
    _verify_sparse_mpm_sources(entity,sparse_base,native_base,extra,root)
    _cpic_cache_endpoint(entity.solver,entity.scene)


def _verify_native_sources(entity, kind, extra):
    expected = ('SPHEntity', 'sph_entity', 'SPHSolver', 'sph_solver') if kind == 'process_liquid' else (
        'MPMEntity', 'mpm_entity', 'MPMSolver', 'mpm_solver')
    cls = type(entity)
    _require(cls.__name__ == expected[0] and cls.__module__ == 'genesis.engine.entities.' + expected[1],
             'Unreviewed native material entity class')
    root = Path(inspect.getsourcefile(cls)).resolve().parents[2]
    pins = {str(root / key): value for key, value in NATIVE_SOURCE_PINS.items()}
    for path, sha in extra.items():
        _require(type(path) is str and type(sha) is str and len(sha) == 64,
                 'Malformed additional solver source binding')
        resolved = str(Path(path).resolve())
        _require(resolved not in pins or pins[resolved] == sha, 'Additional pin cannot replace a reviewed native source')
        pins[resolved] = sha
    base = [c for c in type(entity.solver).__mro__ if c.__name__ == expected[2]
            and c.__module__ == 'genesis.engine.solvers.' + expected[3]]
    _require(len(base) == 1, 'Native material solver base differs')
    solver_cls = type(entity.solver)
    if solver_cls is not base[0]:
        if kind == 'sand' and solver_cls.__name__ == 'ScenePrivateSparseMPM':
            _verify_sparse_mpm_sources(entity, solver_cls, base[0], extra, root)
        elif kind == 'sand' and solver_cls.__name__ == 'ScenePrivateCPICLatticeCacheMPM':
            _verify_cpic_cache_sources(entity,solver_cls,base[0],extra,root)
        else:
            _require(kind == 'process_liquid' and solver_cls.__name__ == 'CupDensityMapSPHSolver',
                     'Only explicitly pinned private density-map SPH subclass is supported')
            path = str(Path(inspect.getsourcefile(solver_cls)).resolve())
            _require(path in pins, 'Private solver implementation is not explicitly source-bound')
    for path, sha in pins.items():
        _require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha,
                 'Reviewed native particle source changed: ' + path)
    return pins


class _SolidUnion:
    """Exact nominal-box membership and binary64 exposed-union depth.

    Rational box endpoints select exposed rectangles without overlap seams.
    Only the distance calculation converts those endpoints to binary64.
    """
    def __init__(self, boxes):
        _require(isinstance(boxes, (list, tuple)) and len(boxes) > 0, 'Explicit solid boxes required')
        bounds = []
        for size, center in boxes:
            _require(len(size) == len(center) == 3 and all(math.isfinite(float(v)) for v in (*size, *center))
                     and all(float(v) > 0 for v in size), 'Invalid vessel solid primitive')
            bounds.append(tuple((Fraction.from_float(float(c)) - Fraction.from_float(float(s)) / 2,
                                 Fraction.from_float(float(c)) + Fraction.from_float(float(s)) / 2)
                                for s, c in zip(size, center)))
        cuts = [sorted({b[a][j] for b in bounds for j in (0, 1)}) for a in range(3)]
        shape = tuple(len(c) - 1 for c in cuts); occupied = np.zeros(shape, bool)
        for index in np.ndindex(shape):
            midpoint = [(cuts[a][index[a]] + cuts[a][index[a] + 1]) / 2 for a in range(3)]
            occupied[index] = any(all(lo <= p <= hi for p, (lo, hi) in zip(midpoint, b)) for b in bounds)
        low = []; high = []
        for index in np.argwhere(occupied):
            for axis in range(3):
                for direction in (-1, 1):
                    other = index.copy(); other[axis] += direction
                    if other[axis] < 0 or other[axis] >= shape[axis] or not occupied[tuple(other)]:
                        lo = [cuts[a][index[a]] for a in range(3)]
                        hi = [cuts[a][index[a] + 1] for a in range(3)]
                        lo[axis] = hi[axis] = cuts[axis][index[axis] + (direction > 0)]
                        low.append(list(map(float, lo))); high.append(list(map(float, hi)))
        self.patch_low = np.asarray(low, float); self.patch_high = np.asarray(high, float)
        def inward(value, lower):
            v = float(value); exact = Fraction.from_float(v)
            return np.nextafter(v, np.inf if lower else -np.inf) if (
                (lower and exact < value) or (not lower and exact > value)) else v
        self.low = np.asarray([[inward(pair[0], True) for pair in b] for b in bounds])
        self.high = np.asarray([[inward(pair[1], False) for pair in b] for b in bounds])

    def read(self, points):
        inside = np.any(np.all((points[:, None, :] >= self.low[None, :, :]) &
                               (points[:, None, :] <= self.high[None, :, :]), axis=2), axis=1)
        depth = np.zeros(len(points), float); selected = np.flatnonzero(inside)
        for start in range(0, len(selected), 128):
            ids = selected[start:start + 128]; p = points[ids, None, :]
            nearest = np.minimum(np.maximum(p, self.patch_low[None]), self.patch_high[None])
            depth[ids] = np.sqrt(np.min(np.sum((nearest - p) ** 2, axis=2), axis=1))
        return inside, depth, inside & (depth > .001)


class ParticleMaterialReader:
    """Bind one existing entity and read complete native-order particle arrays.

    ``read(body_observation, include_rows=False)`` requires the current native
    step/time and ``body_world`` in the caller's actual body observation. Arrays
    include ``pos(N,3)``, ``vel(N,3)``, ``active(N)``, immutable ``mass_kg(N)`` and
    ``native_particle_id(N)``. No inactive or outside ID is dropped or zeroed.
    The caller archives arrays and applies mission/ledger acceptance separately.
    """
    def __init__(self, scene, entity, *, material_kind, source_id, geometry,
                 extra_source_pins=None, domain_warning_margin_m=.025):
        _require(material_kind in ('process_liquid', 'sand'), 'Explicit process_liquid or sand kind required')
        _require(type(source_id) is str and source_id.strip(), 'Explicit persistent source ID required')
        _require(math.isfinite(domain_warning_margin_m) and domain_warning_margin_m >= 0.,
                 'Invalid domain warning margin')
        self.scene = scene; self.entity = entity; self.solver = entity.solver
        self.kind = material_kind; self.source_id = source_id
        self.geometry = copy.deepcopy(geometry); self.union = _SolidUnion(self.geometry['component_boxes'])
        self.warning_margin = float(domain_warning_margin_m)
        self.source_pins = _verify_native_sources(entity, material_kind, extra_source_pins or {})
        self.source_pins[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        import physics.scene_assets as assets
        self.source_pins[str(Path(assets.__file__).resolve())] = hashlib.sha256(Path(assets.__file__).read_bytes()).hexdigest()
        self._getters = tuple(getattr(type(entity), name) for name in GETTERS)
        _require(all(name not in vars(entity) for name in GETTERS), 'Overridden native particle getter instance')
        self.start = _integer(entity.particle_start); self.end = _integer(entity.particle_end)
        self.n = _integer(entity.n_particles); self.entity_idx = _integer(entity.idx)
        _require(self.n > 0 and self.end - self.start == self.n and entity.scene is scene
                 and entity.is_built and scene.is_built and scene.n_envs == 0 and self.solver._B == 1,
                 'Built unbatched nonempty particle entity required')
        _require(self.solver.sim is scene.sim and getattr(scene.sim,
                 'sph_solver' if self.kind == 'process_liquid' else 'mpm_solver') is self.solver,
                 'Native entity solver is not the actual live scene solver')
        self._material = entity.material; self._material_record = copy.deepcopy(entity.material.model_dump())
        self._solver_count = _integer(self.solver.n_particles)
        self._entity_name = str(entity.name); self._solver_class = type(self.solver)
        self._ids = np.arange(self.start, self.end, dtype=np.int64)
        self._lower = np.asarray(self.solver._lower_bound, float).copy()
        self._upper = np.asarray(self.solver._upper_bound, float).copy()
        self._padding = float(self.solver.boundary_padding) if self.kind == 'sand' else 0.
        _require(self._lower.shape == self._upper.shape == (3,) and np.isfinite(self._lower).all()
                 and np.isfinite(self._upper).all() and math.isfinite(self._padding) and self._padding >= 0
                 and np.all(self._lower + self._padding < self._upper - self._padding), 'Invalid actual native domain')
        self._scale = float(self.solver._particle_volume_scale) if self.kind == 'sand' else 1.
        _require(self._scale == (1000. if self.kind == 'sand' else 1.), 'Pinned native SI mass scale changed')
        self._volume = float(self.solver._particle_volume)
        _require(math.isfinite(self._volume) and self._volume > 0, 'Invalid actual native particle volume')
        clock = _clock(scene); _sync_native()
        first = self._native_info()
        _require(_clock(scene) == clock, 'Native clock changed during material binding')
        self._mass_native = first['mass_native'].copy(); self._weights = self._mass_native.astype(float) / self._scale
        self._material_indices = first.get('material_idx'); self._free = first.get('free')
        _require(self._free is None or self._free.all(), 'Original MPM source includes fixed particles')
        self._rho0 = first.get('rho0')
        self._public_mass = None
        if self.kind == 'sand':
            self._public_mass = self._check_public_mass()
        _require(_clock(scene) == clock, 'Public mass getter changed native clock')
        self._contract = dict(schema_version='native-particle-material-mass-contract-v1', material_kind=self.kind,
            source_id=source_id, native_entity_name=self._entity_name, native_entity_idx=self.entity_idx,
            native_solver_class=type(self.solver).__name__, native_particle_start=self.start, particle_count=self.n,
            native_particle_ids=self._ids.tolist(), native_mass_scale=self._scale,
            actual_native_mass_sha256=_array_hash(self._mass_native), particle_masses_kg=self._weights.tolist(),
            initial_total_mass_kg=float(math.fsum(self._weights)), actual_native_particle_volume=self._volume,
            physical_particle_volume_m3=self._volume / self._scale, native_material_settings=self._material_record,
            native_material_settings_sha256=_digest(self._material_record), geometry_sha256=_digest(self.geometry),
            public_entity_mass_kg=self._public_mass,
            public_mass_status='VERIFIED_INITIAL_NATIVE_GET_MASS' if self.kind == 'sand' else
                'UNAVAILABLE_PINNED_SPH_GET_MASS_FIELD_BUG',
            public_mass_note=None if self.kind == 'sand' else
                'Genesis1.3.2 SPH get_mass references nonexistent particles.m. Actual particles_info.mass is authoritative.',
            native_lower_bound_m=self._lower.tolist(), native_upper_bound_m=self._upper.tolist(),
            native_boundary_padding_m=self._padding, effective_lower_bound_m=(self._lower + self._padding).tolist(),
            effective_upper_bound_m=(self._upper - self._padding).tolist(), domain_warning_margin_m=self.warning_margin,
            containment='Existing interior_membership, point centers, radius0, tolerance0, actual body-local transform',
            deep_solid_guard='Nominal exact closed-box union, Euclidean exposed-face depth>0.001m, separate from containment',
            MPM_deformation_gate='Actual current-ring F finite and detF>0. F_tmp is not used.' if self.kind == 'sand' else None,
            source_hashes=copy.deepcopy(self.source_pins), initial_native_clock=list(clock),
            logical_identity='source_id:local_index', native_identity='(native_entity_idx,native_particle_id)',
            particle_reset_or_inactive_omission_permitted=False)
        self.contract_sha256 = _digest(self._contract); self.reads = 0
        if type(self.solver).__name__ == 'ScenePrivateCPICLatticeCacheMPM':
            self._cache_type_chain = type(self.solver).__mro__
            self._cache_methods = tuple(getattr(type(self.solver),n) for n in CPIC_CACHE_METHODS)
            self._cache_initial = _cpic_cache_endpoint(self.solver,self.scene)
            self._cache_invocation_origin = self._cache_initial['cache_invocations']-clock[2]
            _require(self._cache_invocation_origin >= 0, 'Invalid cache build-prefix invocation count')

    @property
    def contract(self):
        return copy.deepcopy(self._contract)

    def verify_source_bindings(self):
        for path, sha in self.source_pins.items():
            _require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha,
                     'Bound material source changed: ' + path)
        self._topology()
        return copy.deepcopy(self.source_pins)

    def _topology(self):
        e = self.entity; s = self.solver
        _require(e.scene is self.scene and e.solver is s and type(s) is self._solver_class
                 and s.sim is self.scene.sim and getattr(self.scene.sim,
                     'sph_solver' if self.kind == 'process_liquid' else 'mpm_solver') is s
                 and e.material is self._material and e.material.model_dump() == self._material_record
                 and e.idx == self.entity_idx and e.name == self._entity_name
                 and e.particle_start == self.start and e.particle_end == self.end and e.n_particles == self.n
                 and s.n_particles == self._solver_count and s._B == 1 and self.scene.n_envs == 0,
                 'Native material topology, count, entity or material binding changed')
        if self._solver_class.__name__ == 'ScenePrivateCPICLatticeCacheMPM':
            _require(type(s).__mro__ == self._cache_type_chain
                     and tuple(getattr(type(s),n) for n in CPIC_CACHE_METHODS) == self._cache_methods,
                     'CPIC cache source-owned method or class chain changed')
            receipt = _cpic_cache_endpoint(s,self.scene)
            _require(receipt['cache_invocations'] == self._cache_invocation_origin+self.scene.sim.cur_substep_global,
                     'CPIC cache invocation count differs from actual internal integration count')
        _require(tuple(getattr(type(e), name) for name in GETTERS) == self._getters
                 and all(name not in vars(e) for name in GETTERS), 'Native material getter was replaced')
        _require(np.array_equal(s._lower_bound, self._lower) and np.array_equal(s._upper_bound, self._upper)
                 and float(s._particle_volume) == self._volume
                 and (self.kind != 'sand' or (float(s._particle_volume_scale) == self._scale
                     and float(s.boundary_padding) == self._padding)), 'Actual native mass/domain settings changed')

    def _field(self, name):
        a = np.asarray(getattr(self.solver.particles_info, name).to_numpy()).copy()
        _require(a.shape == (self._solver_count,), 'Native particle-info field layout changed: ' + name)
        return a[self.start:self.end].copy()

    def _native_info(self):
        mass = self._field('mass')
        _require(mass.dtype.kind == 'f' and np.isfinite(mass).all() and np.all(mass > 0.),
                 'Nonfinite/nonpositive actual per-particle mass')
        result = {'mass_native': mass}
        if self.kind == 'process_liquid':
            rho = self._field('rho')
            _require(rho.dtype.kind == 'f' and np.isfinite(rho).all() and np.all(rho > 0.), 'Invalid actual rest density')
            expected = rho.astype(float) * self._volume
            tolerance = 4 * np.finfo(mass.dtype).eps * np.maximum(expected, np.finfo(mass.dtype).tiny)
            _require(np.all(np.abs(mass.astype(float) - expected) <= tolerance),
                     'SPH actual mass disagrees with actual rest density times native particle volume')
            _require(np.all(rho == float(self.entity.material.rho)), 'SPH native rest density differs from material')
            result['rho0'] = rho
        else:
            indices = self._field('material_idx'); free = self._field('free')
            _require(indices.dtype.kind in 'iu' and np.all(indices == self.entity.material.idx),
                     'MPM native per-particle material ownership changed')
            _require(free.dtype.kind == 'b', 'MPM native free flag layout changed')
            result.update(material_idx=indices, free=free)
        return result

    def _check_public_mass(self):
        value = _host(self.entity.get_mass())
        _require(value.shape == (1,) and np.isfinite(value).all(), 'Invalid public MPM get_mass result')
        measured = float(value[0]); total = math.fsum(self._weights)
        _require(abs(measured - total) <= max(1e-7, 2e-5 * total), 'Public MPM mass disagrees with native per-ID SI sum')
        return measured

    def _deformation(self):
        value = np.asarray(self.solver.particles.F.to_numpy()).copy()
        shape = (self.scene.sim.substeps_local + 1, self._solver_count, 1, 3, 3)
        _require(value.shape == shape and value.dtype.kind == 'f', 'Native MPM current-ring F layout changed')
        frame = _integer(self.scene.sim.cur_substep_local)
        _require(frame < shape[0], 'Native MPM current-ring index outside storage')
        F = value[frame, self.start:self.end, 0].copy()
        with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
            det = np.linalg.det(F.astype(float))
        return F, det

    def read(self, body_observation, *, include_rows=False):
        self._topology(); clock = _clock(self.scene)
        _require(isinstance(body_observation, dict)
                 and type(body_observation.get('native_scene_step')) is int
                 and type(body_observation.get('simulation_s')) in (int, float)
                 and body_observation.get('native_scene_step') == clock[0]
                 and body_observation.get('simulation_s') == clock[1] and 'body_world' in body_observation,
                 'Material classification needs actual same-clock body observation')
        _require(type(include_rows) is bool, 'include_rows must be an explicit Boolean')
        body = copy.deepcopy(body_observation['body_world'])
        p = np.asarray(body['position_m'], float); q = np.asarray(body['quaternion_wxyz'], float)
        _require(body.get('frame') == 'world' and p.shape == (3,) and q.shape == (4,)
                 and np.isfinite(p).all() and np.isfinite(q).all() and abs(np.linalg.norm(q) - 1.) <= 1e-6,
                 'Invalid actual body world pose')
        _sync_native()
        pos = _host(self.entity.get_particles_pos()); vel = _host(self.entity.get_particles_vel())
        active = _host(self.entity.get_particles_active()); info = self._native_info()
        F, detF = self._deformation() if self.kind == 'sand' else (None, None)
        _require(pos.shape == vel.shape == (self.n, 3) and pos.dtype.kind == vel.dtype.kind == 'f'
                 and np.isfinite(pos).all() and np.isfinite(vel).all(), 'Invalid actual native position/velocity arrays')
        _require(active.shape == (self.n,) and active.dtype.kind == 'b', 'Invalid actual native active mask')
        _require(info['mass_native'].dtype == self._mass_native.dtype and
                 np.array_equal(info['mass_native'], self._mass_native), 'Original native particle mass changed')
        for key, initial in (('material_idx', self._material_indices), ('free', self._free), ('rho0', self._rho0)):
            if initial is not None:
                _require(np.array_equal(info[key], initial), 'Original particle info changed: ' + key)
        _require(_clock(self.scene) == clock, 'Native clock changed during particle readback')
        self._topology()
        w, x, y, z = q / np.linalg.norm(q)
        rotation = np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        local = (pos.astype(float) - p) @ rotation
        retained = interior_membership(local, self.geometry, radius_m=0., tolerance_m=0.)
        solid, depth, deep = self.union.read(local)
        low = self._lower + self._padding; high = self._upper - self._padding
        domain = np.any((pos < low) | (pos > high), axis=1)
        warning = np.any((pos < low + self.warning_margin) | (pos > high - self.warning_margin), axis=1)
        arrays = dict(pos=pos, vel=vel, active=active, mass_kg=self._weights.copy(),
            native_particle_id=self._ids.copy(), mass_native=info['mass_native'], local_xyz=local,
            inside_carrier=retained, inside_solid_union=solid, solid_union_depth_m=depth,
            deep_solid=deep, outside_effective_domain=domain, domain_warning=warning)
        arrays.update({key: value for key, value in info.items() if key != 'mass_native'})
        F_finite = F is None or bool(np.isfinite(F).all() and np.isfinite(detF).all())
        F_positive = F is None or bool(F_finite and np.all(detF > 0))
        if F is not None:
            arrays.update(F=F, detF=detF)
        total = float(math.fsum(self._weights)); inactive = ~active
        summary = dict(schema_version='native-particle-material-observation-v1', native_scene_step=clock[0],
            simulation_s=clock[1], native_clock=list(clock), particle_material_contract_sha256=self.contract_sha256,
            material_kind=self.kind, source_id=self.source_id, native_entity_idx=self.entity_idx,
            persistent_id_count=self.n, initial_mass_kg=total, native_mass_kg=total,
            mass_native_sha256=_array_hash(info['mass_native']), identity_and_mass_unchanged=True,
            active_count=int(active.sum()), inactive_count=int(inactive.sum()),
            inactive_mass_kg=float(self._weights[inactive].sum()), retained_count=int(retained.sum()),
            retained_mass_kg=float(self._weights[retained].sum()), outside_carrier_count=int((~retained).sum()),
            outside_carrier_mass_kg=float(self._weights[~retained].sum()),
            deep_solid_count=int(deep.sum()), maximum_solid_union_depth_m=float(depth.max(initial=0.)),
            outside_effective_domain_count=int(domain.sum()), domain_warning_count=int(warning.sum()),
            maximum_speed_m_s=float(np.linalg.norm(vel, axis=1).max(initial=0.)),
            public_entity_mass_kg=self._public_mass, public_mass_status=self._contract['public_mass_status'],
            finite_arrays=F_finite, native_clock_unchanged=True,
            MPM_F_finite=F_finite if self.kind == 'sand' else None,
            MPM_positive_detF=F_positive if self.kind == 'sand' else None,
            MPM_minimum_detF=float(detF.min()) if detF is not None and F_finite else None,
            material_quality_passed=bool(active.all() and not deep.any() and not domain.any() and F_finite and F_positive),
            outside_carrier_is_verified_loss=False, delivered_credit_assigned=False,
            native_particle_order='Persistent public-getter order, not SPH reordered storage')
        result = {'arrays': arrays, 'summary': summary}
        if include_rows:
            result['materials'] = [dict(id=f'{self.source_id}:{i}', entity_idx=self.entity_idx,
                native_particle_id=int(self._ids[i]), mass_kg=float(self._weights[i]),
                mass_basis='initial_actual_native_particle_mass_SI', material_kind=self.kind,
                particle_material_contract_sha256=self.contract_sha256,
                position_world_m=pos[i].tolist(), velocity_world_m_s=vel[i].tolist(),
                active=bool(active[i]), inside_carrier=bool(retained[i])) for i in range(self.n)]
        self.reads += 1
        return result
