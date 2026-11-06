"""Bounded, append-only host archive for ParticleMaterialReader observations.

This module never imports a native runtime. It records supplied arrays, not a
simulation replay or a material/quantity verdict. ``append(read_result, body)``
returns a JSON-safe summary and a *pending* row reference. The reference becomes
durable only when its chunk appears in manifest.json with matching file hashes.
All N particles remain in native order, including inactive and outside IDs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

from transport_challenge.contracts import ContractError


SCHEMA = 'native-particle-archive-v1'
STATIC = {'native_particle_id', 'mass_kg', 'mass_native', 'rho0', 'material_idx', 'free'}
BASE = {'pos', 'vel', 'active', 'mass_kg', 'native_particle_id', 'mass_native',
        'local_xyz', 'inside_carrier', 'inside_solid_union', 'solid_union_depth_m',
        'deep_solid', 'outside_effective_domain', 'domain_warning'}
BOOL = {'active', 'inside_carrier', 'inside_solid_union', 'deep_solid',
        'outside_effective_domain', 'domain_warning', 'free'}
TIME_ULPS = 4


def _require(condition, message):
    if not condition:
        raise ContractError(message)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ContractError('Archive metadata must be finite JSON-safe values') from exc


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(_json(dict(dtype=a.dtype.str, shape=list(a.shape),
        data_sha256=hashlib.sha256(a.tobytes()).hexdigest()))).hexdigest()


def _integer(value):
    return type(value) is int and 0 <= value <= np.iinfo(np.int64).max


class ParticleArchive:
    """Archive one reader contract, with bounded uncompressed NPZ chunks.

    ``outer_dt_s`` is the actual native outer timestep. A record's time must
    agree with step*dt within four binary64 ULPs. Its reader and body clocks
    must match exactly. ``start_step`` explicitly defines the first record.
    Chunk limits bound array payload, not allocator/ZIP/JSON overhead or RSS.
    Stacking can temporarily duplicate one bounded chunk. F and detF are not
    stored at every step. Their quality is retained without promoting false to
    true. Optional first/last F snapshots preserve the original array dtype.
    """
    def __init__(self, directory, contract, *, outer_dt_s, start_step=0,
                 chunk_steps=100, max_chunk_bytes=64 * 1024 * 1024,
                 store_initial_F=True, store_final_F=True):
        _require(type(contract) is dict and contract.get('schema_version') ==
                 'native-particle-material-mass-contract-v1', 'Use the actual reader mass contract')
        self._contract = copy.deepcopy(contract)
        self.contract_sha256 = hashlib.sha256(_json(contract)).hexdigest()
        self.n = contract.get('particle_count'); self.kind = contract.get('material_kind')
        _require(_integer(self.n) and self.n > 0 and self.kind in ('process_liquid', 'sand'),
                 'Archive requires a nonempty supported reader contract')
        _require(type(outer_dt_s) in (int, float) and math.isfinite(outer_dt_s) and outer_dt_s > 0
                 and _integer(start_step), 'Explicit positive outer timestep and start step required')
        _require(_integer(chunk_steps) and chunk_steps > 0 and _integer(max_chunk_bytes)
                 and max_chunk_bytes > 0, 'Positive chunk bounds required')
        _require(type(store_initial_F) is bool and type(store_final_F) is bool,
                 'F snapshot choices must be Boolean')
        ids = contract.get('native_particle_ids'); mass = contract.get('particle_masses_kg')
        _require(type(ids) is list and len(ids) == self.n and all(_integer(i) for i in ids)
                 and len(set(ids)) == self.n and type(mass) is list and len(mass) == self.n
                 and all(type(m) in (int, float) and math.isfinite(m) and m > 0 for m in mass),
                 'Reader contract must contain all original IDs and positive SI masses')
        _require(ids == list(range(contract['native_particle_start'], contract['native_particle_start'] + self.n)),
                 'Reader contract native-order ID range differs')
        self.dt = float(outer_dt_s); self.start = start_step
        self.chunk_steps = chunk_steps; self.max_bytes = max_chunk_bytes
        self.initial_F = store_initial_F; self.final_F = store_final_F
        self.path = Path(directory).resolve(); self.path.mkdir(parents=True, exist_ok=False)
        self._static = None; self._static_sha = None; self._schema = None; self._buffer = []; self._buffer_bytes = 0
        self._samples = 0; self._committed = 0; self._last = None; self._last_F = None
        self._chunks = []; self._snapshots = []; self._status = 'OPEN'; self._error = None
        self._closed = False; self._manifest_sha = None
        self._implementation_sha = _sha(Path(__file__))
        self._write_json_new('contract.json', self._contract)
        self._contract_file_sha = _sha(self.path / 'contract.json')
        self._manifest()

    def _write_json_new(self, filename, value):
        with (self.path / filename).open('xb') as stream:
            stream.write(_json(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())

    def _npz_new(self, filename, arrays):
        with (self.path / filename).open('xb') as stream:
            np.savez(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        return _sha(self.path / filename)

    def _manifest(self):
        path = self.path / 'manifest.json'
        _require(_sha(self.path / 'contract.json') == self._contract_file_sha and
                 (self._static is None or _sha(self.path / 'static_arrays.npz') == self._static_sha),
                 'Archive immutable files changed outside this writer')
        _require((not path.exists() and self._manifest_sha is None) or
                 (path.exists() and _sha(path) == self._manifest_sha),
                 'Archive manifest was replaced outside this writer')
        value = dict(schema_version=SCHEMA, status=self._status, particle_material_contract_sha256=self.contract_sha256,
            archive_implementation_sha256=self._implementation_sha,
            contract_file='contract.json', contract_file_sha256=self._contract_file_sha,
            particle_count=self.n, outer_dt_s=self.dt, first_expected_step=self.start,
            time_tolerance=dict(binary64_ULPs=TIME_ULPS, scope='step*outer_dt only; reader/body/native clock equality is exact'),
            chunk_steps_limit=self.chunk_steps, chunk_uncompressed_array_bytes_limit=self.max_bytes,
            accepted_samples=self._samples, committed_samples=self._committed,
            last_valid_step=self._last, last_committed_step=self._chunks[-1]['last_step'] if self._chunks else None,
            pending_samples=len(self._buffer),
            static_arrays=None if self._static is None else dict(file='static_arrays.npz',
                sha256=self._static_sha),
            storage_policy=dict(immutable_arrays_once=sorted(STATIC), full_N_dynamic_arrays_every_step=True,
                F_and_detF_full_arrays_every_step=False, F_quality_every_step=self.kind == 'sand',
                initial_F_requested=self.initial_F, final_F_requested=self.final_F),
            array_schema=self._schema, chunks=self._chunks, selected_F_snapshots=self._snapshots,
            failure=self._error, native_execution_performed=False, replay_supported=False,
            quantity_or_physical_success_assigned=False)
        temporary = self.path / ('.manifest-' + uuid.uuid4().hex + '.tmp')
        with temporary.open('xb') as stream:
            stream.write(_json(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        self._manifest_sha = _sha(path)

    def _validate(self, result, body_observation):
        _require(type(result) is dict and type(result.get('arrays')) is dict and
                 type(result.get('summary')) is dict, 'Use ParticleMaterialReader.read result')
        a, s = result['arrays'], result['summary']
        expected = BASE | ({'rho0'} if self.kind == 'process_liquid' else {'material_idx', 'free', 'F', 'detF'})
        _require(set(a) == expected, 'Reader array keys changed or a particle array is missing')
        for key, value in a.items():
            _require(type(value) is np.ndarray and value.dtype.kind not in 'OVSUc' and
                     value.shape == ((self.n, 3, 3) if key == 'F' else (self.n, 3)
                         if key in ('pos', 'vel', 'local_xyz') else (self.n,)),
                     'Native particle count, array type or shape changed: ' + key)
            kind = value.dtype.kind
            _require(kind == 'b' if key in BOOL else kind in 'iu' if key in
                     ('native_particle_id', 'material_idx') else kind == 'f', 'Native dtype kind differs: ' + key)
            if key not in ('F', 'detF'):
                _require(np.isfinite(value).all(), 'Nonfinite particle array: ' + key)
        schema = {k: dict(dtype=v.dtype.str, shape=list(v.shape)) for k, v in a.items()}
        _require(self._schema is None or schema == self._schema, 'Particle array dtype or shape changed')
        _require(np.array_equal(a['native_particle_id'], self._contract['native_particle_ids']) and
                 np.array_equal(a['mass_kg'], self._contract['particle_masses_kg']) and
                 _array_hash(a['mass_native']) == self._contract['actual_native_mass_sha256'],
                 'Original particle ID or mass changed')
        if self._static is not None:
            _require(all(np.array_equal(a[k], v) for k, v in self._static.items()),
                     'Original static particle data changed')
        _require(s.get('schema_version') == 'native-particle-material-observation-v1' and
                 s.get('particle_material_contract_sha256') == self.contract_sha256 and
                 s.get('material_kind') == self.kind and s.get('source_id') == self._contract['source_id'] and
                 s.get('native_entity_idx') == self._contract['native_entity_idx'] and
                 s.get('persistent_id_count') == self.n and s.get('identity_and_mass_unchanged') is True and
                 s.get('native_clock_unchanged') is True and s.get('mass_native_sha256') == _array_hash(a['mass_native']),
                 'Reader observation contract or identity differs')
        _require(s.get('initial_mass_kg') == s.get('native_mass_kg') == self._contract['initial_total_mass_kg']
                 and type(s.get('material_quality_passed')) is bool,
                 'Reader total mass or quality flag differs from its immutable contract')
        step, now = s.get('native_scene_step'), s.get('simulation_s')
        _require(_integer(step) and step == (self.start if self._last is None else self._last + 1)
                 and type(now) in (int, float) and math.isfinite(now) and now >= 0,
                 'Archive requires consecutive actual outer steps and finite time')
        expected_time = step * self.dt
        _require(math.isfinite(expected_time) and
                 abs(now - expected_time) <= TIME_ULPS * max(math.ulp(float(now)), math.ulp(expected_time)),
                 'Actual time differs from step*outer_dt beyond four binary64 ULPs')
        clock = s.get('native_clock')
        _require(type(clock) is list and len(clock) == 4 and _integer(clock[0]) and
                 clock[0] == step and type(clock[1]) in (int, float) and clock[1] == now and
                 _integer(clock[2]) and _integer(clock[3]),
                 'Invalid actual native clock')
        _require(type(body_observation) is dict and body_observation.get('native_scene_step') == step and
                 type(body_observation.get('native_scene_step')) is int and
                 type(body_observation.get('simulation_s')) in (int, float) and
                 body_observation.get('simulation_s') == now, 'Body observation is not the exact reader clock')
        body = body_observation.get('body_world', {}); p = np.asarray(body.get('position_m'), dtype=float)
        q = np.asarray(body.get('quaternion_wxyz'), dtype=float)
        _require(body.get('frame') == 'world' and p.shape == (3,) and q.shape == (4,) and
                 np.isfinite(p).all() and np.isfinite(q).all() and abs(np.linalg.norm(q) - 1.) <= 1e-6,
                 'Invalid actual body world pose')
        quality = None
        if self.kind == 'sand':
            finite = bool(np.isfinite(a['F']).all() and np.isfinite(a['detF']).all())
            positive = bool(finite and np.all(a['detF'] > 0))
            minimum = float(a['detF'].min()) if finite else None
            _require(s.get('MPM_F_finite') is finite and s.get('MPM_positive_detF') is positive and
                     s.get('MPM_minimum_detF') == minimum and s.get('finite_arrays') is finite and
                     (finite and positive or s.get('material_quality_passed') is False),
                     'F quality summary differs from supplied F/detF or promotes failure')
            quality = dict(F_finite=finite, detF_positive=positive, minimum_detF=minimum,
                           full_F_stored_per_step=False, original_quality_passed=s.get('material_quality_passed'))
        summary = copy.deepcopy(s)
        _require(len(_json(summary)) <= 65536, 'Scalar summary exceeds bounded metadata limit')
        dynamic = {k: v.copy() for k, v in a.items() if k not in STATIC | {'F', 'detF'}}
        dynamic.update(native_scene_step=np.asarray(step, np.int64), simulation_s=np.asarray(now, np.float64),
            native_substep_global=np.asarray(clock[2], np.int64), native_substep_local=np.asarray(clock[3], np.int64),
            body_position_m=p.copy(), body_quaternion_wxyz=q.copy())
        size = sum(v.nbytes for v in dynamic.values())
        _require(size <= self.max_bytes, 'One full-N observation exceeds max_chunk_bytes')
        return a, schema, dynamic, summary, quality, size

    def append(self, reader_result, body_observation):
        """Copy one complete same-clock reader observation; return pending ref.

        An invalid row is never appended. Its preceding valid buffer is flushed
        before the archive is marked FAILED. This writer cannot authenticate a
        fabricated caller observation or replace the reader's source checks.
        """
        _require(not self._closed, 'Particle archive is already closed or failed')
        try:
            arrays, schema, dynamic, summary, quality, size = self._validate(reader_result, body_observation)
            if self._static is None:
                self._static = {k: v.copy() for k, v in arrays.items() if k in STATIC}
                self._schema = schema; self._static_sha = self._npz_new('static_arrays.npz', self._static)
            if self._buffer and self._buffer_bytes + size > self.max_bytes:
                self._flush()
            if self.kind == 'sand' and (self.final_F or (self.initial_F and self._samples == 0)):
                self._last_F = dict(step=summary['native_scene_step'], simulation_s=summary['simulation_s'],
                    F=arrays['F'].copy(), detF=arrays['detF'].copy())
                if self.initial_F and self._samples == 0:
                    self._snapshot_F('F_initial.npz', 'initial')
            index = len(self._buffer); chunk = len(self._chunks)
            reference = dict(schema_version=SCHEMA, manifest_file='manifest.json',
                particle_material_contract_sha256=self.contract_sha256, chunk_file=f'chunk_{chunk:06d}.npz',
                row_index=index, native_scene_step=summary['native_scene_step'],
                durability='PENDING_UNTIL_HASHED_MANIFEST_COMMIT', full_native_order_count=self.n)
            self._buffer.append((dynamic, summary, quality)); self._buffer_bytes += size
            self._samples += 1; self._last = summary['native_scene_step']
            if len(self._buffer) >= self.chunk_steps:
                self._flush()
                reference['durability'] = 'COMMITTED_IN_HASHED_MANIFEST'
            return dict(summary=copy.deepcopy(summary), particle_archive_reference=reference)
        except Exception as exc:
            self._fail(exc)
            raise

    def _flush(self):
        if not self._buffer:
            return
        index = len(self._chunks); name = f'chunk_{index:06d}.npz'; meta = f'chunk_{index:06d}.jsonl'
        arrays = {k: np.stack([r[0][k] for r in self._buffer]) for k in self._buffer[0][0]}
        digest = self._npz_new(name, arrays)
        with (self.path / meta).open('xb') as stream:
            for i, (_, summary, quality) in enumerate(self._buffer):
                value = dict(summary=summary, F_quality=quality, particle_archive_reference=dict(
                    schema_version=SCHEMA, particle_material_contract_sha256=self.contract_sha256,
                    chunk_file=name, chunk_sha256=digest, row_index=i, native_scene_step=summary['native_scene_step'],
                    durability='COMMITTED_ONLY_WHEN_LISTED_IN_MANIFEST', full_native_order_count=self.n))
                stream.write(_json(value) + b'\n')
            stream.flush(); os.fsync(stream.fileno())
        self._chunks.append(dict(file=name, sha256=digest, metadata_file=meta, metadata_sha256=_sha(self.path / meta),
            rows=len(self._buffer), first_step=self._buffer[0][1]['native_scene_step'],
            last_step=self._buffer[-1][1]['native_scene_step'], uncompressed_array_bytes=self._buffer_bytes))
        self._committed += len(self._buffer); self._buffer = []; self._buffer_bytes = 0
        self._manifest()

    def flush(self):
        _require(not self._closed, 'Particle archive is already closed or failed')
        try:
            self._flush()
        except Exception as exc:
            self._fail(exc); raise

    def _snapshot_F(self, name, role):
        if self._last_F is None:
            return
        s = self._last_F
        digest = self._npz_new(name, dict(F=s['F'], detF=s['detF'], native_scene_step=np.asarray(s['step'], np.int64),
            simulation_s=np.asarray(s['simulation_s'], np.float64)))
        self._snapshots.append(dict(file=name, sha256=digest, role=role, native_scene_step=s['step']))

    def _fail(self, exc):
        if self._closed:
            return
        error = dict(type=type(exc).__name__, message=str(exc), invalid_row_never_appended=True)
        try:
            self._flush()
        except Exception as flush_error:
            error['flush_failure'] = dict(type=type(flush_error).__name__, message=str(flush_error))
        self._status = 'FAILED'; self._error = error; self._closed = True
        try:
            if self.final_F:
                self._snapshot_F('F_final.npz', 'last_valid_before_failure')
            self._manifest()
        except Exception as record_error:
            # Earlier committed manifest/chunks remain authoritative. Never hide
            # the original input/I/O error by replacing it with a cleanup error.
            self._error['failure_record_error'] = str(record_error)

    def close(self):
        if self._closed:
            return
        try:
            self._flush()
            if self.final_F:
                self._snapshot_F('F_final.npz', 'last_valid_on_close')
            self._status = 'CLOSED'; self._manifest(); self._closed = True
        except Exception as exc:
            self._fail(exc); raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc is None:
            self.close()
        else:
            self._fail(exc)
        return False
