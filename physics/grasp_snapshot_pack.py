"""Fresh scene-owned grasp snapshots without repeated equality-force exports.

The pinned public equality kernel copies the same native integer rows and the
public weld getter filters those rows by WELD. No native value is cached here.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import inspect
import math
from pathlib import Path
import time

import numpy as np

from physics.carried_body import WorldPose
from physics.episode_observation import _dependencies as _pose_dependencies, _host
from transport_challenge.contracts import ContractError

EXTRA_SOURCE_PINS = {
    'engine/solvers/rigid/constraint/solver.py': 'f0753ff78bfe93b42d82ecb5c81e440cf0c521a3588efa3a4663a7c82e3b4d0f',
    'utils/array_class.py': '459ff6c67a3080d5ebf71e746d59f6746e6e76f11a1d534dfbfc7a78dd049d34',
    'constants.py': 'd8bbc696685b60b0238b35e1408814ef1da5918b0f65cd611df243c836fb1bc8',
}


def _need(ok, message):
    if not ok:
        raise ContractError(message)


def _dependencies(scene):
    torch, qd, sources = _pose_dependencies(scene)
    root = Path(metadata.distribution('genesis-world').locate_file('genesis')).resolve()
    for relative, expected in EXTRA_SOURCE_PINS.items():
        path = root/relative
        _need(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Unreviewed grasp inventory API: '+relative)
        sources[str(path)] = expected
    from genesis.utils.misc import qd_to_torch
    _need(str(Path(inspect.getsourcefile(qd_to_torch)).resolve()) in sources,
          'Unreviewed zero-copy field API')
    _need(Path(inspect.getsourcefile(type(scene.rigid_solver.constraint_solver))).resolve()
          == root/'engine/solvers/rigid/constraint/solver.py', 'Unreviewed native constraint solver')
    return torch, qd, sources, qd_to_torch


def _buffer(field):
    return field._unwrap() if hasattr(field, '_unwrap') else field


class GraspSnapshotPack:
    def __init__(self, scene, *, cup, hands, body=None):
        _need(type(hands) is dict and bool(hands) and all(type(n) is str and n for n in hands),
              'Named native hands required')
        self.scene, self.r, self.cup = scene, scene.rigid_solver, cup
        self.hands = dict(hands)
        self.body = cup.get_link('payload') if body is None else body
        self.torch, self.qd, self.sources, native_view = _dependencies(scene)
        self._signature = self._topology()
        self._clock()
        self.qd.sync()
        self._fields_bound = self._fields()
        self._buffers = tuple(_buffer(field) for _, field in self._fields_bound)
        self._views = tuple(native_view(field, copy=False) for _, field in self._fields_bound)
        _need(len(self._views) == 5, 'Native equality field inventory differs')
        self._capacity = int(self._views[2].shape[0]) if len(self._views[2].shape) == 2 else 0
        _need(self._capacity > 0, 'Reserved equality buffer must have positive capacity')
        self._shapes = ((1,), (), (self._capacity, 1), (self._capacity, 1), (self._capacity, 1))
        self._integer_device = self._views[0].device
        self._view_pointers = tuple(view.data_ptr() for view in self._views)
        self._validate_views()
        self._float_identity = None
        self.snapshot_calls = self.inventory_calls = self.float_host_copies = self.integer_host_copies = 0
        self.wall_s = dict(snapshot=0., inventory=0., sync=0.)
        self.failed = False

    def _fields(self):
        eq = self.r.dyn_info.equalities
        return (('active_count', self.r.constraint_solver.constraint_state.qd_n_equalities),
                ('capacity', self.r.rigid_info.n_candidate_equalities),
                ('eq_type', eq.eq_type), ('eq_obj1id', eq.eq_obj1id), ('eq_obj2id', eq.eq_obj2id))

    def _topology(self):
        _need(self.scene.is_built and self.scene.rigid_solver is self.r and self.r.n_envs == 0
              and not self.scene.sim.requires_grad and self.cup.solver is self.r
              and self.cup.n_links == 1 and len(self.cup.geoms) == 9 and self.cup.morph.fixed is False
              and self.cup.get_link('payload') is self.body and self.body.entity is self.cup
              and not self.body.is_fixed and int(self.cup.base_link_idx) == int(self.body.idx)
              and all(hand.solver is self.r for hand in self.hands.values()),
              'Snapshot needs the same built forward scene, free one-link nine-box cup and native hands')
        return (id(self.scene), id(self.scene.sim), id(self.r), id(self.r.constraint_solver),
                id(self.cup), int(self.cup.idx), int(self.cup.base_link_idx), id(self.body), int(self.body.idx),
                int(self.r.n_equalities), tuple((n, id(hand), int(hand.idx)) for n, hand in self.hands.items()))

    def _clock(self):
        sim = self.scene.sim
        clock = (self.scene.t, self.scene.cur_t, sim.cur_substep_global, sim.cur_substep_local)
        _need(all(not isinstance(clock[i], (bool, np.bool_)) and isinstance(clock[i], (int, np.integer))
                  and clock[i] >= 0 for i in (0, 2, 3))
              and not isinstance(clock[1], (bool, np.bool_)) and isinstance(clock[1], (int, float, np.number))
              and math.isfinite(float(clock[1])) and sim.dt == .0025
              and sim.substeps in (1, 16, 100) and sim.substeps_local == 1,
              'Unreviewed native grasp clock/profile')
        step, now, global_step, local_step = int(clock[0]), float(clock[1]), int(clock[2]), int(clock[3])
        _need(step == global_step//sim.substeps and local_step == 0
              and now == global_step*(sim.dt/sim.substeps), 'Actual internal grasp clock differs')
        return [step, now, global_step, local_step]

    def _validate_views(self):
        current = self._fields()
        _need(tuple(n for n, _ in current) == tuple(n for n, _ in self._fields_bound)
              and all(_buffer(field) is original for (_, field), original in zip(current, self._buffers)),
              'Native equality allocation/buffer identity changed')
        for view, shape, pointer in zip(self._views, self._shapes, self._view_pointers):
            _need(isinstance(view, self.torch.Tensor) and tuple(view.shape) == shape
                  and view.dtype == self.torch.int32 and view.device == self._integer_device
                  and view.data_ptr() == pointer, 'Native equality field shape/dtype/device/view changed')

    def _begin(self):
        _need(not self.failed and self._topology() == self._signature, 'Bound grasp topology changed or previous read failed')
        self._validate_views()
        clock = self._clock()
        started = time.perf_counter()
        self.qd.sync()
        self.wall_s['sync'] += time.perf_counter()-started
        _need(self._clock() == clock, 'Sync changed native grasp clock')
        return clock

    def _finish(self, clock):
        _need(self._clock() == clock and self._topology() == self._signature,
              'Snapshot changed native clock/topology')
        self._validate_views()

    def _inventory(self):
        self.inventory_calls += 1
        packed = self.torch.cat(tuple(view.reshape(-1) for view in self._views))
        _need(packed.dtype == self.torch.int32 and packed.device == self._integer_device
              and tuple(packed.shape) == (2+3*self._capacity,), 'Native integer pack changed')
        values = np.asarray(packed.detach().cpu().numpy()).copy()
        self.integer_host_copies += 1
        _need(values.dtype == np.dtype('int32') and values.shape == (2+3*self._capacity,), 'Invalid native integer host pack')
        n, capacity = int(values[0]), int(values[1])
        _need(capacity == self._capacity and 0 <= self.r.n_equalities <= n <= capacity,
              'Native active equality count/capacity invalid')
        columns = values[2:].reshape(3, self._capacity)
        active = columns[:, :n].T
        _need((active >= 0).all(), 'Invalid active native equality identity')
        rows = active.tolist()
        return dict(count=n, capacity=capacity, rows=rows,
                    weld_pairs=[row[1:] for row in rows if row[0] == 1])

    def inventory(self):
        """Fresh active native rows, including same-clock add/delete updates."""
        started = time.perf_counter()
        try:
            clock = self._begin()
            result = self._inventory()
            self._finish(clock)
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.wall_s['inventory'] += time.perf_counter()-started

    def _tensor(self, value, shape):
        _need(isinstance(value, self.torch.Tensor) and tuple(value.shape) == shape
              and value.dtype in (self.torch.float32, self.torch.float64)
              and value.device == self._integer_device, 'Native snapshot tensor shape/dtype/device differs')
        identity = value.dtype, value.device
        if self._float_identity is None:
            self._float_identity = identity
        _need(identity == self._float_identity, 'Native snapshot float dtype/device changed')
        return value

    def snapshot(self, arm_id):
        started = time.perf_counter()
        self.snapshot_calls += 1
        try:
            _need(arm_id in self.hands, 'Snapshot arm must name a bound native hand')
            clock = self._begin()
            hand = self.hands[arm_id]
            ids = (int(self.cup.base_link_idx), int(hand.idx))
            values = (self._tensor(self.r.get_links_pos(ids, ref='link_origin', relative=False), (2, 3)),
                      self._tensor(self.r.get_links_quat(ids, relative=False), (2, 4)),
                      self._tensor(self.r.get_links_vel(ids, ref='link_origin'), (2, 3)),
                      self._tensor(self.r.get_links_ang(ids), (2, 3)))
            packed = self.torch.cat(tuple(value.reshape(-1) for value in values))
            self._tensor(packed, (26,))
            host = _host(packed, (26,))
            self.float_host_copies += 1
            positions, quaternions = host[:6].reshape(2, 3), host[6:14].reshape(2, 4)
            velocities, angular = host[14:20].reshape(2, 3), host[20:].reshape(2, 3)
            def pose(index):
                p = dict(frame='world', position_m=positions[index].tolist(), quaternion_wxyz=quaternions[index].tolist())
                WorldPose(p['position_m'], p['quaternion_wxyz'])
                return p
            # The pinned one-link public getter returns the same fresh host
            # inertial-mass value as the original helper. Do not round via f32.
            mass = np.asarray(self.cup.get_mass()).copy()
            _need(mass.size == 1 and mass.dtype.kind == 'f' and np.isfinite(mass).all() and mass.item() > 0,
                  'Invalid public one-link cup mass')
            result = dict(native_clock=clock, body_entity_idx=int(self.cup.idx), cup_link_idx=int(self.body.idx),
                hand_link_idx=int(hand.idx), cup_mass_kg=float(mass.item()),
                body_world=pose(0), hand_world=pose(1), body_linear_velocity_m_s=velocities[0].tolist(),
                body_angular_velocity_rad_s=angular[0].tolist(), hand_linear_velocity_m_s=velocities[1].tolist(),
                hand_angular_velocity_rad_s=angular[1].tolist(), inventory=self._inventory())
            self._finish(clock)
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.wall_s['snapshot'] += time.perf_counter()-started

    def metadata(self):
        return dict(schema_version='fresh-native-grasp-snapshot-pack-v1', snapshot_calls=self.snapshot_calls,
            inventory_calls=self.inventory_calls, float_host_copies=self.float_host_copies,
            integer_host_copies=self.integer_host_copies, reserved_equality_capacity=self._capacity,
            native_equality_fields=[n for n, _ in self._fields_bound], source_hashes=dict(self.sources),
            measured_host_wall_s=dict(self.wall_s), failed=self.failed,
            snapshot_values_cached=False, inactive_reserved_rows_ignored=True,
            public_weld_subset_equivalence_source='Pinned constraint/solver.py get_weld_constraints and kernel_get_equality_constraints',
            scope='Fresh native poses/velocities/active equality inventory per call, including same-clock transactions. Logical two-copy snapshot, not a measured speedup or physical/holder acceptance.')
