"""Exact public-getter ten-pose capture for the existing rigid16 observer.

No physics, contacts, ownership, FK or IK is changed. Native internal clocks
remain g for both detection and produced-state captures; no state is cached.
"""
from __future__ import annotations

import math
import time
import hashlib
from pathlib import Path

import numpy as np

from physics.episode_observation import _dependencies, _host
from transport_challenge.contracts import ContractError


def _need(condition, message):
    if not condition:
        raise ContractError(message)


class RigidPosePack:
    def __init__(self, scene, *, hands, fingers, tray, arm_names, mode='packed', verify_scalar=False,
                 pose_hands=None, pose_fingers=None):
        _need(mode in ('scalar', 'packed') and type(verify_scalar) is bool,
              'Select explicit scalar/packed capture and Boolean parity')
        _need(not verify_scalar or mode == 'packed', 'Scalar parity is an optional packed-capture diagnostic')
        self.names = tuple(arm_names)
        self.single_active_contract = None
        if len(self.names)==1:
            from physics.fixed_peer_arms import validate_scene_contract
            try:self.single_active_contract=validate_scene_contract(scene,{n:hands[n].entity for n in self.names})
            except (KeyError,TypeError,ValueError) as exc:raise ContractError('Invalid single-active native pose contract: '+str(exc)) from exc
            _need(pose_hands is not None and pose_fingers is not None,
                  'Single-active capture needs actual active and fixed peer pose maps')
            expected={self.single_active_contract['active_arm_id'],*self.single_active_contract['frozen_arm_ids']}
            _need(set(pose_hands)==set(pose_fingers)==expected and set(hands)==set(fingers)==set(self.names)
                  and all(pose_hands[n] is hands[n] and all(pose_fingers[n][k] is fingers[n][k] for k in ('left_finger','right_finger')) for n in self.names),
                  'Actual active/fixed pose map differs from its native qualification')
            self.capture_names=tuple(sorted(expected))
            hands,fingers=pose_hands,pose_fingers
            _need(all(hands[n].entity.name==n and hands[n].name=='hand'
                      and all(fingers[n][k].entity is hands[n].entity and fingers[n][k].name==k for k in ('left_finger','right_finger'))
                      for n in self.capture_names)
                  and all(hands[n].entity.n_dofs==0 for n in self.single_active_contract['frozen_arm_ids']),
                  'Fixed peer poses must be their actual zero-DOF native hand/finger links')
        else:
            _need(pose_hands is None and pose_fingers is None,'Legacy pose maps cannot borrow single-active override')
            self.capture_names=self.names
        _need(len(self.capture_names) == 3 and len(set(self.capture_names)) == 3
              and all(type(n) is str and n for n in self.names)
              and set(hands) == set(fingers) == set(self.capture_names)
              and all(set(fingers[n]) == {'left_finger', 'right_finger'} for n in self.capture_names),
              'Capture requires the three ordered hands and their two named fingers')
        self.scene, self.solver, self.tray = scene, scene.rigid_solver, tray
        self.hands = dict(hands)
        self.fingers = {n:dict(fingers[n]) for n in self.capture_names}
        self.links = tuple(link for n in self.capture_names for link in
                           (self.hands[n], self.fingers[n]['left_finger'], self.fingers[n]['right_finger']))
        self.pose_ids = tuple(int(link.idx) for link in self.links)+(int(tray.base_link_idx),)
        _need(len(set(self.pose_ids)) == 10 and min(self.pose_ids) >= 0,
              'Ten distinct native link origins required')
        self.torch, self.qd, self.sources = _dependencies(scene)
        if self.single_active_contract is not None:
            import physics.fixed_peer_arms as fixed
            self.sources[str(Path(fixed.__file__).resolve())]=hashlib.sha256(Path(fixed.__file__).read_bytes()).hexdigest()
        self.mode, self.verify_scalar = mode, verify_scalar
        self._identity = None
        self.device_capture_calls = 0
        self._signature = self._topology()
        self._clock()
        self.capture_calls = self.completed_captures = self.scalar_calls = self.packed_calls = self.parity_checks = 0
        self.logical_host_transfers = 0
        self.wall_s = dict(total_capture=0., sync=0., scalar_read=0., packed_read=0., scalar_parity=0.)
        self.last_native_clock = None
        self.failed = False

    def _topology(self):
        if self.single_active_contract is not None:
            _need(getattr(self.scene,'_ap_single_active_arm_contract',None)==self.single_active_contract,
                  'Qualified compact fixed-peer pose contract changed')
        _need(self.scene.is_built and self.scene.rigid_solver is self.solver
              and self.solver.n_envs == 0 and self.tray.solver is self.solver
              and all(link.solver is self.solver for link in self.links),
              'Pose capture requires the same built unbatched native solver')
        return (id(self.scene), id(self.scene.sim), id(self.solver), self.names,
                tuple((id(link), int(link.idx)) for link in self.links),
                id(self.tray), int(self.tray.base_link_idx), self.pose_ids,
                tuple((n, id(self.hands[n]), id(self.fingers[n]['left_finger']),
                       id(self.fingers[n]['right_finger'])) for n in self.capture_names))

    def _clock(self):
        sim = self.scene.sim
        _need(sim.dt == .0025 and sim.substeps == 16 and sim.substeps_local == 1,
              'Pose capture is bound to the original rigid16 integration profile')
        values = (self.scene.t, self.scene.cur_t, sim.cur_substep_global, sim.cur_substep_local)
        _need(all(not isinstance(values[i], (bool, np.bool_)) and isinstance(values[i], (int, np.integer))
                  and values[i] >= 0 for i in (0, 2, 3))
              and not isinstance(values[1], (bool, np.bool_))
              and isinstance(values[1], (int, float, np.number)) and math.isfinite(float(values[1])),
              'Invalid actual internal native clock')
        step, now, global_step, local_step = int(values[0]), float(values[1]), int(values[2]), int(values[3])
        _need(step == global_step//16 and local_step == 0 and now == global_step*(.0025/16),
              'Capture must use the actual internal g clock, not an outer-clock substitute')
        return step, now, global_step, local_step

    def _tensor(self, value, shape):
        _need(isinstance(value, self.torch.Tensor) and tuple(value.shape) == shape
              and value.dtype in (self.torch.float32, self.torch.float64),
              'Unexpected native pose tensor shape or dtype')
        identity = value.dtype, value.device
        if self._identity is None:
            self._identity = identity
        _need(identity == self._identity, 'Native pose dtype/device changed')
        return value

    def _record(self, positions, quaternions):
        def pose(index):
            return dict(frame='world', position_m=positions[index].tolist(),
                        quaternion_wxyz=quaternions[index].tolist())
        captures=getattr(self,'capture_names',self.names)
        return dict(body_world=pose(9), arms={name:dict(hand_world=pose(3*captures.index(name)),
            finger_world=dict(left=pose(3*captures.index(name)+1), right=pose(3*captures.index(name)+2))) for name in self.names})

    def _packed(self):
        started = time.perf_counter()
        self.packed_calls += 1
        positions = self._tensor(self.solver.get_links_pos(self.pose_ids, ref='link_origin', relative=False), (10, 3))
        quaternions = self._tensor(self.solver.get_links_quat(self.pose_ids, relative=False), (10, 4))
        packed = self.torch.cat((positions.reshape(-1), quaternions.reshape(-1)))
        self._tensor(packed, (70,))
        host = _host(packed, (70,))
        self.logical_host_transfers += 1
        result = self._record(host[:30].reshape(10, 3), host[30:].reshape(10, 4))
        self.wall_s['packed_read'] += time.perf_counter()-started
        return result

    def device_capture(self):
        """Fresh packed CUDA poses for one phase-owned transfer, never a cache."""
        _need(self.mode=='packed' and not self.verify_scalar and not self.failed,
              'Consolidated captures require the original non-parity packed reader')
        self.capture_calls+=1
        self.device_capture_calls+=1
        _need(self._topology()==self._signature, 'Native pose topology changed')
        clock=self._clock()
        started=time.perf_counter();self.qd.sync();self.wall_s['sync']+=time.perf_counter()-started
        positions=self._tensor(self.solver.get_links_pos(self.pose_ids,ref='link_origin',relative=False),(10,3))
        quaternions=self._tensor(self.solver.get_links_quat(self.pose_ids,relative=False),(10,4))
        packed=self.torch.cat((positions.reshape(-1),quaternions.reshape(-1)))
        self._tensor(packed,(70,))
        _need(packed.device.type=='cuda' and self._clock()==clock
              and self._topology()==self._signature,'Device capture requires unchanged CUDA state')
        self.packed_calls+=1
        return clock,packed

    def decode_device_capture(self, clock, values):
        """Validate the owned host slice at the same phase before accepting it."""
        host=_host(values,(70,))
        _need(self._clock()==clock and self._topology()==self._signature,
              'Combined transfer changed native pose clock/topology')
        result=self._record(host[:30].reshape(10,3),host[30:].reshape(10,4))
        self.completed_captures+=1;self.last_native_clock=clock
        return result

    def _scalar(self):
        started = time.perf_counter()
        self.scalar_calls += 1
        positions, quaternions = [], []
        for entity in (*self.links, self.tray):
            positions.append(_host(self._tensor(entity.get_pos(relative=False), (3,)), (3,)))
            quaternions.append(_host(self._tensor(entity.get_quat(relative=False), (4,)), (4,)))
            self.logical_host_transfers += 2
        result = self._record(np.asarray(positions), np.asarray(quaternions))
        self.wall_s['scalar_read'] += time.perf_counter()-started
        return result

    def read(self):
        """Fresh read at the unchanged actual internal clock; never fallback."""
        _need(not self.failed, 'A failed pose capture cannot silently resume')
        started = time.perf_counter()
        self.capture_calls += 1
        try:
            _need(self._topology() == self._signature, 'Native pose topology changed')
            before = self._clock()
            sync_started = time.perf_counter()
            self.qd.sync()
            self.wall_s['sync'] += time.perf_counter()-sync_started
            _need(self._clock() == before, 'Synchronization changed the actual native clock')
            result = self._packed() if self.mode == 'packed' else self._scalar()
            _need(self._clock() == before and self._topology() == self._signature,
                  'Pose read changed native clock/topology')
            if self.verify_scalar:
                parity_started = time.perf_counter()
                expected = self._scalar()
                _need(result == expected, 'Same-clock scalar/packed ten-pose values differ')
                _need(self._clock() == before and self._topology() == self._signature,
                      'Parity changed native clock/topology')
                self.parity_checks += 1
                self.wall_s['scalar_parity'] += time.perf_counter()-parity_started
            self.completed_captures += 1
            self.last_native_clock = before
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.wall_s['total_capture'] += time.perf_counter()-started

    capture = read

    def metadata(self):
        result=dict(schema_version='rigid16-public-ten-pose-pack-v1', mode=self.mode,
            verify_scalar=self.verify_scalar, arm_names=list(self.names), pose_link_indices=list(self.pose_ids),
            body_slot=9, packed_tensor_values=70, public_getters_per_packed_capture=2,
            modeled_host_transfers_per_packed_capture=1, modeled_host_transfers_per_scalar_capture=20,
            capture_calls=self.capture_calls, completed_captures=self.completed_captures,
            device_capture_calls=self.device_capture_calls,
            scalar_calls=self.scalar_calls, packed_calls=self.packed_calls, exact_same_clock_parity_checks=self.parity_checks,
            logical_host_transfers=self.logical_host_transfers, measured_host_wall_s=dict(self.wall_s),
            last_actual_native_clock=None if self.last_native_clock is None else list(self.last_native_clock),
            failed=self.failed, source_hashes=dict(self.sources),
            native_dtype_device=None if self._identity is None else [str(v) for v in self._identity],
            scope='Readback only. Counts are logical API/host conversions, not measured GPU transfers or speedup. Total host capture time includes synchronization and optional nested scalar parity.')
        if self.single_active_contract is not None:
            result.update(schema_version='rigid16-single-active-ten-native-pose-pack-v1',
                capture_arm_names=list(self.capture_names),
                single_active_contract_sha256=self.single_active_contract['contract_sha256'],
                fixed_peer_pose_links_are_real=True,inactive_motor_state_fabricated=False)
        return result
