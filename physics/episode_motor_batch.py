"""Optional public-API motor commands for one persistent three-Panda scene.

Only the selected 27 DOFs receive the supplied actuator biases and targets.
One public force query is copied to the host. The caller retains every effort,
tracking and physical check before stepping. This helper never steps or sets a
joint/body pose. It must not overlap another controller or a physics step.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from numbers import Integral
from pathlib import Path

import numpy as np

from transport_challenge.contracts import ContractError


MOTOR_SOURCE_PINS = {
    'engine/solvers/rigid/abd/accessor.py': '64250f83e393119cdec38f6bfdac243a68fc8701e3a6109e1775c78c3513c54d',
    'engine/couplers/legacy_coupler.py': '29b5778655bca346d74a1f54dff07a5188db22705ea6b384b09c35f202483b9b',
    'engine/scene.py': 'd4b6b2dd11d6745147daaa3fc47e2f172d832c9bfc8c6b68576bb1d4415fce3c',
    'engine/simulator.py': 'b385b363dfb56512ae189d3ad86fd9bd75acc7209e2cf4b9c934ec76e98073da',
}


def motor_source_paths():
    """Reuse installed public-getter pins without importing a native runtime."""
    from physics.episode_observation import observation_source_paths
    from physics.handle_contacts import verify_contact_sources
    root, _ = verify_contact_sources()
    paths = observation_source_paths()
    for relative, expected in MOTOR_SOURCE_PINS.items():
        path = Path(root) / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ContractError('Reviewed public motor source changed: ' + relative)
        paths[str(path.resolve())] = expected
    return paths


def _dependencies(scene, arms):
    sources = motor_source_paths()
    import genesis as gs
    import quadrants as qd
    import torch
    root = Path(gs.__file__).resolve().parent
    objects = [(scene.rigid_solver, 'genesis.engine.solvers.rigid.rigid_solver', 'RigidSolver',
                'engine/solvers/rigid/rigid_solver.py'),
               (scene.sim.coupler, 'genesis.engine.couplers.legacy_coupler', 'LegacyCoupler',
                'engine/couplers/legacy_coupler.py')]
    objects.extend((arm, 'genesis.engine.entities.rigid_entity.rigid_entity', 'RigidEntity',
                    'engine/entities/rigid_entity/rigid_entity.py') for arm in arms.values())
    for obj, module, name, relative in objects:
        if (type(obj).__module__ != module or type(obj).__name__ != name
                or Path(inspect.getsourcefile(type(obj))).resolve() != root / relative):
            raise ContractError('Motor batching requires the reviewed native ' + name)
    if (gs.use_zerocopy is not True or gs.backend != gs.cuda or gs.device.type != 'cuda'
            or gs.tc_float not in (torch.float32, torch.float64)):
        raise ContractError('Motor batching requires the reviewed zero-copy CUDA backend')
    device = gs.device if gs.device.index is not None else torch.device('cuda', torch.cuda.current_device())
    return torch, qd, device, gs.tc_float, sources


def _vector(value, label):
    raw = np.asarray(value)
    if raw.shape != (9,) or raw.dtype.kind not in 'iuf':
        raise ContractError(label + ' must be a numeric nine-vector without booleans')
    # Mixed Python sequences otherwise coerce True to 1 before the dtype check.
    if isinstance(value, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in value):
        raise ContractError(label + ' cannot contain booleans')
    result = raw.astype(float, copy=True)
    if not np.isfinite(result).all():
        raise ContractError(label + ' must be finite')
    return result


class BatchedEpisodeMotors:
    def __init__(self, scene, arms, kp, kv):
        if (len(arms) != 3 or any(not isinstance(n, str) or not n for n in arms)
                or len({id(a) for a in arms.values()}) != 3):
            raise ContractError('Motor batching requires three distinct named arms')
        self.scene = scene
        self.solver = scene.rigid_solver
        self.arms = dict(arms)
        self.names = tuple(self.arms)
        self.torch, self.qd, self.device, self.dtype, self.sources = _dependencies(scene, self.arms)
        self._kp = _vector(kp, 'kp')
        self._kv = _vector(kv, 'kv')
        self._bias1 = np.tile(-self._kp, 3)
        self._bias2 = np.tile(-self._kv, 3)
        self._signature = self._topology()
        self._check()
        indices = [int(a.dof_start) + i for a in self.arms.values() for i in range(9)]
        if len(set(indices)) != 27 or min(indices) < 0 or max(indices) >= self.solver.n_dofs:
            raise ContractError('Motor DOF ranges overlap or leave the built solver')
        self._dof_indices = tuple(indices)
        self.applies = 0

    @property
    def dof_indices(self):
        return self._dof_indices

    def _topology(self):
        return (id(self.scene), id(self.scene.rigid_solver), id(self.scene.sim), id(self.scene.sim.coupler),
                self.scene.rigid_solver.n_dofs, tuple(self.arms), self.names,
                tuple((n, id(a), a.idx, a.dof_start, a.q_start, a.n_dofs, a.n_qs,
                       id(a.scene), id(a.sim)) for n, a in self.arms.items()))

    def _check(self, expected_clock=None):
        if (not self.scene.is_built or not self.solver.is_built or self.solver.n_envs != 0
                or not self.scene.sim._rigid_only or self.solver._use_hibernation
                or self._topology() != self._signature):
            raise ContractError('Motor batching requires its original built unbatched rigid scene')
        entity_ids = []
        for arm in self.arms.values():
            if (arm.scene is not self.scene or arm.sim is not self.scene.sim or not arm.is_built
                    or arm.n_dofs != 9 or arm.n_qs != 9
                    or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
                           for v in (arm.idx, arm.dof_start, arm.q_start))
                    or not 0 <= arm.idx < len(self.scene.entities)
                    or self.scene.entities[arm.idx] is not arm):
                raise ContractError('Motor arm identity or native nine-DOF binding changed')
            entity_ids.append(arm.idx)
        if len(set(entity_ids)) != 3:
            raise ContractError('Motor arms alias a native entity')
        step, now = self.scene.t, self.scene.cur_t
        if (isinstance(step, (bool, np.bool_)) or not isinstance(step, Integral) or step < 0
                or isinstance(now, (bool, np.bool_)) or not isinstance(now, (int, float, np.number))
                or not math.isfinite(float(now)) or now < 0):
            raise ContractError('Invalid motor command scene clock')
        clock = (int(step), float(now))
        if expected_clock is not None and clock != expected_clock:
            raise ContractError('Scene clock changed during motor commands')
        cuda = self.torch.cuda
        if cuda.current_stream(self.device).cuda_stream != cuda.default_stream(self.device).cuda_stream:
            raise ContractError('Motor batching requires the default CUDA stream')
        return clock

    def apply(self, commands):
        """Set supplied commands and return actual bounded control force by arm."""
        clock = self._check()
        if not isinstance(commands, dict) or set(commands) != set(self.names):
            raise ContractError('Motor commands must cover each bound arm exactly once')
        packed = {key: [] for key in ('tau', 'target', 'target_v')}
        for name in self.names:
            row = commands[name]
            if not isinstance(row, dict) or set(row) != set(packed):
                raise ContractError('Motor command requires tau, target and target_v')
            for key in packed:
                packed[key].append(_vector(row[key], name + ' ' + key))
        tau, target, target_v = (np.concatenate(packed[key]) for key in packed)
        self.qd.sync()
        self.solver.set_dofs_act_bias(tau, self._bias1, self._bias2, dofs_idx=self.dof_indices)
        self.solver.control_dofs_position_velocity(target, target_v, dofs_idx=self.dof_indices)
        force = self.solver.get_dofs_control_force(dofs_idx=self.dof_indices)
        if (not isinstance(force, self.torch.Tensor) or tuple(force.shape) != (27,)
                or force.dtype != self.dtype or force.device != self.device):
            raise ContractError('Unexpected public motor-force tensor shape, dtype or device')
        self.qd.sync()
        host = np.asarray(force.detach().cpu().numpy(), dtype=float).copy()
        if not np.isfinite(host).all():
            raise ContractError('Nonfinite native motor force')
        self._check(clock)
        self.applies += 1
        return {name: host[i * 9:(i + 1) * 9].copy() for i, name in enumerate(self.names)}

    def metadata(self):
        return {'schema_version': 'episode-public-motor-batch-v1', 'arm_order': list(self.names),
                'dof_indices': list(self.dof_indices), 'source_hashes': dict(self.sources),
                'applies': self.applies, 'public_setter_calls_per_apply': 2,
                'public_force_queries_per_apply': 1, 'host_transfers_per_apply': 1,
                'scalar_fallback': False, 'default_cuda_stream_required': True}
