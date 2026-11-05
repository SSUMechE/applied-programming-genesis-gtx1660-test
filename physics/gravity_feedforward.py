"""Teacher-only gravity torque from public Genesis model and Jacobian APIs.

This helper reads one built, unbatched entity. It never steps, resets or controls
the robot and never changes material gravity. The returned generalized effort
must be combined with the controller and clamped to the actual actuator limits
by the caller. It is gravity feedforward, not full inverse dynamics.

Every link is included, including fixed children and gripper fingers. World COM
and link poses are read explicitly. Local COM and mass must remain consistent
with the immutable model snapshot. Calls must not overlap a physics step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import inspect
import json
import math
from numbers import Integral
from pathlib import Path
import threading
from time import perf_counter

import numpy as np


QUATERNION_NORM_TOLERANCE = 1e-4
LOCAL_COM_TOLERANCE_M = 1e-5
MASS_RELATIVE_TOLERANCE = 1e-6
MASS_ABSOLUTE_TOLERANCE_KG = 1e-8

PACKED_SOURCE_PINS = {
    "engine/entities/rigid_entity/rigid_entity.py": "b0f86f9c9ebdf62b9df2e5aadc0719c28e06cd52e0ce1bef452b1a269d9b2092",
    "engine/solvers/rigid/rigid_solver.py": "e076854af17262323041544ff692d366402a700357053618c8d89fed1e86b19f",
    "engine/solvers/kinematic_solver.py": "c663445fd2dfbbf2286cec08166b1c45d34dd7e1bb9850d881c7b0b2d408ed20",
    "utils/misc.py": "ba5949408015aa7148dc1518977a958bca15e665b40b44243bfa5d5b2fa62454",
}


def _packed_dependencies(arm):
    """Load the optional CUDA path without importing its dependencies in scalar mode."""
    torch = importlib.import_module("torch")
    qd = importlib.import_module("quadrants")
    gs = importlib.import_module("genesis")
    root = Path(gs.__file__).resolve().parent
    if (type(arm).__name__ != "RigidEntity"
            or type(arm).__module__ != "genesis.engine.entities.rigid_entity.rigid_entity"
            or Path(inspect.getsourcefile(type(arm))).resolve()
            != root / "engine/entities/rigid_entity/rigid_entity.py"):
        raise ValueError("Packed gravity requires the reviewed public RigidEntity")
    for relative, expected in PACKED_SOURCE_PINS.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Packed gravity public getter source changed: " + relative)
    if (gs.use_zerocopy is not True or gs.backend != gs.cuda or gs.device.type != "cuda"
            or gs.tc_float not in (torch.float32, torch.float64)):
        raise ValueError("Packed gravity requires the reviewed zero-copy CUDA backend")
    device = gs.device if gs.device.index is not None else torch.device("cuda", torch.cuda.current_device())
    return torch, qd, arm.scene, device, gs.tc_float


def _contains_boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


def _array(value, shape, label):
    if _contains_boolean(value):
        raise ValueError(f"{label} cannot contain boolean values")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} must be a numeric array of shape {shape}")
    result = raw.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


def rotation_matrix_wxyz(quaternion):
    """Return local-to-world R for a finite near-unit WXYZ quaternion.

    Normalize only floating-point norm drift within the declared tolerance.
    A zero, malformed or substantially non-unit quaternion is rejected.
    """
    q = _array(quaternion, (4,), "world quaternion")
    norm = math.hypot(*q)
    if not math.isfinite(norm) or abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise ValueError("world quaternion must be unit length")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def world_to_local_point(point_world, origin_world, quaternion_wxyz):
    """Convert one world point to the link frame without changing the entity."""
    point = _array(point_world, (3,), "world COM")
    origin = _array(origin_world, (3,), "world link origin")
    with np.errstate(over="raise", invalid="raise"):
        result = rotation_matrix_wxyz(quaternion_wxyz).T @ (point - origin)
    return _array(result, (3,), "local COM")


@dataclass(frozen=True)
class LinkMassEvidence:
    index_local: int
    name: str
    mass_kg: float
    local_com_m: tuple[float, float, float]


@dataclass(frozen=True)
class GravityModelEvidence:
    schema_version: str
    n_dofs: int
    gravity_world_m_s2: tuple[float, float, float]
    links: tuple[LinkMassEvidence, ...]
    model_sha256: str
    initialization_elapsed_s: float
    initialization_public_calls: tuple[tuple[str, int], ...]
    quaternion_norm_tolerance: float = QUATERNION_NORM_TOLERANCE
    local_com_tolerance_m: float = LOCAL_COM_TOLERANCE_M
    mass_relative_tolerance: float = MASS_RELATIVE_TOLERANCE
    mass_absolute_tolerance_kg: float = MASS_ABSOLUTE_TOLERANCE_KG
    scope: str = "public_model_gravity_only_no_control_or_physical_acceptance"

    def to_record(self):
        """Return a detached JSON-serializable copy."""
        return asdict(self)


@dataclass(frozen=True)
class GravityStepEvidence:
    call_index: int
    status: str
    elapsed_s: float
    public_calls: tuple[tuple[str, int], ...]
    maximum_mass_change_kg: float | None
    maximum_local_com_change_m: float | None
    torque_generalized: tuple[float, ...] | None
    error: str | None
    jacobian_evaluation: dict | None = None

    def to_record(self):
        record=asdict(self)
        if record['jacobian_evaluation'] is None:record.pop('jacobian_evaluation')
        return record


class GravityFeedforward:
    """Compute -sum(J_COM_linear.T @ mass*gravity) for all entity-local DOFs.

    torques() returns a fresh one-dimensional NumPy vector (all nine DOFs for
    the current Panda). model_evidence and last_evidence are frozen dataclasses.
    A failed call raises and records FAILED with no torque vector. No stale
    successful vector is returned after a validation or public-API failure.
    """

    def __init__(self, arm, gravity_world, *, readback="scalar"):
        started = perf_counter()
        if readback not in ("scalar", "packed", "fused"):
            raise ValueError("Gravity readback must be scalar, packed or fused")
        self._readback = readback
        self._arm = arm
        self._gravity = tuple(map(float, _array(gravity_world, (3,), "world gravity")))
        n_dofs = arm.n_dofs
        if isinstance(n_dofs, bool) or not isinstance(n_dofs, Integral) or n_dofs <= 0:
            raise ValueError("arm.n_dofs must be a positive integer")
        self._n_dofs = int(n_dofs)
        self._links = tuple(arm.links)
        if not self._links or arm.n_links != len(self._links):
            raise ValueError("arm.links must contain every link in getter order")
        self._identities = self._link_identities()
        self._call_index = 0
        self._last_evidence = None
        self._fused_backend=None
        self._last_fused_inputs=None
        if readback in ("packed", "fused"):
            (self._packed_torch, self._packed_qd, self._packed_scene,
             self._packed_device, self._packed_input_dtype) = _packed_dependencies(arm)
            self._packed_thread = threading.get_ident()
            self._packed_lock = threading.Lock()
            self._packed_identity = None
        calls = {}
        masses, local_coms = self._read_model(calls)
        self._masses = masses.copy()
        self._local_coms = local_coms.copy()
        if readback == 'fused':
            from physics.gravity_jacobian_batch import NativeLinkJacobianBatch
            self._fused_backend=NativeLinkJacobianBatch(arm,local_coms)
            seed_calls=self._fused_backend.configuration_record()['initialization_public_calls']
            for name,count in seed_calls.items():
                if count:calls[name]=calls.get(name,0)+count
        link_records = tuple(LinkMassEvidence(
            index, name, float(masses[index]), tuple(map(float, local_coms[index]))
        ) for index, name in self._identities)
        identity_record = {
            "schema_version": "gravity-model-v1",
            "n_dofs": self._n_dofs,
            "gravity_world_m_s2": self._gravity,
            "links": [asdict(link) for link in link_records],
        }
        digest = hashlib.sha256(json.dumps(
            identity_record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        self._model_evidence = GravityModelEvidence(
            "gravity-model-v1", self._n_dofs, self._gravity, link_records,
            digest, perf_counter() - started, tuple(sorted(calls.items())),
        )

    @property
    def model_evidence(self):
        return self._model_evidence

    @property
    def last_evidence(self):
        """Most recent attempted torque call, or None before the first call."""
        return self._last_evidence

    def _link_identities(self):
        identities = []
        for expected, link in enumerate(self._links):
            index, name = link.idx_local, link.name
            if isinstance(index, bool) or not isinstance(index, Integral) or index != expected:
                raise ValueError("Link local indices must match public getter order")
            if not isinstance(name, str):
                raise ValueError("Each link needs a string name for evidence")
            identities.append((int(index), name))
        return tuple(identities)

    def _call(self, counts, name, *args, **kwargs):
        counts[name] = counts.get(name, 0) + 1
        return getattr(self._arm, name)(*args, **kwargs)

    def _read_model(self, calls):
        if self._readback in ("packed", "fused"):
            return self._read_model_packed(calls)
        n = len(self._links)
        masses = _array(self._call(calls, "get_links_inertial_mass"), (n,), "built link masses")
        if (masses < 0).any() or not (masses > 0).any():
            raise ValueError("Built masses must be nonnegative with positive total mass")
        origins = _array(self._call(calls, "get_links_pos", ref="link_origin", relative=False),
                         (n, 3), "world link origins")
        coms = _array(self._call(calls, "get_links_pos", ref="link_com", relative=False),
                      (n, 3), "world link COMs")
        quats = _array(self._call(calls, "get_links_quat", relative=False),
                       (n, 4), "world link quaternions")
        local_coms = np.stack([
            world_to_local_point(com, origin, quat)
            for com, origin, quat in zip(coms, origins, quats)
        ])
        return masses, local_coms

    def _packed_context(self, expected_clock=None):
        """Caller owns a resting scene on this thread, with no concurrent physics.

        Clock checks detect completed concurrent scene work. They cannot lock a
        foreign caller's simulator, so exclusive scene ownership remains required.
        """
        if threading.get_ident() != self._packed_thread:
            raise ValueError("Packed gravity must remain on its owning thread")
        scene = self._packed_scene
        if self._arm.scene is not scene or not scene.is_built or scene.n_envs != 0:
            raise ValueError("Packed gravity requires its original built unbatched scene")
        clock = (scene.t, scene.cur_t)
        if (isinstance(clock[0], (bool, np.bool_)) or not isinstance(clock[0], Integral)
                or clock[0] < 0 or isinstance(clock[1], (bool, np.bool_))
                or not isinstance(clock[1], (int, float, np.number))
                or not math.isfinite(float(clock[1])) or clock[1] < 0):
            raise ValueError("Packed gravity scene clock is invalid")
        if expected_clock is not None and clock != expected_clock:
            raise ValueError("Scene clock changed during packed gravity readback")
        if self._packed_device is not None:
            cuda = self._packed_torch.cuda
            if (cuda.current_stream(self._packed_device).cuda_stream
                    != cuda.default_stream(self._packed_device).cuda_stream):
                raise ValueError("Packed gravity requires the default CUDA stream")
        return clock

    def _packed_host(self, values, shapes, labels, *, stack=False):
        torch = self._packed_torch
        identity = None
        for value, shape, label in zip(values, shapes, labels):
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.dtype not in (torch.float32, torch.float64)):
                raise ValueError(f"{label} must be a floating public tensor of shape {shape}")
            current = (value.dtype, value.device)
            if identity is not None and current != identity:
                raise ValueError("Packed gravity public tensor dtype/device disagreement")
            if self._packed_device is not None and value.device != self._packed_device:
                raise ValueError("Packed gravity public tensor device changed")
            identity = current
        if self._packed_identity is not None and identity != self._packed_identity:
            raise ValueError("Packed gravity public tensor dtype/device changed")
        self._packed_identity = identity
        # Pinned public getters return independent copies. In particular each
        # Jacobian is already cloned before the native scratch field is reused.
        # Never borrow that field or add a redundant clone here.
        self._packed_qd.sync()
        packed = torch.stack(values) if stack else torch.cat([value.reshape(-1) for value in values])
        host = packed.detach().cpu().numpy().reshape(-1)
        expected = sum(math.prod(shape) for shape in shapes)
        if host.shape != (expected,):
            raise ValueError("Packed gravity host layout changed")
        arrays = []
        offset = 0
        for shape, label in zip(shapes, labels):
            size = math.prod(shape)
            arrays.append(_array(host[offset:offset + size].reshape(shape), shape, label))
            offset += size
        return arrays

    def _read_model_packed(self, calls):
        clock = self._packed_context()
        n = len(self._links)
        values = (
            self._call(calls, "get_links_inertial_mass"),
            self._call(calls, "get_links_pos", ref="link_origin", relative=False),
            self._call(calls, "get_links_pos", ref="link_com", relative=False),
            self._call(calls, "get_links_quat", relative=False),
        )
        masses, origins, coms, quats = self._packed_host(values,
            ((n,), (n, 3), (n, 3), (n, 4)),
            ("built link masses", "world link origins", "world link COMs", "world link quaternions"))
        if (masses < 0).any() or not (masses > 0).any():
            raise ValueError("Built masses must be nonnegative with positive total mass")
        local_coms = np.stack([
            world_to_local_point(com, origin, quat)
            for com, origin, quat in zip(coms, origins, quats)
        ])
        self._packed_context(clock)
        return masses, local_coms

    def _read_jacobians_packed(self, calls, local_coms):
        clock = self._packed_context()
        # Upload this call's freshly measured COMs once with the exact native
        # public getter conversion. Each row is already on its requested dtype
        # and device, so get_jacobian's as_tensor does not upload it again.
        points = self._packed_torch.as_tensor(local_coms, dtype=self._packed_input_dtype,
            device=self._packed_device if self._packed_device is not None else "cpu")
        values = tuple(self._call(calls, "get_jacobian", link, local_point=local_com)
                       for link, local_com in zip(self._links, points))
        shapes = ((6, self._n_dofs),) * len(self._links)
        result = self._packed_host(values, shapes, ("world COM Jacobian",) * len(values), stack=True)
        self._packed_context(clock)
        return result

    def fused_backend_configuration(self):
        return None if self._fused_backend is None else self._fused_backend.configuration_record()

    def verify_fused_jacobians(self):
        """Expensive same-clock diagnostic. Never used for normal control."""
        if self._readback!='fused' or self._last_fused_inputs is None:
            raise ValueError('No current successful fused matrices to verify')
        clock,local_coms,matrices=self._last_fused_inputs
        self._packed_context(clock)
        refs=np.stack([_array(self._arm.get_jacobian(link,local_point=point),
            (6,self._n_dofs),'public reference Jacobian') for link,point in zip(self._links,local_coms)])
        after=_array(self._fused_backend.read_output(),matrices.shape,'fused output after scratch queries')
        self._packed_context(clock)
        return {'links':len(self._links),'entries_compared':int(matrices.size),
            'public_reference_calls':len(self._links),'matrix_exactly_equal':bool(np.array_equal(matrices,refs)),
            'maximum_matrix_difference':float(np.max(np.abs(matrices-refs))),
            'output_not_aliased_to_scratch':bool(np.array_equal(matrices,after)),
            'same_native_clock':True}

    def torques(self):
        started = perf_counter()
        self._call_index += 1
        calls = {}
        max_mass_change = max_com_change = None
        packed_locked = False
        backend_attempted=False
        self._last_fused_inputs=None
        try:
            if self._readback in ("packed", "fused"):
                packed_locked = self._packed_lock.acquire(blocking=False)
                if not packed_locked:
                    raise ValueError("Packed gravity reads must not overlap")
                packed_clock = self._packed_context()
            if (self._arm.n_dofs != self._n_dofs
                    or self._arm.n_links != len(self._links)
                    or len(self._arm.links) != len(self._links)
                    or any(a is not b for a, b in zip(self._arm.links, self._links))
                    or self._link_identities() != self._identities):
                raise ValueError("Robot topology changed after model snapshot")
            masses, local_coms = self._read_model(calls)
            with np.errstate(over="raise", invalid="raise"):
                max_mass_change = float(np.max(np.abs(masses - self._masses)))
                max_com_change = float(np.max(np.linalg.norm(local_coms - self._local_coms, axis=1)))
            if not np.allclose(masses, self._masses, rtol=MASS_RELATIVE_TOLERANCE,
                               atol=MASS_ABSOLUTE_TOLERANCE_KG):
                raise ValueError("Built link mass changed after model snapshot")
            if not math.isfinite(max_com_change) or max_com_change > LOCAL_COM_TOLERANCE_M:
                raise ValueError("Link-local COM changed or world frames are inconsistent")
            torque = np.zeros(self._n_dofs)
            jacobians = self._read_jacobians_packed(calls, local_coms) if self._readback == "packed" else None
            if self._readback=='fused':
                backend_attempted=True
                matrices=_array(self._fused_backend.compute(local_coms),
                    (len(self._links),6,self._n_dofs),'fused native Jacobians')
                jacobians=matrices
            with np.errstate(over="raise", invalid="raise"):
                for index, (link, mass, local_com) in enumerate(zip(self._links, masses, local_coms)):
                    if jacobians is None:
                        jacobian = _array(self._call(calls, "get_jacobian", link, local_point=local_com),
                                          (6, self._n_dofs), "world COM Jacobian")
                    else:
                        jacobian = jacobians[index]
                    torque -= jacobian[:3].T @ (mass * np.asarray(self._gravity))
            torque = _array(torque, (self._n_dofs,), "gravity generalized effort")
            if self._readback in ("packed", "fused"):
                self._packed_context(packed_clock)
            if self._readback=='fused':self._last_fused_inputs=(packed_clock,local_coms.copy(),matrices)
            self._last_evidence = GravityStepEvidence(
                self._call_index, "COMPUTED_NOT_PHYSICALLY_VALIDATED", perf_counter() - started,
                tuple(sorted(calls.items())), max_mass_change, max_com_change,
                tuple(map(float, torque)), None,
                None if self._fused_backend is None else self._fused_backend.last_execution_record(),
            )
            return torque
        except Exception as exc:
            self._last_evidence = GravityStepEvidence(
                self._call_index, "FAILED", perf_counter() - started, tuple(sorted(calls.items())),
                max_mass_change, max_com_change, None, f"{type(exc).__name__}: {exc}",
                self._fused_backend.last_execution_record() if backend_attempted else None,
            )
            raise
        finally:
            if packed_locked:
                self._packed_lock.release()

