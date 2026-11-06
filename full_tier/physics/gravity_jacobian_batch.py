"""Source-pinned native Jacobian collection for the instructor CUDA runner.

One collector invokes Genesis' installed Jacobian function for every link in
getter order. Links execute serially against native scratch. Every completed
matrix is copied to its own output region before the scratch is reused.

The caller owns a resting scene on the constructing thread. No physics, IK or
other Jacobian query may overlap a call. This module does not compute gravity
torque, alter physics or establish physical acceptance.
"""

import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import math
from numbers import Integral
from pathlib import Path
import threading
from time import perf_counter

import numpy as np


GENESIS_SOURCE_PINS = {
    "engine/entities/rigid_entity/rigid_entity.py": "b0f86f9c9ebdf62b9df2e5aadc0719c28e06cd52e0ce1bef452b1a269d9b2092",
    "engine/solvers/rigid/rigid_solver.py": "e076854af17262323041544ff692d366402a700357053618c8d89fed1e86b19f",
    "engine/solvers/kinematic_solver.py": "c663445fd2dfbbf2286cec08166b1c45d34dd7e1bb9850d881c7b0b2d408ed20",
    "utils/geom.py": "57affcc68874030ba0f0562516137d1a5c6373ae606037b2de34d4349a1499e3",
    "utils/array_class.py": "459ff6c67a3080d5ebf71e746d59f6746e6e76f11a1d534dfbfc7a78dd049d34",
    "utils/misc.py": "ba5949408015aa7148dc1518977a958bca15e665b40b44243bfa5d5b2fa62454",
}
QUADRANTS_SOURCE_PINS = {
    "lang/kernel_impl.py": "9d0032e0437dbbea81a8ff86038b107450ec5461c9e1521420498cd57de2ec6d",
    "lang/kernel.py": "7213e562567e73527ac7d313bdaea66bea0aa9df956d734f042c787bd4ab2865",
    "lang/_func_base.py": "8cd1c0de1eb42ffe1702a040f918617f00286d4ff2f96521b78c631e30451089",
}
EXPECTED_VERSIONS = {
    "torch": "2.14.0+cu126", "genesis-world": "1.3.2", "quadrants": "1.2.0",
}
_KERNEL_CACHE = {}


def _load_dependencies():
    torch = importlib.import_module("torch")
    qd = importlib.import_module("quadrants")
    gs = importlib.import_module("genesis")
    arrays = importlib.import_module("genesis.utils.array_class")
    misc = importlib.import_module("genesis.utils.misc")
    return torch, qd, gs, arrays, misc


def _verify_sources(root, pins, label):
    for relative, expected in pins.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Native Jacobian {label} source changed: {relative}")


def _collector_kernel(qd, gs, arrays):
    key = (id(qd), id(gs), id(arrays))
    if key not in _KERNEL_CACHE:
        # No future annotations: the decorator must receive actual ABI types.
        @qd.kernel
        def collect(
            arm: qd.template(),
            points: qd.types.ndarray(),
            output: qd.types.ndarray(),
            dyn_state: arrays.DynState,
            dyn_info: arrays.DynInfo,
        ):
            # This is the only top-level loop. Each environment owns one native
            # scratch slice. Nested link iterations must never be parallelized.
            for i_b in range(arm._solver._B):
                for i_l in range(arm.n_links):
                    p_vec = qd.Vector(
                        [points[i_l, 0], points[i_l, 1], points[i_l, 2]], dt=gs.qd_float
                    )
                    arm._impl_get_jacobian(
                        arm._link_start + i_l, i_b, p_vec, dyn_state, dyn_info
                    )
                    for row, dof in qd.ndrange(6, arm.n_dofs):
                        output[i_l, row, dof, i_b] = arm._jacobian[row, dof, i_b]

        _KERNEL_CACHE[key] = collect
    return _KERNEL_CACHE[key]


def _numeric_points(value, n_links):
    def contains_boolean(item):
        if isinstance(item, (bool, np.bool_)):
            return True
        return isinstance(item, (list, tuple)) and any(contains_boolean(part) for part in item)

    if contains_boolean(value):
        raise ValueError("Native Jacobian local COMs cannot be boolean")
    raw = np.asarray(value)
    if raw.shape != (n_links, 3) or raw.dtype.kind not in "iuf":
        raise ValueError(f"Native Jacobian local COMs need numeric shape {(n_links, 3)}")
    if not np.isfinite(raw).all():
        raise ValueError("Native Jacobian local COMs must be finite")
    return np.ascontiguousarray(raw)


class NativeLinkJacobianBatch:
    """Collect all native link Jacobians without changing their math.

    compute returns a fresh CPU array in native precision, shaped (links, 6, 9).
    read_output rereads the last GPU result without running the collector. A
    failed attempt invalidates that result. No fallback or stale result is used.
    """

    def __init__(self, arm, initial_local_coms):
        started = perf_counter()
        self._last_record = None
        self._valid_output = False
        self._call_index = 0
        self._invalidation_epoch = 0
        self._owner_thread = threading.get_ident()
        self._lock = threading.Lock()
        self._arm = arm
        self._torch, self._qd, self._gs, arrays, misc = _load_dependencies()
        torch, qd, gs = self._torch, self._qd, self._gs
        gs_root = Path(gs.__file__).resolve().parent
        qd_root = Path(qd.__file__).resolve().parent
        if (type(arm).__name__ != "RigidEntity"
                or type(arm).__module__ != "genesis.engine.entities.rigid_entity.rigid_entity"
                or Path(inspect.getsourcefile(type(arm))).resolve()
                != gs_root / "engine/entities/rigid_entity/rigid_entity.py"):
            raise ValueError("Native Jacobian requires the reviewed original RigidEntity")
        _verify_sources(gs_root, GENESIS_SOURCE_PINS, "Genesis")
        _verify_sources(qd_root, QUADRANTS_SOURCE_PINS, "Quadrants")
        versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
        if versions != EXPECTED_VERSIONS or str(torch.__version__) != EXPECTED_VERSIONS["torch"]:
            raise ValueError("Native Jacobian package versions changed")
        if (gs.use_zerocopy is not True or gs.backend != gs.cuda or gs.device.type != "cuda"
                or gs.tc_float not in (torch.float32, torch.float64)
                or gs.qd_float not in (qd.f32, qd.f64)):
            raise ValueError("Native Jacobian requires the reviewed zero-copy CUDA precision")
        if (gs.tc_float == torch.float32) != (gs.qd_float == qd.f32):
            raise ValueError("Native Jacobian native and Torch precision disagree")
        self._dtype = gs.tc_float
        self._qd_dtype = gs.qd_float
        self._device = gs.device if gs.device.index is not None else torch.device(
            "cuda", torch.cuda.current_device()
        )
        self._scene = arm.scene
        self._solver = arm._solver
        self._dyn_state = self._solver.dyn_state
        self._dyn_info = self._solver.dyn_info
        self._links = tuple(arm.links)
        self._n_links = len(self._links)
        self._n_dofs = arm.n_dofs
        if (isinstance(self._n_dofs, bool) or self._n_dofs != 9
                or self._n_links <= 0 or arm.n_links != self._n_links
                or not arm._requires_jac_and_IK):
            raise ValueError("Native Jacobian requires the reviewed nonempty nine-DOF model")
        self._link_start = arm.link_start
        self._link_identities = tuple(
            (link.idx_local, link.idx, link.name) for link in self._links
        )
        if any(local != index or global_ != self._link_start + index
               for index, (local, global_, _) in enumerate(self._link_identities)):
            raise ValueError("Native Jacobian link indices must match native getter order")
        self._methods = tuple((name, inspect.getattr_static(type(arm), name)) for name in (
            "_impl_get_jacobian", "_func_get_jacobian",
        ))
        self._scratch = None
        self._scratch_view = None
        self._output = None
        clock = self._context()
        initial = _numeric_points(initial_local_coms, self._n_links)
        initialization_public_calls = 0
        if arm._jacobian is None:
            # Use Genesis' ordinary lazy allocation, without editing its fields.
            arm.get_jacobian(self._links[0], local_point=initial[0])
            initialization_public_calls = 1
        self._scratch = arm._jacobian
        self._qd.sync()
        self._scratch_view = misc.qd_to_torch(self._scratch, copy=False)
        self._scratch_pointer = self._scratch_view.data_ptr()
        self._output = torch.empty(
            (self._n_links, 6, self._n_dofs, 1), dtype=self._dtype, device=self._device
        )
        self._output_pointer = self._output.data_ptr()
        if self._output_pointer == self._scratch_pointer:
            raise ValueError("Native Jacobian output cannot alias scratch")
        self._kernel = _collector_kernel(qd, gs, arrays)
        self._context(clock)
        self._configuration = {
            "schema_version": "native-link-jacobian-batch-config-v1",
            "versions": versions,
            "genesis_source_pins": dict(GENESIS_SOURCE_PINS),
            "quadrants_source_pins": dict(QUADRANTS_SOURCE_PINS),
            "backend": "cuda", "zero_copy": True,
            "precision": str(self._dtype), "device": str(self._device),
            "n_links": self._n_links, "n_dofs": self._n_dofs, "batch_size": 1,
            "link_identities": [list(item) for item in self._link_identities],
            "output_shape": list(self._output.shape),
            "scratch_shape": list(self._scratch.shape),
            "output_data_ptr": self._output_pointer,
            "scratch_data_ptr": self._scratch_pointer,
            "initialization_public_calls": {"get_jacobian": initialization_public_calls},
            "initialization_elapsed_s": perf_counter() - started,
            "kernel_compile": "lazy_first_compute_normal_settings_no_graph",
            "summation": "outside_collector_original_CPU_float64_link_order",
            "scope": "native_matrices_only_no_physical_acceptance",
        }

    def _tensor(self, value, shape, label, *, pointer=None, contiguous=False):
        torch = self._torch
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != self._dtype or value.device != self._device
                or value.requires_grad or (contiguous and not value.is_contiguous())
                or (pointer is not None and value.data_ptr() != pointer)):
            raise ValueError(f"Native Jacobian {label} shape/dtype/device/binding changed")

    def _context(self, expected_clock=None):
        if threading.get_ident() != self._owner_thread:
            raise ValueError("Native Jacobian must stay on its owning thread")
        arm, scene, sol, gs = self._arm, self._scene, self._solver, self._gs
        if (arm.scene is not scene or arm._solver is not sol
                or not scene.is_built or scene.n_envs != 0 or sol._B != 1
                or sol.dyn_state is not self._dyn_state or sol.dyn_info is not self._dyn_info):
            raise ValueError("Native Jacobian scene/solver ABI changed")
        if (gs.use_zerocopy is not True or gs.backend != gs.cuda
                or gs.tc_float != self._dtype or gs.qd_float != self._qd_dtype
                or gs.device.type != "cuda"
                or self._torch.device(gs.device.type, self._torch.cuda.current_device()
                    if gs.device.index is None else gs.device.index) != self._device):
            raise ValueError("Native Jacobian backend/precision/device changed")
        cuda = self._torch.cuda
        if (cuda.current_stream(self._device).cuda_stream
                != cuda.default_stream(self._device).cuda_stream):
            raise ValueError("Native Jacobian requires the default CUDA stream")
        if (arm.n_dofs != self._n_dofs or arm.n_links != self._n_links
                or arm.link_start != self._link_start or len(arm.links) != self._n_links
                or any(a is not b for a, b in zip(arm.links, self._links))
                or tuple((link.idx_local, link.idx, link.name) for link in arm.links)
                != self._link_identities or not arm._requires_jac_and_IK):
            raise ValueError("Native Jacobian topology changed")
        if any(inspect.getattr_static(type(arm), name) is not method or name in vars(arm)
               for name, method in self._methods):
            raise ValueError("Native Jacobian installed function binding changed")
        if self._scratch is not None:
            if (arm._jacobian is not self._scratch
                    or tuple(self._scratch.shape) != (6, self._n_dofs, 1)
                    or self._scratch.dtype != self._qd_dtype):
                raise ValueError("Native Jacobian scratch binding changed")
        if self._scratch_view is not None:
            self._tensor(self._scratch_view, (6, self._n_dofs, 1), "scratch view",
                         pointer=self._scratch_pointer)
        if self._output is not None:
            self._tensor(self._output, (self._n_links, 6, self._n_dofs, 1), "output",
                         pointer=self._output_pointer, contiguous=True)
        clock = (scene.t, scene.cur_t)
        if (isinstance(clock[0], (bool, np.bool_)) or not isinstance(clock[0], Integral)
                or clock[0] < 0 or isinstance(clock[1], (bool, np.bool_))
                or not isinstance(clock[1], (int, float, np.number))
                or not math.isfinite(float(clock[1])) or clock[1] < 0):
            raise ValueError("Native Jacobian scene clock is invalid")
        if expected_clock is not None and clock != expected_clock:
            raise ValueError("Native Jacobian scene clock changed during collection")
        return clock

    def _host_output(self):
        self._qd.sync()
        host = self._output.detach().cpu().numpy()[..., 0].copy()
        expected_dtype = np.dtype("float32" if self._dtype == self._torch.float32 else "float64")
        if (host.shape != (self._n_links, 6, self._n_dofs)
                or host.dtype != expected_dtype or not np.isfinite(host).all()):
            raise ValueError("Native Jacobian host output is malformed or nonfinite")
        return host

    def compute(self, local_coms):
        started = perf_counter()
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            self._valid_output = False
            self._invalidation_epoch += 1
            self._last_record = {
                "schema_version": "native-link-jacobian-batch-step-v1",
                "call_index": self._call_index, "status": "FAILED",
                "logical_evaluations": 0, "collector_invocations": 0,
                "public_calls": {}, "clock": None,
                "elapsed_s": perf_counter() - started,
                "error": "ValueError: Native Jacobian collection must not overlap",
            }
            raise ValueError("Native Jacobian collection must not overlap")
        self._call_index += 1
        invalidation_epoch = self._invalidation_epoch
        self._valid_output = False
        invocations = 0
        logical = 0
        clock = None
        try:
            clock = self._context()
            values = _numeric_points(local_coms, self._n_links)
            points = self._torch.as_tensor(values, dtype=self._dtype, device=self._device)
            self._tensor(points, (self._n_links, 3), "local COM input", contiguous=True)
            invocations = 1
            logical = None
            self._kernel(self._arm, points, self._output, self._dyn_state, self._dyn_info)
            result = self._host_output()
            logical = self._n_links
            self._context(clock)
            if invalidation_epoch != self._invalidation_epoch:
                raise ValueError("Overlapping attempt invalidated native Jacobian collection")
            self._valid_output = True
            self._output_clock = clock
            self._last_record = {
                "schema_version": "native-link-jacobian-batch-step-v1",
                "call_index": self._call_index, "status": "COLLECTED_NOT_PHYSICALLY_VALIDATED",
                "logical_evaluations": logical, "collector_invocations": invocations,
                "public_calls": {}, "clock": list(clock),
                "elapsed_s": perf_counter() - started, "error": None,
            }
            return result
        except Exception as exc:
            self._valid_output = False
            self._last_record = {
                "schema_version": "native-link-jacobian-batch-step-v1",
                "call_index": self._call_index, "status": "FAILED",
                "logical_evaluations": logical, "collector_invocations": invocations,
                "public_calls": {}, "clock": list(clock) if clock is not None else None,
                "elapsed_s": perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
            raise
        finally:
            self._lock.release()

    def read_output(self):
        """Copy the last successful GPU output without querying native scratch."""
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            self._valid_output = False
            self._invalidation_epoch += 1
            self._last_record = None
            raise ValueError("Native Jacobian output read must not overlap")
        try:
            if not self._valid_output:
                raise ValueError("Native Jacobian has no successful current output")
            self._context(self._output_clock)
            result = self._host_output()
            self._context(self._output_clock)
            return result
        except Exception:
            self._valid_output = False
            self._last_record = None
            raise
        finally:
            self._lock.release()

    def configuration_record(self):
        return copy.deepcopy(self._configuration)

    def last_execution_record(self):
        return copy.deepcopy(self._last_record)
