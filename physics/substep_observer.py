"""Opt-in, version-pinned PRIVATE rigid-substep instrumentation.

This is not a public Genesis callback API. It wraps one built solver instance,
never edits installed files, and never calls collision detection or a setter.
The default is disabled. Source/fake-solver checks cannot establish real contact
coverage, collision-filter approval, trajectory equivalence or grade eligibility.
"""
from copy import deepcopy
from importlib import metadata
import inspect
import math
from pathlib import Path
import hashlib
import json
from time import perf_counter


PINNED_VERSION = "1.3.2"
PINNED_SOURCES = {
    "engine/scene.py": "d4b6b2dd11d6745147daaa3fc47e2f172d832c9bfc8c6b68576bb1d4415fce3c",
    "engine/simulator.py": "b385b363dfb56512ae189d3ad86fd9bd75acc7209e2cf4b9c934ec76e98073da",
    "engine/solvers/rigid/rigid_solver.py": "e076854af17262323041544ff692d366402a700357053618c8d89fed1e86b19f",
    "engine/solvers/rigid/collider/collider.py": "496e2f56acdfb7bd20822c993ba12d275824335759b49cbc965347a64c833cc0",
    "engine/solvers/rigid/collider/broadphase.py": "923e6ebdd3612233ea751d74c2f532ecdbe305c392a97782a13b9b66af8c739f",
    "engine/solvers/base_solver.py": "849a7dd38a36da96e68aad5d777bd29a200479946962424ca90dbc8299f9cbe6",
    "engine/couplers/legacy_coupler.py": "29b5778655bca346d74a1f54dff07a5188db22705ea6b384b09c35f202483b9b",
    "options/solvers.py": "2f799beaf1a5acb7aa55edcc063187f3e6a384ce1ea85092c2ee72f4e202e536",
}
CONTACT_FIELDS = ("link_a", "link_b", "geom_a", "geom_b", "penetration", "position", "normal", "force")


class ObservationError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise ObservationError(message)


def verify_installed_sources():
    """Read distribution metadata and eight source files without importing Genesis."""
    distribution = metadata.distribution("genesis-world")
    _require(distribution.version == PINNED_VERSION, "Unreviewed Genesis version")
    root = Path(distribution.locate_file("genesis")).resolve()
    observed = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in PINNED_SOURCES}
    _require(observed == PINNED_SOURCES, "Installed Genesis source differs from the reviewed private hook")
    return root, observed


def _runtime_contract(solver):
    root, hashes = verify_installed_sources()
    expected_classes = (
        (solver, "genesis.engine.solvers.rigid.rigid_solver", "RigidSolver", "engine/solvers/rigid/rigid_solver.py"),
        (solver.sim, "genesis.engine.simulator", "Simulator", "engine/simulator.py"),
        (solver.collider, "genesis.engine.solvers.rigid.collider.collider", "Collider", "engine/solvers/rigid/collider/collider.py"),
        (solver.sim.coupler, "genesis.engine.couplers.legacy_coupler", "LegacyCoupler", "engine/couplers/legacy_coupler.py"),
    )
    for instance, module, name, path in expected_classes:
        cls = type(instance)
        _require(cls.__module__ == module and cls.__name__ == name, "Unreviewed solver/coupler class")
        _require(Path(inspect.getsourcefile(cls)).resolve() == root/path, "Loaded class is outside the pinned source")
    sim = solver.sim
    _require(solver.is_built and solver.is_active, "Observer needs one built active rigid solver")
    _require(sim._rigid_only and not sim._requires_grad and not solver._requires_grad, "Only rigid-only forward simulation is reviewed")
    _require(solver.n_envs == 0, "Only the single unbatched scene is reviewed")
    _require(solver._enable_collision and not solver._disable_constraint and not solver._use_hibernation,
             "Collision must be enabled, constraints active and hibernation disabled")
    _require(type(sim.substeps) is int and sim.substeps > 0, "Invalid internal-substep count")
    _require(math.isfinite(solver.substep_dt) and solver.substep_dt > 0, "Invalid internal-substep duration")
    _require(sim.cur_substep_global == 0, "Install before the first internal substep to avoid a missing prefix")
    _require("substep" not in vars(solver), "Do not stack or replace an existing instance hook")
    _require(inspect.ismethod(solver.substep) and solver.substep.__func__ is type(solver).substep,
             "Unexpected substep callable")
    _require(tuple(inspect.signature(solver.substep).parameters) == ("f",), "Unreviewed substep signature")
    return {"genesis_version": PINNED_VERSION, "source_sha256": hashes,
            "substeps_per_outer_step": sim.substeps, "substep_dt_s": solver.substep_dt,
            "enable_self_collision": bool(solver._enable_self_collision),
            "enable_neutral_collision": bool(solver._enable_neutral_collision),
            "enable_adjacent_collision": bool(solver._enable_adjacent_collision)}


def _copy_contacts(collider):
    # The extractor reads current solver contacts. Calling detect_collision(),
    # detection() or clear() here would recompute/mutate the simulation instead.
    borrowed = collider.get_contacts(as_tensor=True, to_torch=False)
    _require(set(CONTACT_FIELDS) <= set(borrowed), "Contact extractor omitted a required raw field")
    copied = deepcopy(borrowed)  # numpy arrays can alias solver storage: copy now.
    result = {name: value.tolist() if hasattr(value, "tolist") else value for name, value in copied.items()}
    return _validate_contacts(result)


def _validate_contacts(result):
    """Validate an already-owned contact mapping without another native read."""
    count = len(result["geom_a"])
    _require(all(isinstance(result[name], list) and len(result[name]) == count for name in CONTACT_FIELDS),
             "Contact fields have different row counts")
    for name in CONTACT_FIELDS[:4]:
        _require(all(type(value) is int and value >= 0 for value in result[name]), "Invalid contact identity")
    for name in ("position", "normal", "force"):
        _require(all(isinstance(row, list) and len(row) == 3 and all(math.isfinite(value) for value in row)
                     for row in result[name]), "Invalid contact vector")
    _require(all(math.isfinite(value) for value in result["penetration"]), "Invalid contact penetration")
    return result


def _detection_clock(solver):
    sim=solver.sim
    clock=[sim.cur_substep_global,sim.cur_substep_local,sim.cur_step_global,float(sim.cur_t)]
    _require(all(type(x) is int and x>=0 for x in clock[:3]) and math.isfinite(clock[3]), "Invalid actual detection clock")
    return clock


def _field_integer(field, label):
    value=field.to_numpy()
    _require(getattr(value,'size',None)==1, "Unbatched scalar required: "+label)
    x=value.reshape(-1)[0].item()
    _require(type(x) is int and x>=0, "Native integer required: "+label)
    return x


def _detection_runtime_contract(solver, contract):
    sim=solver.sim
    _require(sim.dt==.0025 and sim.substeps==16 and sim.substeps_local==1
             and solver.substep_dt==.0025/16, "Only explicit rigid16 detection capture is supported")
    _require('_func_constraint_force' not in vars(solver)
             and inspect.ismethod(solver._func_constraint_force)
             and solver._func_constraint_force.__func__ is type(solver)._func_constraint_force
             and tuple(inspect.signature(solver._func_constraint_force).parameters)==(),
             "Unexpected or stacked native constraint hook")
    root=Path(metadata.distribution('genesis-world').locate_file('genesis')).resolve()
    _require(Path(inspect.getsourcefile(solver._func_constraint_force)).resolve()==root/'engine/solvers/rigid/rigid_solver.py',
             "Constraint phase callable is outside pinned native source")
    return solver._func_constraint_force


def _capture_state(callback):
    value=callback()
    owner=getattr(callback,'__self__',None)
    owned=False
    if owner is not None:
        from physics.rigid_pose_pack import RigidPosePack
        owned=(type(owner) is RigidPosePack
               and getattr(callback,'__func__',None) is RigidPosePack.read)
    # This exact producer validates the finite native array in _host, copies
    # its storage, and builds new JSON lists/dicts per read. Detection and post
    # still call it separately even when their native clocks are identical.
    state=value if owned else deepcopy(value)
    _require(type(state) is dict and {'body_world','arms'}<=set(state), "Capture needs actual body/arm state mapping")
    if not owned:
        try:json.dumps(state,allow_nan=False)
        except (TypeError,ValueError) as exc:raise ObservationError('Capture must return finite JSON host data') from exc
    return state


class RigidSubstepObserver:
    """Use with a JSON-serializable snapshot sink; disabled unless opted in.

    ``sink(snapshot)`` receives every raw contact row, including zero-contact
    observations, after the original internal rigid substep returns and before
    the next collision pass overwrites buffers. Contact data describes that
    substep's solver contact calculation, not continuous swept collision tests.
    The helper never declares physical coverage or filter approval by itself.
    """
    def __init__(self, solver, sink=None, *, enabled=False, detection_capture=None, profile=False, fence=None,
                 phase_readback=None):
        _require(type(profile) is bool, "Explicit observer profiling Boolean required")
        _require(fence is None or callable(fence), "Diagnostic fence must be a zero-argument callable")
        self.solver, self.sink, self.enabled = solver, sink, enabled
        self.observed_count = 0
        self.failed = False
        self.active = False
        self.contract = None
        self.last_global_ordinal = None
        self._original = None
        self._wrapper = None
        self.detection_capture=detection_capture
        self.phase_readback=phase_readback
        self.detection_count=0
        self.constraint_original_calls=0
        self._constraint_original=None
        self._constraint_wrapper=None
        self._pending_detection=None
        self._in_substep=False
        self.profile=profile
        self.fence=fence
        self._timing_s={}
        self._timing_calls={}

    def _add_timing(self,name,elapsed):
        self._timing_s[name]=self._timing_s.get(name,0.)+elapsed
        self._timing_calls[name]=self._timing_calls.get(name,0)+1

    def _timed_call(self,name,callback,*args):
        started=perf_counter()
        try:return callback(*args)
        finally:self._add_timing(name,perf_counter()-started)

    def _timed_fence(self,name):
        if self.fence is not None:self._timed_call(name,self.fence)

    def _timed_original(self,name,callback,*args):
        self._timed_fence('fence_before_'+name)
        started=perf_counter()
        try:
            return self._timed_call(name+'_host',callback,*args)
        finally:
            try:self._timed_fence('fence_after_'+name)
            finally:self._add_timing(name+'_inclusive_after_fence',perf_counter()-started)

    def timing_record(self):
        """Detached diagnostic host-wall accounting, not GPU kernel throughput.

        The original substep includes the detection callback and its nested
        constraint solve. Subtract the whole callback, then add back the
        original constraint span. After-fence waits remain in the synchronized
        native spans. Before-fence waits are separate exclusive components.
        """
        s=lambda name:self._timing_s.get(name,0.)
        native=s('original_substep_inclusive_after_fence')-s('detection_callback_total')+s('original_constraint_inclusive_after_fence')
        detection_other=(s('detection_callback_total')-s('original_constraint_inclusive_after_fence')
            -s('fence_before_original_constraint')-s('detection_contact_copy')
            -s('state_capture_detection_full')-s('fields_count_capacity_errno'))
        wrapper_other=(s('observed_substep_wrapper_total')-s('original_substep_inclusive_after_fence')
            -s('fence_before_original_substep')-s('state_capture_post_full')-s('post_contact_copy')-s('sink'))
        exclusive=dict(original_rigid_accounted_with_after_fence_wait=native,
            detection_contact_copy=s('detection_contact_copy'),state_capture_detection_full=s('state_capture_detection_full'),
            fields_count_capacity_errno=s('fields_count_capacity_errno'),detection_wrapper_other=detection_other,
            state_capture_post_full=s('state_capture_post_full'),post_contact_copy=s('post_contact_copy'),sink=s('sink'),
            substep_wrapper_other=wrapper_other,before_native_fence_wait=s('fence_before_original_substep')+s('fence_before_original_constraint'))
        total=s('observed_substep_wrapper_total');accounted=math.fsum(exclusive.values())
        return dict(schema_version='rigid-substep-observer-diagnostic-host-timing-v1',profile_enabled=self.profile,
            observer_enabled=self.enabled,diagnostic_fences_enabled=self.profile and self.fence is not None,
            raw_nested_host_wall_s=dict(self._timing_s),timed_calls=dict(self._timing_calls),exclusive_host_wall_s=exclusive,
            observed_wrapper_wall_s=total,exclusive_accounted_sum_s=accounted,accounting_error_s=total-accounted,
            accounting_consistent=all(v>=-1e-8 for v in exclusive.values()) and abs(total-accounted)<=1e-8,
            diagnostic_fence_wall_s={k:v for k,v in self._timing_s.items() if k.startswith('fence_')},
            contact_copy_total_s=s('detection_contact_copy')+s('post_contact_copy'),
            state_capture_total_s=s('state_capture_detection_full')+s('state_capture_post_full'),
            observed_internal_substeps=self.observed_count,native_constraint_calls=self.constraint_original_calls,
            failed=self.failed,active=self.active,
            scope='Diagnostic perf_counter host wall only. Original substep is inclusive of detection callback. Exclusive accounting subtracts that callback and adds the nested original constraint. State capture includes callback, deepcopy and finite-JSON validation. Sink includes its full classification/coverage work. Fence times are nested diagnostics: after-fence waits are already inside original spans, before-fence waits are separate. Timer/bookkeeping overhead remains in residuals and may slightly affect the native-accounted estimate. No pure GPU kernel time or normal-run throughput claim. Failed runs include attempted partial timings.')

    def __enter__(self):
        _require(not self.active and self._original is None, "Observer cannot be reused or nested")
        if not self.enabled:
            return self
        _require(callable(self.sink), "An enabled observer needs a snapshot sink")
        _require(self.phase_readback is None or callable(self.detection_capture),
                 'Consolidated phase packets require detection and post captures')
        self.contract = _runtime_contract(self.solver)
        if self.detection_capture is not None:
            _require(callable(self.detection_capture), "Detection capture must be a pure callable")
            if self.phase_readback is not None:
                from physics.rigid16_transfer_pack import Rigid16TransferPack
                from physics.rigid16_native_pack import NativeRigid16TransferPack
                _require(type(self.phase_readback) in (Rigid16TransferPack,NativeRigid16TransferPack)
                         and self.phase_readback.solver is self.solver
                         and getattr(self.detection_capture,'__self__',None) is self.phase_readback.pose_reader,
                         'Consolidated capture must bind this exact solver and pose reader')
            self._constraint_original=_detection_runtime_contract(self.solver,self.contract)
            self.contract={**self.contract,'detection_profile':'rigid16_bead_tc2h_v1',
                           'detection_phase':'after_native_constraint_before_rigid_pose_integration'}
            def detection_fields(contacts,clock):
                count=_field_integer(self.solver.collider._collider_state.n_contacts,'contact count')
                capacity=_field_integer(self.solver.collider._collider_info.max_contacts,'contact capacity')
                errno=_field_integer(self.solver._errno,'errno')
                _require(count==len(contacts['geom_a'])<=capacity and errno==0, "Detection buffer/count/errno invalid")
                _require(_detection_clock(self.solver)==clock, "Detection read changed actual clock")
                return count,capacity,errno
            def detected_constraint():
                if not (self._in_substep and self._pending_detection is None and not self.failed):
                    self.failed=True
                    raise ObservationError("Missing/double/out-of-substep constraint detection")
                _require(type(self.solver)._func_constraint_force is self._constraint_original.__func__, "Native constraint class method changed")
                clock=_detection_clock(self.solver)
                self.constraint_original_calls+=1
                result=(self._timed_original('original_constraint',self._constraint_original) if self.profile
                    else self._constraint_original())  # Exactly one original native constraint solve.
                _require(_detection_clock(self.solver)==clock, "Native constraint call advanced clock")
                if self.phase_readback is not None:
                    contacts,state,count,capacity,errno=(self._timed_call('phase_pre_transfer_full',
                        self.phase_readback.capture_pre,clock) if self.profile else self.phase_readback.capture_pre(clock))
                else:
                    contacts=(self._timed_call('detection_contact_copy',_copy_contacts,self.solver.collider) if self.profile
                        else _copy_contacts(self.solver.collider))
                    state=(self._timed_call('state_capture_detection_full',_capture_state,self.detection_capture) if self.profile
                        else _capture_state(self.detection_capture))
                    count,capacity,errno=(self._timed_call('fields_count_capacity_errno',detection_fields,contacts,clock)
                        if self.profile else detection_fields(contacts,clock))
                self._pending_detection={'schema_version':'rigid16-detection-snapshot-v1',
                    'profile_name':'rigid16_bead_tc2h_v1','preintegration_native_clock':clock,
                    'raw_contacts':contacts,'rigid_state':state,'native_contact_count':count,
                    'native_capacity':capacity,'native_errno':errno,'readback_did_not_change_clock':True}
                self.detection_count+=1
                return result
            self._constraint_wrapper=(lambda:self._timed_call('detection_callback_total',detected_constraint)) if self.profile else detected_constraint
            self.solver._func_constraint_force=self._constraint_wrapper
        self._original = self.solver.substep
        def observed_substep(f):
            try:
                ordinal = self.solver.sim.cur_substep_global
                expected = 0 if self.last_global_ordinal is None else self.last_global_ordinal+1
                _require(type(ordinal) is int and ordinal == expected, "Skipped or repeated global substep ordinal")
                if self.detection_capture is not None:
                    _require(not self.failed and self._pending_detection is None
                             and self.solver._func_constraint_force is self._constraint_wrapper
                             and type(self.solver).substep is self._original.__func__
                             and type(self.solver)._func_constraint_force is self._constraint_original.__func__,
                             "Detection hook failed or was replaced")
                self._in_substep=True
                try:result=(self._timed_original('original_substep',self._original,f) if self.profile
                    else self._original(f))  # Exactly one unchanged original invocation.
                finally:self._in_substep=False
                _require(self.solver.sim.cur_substep_global == ordinal, "Simulator advanced its counter inside the rigid call")
                if self.detection_capture is None:
                    contacts=(self._timed_call('post_contact_copy',_copy_contacts,self.solver.collider) if self.profile
                        else _copy_contacts(self.solver.collider))
                else:
                    detection=self._pending_detection
                    clock=_detection_clock(self.solver)
                    _require(detection is not None and detection['preintegration_native_clock']==clock
                             and clock==[ordinal,f,ordinal//16,ordinal*(.0025/16)]
                             and self.detection_count==self.constraint_original_calls==self.observed_count+1,
                             "Missing/repeated/wrong-clock native detection")
                    if self.phase_readback is not None:
                        post=(self._timed_call('phase_post_transfer_full',self.phase_readback.capture_post,clock)
                            if self.profile else self.phase_readback.capture_post(clock))
                    else:
                        post=(self._timed_call('state_capture_post_full',_capture_state,self.detection_capture) if self.profile
                            else _capture_state(self.detection_capture))
                    _require(_detection_clock(self.solver)==clock, "Postintegration read changed clock")
                    contacts=detection['raw_contacts']
                substeps = self.contract["substeps_per_outer_step"]
                snapshot={"global_substep_ordinal": ordinal, "outer_step_ordinal": ordinal//substeps,
                           "internal_substep_index": ordinal%substeps, "local_state_index": f,
                           "interval_start_s": ordinal*self.contract["substep_dt_s"],
                           "interval_end_s": (ordinal+1)*self.contract["substep_dt_s"],
                           "contact_count": len(contacts["geom_a"]), "contacts": contacts,
                           "observation_phase": "after_original_rigid_substep_before_next_detection"}
                if self.detection_capture is not None:
                    snapshot.update(detection=detection,post_integration_state=post,post_read_native_clock=clock,
                                    produced_post_global_substep=ordinal+1,
                                    observation_phase='after_rigid_substep_with_actual_detection_and_post_state')
                if self.profile:self._timed_call('sink',self.sink,snapshot)
                else:self.sink(snapshot)
                self._pending_detection=None
                self.last_global_ordinal = ordinal
                self.observed_count += 1
                return result
            except BaseException:
                self.failed = True
                raise
        self._wrapper=(lambda f:self._timed_call('observed_substep_wrapper_total',observed_substep,f)) if self.profile else observed_substep
        self.solver.substep=self._wrapper
        self.active = True
        return self

    def assert_observed_through(self, completed_outer_steps):
        """Check counts/counters only. This does not approve physical coverage."""
        _require(self.enabled and self.contract is not None and not self.failed, "Observer is disabled or failed")
        _require(type(completed_outer_steps) is int and completed_outer_steps >= 0, "Invalid completed outer-step count")
        expected = completed_outer_steps*self.contract["substeps_per_outer_step"]
        _require(self.observed_count == self.solver.sim.cur_substep_global == expected, "Internal-substep observation count mismatch")
        if self.active:
            _require(self.solver.substep is self._wrapper, "Substep observer was replaced during execution")
        result={"observed_internal_substeps": self.observed_count, "expected_internal_substeps": expected,
                "sequence_complete": True, "physical_coverage_verified": False, "filter_audit_approved": False}
        if self.detection_capture is not None:
            _require(self.detection_count==self.constraint_original_calls==expected and self._pending_detection is None,
                     "Native detection sequence incomplete")
            if self.active:_require(self.solver._func_constraint_force is self._constraint_wrapper, "Detection hook replaced")
            result.update(detection_observed_internal_substeps=self.detection_count,
                          constraint_original_calls=self.constraint_original_calls,
                          detection_profile='rigid16_bead_tc2h_v1')
        return result

    def __exit__(self, exception_type, exception, traceback):
        if self.active:
            owned = self.solver.substep is self._wrapper
            detection_owned=(self._constraint_wrapper is None or self.solver._func_constraint_force is self._constraint_wrapper)
            if self._constraint_wrapper is not None and detection_owned:
                delattr(self.solver,'_func_constraint_force')
                detection_owned=(getattr(self.solver._func_constraint_force,'__func__',None) is self._constraint_original.__func__)
            if owned:
                delattr(self.solver, "substep")  # Restore inherited method exactly.
                if self.detection_capture is not None:
                    owned=(getattr(self.solver.substep,'__func__',None) is self._original.__func__)
            else:
                self.failed = True
            self.active = False
            if not detection_owned:self.failed=True
            if (not owned or not detection_owned) and exception is None:
                raise ObservationError("Substep observer replaced externally; did not overwrite its replacement")
        return False
