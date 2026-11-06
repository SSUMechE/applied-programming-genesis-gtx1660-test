"""Read-only, source-bound coverage of a built Genesis 1.3.2 rigid scene.

This is a pair-policy certificate, not contact-buffer completeness or continuous
collision proof. Build once after all entities/owners exist and before stepping.
The per-interval adapter must separately copy every retained native contact,
classify it and bind the exact pre/post clock and counts. No scene operation is
performed here. In particular there is no collision refresh, state setter or
physics step. The scene's source-bound pair metadata is immutable after build.
Dynamic constraints, error flags and runtime pruning are checked by each stamp.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import inspect
import json
from importlib import metadata
from pathlib import Path
import weakref

import numpy as np

from physics.handle_contacts import verify_contact_sources
from physics.self_collision_audit import adjacent_by_installed_rule, built_self_pair_ledger
from transport_challenge.contracts import ContractError


EXTRA_SOURCES = {
    "engine/simulator.py": "b385b363dfb56512ae189d3ad86fd9bd75acc7209e2cf4b9c934ec76e98073da",
    "engine/scene.py": "d4b6b2dd11d6745147daaa3fc47e2f172d832c9bfc8c6b68576bb1d4415fce3c",
    "utils/array_class.py": "459ff6c67a3080d5ebf71e746d59f6746e6e76f11a1d534dfbfc7a78dd049d34",
    "constants.py": "d8bbc696685b60b0238b35e1408814ef1da5918b0f65cd611df243c836fb1bc8",
    "engine/entities/rigid_entity/rigid_equality.py": "2a2143af7d644dd2917947fffbadd6933bd384d2dc93ccdf0f5ae197f9e2b369",
    "engine/solvers/rigid/abd/misc.py": "bb3d2c80cf68c3fe89536b1b798c5913f62390ef42f58747021b63cdbf9f8ca2",
}
ROLES = frozenset(("arm", "tray", "bead", "support", "floor", "obstacle"))
RIGID16_PROFILE = "rigid16_bead_tc2h_v1"
RIGID16_DT = .0025
RIGID16_H = RIGID16_DT/16
_BUILT = weakref.WeakKeyDictionary()
_READBACKS = weakref.WeakKeyDictionary()
_EQUALITY_ORDER = weakref.WeakKeyDictionary()
READBACK_API_SOURCES = {
    "genesis/utils/misc.py": "ba5949408015aa7148dc1518977a958bca15e665b40b44243bfa5d5b2fa62454",
    "quadrants/lang/field.py": "1e7b8dbbaade9f978bd1ba3cec1c10a56a0fc05e4d846e10785ae790fd0fcd3d",
    "quadrants/lang/_ndarray.py": "6e4db65b6611332ac1a316b3bfe9aa53805af09d2ea63bd68ecac600fdfb9889",
    "quadrants/_tensor_wrapper.py": "83796951254166bd841ed7624ba7520750d16913ced6f84ff851104e4591e0d8",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def verify_live_contact_sources():
    root, sources = verify_contact_sources()
    for name, expected in EXTRA_SOURCES.items():
        observed = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if observed != expected:
            raise ContractError(f"Unreviewed live contact source: {name}")
        sources[name] = observed
    return root, sources


def live_contact_readback_source_paths():
    """Verified absolute installed paths for caller snapshots, with SHA256.

    Quadrants lives beside Genesis, not below it. This read-only function does
    not import either engine or Torch. Freeze all returned paths before launch.
    """
    site = Path(metadata.distribution("genesis-world").locate_file("genesis")).resolve().parent
    paths = {}
    for name, expected in READBACK_API_SOURCES.items():
        path = site/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ContractError("Unreviewed native read API: " + name)
        paths[str(path)] = expected
    return paths


def _native_read_fields(scene):
    solver = scene.rigid_solver
    collider = solver.collider
    fields = [("pruning", collider._collider_info.contact_pruning_tolerance, None),
              ("static_equalities", solver.rigid_info.n_equalities, None),
              ("active_equalities", solver.constraint_solver.constraint_state.qd_n_equalities, 0),
              ("errno", solver._errno, 0)]
    # The pinned RigidSolver._init_equality_fields flattens this sequence into
    # native rows. The public RigidEquality.idx is documented as global, but
    # the observed pinned two-Panda build reports zero for both wrappers.
    # Preserve that value as evidence, never use it to address native rows.
    for ordinal, eq in enumerate(solver.equalities):
        for name in ("eq_type", "eq_obj1id", "eq_obj2id"):
            fields.append((f"equality_{ordinal}_{name}", getattr(solver.dyn_info.equalities, name), (ordinal, 0)))
    fields.extend((("contact_count", collider._collider_state.n_contacts, 0),
                   ("contact_capacity", collider._collider_info.max_contacts, None)))
    return fields


def _native_buffer(field):
    return field._unwrap() if hasattr(field, "_unwrap") else field


def _decode_native_pack(values, names):
    if len(values) != len(names) or not names or names[0] != "pruning":
        raise ContractError("Native read layout mismatch")
    result = {}
    for index, (name, value) in enumerate(zip(names, values)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ContractError("Nonfinite or malformed native read")
        if index:
            if value != int(value) or abs(value) > 2**53:
                raise ContractError("Native integer read lost precision")
            value = int(value)
        result[name] = value
    count, capacity = result["contact_count"], result["contact_capacity"]
    if capacity < 0 or not 0 <= count <= capacity:
        raise ContractError("Native contact count is outside its declared capacity")
    return result


class _BatchedReadback:
    """Scene-bound zero-copy views. No scalar fallback or native write."""

    def __init__(self, scene):
        sources = live_contact_readback_source_paths()
        import torch
        import quadrants as qd
        from genesis.utils.misc import qd_to_torch
        if str(Path(inspect.getsourcefile(qd_to_torch)).resolve()) not in sources:
            raise ContractError("Unreviewed imported native read API")
        self.fields = _native_read_fields(scene)
        self.buffers = [_native_buffer(field) for _, field, _ in self.fields]
        self.names = [name for name, _, _ in self.fields]
        self.views = []
        qd.sync()
        for name, field, index in self.fields:
            view = qd_to_torch(field, copy=False)
            if name == "contact_count" and tuple(view.shape) != (1,):
                raise ContractError("Expected one unbatched native contact count")
            value = view.reshape(()) if index is None else view[index]
            if value.shape != torch.Size([]):
                raise ContractError("Expected one native scalar")
            allowed = (torch.float32, torch.float64) if name == "pruning" else (torch.int32,)
            if value.dtype not in allowed:
                raise ContractError("Unreviewed native read dtype")
            self.views.append(value)
        if len({view.device for view in self.views}) != 1:
            raise ContractError("Native read fields use different devices")
        self.torch, self.qd = torch, qd
        self.record = {"schema_version": "native-contact-readback-v1", "mode": "batched",
                       "field_names": self.names, "dtypes": [str(v.dtype) for v in self.views],
                       "device": str(self.views[0].device), "source_hashes": sources,
                       "zero_copy_required": True, "quadrants_sync_per_stamp": 1,
                       "cpu_transfers_per_stamp": 1, "packed_dtype": "float64",
                       "scalar_fallback": False}

    def device_values(self, scene, *, synchronized=False):
        current = _native_read_fields(scene)
        if ([(name, index) for name, _, index in current] != [(name, index) for name, _, index in self.fields]
                or any(_native_buffer(field) is not original for (_, field, _), original in zip(current, self.buffers))):
            raise ContractError("Scene native read buffers changed after build")
        if not synchronized:self.qd.sync()
        # Float64 represents every actual int32 field exactly. Wider integer
        # values are checked before acceptance, never silently rounded to f32.
        packed = self.torch.stack([self.views[0].to(self.torch.float64), *self.views[1:]])
        if packed.dtype != self.torch.float64:
            raise ContractError("Native packed dtype changed")
        return packed

    def read(self, scene):
        packed=self.device_values(scene)
        return _decode_native_pack(packed.detach().cpu().numpy().tolist(), self.names)


def _verify_runtime_classes(scene, root):
    for obj, module, name, source in (
        (scene, "genesis.engine.scene", "Scene", "engine/scene.py"),
        (scene.sim, "genesis.engine.simulator", "Simulator", "engine/simulator.py"),
        (scene.rigid_solver, "genesis.engine.solvers.rigid.rigid_solver", "RigidSolver",
         "engine/solvers/rigid/rigid_solver.py"),
        (scene.rigid_solver.collider, "genesis.engine.solvers.rigid.collider.collider", "Collider",
         "engine/solvers/rigid/collider/collider.py"),
    ):
        cls = type(obj)
        if (cls.__module__ != module or cls.__name__ != name
                or Path(inspect.getsourcefile(cls)).resolve() != root/source):
            raise ContractError("Unreviewed live contact runtime class")


def _integer(value, label):
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iu":
        raise ContractError(f"{label} must be a scalar integer")
    return int(raw)


def _array(value, label):
    # qd.Tensor exposes its backing ndarray/field through _unwrap. Reading the
    # backing buffer directly avoids importing Genesis and does not run kernels.
    if hasattr(value, "_unwrap"):
        value = value._unwrap()
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    elif hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in "iu":
        raise ContractError(f"{label} must be an integer native buffer")
    return result.copy()


def _runtime(scene, native_read=None, *, runtime_profile=None):
    if runtime_profile not in (None, RIGID16_PROFILE):
        raise ContractError("Unknown live rigid runtime profile")
    solver = scene.rigid_solver
    info = solver.collider._collider_info
    if native_read is None:
        pruning = float(info.contact_pruning_tolerance[None])
        static_eq = _integer(solver.rigid_info.n_equalities[None], "native static equalities")
        live_eq = _integer(solver.constraint_solver.constraint_state.qd_n_equalities[0],
                           "native active equalities")
        errno = _integer(solver._errno[0], "native errno")
    else:
        pruning = native_read["pruning"]
        static_eq, live_eq, errno = (native_read[k] for k in ("static_equalities", "active_equalities", "errno"))
    equalities = []
    local_counts = Counter()
    for i, eq in enumerate(solver.equalities):
        entity_name = str(eq.entity.name)
        local_index = local_counts[entity_name]
        local_counts[entity_name] += 1
        native = solver.dyn_info.equalities
        observed = ([_integer(getattr(native, name)[i, 0], "native equality " + name)
                     for name in ("eq_type", "eq_obj1id", "eq_obj2id")] if native_read is None
                    else [native_read[f"equality_{i}_{name}"] for name in ("eq_type", "eq_obj1id", "eq_obj2id")])
        equalities.append({"idx": int(eq.idx), "public_wrapper_idx": int(eq.idx),
                           "native_row_index": i, "entity_local_index": local_index,
                           "entity": entity_name, "type": int(eq.type),
                           "obj1id": int(eq.eq_obj1id), "obj2id": int(eq.eq_obj2id),
                           "native_type": observed[0], "native_obj1id": observed[1], "native_obj2id": observed[2]})
    flags = {
        "built": bool(scene._is_built), "n_envs": int(solver.n_envs),
        "rigid_only": bool(scene.sim._rigid_only), "substeps": int(scene.sim.substeps),
        "use_hibernation": bool(solver._use_hibernation),
        "native_use_hibernation": bool(solver.rigid_config.use_hibernation),
        "requires_grad": bool(solver._requires_grad),
        "native_requires_grad": bool(solver.rigid_config.requires_grad),
        "enable_self_collision": bool(solver._enable_self_collision),
        "enable_neutral_collision": bool(solver._enable_neutral_collision),
        "enable_adjacent_collision": bool(solver._enable_adjacent_collision),
        "enable_collision": bool(solver._enable_collision),
        "native_enable_collision": bool(solver.rigid_config.enable_collision),
        "disable_constraint": bool(solver._disable_constraint),
        "contact_pruning_tolerance": solver._options.contact_pruning_tolerance,
        "native_pruning_tolerance": pruning,
        "static_equalities": static_eq, "active_equalities": live_eq,
        "equality_identity": equalities,
        "native_errno": errno, "geometry_count": int(solver.n_geoms),
    }
    failures = []
    if not flags["built"] or flags["n_envs"] != 0 or not flags["rigid_only"]:
        failures.append("requires_built_unbatched_rigid_only_scene")
    if runtime_profile == RIGID16_PROFILE:
        if (flags["substeps"] != 16 or float(scene.sim.dt) != RIGID16_DT
                or int(scene.sim.substeps_local) != 1):
            failures.append("requires_exact_named_rigid16_native_timestep")
        flags.update(runtime_profile=runtime_profile, dt_s=float(scene.sim.dt),
                     substeps_local=int(scene.sim.substeps_local))
    elif flags["substeps"] != 1:
        failures.append("requires_one_internal_substep_per_observation")
    if any(flags[key] for key in ("use_hibernation", "native_use_hibernation",
                                 "requires_grad", "native_requires_grad")):
        failures.append("hibernation_or_differentiable_detection_not_supported")
    if not flags["enable_self_collision"] or not flags["enable_neutral_collision"]:
        failures.append("self_and_neutral_collision_must_be_enabled")
    if not flags["enable_collision"] or not flags["native_enable_collision"] or flags["disable_constraint"]:
        failures.append("collision_and_constraint_solving_must_be_enabled")
    if flags["contact_pruning_tolerance"] is not None or pruning != 0.:
        failures.append("contact_pruning_not_explicitly_disabled")
    if static_eq != len(equalities) or live_eq != static_eq:
        failures.append("extra_or_mismatched_equality_constraints")
    if any(e["type"] != e["native_type"] or e["obj1id"] != e["native_obj1id"]
           or e["obj2id"] != e["native_obj2id"] for e in equalities):
        failures.append("native_equality_identity_mismatch")
    if errno != 0:
        failures.append("native_error_or_contact_capacity_failure")
    return flags, failures


def _geometry_record(geom, owner):
    link = geom.link
    expected = {"entity": str(geom.entity.name), "link": str(link.name), "link_id": int(link.idx)}
    if not isinstance(owner, dict) or owner.get("kind") not in ROLES:
        raise ContractError("Every collision geometry needs an explicit known role")
    if any(owner.get(k) != value for k, value in expected.items()):
        raise ContractError("Geometry ownership disagrees with the native entity/link")
    return {"geom_id": int(geom.idx), "kind": owner["kind"], **expected,
            "entity_idx": int(geom.entity.idx), "root_link_id": int(link.root_idx),
            "parent_link_id": int(link.parent_idx), "fixed_relative_to_world": bool(link.is_fixed),
            "contype": int(geom.contype), "conaffinity": int(geom.conaffinity),
            "local_collision_mask": bool(geom.entity.is_local_collision_mask)}


def build_live_contact_coverage(scene, *, arms, owners, readback="scalar", runtime_profile=None, payload_mode=None,
                                single_active_contract=None):
    """Return detached actual pair ledger. Unknown filtered pairs fail closed.

    Narrow exclusions are same-link solids, the pinned model's adjacent/fixed
    joint chain, and explicitly world-fixed pairs. Static pairs still require
    separate installation/initial-overlap validation. No dynamic hand/tray,
    tray/support, content/tray or content/content pair is exempted here.
    """
    if readback not in ("scalar", "batched"):
        raise ContractError("Native readback must be scalar or batched")
    if payload_mode not in (None,'empty') or payload_mode=='empty' and runtime_profile is not None:
        raise ContractError('Empty contact coverage requires explicit empty payload with the default rigid profile')
    root, sources = verify_live_contact_sources()
    _verify_runtime_classes(scene, root)
    reader = _BatchedReadback(scene) if readback == "batched" else None
    flags, failures = _runtime(scene, reader.read(scene) if reader is not None else None,
                               runtime_profile=runtime_profile)
    solver = scene.rigid_solver
    geoms = list(solver.geoms)
    if [int(g.idx) for g in geoms] != list(range(solver.n_geoms)):
        raise ContractError("Native geometries must have the exact contiguous scene IDs")
    normalized = {}
    for key, owner in owners.items():
        if isinstance(key, bool) or (not isinstance(key, (str, int, np.integer))):
            raise ContractError("Geometry owner keys must be exact integer IDs")
        idx = int(key)
        if str(idx) != str(key) or idx in normalized:
            raise ContractError("Ambiguous geometry owner ID")
        normalized[idx] = owner
    if set(normalized) != set(range(len(geoms))):
        raise ContractError("Owner ledger must cover every native collision geometry exactly")
    single=None
    native_mode=getattr(scene,'_ap_single_active_arm_contract',None)
    if native_mode is not None or single_active_contract is not None or len(arms)==1:
        from physics.fixed_peer_arms import validate_scene_contract
        from physics.episode_contacts import validate_single_active_contact_contract
        try:single=validate_scene_contract(scene,{arm.name:arm for arm in arms})
        except (KeyError,TypeError,ValueError) as exc:raise ContractError('Missing qualified single-active native scene: '+str(exc)) from exc
        if single_active_contract is not None and single_active_contract!=single:
            raise ContractError('Foreign single-active coverage contract')
        validate_single_active_contact_contract(single,{arm.name for arm in arms},normalized)
        import physics.fixed_peer_arms as fixed
        fixed_path=Path(fixed.__file__).resolve()
        sources[str(fixed_path)]=hashlib.sha256(fixed_path.read_bytes()).hexdigest()
    geometry = [_geometry_record(g, normalized[g.idx]) for g in geoms]
    required_roles=ROLES-{'bead'} if payload_mode=='empty' else ROLES
    if {r["kind"] for r in geometry} != required_roles:
        failures.append("required_role_family_missing")
    arm_ids = {int(g.idx) for arm in arms for g in arm.geoms}
    if not arms or arm_ids != {r["geom_id"] for r in geometry if r["kind"] == "arm"}:
        raise ContractError("Arms do not exactly match the actual ownership ledger")
    # The pinned Panda XML has exactly one JOINT coupling per arm. JOINT (2)
    # is not a WELD (1) and does not suppress collision pairs in broadphase.py.
    # Accept only those exact named joint identities, never arbitrary equality.
    expected_equalities = []
    for arm in arms:
        fingers = {j.name: int(j.idx) for j in arm.joints if j.name in ("finger_joint1", "finger_joint2")}
        if set(fingers) != {"finger_joint1", "finger_joint2"}:
            raise ContractError("Pinned Panda finger joint identities missing")
        expected_equalities.append((str(arm.name), 2, fingers["finger_joint1"], fingers["finger_joint2"]))
    actual_equalities = [(e["entity"], e["type"], e["obj1id"], e["obj2id"])
                         for e in flags["equality_identity"]]
    if sorted(actual_equalities) != sorted(expected_equalities):
        failures.append("unreviewed_static_equality_not_exact_panda_finger_coupling")
    # The reused adjacency helper uses n_dofs==0. Verify equivalence to the
    # pinned installed rule (JOINT_TYPE.FIXED==0) on every built model joint.
    for link in solver.links:
        if any((int(j.type) == 0) != (int(j.n_dofs) == 0) for j in link.joints):
            raise ContractError("Unreviewed fixed-joint adjacency semantics")
    host = _array(solver.collider._collision_pair_idx, "host pair matrix")
    native = _array(solver.collider._collider_info.collision_pair_idx, "native pair matrix")
    valid = _array(solver.collider._valid_collision_pairs, "host enabled pairs")
    expected_shape = (len(geoms), len(geoms))
    if host.shape != expected_shape or native.shape != expected_shape or not np.array_equal(host, native):
        raise ContractError("Native and host collision pair matrices disagree")
    if valid.ndim != 2 or valid.shape[1] != 2:
        raise ContractError("Malformed built pair list")
    rebuilt = np.full(expected_shape, -1, dtype=np.int64)
    for ordinal, (a, b) in enumerate(valid):
        if not 0 <= a < b < len(geoms) or rebuilt[a, b] != -1:
            raise ContractError("Built pairs must be unique ascending native geometry IDs")
        rebuilt[a, b] = ordinal
    if not np.array_equal(rebuilt, host):
        raise ContractError("Enabled pair list and native pair-index ledger disagree")
    native_valid = _array(solver.collider._collider_info.valid_collision_pairs, "native enabled pairs")
    if native_valid.shape != valid.shape or not np.array_equal(native_valid, valid):
        raise ContractError("Native enabled-pair array differs from built pairs")
    pairs = []
    for a, first in enumerate(geoms):
        for b in range(a+1, len(geoms)):
            second = geoms[b]
            enabled = bool(host[a, b] >= 0)
            exclusion = None
            if not enabled:
                if first.link.idx == second.link.idx:
                    exclusion = "same_rigid_link_compound_solids"
                elif first.link.is_fixed and second.link.is_fixed:
                    exclusion = "world_fixed_pair_requires_separate_installation_check"
                elif (a in arm_ids and b in arm_ids and first.entity is second.entity
                      and first.link.root_idx == second.link.root_idx
                      and not solver._enable_adjacent_collision
                      and adjacent_by_installed_rule(first.link, second.link, solver.links)):
                    exclusion = "pinned_robot_adjacent_or_fixed_chain"
                else:
                    exclusion = "UNEXPLAINED_REQUIRED_PAIR_FILTERED"
                    failures.append(f"required_pair_filtered:{a}:{b}")
            pairs.append({"geom_ids": [a, b], "roles": [geometry[a]["kind"], geometry[b]["kind"]],
                          "entities": [geometry[a]["entity"], geometry[b]["entity"]],
                          "enabled": enabled, "exclusion": exclusion})
    payload = {"schema_version": "live-contact-coverage-v1", "installed_version": "1.3.2",
               "policy_extension_version": "native-equality-ordinal-mapping-v2",
               "readback": copy.deepcopy(reader.record) if reader is not None else {
                   "schema_version": "native-contact-readback-v1", "mode": "scalar"},
               "source_hashes": sources, "runtime_at_build": flags,
               "geometry": geometry, "pairs": pairs,
               "counts": {"geometries": len(geoms), "all_unordered_pairs": len(pairs),
                          "enabled": len(valid), "excluded": len(pairs)-len(valid)},
               "exclusions_by_reason": dict(Counter(p["exclusion"] for p in pairs if not p["enabled"])),
               "native_pair_matrix_sha256": hashlib.sha256(host.astype("<i8").tobytes()).hexdigest(),
               "robot_self_pair_ledger": built_self_pair_ledger(solver, arms),
               "eligible_for_sampled_contact_assessment": not failures,
               "failures": failures,
               "scope": {"continuous_collision_proof": False, "buffer_completeness_proof": False,
                         "initial_static_overlap_checked": False, "contact_allowances_granted": False,
                         "pair_metadata_immutable_after_build": True,
                         "per_step_adapter_must_copy_count_and_classify_all_retained_contacts": True}}
    if runtime_profile == RIGID16_PROFILE:
        payload.update(schema_version="live-contact-coverage-rigid16-v1", runtime_profile=runtime_profile)
    if single is not None:
        if runtime_profile!=RIGID16_PROFILE or payload_mode is not None:
            raise ContractError('Single-active coverage is only the explicit rigid16 four-bead operation')
        payload['single_active_contract']=single
    if payload_mode=='empty':
        payload.update(schema_version='live-contact-coverage-empty-v1',payload_mode='empty',material_count=0,
            native_dt_s=float(scene.sim.dt),native_substeps_local=int(scene.sim.substeps_local))
    ledger = {**payload, "coverage_policy_sha256": _digest(payload)}
    # This cache binds subsequent stamps to this exact live scene. It contains
    # a detached ledger, never live tensors or mutable caller dictionaries.
    _BUILT[scene] = copy.deepcopy(ledger)
    # Retain the exact host object sequence. Comparing detached identity fields
    # also detects semantic changes, while this check rejects replacement or
    # reordering even if a caller edits host and native metadata together.
    _EQUALITY_ORDER[scene] = tuple(solver.equalities)
    if reader is not None:
        _READBACKS[scene] = reader
    else:
        _READBACKS.pop(scene, None)
    return ledger


def require_live_contact_coverage(ledger):
    """Reject malformed/tampered records and every unsupported pair policy."""
    if not isinstance(ledger, dict) or ledger.get("schema_version") not in ("live-contact-coverage-v1", "live-contact-coverage-rigid16-v1",'live-contact-coverage-empty-v1'):
        raise ContractError("Expected a live coverage ledger")
    if ledger.get('schema_version')=='live-contact-coverage-empty-v1':
        flags=ledger.get('runtime_at_build',{})
        if (ledger.get('payload_mode')!='empty' or type(ledger.get('material_count')) is not int
                or ledger['material_count']!=0 or ledger.get('runtime_profile') is not None
                or {r['kind'] for r in ledger.get('geometry',[])}!=ROLES-{'bead'}
                or flags.get('substeps')!=1 or ledger.get('native_dt_s')!=.0025
                or ledger.get('native_substeps_local')!=1):
            raise ContractError('Empty coverage requires exactly all non-content roles, zero contents and native substeps1')
    elif 'payload_mode' in ledger or 'material_count' in ledger:
        raise ContractError('Empty payload annotation requires its explicit coverage schema')
    if ledger.get("schema_version") == "live-contact-coverage-rigid16-v1":
        flags = ledger.get("runtime_at_build", {})
        if (ledger.get("runtime_profile") != RIGID16_PROFILE or flags.get("substeps") != 16
                or flags.get("dt_s") != RIGID16_DT or flags.get("substeps_local") != 1):
            raise ContractError("Rigid16 ledger requires the exact named profile")
    dynamic_ids={r['entity'] for r in ledger.get('geometry',[]) if r['kind']=='arm'}
    single=ledger.get('single_active_contract')
    if len(dynamic_ids)==1 or single is not None:
        from physics.episode_contacts import validate_single_active_contact_contract
        owners={r['geom_id']:r for r in ledger['geometry']}
        proof=validate_single_active_contact_contract(single,dynamic_ids,owners)
        if proof is None or ledger.get('schema_version')!='live-contact-coverage-rigid16-v1':
            raise ContractError('One dynamic arm needs the qualified fixed-peer rigid16 mode')
        if any(not r['fixed_relative_to_world'] for r in ledger['geometry'] if r['entity'] in proof['frozen_arm_ids']):
            raise ContractError('Qualified peer geometry is not actually world-fixed')
    payload = {k: v for k, v in ledger.items() if k != "coverage_policy_sha256"}
    if ledger.get("coverage_policy_sha256") != _digest(payload):
        raise ContractError("Live coverage ledger hash mismatch")
    if ledger.get("eligible_for_sampled_contact_assessment") is not True or ledger.get("failures") != []:
        raise ContractError("Live contact coverage is not eligible: " + str(ledger.get("failures")))
    return ledger["coverage_policy_sha256"]



AUTOMATIC_GRASP_HELPER_SHA256 = "7458d4f339da20d91d819f981cc8e30c3b018dd5cce375e9d61f32eb55e3c291"
AUTOMATIC_GRASP_SCOPE = "Native current-pose weld; standard entire hand-link/cup-link collision exclusion only. No bilateral-contact or no-weld certificate."


def _validate_automatic_grasp_flags(proof, flags):
    """Bind runtime flags to an already owned/fully validated witness only."""
    if flags is None:
        return
    baseline, current = proof["baseline_inventory"], proof["snapshot"]["inventory"]
    rows = [[e["native_type"],e["native_obj1id"],e["native_obj2id"]] for e in flags["equality_identity"]]
    if (rows != baseline["rows"] or flags["static_equalities"] != baseline["count"]
            or flags["active_equalities"] != current["count"]):
        raise ContractError("Automatic grasp native count/static identity differs")


def validate_automatic_grasp_witness(proof, *, coverage_policy_sha256=None, flags=None, rigid_profile=None):
    """Pure archive consistency check; live provenance comes from the owned helper."""
    if type(proof) is not dict:
        raise ContractError("Explicit automatic-grasp witness required")
    try:
        payload = {k: v for k, v in proof.items() if k != "witness_sha256"}
        expected_schema = ("provided-automatic-grasp-rigid16-internal-witness-v1"
                           if rigid_profile == RIGID16_PROFILE else "provided-automatic-grasp-witness-v1")
        if rigid_profile not in (None, RIGID16_PROFILE):
            raise ContractError("Unknown automatic witness profile")
        if (proof["schema_version"] != expected_schema
                or proof["witness_sha256"] != _digest(payload)
                or proof["helper_sha256"] != AUTOMATIC_GRASP_HELPER_SHA256
                or proof["model"] != "provided-ideal-native-weld-v1"
                or proof["scope"] != AUTOMATIC_GRASP_SCOPE
                or proof["bound_scene_object_verified"] is not True):
            raise ContractError("Invalid automatic-grasp witness identity")
        if coverage_policy_sha256 is not None and proof["coverage_policy_sha256"] != coverage_policy_sha256:
            raise ContractError("Automatic grasp belongs to another pair policy")
        clock = proof["snapshot"]["native_clock"]
        if (type(clock) is not list or len(clock) != 4 or any(type(clock[i]) is not int or clock[i] < 0 for i in (0,2,3))
                or type(clock[1]) not in (int,float) or not np.isfinite(clock[1])):
            raise ContractError("Automatic grasp native clock is malformed")
        snapshot = proof["snapshot"]
        hands = proof["hand_links_by_arm"]
        if (type(hands) is not dict or not hands or any(type(k) is not str or not k or type(v) is not int or v < 0 for k,v in hands.items())
                or len(set(hands.values())) != len(hands)
                or any(type(snapshot[k]) is not int or snapshot[k] < 0 for k in ("body_entity_idx","cup_link_idx","hand_link_idx"))
                or snapshot["hand_link_idx"] not in hands.values()
                or type(snapshot["cup_mass_kg"]) not in (int,float) or not np.isfinite(snapshot["cup_mass_kg"]) or snapshot["cup_mass_kg"] <= 0):
            raise ContractError("Automatic grasp body/hand identity or native mass is malformed")
        baseline, current = proof["baseline_inventory"], snapshot["inventory"]
        holder, row = proof["holder_arm_id"], proof["owned_weld_row"]
        for inv in (baseline,current):
            if (type(inv["count"]) is not int or type(inv["capacity"]) is not int
                    or not 0 <= inv["count"] <= inv["capacity"] or inv["count"] != len(inv["rows"])
                    or any(type(r) is not list or len(r) != 3 or any(type(v) is not int or v < 0 for v in r) for r in inv["rows"])
                    or inv["weld_pairs"] != [r[1:] for r in inv["rows"] if r[0] == 1]):
                raise ContractError("Automatic grasp ordered native inventory is malformed")
        if baseline["weld_pairs"] or any(r[0] != 2 for r in baseline["rows"]):
            raise ContractError("Automatic grasp baseline must be the unchanged Panda joint rows")
        if holder is None:
            if row is not None or current != baseline:
                raise ContractError("Unheld automatic grasp has an extra native equality")
        elif (type(holder) is not str or holder not in proof["hand_links_by_arm"]
                or row != [1,proof["hand_links_by_arm"][holder],proof["snapshot"]["cup_link_idx"]]
                or proof["snapshot"]["hand_link_idx"] != row[1]
                or current["capacity"] != baseline["capacity"]
                or current["rows"] != baseline["rows"]+[row]
                or current["weld_pairs"] != [row[1:]]):
            raise ContractError("Automatic holder/owned weld membership differs")
        _validate_automatic_grasp_flags(proof, flags)
        return holder
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("Malformed automatic-grasp witness: "+str(exc)) from exc



def validate_automatic_grasp_stamp(stamp, *, step, dt, substeps):
    """Call alongside the unchanged copied-contact/count/pruning checks."""
    schema = ("live-contact-policy-stamp-rigid16-automatic-grasp-outer-v1" if substeps == 16 else
              "live-contact-policy-stamp-automatic-grasp-v1" if substeps == 1 else "particle-contact-policy-stamp-automatic-grasp-v1")
    if (type(stamp) is not dict or type(substeps) is not int or substeps not in (1,16,100)
            or type(dt) not in (int,float) or not np.isfinite(dt) or dt <= 0
            or stamp.get("schema_version") != schema or type(step) is not int or step < 1):
        raise ContractError("Automatic policy cannot use a legacy no-weld stamp")
    if substeps == 16 and (dt != RIGID16_DT or stamp.get("runtime_profile") != RIGID16_PROFILE):
        raise ContractError("Automatic16 requires the named native profile")
    proof = stamp.get("automatic_grasp")
    validate_automatic_grasp_witness(proof, coverage_policy_sha256=stamp["coverage_policy_sha256"], flags=stamp.get("runtime_flags"))
    if (type(stamp["active_equalities"]) is not int or stamp["active_equalities"] != proof["snapshot"]["inventory"]["count"]
            or proof["snapshot"]["native_clock"] != [step,(step*substeps)*(dt/substeps),step*substeps,0]):
        raise ContractError("Automatic stamp count or actual outer post clock differs")
    return proof["holder_arm_id"]


def automatic_grasp_witness(helper, *, scene=None, coverage_policy_sha256=None, rigid_profile=None):
    """One fresh read of the exact trusted helper. Never accepts caller arrays."""
    from physics.provided_automatic_grasp import ProvidedAutomaticGrasp
    path = Path(inspect.getsourcefile(ProvidedAutomaticGrasp)).resolve()
    if (type(helper) is not ProvidedAutomaticGrasp or Path(inspect.getsourcefile(type(helper))).resolve() != path
            or hashlib.sha256(path.read_bytes()).hexdigest() != AUTOMATIC_GRASP_HELPER_SHA256
            or (scene is not None and helper.scene is not scene)
            or any(k in vars(helper) for k in ("snapshot","_inventory","_check_inventory"))):
        raise ContractError("Exact scene-bound provided automatic grasp required")
    arm = helper.holder if helper.holder is not None else next(iter(helper.arms))
    snapshot = helper.snapshot(arm)
    step, substeps = int(helper.scene.t), int(helper.scene.sim.substeps)
    expected_time = (step*substeps)*(float(helper.scene.sim.dt)/substeps)
    if rigid_profile is None:
        if snapshot["native_clock"] != [step,expected_time,step*substeps,0]:
            raise ContractError("Automatic grasp witness requires this completed outer clock")
    elif rigid_profile == RIGID16_PROFILE:
        clock = snapshot["native_clock"]
        if (substeps != 16 or float(helper.scene.sim.dt) != RIGID16_DT
                or type(clock[2]) is not int or clock != [clock[2]//16,clock[2]*RIGID16_H,clock[2],0]):
            raise ContractError("Automatic16 witness requires the actual internal native read clock")
    else:
        raise ContractError("Unknown automatic witness runtime profile")
    helper._check_inventory(snapshot)
    proof = dict(schema_version="provided-automatic-grasp-witness-v1", model="provided-ideal-native-weld-v1",
        helper_sha256=AUTOMATIC_GRASP_HELPER_SHA256, bound_scene_object_verified=True,
        coverage_policy_sha256=coverage_policy_sha256, holder_arm_id=helper.holder,
        owned_weld_row=copy.deepcopy(helper._owned_row), baseline_inventory=copy.deepcopy(helper._baseline),
        hand_links_by_arm={name:int(link.idx) for name,link in helper.hands.items()},
        snapshot=snapshot, scope=AUTOMATIC_GRASP_SCOPE)
    if rigid_profile == RIGID16_PROFILE:
        proof.update(schema_version="provided-automatic-grasp-rigid16-internal-witness-v1",
                     runtime_profile=rigid_profile, native_clock_scope="actual_internal_read_clock_not_produced_state_clock")
    proof["witness_sha256"] = _digest(proof)
    validate_automatic_grasp_witness(proof, coverage_policy_sha256=coverage_policy_sha256, rigid_profile=rigid_profile)
    return proof


def _automatic_policy(scene, ledger, helper, flags, *, rigid_profile=None):
    proof = automatic_grasp_witness(helper, scene=scene, coverage_policy_sha256=ledger["coverage_policy_sha256"], rigid_profile=rigid_profile)
    # The owned constructor above already verified the hash, holder and full
    # native inventory. External/archive callers still use the public validator.
    try:
        _validate_automatic_grasp_flags(proof, flags)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("Malformed automatic-grasp witness: "+str(exc)) from exc
    geometry = ledger["geometry"]
    cup = [g for g in geometry if g["link_id"] == proof["snapshot"]["cup_link_idx"]]
    if (len(cup) != 9 or any(g["kind"] != "tray" or g["entity_idx"] != proof["snapshot"]["body_entity_idx"] for g in cup)
            or any(not any(g["kind"] == "arm" and g["entity"] == arm and g["link_id"] == link for g in geometry)
                for arm,link in proof["hand_links_by_arm"].items())):
        raise ContractError("Automatic hand/cup geometry ownership differs from built pair policy")
    return proof


def live_contact_coverage_stamp(scene, ledger, *, automatic_grasp=None, internal_capture=None, internal_readback=None):
    """Small live policy stamp. It does NOT certify a copied contact buffer.

    The cached complete ledger is bound to the same live scene. Reusing a saved
    ledger on another scene fails. Its immutable native pair arrays were checked
    at build. Dynamic constraint counts, pruning, errno and flags are read now.
    """
    frozen = _BUILT.get(scene)
    reader = _READBACKS.get(scene)
    native_read = None
    automatic_proof = None
    failures = []
    profile = ledger.get("runtime_profile")
    if profile not in (None, RIGID16_PROFILE):
        raise ContractError("Unknown live policy profile")
    if internal_capture is not None and profile != RIGID16_PROFILE:
        raise ContractError("Internal captures require the explicit rigid16 ledger")
    if frozen is None or ledger.get("coverage_policy_sha256") != frozen["coverage_policy_sha256"]:
        failures.append("no_matching_built_scene_policy")
    elif frozen["eligible_for_sampled_contact_assessment"] is not True:
        failures.append("built_pair_policy_ineligible")
    try:
        order = _EQUALITY_ORDER.get(scene)
        current = tuple(scene.rigid_solver.equalities)
        if order is None or len(order) != len(current) or any(a is not b for a, b in zip(order, current)):
            raise ContractError("Host equality object order changed after build")
        if frozen is not None and frozen.get("readback", {}).get("mode") == "batched" and reader is None:
            raise ContractError("Batched readback is unavailable. Scalar fallback is prohibited")
        if internal_readback is not None:
            from physics.rigid16_transfer_pack import Rigid16TransferPack
            from physics.rigid16_native_pack import NativeRigid16TransferPack
            if (type(internal_readback) not in (Rigid16TransferPack,NativeRigid16TransferPack) or profile!=RIGID16_PROFILE
                    or internal_capture is None or internal_readback.policy_reader is not reader):
                raise ContractError('Untrusted or out-of-phase combined native policy read')
            native_read=internal_readback.claim_policy(scene,internal_capture)
        else:
            native_read = reader.read(scene) if reader is not None else None
        flags, runtime_failures = _runtime(scene, native_read, runtime_profile=profile)
        if automatic_grasp is not None:
            automatic_proof = _automatic_policy(scene, frozen, automatic_grasp, flags,
                rigid_profile=profile if internal_capture is not None else None)
            runtime_failures = [f for f in runtime_failures if f != "extra_or_mismatched_equality_constraints"]
        failures.extend(runtime_failures)
        if frozen is not None:
            for key, original in frozen["runtime_at_build"].items():
                if key != "native_errno" and not (automatic_proof is not None and key == "active_equalities") and flags[key] != original:
                    failures.append("runtime_policy_changed:"+key)
    except (AttributeError, TypeError, ValueError, ContractError) as exc:
        flags = {}
        failures.append("runtime_policy_unavailable:"+str(exc))
    result = {"schema_version": "live-contact-policy-stamp-v1",
            "coverage_policy_sha256": ledger.get("coverage_policy_sha256"),
            "policy_valid_now": not failures, "failures": failures,
            "pruning_disabled": flags.get("contact_pruning_tolerance", "missing") is None
                                and flags.get("native_pruning_tolerance") == 0.,
            "single_substep": flags.get("substeps") == 1,
            "native_errno": flags.get("native_errno"),
            "active_equalities": flags.get("active_equalities"),
            "enable_collision": flags.get("enable_collision"),
            "native_enable_collision": flags.get("native_enable_collision"),
            "disable_constraint": flags.get("disable_constraint"),
            "readback_mode": "batched" if reader is not None else "scalar",
            "native_contact_count": native_read.get("contact_count") if native_read is not None else None,
            "native_contact_capacity": native_read.get("contact_capacity") if native_read is not None else None,
            "contact_buffer_completeness_asserted": False}
    if automatic_grasp is not None:
        result.update(schema_version="live-contact-policy-stamp-automatic-grasp-v1", automatic_grasp=automatic_proof)
    if profile == RIGID16_PROFILE:
        result.update(runtime_profile=profile, single_substep=False, substeps=16,
                      runtime_flags=copy.deepcopy(flags))
        if internal_capture is None:
            result["schema_version"] = ("live-contact-policy-stamp-rigid16-automatic-grasp-outer-v1"
                                        if automatic_grasp is not None else "live-contact-policy-stamp-rigid16-outer-v1")
        else:
            g = internal_capture.get("global_substep_ordinal")
            clock = [int(scene.sim.cur_substep_global),int(scene.sim.cur_substep_local),int(scene.t),float(scene.cur_t)]
            if type(g) is not int or clock != [g,0,g//16,g*RIGID16_H]:
                raise ContractError("Rigid16 stamp must read this actual internal g clock")
            result.update(schema_version="live-contact-policy-stamp-rigid16-automatic-grasp-internal-v1"
                          if automatic_grasp is not None else "live-contact-policy-stamp-rigid16-internal-v1",
                          internal_native_read_clock=clock, produced_post_global_substep=g+1,
                          automatic_pose_phase="post_integration_state_g_plus_1_native_counter_still_g")
            if native_read is None:
                result.update(native_contact_count=native_live_contact_count(scene),
                    native_contact_capacity=_integer(scene.rigid_solver.collider._collider_info.max_contacts[None], "native contact capacity"))
    return result


def validate_rigid16_policy_stamp(stamp, *, global_substep, post_state=None,
                                  coverage_policy_sha256=None):
    """Pure exact16 archive contract. Actual provenance belongs to the live reader.

    Native clock remains g while the captured post pose is produced state g+1.
    Neither the completed outer stamp nor a particle100 stamp can stand in.
    """
    g = global_substep
    schemas = ("live-contact-policy-stamp-rigid16-internal-v1",
               "live-contact-policy-stamp-rigid16-automatic-grasp-internal-v1")
    if (type(g) is not int or g < 0 or type(stamp) is not dict
            or stamp.get("schema_version") not in schemas
            or stamp.get("runtime_profile") != RIGID16_PROFILE
            or stamp.get("substeps") != 16 or stamp.get("single_substep") is not False
            or stamp.get("internal_native_read_clock") != [g,0,g//16,g*RIGID16_H]
            or stamp.get("produced_post_global_substep") != g+1
            or stamp.get("policy_valid_now") is not True or stamp.get("failures") != []
            or stamp.get("pruning_disabled") is not True or type(stamp.get("native_errno")) is not int
            or stamp.get("native_errno") != 0):
        raise ContractError("Expected the exact actual internal rigid16 policy stamp")
    identity = stamp.get("coverage_policy_sha256")
    if (type(identity) is not str or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)
            or (coverage_policy_sha256 is not None and identity != coverage_policy_sha256)):
        raise ContractError("Rigid16 stamp belongs to another or missing pair policy")
    count, capacity = stamp.get("native_contact_count"), stamp.get("native_contact_capacity")
    if type(count) is not int or type(capacity) is not int or not 0 <= count <= capacity:
        raise ContractError("Rigid16 native buffer count/capacity missing")
    flags = stamp.get("runtime_flags", {})
    if (flags.get("runtime_profile") != RIGID16_PROFILE or flags.get("substeps") != 16
            or flags.get("dt_s") != RIGID16_DT or flags.get("substeps_local") != 1
            or flags.get("native_errno") != 0 or flags.get("native_pruning_tolerance") != 0.
            or flags.get("contact_pruning_tolerance", "missing") is not None
            or flags.get("enable_collision") is not True or flags.get("native_enable_collision") is not True
            or flags.get("disable_constraint") is not False):
        raise ContractError("Rigid16 native policy flags changed")
    if (any(flags.get(k) is not True for k in ('built','rigid_only','enable_self_collision','enable_neutral_collision'))
            or any(flags.get(k) is not False for k in ('use_hibernation','native_use_hibernation','requires_grad','native_requires_grad'))
            or type(flags.get('n_envs')) is not int or flags['n_envs'] != 0
            or type(flags.get('static_equalities')) is not int or type(flags.get('active_equalities')) is not int
            or type(flags.get('equality_identity')) is not list or flags['static_equalities'] != len(flags['equality_identity'])
            or any(e['type'] != e['native_type'] or e['obj1id'] != e['native_obj1id'] or e['obj2id'] != e['native_obj2id'] for e in flags['equality_identity'])):
        raise ContractError("Rigid16 original built model/equality guards changed")
    if stamp["schema_version"] == schemas[1]:
        proof = stamp.get("automatic_grasp")
        validate_automatic_grasp_witness(proof, coverage_policy_sha256=identity,
                                         flags=flags, rigid_profile=RIGID16_PROFILE)
        if (proof.get("runtime_profile") != RIGID16_PROFILE
                or proof["snapshot"]["native_clock"] != [g//16,g*RIGID16_H,g,0]
                or stamp.get("active_equalities") != proof["snapshot"]["inventory"]["count"]):
            raise ContractError("Rigid16 automatic owned-weld inventory/read clock mismatch")
        if post_state is None or proof["snapshot"]["body_world"] != post_state.get("body_world"):
            raise ContractError("Rigid16 automatic proof is not this actual produced post body")
        names = [n for n,i in proof["hand_links_by_arm"].items() if i == proof["snapshot"]["hand_link_idx"]]
        if len(names) != 1 or proof["snapshot"]["hand_world"] != post_state["arms"][names[0]]["hand_world"]:
            raise ContractError("Rigid16 automatic proof is not this actual produced post hand")
    elif flags.get("active_equalities") != flags.get("static_equalities"):
        raise ContractError("Rigid16 no-weld stamp contains a dynamic equality")
    return identity


def native_live_contact_count(scene):
    """Read the retained unbatched count used by installed get_contacts().

    Requires a source-verified build ledger for this live scene. This is the
    count after native capacity handling, not the number of all geometric
    witnesses that might exist. The caller must also retain the policy stamp's
    errno result, and compare this count with its copied/classified rows.
    """
    if scene not in _BUILT or scene.rigid_solver.n_envs != 0:
        raise ContractError("Contact count needs this source-verified unbatched scene")
    collider = scene.rigid_solver.collider
    buffer = _array(collider._collider_state.n_contacts, "native contact count")
    if buffer.shape != (1,):
        raise ContractError("Expected one native contact count for the unbatched scene")
    count = int(buffer[0])
    capacity = _integer(collider._collider_info.max_contacts[None], "native contact capacity")
    if capacity < 0 or not 0 <= count <= capacity:
        raise ContractError("Native contact count is outside its declared capacity")
    return count
