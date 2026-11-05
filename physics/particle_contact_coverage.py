"""Separate coupled-particle rigid-contact pair policy and sampled coverage.

Only the last25us preintegration rigid contact/pose sample in each2.5ms outer
interval is assessed. Neither all100 internal intervals nor continuous safety
is certified. Particle/rigid coupling is not a rigid contact-buffer witness.
This module performs readback only. It never steps, sets or refreshes physics.
The unchanged rigid-only live_contact_coverage contract is not reused as a
coupled-mode certificate, and actual mode flags are never rewritten.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import weakref

import numpy as np

from physics import live_contact_coverage as base
from physics.handle_contacts import validate_contact_snapshot
raise RuntimeError('This public fixed example supports four rigid beads only')
from transport_challenge.contracts import ContractError

DT, SUBSTEPS = .0025, 100
INTERNAL_DT = DT/SUBSTEPS
ROLES = frozenset(("arm", "tray", "support", "floor", "obstacle"))
CONTACT_COPY_SHA = "df80451f98fbe0058ea5754ee5eb1142cfc39a52a8da8e6e6ec60518ad68aafe"
NATIVE_PINS = {
    "engine/solvers/rigid/rigid_solver.py": "e076854af17262323041544ff692d366402a700357053618c8d89fed1e86b19f",
    "engine/solvers/mpm_solver.py": "3f4e29b668a77c10e276f499e9d40259001bf61ae448a65061874b1efd5ccd5f",
    "engine/solvers/sph_solver.py": "7e06ddbdafcc3f607798b4dba16da653a859bc0b66fbbc1ea9cde1f17db69849",
    "engine/couplers/legacy_coupler.py": "29b5778655bca346d74a1f54dff07a5188db22705ea6b384b09c35f202483b9b",
    "engine/simulator.py": "b385b363dfb56512ae189d3ad86fd9bd75acc7209e2cf4b9c934ec76e98073da"}
QA = Path(__file__).resolve().parents[1]/"qa/all_material_physics_20261031"
COPY_PATH = QA/"runtime_bridge_branch/coupled_contact_sample_candidate.py"
PROFILES = {
    "sparse_mpm": dict(slot="mpm_solver", material="ScenePrivateSparseMPM", coupler="ScenePrivateSparseLegacyCoupler",
        private=("performance_branch/private_sparse_mpm_candidate.py", "90bba1570f08dae3a54f7a42bfbd578f5d43bdf7071ab1303bb77f0fd92e8106")),
    "process_liquid": dict(slot="sph_solver", material="CupDensityMapSPHSolver", coupler="CupDensityMapLegacyCoupler",
        private=("liquid_integration_branch/private_density_map_canonical_candidate.py", "36d228f91b291bc8205a3fa987c58c08fa7685471f6d0ba489eab85db6b3a933")),
    "cpic_cache_mpm": dict(slot="mpm_solver", material="ScenePrivateCPICLatticeCacheMPM", coupler="ScenePrivateSparseLegacyCoupler",
        private=("cpic_cache_branch/private_cpic_lattice_cache_candidate.py", "ffa5e4f971185f845ea308b43601713b2e4be9d24c28045aecba7cd2b63f8b96"),
        base_private=("performance_branch/private_sparse_mpm_candidate.py", "90bba1570f08dae3a54f7a42bfbd578f5d43bdf7071ab1303bb77f0fd92e8106"))}
SAMPLE_SCOPE = dict(outer_dt_s=DT, internal_substeps=SUBSTEPS, internal_dt_s=INTERNAL_DT,
    sampled_internal_offset=SUBSTEPS-1, rigid_only=False,
    observation_phase="after_native_constraint_before_rigid_pose_integration",
    last_internal_only=True, all_internal_substeps_covered=False,
    continuous_collision_proof=False, particle_rigid_coupling_is_contact_buffer=False)
LAST_SAMPLE_SCOPE = "Only final25us rigid preintegration contact sample in this2.5ms outer interval. No continuous/100-subinterval certificate."
_BUILT = weakref.WeakKeyDictionary()
_READBACKS = weakref.WeakKeyDictionary()
_EQUALITY_ORDER = weakref.WeakKeyDictionary()
_BINDINGS = weakref.WeakKeyDictionary()
_TOPOLOGY = weakref.WeakKeyDictionary()
_digest = base._digest
_integer = base._integer


def _require(value, message):
    if not value:
        raise ContractError(message)


def _source_bundle(profile):
    _require(type(profile) is str and profile in PROFILES, "Explicit allowed coupled particle profile required")
    root, sources = base.verify_live_contact_sources()
    sources = dict(sources)
    _require(contact_copy.NATIVE_PINS == NATIVE_PINS
             and contact_copy.NATIVE.resolve() == root.resolve()
             and contact_copy.QA.resolve() == QA.resolve(), "Contact-copy source globals differ")
    for relative, expected in NATIVE_PINS.items():
        path = root/relative
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                 "Unreviewed coupled native source: "+relative)
        sources[str(path)] = expected
    private, private_sha = PROFILES[profile]["private"]
    for path, expected in ((COPY_PATH, CONTACT_COPY_SHA), (QA/private, private_sha)):
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                 "Unreviewed coupled profile/copy source: "+str(path))
        sources[str(path)] = expected
    if profile == "cpic_cache_mpm":
        for path, expected in contact_copy._profile_sources(profile).items():
            _require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected,
                     "Unreviewed complete cache source closure: "+path)
            sources[path] = expected
    _require(Path(contact_copy.__file__).resolve() == COPY_PATH.resolve()
             and contact_copy.PROFILES[profile] == PROFILES[profile]
             and contact_copy.DT == DT and contact_copy.SUBSTEPS == SUBSTEPS,
             "Imported last-internal contact implementation differs")
    contact_copy._definition_check(contact_copy.LastInternalContactCopy, "__enter__", COPY_PATH)
    for module in (base, __import__("physics.self_collision_audit", fromlist=["x"])):
        path = Path(module.__file__).resolve()
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = Path(__file__).resolve()
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return root, sources


def _authenticate_profile(scene, profile, root):
    """Capture c137's exact native phases and aliases without installing a hook."""
    base._verify_runtime_classes(scene, root)
    guard = contact_copy.LastInternalContactCopy(scene, lambda: None, profile=profile)
    spec = PROFILES[profile]
    s, r = scene.sim, scene.rigid_solver
    c, m = s.coupler, getattr(s, spec["slot"])
    classes = contact_copy._native_classes()
    material_base = classes["SPHSolver" if spec["slot"] == "sph_solver" else "MPMSolver"]
    _require(type(r) is classes["RigidSolver"] and type(s) is classes["Simulator"],
             "Exact native rigid/Simulator instances required")
    if profile == "cpic_cache_mpm":
        contact_copy._authenticate_cache_family(scene, material_base, classes["LegacyCoupler"])
    else:
        _require(type(m).__name__ == spec["material"] and type(c).__name__ == spec["coupler"]
                 and type(m).__bases__ == (material_base,) and type(c).__bases__ == (classes["LegacyCoupler"],),
                 "Unreviewed coupled material/coupler class pair")
    _require(m.is_active and r.is_active and getattr(c, "_rigid_sph" if spec["slot"] == "sph_solver" else "_rigid_mpm") is True,
             "Actual rigid/particle coupling must be active")
    guard.sim, guard.coupler, guard.material = s, c, m
    guard.solvers, guard.active_solvers = tuple(s._solvers), tuple(s._active_solvers)
    _require(len(guard.active_solvers) == 2 and set(map(id, guard.active_solvers)) == {id(r), id(m)}
             and sum(x is r for x in guard.solvers) == sum(x is m for x in guard.solvers) == 1,
             "Exactly one rigid and one allowed particle solver must be active")
    guard.slot_aliases = {name: getattr(s, name) for name in contact_copy.SOLVER_SLOTS}
    guard.methods = []
    rigid_path = root/"engine/solvers/rigid/rigid_solver.py"
    guard.original = r._func_constraint_force
    _require("_func_constraint_force" not in vars(r) and inspect.ismethod(guard.original)
             and guard.original.__func__ is type(r)._func_constraint_force
             and tuple(inspect.signature(guard.original).parameters) == (),
             "Build before installing a contact-copy hook")
    contact_copy._definition_check(type(r), "_func_constraint_force", rigid_path)
    guard._capture_methods(r, ("substep", "substep_pre_coupling", "substep_post_coupling"), rigid_path)
    guard._capture_methods(s, ("substep", "substep_pre_coupling", "substep_post_coupling"), root/"engine/simulator.py")
    private_path = QA/spec["private"][0]
    for obj, names in ((m, ("substep_pre_coupling", "substep_post_coupling")), (c, ("preprocess", "couple"))):
        for name in names:
            owner = next(cls for cls in type(obj).__mro__ if name in cls.__dict__)
            path = private_path if owner is type(obj) else root/("engine/couplers/legacy_coupler.py" if obj is c
                else "engine/solvers/sph_solver.py" if spec["slot"] == "sph_solver" else "engine/solvers/mpm_solver.py")
            contact_copy._definition_check(owner, name, path)
            guard._capture_methods(obj, (name,), path, definitions_owned=False)
    anchor = "init_grid_fields" if profile in ("sparse_mpm", "cpic_cache_mpm") else "substep_pre_coupling"
    kernel = "mpm_grid_op" if profile in ("sparse_mpm", "cpic_cache_mpm") else "sph_rigid"
    contact_copy._definition_check(type(m), anchor, private_path)
    coupler_path = QA/spec["base_private"][0] if profile == "cpic_cache_mpm" else private_path
    contact_copy._definition_check(type(c), kernel, coupler_path)
    guard._capture_methods(c, (kernel,), coupler_path, definitions_owned=False)
    if profile == "cpic_cache_mpm": contact_copy._capture_cache_methods(guard, m)
    guard._check_bindings()
    return guard


def _check_profile(binding):
    binding._check_bindings()
    r = binding.solver
    if "_func_constraint_force" in vars(r):
        method = r._func_constraint_force
        _require(inspect.ismethod(method), "Unexpected instance constraint callback")
        nonlocals = inspect.getclosurevars(method.__func__).nonlocals
        observer = nonlocals.get("self")
        _require(type(observer) is contact_copy.LastInternalContactCopy
                 and observer.scene is binding.scene and observer.profile == binding.profile
                 and observer.solver is r and observer.active is True and observer.wrapper is method
                 and observer.original.__func__ is binding.original.__func__
                 and method.__func__.__qualname__ == "LastInternalContactCopy.__enter__.<locals>.copied_after_original"
                 and Path(inspect.getsourcefile(method.__func__)).resolve() == COPY_PATH.resolve(),
                 "Only the source-pinned last-internal copy may wrap the native constraint")
        observer._check_bindings()


def _geometry_record(geom, owner):
    _require(type(owner) is dict and owner.get("kind") in ROLES,
             "Every actual rigid geometry needs an exact allowed nonparticle role")
    return base._geometry_record(geom, owner)


def _capture_topology(scene, arms, geometry):
    r = scene.rigid_solver
    collider = r.collider
    return dict(solver=r, collider=collider, geoms=tuple(r.geoms), links=tuple(r.links), arms=tuple(arms),
        geometry=copy.deepcopy(geometry), host_pairs=collider._collision_pair_idx,
        host_valid=collider._valid_collision_pairs,
        native_pairs=base._native_buffer(collider._collider_info.collision_pair_idx),
        native_valid=base._native_buffer(collider._collider_info.valid_collision_pairs),
        host_matrix_sha256=hashlib.sha256(base._array(collider._collision_pair_idx, "host pairs").astype("<i8").tobytes()).hexdigest(),
        host_valid_sha256=hashlib.sha256(base._array(collider._valid_collision_pairs, "host pairs").astype("<i8").tobytes()).hexdigest(),
        arm_geoms=tuple(tuple(arm.geoms) for arm in arms),
        arm_joints=tuple(tuple(arm.joints) for arm in arms),
        finger_identity=tuple(tuple((j.name, int(j.idx)) for j in arm.joints
            if j.name in ("finger_joint1", "finger_joint2")) for arm in arms),
        link_joint_semantics=tuple(tuple((int(j.type), int(j.n_dofs)) for j in link.joints) for link in r.links))


def _same_objects(first, second):
    return len(first) == len(second) and all(a is b for a, b in zip(first, second))


def _check_topology(scene, frozen):
    r, c = scene.rigid_solver, scene.rigid_solver.collider
    _require(r is frozen["solver"] and c is frozen["collider"]
             and _same_objects(tuple(r.geoms), frozen["geoms"]) and _same_objects(tuple(r.links), frozen["links"]),
             "Built rigid geometry/link object identity changed")
    _require(c._collision_pair_idx is frozen["host_pairs"] and c._valid_collision_pairs is frozen["host_valid"]
             and base._native_buffer(c._collider_info.collision_pair_idx) is frozen["native_pairs"]
             and base._native_buffer(c._collider_info.valid_collision_pairs) is frozen["native_valid"],
             "Built host/native pair buffer identity changed")
    _require(hashlib.sha256(base._array(c._collision_pair_idx, "host matrix").astype("<i8").tobytes()).hexdigest() == frozen["host_matrix_sha256"]
             and hashlib.sha256(base._array(c._valid_collision_pairs, "host pairs").astype("<i8").tobytes()).hexdigest() == frozen["host_valid_sha256"],
             "Built host pair metadata changed")
    actual = [_geometry_record(g, owner) for g, owner in zip(r.geoms, frozen["geometry"])]
    _require(actual == frozen["geometry"], "Built geometry ownership/masks changed")
    for arm, geoms, joints in zip(frozen["arms"], frozen["arm_geoms"], frozen["arm_joints"]):
        _require(_same_objects(tuple(arm.geoms), geoms) and _same_objects(tuple(arm.joints), joints),
                 "Arm geometry/joint object identity changed")
    _require(tuple(tuple((j.name, int(j.idx)) for j in arm.joints
        if j.name in ("finger_joint1", "finger_joint2")) for arm in frozen["arms"]) == frozen["finger_identity"]
        and tuple(tuple((int(j.type), int(j.n_dofs)) for j in link.joints) for link in r.links) == frozen["link_joint_semantics"],
        "Built finger identity or fixed-joint adjacency semantics changed")


def _runtime(scene, profile, native_read=None):
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
        "sim_requires_grad": bool(scene.sim.requires_grad),
        "substeps_local": int(scene.sim.substeps_local), "dt_s": float(scene.sim.dt),
        "rigid_substep_dt_s": float(solver.substep_dt),
        "material_substep_dt_s": float(getattr(scene.sim, PROFILES[profile]["slot"]).substep_dt),
    }
    failures = []
    if not flags["built"] or flags["n_envs"] != 0 or flags["rigid_only"]:
        failures.append("requires_built_unbatched_coupled_particle_scene")
    if (flags["substeps"] != SUBSTEPS or flags["substeps_local"] != 1
            or flags["dt_s"] != DT or abs(flags["rigid_substep_dt_s"]-INTERNAL_DT) > 1e-15
            or abs(flags["material_substep_dt_s"]-INTERNAL_DT) > 1e-15):
        failures.append("requires_coupled_100_internal_25us_steps_per_outer_interval")
    if any(flags[key] for key in ("use_hibernation", "native_use_hibernation",
                                 "requires_grad", "native_requires_grad", "sim_requires_grad")):
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


def build_particle_contact_coverage(scene, *, arms, owners, profile, readback="batched"):
    """Return detached actual pair ledger. Unknown filtered pairs fail closed.

    Narrow exclusions are same-link solids, the pinned model's adjacent/fixed
    joint chain, and explicitly world-fixed pairs. Static pairs still require
    separate installation/initial-overlap validation. No dynamic hand/tray,
    tray/support pair is exempted here. Particle coupling is not included in
    this rigid pair matrix or represented by proxy rigid particle entities.
    """
    arms = tuple(arms)
    if readback not in ("scalar", "batched"):
        raise ContractError("Native readback must be scalar or batched")
    root, sources = _source_bundle(profile)
    binding = _authenticate_profile(scene, profile, root)
    reader = base._BatchedReadback(scene) if readback == "batched" else None
    flags, failures = _runtime(scene, profile, reader.read(scene) if reader is not None else None)
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
    geometry = [_geometry_record(g, normalized[g.idx]) for g in geoms]
    if {r["kind"] for r in geometry} != ROLES:
        failures.append("required_role_family_missing")
    arm_ids = {int(g.idx) for arm in arms for g in arm.geoms}
    if not arms or arm_ids != {r["geom_id"] for r in geometry if r["kind"] == "arm"}:
        raise ContractError("Arms do not exactly match the actual ownership ledger")
    if (len({id(arm) for arm in arms}) != len(arms)
            or any(g.entity is not arm or geoms[int(g.idx)] is not g for arm in arms for g in arm.geoms)):
        raise ContractError("Arms must own their exact native geometry objects")
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
    host = base._array(solver.collider._collision_pair_idx, "host pair matrix")
    native = base._array(solver.collider._collider_info.collision_pair_idx, "native pair matrix")
    valid = base._array(solver.collider._valid_collision_pairs, "host enabled pairs")
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
    native_valid = base._array(solver.collider._collider_info.valid_collision_pairs, "native enabled pairs")
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
                      and base.adjacent_by_installed_rule(first.link, second.link, solver.links)):
                    exclusion = "pinned_robot_adjacent_or_fixed_chain"
                else:
                    exclusion = "UNEXPLAINED_REQUIRED_PAIR_FILTERED"
                    failures.append(f"required_pair_filtered:{a}:{b}")
            pairs.append({"geom_ids": [a, b], "roles": [geometry[a]["kind"], geometry[b]["kind"]],
                          "entities": [geometry[a]["entity"], geometry[b]["entity"]],
                          "enabled": enabled, "exclusion": exclusion})
    payload = {"schema_version": "particle-contact-coverage-v1", "installed_version": "1.3.2",
               "policy_extension_version": "coupled100-final25us-native-pair-policy-v1",
               "readback": copy.deepcopy(reader.record) if reader is not None else {
                   "schema_version": "native-contact-readback-v1", "mode": "scalar"},
               "source_hashes": sources, "runtime_at_build": flags,
               "material_profile": profile, "sample_scope": copy.deepcopy(SAMPLE_SCOPE),
               "geometry": geometry, "pairs": pairs,
               "counts": {"geometries": len(geoms), "all_unordered_pairs": len(pairs),
                          "enabled": len(valid), "excluded": len(pairs)-len(valid)},
               "exclusions_by_reason": dict(Counter(p["exclusion"] for p in pairs if not p["enabled"])),
               "native_pair_matrix_sha256": hashlib.sha256(host.astype("<i8").tobytes()).hexdigest(),
               "robot_self_pair_ledger": base.built_self_pair_ledger(solver, arms),
               "eligible_for_sampled_contact_assessment": not failures,
               "failures": failures,
               "scope": {"continuous_collision_proof": False, "buffer_completeness_proof": False,
                         "initial_static_overlap_checked": False, "contact_allowances_granted": False,
                         "pair_metadata_immutable_after_build": True,
                         "per_step_adapter_must_copy_count_and_classify_all_retained_contacts": True}}
    ledger = {**payload, "coverage_policy_sha256": _digest(payload)}
    # This cache binds subsequent stamps to this exact live scene. It contains
    # a detached ledger, never live tensors or mutable caller dictionaries.
    _BUILT[scene] = copy.deepcopy(ledger)
    _BINDINGS[scene] = binding
    _TOPOLOGY[scene] = _capture_topology(scene, arms, geometry)
    # Retain the exact host object sequence. Comparing detached identity fields
    # also detects semantic changes, while this check rejects replacement or
    # reordering even if a caller edits host and native metadata together.
    _EQUALITY_ORDER[scene] = tuple(solver.equalities)
    if reader is not None:
        _READBACKS[scene] = reader
    else:
        _READBACKS.pop(scene, None)
    return ledger


def _record_flags_valid(flags, automatic_grasp=None):
    """Check actual coupled flags without converting a rigid-only record."""
    if automatic_grasp is not None:
        try: base.validate_automatic_grasp_witness(automatic_grasp, flags=flags)
        except ContractError: return False
    return (flags["built"] is True and type(flags["n_envs"]) is int and flags["n_envs"] == 0
        and flags["rigid_only"] is False and type(flags["substeps"]) is int and flags["substeps"] == SUBSTEPS
        and type(flags["substeps_local"]) is int and flags["substeps_local"] == 1
        and flags["dt_s"] == DT
        and abs(flags["rigid_substep_dt_s"]-INTERNAL_DT) <= 1e-15
        and abs(flags["material_substep_dt_s"]-INTERNAL_DT) <= 1e-15
        and all(flags[k] is False for k in ("use_hibernation", "native_use_hibernation", "requires_grad",
            "native_requires_grad", "sim_requires_grad", "disable_constraint"))
        and all(flags[k] is True for k in ("enable_self_collision", "enable_neutral_collision", "enable_collision", "native_enable_collision"))
        and flags["contact_pruning_tolerance"] is None and flags["native_pruning_tolerance"] == 0.
        and type(flags["native_errno"]) is int and flags["native_errno"] == 0
        and type(flags["static_equalities"]) is int and type(flags["active_equalities"]) is int
        and flags["static_equalities"] == len(flags["equality_identity"])
        and (flags["active_equalities"] == flags["static_equalities"] if automatic_grasp is None
             else flags["active_equalities"] == automatic_grasp["snapshot"]["inventory"]["count"])
        and all(e["type"] == e["native_type"] and e["obj1id"] == e["native_obj1id"]
            and e["obj2id"] == e["native_obj2id"] for e in flags["equality_identity"]))


def require_particle_contact_coverage(ledger):
    """Validate only the separate coupled-particle ledger, never a bead policy."""
    try:
        _require(type(ledger) is dict and ledger["schema_version"] == "particle-contact-coverage-v1",
                 "Expected a coupled particle coverage ledger")
        payload = {key: value for key, value in ledger.items() if key != "coverage_policy_sha256"}
        _require(ledger["coverage_policy_sha256"] == _digest(payload), "Particle coverage ledger hash mismatch")
        _require(ledger["eligible_for_sampled_contact_assessment"] is True and ledger["failures"] == []
                 and ledger["material_profile"] in PROFILES and ledger["sample_scope"] == SAMPLE_SCOPE
                 and _record_flags_valid(ledger["runtime_at_build"])
                 and {g["kind"] for g in ledger["geometry"]} == ROLES,
                 "Particle pair policy is ineligible or has a different actual mode")
        return ledger["coverage_policy_sha256"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("Malformed particle coverage ledger: "+str(exc)) from exc


def _scalar_contact_count(scene):
    c = scene.rigid_solver.collider
    value = base._array(c._collider_state.n_contacts, "native contact count")
    _require(value.shape == (1,), "Expected one unbatched native contact count")
    capacity = _integer(c._collider_info.max_contacts[None], "native contact capacity")
    count = int(value[0])
    _require(capacity >= 0 and 0 <= count <= capacity, "Native contact count is outside capacity")
    return count, capacity


def particle_contact_coverage_stamp(scene, ledger, *, automatic_grasp=None):
    """Read actual coupled flags and source/host identities for this built scene.

    The stamp does not certify the copied buffer. The last-internal sample and
    detection-state binding are independently required by the row validator.
    """
    frozen = _BUILT.get(scene)
    reader = _READBACKS.get(scene)
    flags, failures, native_read = {}, [], None
    count = capacity = None
    automatic_proof = None
    try:
        require_particle_contact_coverage(ledger)
        _require(frozen is not None and _digest(ledger) == _digest(frozen), "No exact matching built particle scene policy")
        profile = frozen["material_profile"]
        _, sources = _source_bundle(profile)
        _require(sources == frozen["source_hashes"], "Particle contact source binding changed")
        binding = _BINDINGS.get(scene)
        _require(binding is not None, "Missing built coupled runtime identity")
        _check_profile(binding)
        topology = _TOPOLOGY.get(scene)
        _require(topology is not None, "Missing built rigid geometry identity")
        _check_topology(scene, topology)
        _require(_same_objects(tuple(scene.rigid_solver.equalities), _EQUALITY_ORDER.get(scene, ())),
                 "Host equality object order changed after build")
        _require(frozen["readback"]["mode"] != "batched" or reader is not None,
                 "Batched readback is unavailable. Scalar fallback is prohibited")
        native_read = reader.read(scene) if reader is not None else None
        flags, runtime_failures = _runtime(scene, profile, native_read)
        if automatic_grasp is not None:
            automatic_proof = base._automatic_policy(scene, frozen, automatic_grasp, flags)
            runtime_failures = [f for f in runtime_failures if f != "extra_or_mismatched_equality_constraints"]
        failures.extend(runtime_failures)
        for key, original in frozen["runtime_at_build"].items():
            if key != "native_errno" and not (automatic_proof is not None and key == "active_equalities") and flags[key] != original:
                failures.append("runtime_policy_changed:"+key)
        if reader is not None:
            count, capacity = native_read["contact_count"], native_read["contact_capacity"]
        else:
            count, capacity = _scalar_contact_count(scene)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, ContractError) as exc:
        failures.append("particle_policy_unavailable:"+str(exc))
    result = dict(schema_version="particle-contact-policy-stamp-v1",
        coverage_policy_sha256=ledger.get("coverage_policy_sha256") if type(ledger) is dict else None,
        material_profile=ledger.get("material_profile") if type(ledger) is dict else None, policy_valid_now=not failures, failures=failures,
        runtime_flags=flags, sample_scope=copy.deepcopy(SAMPLE_SCOPE),
        pruning_disabled=flags.get("contact_pruning_tolerance", "missing") is None
            and flags.get("native_pruning_tolerance") == 0.,
        rigid_only=flags.get("rigid_only"), substeps=flags.get("substeps"),
        substeps_local=flags.get("substeps_local"), dt_s=flags.get("dt_s"),
        rigid_substep_dt_s=flags.get("rigid_substep_dt_s"), material_substep_dt_s=flags.get("material_substep_dt_s"),
        native_errno=flags.get("native_errno"), active_equalities=flags.get("active_equalities"),
        enable_collision=flags.get("enable_collision"), native_enable_collision=flags.get("native_enable_collision"),
        disable_constraint=flags.get("disable_constraint"), readback_mode="batched" if reader is not None else "scalar",
        native_contact_count=count, native_contact_capacity=capacity, contact_buffer_completeness_asserted=False)
    if automatic_grasp is not None:
        result.update(schema_version="particle-contact-policy-stamp-automatic-grasp-v1", automatic_grasp=automatic_proof)
    return result


def _number(value, label):
    _require(type(value) in (int, float) and math.isfinite(value), label+" must be finite numeric")
    return value


def _at(value, expected, label):
    _require(abs(_number(value, label)-expected) <= 1e-12, label+" does not match the actual coupled clock")


def validate_particle_contact_coverage(row, step, dt, contacts):
    """Bind copied/classified rigid contacts to c137's final25us measured state.

    This pure validator cannot establish native provenance of a fabricated row.
    Source-authenticated build/stamp/copy integration supplies that provenance.
    Passing describes only this final internal sample, not all100 intervals.
    """
    try:
        _require(type(step) is int and step > 0 and type(dt) in (int, float) and dt == DT,
                 "Particle coverage uses the fixed2.5ms outer clock")
        coverage = row["coverage"]
        stamp, raw, last = coverage["policy_stamp"], row["raw_contacts"], coverage["last_internal_sample"]
        _require(type(contacts) is list, "Classified contacts must be an explicit complete list")
        validated = validate_contact_snapshot(raw)
        count = len(contacts)
        for key in ("native_contact_count", "copied_contact_count", "classified_contact_count"):
            _require(type(coverage[key]) is int and coverage[key] == count, "Native/copied/classified contact counts differ")
        _require(len(validated["geom_a"]) == count and coverage["contacts_complete"] is True
                 and type(coverage["step"]) is int and coverage["step"] == step,
                 "Incomplete same-interval particle contact coverage")
        _at(coverage["pre_time_s"], (step-1)*DT, "Outer pre time")
        _at(coverage["post_time_s"], step*DT, "Outer post time")
        automatic_proof = stamp.get("automatic_grasp")
        if automatic_proof is not None:
            base.validate_automatic_grasp_stamp(stamp, step=step, dt=DT, substeps=SUBSTEPS)
        _require(stamp["schema_version"] == ("particle-contact-policy-stamp-v1" if automatic_proof is None else "particle-contact-policy-stamp-automatic-grasp-v1")
                 and "single_substep" not in stamp and stamp["material_profile"] in PROFILES
                 and stamp["policy_valid_now"] is True and stamp["failures"] == []
                 and stamp["pruning_disabled"] is True and stamp["sample_scope"] == SAMPLE_SCOPE
                 and stamp["contact_buffer_completeness_asserted"] is False
                 and _record_flags_valid(stamp["runtime_flags"], automatic_proof),
                 "Invalid source-bound coupled particle contact stamp")
        for key in ("rigid_only", "substeps", "substeps_local", "dt_s", "rigid_substep_dt_s", "material_substep_dt_s",
                    "native_errno", "active_equalities", "enable_collision", "native_enable_collision", "disable_constraint"):
            _require(type(stamp[key]) is type(stamp["runtime_flags"][key]) and stamp[key] == stamp["runtime_flags"][key],
                     "Stamp actual-mode flags disagree: "+key)
        identity = stamp["coverage_policy_sha256"]
        _require(type(identity) is str and len(identity) == 64 and all(c in "0123456789abcdef" for c in identity),
                 "Missing exact particle coverage policy identity")
        _require(type(stamp["native_contact_count"]) is int and stamp["native_contact_count"] == count
                 and type(stamp["native_contact_capacity"]) is int and stamp["native_contact_capacity"] >= count,
                 "Stamp/native contact count or capacity differs")
        _require(type(last["outer_interval_ending_step"]) is int and last["outer_interval_ending_step"] == step
                 and type(last["preintegration_global_substep"]) is int and last["preintegration_global_substep"] == step*SUBSTEPS-1
                 and last["readback_did_not_change_state"] is True and last["scope"] == LAST_SAMPLE_SCOPE,
                 "Missing or stale actual last-internal sample")
        clock = last["preintegration_native_clock"]
        _require(type(clock) is list and len(clock) == 4 and all(type(v) is int for v in clock[:3])
                 and clock[:3] == [step*SUBSTEPS-1, 0, step-1], "Actual last-internal native clock differs")
        sample_time = (step*SUBSTEPS-1)*INTERNAL_DT
        _at(clock[3], sample_time, "Native preintegration clock")
        _at(last["preintegration_time_s"], sample_time, "Preintegration time")
        _at(last["outer_interval_ending_time_s"], step*DT, "Outer ending time")
        sample = last["sample"]
        _require(type(sample) is dict and set(sample) == {"raw_contacts", "rigid_state"}
                 and last["same_instant_copy_before_after_sha256"] == _digest(sample)
                 and _digest(sample["raw_contacts"]) == _digest(raw)
                 and _digest(sample["rigid_state"]) == _digest(row["contact_detection_state"]),
                 "Same-instant sample/raw/detection-state digest differs")
        state = sample["rigid_state"]
        _require(type(state) is dict and type(state["body_world"]) is dict and type(state["arms"]) is dict
                 and bool(state["arms"]) and type(state["native_contact_count"]) is int
                 and state["native_contact_count"] == count,
                 "Actual contact-detection rigid state/count is incomplete")
        if "contact_detection_time_s" in row:
            _at(row["contact_detection_time_s"], sample_time, "Contact detection time")
        for index, contact in enumerate(contacts):
            _require(type(contact) is dict and type(contact["raw_row"]) is int and contact["raw_row"] == index
                     and contact["geom_ids"] == [raw["geom_a"][index], raw["geom_b"][index]],
                     "Classified contact order/geometry differs from copied native buffer")
        return identity
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ContractError("Malformed coupled particle contact coverage: "+str(exc)) from exc
