"""Supplied bead-only batch tail with a separate, absolute native-clock lineage.

This module neither commits nor alters the old mission dispatcher. Mathematical
rest projections are explicitly backed by owned device histories. They are not
legacy raw contact rows or a legacy whole-episode coverage certificate.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import types

import numpy as np

from transport_challenge.contracts import ContractError, digest

DT = .0025
SUBSTEPS = 16
SUPPORT_COLUMNS = ('native_global_substep', 'named_goal_support', 'any_named_support',
                   'holder_present', 'classifier_failed', 'classified_global_substep')
_KERNEL = {}


def require(value, message):
    if not value:
        raise ContractError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def support_mask(stations, support_id, geometry_count):
    """Only the original validated station registry assigns a support identity."""
    require(type(support_id) is str and bool(support_id), 'Require an exact named goal support')
    mask = np.zeros(geometry_count, np.int32)
    for geom, station in stations.items():
        require(type(geom) is int and 0 <= geom < geometry_count
                and station.get('geom_id') == geom, 'Station registry geometry differs')
        if station['id'] == support_id:
            mask[geom] = 1
    require(bool(mask.any()), 'Goal support has no actual registered native geometry')
    return mask


def contact_roles(phase, actual_holder, permission_arm, arm_ids):
    """Original classifier permission is distinct from native weld membership."""
    require(phase in ('carry', 'place', 'release', 'released_settle'), 'Unknown provided phase')
    require(actual_holder is None or actual_holder in arm_ids, 'Unknown actual native holder')
    require(permission_arm is None or permission_arm in arm_ids, 'Unknown contact permission arm')
    if phase == 'release':
        require(actual_holder is None and permission_arm in arm_ids,
                'Post-detach release retains the original active-arm contact permission')
    elif phase == 'released_settle':
        require(actual_holder is None and permission_arm is None,
                'Released rest grants no holder or finger contact permission')
    else:
        require(actual_holder in arm_ids and permission_arm == actual_holder,
                'Held carrying/placement must authorize its actual holder only')
    return permission_arm


def assess_support_history(history, *, start_ordinal, phase, holder_arm_id):
    """Assess actual 16-sample intervals, never infer a timeline from totals."""
    a = np.asarray(history)
    require(a.ndim == 2 and a.shape[1] == len(SUPPORT_COLUMNS)
            and a.dtype == np.dtype('int64') and len(a) > 0 and len(a) % SUBSTEPS == 0,
            'Require complete exact int64 native support history')
    require(type(start_ordinal) is int and start_ordinal >= 0 and start_ordinal % SUBSTEPS == 0,
            'Support history must start on the real outer boundary')
    expected = np.arange(start_ordinal, start_ordinal + len(a), dtype=np.int64)
    require(np.array_equal(a[:, 0], expected) and np.array_equal(a[:, 5], expected),
            'Skipped, repeated or stale native support/classifier ordinal')
    require(np.isin(a[:, 1:4], [0, 1]).all() and (a[:, 4] == 0).all()
            and (a[:, 1] <= a[:, 2]).all(), 'Failed or malformed support classification')
    held = holder_arm_id is not None
    require((a[:, 3] == int(held)).all(), 'Actual phase holder permission changed')
    grouped = a.reshape(-1, SUBSTEPS, len(SUPPORT_COLUMNS))
    goal = grouped[:, :, 1].all(axis=1)
    any_support = grouped[:, :, 2].any(axis=1)
    longest = current = 0
    for supported in any_support:
        current = current + 1 if phase == 'carry' and held and not supported else 0
        longest = max(longest, current)
    return dict(schema_version='deferred-named-support-history-v1',
        start_native_step=start_ordinal // SUBSTEPS,
        end_native_step=(start_ordinal + len(a)) // SUBSTEPS,
        actual_outer_intervals=len(grouped), actual_internal_samples=len(a),
        phase=phase, holder_arm_id=holder_arm_id,
        named_goal_support_every_internal_by_outer=goal.tolist(),
        any_named_support_by_outer=any_support.tolist(),
        maximum_consecutive_unsupported_held_s=longest * DT,
        raw_history_sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
        columns=list(SUPPORT_COLUMNS), legacy_coverage_claimed=False)


def _support_kernel(qd, arrays):
    if id(qd) not in _KERNEL:
        @qd.kernel
        def collect(mask: qd.types.ndarray(), categories: qd.types.ndarray(),
                    classifier: qd.types.ndarray(), result: qd.types.ndarray(),
                    cs: arrays.ColliderState, index: qd.i32, ordinal: qd.i32,
                    held: qd.i32, capacity: qd.template(), n_geom: qd.template(),
                    permuted: qd.template()):
            for singleton in range(1):
                goal = 0
                supported = 0
                count = cs.n_contacts[0]
                valid = classifier[0] == 0 and classifier[15] == ordinal
                if count < 0 or count > capacity:
                    valid = False
                if valid:
                    for i in range(count):
                        if categories[i] == 1:
                            slot = i
                            if qd.static(permuted):
                                slot = cs.contact_sort_idx[i, 0]
                            if slot >= 0 and slot < capacity:
                                a = cs.contact_data.geom_a[slot, 0]
                                b = cs.contact_data.geom_b[slot, 0]
                                if a >= 0 and a < n_geom and b >= 0 and b < n_geom:
                                    supported = 1
                                    if mask[a] == 1 or mask[b] == 1:
                                        goal = 1
                                else:
                                    valid = False
                            else:
                                valid = False
                result[index, 0] = ordinal
                result[index, 1] = goal
                result[index, 2] = supported
                result[index, 3] = held
                result[index, 4] = qd.cast(not valid, qd.i64)
                result[index, 5] = classifier[15]
        _KERNEL[id(qd)] = collect
    return _KERNEL[id(qd)]


class DeviceNamedSupportHistory:
    """Use directly after the same owned DeviceContactAudit.pre, including carry."""
    def __init__(self, contact_audit, *, support_id, capacity_steps, phase, holder_arm_id,
                 permission_arm_id=None):
        from physics.deferred_gpu_contacts import DeviceContactAudit
        require(type(contact_audit) is DeviceContactAudit and contact_audit.sample_mode == 'bead16',
                'Support history requires the actual original bead16 device classifier')
        require(type(capacity_steps) is int and 0 < capacity_steps <= 20000,
                'Bound actual support history allocation')
        permission = holder_arm_id if permission_arm_id is None else permission_arm_id
        contact_roles(phase, holder_arm_id, permission, contact_audit.policy['arm_ids'])
        require(contact_audit.phases[-1]['phase'] == phase
                and contact_audit.phases[-1]['active_holder_arm_id'] == permission,
                'Support recorder phase differs from native equality/contact policy')
        self.audit = contact_audit
        self.scene = contact_audit.scene
        self.support_id, self.phase, self.holder = support_id, phase, holder_arm_id
        self.permission_arm_id = permission
        self.steps, self.calls = capacity_steps, 0
        self.start = int(self.scene.sim.cur_substep_global)
        require(self.start == int(self.scene.t) * SUBSTEPS, 'Actual native start clock differs')
        t, qd = contact_audit.torch, contact_audit.qd
        mask = support_mask(contact_audit.policy['registries'][3], support_id, contact_audit.r.n_geoms)
        self.mask = t.tensor(mask, dtype=t.int32, device=contact_audit.gs.device)
        self.data = t.zeros((capacity_steps * SUBSTEPS, len(SUPPORT_COLUMNS)),
                            dtype=t.int64, device=contact_audit.gs.device)
        from genesis.utils import array_class as arrays
        self.collect = _support_kernel(qd, arrays)
        self.sources = {str(Path(__file__).resolve()): _sha(__file__)}
        self.result = None

    def capture_pre(self, ordinal):
        require(self.result is None and type(ordinal) is int and ordinal == self.start + self.calls
                and self.calls < self.steps * SUBSTEPS
                and ordinal == self.scene.sim.cur_substep_global
                and self.audit.calls['pre'] == self.calls + 1,
                'Support capture must immediately follow every actual native contact PRE')
        a = self.audit
        self.collect(self.mask, a.categories, a.state, self.data, a.r.collider._collider_state,
                     self.calls, ordinal, int(self.holder is not None), a.capacity,
                     a.r.n_geoms, a.permuted)
        self.calls += 1

    def finish(self, contact_receipt):
        require(self.result is None and self.calls == self.steps * SUBSTEPS
                and int(self.scene.t) * SUBSTEPS == self.start + self.calls,
                'Support recorder lacks the actual complete physical interval sequence')
        require(self.audit.finalized and contact_receipt.get('passed') is True
                and contact_receipt.get('PRE_samples') == self.calls
                and contact_receipt.get('POST_samples') == self.calls
                and contact_receipt.get('coverage_policy_sha256') == self.audit.coverage_sha256
                and contact_receipt.get('source_hashes') == self.audit.sources,
                'Support facts require the successful exact owned native contact audit')
        require(all(_sha(p) == h for p, h in self.sources.items()), 'Support source changed')
        raw = self.data.detach().cpu().numpy().copy()
        raw.setflags(write=False)
        self.result = assess_support_history(raw, start_ordinal=self.start,
            phase=self.phase, holder_arm_id=self.holder)
        self.result.update(support_id=self.support_id, source_hashes=copy.deepcopy(self.sources),
            contact_audit_sha256=digest(contact_receipt), passed=True,
            contact_permission_arm_id=self.permission_arm_id)
        self.raw = raw
        return copy.deepcopy(self.result)


def tail_commands(last_row, arm_order, active_arm, finger_bounds, *, phase, duration_s):
    """Exact supplied hold and EpisodeDispatcher cubic open samples, unchanged."""
    q = np.asarray(last_row['position'] if type(last_row) is dict else last_row, np.float64)
    require(q.shape == (9*len(arm_order),) and np.isfinite(q).all() and len(arm_order) in (1,3)
            and len(set(arm_order)) == len(arm_order) and active_arm in arm_order, 'Require exact final q9 commands for actual controlled arms')
    count = round(duration_s / DT)
    require(count > 0 and abs(duration_s - count * DT) <= 1e-12
            and phase in ('place', 'release', 'released_settle'), 'Tail phase must use the original dt grid')
    pos = np.repeat(q[None], count, axis=0)
    vel = np.zeros_like(pos)
    if phase == 'release':
        bounds = np.asarray(finger_bounds[active_arm], np.float64)
        require(bounds.shape == (2, 2) and np.isfinite(bounds).all(), 'Actual native finger bounds required')
        target = min(bounds[:, 1])
        start = q[9 * arm_order.index(active_arm) + 7:9 * arm_order.index(active_arm) + 9]
        require(np.all(start <= target) and np.all(bounds[:, 0] <= target), 'Provided open cannot close/clamp fingers')
        offset = 9 * arm_order.index(active_arm) + 7
        for i in range(count):
            if i + 1 == count:
                pos[i, offset:offset + 2] = target
            else:
                u = (i + 1) / count
                blend = u * u * (3. - 2. * u)
                pos[i, offset:offset + 2] = start + (target - start) * blend
                vel[i, offset:offset + 2] = (target - start) * 6. * u * (1. - u) / duration_s
    return dict(position=pos, velocity=vel)


@contextmanager
def _hooks(scene, audit, supports, handling):
    solver = scene.rigid_solver
    require('_func_constraint_force' not in vars(solver) and 'substep' not in vars(solver),
            'Original observers must be restored before the separate batch tail')
    constraint, substep = solver._func_constraint_force, solver.substep
    def pre(_self, *args, **kwargs):
        result = constraint(*args, **kwargs)
        ordinal = int(scene.sim._cur_substep_global)
        audit.pre(ordinal)
        supports.capture_pre(ordinal)
        handling.pre(ordinal)
        return result
    def post(_self, *args, **kwargs):
        result = substep(*args, **kwargs)
        ordinal = int(scene.sim._cur_substep_global)
        audit.post(ordinal)
        handling.post(ordinal)
        return result
    solver._func_constraint_force = types.MethodType(pre, solver)
    solver.substep = types.MethodType(post, solver)
    try:
        yield
    finally:
        delattr(solver, '_func_constraint_force')
        delattr(solver, 'substep')
        require(solver._func_constraint_force.__func__ is constraint.__func__
                and solver.substep.__func__ is substep.__func__, 'Original native hooks were not restored')


def handling_checkpoint(owner, scene, start_ordinal, *, expected=None, final=False):
    """Read one exact persistent owner. This function never initializes an anchor."""
    from physics.deferred_gpu_handling import DeviceHandlingAudit
    require(type(owner) is DeviceHandlingAudit and owner.scene is scene and not owner.closed,
            'Retain the actual original-prefix handling owner through the complete tail')
    record = owner.finish() if final else owner.checkpoint()
    end = int(scene.t) * SUBSTEPS
    valid = (record.get('schema_version') == 'deferred-device-handling-audit-v1'
            and record.get('passed') is True and record.get('sticky_failed') is False
            and record.get('source_bindings_verified') is True
            and record.get('every_original_internal_sample_preserved') is True
            and record.get('start_global_substep') == start_ordinal
            and record.get('last_accepted_global_substep') == end
            and record.get('PRE_samples') == record.get('POST_samples') == end-start_ordinal
            and record.get('source_hashes') == owner.sources
            and record.get('prefix_guard') == owner.seed
            and record.get('held_root_to_hand_limit_m') == .20
            and record.get('unheld_displacement_limit_m') == .005
            and owner.closed is bool(final))
    if not valid or (expected is not None and record != expected):
        error = ContractError('Persistent GPU handling failed or differs from its actual original seed/source/clock/owner checkpoint')
        error.handling_record = copy.deepcopy(record)
        raise error
    return record


def project_history(control, payload, support, *, start_step, arm_order, identities,
                    phase, holder, previous, payload_owner_clock):
    """Mathematical projections of authentic histories, never dispatcher commits."""
    c = np.asarray(control['history'])
    p = np.asarray(payload['history'])
    n = len(c)
    require(control.get('passed') is True and payload.get('passed') is True
            and support.get('passed') is True and len(arm_order) in (1,3)
            and len(set(arm_order))==len(arm_order) and c.shape == (n, 9*len(arm_order), 13)
            and p.shape == (n, 46) and np.isfinite(c).all() and np.isfinite(p).all()
            and control['start_native_step'] == support['start_native_step'] == start_step
            and control['end_native_step'] == support['end_native_step'] == start_step + n
            and payload['actual_POST_samples'] == support['actual_outer_intervals'] == n,
            'Only successful exact owned control/payload/support histories may be projected')
    require(payload_owner_clock == dict(start_native_step=start_step,
            end_native_step=start_step+n, actual_POST_samples=n),
            'Actual payload owner start/end/capture clock differs')
    require(payload['original_bead_ids'] == identities and len(identities) == 4,
            'Original actual bead identities changed')
    from physics.deferred_gpu_control import CHANNELS
    require(tuple(control['arm_order']) == tuple(arm_order)
            and tuple(control['history_channels']) == CHANNELS
            and support['phase'] == phase and support['holder_arm_id'] == holder,
            'Owned history layout/phase/arm identity changed')
    native_masses = np.asarray(payload['expected_native_masses_kg'])
    require(native_masses.shape == (4,) and np.isfinite(native_masses).all()
            and all((p[:, 21 + 8 * j] == native_masses[j]).all() for j in range(4)),
            'Actual native mass history differs from its verified dtype-cast binding')
    channels = {name: i for i, name in enumerate(control['history_channels'])}
    rows = []
    for i in range(n):
        step = start_step + i + 1
        row = dict(schema_version='deferred-native-bead-tail-endpoint-v1', step=step,
            native_scene_step=step, simulation_s=step * DT, phase=phase,
            post_processing_complete=True, failures=[], actual_holder_arm_id=holder,
            native_history_projection=True, legacy_dispatcher_commit_claimed=False,
            named_goal_support_id=support['support_id'],
            body_world=dict(frame='world', position_m=p[i, :3].tolist(), quaternion_wxyz=p[i, 3:7].tolist()),
            body_linear_velocity_m_s=p[i, 7:10].tolist(), body_angular_velocity_rad_s=p[i, 10:13].tolist(),
            body_up_tilt_deg=float(p[i, 13]), arms={}, materials=[],
            named_goal_support_every_internal=support['named_goal_support_every_internal_by_outer'][i],
            classified_contacts=[])
        if holder is None:
            row['provided_grasp'] = None
        for j, arm in enumerate(arm_order):
            a = c[i, 9 * j:9 * j + 9]
            row['arms'][arm] = dict(q_all_dofs=a[:, channels['post_q']].tolist(),
                qd_all_dofs=a[:, channels['post_qd']].tolist(),
                qdd_fd_rad_s2=a[:7, channels['post_qdd_fd']].tolist())
        for j, identity in enumerate(identities):
            offset = 14 + 8 * j
            require(p[i, offset + 6] == 1., 'GPU history contains an unretained bead')
            row['materials'].append(dict(id=identity['id'], entity_idx=identity['entity_idx'],
                mass_kg=identity['mass_kg'], inside_carrier=True,
                position_world_m=p[i, offset:offset + 3].tolist(),
                velocity_world_m_s=p[i, offset + 3:offset + 6].tolist()))
        if row['named_goal_support_every_internal']:
            row['classified_contacts'] = [dict(category='named_tray_bottom_support_top',
                support_station_id=support['support_id'],
                evidence_schema='deferred-named-support-history-v1')]
        row['pre'] = previous
        rows.append(row)
        previous = {k: copy.deepcopy(v) for k, v in row.items() if k != 'pre'}
    return rows


def released_numeric_gate(rows, *, settings, rest_policy, goal, arm_order,
                          finger_targets, load_bearing_s, model, identities, strict_material):
    """Original released numeric/rest conditions on explicit typed device facts."""
    from physics.episode_runtime import measured_rest, body_rest, PROTOCOL
    from physics.episode_transport_assessment import _angle, _materials, bead_material_rest_gate
    now = rows[-1]['simulation_s']
    end = rows[-1]['native_scene_step']
    selected = [r for r in rows if now - .5 + 1e-10 < r['simulation_s'] <= now + 1e-10]
    require(len(selected) == 200 and load_bearing_s >= .5, 'Incomplete released/load-bearing window')
    nominal = goal.get('seating', {}).get('nominal_goal', goal)
    execution = settings['execution_limits']
    supply = _materials(identities)
    for row, step in zip(selected, range(end - 199, end + 1)):
        require(row['step'] == row['native_scene_step'] == step
                and abs(row['simulation_s'] - step * DT) <= 1e-9
                and row['post_processing_complete'] is True and row['failures'] == []
                and row['phase'] == 'released_settle'
                and row['named_goal_support_id'] == nominal['support_id']
                and row['named_goal_support_every_internal'] is True,
                'Released support, phase or clock history failed')
        require(row['pre']['native_scene_step'] == step - 1
                and abs(row['pre']['simulation_s'] - (step - 1) * DT) <= 1e-9,
                'Missing or stale actual preceding native endpoint')
        for value in (row['pre'], row):
            require(value.get('actual_holder_arm_id') is None and value.get('provided_grasp') is None
                    and set(value['arms']) == set(arm_order), 'Released endpoint has holder or missing arm')
            for name in arm_order:
                q = np.asarray(value['arms'][name]['q_all_dofs'])
                v = np.asarray(value['arms'][name]['qd_all_dofs'])
                require(q.shape == v.shape == (9,) and np.isfinite(q).all() and np.isfinite(v).all()
                    and max(abs(v[:7])) <= execution['rest_speed_rad_s']
                    and max(abs(v[7:])) <= PROTOCOL['finger_rest_speed_m_s']
                    and max(abs(q[7:] - finger_targets[name])) <= settings['closure_match_tolerances']['finger_m'],
                    'Original released arm/finger rest or exact command match failed')
            pose = value['body_world']
            require(math.dist(pose['position_m'], nominal['body_world']['position_m']) <= execution['final_body_position_error_m']
                and _angle(pose['quaternion_wxyz'], nominal['body_world']['quaternion_wxyz']) <= execution['final_body_orientation_error_rad']
                and math.hypot(*value['body_linear_velocity_m_s']) <= rest_policy['maximum_body_linear_speed_m_s']
                and math.hypot(*value['body_angular_velocity_rad_s']) <= rest_policy['maximum_body_angular_speed_rad_s'],
                'Original released body goal/rest failed')
            current = _materials(value['materials'])
            require(list(current) == list(supply), 'Original bead identity order changed')
            for name, item in current.items():
                require((item['mass_kg'], item['entity_idx']) == (supply[name]['mass_kg'], supply[name]['entity_idx'])
                    and item['inside_carrier'] is True
                    and (not strict_material or math.hypot(*item['velocity_world_m_s']) <= settings['scenario']['stable_speed_m_s']),
                    'Original released bead identity/retention/speed failed')
    arms = measured_rest(rows, now, {**execution, 'rest_window_s': .5}, arm_order)
    body = body_rest(rows, now, .5, rest_policy)
    bulk = bead_material_rest_gate(rows, now, identities, model)
    require(arms['passed'] and body['passed'] and bulk['quality_passed'], 'Original arm/body/material quality failed')
    if strict_material:
        require(bulk['passed'], 'Original complete201-endpoint bead bulk rest failed')
    return dict(schema_version='deferred-bead-released-rest-gate-v1', passed=True,
        support_id=nominal['support_id'], expected_intervals=200, expected_material_endpoints=201,
        ending_s=now, all_arms_rest=arms, body_rest=body, bead_bulk_rest=bulk,
        actual_unsupported_held_s=load_bearing_s, strict_material=strict_material,
        legacy_coverage_claimed=False)


def run_supported_tail(runtime_context, last_command_table_row, *, output, wall_deadline):
    """Execute the original supplied bead tail. Failures never become completion."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = dict(schema_version='provided-deferred-bead-tail-v1', status='FAILED', passed=False,
        complete_one_carry_verified=False, full_mission_passed=False, no_fake_dispatcher_commits=True,
        legacy_raw_archive_claimed=False, phases=[], failure=None)
    c = runtime_context
    scene = c['scene']
    receipt['start_native_step'] = int(scene.t)
    try:
        from physics.deferred_gpu_control import DeviceCarryControl
        from physics.deferred_gpu_contacts import DeviceContactAudit
        from physics.deferred_gpu_beads import DeferredPayloadAudit
        from physics.episode_runtime import measured_rest
        from physics.provided_manipulation import assess_place_ready
        from physics.episode_transport_assessment import _bead_material_sample, _materials
        from physics.automatic_runtime_binding import physical_observation
        driver = c['student_driver']
        helper = c['automatic_grasp']
        arm = helper.holder
        require(len(c['beads']) == len(c['ids']) == 4 and not driver.particle_mode
                and helper is driver.automatic_grasp and arm in c['arms']
                and scene.sim.substeps == SUBSTEPS and scene.sim.dt == DT,
                'Supplied deferred tail is only the original four-bead automatic scene')
        require(type(wall_deadline) in (int, float) and math.isfinite(wall_deadline), 'Require absolute host deadline')
        def check(*, sources=True):
            if time.perf_counter() >= wall_deadline:
                raise TimeoutError('Supplied tail exceeded the whole-process wall deadline')
            if sources:
                require(c['source_check']() is True, 'Source bindings changed during supplied tail')
            require(float(scene.cur_t) <= driver.simulation_deadline_s + 1e-9, 'Original simulation deadline exceeded')
        check()
        carry = c['deferred_carry_result']
        support_owner = c['deferred_carry_support']
        require(type(support_owner) is DeviceNamedSupportHistory and support_owner.scene is scene
                and support_owner.result is not None and support_owner.phase == 'carry'
                and support_owner.holder == arm and support_owner.support_id == driver.goal['support_id']
                and support_owner.result['end_native_step'] == int(scene.t),
                'Actual carried load/support owner and final clock required')
        require(carry.get('material') == 'beads' and carry.get('actual_final_step') == int(scene.t)
                and all(carry.get(key, {}).get('passed') is True for key in ('control', 'contacts', 'payload'))
                and carry['contacts'] and digest(carry['contacts']) == support_owner.result['contact_audit_sha256'],
                'The exact carried control/contact/payload prefix must pass before the tail')
        handling = c['deferred_handling_audit']
        initial_handling = handling_checkpoint(handling, scene, support_owner.start,
                                               expected=carry['handling'])
        receipt['carried_handling_checkpoint'] = initial_handling
        verified_load = assess_support_history(support_owner.raw, start_ordinal=support_owner.start,
            phase='carry', holder_arm_id=arm)
        require(all(support_owner.result.get(k) == v for k, v in verified_load.items()),
                'Carried support/load receipt changed from its owned actual history')
        load = support_owner.result['maximum_consecutive_unsupported_held_s']
        require(load >= .5, 'Actual carried unsupported held load was shorter than the unchanged 0.5-second gate')
        try:
            receipt['carried_evidence_sha256'] = digest(carry)
        except (TypeError, ValueError) as exc:
            raise ContractError('Pass the artifact-bound JSON carry receipt. Keep actual arrays in its hash-bound NPZ files') from exc
        for kind in ('control', 'payload'):
            record = carry[kind].get('history_record', {})
            path = record.get('file')
            require(type(path) is str and Path(path).is_file() and _sha(path) == record.get('sha256'),
                    'Actual carried ' + kind + ' NPZ history binding is missing or changed')
        receipt['carried_support'] = copy.deepcopy(support_owner.result)
        names = tuple(sorted(c['arms']))
        q = np.asarray(last_command_table_row['position'] if type(last_command_table_row) is dict
                       else last_command_table_row, np.float64).copy()
        require(q.shape == (9*len(names),) and np.isfinite(q).all(), 'Final carried command row differs')
        previous = c['observe']()
        require(previous['native_scene_step'] == int(scene.t), 'Initial tail snapshot has stale native clock')
        previous = copy.deepcopy(previous)
        previous.update(step=int(scene.t), post_processing_complete=True, failures=[],
                        actual_holder_arm_id=arm, native_history_projection=False)
        rows = [previous]
        sources = {str(Path(__file__).resolve()): _sha(__file__)}
        def segment(phase, duration):
            nonlocal q, previous
            check()
            begin = int(scene.t)
            commands = tail_commands(q, names, arm, c['finger_bounds'], phase=phase, duration_s=duration)
            count = len(commands['position'])
            holder = helper.holder
            snapshot = helper.snapshot(arm)
            inventory = snapshot['inventory']
            permission = contact_roles(phase, holder, arm if phase == 'release' else holder, names)
            contact = DeviceContactAudit(scene, c['coverage'], owners=c['owners'], components_by_geom=c['components'],
                pads_by_arm=c['pads'], stations_by_geom=c['station_map'], sample_mode='bead16',
                expected_inventory=inventory, model=c['model'])
            contact.set_phase(phase, permission, expected_inventory=inventory)
            handling.bind_contact(contact, phase, holder, snapshot, permission_arm_id=permission)
            support = DeviceNamedSupportHistory(contact, support_id=driver.goal['support_id'],
                capacity_steps=count, phase=phase, holder_arm_id=holder, permission_arm_id=permission)
            control = DeviceCarryControl(scene, c['arms'], c['gravity'], c['limits_by_arm'], commands,
                kp=c['kp'], kv=c['kv'], effort=c['effort'], finger_bounds=c['finger_bounds'],
                hands=c['hands'], tool_workspace_m=c['case']['tool_workspace_m'],
                tracking_error_rad=c['execution']['tracking_error_rad'], dt_s=DT)
            payload = DeferredPayloadAudit(scene, c['beads'], c['tray'], c['model'], c['ids'],
                tilt_limit_deg=driver.body_tilt_limit_deg, capacity_steps=count)
            with _hooks(scene, contact, support, handling):
                for i in range(count):
                    check(sources=False)
                    control.apply(i)
                    scene.step()
                    control.capture_post(i)
                    payload.capture_post(i)
            control_record = control.finish()
            payload_record = payload.finish()
            payload_owner_clock = dict(start_native_step=payload.start,
                end_native_step=int(scene.t), actual_POST_samples=payload.captures)
            contact_record = contact.final_readback(expected_pre_samples=count * SUBSTEPS,
                                                     expected_post_samples=count * SUBSTEPS)
            index = len(receipt['phases'])
            path = output / f'phase_{index:02d}_actual_histories.npz'
            # Persist actual histories before interpreting any failed final audit.
            support_raw = support.data.detach().cpu().numpy().copy()
            np.savez(path, control=control_record['history'], payload=payload_record['history'],
                     support=support_raw, command_position=commands['position'],
                     command_velocity=commands['velocity'])
            receipt['last_actual_history'] = dict(file=str(path), sha256=_sha(path))
            from physics.transport_outcomes import check_diagnostic, reject_failed_checks
            checks = check_diagnostic(control_record, payload_record, contact_record, handling.checkpoint())
            _save(output / f'phase_{index:02d}_checks.json', checks)
            reject_failed_checks(checks['assessment'])
            support_record = support.finish(contact_record)
            handling_record = handling_checkpoint(handling, scene, support_owner.start)
            require(control_record['passed'] and payload_record['passed'], 'Native tail control/payload audit failed')
            projected = project_history(control_record, payload_record, support_record, start_step=begin,
                arm_order=names, identities=c['ids'], phase=phase, holder=holder, previous=previous,
                payload_owner_clock=payload_owner_clock)
            rows.extend(projected)
            previous = {k: copy.deepcopy(v) for k, v in projected[-1].items() if k != 'pre'}
            q = commands['position'][-1].copy()
            control_record.pop('history')
            payload_record.pop('history')
            handling_path = output / f'phase_{index:02d}_handling_checkpoint.json'
            _save(handling_path, handling_record)
            item = dict(phase=phase, actual_intervals=count, start_native_step=begin,
                end_native_step=int(scene.t), control=control_record, payload=payload_record,
                payload_actual_owner_clock=payload_owner_clock,
                contacts=contact_record, support=support_record,
                actual_holder_arm_id=holder, contact_permission_arm_id=permission,
                handling_checkpoint=handling_record,
                handling_checkpoint_file=str(handling_path),
                handling_checkpoint_file_sha256=_sha(handling_path),
                history_file=str(path), history_file_sha256=_sha(path))
            receipt['phases'].append(item)
            _save(output / f'phase_{index:02d}_receipt.json', item)
            check()
        segment('place', .5)
        before = c['observe']()
        arm_rest = measured_rest(rows, before['simulation_s'], c['settings']['execution_limits'], names)
        ready = assess_place_ready(before, rows, c['settings'], c['rest_policy'], driver.goal)
        _bead_material_sample(rows[-1], _materials(c['ids']), c['model'].geometry_record())
        require(arm_rest['passed'] and ready['passed'], 'Original actual support/arm/body placement readiness failed')
        snapshot = helper.snapshot(arm)
        require(before['provided_grasp']['current_snapshot'] == snapshot, 'Fresh supported snapshot differs')
        evidence = dict(ready=ready, arm_rest=arm_rest, snapshot=snapshot,
                        endpoint_sha256=digest(before), support=receipt['phases'][-1]['support'])
        def supported(actual):
            check()
            require(actual == snapshot, 'Support callback changed from actual same-clock native state')
            return dict(passed=True, native_clock=actual['native_clock'], body_entity_idx=actual['body_entity_idx'],
                support_id=driver.goal['support_id'], evidence_sha256=digest(evidence))
        release = helper.release(arm, support_ready=supported)
        driver.automatic_attachment_receipt = None
        driver.held = None
        after = c['observe']()
        require(physical_observation(before) == physical_observation(after)
                and after.get('provided_grasp') is None and helper.holder is None,
                'Actual native detach changed physical state/time or retained its holder')
        receipt['actual_release'] = release
        receipt['supported_release_evidence'] = evidence
        _save(output / 'actual_release.json', dict(receipt=release, evidence=evidence,
            before_observation_sha256=digest(before), after_observation_sha256=digest(after)))
        previous = copy.deepcopy(after)
        previous.update(step=int(scene.t), post_processing_complete=True, failures=[], actual_holder_arm_id=None)
        segment('release', 1.)
        segment('released_settle', 1.)
        targets = {name: q[9 * j + 7:9 * j + 9].copy() for j, name in enumerate(names)}
        extra = 0.
        while True:
            check()
            hard = released_numeric_gate(rows, settings=c['settings'], rest_policy=c['rest_policy'],
                goal=driver.goal, arm_order=names, finger_targets=targets, load_bearing_s=load,
                model=c['model'], identities=c['ids'], strict_material=False)
            if hard['bead_bulk_rest']['passed']:
                gate = released_numeric_gate(rows, settings=c['settings'], rest_policy=c['rest_policy'],
                    goal=driver.goal, arm_order=names, finger_targets=targets, load_bearing_s=load,
                    model=c['model'], identities=c['ids'], strict_material=True)
                break
            require(extra + .5 <= 5. + 1e-9, 'Original extra material-rest allowance exhausted')
            segment('released_settle', .5)
            extra += .5
        final = c['observe']()
        require(final['native_scene_step'] == rows[-1]['native_scene_step']
                and final.get('provided_grasp') is None and helper.holder is None,
                'Final actual no-weld native boundary differs')
        final_handling = handling_checkpoint(handling, scene, support_owner.start, final=True)
        require(final_handling['qualified_holder_id'] is None
                and final_handling['release_count'] == handling.seed['release_count'] + 1
                and final_handling['supported_pickup_count'] == handling.seed['supported_pickup_count']
                    + int(handling.seed['qualified_holder_id'] is None),
                'Persistent handling must retain the actual supported pickup and single release chronology')
        final_handling_path = output / 'actual_final_handling.json'
        _save(final_handling_path, final_handling)
        require(all(_sha(p) == h for p, h in sources.items()), 'Tail implementation changed during execution')
        receipt.update(status='PROVIDED_DEFERRED_BEAD_TAIL_PASS', passed=True,
            actual_final_observation=final, final_released_gate=gate, additional_material_wait_s=extra,
            source_hashes=sources, source_bindings_verified=True,
            original_rest_thresholds_unchanged=True, native_zero_time_release=True,
            actual_release_verified=True, summary='Distinct actual supported-release and rest evidence. Caller must bind the complete saved carry and resource receipt separately.')
        receipt.update(final_handling=final_handling, final_handling_file=str(final_handling_path),
            final_handling_file_sha256=_sha(final_handling_path),
            persistent_original_handling_verified=True)
    except BaseException as exc:
        receipt['failure'] = dict(type=type(exc).__name__, message=str(exc))
        failed_handling = getattr(exc, 'handling_record', None)
        if failed_handling is not None:
            failed_path = output / 'actual_failed_handling.json'
            _save(failed_path, failed_handling)
            receipt['failed_handling_file'] = str(failed_path)
            receipt['failed_handling_file_sha256'] = _sha(failed_path)
        raise
    finally:
        receipt.update(final_native_step=int(scene.t), actual_tail_intervals=int(scene.t)-receipt['start_native_step'],
            tail_host_wall_s=time.perf_counter()-started)
        _save(output / 'TAIL_RESULT.json', receipt)
    return receipt
