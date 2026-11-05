"""Persistent, opt-in GPU port of the original sampled bead handling guard.

This module reads native state. It never changes physics, a holder, or a pose.
The exact legacy guard is transferred once, including its old unheld anchor.
Checks remain at every original 16-step PRE/POST pair. Host readback is only
at explicit boundaries. Deferred sticky failure is not immediate CPU stopping.
"""
import copy
import hashlib
import math
from pathlib import Path
import threading
import numpy as np

from transport_challenge.contracts import ContractError, digest
from physics.episode_handling_guard import MAX_UNHELD_DISPLACEMENT_M, CLOCK_TOLERANCE_S
from physics.episode_handling_enforcement import MAX_HELD_ROOT_TO_HAND_M, SampledHandlingEnforcer

DT = .0025
SUBSTEPS = 16
REASONS = {1:'contact_witness_failed', 2:'native_clock_or_pair_failed',
    3:'nonfinite_native_root_or_hand', 4:'cup_native_mass_changed',
    5:'held_root_to_hand_exceeded', 6:'authenticated_hand_reach_exceeded',
    7:'authenticated_root_reach_exceeded', 8:'direct_holder_switch',
    9:'unheld_root_displacement_exceeded', 10:'acquisition_without_named_fixed_support'}
_KERNELS = {}


def require(value, message):
    if not value:
        raise ContractError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contact_permission(actual_holder, permission_arm_id, names):
    """A pad-contact permission cannot invent a native qualified holder."""
    require(actual_holder is None or actual_holder in names, 'Unknown native holder')
    permission = actual_holder if permission_arm_id is None else permission_arm_id
    require(permission is None or permission in names, 'Unknown actual contact permission arm')
    require(actual_holder is None or actual_holder == permission,
            'A qualified holder must be the actual authorized manipulation actor')
    return permission


def bind_cup_mass(public_mass, actual_native_mass, native_dtype):
    """Keep original host mass and its exact declared native cast distinct."""
    require(type(public_mass) in (int,float) and math.isfinite(public_mass) and public_mass > 0,
            'Original positive public one-link cup mass required')
    dtype = np.dtype(native_dtype)
    require(dtype in (np.dtype('float32'),np.dtype('float64')), 'Original native float dtype required')
    actual = np.asarray(actual_native_mass)
    expected = np.asarray(public_mass, dtype=dtype)
    require(actual.size == 1 and actual.dtype == dtype and np.isfinite(actual).all()
            and float(actual.reshape(-1)[0]) == float(expected),
            'Initial native cup mass differs from the exact declared precision cast')
    return float(expected)


def transition_cpu(previous, *, root, holder, support, hand=None, shoulder=None, radius=None):
    """CPU-only differential fixture for the port, never a native certificate."""
    s = copy.deepcopy(previous)
    root = np.asarray(root, np.float64)
    require(root.shape == (3,) and np.isfinite(root).all(), 'Finite measured root required')
    if holder is not None:
        require(hand is not None and shoulder is not None and radius is not None,
                'Actual qualified hand and authenticated reach are required')
        hand, shoulder = np.asarray(hand, float), np.asarray(shoulder, float)
        if not np.isfinite(hand).all() or not np.isfinite(shoulder).all():
            return s, 3
        if math.dist(root, hand) > MAX_HELD_ROOT_TO_HAND_M:
            return s, 5
        if math.dist(hand, shoulder) > radius:
            return s, 6
        if math.dist(root, shoulder) > radius + MAX_HELD_ROOT_TO_HAND_M:
            return s, 7
    if s['holder'] is not None and holder is not None and holder != s['holder']:
        return s, 8
    if s['holder'] is None or holder is None:
        distance = math.dist(root, s['anchor'])
        if distance > MAX_UNHELD_DISPLACEMENT_M:
            return s, 9
        s['maximum_displacement_m'] = max(s['maximum_displacement_m'], distance)
    if s['holder'] is None and holder is not None:
        if not support:
            return previous, 10
        s['pickup_count'] += 1
    if s['holder'] is not None and holder is None:
        s['release_count'] += 1
    if holder is not None:
        s['anchor'] = root.tolist()
    s['holder'] = holder
    return s, 0


def _kernels(qd, arrays):
    if id(qd) in _KERNELS:
        return _KERNELS[id(qd)]

    @qd.func
    def finite(value: qd.f64):
        mask = qd.i64(9218868437227405312)
        return (qd.bit_cast(value, qd.i64) & mask) != mask

    @qd.kernel
    def capture(ds: arrays.DynState, di: arrays.DynInfo, rc: qd.template(),
                cs: arrays.ColliderState, classifier: qd.types.ndarray(),
                categories: qd.types.ndarray(), links: qd.types.ndarray(),
                reach: qd.types.ndarray(), cfg: qd.types.ndarray(),
                state: qd.types.ndarray(), values: qd.types.ndarray(),
                expected_mass: qd.f64, ordinal: qd.i32, pre: qd.template(),
                capacity: qd.template()):
        for singleton in range(1):
            bad = 0
            holder = cfg[0]
            root = qd.cast(ds.links.pos[links[0], 0], qd.f64)
            hand = root
            shoulder = root
            radius = qd.f64(0.0)
            root_hand = qd.f64(0.0)
            hand_reach = qd.f64(0.0)
            root_reach = qd.f64(0.0)
            displacement = qd.f64(0.0)
            support = 0
            mass = qd.f64(0.0)
            if qd.static(rc.batch_links_info):
                mass = qd.cast(di.links.inertial_mass[links[0], 0], qd.f64)
            else:
                mass = qd.cast(di.links.inertial_mass[links[0]], qd.f64)
            if classifier[0] != 0:
                bad = 1
            if not finite(mass) or mass <= qd.f64(0.0) or mass != expected_mass:
                if bad == 0:
                    bad = 4
            for j in qd.static(range(3)):
                if not finite(root[j]):
                    if bad == 0:
                        bad = 3
            if qd.static(pre):
                if (classifier[15] != ordinal or state[9] != state[10]
                        or ordinal != state[10] + 1):
                    if bad == 0:
                        bad = 2
                count = cs.n_contacts[0]
                if count < 0 or count > capacity:
                    if bad == 0:
                        bad = 1
                else:
                    for i in range(count):
                        if categories[i] == 1:
                            support = 1
                state[7] += 1
                state[9] = ordinal
                state[11] = support
                state[12] = holder
                values[6] = mass
            else:
                if (classifier[16] != ordinal or classifier[15] != ordinal
                        or state[9] != ordinal or state[10] != ordinal - 1
                        or state[12] != holder or values[6] != mass):
                    if bad == 0:
                        bad = 2
                support = state[11]
                if holder >= 0:
                    hand = qd.cast(ds.links.pos[links[holder + 1], 0], qd.f64)
                    shoulder = qd.Vector([reach[holder, 0], reach[holder, 1], reach[holder, 2]])
                    radius = reach[holder, 3]
                    for j in qd.static(range(3)):
                        if not finite(hand[j]) or not finite(shoulder[j]):
                            if bad == 0:
                                bad = 3
                    delta = root - hand
                    root_hand = qd.sqrt(delta.dot(delta))
                    delta = hand - shoulder
                    hand_reach = qd.sqrt(delta.dot(delta))
                    delta = root - shoulder
                    root_reach = qd.sqrt(delta.dot(delta))
                    if not finite(root_hand) or root_hand > qd.f64(.20):
                        if bad == 0:
                            bad = 5
                    if not finite(hand_reach) or hand_reach > radius:
                        if bad == 0:
                            bad = 6
                    if not finite(root_reach) or root_reach > radius + qd.f64(.20):
                        if bad == 0:
                            bad = 7
                    values[5] = qd.max(values[5], root_hand)
                if state[4] >= 0 and holder >= 0 and state[4] != holder:
                    if bad == 0:
                        bad = 8
                if state[4] < 0 or holder < 0:
                    anchor = qd.Vector([values[0], values[1], values[2]])
                    delta = root - anchor
                    displacement = qd.sqrt(delta.dot(delta))
                    if not finite(displacement) or displacement > qd.f64(.005):
                        if bad == 0:
                            bad = 9
                if state[4] < 0 and holder >= 0 and support == 0:
                    if bad == 0:
                        bad = 10
                if bad == 0 and state[0] == 0:
                    if state[4] < 0 or holder < 0:
                        values[3] = displacement
                        values[4] = qd.max(values[4], displacement)
                    if state[4] < 0 and holder >= 0:
                        state[5] += 1
                    if state[4] >= 0 and holder < 0:
                        state[6] += 1
                    if holder >= 0:
                        for j in qd.static(range(3)):
                            values[j] = root[j]
                        values[3] = qd.f64(0.0)
                        state[17] = ordinal + 1
                    state[4] = holder
                    state[3] = ordinal + 1
                    state[15] += 1
                state[8] += 1
                state[10] = ordinal
            if bad != 0 and state[0] == 0:
                state[0] = 1
                state[1] = ordinal + qd.cast(not pre, qd.i32)
                state[2] = bad
                state[13] = qd.cast(not pre, qd.i32)
                state[14] = holder
                for j in qd.static(range(3)):
                    values[8+j] = root[j]
                    values[11+j] = hand[j]
                    values[14+j] = values[j]
                values[17] = root_hand
                values[18] = hand_reach
                values[19] = root_reach
                values[20] = displacement
                values[21] = mass
                state[16] = support
    _KERNELS[id(qd)] = capture
    return capture


class DeviceHandlingAudit:
    """Transfer the live legacy state once, then preserve it across all phases."""
    def __init__(self, context, contact_audit):
        from physics.deferred_gpu_contacts import DeviceContactAudit
        e = context['handling_enforcer']
        require(type(e) is SampledHandlingEnforcer and e.rigid_profile == 'rigid16_bead_tc2h_v1',
                'Transfer the actual live original rigid16 handling owner')
        require(type(contact_audit) is DeviceContactAudit and contact_audit.sample_mode == 'bead16',
                'Actual original bead16 native classifier required')
        self.scene = context['scene']; self.r = self.scene.rigid_solver
        self.tray = context['tray']; self.helper = context['automatic_grasp']
        self.arms = dict(context['arms']); self.hands = dict(context['hands'])
        if getattr(self.scene,'_ap_single_active_arm_contract',None) is not None:
            from physics.fixed_peer_arms import validate_scene_contract
            validate_scene_contract(self.scene,self.arms)
        self.names = tuple(contact_audit.policy['arm_ids'])
        self.qd, self.torch, self.gs = contact_audit.qd, contact_audit.torch, contact_audit.gs
        require(self.scene is contact_audit.scene and self.tray.n_links == 1
                and self.tray._solver is self.r and self.tray.scene is self.scene
                and self.scene.sim.dt == DT and self.scene.sim.substeps == SUBSTEPS,
                'Handling must use the original native one-link cup and clock')
        self.thread = threading.get_ident(); self.closed = False
        from physics.gravity_jacobian_batch import _load_dependencies
        _, _, _, _, misc = _load_dependencies()
        self.mass_view = misc.qd_to_torch(self.r.dyn_info.links.inertial_mass, copy=False)
        mass_shape = (self.r.n_links,1) if self.r.rigid_config.batch_links_info else (self.r.n_links,)
        require(tuple(self.mass_view.shape) == mass_shape and self.mass_view.dtype == self.gs.tc_float
                and self.mass_view.device.type == 'cuda', 'Actual native cup mass field ABI differs')
        self.seed = e.guard.status(); self.envelope = copy.deepcopy(e.envelope)
        self.reach = copy.deepcopy(e.reach_envelopes); self.identity = copy.deepcopy(e.identity)
        require(self.seed['passed'] is True and self.seed['first_failure'] is None
                and self.seed['body_identity'] == self.identity
                and self.identity['entity_idx'] == int(self.tray.idx)
                and self.identity['model_sha256'] == context['model'].geometry_sha256
                and self.envelope['passed'] is True and self.envelope['radius_limit_m'] == .20,
                'Original body identity/envelope or prefix guard failed')
        require(self.seed['dt_s'] == DT/SUBSTEPS
                and self.seed['last_accepted_scene_step'] == int(self.scene.t)*SUBSTEPS
                and abs(self.seed['last_accepted_simulation_s']-int(self.scene.t)*DT) <= CLOCK_TOLERANCE_S
                and self.seed['unheld_displacement_limit_m'] == .005
                and set(self.names) == set(self.seed['known_holder_ids']) == set(self.arms) == set(self.hands),
                'Transfer the exact current legacy clock, holders and old anchor')
        require(self.reach['passed'] is True and set(self.reach['by_arm']) == set(self.names),
                'Actual authenticated fixed-base reach proof required')
        links = [self.tray.base_link_idx] + [int(self.hands[n].idx) for n in self.names]
        require(links == list(contact_audit.policy['pose_link_ids'][:1+len(self.names)])
                and all(self.hands[n].entity is self.arms[n] for n in self.names),
                'Contact and handling must read the same actual cup/hand links')
        reach = [[*self.reach['by_arm'][n]['shoulder_world_m'],
                  self.reach['by_arm'][n]['accepted_hand_radius_m']] for n in self.names]
        require(np.asarray(reach).shape == (len(self.names),4) and np.isfinite(reach).all()
                and all(row[3] > 0 for row in reach), 'Finite original reach radii required')
        snapshot = self.helper.snapshot(self.helper.holder or self.names[0])
        self.mass = float(snapshot['cup_mass_kg'])
        require(math.isfinite(self.mass) and self.mass > 0, 'Actual positive native cup mass required')
        native_mass = self.mass_view[self.tray.base_link_idx].detach().cpu().numpy().copy()
        self.native_dtype = np.dtype(self.gs.np_float)
        self.expected_native_mass = bind_cup_mass(self.mass,native_mass,self.native_dtype)
        self.initial_native_mass_sha256 = hashlib.sha256(native_mass.tobytes()).hexdigest()
        def upload(v, integer=False):
            return self.torch.tensor(v, dtype=self.torch.int64 if integer else self.torch.float64,
                                     device=self.gs.device)
        s = np.zeros(24, np.int64); s[1] = -1; s[3] = self.seed['last_accepted_scene_step']
        s[4] = -1 if self.seed['qualified_holder_id'] is None else self.names.index(self.seed['qualified_holder_id'])
        s[5:7] = [self.seed['supported_pickup_count'], self.seed['release_count']]
        s[9:11] = s[3]-1; s[15] = self.seed['observations_accepted']
        s[17] = self.seed['anchor_scene_step']
        v = np.zeros(24, np.float64); v[:3] = self.seed['anchor_world_m']
        v[3] = self.seed['current_unheld_displacement_m'] or 0.
        v[4] = self.seed['maximum_unheld_displacement_m']; v[5] = e.maximum_held_root_to_hand_m
        self.state, self.values = upload(s, True), upload(v)
        self.links, self.radii, self.config = upload(links, True), upload(reach), upload([-1], True)
        from genesis.utils import array_class as arrays
        self.capture = _kernels(self.qd, arrays)
        self.pre_count = self.post_count = 0; self.last_ordinal = s[3]-1
        self.phases = []; self.binding = self._signature()
        self.sources = {str(Path(__file__).resolve()): _sha(__file__)}
        if getattr(self.scene,'_ap_single_active_arm_contract',None) is not None:
            p=Path(__file__).with_name('fixed_peer_arms.py')
            self.sources[str(p)]=_sha(p)
        for name in ('episode_handling_guard.py','episode_handling_enforcement.py',
                     'deferred_gpu_contacts.py','provided_automatic_grasp.py'):
            p = Path(__file__).with_name(name); self.sources[str(p)] = _sha(p)
        self.bind_contact(contact_audit, contact_audit.phases[-1]['phase'], self.helper.holder, snapshot)

    def _signature(self):
        from physics.live_contact_coverage import _native_buffer
        return (id(self.scene), id(self.r), id(self.r.dyn_state), id(self.r.dyn_info),
                id(self.tray), self.tray.base_link_idx,
                id(_native_buffer(self.r.dyn_info.links.inertial_mass)), self.mass_view.data_ptr(),
                tuple(self.mass_view.shape), str(self.mass_view.dtype),
                tuple((n,id(self.arms[n]),id(self.hands[n]),int(self.hands[n].idx)) for n in self.names))

    def bind_contact(self, contact_audit, phase, actual_holder, snapshot, *, permission_arm_id=None):
        from physics.deferred_gpu_contacts import DeviceContactAudit
        from physics.live_contact_coverage import automatic_grasp_witness
        permission = contact_permission(actual_holder, permission_arm_id, self.names)
        require(not self.closed and self.pre_count == self.post_count
                and self._signature() == self.binding and threading.get_ident() == self.thread,
                'Handling boundaries cannot split an internal pair or replace native owners')
        require(type(contact_audit) is DeviceContactAudit and not contact_audit.finalized
                and contact_audit.scene is self.scene and contact_audit.sample_mode == 'bead16'
                and contact_audit.policy['arm_ids'] == self.names
                and contact_audit.phases[-1]['phase'] == phase
                and contact_audit.phases[-1]['active_holder_arm_id'] == permission
                and self.helper.holder == actual_holder,
                'Rebind from the actual same-clock helper and contact phase only')
        proof = automatic_grasp_witness(self.helper,scene=self.scene,
                                       coverage_policy_sha256=contact_audit.coverage_sha256)
        require(proof['holder_arm_id'] == actual_holder and snapshot == proof['snapshot'],
                'Boundary holder must have the unchanged actual owned-weld membership witness')
        require(snapshot['native_clock'][:2] == [int(self.scene.t),int(self.scene.t)*DT]
                and snapshot['body_entity_idx'] == self.identity['entity_idx']
                and snapshot['cup_link_idx'] == self.tray.base_link_idx
                and snapshot['cup_mass_kg'] == self.mass
                and snapshot['inventory'] == contact_audit.phases[-1]['expected_inventory'],
                'Handling phase must bind the exact native clock, identity, mass and equality rows')
        require(set(v['id'] for v in contact_audit.policy['registries'][3].values())
                <= set(self.seed['allowed_support_ids']), 'Classifier names an unauthenticated support')
        self.audit = contact_audit; self.audit_offset = dict(contact_audit.calls)
        self.phase_start = self.pre_count
        holder = -1 if actual_holder is None else self.names.index(actual_holder)
        self.config.copy_(self.torch.tensor([holder],dtype=self.torch.int64,device=self.gs.device))
        self.phases.append(dict(phase=phase,actual_holder=actual_holder,
            active_contact_permission_arm=contact_audit.phases[-1]['active_holder_arm_id'],
            global_ordinal=int(self.scene.sim.cur_substep_global),snapshot_sha256=digest(snapshot),
            automatic_witness_sha256=proof['witness_sha256'],
            contact_policy_sha256=contact_audit.policy['sha256']))

    def _capture(self, ordinal, pre):
        require(not self.closed and threading.get_ident() == self.thread and self._signature() == self.binding
                and type(ordinal) is int and ordinal == self.scene.sim.cur_substep_global,
                'Handling capture requires the actual native phase and unchanged owner')
        expected = self.post_count-self.phase_start+1
        require(self.audit.calls['pre']-self.audit_offset['pre'] == expected
                and self.audit.calls['post']-self.audit_offset['post'] == expected-int(pre)
                and self.pre_count == self.post_count + int(not pre),
                'Handling must follow each same-clock owned PRE/POST classifier exactly once')
        require(ordinal == self.last_ordinal+int(pre), 'Skipped or repeated internal handling clock')
        self.capture(self.r.dyn_state,self.r.dyn_info,self.r.rigid_config,self.r.collider._collider_state,
            self.audit.state,self.audit.categories,self.links,self.radii,self.config,self.state,self.values,
            self.expected_native_mass,ordinal,pre,self.audit.capacity)
        if pre:
            self.pre_count += 1; self.last_ordinal = ordinal
        else:
            self.post_count += 1

    def pre(self, ordinal):
        self._capture(ordinal, True)

    def post(self, ordinal):
        self._capture(ordinal, False)

    def checkpoint(self):
        require(not self.closed and self.pre_count == self.post_count
                and self._signature() == self.binding
                and int(self.scene.t)*SUBSTEPS == self.last_ordinal+1,
                'Handling checkpoint requires a completed actual outer boundary')
        s = self.state.detach().cpu().numpy().copy(); v = self.values.detach().cpu().numpy().copy()
        require(int(s[7]) == self.pre_count and int(s[8]) == self.post_count,
                'GPU handling inspection count differs from actual submitted phases')
        source_ok = all(_sha(p) == h for p,h in self.sources.items())
        holder = None if s[4] < 0 else self.names[int(s[4])]
        def number(value):
            value = float(value)
            return value if math.isfinite(value) else 'nonfinite:'+repr(value)
        first = None if not s[0] else dict(produced_global_substep=int(s[1]),reason=REASONS[int(s[2])],
            phase='POST' if s[13] else 'PRE',qualified_holder=None if s[14]<0 else self.names[int(s[14])],
            named_support=bool(s[16]),root_world_m=list(map(number,v[8:11])),hand_world_m=list(map(number,v[11:14])),
            anchor_world_m=list(map(number,v[14:17])),root_to_hand_m=number(v[17]),hand_to_shoulder_m=number(v[18]),
            root_to_shoulder_m=number(v[19]),unheld_displacement_m=number(v[20]),cup_mass_kg=number(v[21]))
        return dict(schema_version='deferred-device-handling-audit-v1',passed=not bool(s[0]) and source_ok,
            sticky_failed=bool(s[0]),first_failure=first,PRE_samples=self.pre_count,POST_samples=self.post_count,
            start_global_substep=self.seed['last_accepted_scene_step'],last_accepted_global_substep=int(s[3]),
            qualified_holder_id=holder,anchor_world_m=v[:3].tolist(),
            anchor_scene_step=int(s[17]),anchor_simulation_s=int(s[17])*(DT/SUBSTEPS),
            maximum_unheld_displacement_m=float(v[4]),maximum_held_root_to_hand_m=float(v[5]),
            current_unheld_displacement_m=None if holder is not None else float(v[3]),
            supported_pickup_count=int(s[5]),release_count=int(s[6]),
            actual_cup_mass_kg=self.expected_native_mass,source_public_cup_mass_kg=self.mass,
            native_mass_dtype=self.native_dtype.str,initial_native_cup_mass_sha256=self.initial_native_mass_sha256,
            held_root_to_hand_limit_m=.20,unheld_displacement_limit_m=.005,
            prefix_guard=copy.deepcopy(self.seed),native_reach_envelopes=copy.deepcopy(self.reach),
            body_root_envelope=copy.deepcopy(self.envelope),phase_bindings=copy.deepcopy(self.phases),
            source_hashes=dict(self.sources),source_bindings_verified=source_ok,
            every_original_internal_sample_preserved=True,per_sample_host_transfers=0,
            immediate_CPU_stop_preserved=False,legacy_handling_record_claimed=False)

    def finish(self):
        result = self.checkpoint(); self.closed = True
        return result
