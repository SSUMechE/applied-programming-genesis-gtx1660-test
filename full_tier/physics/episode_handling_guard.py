"""Pure, sampled handling restrictions for one persistent authored-root vessel.

The caller must bind measured body-root and native contact facts to their actual
clocks. A qualified holder means validated bilateral finger-pad/handle contact,
not a command, phase name, IK pose or cached grasp. Support IDs mean validated
bottom/support witnesses. ``manipulation_contacts_allowed`` comes from complete
native ownership/contact checks. If contacts describe the pre-integration pose
but the root is post-step, this is an interval-qualified sampled assertion, NOT
proof that the post-step contact still exists. That integration must disclose
both clocks and separately enforce the post-root reach bound before accepting
the assertion. This module cannot authenticate facts, certify unsampled motion
or prove physical delivery. It never imports a simulator or changes state.

The unheld budget is displacement from one fixed anchor, not an allowance per
step. Initially that anchor is the measured source root. At loss of grasp it is
the LAST observed held root, including the first unheld sample's displacement.
Only a valid, supported acquisition ends that unheld episode. Phase, pause,
support contact and status reads cannot reset its anchor.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from numbers import Integral, Real
import re

from transport_challenge.contracts import ContractError

MAX_UNHELD_DISPLACEMENT_M = .005
MAX_BODY_ROOT_RADIUS_M = .20
CLOCK_TOLERANCE_S = 1e-9
_OBSERVATION_KEYS = {
    'schema_version', 'body_identity', 'scene_step', 'simulation_s',
    'body_root_world_m', 'qualified_holder_id', 'support_ids',
    'manipulation_contacts_allowed',
}


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractError(label+' must be a finite real number, not bool')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ContractError(label+' must be finite'+(' and positive' if positive else ''))
    return value


def _point(value, label):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ContractError(label+' must contain exactly three coordinates')
    return tuple(_number(v, label) for v in value)


def _index(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ContractError(label+' must be a nonnegative integer')
    return int(value)


def _name(value, label):
    if type(value) is not str or not value or value.strip() != value:
        raise ContractError(label+' must be an explicit nonempty name')
    return value


def _names(value, label, *, nonempty=False):
    if not isinstance(value, (tuple, list)):
        raise ContractError(label+' must be an explicit list/tuple of names')
    names = tuple(_name(v, label) for v in value)
    if len(set(names)) != len(names) or (nonempty and not names):
        raise ContractError(label+' must contain unique'+(' nonempty' if nonempty else '')+' names')
    return tuple(sorted(names))


def _identity(value):
    if not isinstance(value, dict) or set(value) != {'id', 'entity_idx', 'model_sha256'}:
        raise ContractError('Body identity requires id, native entity_idx and model_sha256')
    digest = value['model_sha256']
    if type(digest) is not str or re.fullmatch('[0-9a-f]{64}', digest) is None:
        raise ContractError('Body model_sha256 must be a lowercase SHA-256 of the authored-root model')
    return {'id': _name(value['id'], 'body id'),
            'entity_idx': _index(value['entity_idx'], 'body entity_idx'),
            'model_sha256': digest}


def check_body_root_envelope(component_boxes):
    """Check every corner of actual axis-aligned boxes in the fixed body frame.

    Input is exactly ``build_container_geometry(...)["component_boxes"]``:
    a nonempty list/tuple of (size_xyz_m, center_xyz_m) pairs. Dimensions and
    centers are in the immutable authored root, never a recentered COM or hand
    frame. Every positive-size box's farthest point is a corner, so this checks
    all points of that box. Other/rotated primitive formats are rejected rather
    than silently reduced to these pairs. Caller must bind this inventory to
    ALL actual native solids/tools. Passing this input cannot prove that an
    unlisted attachment is absent from a simulator.
    """
    if not isinstance(component_boxes, (tuple, list)) or not component_boxes:
        raise ContractError('Provide all nonempty actual authored-root component boxes')
    normalized = []
    maximum = -1.
    farthest = None
    outside = []
    for index, box in enumerate(component_boxes):
        if not isinstance(box, (tuple, list)) or len(box) != 2:
            raise ContractError('Only explicit (size_xyz_m, center_xyz_m) box pairs are supported')
        size, center = _point(box[0], 'component size'), _point(box[1], 'component center')
        if min(size) <= 0:
            raise ContractError('Every component size must be positive')
        normalized.append([list(size), list(center)])
        component_maximum = -1.
        for signs in itertools.product((-1, 1), repeat=3):
            corner = tuple(c + s*h/2 for c, s, h in zip(center, signs, size))
            if not all(math.isfinite(v) for v in corner):
                raise ContractError('A component corner overflowed')
            radius = math.hypot(*corner)
            if not math.isfinite(radius):
                raise ContractError('A component radius overflowed')
            component_maximum = max(component_maximum, radius)
            if radius > maximum:
                maximum = radius
                farthest = {'component_index': index, 'corner_body_m': list(corner)}
        if component_maximum > MAX_BODY_ROOT_RADIUS_M:
            outside.append(index)
    digest = hashlib.sha256(json.dumps(normalized, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return {'schema_version': 'authored-body-root-envelope-check-v1',
            'passed': not outside, 'radius_limit_m': MAX_BODY_ROOT_RADIUS_M,
            'maximum_corner_radius_m': maximum, 'farthest_corner': farthest,
            'violating_component_indices': outside, 'component_count': len(normalized),
            'corner_count': 8*len(normalized), 'component_boxes_sha256': digest,
            'frame': 'immutable_authored_body_root', 'numeric_allowance_m': 0.,
            'native_inventory_authenticated': False}


class UnheldMotionGuard:
    """Persistent sampled state machine. No reset, phase or pause mutation API.

    Construction binds an INITIAL SOURCE anchor and its actual clock. The first
    observe call must be the next native step. All subsequent new observations
    are contiguous. An exact factual repeat at the same step/time is idempotent
    for a pause. Malformed/new-clock failures latch and return failed evidence.
    Constructor errors raise ContractError. A new holder must have a current
    named support witness, and must pass the OLD anchor limit before acquiring.
    Holder A -> B without an intervening unheld observation is rejected by this
    supported-handoff contract. Continuing the same holder needs no support.
    """

    def __init__(self, *, body_identity, known_holder_ids, allowed_support_ids,
                 initial_root_world_m, initial_scene_step, initial_simulation_s,
                 dt_s=.0025):
        self._identity = _identity(body_identity)
        self._holders = _names(known_holder_ids, 'known holder IDs', nonempty=True)
        if len(self._holders) > 3:
            raise ContractError('This contract permits at most three known holders')
        self._supports = _names(allowed_support_ids, 'allowed fixed support IDs', nonempty=True)
        self._dt = _number(dt_s, 'dt_s', positive=True)
        self._initial_step = _index(initial_scene_step, 'initial scene step')
        self._initial_time = _number(initial_simulation_s, 'initial simulation time')
        if self._initial_time < 0 or not self._on_grid(self._initial_step, self._initial_time):
            raise ContractError('Initial native step and time must agree on the configured grid')
        self._initial_root = _point(initial_root_world_m, 'initial measured source root')
        self._anchor = self._initial_root
        self._anchor_step = self._initial_step
        self._anchor_time = self._initial_time
        self._last_step = self._initial_step
        self._last_time = self._initial_time
        self._last_facts = None
        self._holder = None
        self._count = 0
        self._release_count = 0
        self._pickup_count = 0
        self._maximum_displacement = 0.
        self._current_displacement = 0.
        self._failure = None

    def _on_grid(self, step, time_s):
        try:
            expected = step*self._dt
            return math.isfinite(expected) and abs(time_s-expected) <= CLOCK_TOLERANCE_S
        except OverflowError:
            return False

    def _facts(self, value):
        if not isinstance(value, dict) or set(value) != _OBSERVATION_KEYS:
            raise ContractError('Supply exactly the handling observation schema fields')
        if value['schema_version'] != 'episode-handling-observation-v1':
            raise ContractError('Unknown handling observation schema')
        holder = value['qualified_holder_id']
        if holder is not None:
            holder = _name(holder, 'qualified holder')
        if type(value['manipulation_contacts_allowed']) is not bool:
            raise ContractError('manipulation_contacts_allowed must be an explicit measured-classifier bool')
        return {'schema_version': value['schema_version'],
                'body_identity': _identity(value['body_identity']),
                'scene_step': _index(value['scene_step'], 'scene step'),
                'simulation_s': _number(value['simulation_s'], 'simulation time'),
                'body_root_world_m': list(_point(value['body_root_world_m'], 'measured body root')),
                'qualified_holder_id': holder,
                'support_ids': list(_names(value['support_ids'], 'observed support IDs')),
                'manipulation_contacts_allowed': value['manipulation_contacts_allowed']}

    def _fail(self, reason, *, facts=None, detail=None):
        if self._failure is None:
            self._failure = {'reason': reason, 'detail': detail,
                             'factual_observation': copy.deepcopy(facts),
                             'anchor_world_m': list(self._anchor),
                             'anchor_scene_step': self._anchor_step,
                             'anchor_simulation_s': self._anchor_time}
        return self.status()

    def observe(self, observation):
        """Consume one detached factual dict. Every rejection is permanently latched."""
        if self._failure is not None:
            return self.status()
        try:
            f = self._facts(observation)
        except (ContractError, TypeError, ValueError, OverflowError) as exc:
            return self._fail('malformed_observation', detail=str(exc))
        if f['body_identity'] != self._identity:
            return self._fail('persistent_body_identity_changed', facts=f)
        if f['qualified_holder_id'] is not None and f['qualified_holder_id'] not in self._holders:
            return self._fail('unknown_qualified_holder', facts=f)
        if not set(f['support_ids']) <= set(self._supports):
            return self._fail('unknown_or_nonfixed_support', facts=f)
        if f == self._last_facts:
            return self.status()
        if (f['scene_step'], f['simulation_s']) == (self._last_step, self._last_time):
            return self._fail('conflicting_same_clock_observation', facts=f)
        if f['scene_step'] != self._last_step+1:
            return self._fail('noncontiguous_native_step', facts=f)
        if (not self._on_grid(f['scene_step'], f['simulation_s'])
                or f['simulation_s'] <= self._last_time
                or abs(f['simulation_s']-self._last_time-self._dt) > CLOCK_TOLERANCE_S):
            return self._fail('native_time_grid_mismatch', facts=f)
        if f['manipulation_contacts_allowed'] is not True:
            return self._fail('forbidden_manipulation_contact', facts=f)
        holder = f['qualified_holder_id']
        root = tuple(f['body_root_world_m'])
        if self._holder is not None and holder is not None and holder != self._holder:
            return self._fail('direct_holder_switch_without_supported_regrasp', facts=f)
        # The anchor still is the last held observation, even on the FIRST loss
        # sample. Acquisition is checked before clearing this old anchor.
        if self._holder is None or holder is None:
            distance = math.dist(root, self._anchor)
            if not math.isfinite(distance):
                return self._fail('nonfinite_root_displacement', facts=f)
            self._current_displacement = distance
            self._maximum_displacement = max(self._maximum_displacement, distance)
            if distance > MAX_UNHELD_DISPLACEMENT_M:
                return self._fail('unheld_root_displacement_exceeded', facts=f,
                                  detail={'displacement_m': distance, 'limit_m': MAX_UNHELD_DISPLACEMENT_M})
        if self._holder is None and holder is not None:
            if not f['support_ids']:
                return self._fail('acquisition_without_named_fixed_support', facts=f)
            self._pickup_count += 1
        if self._holder is not None and holder is None:
            self._release_count += 1
        if holder is not None:
            self._anchor = root
            self._anchor_step = f['scene_step']
            self._anchor_time = f['simulation_s']
            self._current_displacement = None
        self._holder = holder
        self._last_step = f['scene_step']
        self._last_time = f['simulation_s']
        self._last_facts = f
        self._count += 1
        return self.status()

    def status(self):
        """Detached bounded evidence. Reading it never resets or advances state."""
        return copy.deepcopy({
            'schema_version': 'episode-handling-guard-status-v1',
            'passed': self._failure is None,
            'status': ('SAMPLED_HANDLING_CONSISTENT' if self._failure is None else 'SAMPLED_HANDLING_GUARD_FAIL'),
            'scope': 'Pure sampled caller facts only; native authentication and runtime enforcement are separate.',
            'body_identity': self._identity,
            'known_holder_ids': list(self._holders), 'allowed_support_ids': list(self._supports),
            'dt_s': self._dt, 'clock_tolerance_s': CLOCK_TOLERANCE_S,
            'unheld_displacement_limit_m': MAX_UNHELD_DISPLACEMENT_M,
            'initial_source_anchor_world_m': list(self._initial_root),
            'initial_scene_step': self._initial_step, 'initial_simulation_s': self._initial_time,
            'last_accepted_scene_step': self._last_step, 'last_accepted_simulation_s': self._last_time,
            'observations_accepted': self._count,
            'qualified_holder_id': self._holder,
            'anchor_world_m': list(self._anchor), 'anchor_scene_step': self._anchor_step,
            'anchor_simulation_s': self._anchor_time,
            'current_unheld_displacement_m': self._current_displacement,
            'maximum_unheld_displacement_m': self._maximum_displacement,
            'release_count': self._release_count, 'supported_pickup_count': self._pickup_count,
            'first_failure': self._failure,
        })

    def snapshot(self):
        return self.status()
