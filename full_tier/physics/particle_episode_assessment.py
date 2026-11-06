"""Particle-specific sampled relay and current-holder assessments.

The nonmaterial guards follow episode_transport_assessment.released_relay_gate
and provided_placement_plan.refresh_measured_held. No native call, adapter
authentication, route, material reset or delivery verdict is provided here.
The runtime supplies a source-authenticated coupled coverage validator. Never
pretend a 100-substep interval satisfies the rigid-only single-substep policy.
"""
from __future__ import annotations

import math

from physics.carried_body import HeldBodySpec, WorldPose
from physics.episode_transport_assessment import _number, _vector, _pose, _angle, _active_arm
from physics.handle_contacts import contents_contact_witness_valid
from physics.particle_episode_state import validate_particle_packet, window_rest, REST_LIMITS, _hash, _require
from physics.provided_placement_plan import HOLDER_WINDOW_ROWS
from transport_challenge.contracts import ContractError, digest, point


def released_particle_relay_gate(recent_rows, ending_s, settings, arm_ids, rest_policy, goal,
                                 finger_bounds, original_contract, dt=.0025, window_s=.5, *,
                                 coverage_validator, active_arm_id='arm_0'):
    """Check one released particle placement with unchanged nonmaterial guards.

    Required coverage_validator(row, step, dt, contacts) returns one policy
    SHA256 or raises ContractError. The caller owns its source/native binding,
    complete copied witnesses, actual detection poses and temporal scope. This
    pure gate cannot authenticate an arbitrary callable or fabricated packet.
    No default/fallback rewrites the existing rigid-only coverage check.
    """
    from physics.episode_runtime import measured_rest, body_rest, DT, PROTOCOL
    _number(dt, 'dt', 0.); _number(window_s, 'window', 0.); _number(ending_s, 'ending time', 0.)
    if dt != DT or window_s != .5 or abs(ending_s/dt-round(ending_s/dt)) > 1e-9:
        raise ContractError('Particle relay uses the fixed actual step grid and .5-second rest window')
    ids = tuple(arm_ids); active = _active_arm(active_arm_id, ids)
    if set(finger_bounds) != set(ids):
        raise ContractError('Relay gate needs the exact named arms and native finger bounds')
    bounds = {name: tuple(_vector(pair, 2, 'finger bounds') for pair in finger_bounds[name]) for name in ids}
    if any(len(value) != 2 or any(lo > hi for lo, hi in value) for value in bounds.values()):
        raise ContractError('Each arm needs two ordered native finger bounds')
    execution = settings['execution_limits']; expected = round(window_s/dt); end = round(ending_s/dt)
    limits = {'finger_m': _number(settings['closure_match_tolerances']['finger_m'], 'finger tolerance', 0.),
              'body_position_m': _number(execution['final_body_position_error_m'], 'body position tolerance', 0.),
              'body_orientation_rad': _number(execution['final_body_orientation_error_rad'], 'body orientation tolerance', 0.)}
    target = _pose(goal['body_world']); support_id = goal['support_id']
    if type(support_id) is not str or not support_id:
        raise ContractError('Relay support must be explicitly named')
    failures = []; selected = []; policy_ids = set()
    maxima = {'body_position_m': None, 'body_orientation_rad': None, 'finger_open_error_m': None}
    rows = list(recent_rows)
    try:
        _require(callable(coverage_validator), 'Source-bound coupled coverage validator is required')
        for row in rows:
            clock = _number(row['simulation_s'], 'row clock', 0.)
            if ending_s-window_s+1e-10 < clock <= ending_s+1e-10:
                selected.append(row)
        if len(selected) != expected:
            raise ContractError('Incomplete released relay observation window')
        for row, step in zip(selected, range(end-expected+1, end+1)):
            if (type(row['step']) is not int or row['step'] != step or type(row['native_scene_step']) is not int
                    or row['native_scene_step'] != step or abs(row['simulation_s']-step*dt) > 1e-9
                    or row['post_processing_complete'] is not True or row['failures'] != []
                    or row['phase'] != 'released_settle'):
                raise ContractError('Failed, incomplete or nonconsecutive released interval')
            pre = row['pre']
            if (type(pre['native_scene_step']) is not int or pre['native_scene_step'] != step-1
                    or abs(_number(pre['simulation_s'], 'pre clock')-(step-1)*dt) > 1e-9):
                raise ContractError('Pre-step observation is not from this interval')
            contacts = row['classified_contacts']
            if not isinstance(contacts, list):
                raise ContractError('Classified contacts must retain every copied row')
            policy_ids.add(_hash(coverage_validator(row, step, dt, contacts)))
            supported = False
            for i, contact in enumerate(contacts):
                if (type(contact['raw_row']) is not int or contact['raw_row'] != i
                        or contact['geom_ids'] != [row['raw_contacts']['geom_a'][i], row['raw_contacts']['geom_b'][i]]):
                    raise ContractError('Classified witness identity/order differs')
                if contact['category'] not in ('named_tray_bottom_support_top', 'contents_contact', 'named_support_floor'):
                    raise ContractError('Unallowed or unreleased pad contact')
                if contact['category'] == 'contents_contact' and _number(contact['penetration_m'], 'penetration') < 0.:
                    valid_contact = contents_contact_witness_valid(contact['penetration_m'],
                        contact['normal_unit_error'], contact.get('owners'))
                else:
                    valid_contact = (_number(contact['penetration_m'], 'penetration', 0.) <= .001
                        and _number(contact['normal_unit_error'], 'normal error', 0.) <= .001)
                if not valid_contact:
                    raise ContractError('Existing contact accuracy guard failed')
                supported |= contact['category'] == 'named_tray_bottom_support_top' and contact.get('support_station_id') == support_id
            if not supported:
                raise ContractError('Named relay support is absent in this interval')
            for observation in (pre, row):
                if set(observation['arms']) != set(ids):
                    raise ContractError('Missing named arm observation')
                for name in ids:
                    q = _vector(observation['arms'][name]['q_all_dofs'], 9, 'measured q')
                    qd = _vector(observation['arms'][name]['qd_all_dofs'], 9, 'measured velocity')
                    if (max(map(abs, qd[:7])) > execution['rest_speed_rad_s']
                            or max(map(abs, qd[7:])) > PROTOCOL['finger_rest_speed_m_s']):
                        raise ContractError('Measured interval endpoint is not at joint/finger rest')
                    if name == active:
                        error = max(abs(q[7+i]-bounds[name][i][1]) for i in range(2))
                        maxima['finger_open_error_m'] = max(maxima['finger_open_error_m'] or 0., error)
                        if error > limits['finger_m']:
                            raise ContractError('Active fingers are not at native open bounds')
                actual = _pose(observation['body_world'])
                if (math.hypot(*_vector(observation['body_linear_velocity_m_s'], 3, 'body velocity')) > rest_policy['maximum_body_linear_speed_m_s']
                        or math.hypot(*_vector(observation['body_angular_velocity_rad_s'], 3, 'body angular velocity')) > rest_policy['maximum_body_angular_speed_rad_s']):
                    raise ContractError('Measured interval endpoint is not at body rest')
                pe = math.dist(actual.position_m, target.position_m); re = _angle(actual.quaternion_wxyz, target.quaternion_wxyz)
                maxima['body_position_m'] = max(maxima['body_position_m'] or 0., pe)
                maxima['body_orientation_rad'] = max(maxima['body_orientation_rad'] or 0., re)
                if pe > limits['body_position_m'] or re > limits['body_orientation_rad']:
                    raise ContractError('Measured body is outside the relay goal tolerance')
                packet = validate_particle_packet(observation['particle_material'], original_contract, observation, require_retained=True)
                if (packet['bulk']['relative_RMS_m_s'] > REST_LIMITS['relative_RMS_max_m_s']
                        or packet['bulk']['relative_q95_m_s'] > REST_LIMITS['relative_q95_max_m_s']):
                    raise ContractError('Particle interval endpoint is not at the fixed bulk-rest speed')
            _require(pre['particle_material']['native_clock'][2] < row['particle_material']['native_clock'][2],
                     'Native particle interval did not advance')
        if len(policy_ids) != 1:
            raise ContractError('Contact policy changed during the released window')
    except (ContractError, KeyError, TypeError, ValueError, OverflowError) as exc:
        failures.append(str(exc))
    rest = None; body = None; bulk = None
    if not failures:
        full_window = {**execution, 'rest_window_s': window_s}
        try:
            rest = measured_rest(selected, ending_s, full_window, ids)
            body = body_rest(selected, ending_s, window_s, rest_policy)
            bulk = window_rest(rows, ending_s, original_contract, dt_s=dt)
            if not rest['passed']: failures.append('All-arm joint/finger rest failed')
            if not body['passed']: failures.append('Body rest failed')
            if not bulk['passed']: failures.append('Particle bulk rest failed: '+' | '.join(bulk['failures']))
        except (ContractError, KeyError, TypeError, ValueError, OverflowError) as exc:
            failures.append(str(exc))
    passed = not failures
    return dict(schema_version='sampled-released-particle-relay-gate-v1', passed=passed,
        status='PARTICLE_RELAY_PLACEMENT_SAMPLED_PASS' if passed else 'PARTICLE_RELAY_PLACEMENT_NOT_VERIFIED',
        ending_s=ending_s, window_s=window_s, dt_s=dt, expected_rows=expected, observed_rows=len(selected),
        support_id=support_id, active_arm_id=active, limits=limits, maxima=maxima,
        all_arms_rest=rest, body_rest=body, particle_bulk_rest=bulk, failures=failures,
        coverage_policy_sha256=next(iter(policy_ids)) if len(policy_ids) == 1 else None,
        particle_material_contract_sha256=digest(original_contract),
        material_delivery_accepted=False, physical_handoff_verified=False, full_mission_passed=False,
        continuous_collision_certificate=False,
        scope='One completed sampled released particle placement. Caller separately rejects earlier episode failures and authenticates coverage/native sources.')


def refresh_particle_measured_held(world, scenario, model, observation, recent_rows, previous_held,
                                   original_contract, arm):
    """Preserve the current body/holder binding, using the full typed inventory.

    This is the original current held-frame refresh, not a new acquisition or
    a replacement of native handling guards. Caller owns actual native truth.
    """
    from physics.episode_held_phase_inputs import observed_held_spec
    if (type(previous_held) is not HeldBodySpec or previous_held.object_id != world['object_id']
            or previous_held.model.geometry_sha256 != model.geometry_sha256
            or original_contract.get('geometry_sha256') != model.geometry_sha256
            or set(observation['arms']) != {'arm_0', 'arm_1', 'arm_2'} or arm not in observation['arms']):
        raise ContractError('Measured seating requires the same currently held body and exact three arms')
    selected = list(recent_rows)[-HOLDER_WINDOW_ROWS:]; end = observation['native_scene_step']
    if (len(selected) != HOLDER_WINDOW_ROWS or selected[-1]['native_scene_step'] != end
            or abs(observation['simulation_s']-end*.0025) > 1e-9):
        raise ContractError('Current held-frame refresh requires a complete native interval window')
    for index, row in enumerate(selected, end-HOLDER_WINDOW_ROWS+1):
        interval = row.get('handling_interval', {})
        if (row.get('step') != index or row.get('native_scene_step') != index
                or abs(row['simulation_s']-index*.0025) > 1e-9 or row.get('post_processing_complete') is not True
                or row.get('failures') != [] or row.get('coverage', {}).get('contacts_complete') is not True
                or interval.get('passed') is not True or interval.get('qualified_holder_id') != arm
                or interval.get('first_failure') is not None or row.get('bilateral_by_arm', {}).get(arm) is not True):
            raise ContractError('Measured seating cannot inherit a missing, switched or unqualified holder')
        validate_particle_packet(row['particle_material'], original_contract, row, require_retained=True)
    validate_particle_packet(observation['particle_material'], original_contract, observation, require_retained=True)
    hand = observation['arms'][arm]['hand_world']; body = observation['body_world']
    if math.dist(point(hand['position_m'], 3), point(body['position_m'], 3)) > .20:
        raise ContractError('Current held root leaves the existing native grasp reach bound')
    spec = HeldBodySpec(world['object_id'], model, WorldPose(hand['position_m'], hand['quaternion_wxyz']),
        WorldPose(body['position_m'], body['quaternion_wxyz']), tuple(point(observation['arms'][arm]['q_all_dofs'], 9)[7:]),
        observation['simulation_s'], f'episode_step_{end}')
    return observed_held_spec(world, scenario, observation, spec.to_record(), active_arm_id=arm, repeated_three_arm_mode=True)
