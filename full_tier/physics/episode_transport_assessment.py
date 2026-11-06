"""Pure assessment of a supported grasp and one released relay placement.

Inputs are completed observations from the persistent episode runtime. This
module cannot authenticate a fabricated adapter record and makes no native
calls. The caller retains source, geometry, live coverage and whole-episode
failure evidence. A relay placement is neither a recipient grasp nor delivery.
"""
from __future__ import annotations

import copy
import math

from physics.carried_body import HeldBodySpec, WorldPose, measure_hand_to_body
from physics.handle_contacts import contents_contact_witness_valid
from transport_challenge.contracts import ContractError


ACTIVE_ARM_ID = 'arm_0'  # Legacy default. A caller may explicitly select a known holder.


def _number(value, label, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ContractError(label+' must be finite numeric data')
    if minimum is not None and value < minimum:
        raise ContractError(label+' is below its minimum')
    return float(value)


def _vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ContractError(label+' has the wrong dimension')
    return tuple(_number(v, label) for v in value)


def _pose(value):
    if not isinstance(value, dict) or value.get('frame', 'world') != 'world':
        raise ContractError('Expected an observed world-frame pose')
    return WorldPose(_vector(value['position_m'], 3, 'position'),
                     _vector(value['quaternion_wxyz'], 4, 'quaternion'))


def _angle(a, b):
    # Identical arithmetic to the supplied quaternion_error used by the runtime.
    qa = WorldPose((0, 0, 0), a).quaternion_wxyz
    qb = WorldPose((0, 0, 0), b).quaternion_wxyz
    return 2*math.acos(min(1., abs(sum(x*y for x, y in zip(qa, qb)))))


def _active_arm(value, arm_ids, *, single_active_contract=None):
    ids = tuple(arm_ids)
    from physics.episode_contacts import validate_single_active_contact_contract
    single = validate_single_active_contact_contract(single_active_contract, ids)
    if (type(value) is not str or not value or any(type(name) is not str or not name for name in ids)
            or (len(ids) != 1 if single is not None else len(ids) not in (2, 3))
            or len(set(ids)) != len(ids) or value not in ids):
        raise ContractError('Select one of the exact named arms')
    return value


def _grasp_targets(observation, arm_ids, scenario, active_arm_id, support_id,
                   endpoint_hand_pose, body_id, *, single_active_contract=None):
    """Resolve supplied phase identities without moving or rewriting a fixture."""
    ids = tuple(arm_ids)
    active = _active_arm(scenario['active_arm_id'] if active_arm_id is None else active_arm_id, ids,
                         single_active_contract=single_active_contract)
    declared_ids = set(ids)
    if single_active_contract is not None:
        from physics.episode_contacts import validate_single_active_contact_contract
        proof = validate_single_active_contact_contract(single_active_contract, ids)
        declared_ids.update(proof['frozen_arm_ids'])
    if (set(observation['arms']) != set(ids)
            or {r['id'] for r in scenario['robots']} != declared_ids):
        raise ContractError('Supported grasp requires the known scenario and observed arms')
    stations = [s['id'] for s in scenario['support_stations']]
    if (not stations or any(type(s) is not str or not s for s in stations)
            or len(stations) != len(set(stations))):
        raise ContractError('Require distinct named scenario supports')
    support = stations[0] if support_id is None else support_id
    if type(support) is not str or support not in stations:
        raise ContractError('Select a known scenario support')
    body = scenario['body_id'] if body_id is None else body_id
    if type(body) is not str or not body or body != scenario['body_id']:
        raise ContractError('Supported grasp body must match the known scenario body')
    endpoint = scenario['endpoint_hand_pose'] if endpoint_hand_pose is None else endpoint_hand_pose
    _pose(endpoint)  # Validate, without normalizing or replacing the supplied target.
    return active, support, endpoint, body


def supported_grasp_gate(observation, recent_rows, settings, arm_ids, scenario,
                         rest_policy, contact_rows, bilateral_since, stable_since,
                         support_since, model, *, active_arm_id=None, support_id=None,
                         endpoint_hand_pose=None, body_id=None, single_active_contract=None):
    """Assess the selected holder at a named support and supplied hand endpoint.

Omitted targets retain the source-grasp defaults. The legacy source_support_*
output keys describe the selected support. Timers must already belong to this
holder/support phase. This function does not create contacts or prove lifting.
"""
    from physics.episode_runtime import measured_rest, body_rest
    final = observation
    active, selected_support, endpoint, selected_body = _grasp_targets(
        observation, arm_ids, scenario, active_arm_id, support_id, endpoint_hand_pose, body_id,
        single_active_contract=single_active_contract)
    execution = settings['execution_limits']
    rest = measured_rest(recent_rows, final['simulation_s'], execution, arm_ids)
    final_body_rest = body_rest(recent_rows, final['simulation_s'],
                                execution['rest_window_s'], rest_policy)
    supported = any(r['category'] == 'named_tray_bottom_support_top'
                    and r.get('support_station_id') == selected_support
                    for r in contact_rows)
    position = math.dist(final['arms'][active]['hand_world']['position_m'],
                         endpoint['position_m'])
    orientation = _angle(final['arms'][active]['hand_world']['quaternion_wxyz'],
                          endpoint['quaternion_wxyz'])
    gate = {'all_arms_rest': rest, 'body_rest': final_body_rest,
            'bilateral_window_s': 0. if bilateral_since is None else final['simulation_s']-bilateral_since,
            'stable_contents_window_s': 0. if stable_since is None else final['simulation_s']-stable_since,
            'source_support_window_s': 0. if support_since is None else final['simulation_s']-support_since,
            'source_support_observed': supported, 'hand_position_error_m': position,
            'hand_orientation_error_rad': orientation, 'load_bearing_verified': False}
    gate['passed'] = bool(rest['passed'] and final_body_rest['passed'] and supported
        and gate['bilateral_window_s'] >= settings['scenario']['bilateral_window_s']-1e-9
        and gate['stable_contents_window_s'] >= settings['scenario']['contained_stability_s']-1e-9
        and gate['source_support_window_s'] >= settings['scenario']['bilateral_window_s']-1e-9
        and position <= execution['final_hand_position_error_m']
        and orientation <= execution['final_hand_orientation_error_rad'])
    h = final['arms'][active]['hand_world']; b = final['body_world']
    spec = HeldBodySpec(selected_body, model, WorldPose(h['position_m'], h['quaternion_wxyz']),
        WorldPose(b['position_m'], b['quaternion_wxyz']),
        tuple(final['arms'][active]['q_all_dofs'][7:]), final['simulation_s'],
        f"episode_step_{final['native_scene_step']}")
    return gate, spec


def _materials(values):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ContractError('Require the four persistent material identities')
    result = {}
    for item in values:
        name = item['id']
        if type(name) is not str or not name or name in result:
            raise ContractError('Material identities must be unique names')
        _number(item['mass_kg'], 'material mass', 0.)
        if item['mass_kg'] <= 0:
            raise ContractError('Material mass must be positive')
        if type(item['entity_idx']) is not int or item['entity_idx'] < 0:
            raise ContractError('Material native entity must be an exact integer')
        result[name] = item
    if len({m['entity_idx'] for m in result.values()}) != 4:
        raise ContractError('Material native entities must be unique')
    return result


BEAD_MATERIAL_REST_LIMITS = {'window_s': .5, 'relative_RMS_max_m_s': .03,
    'relative_q95_max_m_s': .1, 'COM_excursion_max_m': .0005}


def _bead_material_sample(row, supply, geometry):
    """One committed rigid-bead observation, never a particle surrogate."""
    import numpy as np
    from physics.gravity_feedforward import rotation_matrix_wxyz
    from physics.scene_assets import interior_membership
    step=row['native_scene_step'];now=_number(row['simulation_s'],'bead clock',0.)
    if (type(step) is not int or step<1 or type(row.get('step')) is not int or row['step']!=step or abs(now-step*.0025)>1e-9
            or row.get('post_processing_complete') is not True or row.get('failures')!=[]):
        raise ContractError('Bead rest requires a successful committed native endpoint')
    current=_materials(row['materials'])
    if (list(current)!=list(supply) or any((current[k]['entity_idx'],current[k]['mass_kg'])
            !=(supply[k]['entity_idx'],supply[k]['mass_kg']) or current[k].get('inside_carrier') is not True for k in supply)):
        raise ContractError('Original bead IDs, entities, mass or containment changed')
    pose=row['body_world']
    if type(pose) is not dict or pose.get('frame')!='world':raise ContractError('Actual world body pose required')
    _pose(pose)
    p=np.asarray(_vector(pose['position_m'],3,'body position'))
    q=_vector(pose['quaternion_wxyz'],4,'body quaternion')
    v=np.asarray(_vector(row['body_linear_velocity_m_s'],3,'body velocity'))
    w=np.asarray(_vector(row['body_angular_velocity_rad_s'],3,'body angular velocity'))
    x=np.asarray([_vector(current[k]['position_world_m'],3,'bead position') for k in supply])
    u=np.asarray([_vector(current[k]['velocity_world_m_s'],3,'bead velocity') for k in supply])
    mass=np.asarray([supply[k]['mass_kg'] for k in supply])
    with np.errstate(over='ignore',invalid='ignore'):
        local=(x-p)@rotation_matrix_wxyz(q)
        relative=u-v-np.cross(w,x-p);speed=np.linalg.norm(relative,axis=1)
        com=np.sum(local*mass[:,None],axis=0)/mass.sum()
        rms=float(np.sqrt(np.sum(speed**2*mass)/mass.sum()));q95=float(np.quantile(speed,.95))
    if (not np.isfinite(speed).all() or not np.isfinite(com).all()
            or not math.isfinite(rms) or not math.isfinite(q95)
            or not interior_membership(local,geometry,.012,.001).all()):
        raise ContractError('Bead rest has invalid data or actual geometric containment')
    return dict(relative_RMS_m_s=rms,relative_q95_m_s=q95,COM_body_m=com.tolist())


def bead_material_rest_gate(recent_rows, ending_s, initial_supply, model):
    """Fixed201-endpoint bulk rest. A failed quality check is never waitable."""
    import numpy as np
    from physics.carried_body import VesselModel
    selected=[];metrics=None;quality_failures=[];settling_failures=[]
    try:
        _number(ending_s,'bead rest ending time',0.)
        end=round(ending_s/.0025)
        if abs(ending_s-end*.0025)>1e-9 or end<200 or type(model) is not VesselModel:
            raise ContractError('Bead rest needs the fixed actual clock and vessel model')
        supply=_materials(initial_supply);geometry=model.geometry_record()
        for row in recent_rows:
            now=_number(row['simulation_s'],'row clock',0.)
            if ending_s-.5-1e-10<=now<=ending_s+1e-10:selected.append(row)
        if len(selected)!=201:raise ContractError('Incomplete201-endpoint bead rest window')
        values=[]
        for row,step in zip(selected,range(end-200,end+1)):
            if row.get('native_scene_step')!=step:raise ContractError('Nonconsecutive bead rest endpoints')
            values.append(_bead_material_sample(row,supply,geometry))
        com=np.asarray([v['COM_body_m'] for v in values])
        metrics=dict(relative_RMS_max_m_s=max(v['relative_RMS_m_s'] for v in values),
            relative_q95_max_m_s=max(v['relative_q95_m_s'] for v in values),
            COM_excursion_max_m=float(np.linalg.norm(com-com[0],axis=1).max()))
        for key,value in metrics.items():
            _number(value,key,0.)
            if value>BEAD_MATERIAL_REST_LIMITS[key]:settling_failures.append(key)
    except (ContractError,KeyError,TypeError,ValueError,OverflowError) as exc:
        metrics=None;quality_failures.append(str(exc))
    return dict(schema_version='bead-material-window-rest-v1',passed=not quality_failures and not settling_failures,
        quality_passed=not quality_failures,ending_s=ending_s,window_s=.5,dt_s=.0025,
        expected_rows=201,observed_rows=len(selected),limits=dict(BEAD_MATERIAL_REST_LIMITS),metrics=metrics,
        quality_failures=quality_failures,settling_failures=settling_failures,
        source_authentication='Caller owns actual native observations, source bindings and every physical interval.')


def reference_agreement(observation, reference_final, settings, *, active_arm_id=ACTIVE_ARM_ID,
                        single_active_contract=None):
    """Compare fresh observations without moving a pose or widening a limit.

All arms' seven q, two fingers and hand positions are checked. The measured
hand/body rotation belongs to the explicitly selected holder. Other rotations and
material positions/velocities are diagnostics, without new tolerance rules.
"""
    actual_arms = observation['arms']; expected_arms = reference_final['arms']
    active = _active_arm(active_arm_id, actual_arms, single_active_contract=single_active_contract)
    if set(actual_arms) != set(expected_arms):
        raise ContractError('Fresh reference requires the same named arms and holder')
    tolerances = settings['closure_match_tolerances']
    limits = {k: _number(tolerances[k], k, 0.) for k in
              ('joint_rad', 'finger_m', 'hand_position_m', 'body_position_m', 'relative_orientation_rad')}
    differences = {}; failed = []
    for name in actual_arms:
        a = actual_arms[name]; b = expected_arms[name]
        aq = _vector(a['q_all_dofs'], 9, 'actual q'); bq = _vector(b['q_all_dofs'], 9, 'reference q')
        ap = _pose(a['hand_world']); bp = _pose(b['hand_world'])
        errors = {'joint_rad': max(abs(x-y) for x, y in zip(aq[:7], bq[:7])),
                  'finger_m': max(abs(x-y) for x, y in zip(aq[7:], bq[7:])),
                  'hand_position_m': math.dist(ap.position_m, bp.position_m),
                  'hand_orientation_rad_diagnostic': _angle(ap.quaternion_wxyz, bp.quaternion_wxyz)}
        differences[name] = errors
        failed.extend(name+'.'+k for k in ('joint_rad', 'finger_m', 'hand_position_m') if errors[k] > limits[k])
    ab = _pose(observation['body_world']); bb = _pose(reference_final['body_world'])
    ah = _pose(actual_arms[active]['hand_world']); bh = _pose(expected_arms[active]['hand_world'])
    ar = measure_hand_to_body(ah, ab); br = measure_hand_to_body(bh, bb)
    body = {'body_position_m': math.dist(ab.position_m, bb.position_m),
            'body_orientation_rad_diagnostic': _angle(ab.quaternion_wxyz, bb.quaternion_wxyz),
            'relative_position_m_diagnostic': math.dist(ar.translation_hand_m, br.translation_hand_m),
            'relative_orientation_rad': _angle(ar.quaternion_hand_body_wxyz, br.quaternion_hand_body_wxyz)}
    failed.extend(k for k in ('body_position_m', 'relative_orientation_rad') if body[k] > limits[k])
    actual_materials = _materials(observation['materials']); expected_materials = _materials(reference_final['materials'])
    if set(actual_materials) != set(expected_materials):
        raise ContractError('Fresh reference material identities changed')
    material_differences = {}
    for name, a in actual_materials.items():
        b = expected_materials[name]
        if (a['entity_idx'], a['mass_kg']) != (b['entity_idx'], b['mass_kg']):
            raise ContractError('Fresh reference material entity/mass changed')
        material_differences[name] = {
            'position_difference_m': math.dist(_vector(a['position_world_m'], 3, 'material position'),
                                               _vector(b['position_world_m'], 3, 'reference material position')),
            'velocity_difference_m_s': math.dist(_vector(a['velocity_world_m_s'], 3, 'material velocity'),
                                                  _vector(b['velocity_world_m_s'], 3, 'reference material velocity'))}
    return {'passed': not failed, 'failed': failed, 'arms': differences, 'body': body,
            'limits': limits, 'holder_arm_id': active, 'material_differences': material_differences,
            'material_position_or_velocity_tolerance_applied': False, 'pose_reset_permitted': False,
            'scope': f'Fresh observed agreement for the explicit {active}-holder prototype. No pose correction or new material-position tolerance.'}


def rigid16_interval_facts(samples, *, step, dt=.0025, coverage_policy_sha256=None):
    """Validate all actual16 detection/post captures and expose their h timeline.

    Unsupported here means no named bottom/support witness in that subinterval.
    It does not silently turn an automatic or bilateral holder into support.
    The runtime combines the returned actual holder/contact facts with its
    existing phase rules. No planner command or outer endpoint is interpolated.
    """
    from physics.live_contact_coverage import (RIGID16_PROFILE, RIGID16_H,
                                               validate_rigid16_policy_stamp)
    if (type(step) is not int or step < 1 or dt != .0025
            or type(samples) is not list or len(samples) != 16):
        raise ContractError('Rigid16 evidence needs the named exact16 actual captures')
    timeline = []; previous = None; identity = coverage_policy_sha256
    unsupported_run = maximum_unsupported_run = 0.
    for index, sample in enumerate(samples):
        g = (step-1)*16+index; start = g*RIGID16_H; end = (g+1)*RIGID16_H
        detection = sample['detection']; raw = sample['contacts']; post = sample['post_integration_state']
        pre = detection['rigid_state']; clock = [g,0,step-1,start]
        if (any(type(sample[k]) is not int for k in ('global_substep_ordinal','outer_step_ordinal','internal_substep_index','local_state_index','produced_post_global_substep','contact_count'))
                or any(type(v) is not int for v in detection['preintegration_native_clock'][:3])
                or any(type(v) is not int for v in sample['post_read_native_clock'][:3])
                or type(detection['preintegration_native_clock'][3]) not in (int,float)
                or type(sample['post_read_native_clock'][3]) not in (int,float)
                or sample['global_substep_ordinal'] != g or sample['outer_step_ordinal'] != step-1
                or sample['internal_substep_index'] != index or sample['local_state_index'] != 0
                or sample['interval_start_s'] != start or sample['interval_end_s'] != end
                or detection.get('schema_version') != 'rigid16-detection-snapshot-v1'
                or detection.get('profile_name') != RIGID16_PROFILE
                or detection['preintegration_native_clock'] != clock
                or sample['post_read_native_clock'] != clock
                or sample['produced_post_global_substep'] != g+1
                or detection['readback_did_not_change_clock'] is not True
                or detection['raw_contacts'] != raw):
            raise ContractError('Rigid16 actual detection g/post g+1 clocks or raw identity differ')
        for state in (pre, post):
            _pose(state['body_world'])
            if type(state['arms']) is not dict or not state['arms']:
                raise ContractError('Rigid16 actual named hand/finger states missing')
            for arm in state['arms'].values():
                _pose(arm['hand_world'])
                if set(arm['finger_world']) != {'left','right'}:
                    raise ContractError('Rigid16 both actual finger poses required')
                for finger in arm['finger_world'].values(): _pose(finger)
        if previous is not None and (pre['body_world'] != previous['body_world'] or pre['arms'] != previous['arms']):
            raise ContractError('Rigid16 detection state is not the preceding actual produced state')
        if set(pre['arms']) != set(post['arms']):
            raise ContractError('Rigid16 named arm inventory changed')
        classified = sample['classified_contacts']; count = len(raw['geom_a'])
        if (type(classified) is not list or len(classified) != count
                or any(len(raw[k]) != count for k in ('geom_b','link_a','link_b','position','normal','force','penetration'))
                or type(detection['native_contact_count']) is not int or detection['native_contact_count'] != count
                or sample['contact_count'] != count or type(detection['native_errno']) is not int or detection['native_errno'] != 0
                or type(detection['native_capacity']) is not int or not 0 <= count <= detection['native_capacity']):
            raise ContractError('Rigid16 native/copied/classified count/capacity/errno mismatch')
        stamp = sample['policy_stamp']
        observed_identity = validate_rigid16_policy_stamp(stamp, global_substep=g,
            post_state=post, coverage_policy_sha256=identity)
        if stamp['native_contact_count'] != count or stamp['native_contact_capacity'] != detection['native_capacity']:
            raise ContractError('Rigid16 live stamp and actual detection buffer differ')
        identity = observed_identity
        support_ids = set(); allowed = True
        for i, contact in enumerate(classified):
            if (type(contact['raw_row']) is not int or contact['raw_row'] != i
                    or contact['geom_ids'] != [raw['geom_a'][i],raw['geom_b'][i]]
                    or contact['penetration_m'] != raw['penetration'][i]):
                raise ContractError('Rigid16 classification dropped or reordered an actual contact')
            for key in ('position','normal','force'): _vector(raw[key][i],3,'actual raw '+key)
            depth = _number(contact['penetration_m'],'penetration')
            error = _number(contact['normal_unit_error'],'normal error',0.)
            actual_normal_error = abs(math.hypot(*raw['normal'][i])-1.)
            if actual_normal_error > .001 or abs(actual_normal_error-error) > 1e-12:
                raise ContractError('Rigid16 classified normal error differs from the actual native vector')
            valid = (contents_contact_witness_valid(depth,error,contact.get('owners'))
                     if depth < 0 and contact['category']=='contents_contact' else 0 <= depth <= .001 and error <= .001)
            if not valid: raise ContractError('Rigid16 unchanged actual contact accuracy guard failed')
            allowed &= contact['category'] != 'UNALLOWED_CONTACT'
            if contact['category'] == 'named_tray_bottom_support_top':
                name = contact.get('support_station_id')
                if type(name) is not str or not name: raise ContractError('Actual support must be explicitly named')
                support_ids.add(name)
        bilateral = sample['bilateral_by_arm']
        if type(bilateral) is not dict or set(bilateral) != set(pre['arms']) or any(type(v) is not bool for v in bilateral.values()):
            raise ContractError('Rigid16 actual bilateral result needs every named arm')
        holder = stamp.get('automatic_grasp',{}).get('holder_arm_id')
        if not support_ids:
            unsupported_run += RIGID16_H; maximum_unsupported_run = max(maximum_unsupported_run,unsupported_run)
        else: unsupported_run = 0.
        timeline.append(dict(global_substep=g,start_s=start,end_s=end,dt_s=RIGID16_H,
            support_ids=sorted(support_ids),bilateral_by_arm=copy.deepcopy(bilateral),
            automatic_holder_arm_id=holder,contacts_allowed=bool(allowed)))
        previous = post
    return dict(schema_version='rigid16-interval-facts-v1',runtime_profile=RIGID16_PROFILE,
        step=step,substeps=16,dt_s=dt,internal_dt_s=RIGID16_H,coverage_policy_sha256=identity,
        timeline=timeline,supported_s=sum(t['dt_s'] for t in timeline if t['support_ids']),
        unsupported_s=sum(t['dt_s'] for t in timeline if not t['support_ids']),
        max_consecutive_unsupported_s=maximum_unsupported_run,all_contacts_allowed=all(t['contacts_allowed'] for t in timeline),
        complete=True,continuous_time_collision_proof=False)


def _coverage(row, step, dt, contacts, *, required_support_id=None):
    if row.get('rigid_profile') == 'rigid16_bead_tc2h_v1':
        if (type(required_support_id) is not str or not required_support_id
                or required_support_id.strip() != required_support_id):
            raise ContractError('Rigid16 released coverage requires the actual selected goal support')
        from physics.live_contact_coverage import validate_automatic_grasp_stamp
        samples = row['rigid_substep_samples']; coverage = row['coverage']; stamp = coverage['policy_stamp']
        facts = rigid16_interval_facts(samples,step=step,dt=dt)
        last = samples[-1]
        if (contacts != last['classified_contacts'] or row['raw_contacts'] != last['contacts']
                or type(coverage['step']) is not int or coverage['step'] != step or coverage['contacts_complete'] is not True
                or abs(coverage['pre_time_s']-(step-1)*dt)>1e-9 or abs(coverage['post_time_s']-step*dt)>1e-9
                or any(type(coverage[k]) is not int or coverage[k] != len(contacts) for k in ('native_contact_count','copied_contact_count','classified_contact_count'))
                or stamp.get('runtime_profile') != 'rigid16_bead_tc2h_v1'
                or stamp.get('single_substep') is not False or stamp.get('substeps') != 16
                or stamp.get('coverage_policy_sha256') != facts['coverage_policy_sha256']
                or stamp.get('policy_valid_now') is not True or stamp.get('failures') != []
                or stamp.get('pruning_disabled') is not True or stamp.get('native_errno') != 0):
            raise ContractError('Rigid16 outer row is not its actual complete last-internal capture')
        if stamp.get('schema_version') == 'live-contact-policy-stamp-rigid16-automatic-grasp-outer-v1':
            validate_automatic_grasp_stamp(stamp,step=step,dt=dt,substeps=16)
        elif stamp.get('schema_version') != 'live-contact-policy-stamp-rigid16-outer-v1':
            raise ContractError('Rigid16 requires the explicit outer schema, not a legacy stamp')
        # Existing automatic released-window callers still iterate the final
        # outer contact rows. Enforce their unchanged category/support rule on
        # every internal capture here, before those callers can accept it.
        release_categories = ('named_tray_bottom_support_top','contents_contact','named_support_floor')
        if any(c['category'] not in release_categories for sample in samples for c in sample['classified_contacts']):
            raise ContractError('Unreleased contact exists in an actual rigid16 released interval')
        if any(required_support_id not in t['support_ids'] for t in facts['timeline']):
            raise ContractError('Selected goal support is absent in an actual rigid16 subinterval')
        return facts['coverage_policy_sha256']
    coverage = row['coverage']; stamp = coverage['policy_stamp']; raw = row['raw_contacts']
    count = len(contacts)
    for key in ('native_contact_count', 'copied_contact_count', 'classified_contact_count'):
        if type(coverage[key]) is not int or coverage[key] != count:
            raise ContractError('Native/copied/classified contact counts differ')
    if any(len(raw[k]) != count for k in ('geom_a', 'geom_b', 'position', 'normal', 'penetration')):
        raise ContractError('Copied native witness arrays are incomplete')
    if (type(coverage['step']) is not int or coverage['step'] != step
            or abs(_number(coverage['pre_time_s'], 'contact pre time')-(step-1)*dt) > 1e-9
            or abs(_number(coverage['post_time_s'], 'contact post time')-step*dt) > 1e-9
            or coverage['contacts_complete'] is not True
            or any(stamp[k] is not True for k in ('policy_valid_now', 'pruning_disabled', 'single_substep'))
            or stamp['failures'] != []):
        raise ContractError('Incomplete same-interval contact coverage')
    if stamp.get('schema_version')=='live-contact-policy-stamp-automatic-grasp-v1':
        from physics.live_contact_coverage import validate_automatic_grasp_stamp
        validate_automatic_grasp_stamp(stamp,step=step,dt=dt,substeps=1)
    identity = stamp['coverage_policy_sha256']
    if type(identity) is not str or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
        raise ContractError('Missing exact coverage policy identity')
    if stamp.get('native_contact_count') is not None and (type(stamp['native_contact_count']) is not int
            or stamp['native_contact_count'] != count):
        raise ContractError('Stamp/native contact count differs')
    return identity


def released_relay_gate(recent_rows, ending_s, settings, arm_ids, rest_policy, goal,
                        finger_bounds, initial_supply, dt=.0025, window_s=.5, *,
                        active_arm_id=ACTIVE_ARM_ID, single_active_contract=None):
    """Assess one sampled released relay window, with no whole-mission verdict.

The current episode uses dt=.0025 and inherited rest helpers with that clock.
Every pre/post state and every copied contact interval must pass. A future dt
needs an explicit runtime/helper change rather than silently changing counts.
"""
    from physics.episode_runtime import measured_rest, body_rest, DT, PROTOCOL
    _number(dt, 'dt', 0.); _number(window_s, 'window', 0.); _number(ending_s, 'ending time', 0.)
    if dt != DT or window_s <= 0 or abs(window_s/dt-round(window_s/dt)) > 1e-9 or abs(ending_s/dt-round(ending_s/dt)) > 1e-9:
        raise ContractError('Relay window must use the actual supported integral step grid')
    ids = tuple(arm_ids)
    active = _active_arm(active_arm_id, ids, single_active_contract=single_active_contract)
    if set(finger_bounds) != set(ids):
        raise ContractError('Relay gate needs the exact named arms and native finger bounds')
    bounds = {name: tuple(_vector(pair, 2, 'finger bounds') for pair in finger_bounds[name]) for name in ids}
    if any(len(value) != 2 or any(lo > hi for lo, hi in value) for value in bounds.values()):
        raise ContractError('Each arm needs two ordered native finger bounds')
    supply = _materials(initial_supply)
    execution = settings['execution_limits']; expected = round(window_s/dt); end = round(ending_s/dt)
    limits = {'finger_m': _number(settings['closure_match_tolerances']['finger_m'], 'finger tolerance', 0.),
              'content_speed_m_s': _number(settings['scenario']['stable_speed_m_s'], 'content speed', 0.),
              'body_position_m': _number(execution['final_body_position_error_m'], 'body position tolerance', 0.),
              'body_orientation_rad': _number(execution['final_body_orientation_error_rad'], 'body orientation tolerance', 0.)}
    target = _pose(goal['body_world']); support_id = goal['support_id']
    if type(support_id) is not str or not support_id:
        raise ContractError('Relay support must be explicitly named')
    failures = []; selected = []; policy_ids = set(); maxima = {'body_position_m': None, 'body_orientation_rad': None,
                                                             'content_speed_m_s': None, 'finger_open_error_m': None}
    try:
        for row in recent_rows:
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
            policy_ids.add(_coverage(row, step, dt, contacts, required_support_id=support_id))
            supported = False
            contact_groups = [(contacts,row['raw_contacts'])]
            if row.get('rigid_profile') == 'rigid16_bead_tc2h_v1':
                facts = rigid16_interval_facts(row['rigid_substep_samples'],step=step,dt=dt)
                if any(support_id not in t['support_ids'] for t in facts['timeline']):
                    raise ContractError('Named relay support is absent in an actual rigid16 subinterval')
                contact_groups = [(sample['classified_contacts'],sample['contacts']) for sample in row['rigid_substep_samples']]
            for actual_contacts, actual_raw in contact_groups:
                for i, contact in enumerate(actual_contacts):
                    if (type(contact['raw_row']) is not int or contact['raw_row'] != i
                            or contact['geom_ids'] != [actual_raw['geom_a'][i], actual_raw['geom_b'][i]]):
                        raise ContractError('Classified witness identity/order differs')
                    if contact['category'] not in ('named_tray_bottom_support_top', 'contents_contact', 'named_support_floor'):
                        raise ContractError('Unallowed or unreleased pad contact')
                    if (contact['category'] == 'contents_contact'
                            and _number(contact['penetration_m'], 'penetration') < 0.):
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
                contents = _materials(observation['materials'])
                if set(contents) != set(supply):
                    raise ContractError('Persistent material IDs changed')
                for name, material in contents.items():
                    initial = supply[name]
                    if (material['mass_kg'], material['entity_idx']) != (initial['mass_kg'], initial['entity_idx']):
                        raise ContractError('Persistent material mass/entity changed')
                    _vector(material['position_world_m'], 3, 'material position')
                    speed = math.hypot(*_vector(material['velocity_world_m_s'], 3, 'material velocity'))
                    maxima['content_speed_m_s'] = max(maxima['content_speed_m_s'] or 0., speed)
                    if material['inside_carrier'] is not True or speed > limits['content_speed_m_s']:
                        raise ContractError('Material is outside or not stable')
        if len(policy_ids) != 1:
            raise ContractError('Contact policy changed during the released window')
    except (ContractError, KeyError, TypeError, ValueError, OverflowError) as exc:
        failures.append(str(exc))
    rest = None; body = None
    if not failures:
        full_window = {**execution, 'rest_window_s': window_s}
        try:
            rest = measured_rest(selected, ending_s, full_window, ids)
            body = body_rest(selected, ending_s, window_s, rest_policy)
            if not rest['passed']: failures.append('All-arm joint/finger rest failed')
            if not body['passed']: failures.append('Body rest failed')
        except (ContractError, KeyError, TypeError, ValueError, OverflowError) as exc:
            failures.append(str(exc))
    passed = not failures
    return {'schema_version': 'sampled-released-relay-gate-v1', 'passed': passed,
            'status': 'RELAY_PLACEMENT_SAMPLED_PASS' if passed else 'RELAY_PLACEMENT_NOT_VERIFIED',
            'ending_s': ending_s, 'window_s': window_s, 'dt_s': dt,
            'expected_rows': expected, 'observed_rows': len(selected), 'support_id': support_id,
            'active_arm_id': active, 'limits': limits, 'maxima': maxima,
            'all_arms_rest': rest, 'body_rest': body, 'failures': failures,
            'coverage_policy_sha256': next(iter(policy_ids)) if len(policy_ids) == 1 else None,
            'persistent_material_ids': sorted(supply), 'material_delivery_accepted': False,
            'physical_handoff_verified': False, 'full_mission_passed': False,
            'continuous_collision_certificate': False,
            'scope': 'One released relay placement from completed sampled intervals. The runtime must separately reject any earlier episode failure.'}
