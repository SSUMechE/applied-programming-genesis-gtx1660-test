"""Trusted automatic-grasp observations and explicit initialization overlay.

No native step, state setter, student route or material operation occurs here.
The runtime owns the helper and the actual receipt returned by its transaction.
"""
import copy
from transport_challenge.contracts import ContractError, digest

HELPER_SHA = '7458d4f339da20d91d819f981cc8e30c3b018dd5cce375e9d61f32eb55e3c291'


def require(ok, message):
    if not ok:
        raise ContractError(message)


def prepare_automatic_scenario(scenario):
    """Supply clear fingers at INITIAL, preserving every seven-joint input."""
    actual = copy.deepcopy(scenario)
    before = {r['id']:list(r['finger_q_m']) for r in actual['robots']}
    for robot in actual['robots']:
        robot['finger_q_m'] = [.04, .04]
    record = dict(schema_version='provided-ideal-initialization-v1',
        input_scenario_sha256=digest(scenario), effective_scenario_sha256=digest(actual),
        original_finger_q_m=before, effective_finger_q_m={r['id']:list(r['finger_q_m']) for r in actual['robots']},
        authored_finger_target_m=.04,native_representation='Existing native_snapshot_command performs one actual-bound-dtype conversion and exact readback. No clipping or post-INITIAL setter.',
        scope='Provided INITIAL finger coordinates only. Authored bases, relays, seven-joint states, vessel and materials are unchanged. No post-INITIAL setter.')
    return actual, record


def automatic_observation_binding(helper, driver, observation):
    holder = helper.holder
    if holder is None:
        return None
    require(holder in observation['arms'], 'Native provided holder is absent from observation')
    snapshot = helper.snapshot(holder)
    receipt = driver.automatic_attachment_receipt
    require(type(receipt) is dict and receipt.get('model')=='provided-ideal-native-weld-v1'
        and receipt.get('event')=='attach' and receipt.get('arm_id')==holder,
        'Current native attachment lacks its actual attach receipt')
    require(snapshot['native_clock'][:2]==[observation['native_scene_step'],observation['simulation_s']]
        and snapshot['body_world']==observation['body_world']
        and snapshot['hand_world']==observation['arms'][holder]['hand_world']
        and snapshot['body_linear_velocity_m_s']==observation['body_linear_velocity_m_s']
        and snapshot['body_angular_velocity_rad_s']==observation['body_angular_velocity_rad_s'],
        'Automatic binding is not this actual rigid observation')
    require(snapshot['body_entity_idx']==receipt['after']['body_entity_idx']
        and snapshot['cup_mass_kg']==receipt['after']['cup_mass_kg']
        and snapshot['inventory']==receipt['after']['inventory'],
        'Native provided mass/body/ordered holder inventory changed')
    return dict(schema_version='provided-automatic-grasp-binding-v1',model='provided_ideal',
        holder_arm_id=holder,helper_source_sha256=HELPER_SHA,
        attach_receipt=copy.deepcopy(receipt),current_snapshot=snapshot)


def physical_observation(observation):
    """Remove only the provided event annotation and post-step diagnostics."""
    result = copy.deepcopy(observation)
    result.pop('provided_grasp',None)
    for arm in result['arms'].values():
        for name in ('qdd_fd_rad_s2','transmitted_effort_diagnostic','tracking_error_rad'):
            arm.pop(name,None)
    return result
