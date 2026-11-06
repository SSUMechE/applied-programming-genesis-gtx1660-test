"""Explicit instructor-authorized robot INITIAL reset after supported release.

This narrow operation changes one completed donor's nine joint coordinates and
joint velocity/control targets. It never writes a payload, bead, base or clock.
It is a zero-time reset, not a physical trajectory or swept collision proof.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import math
from pathlib import Path

from transport_challenge.contracts import ContractError, digest

SCHEMA = 'provided-instant-robot-reset-v1'
MODEL = 'provided-post-release-robot-initial-reset-v1'
ENTITY_SOURCE_SHA256 = 'b0f86f9c9ebdf62b9df2e5aadc0719c28e06cd52e0ce1bef452b1a269d9b2092'


def require(condition, message):
    if not condition:
        raise ContractError(message)


def _vector(value, size):
    import numpy as np
    result = np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value).copy()
    require(result.shape == (size,) and result.dtype.kind == 'f' and np.isfinite(result).all(),
            'Reset requires an owned finite native vector')
    return result.tolist()


def _pose(entity):
    return {'position_m': _vector(entity.get_pos(relative=False), 3),
            'quaternion_wxyz': _vector(entity.get_quat(relative=False), 4)}


def _entity(entity):
    import numpy as np
    value = entity.get_mass()
    mass = np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value).copy()
    require(mass.dtype.kind == 'f' and np.isfinite(mass).all() and float(mass.sum()) > 0.,
            'Reset payload native mass is invalid')
    state=dict(entity_idx=int(entity.idx), mass_kg=float(mass.sum()), qpos=_vector(entity.get_qpos(), 7),
        qd=_vector(entity.get_dofs_velocity(), 6), **_pose(entity),
        linear_velocity_m_s=_vector(entity.get_vel(), 3), angular_velocity_rad_s=_vector(entity.get_ang(), 3))
    # Preserve float32 signed-zero and quaternion bits as well as public values.
    state['native_state_bits_sha256']=hashlib.sha256(b''.join(
        np.asarray(state[key],dtype=np.float32).tobytes() for key in
        ('qpos','qd','position_m','quaternion_wxyz','linear_velocity_m_s','angular_velocity_rad_s'))+mass.tobytes()).hexdigest()
    return state


def _targets(scene, arms):
    import numpy as np
    values = {key: np.asarray(getattr(scene.rigid_solver.dyn_state.dofs, key).to_numpy()).copy()
              for key in ('ctrl_pos', 'ctrl_vel', 'ctrl_mode')}
    require(all(v.ndim == 2 and v.shape[1] == 1 for v in values.values()),
            'Reset native control layout differs from source-bound DofsState')
    return {name: {key: value[arm.dof_start:arm.dof_start+9, 0].tolist() for key, value in values.items()}
            for name, arm in arms.items()}


def _snapshot(scene, arms, cup, beads, identities, helper, arm_id):
    from physics.provided_automatic_grasp import _clock
    clock = _clock(scene)
    result = dict(native_clock=clock,
        arms={name: dict(entity_idx=int(arm.idx), base_world=_pose(arm),
            q_all_dofs=_vector(arm.get_qpos(), 9), qd_all_dofs=_vector(arm.get_dofs_velocity(), 9))
            for name, arm in arms.items()},
        payload=dict(cup=_entity(cup), beads=[dict(id=identity['id'], **_entity(bead))
            for bead, identity in zip(beads, identities)]),
        provided_snapshot=helper.snapshot(arm_id), control_targets=_targets(scene, arms))
    require(clock == _clock(scene), 'Native clock changed during reset snapshot')
    return result


def validate_robot_reset_receipt(receipt, *, before_observation=None, after_observation=None,
                                 expected_command=None, expected_arm_id=None, expected_stage_id=None):
    """Validate a reset transcript. Native authentication remains runtime-owned."""
    require(type(receipt) is dict and receipt.get('schema_version') == SCHEMA
        and receipt.get('model') == MODEL and receipt.get('passed') is True,
        'Explicit supplied robot-reset receipt required')
    arm = receipt.get('arm_id')
    require(arm in ('arm_0', 'arm_1') and receipt.get('after_stage_id') == arm+'__carry',
            'Only a completed nonfinal donor may reset')
    if expected_arm_id is not None:
        require(arm == expected_arm_id, 'Reset donor differs from expected arm')
    if expected_stage_id is not None:
        require(receipt['after_stage_id'] == expected_stage_id, 'Reset carry identity differs')
    command = receipt.get('target_command')
    require(type(command) is dict and set(command) == {'q_rad', 'finger_q_m'}
        and len(command['q_rad']) == 7 and len(command['finger_q_m']) == 2,
        'Reset needs the declared seven-joint/two-finger INITIAL command')
    target = command['q_rad']+command['finger_q_m']
    require(all(type(v) in (int, float) and math.isfinite(v) for v in target), 'Invalid reset INITIAL command')
    if expected_command is not None:
        require(command == expected_command, 'Reset command differs from immutable INITIAL command')
    before, after = receipt.get('native_before'), receipt.get('native_after')
    require(type(before) is dict and type(after) is dict, 'Actual reset snapshots required')
    try: digest(receipt)  # Reject every nonfinite nested native value.
    except (ValueError, TypeError, OverflowError) as exc:
        raise ContractError('Reset receipt contains nonfinite or malformed native values') from exc
    for snapshot in (before, after):
        require(set(snapshot) == {'native_clock','arms','payload','provided_snapshot','control_targets'}
            and type(snapshot['payload']) is dict and set(snapshot['payload']) == {'cup','beads'}
            and type(snapshot['payload']['cup']) is dict and len(snapshot['payload'].get('beads', [])) == 4
            and {r['id'] for r in snapshot['payload']['beads']} == {'tray_0:0','tray_0:1','tray_0:2','tray_0:3'},
            'Reset requires complete native cup and four-bead snapshots')
    step, seconds = receipt.get('native_scene_step'), receipt.get('simulation_s')
    clock = [step, seconds, step*16, 0] if type(step) is int else None
    require(type(step) is int and step > 0 and type(seconds) in (int, float)
        and math.isfinite(seconds) and abs(seconds-step*.0025) < 1e-9
        and before.get('native_clock') == after.get('native_clock') == clock
        and receipt.get('physical_steps') == 0 and receipt.get('simulation_duration_s') == 0.,
        'Robot reset must retain the exact completed rigid16 clock')
    require(before.get('payload') == after.get('payload'), 'Reset changed the vessel or a bead pose, velocity, mass or identity')
    require(set(before.get('arms', {})) == set(after.get('arms', {})) == {'arm_0', 'arm_1', 'arm_2'},
            'Reset lacks exact native robot identities')
    for name, old in before['arms'].items():
        new = after['arms'][name]
        if name != arm:
            require(old == new, 'Reset changed another robot')
        else:
            require(old['entity_idx'] == new['entity_idx'] and old['base_world'] == new['base_world']
                and new['q_all_dofs'] == target and new['qd_all_dofs'] == [0.]*9,
                'Reset must set only its own native q9/qd0 and preserve its base')
    for snapshot in (before, after):
        provided = snapshot.get('provided_snapshot', {})
        require(provided.get('native_clock') == clock and provided.get('inventory', {}).get('weld_pairs') == []
            and provided.get('body_entity_idx') == snapshot['payload']['cup']['entity_idx'],
            'Reset requires fresh native proof that the cup is detached')
    require(before['provided_snapshot']['inventory'] == after['provided_snapshot']['inventory'],
            'Reset changed native equality membership')
    for name, old in before['control_targets'].items():
        new = after['control_targets'][name]
        require(new == old if name != arm else new == {'ctrl_pos': target, 'ctrl_vel': [0.]*9, 'ctrl_mode': [0]*9},
                'Reset control targets differ or another robot target changed')
    require(receipt.get('source_bindings_verified') is True
        and receipt.get('native_contact_check', {}).get('passed') is True
        and receipt['native_contact_check'].get('fresh_detection') is True
        and receipt['native_contact_check'].get('unallowed_contact_count') == 0
        and receipt.get('controller_reset', {}).get('passed') is True
        and receipt.get('pre_reset_rest', {}).get('arms', {}).get('passed') is True
        and receipt.get('pre_reset_rest', {}).get('body', {}).get('passed') is True,
        'Reset lacks source, fresh endpoint-contact or controller proof')
    for key in ('before_observation_sha256', 'after_observation_sha256', 'released_gate_sha256', 'release_event_sha256'):
        checksum = receipt.get(key)
        require(type(checksum) is str and len(checksum) == 64 and all(c in '0123456789abcdef' for c in checksum),
                'Reset lacks an exact evidence hash: '+key)
    if before_observation is not None:
        require(digest(before_observation) == receipt['before_observation_sha256'], 'Reset before-observation binding differs')
    if after_observation is not None:
        require(digest(after_observation) == receipt['after_observation_sha256'], 'Reset after-observation binding differs')
    return True


def perform_released_robot_reset(*, scene, arms, cup, beads, identities, helper, arm_id,
        after_stage_id, target_command, initial_observation, before_observation, observe,
        released_gate, release_event, pre_reset_rest, source_check, retarget_control, check_contacts, record_setter=None,
        set_pose=True):
    """Trusted native operation. The runtime authenticates its released gate."""
    import numpy as np
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity
    from physics.provided_automatic_grasp import ProvidedAutomaticGrasp, _clock
    from physics.automatic_runtime_binding import physical_observation
    require(type(helper) is ProvidedAutomaticGrasp and helper.scene is scene and helper.cup is cup
        and helper.holder is None and set(arms) == set(helper.arms)
        and all(arms[n] is helper.arms[n] for n in arms), 'Reset requires the exact detached native provided-grasp scene')
    require(arm_id in ('arm_0', 'arm_1') and after_stage_id == arm_id+'__carry'
        and type(released_gate) is dict and released_gate.get('passed') is True,
        'Reset requires its actual completed released-support gate')
    require(len(beads) == len(identities) == 4 and len({r['id'] for r in identities}) == 4
        and all(type(bead) is RigidEntity and bead.solver is scene.rigid_solver and bead.n_qs == 7 and bead.n_dofs == 6
            and int(bead.idx) == identity['entity_idx'] for bead, identity in zip(beads, identities)),
        'Reset is scoped to the original four native bead identities')
    arm = arms[arm_id]
    require(type(arm) is RigidEntity and arm.solver is scene.rigid_solver and arm.n_qs == arm.n_dofs == 9
        and arm.base_link.is_fixed is True, 'Reset donor must be the original fixed-base native nine-DOF robot')
    source = Path(inspect.getsourcefile(RigidEntity)).resolve()
    require(hashlib.sha256(source.read_bytes()).hexdigest() == ENTITY_SOURCE_SHA256,
            'Unreviewed installed robot setter implementation')
    require(source_check() is True, 'Sources changed before native robot reset')
    scene.rigid_solver.check_errno()
    current = observe()
    require(physical_observation(current) == physical_observation(before_observation),
            'Reset starts at a stale physical observation')
    target = target_command['q_rad']+target_command['finger_q_m']
    require(target == initial_observation['arms'][arm_id]['q_all_dofs'], 'Reset target must be exact native INITIAL q9')
    before = _snapshot(scene, arms, cup, beads, identities, helper, arm_id)
    require(release_event.get('model') == 'provided-ideal-native-weld-v1'
        and release_event.get('event') == 'release' and release_event.get('arm_id') == arm_id
        and release_event.get('support_ready', {}).get('passed') is True
        and release_event.get('after', {}).get('inventory') == before['provided_snapshot']['inventory'],
        'Reset lacks its authentic native supported release and current detached inventory')
    # The pinned entity-local setter clears collision/constraint caches. It does
    # not integrate time. Fresh endpoint collision detection is required below.
    require(type(set_pose) is bool, 'Explicit reset or already-at-home endpoint verification required')
    if set_pose:
        if record_setter is not None:record_setter('attempt')
        arm.set_qpos(np.asarray(target, dtype=np.float32), zero_velocity=True)
        if record_setter is not None:record_setter('executed')
        arm.set_dofs_velocity(np.zeros(9, dtype=np.float32))
    else:
        require(before['arms'][arm_id]['q_all_dofs']==target and before['arms'][arm_id]['qd_all_dofs']==[0.]*9,
                'Verify-only endpoint must already be exact native INITIAL with zero velocity')
    control = retarget_control(arm_id, target)
    contacts = check_contacts()
    current_after = observe()
    after_observation = copy.deepcopy(before_observation)
    after_observation['arms'][arm_id] = copy.deepcopy(current_after['arms'][arm_id])
    require(physical_observation(after_observation) == physical_observation(current_after),
            'Reset changed physical observations outside its donor')
    after = _snapshot(scene, arms, cup, beads, identities, helper, arm_id)
    require(before['native_clock'] == _clock(scene), 'Reset advanced native time')
    require(source_check() is True, 'Sources changed during native robot reset')
    receipt = dict(schema_version=SCHEMA, model=MODEL, passed=True, arm_id=arm_id,
        after_stage_id=after_stage_id, native_scene_step=int(scene.t), simulation_s=float(scene.cur_t),
        target_command=copy.deepcopy(target_command), before_observation_sha256=digest(before_observation),
        after_observation_sha256=digest(after_observation), released_gate_sha256=digest(released_gate),
        release_event_sha256=digest(release_event), native_before=before, native_after=after,
        native_contact_check=contacts, controller_reset=control, source_bindings_verified=True,
        pre_reset_rest=copy.deepcopy(pre_reset_rest),
        physical_steps=0, simulation_duration_s=0., installed_setter_sha256=ENTITY_SOURCE_SHA256,
        pose_setter_performed=set_pose,
        scope='Explicit post-release donor INITIAL reset. No payload setters, time advance or swept collision claim.')
    validate_robot_reset_receipt(receipt, before_observation=before_observation,
        after_observation=after_observation, expected_command=target_command,
        expected_arm_id=arm_id, expected_stage_id=after_stage_id)
    return after_observation, receipt
