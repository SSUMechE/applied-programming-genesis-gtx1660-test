"""Typed measured active coordinates and separately declared fixed-peer geometry."""
import copy
import numpy as np
from transport_challenge.contracts import ContractError, point


def contract_from_observation(observation, scenario=None):
    value=observation.get('single_active_arm_contract')
    if value is None:return None
    from physics.fixed_peer_arms import validated_single_active_contract
    contract=validated_single_active_contract(value)
    active=contract['active_arm_id']
    if set(observation['arms'])!={active}:
        raise ContractError('Only the actual active native arm belongs in measured joint observations')
    point(observation['arms'][active]['q_all_dofs'],9)
    if scenario is not None:
        records={r['id']:r for r in scenario['robots']}
        if set(records)!={active,*contract['frozen_arm_ids']}:
            raise ContractError('Original scenario robot geometry was dropped')
        for name in contract['frozen_arm_ids']:
            q=list(point(contract['fixed_peers'][name]['native_initial_q'],9))
            expected=[float(np.float32(v)) for v in [*records[name]['q_rad'],*records[name]['finger_q_m']]]
            if q!=expected:raise ContractError('Fixed peer is not at its exact supplied native INITIAL configuration')
    return contract


def robot_coordinates(observation, name):
    """Return real native q9 or source-qualified frozen configuration, never fake qd."""
    if name in observation['arms']:
        return list(point(observation['arms'][name]['q_all_dofs'],9))
    contract=contract_from_observation(observation)
    if contract is None or name not in contract['frozen_arm_ids']:
        raise ContractError('Missing actual arm or qualified static peer')
    return list(point(contract['fixed_peers'][name]['native_initial_q'],9))


def with_static_commands(commands, observation):
    result=copy.deepcopy(commands)
    contract=contract_from_observation(observation)
    if contract is not None:
        for name in contract['frozen_arm_ids']:
            q=robot_coordinates(observation,name)
            frozen={'q_rad':q[:7],'finger_q_m':q[7:]}
            if name in result and result[name]!=frozen:
                raise ContractError('A command attempted to move a qualified fixed peer')
            result.setdefault(name,frozen)
    return result
