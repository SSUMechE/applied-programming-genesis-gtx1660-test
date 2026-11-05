"""Compute a short post-release endpoint from geometry, never a home target.

The search uses the same full CPU oracle and supplied endpoint solver. Its
recipient result is only a nominal CPU endpoint. Fresh native replay and the
unchanged support, handling, material and rest gates remain mandatory.
"""
from __future__ import annotations

import copy
import itertools
import math
import time

from physics.provided_manipulation import grasp_endpoint_candidates, solve_endpoint
from transport_challenge.contracts import ContractError
from transport_challenge.joint_kinematics import BoundedKinematics


RADII_M = (.05, .10, .15, .20, .25, .30)
DIRECTIONS = tuple(tuple(value / math.sqrt(sum(x*x for x in direction)) for value in direction)
                   for direction in itertools.product((-1, 0, 1), repeat=3) if any(direction))


def select_clearance_goal(geometry, commands, body_pose, model, arm, next_arm, *, recipient_seed, check):
    """Prefer the shortest searched shell with a valid direct withdrawal.

    Every candidate endpoint is checked against all three robots, the fixed map,
    released vessel and nominal contents. The next recipient's actual taught
    grasp target must have a valid CPU IK solution with the donor parked there.
    The caller still invokes the original RRT for the selected endpoint. A
    non-direct candidate is considered only if no direct candidate was found.
    """
    started = time.perf_counter()
    start = tuple(commands[arm]['q_rad'])
    oracle = geometry.configure(arm, commands, body_pose)
    if not oracle.state_valid(start):
        raise ContractError('Released donor start is invalid before handoff clearance: ' + str(oracle.last_failure))
    hand = oracle.fk(start)
    recipient_target = grasp_endpoint_candidates(model, body_pose.to_record())[0].to_record()
    attempts = []
    fallback = None
    # Both supported handoffs in this task require a real withdrawal. A zero
    # edge would add an artificial hold rather than remove unnecessary motion.
    candidates = ((radius, direction) for radius in RADII_M for direction in DIRECTIONS)
    for radius, direction in candidates:
        check()
        candidate = {'radius_m': radius, 'direction_world': list(direction)}
        donor = geometry.configure(arm, commands, body_pose)
        target_position = [value + radius*unit for value, unit in zip(hand.position_m, direction)]
        q = donor.ik(target_position, hand.quaternion_wxyz, start)
        if q is None or not donor.state_valid(q):
            candidate.update(status='DONOR_ENDPOINT_REJECTED', diagnostic=copy.deepcopy(donor.last_failure))
            attempts.append(candidate)
            continue
        direct = donor.edge_valid(start, q)
        achieved = donor.fk(q)
        nominal_hand = {'frame': 'world', 'position_m': list(achieved.position_m),
                        'quaternion_wxyz': list(achieved.quaternion_wxyz)}
        peers = copy.deepcopy(commands)
        peers[arm]['q_rad'] = list(q)
        recipient = geometry.configure(next_arm, peers, body_pose)
        kinematics = BoundedKinematics(recipient, recipient.bounds, geometry.scenario['case']['tool_workspace_m'],
            max_fk_calls=32, max_ik_calls=32, wall_check=check)
        endpoint = solve_endpoint(kinematics, recipient, tuple(peers[next_arm]['q_rad']),
            (recipient_target,), attempts=32, seed=recipient_seed, wall_check=check)
        candidate.update(direct_withdrawal_edge_valid=direct,
                         recipient_terminal_status=endpoint['status'])
        if endpoint['status'] != 'ENDPOINT_FOUND':
            candidate.update(status='RECIPIENT_GRASP_REJECTED', diagnostic=copy.deepcopy(recipient.last_failure))
            attempts.append(candidate)
            continue
        candidate['status'] = 'CPU_ENDPOINT_AND_RECIPIENT_VERIFIED'
        attempts.append(candidate)
        selected = {'schema_version': 'cpu-computed-handoff-clearance-endpoint-v1',
            'q_rad': list(q), 'nominal_goal_hand_world': nominal_hand,
            'nominal_search_target_hand_world': {'position_m': target_position,
                'quaternion_wxyz': list(hand.quaternion_wxyz), 'frame': 'world'},
            'selected_from_generic_search': True, 'home_joint_goal_used': False,
            'recipient_grasp_endpoint_verified': True, 'recipient_arm_id': next_arm,
            'recipient_terminal': endpoint, 'recipient_nominal_hand_world': recipient_target,
            'direct_withdrawal_edge_valid': direct, 'selected_radius_m': radius,
            'selected_direction_world': list(direction), 'attempts': copy.deepcopy(attempts),
            'search_policy': {'radii_m': list(RADII_M), 'directions': '26 normalized nonzero {-1,0,1}^3 directions',
                'selection': 'First valid direct candidate by increasing radius and deterministic direction order. First valid endpoint fallback only after the complete direct search.'},
            'physical_grasp_verified': False, 'physical_transport_verified': False,
            'scope': 'Computed CPU endpoint and nominal next-grasp feasibility only. Original RRT and final full-path checks still follow.'}
        if direct:
            selected['search_wall_s'] = time.perf_counter()-started
            return selected
        if fallback is None:
            fallback = selected
    if fallback is not None:
        fallback['attempts'] = attempts
        fallback['search_wall_s'] = time.perf_counter()-started
        return fallback
    raise ContractError('No bounded computed handoff-clearance endpoint permits the next recipient grasp: ' + str(attempts))
