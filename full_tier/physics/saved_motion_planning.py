"""Validate an immutable CPU mission against fresh native snapshots, without search.

CPU nominal poses are planning inputs, never native grasp receipts. Each actual
attachment is supplied by the unchanged runtime. This adapter validates the exact
saved joints and times against that actual held relation. It performs no IK,
RRT, time rescaling, pose repair, physical step or reference-plan substitution.
"""
from __future__ import annotations

import copy
import ast
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

from transport_challenge.contracts import ContractError, digest, point
from transport_challenge.joint_planning import JointLimits, JointTrajectory, make_joint_request, validate_joint_path


STAGES = (
    ('arm_0__approach', 'approach', 'arm_0'),
    ('arm_0__carry', 'carry', 'arm_0'),
    ('arm_0__return_initial', 'retreat', 'arm_0'),
    ('arm_1__approach', 'approach', 'arm_1'),
    ('arm_1__carry', 'carry', 'arm_1'),
    ('arm_1__return_initial', 'retreat', 'arm_1'),
    ('arm_2__approach', 'approach', 'arm_2'),
    ('arm_2__carry', 'carry', 'arm_2'),
)

REPLAY_POLICY_SOURCE_CHANGE_ALLOWLIST = frozenset((
    'physics/saved_motion_planning.py', 'physics/student_episode_driver.py',
    'physics/automatic_manipulation.py', 'physics/run_student_three_arm.py',
    'physics/episode_runtime.py', 'transport_challenge/episode_continuation.py',
    'physics/provided_robot_reset.py',
    'qa/transport_refinement_20261102/plan_cpu_mission.py',
))


def stages_for_policy(post_place_policy='return_home'):
    """Keep the legacy route and explicitly name computed clearance motions."""
    require(post_place_policy in ('return_home', 'clear_handoff', 'instant_reset', 'kinematic_return'), 'Unknown saved post-placement policy')
    if post_place_policy == 'return_home':
        return STAGES
    if post_place_policy in ('instant_reset', 'kinematic_return'):
        return tuple(stage for stage in STAGES if stage[1] != 'retreat')
    return tuple((stage_id.replace('__return_initial', '__clear_handoff'), kind, arm)
                 for stage_id, kind, arm in STAGES)


def require(value, message):
    if not value:
        raise ContractError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kinematic_return_schedule(stages, initial_commands):
    """Bind only the two existing outbound trajectories, without path search."""
    by_id = {stage['stage_id']: stage for stage in stages}
    result = []
    for arm in ('arm_0', 'arm_1'):
        carry_id, approach_id = arm+'__carry', arm+'__approach'
        if carry_id not in by_id:
            continue  # An explicitly incomplete CPU debug prefix has no bundle.
        require(approach_id in by_id, 'A return needs this donor\'s original approach')
        legs = []
        for stage_id in (carry_id, approach_id):
            trajectory = copy.deepcopy(by_id[stage_id]['trajectory'])
            legs.append({'source_stage_id': stage_id, 'trajectory': trajectory,
                         'trajectory_sha256': digest(trajectory)})
        require(legs[0]['trajectory']['waypoints'][0] == legs[1]['trajectory']['waypoints'][-1]
                and legs[1]['trajectory']['waypoints'][0] == initial_commands[arm]['q_rad'],
                'Reverse carry/approach does not connect exactly to INITIAL')
        result.append({'after_stage_id': carry_id, 'arm_id': arm,
            'target_command': copy.deepcopy(initial_commands[arm]), 'legs': legs,
            'sampling_dt_s': 1/60, 'display_sampling_hz': 60, 'include_waypoint_times': True,
            'physical_steps': 0, 'simulation_duration_s': 0.,
            'display_duration_s': sum(leg['trajectory']['duration_s'] for leg in legs)})
    return result


def iter_kinematic_return_samples(schedule, *, limits=None):
    """Reverse stored interpolation at 60 Hz and every waypoint boundary.

    The actual released native state is not synthesized here. The native helper
    records that state and its tracking delta before applying the first sample.
    """
    require(type(schedule) is dict and schedule.get('sampling_dt_s') == 1/60
            and schedule.get('display_sampling_hz') == 60 and schedule.get('include_waypoint_times') is True
            and type(schedule.get('sampling_dt_s')) in (int, float)
            and type(schedule.get('physical_steps')) is int and schedule['physical_steps'] == 0
            and schedule.get('simulation_duration_s') == 0., 'Explicit nonphysical reverse schedule required')
    legs = schedule.get('legs')
    require(type(legs) is list and len(legs) == 2, 'Return must reverse exactly carry then approach')
    trajectories = []
    for leg in legs:
        record = leg['trajectory']
        owned_limits = JointLimits(**record['limits']) if limits is None else limits
        require(owned_limits.to_record() == record['limits'], 'Reverse replay changes saved native limits')
        trajectory = JointTrajectory(record['waypoints'], record['edge_durations_s'], owned_limits)
        require(trajectory.to_record() == record and digest(record) == leg['trajectory_sha256'],
                'Reverse replay changes an original trajectory or its timing')
        trajectories.append(trajectory)
    require(trajectories[0].waypoints[0] == trajectories[1].waypoints[-1]
            and list(trajectories[1].waypoints[0]) == schedule['target_command']['q_rad'],
            'Reverse replay endpoint continuity differs')
    require(schedule.get('display_duration_s') == sum(tr.duration_s for tr in trajectories),
            'Reverse display duration must equal the original two durations')
    previous = None
    elapsed = 0.
    for leg_index, (leg, trajectory) in enumerate(zip(legs, trajectories)):
        times = sorted({0., trajectory.duration_s, *trajectory.waypoint_times_s,
            *(min(index/60, trajectory.duration_s) for index in range(1, math.ceil(trajectory.duration_s*60)+1))})
        for source_time in reversed(times):
            q = list(trajectory.sample(source_time).q_rad)
            display = elapsed+(trajectory.duration_s-source_time)
            if previous is not None and previous['display_s'] == display and previous['q_rad'] == q:
                continue
            require(previous is None or display > previous['display_s'], 'Reverse display clock does not advance')
            sample = {'leg_index': leg_index, 'source_stage_id': leg['source_stage_id'],
                'trajectory_sha256': leg['trajectory_sha256'], 'source_time_s': source_time,
                'display_s': display, 'q_rad': q}
            previous = sample
            yield sample
        elapsed += trajectory.duration_s


def verify_imported_forward_paths(value):
    """Verify explicit old-plan provenance, never re-sign old source bindings."""
    origin = value.get('plan_import')
    if origin is None:
        return
    require(origin.get('schema_version') == 'imported-unchanged-forward-cpu-paths-v1'
            and value.get('post_place_policy') == 'kinematic_return', 'Unsupported forward-path import')
    path = Path(origin['original_bundle_file'])
    require(path.is_absolute() and file_sha(path) == origin['original_bundle_file_sha256']
            == value['source_pins'].get(str(path)), 'Original immutable bundle bytes changed')
    original = json.loads(path.read_bytes())
    require(original.get('bundle_sha256') == origin['original_bundle_sha256']
            == digest({k: v for k, v in original.items() if k != 'bundle_sha256'}),
            'Original CPU bundle digest differs')
    for key in ('stages', 'initial_commands_by_arm', 'world_sha256', 'scenario_sha256', 'design_sha256',
                'task_manifest_sha256', 'provided_wait_policy', 'offline_planner', 'cpu_planning_result',
                'cpu_planning_wall_s', 'stored_motion_duration_s'):
        require(value.get(key) == original.get(key), 'Imported forward planning data changed: '+key)
    require(origin.get('original_source_pins') == original['source_pins'], 'Original CPU source provenance changed')
    changes = origin.get('replay_source_changes', {})
    changed = {name for name, old in original['source_pins'].items() if value['source_pins'].get(name) != old}
    require(set(changes) == changed, 'Imported source changes are not completely declared')
    root = Path(__file__).resolve().parents[1]
    for name in changed:
        old, change = original['source_pins'][name], changes[name]
        path = Path(name)
        require(path.is_relative_to(root) and path.relative_to(root).as_posix() in REPLAY_POLICY_SOURCE_CHANGE_ALLOWLIST,
                'Geometry, input or RRT source changed during path import: '+name)
        snapshot = change.get('original_source_snapshot')
        require(change.get('old_sha256') == old and change.get('new_sha256') == value['source_pins'].get(name)
                and value['source_pins'].get(snapshot) == old, 'Changed replay source lacks its exact historical bytes')
    require(all(type(origin.get(key)) is int and origin[key] == 0
                for key in ('new_rrt_calls', 'new_ik_calls', 'native_physical_steps')),
            'Imported paths must not claim new planning or physics')
    elapsed = origin.get('conversion_wall_s')
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0
            and value.get('cpu_path_preparation_wall_s') == original['cpu_planning_wall_s']+elapsed,
            'Path import omits original CPU work or conversion time')


def saved_wait_policy():
    """Read the actual provided wait constants without importing a simulator."""
    directory = Path(__file__).resolve().parent
    def literals(name, wanted):
        tree = ast.parse((directory/name).read_text(encoding='utf-8'))
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in wanted:
                        values[target.id] = ast.literal_eval(node.value)
        require(set(values) == set(wanted), 'Actual provided wait policy constants are missing')
        return values
    protocol = literals('episode_runtime.py', {'PROTOCOL', 'TRANSPORT_PROTOCOL'})
    extra = literals('automatic_manipulation.py', {'RELEASED_EXTRA_HOLD_S', 'RELEASED_EXTRA_LIMIT_S'})
    return {**protocol, 'additional_material_wait_s': extra['RELEASED_EXTRA_HOLD_S'],
            'maximum_additional_material_wait_s': extra['RELEASED_EXTRA_LIMIT_S']}


def pose_error(a, b):
    from physics.carried_body import WorldPose
    a = WorldPose(a['position_m'], a['quaternion_wxyz'])
    b = WorldPose(b['position_m'], b['quaternion_wxyz'])
    dot = abs(math.fsum(x*y for x, y in zip(a.quaternion_wxyz, b.quaternion_wxyz)))
    cosine = dot / (math.hypot(*a.quaternion_wxyz)*math.hypot(*b.quaternion_wxyz))
    return {'position_m': math.dist(a.position_m, b.position_m),
            'orientation_rad': 2*math.acos(min(1., cosine))}


class SavedMissionPlan:
    """Deeply owned transport and post-placement motions with exact identities."""

    def __init__(self, record, *, path):
        self.path = Path(path).resolve(strict=True)
        self.file_sha256 = file_sha(self.path)
        value = copy.deepcopy(record)
        require(value.get('schema_version') == 'saved-three-arm-mission-v1', 'Unsupported saved mission schema')
        expected = value.get('bundle_sha256')
        require(expected == digest({k: v for k, v in value.items() if k != 'bundle_sha256'}), 'Saved mission bundle digest differs')
        self.sha256 = expected
        self.post_place_policy = value.get('post_place_policy', 'return_home')
        self.stage_signatures = stages_for_policy(self.post_place_policy)
        stages = value.get('stages')
        require(isinstance(stages, list) and [(s.get('stage_id'), s.get('kind'), s.get('active_arm_id')) for s in stages] == list(self.stage_signatures),
                'Saved mission motion count/order differs from its explicit post-placement policy')
        require(isinstance(value.get('source_pins'), dict) and bool(value['source_pins']), 'Saved CPU source pins are missing')
        waits = value.get('provided_wait_policy', {})
        require(all(waits.get(key) == expected for key, expected in saved_wait_policy().items()),
                'Saved provided operation/wait policy differs from the actual unchanged runtime')
        offline = value.get('offline_planner', {})
        require(all(isinstance(offline.get(k), str) and offline[k] for k in ('module', 'source_sha256', 'plan_path_source_sha256')),
                'Saved mission lacks the actual offline planner identity')
        require(all(part.isidentifier() for part in offline['module'].split('.')), 'Saved offline planner module name is invalid')
        initial = value.get('initial_commands_by_arm', {})
        require(set(initial) == {'arm_0', 'arm_1', 'arm_2'}, 'Saved mission initial commands omit an arm')
        commands = copy.deepcopy(initial)
        for name, command in commands.items():
            point(command['q_rad'], 7)
            point(command['finger_q_m'], 2)
        resets = value.get('post_place_resets', [])
        expected_resets = ([{'after_stage_id': arm+'__carry', 'arm_id': arm,
            'target_command': copy.deepcopy(initial[arm]), 'zero_joint_velocity': True,
            'physical_steps': 0, 'simulation_duration_s': 0.} for arm in ('arm_0', 'arm_1')]
            if self.post_place_policy == 'instant_reset' else [])
        require(resets == expected_resets, 'Saved post-placement reset schedule or exact INITIAL target differs')
        require(type(resets) is list and all(type(item.get('physical_steps')) is int
            and item['physical_steps'] == 0 and item.get('zero_joint_velocity') is True
            and type(item.get('simulation_duration_s')) in (int, float)
            and item['simulation_duration_s'] == 0. for item in resets), 'Malformed zero-time robot reset schedule')
        limits_record = None
        for stage in stages:
            tr = stage['trajectory']
            limits = JointLimits(**tr['limits'])
            trajectory = JointTrajectory(tr['waypoints'], tr['edge_durations_s'], limits)
            require(trajectory.to_record() == tr, 'Saved trajectory is noncanonical or has changed timing')
            require(tr['waypoints'][0] == commands[stage['active_arm_id']]['q_rad'], 'Saved stage command sequence is discontinuous')
            peers = {name: command for name, command in commands.items() if name != stage['active_arm_id']}
            require(stage['expected_peer_commands_by_arm'] == peers, 'Saved stage peer command snapshot differs')
            require(limits_record is None or limits_record == tr['limits'], 'Saved native limits change between stages')
            limits_record = tr['limits']
            commands[stage['active_arm_id']]['q_rad'] = copy.deepcopy(tr['waypoints'][-1])
            if self.post_place_policy in ('instant_reset', 'kinematic_return') and stage['kind'] == 'carry' and stage['active_arm_id'] != 'arm_2':
                commands[stage['active_arm_id']] = copy.deepcopy(initial[stage['active_arm_id']])
            if stage['kind'] == 'retreat':
                if self.post_place_policy == 'return_home':
                    require(tr['waypoints'][-1] == initial[stage['active_arm_id']]['q_rad'], 'Saved retreat does not return to the declared initial command')
                else:
                    terminal = stage.get('offline_planning', {}).get('terminal', {})
                    require(terminal.get('schema_version') == 'cpu-computed-handoff-clearance-endpoint-v1'
                            and terminal.get('q_rad') == tr['waypoints'][-1]
                            and isinstance(stage.get('nominal_goal_hand_world'), dict)
                            and terminal.get('selected_from_generic_search') is True
                            and terminal.get('recipient_grasp_endpoint_verified') is True,
                            'Saved handoff clearance lacks its computed endpoint and next-recipient geometry evidence')
        returns = value.get('post_place_returns', [])
        expected_returns = kinematic_return_schedule(stages, initial) if self.post_place_policy == 'kinematic_return' else []
        require(returns == expected_returns, 'Kinematic returns must bind exact own carry then approach and original timing')
        if self.post_place_policy == 'kinematic_return':
            require(len(returns) == 2, 'The six-motion kinematic policy requires two nonphysical returns')
        verify_imported_forward_paths(value)
        self._json = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
        self.verify_sources()

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_bytes()), path=path)

    def record(self):
        return json.loads(self._json)

    @property
    def offline_planner(self):
        return self.record()['offline_planner']

    def verify_sources(self):
        require(file_sha(self.path) == self.file_sha256, 'Saved plan file changed')
        for name, expected in self.record()['source_pins'].items():
            require(Path(name).is_absolute() and file_sha(name) == expected, 'Saved CPU source or input changed: ' + name)
        return True

    def verify_bound_inputs(self, world, scenario, *, design_sha256=None, task_manifest_sha256=None):
        value = self.record()
        require(value['world_sha256'] == digest(world) and value['scenario_sha256'] == digest(scenario), 'Saved plan world or initialization differs')
        require(world.get('rigid_runtime_profile') == 'rigid16_bead_tc2h_v1'
                and world.get('payload_mode') != 'empty' and 'particle_material_request' not in world,
                'This saved replay branch requires the explicit original four-bead rigid16 profile')
        route = [world['source_support']['id'], *[s['id'] for s in world['relay_supports']], world['goal_support']['id']]
        require(len(route) == 4 and len(set(route)) == 4, 'Saved replay requires two distinct supported handoffs')
        for stage in value['stages']:
            arm = int(stage['active_arm_id'][-1])
            support = route[arm+1] if stage['kind'] == 'retreat' else route[arm]
            destination = route[arm+1] if stage['kind'] == 'carry' else None
            require(stage['support_id'] == support and stage.get('destination_support_id') == destination,
                    'Saved stage source/destination support order differs from the complete mission')
        if design_sha256 is not None:
            require(value['design_sha256'] == design_sha256, 'Saved plan installation/design differs')
        if task_manifest_sha256 is not None:
            require(value['task_manifest_sha256'] == task_manifest_sha256, 'Saved plan task manifest differs')
        return True

    def verify_initial(self, world, scenario, commands, limits_by_arm):
        self.verify_bound_inputs(world, scenario)
        value = self.record()
        require(commands == value['initial_commands_by_arm'], 'Actual initial command representation differs from the frozen CPU plan')
        for stage in value['stages']:
            require(stage['trajectory']['limits'] == limits_by_arm[stage['active_arm_id']].to_record(), 'Saved plan does not use these exact native limits')
        return True

    def stage(self, index):
        require(type(index) is int and 0 <= index < len(self.stage_signatures), 'Saved stage index is outside the declared motion sequence')
        return self.record()['stages'][index]

    def reset_after(self, stage_id):
        return next((item for item in self.record().get('post_place_resets', [])
                     if item['after_stage_id'] == stage_id), None)

    def kinematic_return_after(self, stage_id):
        return next((item for item in self.record().get('post_place_returns', [])
                     if item['after_stage_id'] == stage_id), None)


def _endpoint(query, request, stage, spec, oracle):
    """Check a frozen nominal endpoint, retaining actual physical placement gates.

    This is explicitly not the original precise actual-grasp seating IK receipt.
    A nominal saved target is checked by FK. The actual held body must separately
    satisfy the original physical target tolerances and named support geometry.
    Runtime support duration, rest, release and material gates remain required.
    """
    from physics.carried_body import WorldPose
    from physics.episode_transport_inputs import _bottom_corners, VERIFY_POSITION_TOLERANCE_M, VERIFY_ORIENTATION_TOLERANCE_RAD
    from physics.provided_automatic_grasp import POSITION_TOLERANCE_M, ORIENTATION_TOLERANCE_RAD
    from physics.handle_contacts import MAX_PENETRATION_M
    goal_q = tuple(stage['trajectory']['waypoints'][-1])
    require(oracle.state_valid(goal_q), 'Frozen saved endpoint fails the actual native state/contact check')
    witness = copy.deepcopy(getattr(query, '_last_detection', None))
    native_fk = query.fk(goal_q)
    hand = WorldPose(native_fk['position_m'], native_fk['quaternion_wxyz']).to_record()
    result = {'schema_version': 'saved-nominal-endpoint-v1', 'passed': False,
              'fresh_ik_calls': 0, 'frozen_endpoint_q_rad': list(goal_q), 'native_hand_world': hand,
              'old_actual_grasp_seating_ik_certificate_claimed': False}
    if request['kind'] == 'retreat':
        require(list(goal_q) == request['goal']['q_rad'], 'Saved retreat changes the declared exact terminal command')
        if request['goal']['kind'] == 'clear_handoff':
            nominal_error = pose_error(hand, stage['nominal_goal_hand_world'])
            require(nominal_error['position_m'] <= VERIFY_POSITION_TOLERANCE_M
                    and nominal_error['orientation_rad'] <= VERIFY_ORIENTATION_TOLERANCE_RAD,
                    'Computed clearance endpoint differs from its saved nominal FK')
            result.update(passed=True, kind='exact_computed_handoff_clearance',
                          frozen_nominal_target_fk_error=nominal_error, endpoint_hand_pose=hand)
        else:
            require(request['goal']['kind'] == 'return_initial', 'Unknown saved retreat endpoint kind')
            result.update(passed=True, kind='exact_return_initial')
        return result
    nominal_hand = stage['nominal_goal_hand_world']
    nominal_error = pose_error(hand, nominal_hand)
    nominal_limits = ((VERIFY_POSITION_TOLERANCE_M, VERIFY_ORIENTATION_TOLERANCE_RAD)
                      if request['kind'] == 'carry' else (POSITION_TOLERANCE_M, ORIENTATION_TOLERANCE_RAD))
    result['frozen_nominal_target_fk_error'] = nominal_error
    result['frozen_nominal_target_fk_limits'] = {'position_m': nominal_limits[0], 'orientation_rad': nominal_limits[1]}
    require(nominal_error['position_m'] <= nominal_limits[0] and nominal_error['orientation_rad'] <= nominal_limits[1],
            'Frozen endpoint native FK differs from its saved CPU nominal target: '+repr(nominal_error))
    if request['kind'] == 'approach':
        actual_error = pose_error(hand, request['goal'])
        result['actual_current_grasp_target_error'] = actual_error
        require(actual_error['position_m'] <= POSITION_TOLERANCE_M and actual_error['orientation_rad'] <= ORIENTATION_TOLERANCE_RAD,
                'Frozen approach does not align with the actual current vessel for provided attachment: '+repr(actual_error))
        result.update(passed=True, kind='actual_current_provided_grasp_alignment', endpoint_hand_pose=request['goal'])
        return result
    require(spec is not None and spec.grasp_model == 'provided_ideal', 'Saved carry requires its actual native provided attachment')
    require(isinstance(witness, dict) and witness.get('pair_witness_multiset_agrees') is True, 'Frozen carry endpoint lacks same-query native contact witnesses')
    actual_body = spec.body_pose_at(WorldPose(hand['position_m'], hand['quaternion_wxyz']))
    placement = request['placement_goal']
    nominal = placement.get('seating', {}).get('nominal_goal', placement)
    require(stage['destination_support_id'] == nominal['support_id'], 'Saved carry changes its actual destination support')
    require(stage['nominal_goal_body_world'] == nominal['body_world'], 'Saved canonical destination body pose differs')
    body_error = pose_error(actual_body.to_record(), nominal['body_world'])
    execution = request['world']['physics_settings']['execution_limits']
    require(body_error['position_m'] <= execution['final_body_position_error_m']
            and body_error['orientation_rad'] <= execution['final_body_orientation_error_rad'],
            'Frozen carry with the actual grasp misses the unchanged physical body target limits: '+repr(body_error))
    support = nominal['support']
    center = support['center_world_m']
    size = support['size_m']
    top = center[2] + size[2]/2
    corners = _bottom_corners(spec.model, actual_body)
    footprint_ok = all(abs(c[i]-center[i]) <= size[i]/2 for c in corners for i in (0, 1))
    gaps = [corner[2]-top for corner in corners]
    require(footprint_ok and all(-MAX_PENETRATION_M < gap < 0. for gap in gaps),
            'Frozen carry actual-grasp body does not seat within the original support footprint/contact-depth range: '
            +repr({'footprint_inside': footprint_ok, 'bottom_signed_gaps_m': gaps}))
    contacts = [row for row in witness['rows'] if row.get('category') == 'named_tray_bottom_support_top'
                and row.get('support_station_id') == nominal['support_id'] and row.get('tray_component') == 'solid_0']
    require(bool(contacts), 'Frozen carry endpoint has no actual same-query named bottom/support contact')
    result.update(passed=True, kind='nominal_plan_actual_grasp_physical_target_geometry',
                  predicted_actual_body_world=actual_body.to_record(), canonical_goal_body_world=nominal['body_world'],
                  actual_grasp_body_target_error=body_error,
                  unchanged_physical_body_limits={'position_m': execution['final_body_position_error_m'],
                                                  'orientation_rad': execution['final_body_orientation_error_rad']},
                  bottom_footprint_inside_support=footprint_ok, bottom_signed_gaps_m=gaps,
                  original_maximum_contact_penetration_m=MAX_PENETRATION_M,
                  named_bottom_support_witness_count=len(contacts), same_query_sha256=digest(witness),
                  physical_support_rest_release_required=True, endpoint_hand_pose=nominal_hand)
    return result


def validate_saved_motion(request, saved_plan, index, *, previous_commands, output, source_check, wall_check, fixture_readback='scalar'):
    from physics import student_motion_planning as motion
    from physics.run_joint_path import _json_safe
    require(isinstance(saved_plan, SavedMissionPlan), 'Require an owned saved CPU mission')
    require(callable(source_check) and callable(wall_check), 'Saved replay requires live wall and source guards')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not any((output/name).exists() for name in ('saved_motion_request.json', 'saved_motion_result.json', 'path_validation.jsonl', 'assets')),
            'Saved motion validation output must be fresh')
    stage = saved_plan.stage(index)
    request = motion._request(copy.deepcopy(request),
        saved_clearance=saved_plan.post_place_policy == 'clear_handoff' and stage['kind'] == 'retreat')
    require(tuple(request[k] for k in ('stage_id', 'kind', 'active_arm_id')) == saved_plan.stage_signatures[index], 'Current operation differs from the saved stage')
    saved_plan.verify_bound_inputs(request['world'], request['scenario'])
    active = request['active_arm_id']
    from physics.single_active_boundary import contract_from_observation,robot_coordinates
    single=contract_from_observation(request['observation'],request['scenario'])
    require(stage['expected_peer_commands_by_arm']=={k:v for k,v in previous_commands.items() if k!=active},
            'Actual preceding peer commands differ from the frozen stage')
    require(stage['trajectory']['waypoints'][0] == previous_commands[active]['q_rad'] == request['command_start_q_rad'],
            'Frozen trajectory start differs from the exact preceding command')
    query = None
    started = time.perf_counter()
    result = {'schema_version': 'saved-motion-attempt-v1', 'status': 'SAVED_MOTION_VALIDATION_FAILED',
              'stage_id': request['stage_id'], 'request_sha256': digest(request), 'saved_bundle_sha256': saved_plan.sha256,
              'saved_stage_sha256': digest(stage), 'physical_steps': 0, 'fresh_ik_calls': 0,
              'provided_adapter_plan_path_invocations': 0, 'runtime_rrt_calls': 0, 'measured_host_timings_s': {},
              'full_mission_passed': False, 'student_release_approved': False, 'cleanup_errors': []}
    response = failure = None

    def save(name, value):
        with (output/name).open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(_json_safe(value), indent=2, allow_nan=False)+'\n')

    def guard():
        wall_check()
        require(source_check() is True, 'Saved replay source bindings changed')

    def timed(name, action):
        tick = time.perf_counter()
        try:
            return action()
        finally:
            result['measured_host_timings_s'][name] = time.perf_counter()-tick

    try:
        guard()
        timed('saved_cpu_source_boundary_s', saved_plan.verify_sources)
        save('saved_motion_request.json', {'request': request, 'saved_stage': stage, 'saved_bundle_sha256': saved_plan.sha256})
        actual_body_error = pose_error(request['observation']['body_world'], stage['nominal_body_world'])
        execution = request['world']['physics_settings']['execution_limits']
        peer_errors = {name: max(abs(a-b) for a, b in zip(robot_coordinates(request['observation'],name)[:7], command['q_rad']))
                       for name, command in stage['expected_peer_commands_by_arm'].items()}
        result['nominal_actual_boundary'] = {'nominal_body_world': stage['nominal_body_world'],
            'actual_body_world': request['observation']['body_world'], 'body_pose_error': actual_body_error,
            'original_body_limits': {'position_m': execution['final_body_position_error_m'], 'orientation_rad': execution['final_body_orientation_error_rad']},
            'peer_measured_command_maximum_joint_error_rad': peer_errors if single is None else {},
            'fixed_peer_configuration_maximum_joint_error_rad': peer_errors if single is not None else {},
            'original_joint_rest_boundary_tolerance_rad': execution['final_joint_error_rad']}
        require(actual_body_error['position_m'] <= execution['final_body_position_error_m']
                and actual_body_error['orientation_rad'] <= execution['final_body_orientation_error_rad'],
                'Actual supported vessel differs from the saved nominal boundary beyond original body limits: '+repr(actual_body_error))
        require(all(error <= execution['final_joint_error_rad'] for error in peer_errors.values()),
                'Actual resting peers differ from the frozen command snapshot beyond original joint limits: '+repr(peer_errors))
        query, spec = timed('query_scene_construction_s', lambda: motion._query_scene(request, output))
        case, policy = motion.effective_query_case(request)
        require(fixture_readback in ('scalar', 'batched'), 'Unknown fixture readback mode')
        if fixture_readback == 'batched':
            result['fixture_readback'] = timed('fixture_readback_setup_s', query.enable_batched_fixture_readback)
        motion.install_open_finger_margin(query)
        oracle = motion._Oracle(query, guard)
        observed = point(request['observation']['arms'][active]['q_all_dofs'][:7], 7)
        start = point(request['command_start_q_rad'], 7)
        tolerance = request['world']['physics_settings']['execution_limits']['final_joint_error_rad']
        require(max(abs(a-b) for a, b in zip(observed, start)) <= tolerance, 'Actual arm start exceeds the original rest-boundary command tolerance')
        def bridge():
            require(oracle.state_valid(observed) and oracle.state_valid(start) and oracle.edge_valid(observed, start),
                    'Actual observed start, frozen command or sampled bridge is invalid')
        timed('observed_command_bridge_validation_s', bridge)
        limits = JointLimits(query.bounds, case['motion_limits']['velocity_rad_s'], case['motion_limits']['acceleration_rad_s2'])
        tr = stage['trajectory']
        require(tr['limits'] == limits.to_record(), 'Frozen trajectory native limits differ')
        trajectory = JointTrajectory(tr['waypoints'], tr['edge_durations_s'], limits)
        require(trajectory.to_record() == tr, 'Frozen saved waypoints or edge times changed')
        endpoint = timed('saved_endpoint_native_validation_s', lambda: _endpoint(query, request, stage, spec, oracle))
        save('saved_endpoint_receipt.json', endpoint)
        planning_request = make_joint_request(start, trajectory.waypoints[-1], limits, max_queries=request['planning']['max_queries'],
                                             seed=request['planning']['seed'], options=request['planning']['options'])
        timed('common_final_state_edge_validation_s', lambda: validate_joint_path(trajectory.waypoints, planning_request, limits, oracle))
        validation = timed('durable_path_collision_validation_s', lambda: motion._durable_validation(query, trajectory, case, output, guard))
        require(query.scene.t == 0 and float(query.scene.cur_t) == 0., 'Saved path native query advanced physics')
        offline = saved_plan.offline_planner
        evidence = {'trajectory_sha256': digest(tr), 'returned_path_sha256': digest(tr['waypoints']),
                    'plan_path_source_sha256': offline['plan_path_source_sha256'], 'dimension': 7,
                    'motion_source': 'saved_cpu_plan', 'offline_planner': offline, 'saved_bundle_sha256': saved_plan.sha256,
                    'runtime_plan_path_invocations': 0, 'runtime_ik_calls': 0, 'runtime_rrt_calls': 0,
                    'physical_execution_performed': False, 'offline_planning': stage['offline_planning']}
        response = {'schema_version': 'saved-motion-response-v1', 'request_sha256': digest(request), 'stage_id': request['stage_id'],
                    'kind': request['kind'], 'active_arm_id': active, 'plan': {'schema_version': 'joint-plan-v1', 'trajectory': tr, 'evidence': evidence},
                    'endpoint_hand_pose': None if request['kind'] == 'retreat' else endpoint['endpoint_hand_pose'],
                    'source_bindings_verified': True, 'physical_steps': 0, 'observed_start_valid': True,
                    'command_start_valid': True, 'observed_to_command_edge_valid': True, 'durable_path_validation': validation,
                    'provided_adapter_plan_path_invocations': 0, 'runtime_ik_calls': 0, 'runtime_rrt_calls': 0,
                    'saved_bundle_sha256': saved_plan.sha256, 'saved_stage_sha256': digest(stage), 'cleanup_errors': [],
                    'query_geometry_policy': policy, 'endpoint_assessment': endpoint}
        if 'particle_material_request' in request['world']:
            response.update(contents_collision_queried=False,
                contents_query_scope='Frozen robot/vessel/map path only. Actual native particles are authenticated and checked by the unchanged physical runtime.',
                validation_backend='native_full_geometry_reference')
        result.update(status='SAVED_MOTION_DISCRETE_PATH_VALIDATED', response=response)
    except Exception as exc:
        failure = exc
        result.update(failure={'type': type(exc).__name__, 'message': str(exc)}, traceback=traceback.format_exc())
    finally:
        if query is not None:
            try:
                evidence = timed('query_evidence_export_s', query.evidence)
                save('query_evidence.json', evidence)
                result['query_evidence_sha256'] = file_sha(output/'query_evidence.json')
            except Exception as exc:
                result['cleanup_errors'].append('query evidence: '+repr(exc))
            try:
                timed('query_cleanup_s', query.close)
            except Exception as exc:
                result['cleanup_errors'].append('query close: '+repr(exc))
        try:
            timed('saved_cpu_final_source_boundary_s', saved_plan.verify_sources)
            require(source_check() is True, 'Saved replay sources changed by validation completion')
        except Exception as exc:
            result['cleanup_errors'].append('source verification: '+repr(exc))
        if result['cleanup_errors']:
            result['status'] = 'SAVED_MOTION_VALIDATION_FAILED'
            result.pop('response', None)
        result['measured_host_timings_s']['whole_attempt_including_cleanup_s'] = time.perf_counter()-started
        if response is not None:
            response['measured_host_timings_s'] = copy.deepcopy(result['measured_host_timings_s'])
        save('saved_motion_result.json', result)
    if failure is not None:
        raise ContractError('Saved native path validation failed: '+str(failure)) from failure
    require(not result['cleanup_errors'], 'Saved native path source/cleanup checks failed')
    return response
