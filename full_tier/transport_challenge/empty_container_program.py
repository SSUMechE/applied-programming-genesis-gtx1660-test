"""Explicit empty-container seven-joint input. No planner or native calls.

Students edit student/empty_container_path.json. Times are simulation seconds
on the native .0025s grid and are never scaled, stretched, clipped or replaced.
The supplied runtime separately checks the measured-to-command connection.
"""
import copy
from decimal import Decimal
import json
import math
from pathlib import Path

from .contracts import ContractError, digest
from .joint_planning import JointLimits, JointTrajectory

SCHEMA = 'empty-container-joint-path-v1'
PROFILE = 'empty_rigid_default_v1'
DT = .0025
FIELDS = {'schema_version', 'arm_id', 'coordinate_mode', 'waypoints_rad',
          'edge_durations_s', 'end_hold_s'}


def _need(ok, message):
    if not ok:
        raise ContractError(message)


def _number(value, label):
    _need(type(value) in (int, float) and math.isfinite(value), label+' must be finite numeric data')
    return float(value)


def _q(value, label):
    _need(type(value) in (list, tuple) and len(value) == 7, label+' needs seven joint angles in radians')
    return [_number(v, label) for v in value]


def _steps(seconds, label):
    ratio = Decimal(str(seconds)) / Decimal('.0025')
    _need(ratio > 0 and ratio == ratio.to_integral_value() and ratio <= 2**53-1,
          label+' must be a positive integer multiple of native dt=.0025s')
    return int(ratio)


def validate_empty_program(record):
    """Return detached six-field input, without adding evidence or default data."""
    _need(type(record) is dict and set(record) == FIELDS, 'Empty program requires exactly its six fields')
    _need(record['schema_version'] == SCHEMA, 'Unknown empty program schema')
    _need(record['arm_id'] == 'arm_0', 'This first empty-container path explicitly supports arm_0 only')
    mode = record['coordinate_mode']
    _need(mode in ('offset_from_grasp_start', 'absolute'), 'Unknown joint coordinate_mode')
    rows = record['waypoints_rad']
    _need(type(rows) in (list, tuple) and len(rows) >= 2, 'Supply at least two seven-joint waypoints')
    points = [_q(q, 'waypoint') for q in rows]
    _need(all(a != b for a, b in zip(points, points[1:])), 'Adjacent waypoints must differ')
    _need(mode != 'offset_from_grasp_start' or points[0] == [0.]*7,
          'The first offset must be exactly seven zeros')
    raw_times = record['edge_durations_s']
    _need(type(raw_times) in (list, tuple) and len(raw_times) == len(points)-1,
          'Supply exactly one duration per waypoint edge')
    times = [_number(v, 'edge duration') for v in raw_times]
    for value in times:
        _steps(value, 'edge duration')
    hold = _number(record['end_hold_s'], 'end_hold_s')
    _need(hold >= .2, 'end_hold_s must be at least .2 simulation seconds')
    _steps(hold, 'end_hold_s')
    return dict(schema_version=SCHEMA, arm_id='arm_0', coordinate_mode=mode,
                waypoints_rad=points, edge_durations_s=times, end_hold_s=hold)


def load_empty_program(path):
    try:
        record = json.loads(Path(path).read_bytes())
    except (OSError, ValueError, UnicodeError) as exc:
        raise ContractError('Cannot read empty-container joint path: '+str(exc)) from exc
    return validate_empty_program(record)


def resolve_empty_program(program, start_q_rad, limits):
    """Resolve once from the supplied commanded7q anchor after native attach.

    start_q_rad is the preceding supplied Dispatcher command, not measured q.
    Native measured error and the connection edge are separately verified by
    the runtime. This pure conversion never changes pose or an input duration.
    """
    program = validate_empty_program(program)
    start = _q(start_q_rad, 'preceding commanded anchor')
    _need(type(limits) is JointLimits and limits.dimension == 7, 'Use the actual seven-joint limits')
    if program['coordinate_mode'] == 'offset_from_grasp_start':
        points = [[a+b for a, b in zip(start, delta)] for delta in program['waypoints_rad']]
        _need(all(math.isfinite(v) for q in points for v in q), 'Resolved joint offset overflowed')
        # Includes signed-zero anchors. No measured-state substitution occurs.
        points[0] = list(start)
    else:
        points = copy.deepcopy(program['waypoints_rad'])
        _need(points[0] == start, 'Absolute first waypoint must exactly equal the preceding supplied command')
    trajectory = JointTrajectory(points, program['edge_durations_s'], limits)
    edge_steps = [_steps(v, 'edge duration') for v in program['edge_durations_s']]
    return dict(schema_version='resolved-empty-container-joint-path-v1', arm_id=program['arm_id'],
        coordinate_mode=program['coordinate_mode'], start_q_rad=start,
        start_anchor='automatic-grasp boundary preceding supplied commanded7q, not measured q',
        waypoints_rad=points, edge_durations_s=list(program['edge_durations_s']),
        edge_steps=edge_steps, motion_steps=sum(edge_steps), end_hold_s=program['end_hold_s'],
        end_hold_steps=_steps(program['end_hold_s'], 'end_hold_s'), dt_s=DT,
        trajectory=trajectory.to_record(), input_program_sha256=digest(program),
        clipping_or_retiming=False, plan_path_invocations=0)


def empty_container_overlay(world, scenario):
    """After manifest verification/design binding, derive a fresh empty recipe.

    Only the payload selection changes. No native scene or state exists here.
    Original case/map, vessel, source/goal, arm layout and controller stay bound.
    The CLI records and source-pins the effective recipe before scene creation.
    """
    _need(type(world) is dict and type(scenario) is dict, 'Explicit world and initialization mappings required')
    _need(scenario.get('case') == world.get('case_record') and scenario.get('body_id') == world.get('object_id')
          and scenario.get('provenance', {}).get('world_sha256') == digest(world),
          'Empty overlay requires the exact already-bound world/scenario')
    w, s = copy.deepcopy(world), copy.deepcopy(scenario)
    w['payload_mode'] = 'empty'
    w['initial_contents'] = []
    w['initial_content_world_centers_m'] = []
    w.pop('particle_material_request', None)
    w['rigid_runtime_profile'] = PROFILE
    w['physics_settings_sha256'] = digest(w['physics_settings'])
    s['contents'] = []
    s['provenance']['empty_payload_source_world_sha256'] = digest(world)
    s['provenance']['world_sha256'] = digest(w)
    s['provenance']['scope'] = 'Explicit empty-container input overlay. Actual geometry/dynamics remain native. No material, delivery or whole-mission credit.'
    receipt = dict(schema_version='provided-empty-container-overlay-v1', payload_mode='empty',
        source_world_sha256=digest(world), source_scenario_sha256=digest(scenario),
        effective_world_sha256=digest(w), effective_scenario_sha256=digest(s),
        previous_rigid_profile=world.get('rigid_runtime_profile'), rigid_runtime_profile=PROFILE,
        dt_s=DT, substeps=1, geometry_timeconst_s=.005,
        vessel_map_layout_controller_unchanged=True, native_operations=0,
        material_quantity_drop_delivery='NOT_ASSESSED')
    return w, s, receipt
