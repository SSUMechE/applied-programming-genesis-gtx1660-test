"""Pure binding of student layout choices to one fixed three-arm workcell.

The student record contains installation and placement choices, never joint
states, grasp offsets, retreat targets or a route. Native reachability, collision,
grasp and transport validation remain the supplied runner's responsibility.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
import json
import math

from .contracts import ContractError, digest, number, point
from .joint_context import JointBasePlacement


ARM_IDS = ('arm_0', 'arm_1', 'arm_2')
BASE_SURFACE_CLEARANCE_M = .001


def _record(value, keys, label):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ContractError(label+' requires exactly '+', '.join(keys))
    return value


def _name(value, label):
    if type(value) is not str or not value or value.strip() != value:
        raise ContractError(label+' must be a nonempty exact identifier')
    return value


def _interval(value, label):
    lo, hi = point(value, 2)
    if lo >= hi or not math.isfinite(hi-lo):
        raise ContractError(label+' requires a finite increasing interval')
    return lo, hi


def _world_inputs(world):
    if not isinstance(world, Mapping):
        raise ContractError('Supply the instructor-owned fixed world')
    try:
        case = world['case_record']
        if case['schema_version'] != 'joint-map-v1':
            raise ContractError('Three-arm design requires the actual joint-map-v1 case')
        if case['arm_joint_names'] != [f'joint{i}' for i in range(1, 8)]:
            raise ContractError('Three-arm design requires the supplied seven native joints')
        stations = world['support_stations']
        if (not isinstance(stations, list) or len(stations) != 6
                or len({s['id'] for s in stations}) != 6
                or stations != [world['source_support'], *world['relay_supports'],
                                world['goal_support'], *world['quality_supports']]
                or len(world['relay_supports']) != 2 or len(world['quality_supports']) != 2):
            raise ContractError('Require the six existing source/relay/goal/quality supports')
        targets = world['support_targets_by_id']
        if not isinstance(targets, Mapping) or set(targets) != {s['id'] for s in stations}:
            raise ContractError('Every fixed physical support needs its supplied endpoint target')
        for station in stations:
            _record(station, ('id', 'frame', 'units', 'axis_aligned', 'size_m', 'center_world_m'), 'Fixed support')
            if station['frame'] != 'world' or station['units'] != 'm' or station['axis_aligned'] is not True:
                raise ContractError('Supports must remain fixed axis-aligned world boxes')
            if min(point(station['size_m'])) <= 0:
                raise ContractError('Fixed support dimensions must be positive')
            point(station['center_world_m'])
        return case, stations, targets
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError('Malformed supplied three-arm world') from exc


def _payload_offsets(world, target):
    # This uses the actual immutable vessel component boxes, not an adjustable
    # student envelope. A full-model runtime still verifies actual placement.
    from physics.carried_body import VesselModel, WorldPose
    model = VesselModel.from_container(world['vessel_model']['container'])
    if model.geometry_sha256 != world['vessel_model']['geometry_sha256']:
        raise ContractError('Fixed vessel geometry identity differs')
    body = target['body_world']
    if body.get('frame') != 'world':
        raise ContractError('Supplied placement target needs world coordinates')
    pose = WorldPose((0., 0., 0.), body['quaternion_wxyz'])
    vertices = [pose.transform_point(tuple(c+s*sign/2 for c, s, sign in
                 zip(box.center_body_m, box.size_m, signs)))
                for box in model.component_boxes for signs in product((-1., 1.), repeat=3)]
    return tuple((min(v[a] for v in vertices), max(v[a] for v in vertices)) for a in range(3))


def design_space(world):
    """Describe only installation and relay faces present in the actual map.

    Installation rectangles are matched to their named ``terrain_*`` physical
    top plate. The highest containing terrain plate is used when an equipment
    component also lies below that same rectangle. No new surface is added.
    Relay XY limits conservatively contain the complete supplied vessel.
    """
    case, stations, targets = _world_inputs(world)
    try:
        obstacles = case['obstacles']
        if (not isinstance(obstacles, list) or not obstacles
                or len({o['id'] for o in obstacles}) != len(obstacles)):
            raise ContractError('Actual physical obstacle IDs must be distinct')
        bounds = {}
        for obstacle in obstacles:
            lo, hi = point(obstacle['lower_m']), point(obstacle['upper_m'])
            if any(a >= b for a, b in zip(lo, hi)):
                raise ContractError('Actual obstacle bounds must increase')
            bounds[_name(obstacle['id'], 'Physical component')] = (lo, hi)
        mounts = []
        for zone in case['installation_zones']:
            _record(zone, ('x_m', 'y_m', 'yaw_rad'), 'Installation rectangle')
            x = _interval(zone['x_m'], 'Installation X')
            y = _interval(zone['y_m'], 'Installation Y')
            yaw = _interval(zone['yaw_rad'], 'Installation yaw')
            candidates = [(name, lo, hi) for name, (lo, hi) in bounds.items()
                          if name.startswith('terrain_') and hi[2] > 0.
                          and lo[0] <= x[0] <= x[1] <= hi[0]
                          and lo[1] <= y[0] <= y[1] <= hi[1]]
            if not candidates:
                raise ContractError('An installation rectangle lacks its actual physical terrain plate')
            height = max(hi[2] for _, _, hi in candidates)
            selected = [(name, lo, hi) for name, lo, hi in candidates if hi[2] == height]
            if len(selected) != 1:
                raise ContractError('Installation rectangle has ambiguous physical top plates')
            name, lo, hi = selected[0]
            mounts.append({'id': name, 'surface_component_id': name, 'x_m': list(x), 'y_m': list(y),
                           'yaw_rad': list(yaw), 'surface_z_m': height,
                           'base_z_m': height+BASE_SURFACE_CLEARANCE_M})
        if not mounts or len({m['id'] for m in mounts}) != len(mounts):
            raise ContractError('Installation surfaces must have distinct bounded rectangles')
        excluded = {world['source_support']['id'], world['goal_support']['id']}
        relays = []
        for station in stations:
            if station['id'] in excluded:
                continue
            offsets = _payload_offsets(world, targets[station['id']])
            size, center = station['size_m'], station['center_world_m']
            intervals = [[center[a]-size[a]/2-offsets[a][0],
                          center[a]+size[a]/2-offsets[a][1]] for a in (0, 1)]
            if any(lo > hi for lo, hi in intervals):
                raise ContractError('A declared relay face is smaller than the fixed vessel footprint')
            relays.append({'support_id': station['id'], 'x_m': intervals[0], 'y_m': intervals[1],
                           'support_top_z_m': center[2]+size[2]/2,
                           'body_root_z_m': targets[station['id']]['body_world']['position_m'][2]})
        return {'schema_version': 'three-arm-design-space-v1', 'arm_ids': list(ARM_IDS),
                'mount_regions': mounts, 'relay_regions': relays,
                'source_support_id': world['source_support']['id'],
                'goal_support_id': world['goal_support']['id'],
                'base_height_editable': False, 'intermediate_states_supplied': False,
                'scope': 'Structural bounds only. Actual full-model reachability, collision and physics checks remain required.'}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError('Malformed supplied installation/relay geometry') from exc


@dataclass(frozen=True)
class ThreeArmDesign:
    """Detached immutable student design authenticated against one fixed world."""
    _record_json: str = field(repr=False)
    source_world_sha256: str
    _space_json: str = field(repr=False)

    @classmethod
    def from_record(cls, record, world):
        _record(record, ('schema_version', 'arms', 'relays', 'joint_speed_scale'), 'Three-arm design')
        if record['schema_version'] != 'three-arm-design-v1':
            raise ContractError('Use three-arm-design-v1')
        space = design_space(world)
        mounts = {m['id']: m for m in space['mount_regions']}
        regions = {r['support_id']: r for r in space['relay_regions']}
        arms = record['arms']
        if not isinstance(arms, (list, tuple)) or len(arms) != 3:
            raise ContractError('Supply exactly three complete arm installations')
        normalized_arms = []
        for expected, arm in zip(ARM_IDS, arms):
            _record(arm, ('id', 'mount_region', 'base_placement'), 'Arm installation')
            if arm['id'] != expected:
                raise ContractError('List the three distinct arms as arm_0, arm_1, arm_2')
            region = _name(arm['mount_region'], 'Mount region')
            if region not in mounts:
                raise ContractError('Mount region is not an actual permitted physical surface')
            base = _record(arm['base_placement'], ('x_m', 'y_m', 'yaw_rad'), 'Named base placement')
            base = JointBasePlacement(**base)
            mount = mounts[region]
            if any(not lo <= value <= hi for value, (lo, hi) in
                   zip((base.x_m, base.y_m, base.yaw_rad), (mount['x_m'], mount['y_m'], mount['yaw_rad']))):
                raise ContractError('Arm placement leaves its selected installation rectangle')
            normalized_arms.append({'id': expected, 'mount_region': region, 'base_placement': base.to_record()})
        roots = [(a['base_placement']['x_m'], a['base_placement']['y_m'], mounts[a['mount_region']]['base_z_m']) for a in normalized_arms]
        if len(set(roots)) != 3:
            raise ContractError('Distinct arms cannot share the exact same physical base origin')
        relays = record['relays']
        if not isinstance(relays, (list, tuple)) or len(relays) != 2:
            raise ContractError('Select two distinct existing physical relay faces and placement points')
        normalized_relays = []
        for relay in relays:
            _record(relay, ('support_id', 'x_m', 'y_m'), 'Relay placement')
            support = _name(relay['support_id'], 'Relay support')
            if support not in regions:
                raise ContractError('Relay selection must name an existing non-source/non-goal physical support')
            region = regions[support]
            x = number(relay['x_m'], 'Relay X', minimum=region['x_m'][0], maximum=region['x_m'][1])
            y = number(relay['y_m'], 'Relay Y', minimum=region['y_m'][0], maximum=region['y_m'][1])
            normalized_relays.append({'support_id': support, 'x_m': x, 'y_m': y})
        if len({r['support_id'] for r in normalized_relays}) != 2:
            raise ContractError('Both handoffs require distinct physical relay supports')
        speed = number(record['joint_speed_scale'], 'joint_speed_scale', minimum=0., maximum=1.)
        if speed == 0.:
            raise ContractError('joint_speed_scale must be in (0,1]')
        normalized = {'schema_version': 'three-arm-design-v1', 'arms': normalized_arms,
                      'relays': normalized_relays, 'joint_speed_scale': speed}
        return cls(json.dumps(normalized, sort_keys=True, separators=(',', ':'), allow_nan=False),
                   digest(world), json.dumps(space, sort_keys=True, separators=(',', ':'), allow_nan=False))

    def to_record(self):
        return json.loads(self._record_json)

    def space_record(self):
        return json.loads(self._space_json)

    @property
    def sha256(self):
        return digest(self.to_record())

    @property
    def joint_speed_scale(self):
        return self.to_record()['joint_speed_scale']


def bind_three_arm_design(design, world, scenario):
    """Return detached actual scene recipes and an input-binding receipt.

    Physical supports and obstacles never move. Relay selection only changes
    their task roles and the selected body placement XY on those same surfaces.
    Teacher-owned initial arm/finger states are retained as INITIAL conditions,
    not a supplied intermediate motion. Fresh endpoint IK/planning is required.
    """
    if type(design) is not ThreeArmDesign or design.source_world_sha256 != digest(world):
        raise ContractError('Bind the validated design to its exact instructor-owned source world')
    _, stations, _ = _world_inputs(world)
    if (not isinstance(scenario, Mapping) or scenario.get('case') != world['case_record']
            or scenario.get('support_stations') != stations
            or scenario.get('provenance', {}).get('world_sha256') != digest(world)
            or [r.get('id') for r in scenario.get('robots', ())] != list(ARM_IDS)):
        raise ContractError('Supply the matching exact three-arm scenario')
    # Revalidate the immutable record to avoid trusting direct dataclass construction.
    checked = ThreeArmDesign.from_record(design.to_record(), world)
    if checked != design:
        raise ContractError('Three-arm design identity or derived geometry differs')
    data = design.to_record(); space = design.space_record()
    mounts = {m['id']: m for m in space['mount_regions']}
    w = copy.deepcopy(dict(world)); s = copy.deepcopy(dict(scenario))
    for recipe, choice in zip(s['robots'], data['arms']):
        recipe['base_placement'] = copy.deepcopy(choice['base_placement'])
        recipe['base_z_m'] = mounts[choice['mount_region']]['base_z_m']
    by_id = {station['id']: station for station in stations}
    selected = [relay['support_id'] for relay in data['relays']]
    pool = [station['id'] for station in [*world['relay_supports'], *world['quality_supports']]]
    remaining = [name for name in pool if name not in selected]
    w['relay_supports'] = [copy.deepcopy(by_id[name]) for name in selected]
    w['quality_supports'] = [copy.deepcopy(by_id[name]) for name in remaining]
    w['support_stations'] = [w['source_support'], *w['relay_supports'], w['goal_support'], *w['quality_supports']]
    for relay in data['relays']:
        target = w['support_targets_by_id'][relay['support_id']]
        body = target['body_world']['position_m']; hand = target['nominal_hand_endpoint_world']['position_m']
        delta = (relay['x_m']-body[0], relay['y_m']-body[1])
        body[:2] = [relay['x_m'], relay['y_m']]
        hand[:2] = [hand[0]+delta[0], hand[1]+delta[1]]
    w['physical_feasibility_verified'] = False
    s['support_stations'] = copy.deepcopy(w['support_stations'])
    s['provenance']['world_sha256'] = digest(w)
    s['provenance']['student_three_arm_design_sha256'] = design.sha256
    s['provenance']['scope'] = 'Student layout binding. Fresh native endpoint planning, grasp and physical validation required.'
    geometry_preserved = (s['case'] == scenario['case'] == w['case_record']
                          and {v['id']: v for v in w['support_stations']} == by_id)
    if not geometry_preserved:
        raise ContractError('Student design must preserve every actual fixed support and obstacle')
    targets = [w['source_support']['id'], *selected, w['goal_support']['id']]
    receipt = {'schema_version': 'three-arm-design-binding-v1',
               'source_world_sha256': digest(world), 'source_scenario_sha256': digest(scenario),
               'design_sha256': design.sha256, 'bound_world_sha256': digest(w), 'bound_scenario_sha256': digest(s),
               'catalog': space, 'joint_speed_scale': design.joint_speed_scale,
               'arm_native_bases': [{'id': a['id'], 'mount_region': choice['mount_region'],
                                     'position_world_m': [a['base_placement']['x_m'], a['base_placement']['y_m'], a['base_z_m']],
                                     'yaw_rad': a['base_placement']['yaw_rad']} for a, choice in zip(s['robots'], data['arms'])],
               'route_support_ids': targets,
               'route_body_targets': [copy.deepcopy(w['support_targets_by_id'][name]['body_world']) for name in targets],
               'fixed_physical_geometry_preserved': True,
               'initial_teacher_arm_states_preserved': True,
               'student_intermediate_states_supplied': False,
               'native_scene_initialized': False,
               'scope': 'Input binding only. All motion must come from student path callbacks between supplied operation endpoints. No collision, grasp or delivery success implied.'}
    return {'world': w, 'scenario': s, 'joint_speed_scale': design.joint_speed_scale, 'receipt': receipt}
