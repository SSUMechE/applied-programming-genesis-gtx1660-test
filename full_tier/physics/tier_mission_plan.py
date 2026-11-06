"""Source-bound original-INITIAL one/two/three-arm CPU mission plans.

Nominal CPU endpoints and trajectories are not native grasp/handoff receipts.
The native runtime must authenticate each actual boundary and replay without
IK, search, time rescaling, state restoration or trajectory substitution.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path

from transport_challenge.contracts import digest,point
from transport_challenge.joint_planning import JointLimits,JointTrajectory
from physics.saved_motion_planning import (SavedMissionPlan,file_sha,require,
    kinematic_return_schedule,saved_wait_policy)

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='saved-tier-mission-v1'
TASK_SCHEMA='tier-transport-task-v1'
CPU_STATUS='CPU_GEOMETRIC_TIER_MISSION_PLANNED'
COUNTS={'easy':1,'medium':2,'hard':3}
LIMIT_TICKS={'easy':16000,'medium':34000,'hard':42000}
TASK_FIELDS={'schema_version','difficulty','required_arm_ids','support_route',
    'required_handoff_count','post_place_policy','terminal_support_id',
    'terminal_support_target','terminal_support_geometry','source_task_manifest_sha256',
    'source_world_sha256','source_initialization_sha256','declared_initialization_mode',
    'material','dt_s','substeps','clock_start_step','completion_limit_ticks'}


def task_stages(contract):
    return tuple((arm+'__'+kind,kind,arm) for arm in contract['required_arm_ids']
        for kind in ('approach','carry'))


def validate_task_contract(value):
    require(type(value) is dict and set(value)==TASK_FIELDS,'Malformed typed tier task contract')
    tier=value['difficulty'];require(tier in COUNTS,'Unknown tier difficulty')
    n=COUNTS[tier];arms=['arm_'+str(i) for i in range(n)]
    route=['source_support','relay_1_support','relay_2_support','goal_support'][:n+1]
    require(value['schema_version']==TASK_SCHEMA and value['required_arm_ids']==arms
        and value['support_route']==route and type(value['required_handoff_count']) is int and value['required_handoff_count']==n-1,
        'Tier arm/stage/support requirements differ')
    require(value['post_place_policy']=='kinematic_return'
        and value['declared_initialization_mode']=='original_task_initial'
        and value['material']=='beads' and value['dt_s']==.0025
        and type(value['substeps']) is int and value['substeps']==16
        and type(value['clock_start_step']) is int and value['clock_start_step']==200
        and type(value['completion_limit_ticks']) is int
        and value['completion_limit_ticks']==LIMIT_TICKS[tier],
        'Tier initialization, material, physics, return or clock policy differs')
    require(value['terminal_support_id']==route[-1],'Tier terminal is not the fixed task destination')
    for name in ('source_task_manifest_sha256','source_world_sha256','source_initialization_sha256'):
        v=value[name];require(type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v),'Malformed tier input digest')
    target=value['terminal_support_target'];station=value['terminal_support_geometry']
    require(type(target) is dict and target.get('support_id',route[-1])==route[-1]
        and type(station) is dict and station.get('id')==route[-1],
        'Tier immutable terminal identity differs')
    require(target['body_world'].get('frame')=='world','Tier terminal body target must be world-frame')
    point(target['body_world']['position_m'],3);point(target['body_world']['quaternion_wxyz'],4)
    point(station['center_world_m'],3)
    require(all(x>0 for x in point(station['size_m'],3)),'Tier terminal size must be positive')
    return copy.deepcopy(value)


def make_task_contract(original_world,original_scenario,*,difficulty,task_manifest_sha256):
    require(difficulty in COUNTS,'Select easy, medium or hard explicitly')
    require(original_world['case_record']==original_scenario['case'],'Tier source world/scenario disagree')
    n=COUNTS[difficulty]
    route=[original_world['source_support']['id'],*[s['id'] for s in original_world['relay_supports']],original_world['goal_support']['id']][:n+1]
    terminal=route[-1]
    record=dict(schema_version=TASK_SCHEMA,difficulty=difficulty,
        required_arm_ids=['arm_'+str(i) for i in range(n)],support_route=route,
        required_handoff_count=n-1,post_place_policy='kinematic_return',terminal_support_id=terminal,
        terminal_support_target=copy.deepcopy(original_world['support_targets_by_id'][terminal]),
        terminal_support_geometry=copy.deepcopy(next(s for s in original_world['support_stations'] if s['id']==terminal)),
        source_task_manifest_sha256=task_manifest_sha256,source_world_sha256=digest(original_world),
        source_initialization_sha256=digest(original_scenario),declared_initialization_mode='original_task_initial',
        material='beads',dt_s=.0025,substeps=16,clock_start_step=200,completion_limit_ticks=LIMIT_TICKS[difficulty])
    return validate_task_contract(record)


def verify_task_design(contract,world,scenario):
    contract=validate_task_contract(contract)
    terminal=contract['terminal_support_id']
    require(world.get('rigid_runtime_profile')=='rigid16_bead_tc2h_v1'
        and world.get('payload_mode')!='empty' and 'particle_material_request' not in world,
        'Tier replay preserves the original four-bead rigid16 profile')
    require('one_carry_fixture' not in scenario.get('provenance',{}),
        'Full tier begins at original INITIAL, not a grasp-ready one-carry fixture')
    require([r['id'] for r in scenario['robots']]==['arm_0','arm_1','arm_2'],
        'Original INITIAL requires all three declared robot identities')
    require(world['support_targets_by_id'][terminal]==contract['terminal_support_target']
        and next(s for s in world['support_stations'] if s['id']==terminal)==contract['terminal_support_geometry']
        and scenario['supported_body_poses'][terminal]==contract['terminal_support_target']['body_world'],
        'Student design changed the immutable tier final support or body target')
    return True


def donor_return_schedule(stages,initial_commands,contract):
    contract=validate_task_contract(contract)
    donors=set(contract['required_arm_ids'][:-1])
    return [r for r in kinematic_return_schedule(stages,initial_commands) if r['arm_id'] in donors]


def seal_tier_bundle(record):
    result=copy.deepcopy(record);result.pop('bundle_sha256',None)
    result['bundle_sha256']=digest(result)
    return result


class TierMissionPlan(SavedMissionPlan):
    """Strict separate tier schema, inheriting only generic saved-plan methods."""
    def __init__(self,record,*,path,require_supervisor=True):
        self.path=Path(path).resolve(strict=True);self.file_sha256=file_sha(self.path)
        value=copy.deepcopy(record)
        require(value.get('schema_version')==SCHEMA and value.get('scope')=='tier_mission',
            'Require an explicit saved original-INITIAL tier mission')
        require(value.get('bundle_sha256')==digest({k:v for k,v in value.items() if k!='bundle_sha256'}),'Tier bundle digest differs')
        contract=validate_task_contract(value.get('task_contract'))
        require(value.get('task_contract_sha256')==digest(contract),'Tier task contract changed')
        self.sha256=value['bundle_sha256'];self.post_place_policy='kinematic_return'
        self.stage_signatures=task_stages(contract)
        require(value.get('post_place_policy')==self.post_place_policy and value.get('payload')=='beads'
            and value.get('rigid_runtime_profile')=='rigid16_bead_tc2h_v1','Tier recipe/post-place policy differs')
        require(value.get('nominal_only') is True and all(value.get(k) is False for k in
            ('physical_grasp_verified','physical_transport_verified','full_mission_passed','student_release_approved')),
            'CPU tier planning cannot claim native acquisition/handoff/completion')
        require(type(value.get('source_pins')) is dict and bool(value['source_pins']),'Tier CPU source closure missing')
        require(value.get('provided_wait_policy') is not None and all(value['provided_wait_policy'].get(k)==v for k,v in saved_wait_policy().items()),'Tier changed provided waits')
        offline=value.get('offline_planner',{})
        require(all(type(offline.get(k)) is str and offline[k] for k in ('module','source_sha256','plan_path_source_sha256'))
            and all(p.isidentifier() for p in offline['module'].split('.')),'Selected tier planner identity missing')
        stages=value.get('stages')
        require(type(stages) is list and [(s.get('stage_id'),s.get('kind'),s.get('active_arm_id')) for s in stages]==list(self.stage_signatures),
            'Tier requires every exact ordered approach/carry stage')
        initial=value.get('initial_commands_by_arm',{})
        require(set(initial)=={'arm_0','arm_1','arm_2'},'Tier INITIAL commands omit a real robot')
        for command in initial.values():point(command['q_rad'],7);point(command['finger_q_m'],2)
        commands=copy.deepcopy(initial);limit_record=None
        for stage in stages:
            arm=stage['active_arm_id'];i=int(arm[-1]);kind=stage['kind'];tr=stage['trajectory']
            limits=JointLimits(**tr['limits']);trajectory=JointTrajectory(tr['waypoints'],tr['edge_durations_s'],limits)
            require(trajectory.to_record()==tr and tr['waypoints'][0]==commands[arm]['q_rad'],'Tier trajectory/time/preceding endpoint changed')
            require(stage['expected_peer_commands_by_arm']=={n:c for n,c in commands.items() if n!=arm},'Tier preceding peer commands changed')
            require(limit_record is None or limit_record==tr['limits'],'Native tier limits change between stages');limit_record=tr['limits']
            require(stage['support_id']==contract['support_route'][i]
                and stage.get('destination_support_id')==(contract['support_route'][i+1] if kind=='carry' else None),
                'Tier stage source/destination differs')
            commands[arm]['q_rad']=copy.deepcopy(tr['waypoints'][-1])
            if kind=='carry' and arm in contract['required_arm_ids'][:-1]:commands[arm]=copy.deepcopy(initial[arm])
        require(value.get('post_place_resets')==[],'Tier cannot use instant resets')
        require(value.get('post_place_returns')==donor_return_schedule(stages,initial,contract),
            'Tier donor returns must reverse own exact paths/times, with no final return')
        require(len(value['post_place_returns'])==len(contract['required_arm_ids'])-1,'Tier donor return count differs')
        summary=value.get('cpu_planning_result',{})
        require(summary.get('status')==CPU_STATUS and summary.get('planning_scope')=='tier-mission'
            and summary.get('completed_stages')==len(stages) and summary.get('failure') is None
            and summary.get('source_bindings_unchanged') is True
            and summary.get('genesis_initialized') is False and summary.get('genesis_imported_modules')==[]
            and summary.get('blocked_genesis_import_attempts')==[] and summary.get('blocked_integration_attempts')==[]
            and summary.get('native_physical_steps')==summary.get('cpu_physical_steps')==0,
            'Full tier requires successful CPU-only planning with every stage')
        supervisor=value.get('cpu_process_supervisor',{})
        if require_supervisor:
            require(supervisor==summary.get('cpu_process_supervisor')
                and supervisor.get('schema_version')=='owned-cpu-worker-budget-v1'
                and supervisor.get('status')=='COMPLETE_WITHIN_BUDGET' and supervisor.get('timed_out') is False
                and supervisor.get('worker_exit_code')==0 and supervisor.get('cleanup_errors')==[]
                and supervisor.get('source_bindings_verified') is True
                and supervisor.get('parent_genesis_imported_modules')==[]
                and supervisor.get('parent_blocked_genesis_import_attempts')==[],
                'Full tier needs completed owned-process CPU publication')
            cost=summary.get('process_wall_s');limit=supervisor.get('requested_wall_limit_s')
            require(type(cost) in (int,float) and math.isfinite(cost) and type(limit) in (int,float)
                and 0<cost<limit<=3600 and limit>=10 and cost==value.get('cpu_planning_wall_s')==supervisor.get('parent_wall_s_before_publication'),
                'Tier CPU attributed wall budget is inconsistent')
            identity=supervisor.get('worker_identity',{})
            require(type(identity.get('pid')) is int and identity['pid']>0
                and type(identity.get('windows_creation_filetime_100ns')) is int and identity['windows_creation_filetime_100ns']>0,
                'Tier CPU receipt lacks its exact owned worker identity')
            elapsed=supervisor.get('worker_elapsed_including_cleanup_s');allowance=supervisor.get('timeout_s')
            require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=cost
                and type(allowance) in (int,float) and math.isfinite(allowance) and 0<allowance<=limit,
                'Tier CPU worker elapsed/allowance differs from its parent budget')
        else:require(supervisor.get('status')=='WORKER_CANDIDATE_PENDING','Only trusted CPU parent may admit a worker tier candidate')
        self._json=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
        self.verify_sources()

    @property
    def required_arm_ids(self):return tuple(self.record()['task_contract']['required_arm_ids'])
    @property
    def support_route(self):return tuple(self.record()['task_contract']['support_route'])
    @property
    def task_contract(self):return self.record()['task_contract']

    def verify_bound_inputs(self,world,scenario,*,design_sha256=None,task_manifest_sha256=None):
        value=self.record()
        require(value['world_sha256']==digest(world) and value['scenario_sha256']==digest(scenario),'Tier bound world/INITIAL differs')
        verify_task_design(self.task_contract,world,scenario)
        if design_sha256 is not None:require(value['design_sha256']==design_sha256,'Tier student design differs')
        if task_manifest_sha256 is not None:require(value['task_manifest_sha256']==task_manifest_sha256==self.task_contract['source_task_manifest_sha256'],'Tier task manifest differs')
        for stage in value['stages']:
            if stage['kind']=='carry':
                require(stage['nominal_goal_body_world']==scenario['supported_body_poses'][stage['destination_support_id']],
                    'Tier nominal destination does not match bound scenario')
        return True
