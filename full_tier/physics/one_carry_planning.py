"""Owned CPU one-carry recipe and saved path. No native engine import or step."""
from __future__ import annotations
import copy
import json
import math
import struct
from pathlib import Path

from transport_challenge.contracts import ContractError, digest
from transport_challenge.joint_planning import JointLimits, JointTrajectory
from physics.saved_motion_planning import SavedMissionPlan, file_sha, point, require, saved_wait_policy

SCHEMA = 'saved-one-carry-v1'
FIXTURE_SCHEMA = 'cpu-supported-carry-initial-v1'
FIXTURE_SCOPE = 'fresh_declared_INITIAL_not_restored_state'
SIGNATURE = ('arm_0__carry', 'carry', 'arm_0')
MATERIALS = ('beads','process_liquid','sand')


def material_inputs(world,scenario,material,precision=None,runtime_profile=None):
    """Derive a source-bound particle recipe from the selected geometric layout."""
    require(material in MATERIALS,'Select beads, process_liquid or sand explicitly')
    if material=='beads':
        require(precision in (None,'32') and runtime_profile in (None,'default'),
            'The original bead16 profile uses float32')
        require('particle_material_request' not in world,'Do not substitute particles with rigid beads')
        return copy.deepcopy(world),copy.deepcopy(scenario)
    expected_precision='64' if material=='sand' else '32'
    expected_profile='cpic_cache_64_tol12' if material=='sand' else 'default'
    require(precision in (None,expected_precision) and runtime_profile in (None,expected_profile),
        'Preserve liquid32/default or Sand64/cache/tolerance1e-12')
    require('particle_material_request' not in world and 'native_particle_contract' not in world,
        'Material conversion starts from a declared source layout, never a native state')
    w,s=copy.deepcopy(world),copy.deepcopy(scenario)
    require(s['provenance']['world_sha256']==digest(w),'Selected source layout identity differs')
    removed=w.pop('rigid_runtime_profile',None)
    require(removed in (None,'rigid16_bead_tc2h_v1'),'Unknown rigid profile cannot become native particles')
    w['one_carry_material_derivation']=dict(schema_version='selected-layout-particle-recipe-v1',
        source_world_sha256=digest(world),source_scenario_sha256=digest(scenario),
        removed_bead_profile=removed,material_kind=material,precision=expected_precision,
        runtime_profile=expected_profile,geometry_supports_bases_and_source_preserved=True,
        scope='Recipe conversion only. No native state, particles or transport result replayed.')
    s['provenance']['world_sha256']=digest(w)
    from physics.particle_episode_bridge import particle_inputs
    return particle_inputs(w,s,material,precision=expected_precision,runtime_profile=expected_profile)


def native_precision(world):
    request=world.get('particle_material_request')
    return request.get('precision','32') if request is not None else '32'


def native_number(value,world):
    return float(value) if native_precision(world)=='64' else _f32(value)


def unbound_particle_inputs(world,scenario):
    """Remove only the runtime-owned inventory binding for geometric identity."""
    w,s=copy.deepcopy(world),copy.deepcopy(scenario)
    if 'particle_material_request' in w:
        require(s['provenance']['world_sha256']==digest(w),'Actual particle world/scenario binding differs')
        if 'native_particle_contract' in w:
            from physics.particle_episode_bridge import bind_native_contract
            rebound=bind_native_contract(w,s,w['native_particle_contract'])
            require(rebound==(w,s),'Actual native inventory binding is inconsistent')
        w.pop('native_particle_contract',None)
        s['provenance']['world_sha256']=digest(w)
        from physics.particle_episode_bridge import validate_request
        validate_request(w,s)
    return w,s


def _f32(value):
    require(type(value) in (int, float), 'INITIAL coordinates must be real numbers')
    try:return struct.unpack('f', struct.pack('f', float(value)))[0]
    except (OverflowError,struct.error) as exc:
        raise ContractError('INITIAL coordinate exceeds the native float32 range') from exc


def make_initial_fixture(world, scenario, q_rad):
    q = [native_number(v,world) for v in point(q_rad, 7)]
    return dict(schema_version=FIXTURE_SCHEMA, world_sha256=digest(world),
        source_scenario_sha256=digest(scenario), active_arm_id='arm_0',
        support_id=world['source_support']['id'], initial_q_rad=q,
        initial_finger_q_m=[native_number(.04,world)]*2,native_precision=native_precision(world),scope=FIXTURE_SCOPE)


def apply_initial_fixture(world, scenario, fixture):
    expected_keys = {'schema_version','world_sha256','source_scenario_sha256',
        'active_arm_id','support_id','initial_q_rad','initial_finger_q_m','native_precision','scope'}
    require(type(fixture) is dict and set(fixture) == expected_keys,
        'Require the exact CPU-declared one-carry INITIAL recipe')
    require(fixture['schema_version'] == FIXTURE_SCHEMA and fixture['scope'] == FIXTURE_SCOPE
        and fixture['active_arm_id'] == 'arm_0' and fixture['support_id'] == world['source_support']['id']
        and fixture['world_sha256'] == digest(world) and fixture['source_scenario_sha256'] == digest(scenario),
        'One-carry INITIAL source identity differs')
    q = list(point(fixture['initial_q_rad'], 7))
    require(fixture['native_precision']==native_precision(world)
        and q == [native_number(v,world) for v in q]
        and fixture['initial_finger_q_m']==[native_number(.04,world)]*2,
        'One-carry INITIAL differs from its declared native precision/open fingers')
    if 'particle_material_request' in world:
        from physics.particle_episode_bridge import validate_request
        validate_request(world,scenario)
        require(world.get('rigid_runtime_profile') is None,'Particles cannot use bead16 handling')
    else:
        require(world.get('rigid_runtime_profile')=='rigid16_bead_tc2h_v1'
            and world.get('payload_mode')!='empty','Require the original bead16 profile')
    result = copy.deepcopy(scenario)
    require([r['id'] for r in result['robots']] == ['arm_0','arm_1','arm_2']
        and all(r['finger_q_m'] == [.04,.04] for r in result['robots']),
        'Apply the original automatic finger initialization before the one-carry recipe')
    require('one_carry_fixture' not in result.get('provenance', {}),
        'Do not reapply or restore an intermediate one-carry state')
    result['robots'][0]['q_rad'] = list(q)
    result.setdefault('provenance', {})['one_carry_fixture'] = copy.deepcopy(fixture)
    return result


def seal_one_carry_bundle(record):
    value = copy.deepcopy(record)
    value.pop('bundle_sha256', None)
    value['bundle_sha256'] = digest(value)
    return value


class SavedOneCarryPlan(SavedMissionPlan):
    """One real carry with a separately declared fresh INITIAL, never a mission prefix."""
    def __init__(self, record, *, path, require_supervisor=True):
        self.path = Path(path).resolve(strict=True)
        self.file_sha256 = file_sha(self.path)
        value = copy.deepcopy(record)
        require(value.get('schema_version') == SCHEMA and value.get('scope') == 'one_carry'
            and value.get('post_place_policy') == 'one_carry', 'Require explicit one-carry scope/schema')
        require(value.get('bundle_sha256') == digest({k:v for k,v in value.items() if k!='bundle_sha256'}),
            'One-carry bundle digest differs')
        require(value.get('nominal_only') is True and all(value.get(k) is False for k in
            ('physical_grasp_verified','physical_transport_verified','full_mission_passed','student_release_approved')),
            'A CPU one-carry plan cannot claim physical completion')
        require(value.get('payload') in MATERIALS, 'Select an explicit supported material name')
        if value['payload']!='beads':
            from physics.particle_episode_bridge import request as particle_request
            precision='64' if value['payload']=='sand' else '32'
            profile='cpic_cache_64_tol12' if value['payload']=='sand' else 'default'
            require(value.get('particle_material_request')==particle_request(value['payload'],precision,profile),
                'Saved particle recipe differs from its exact supported native model')
            require(value.get('particle_recheck_mode') in ('cpu_geometry_actual_boundary','native_full'),
                'Saved particle static admission mode is missing')
            reference=Path(__file__).resolve().parents[1]/'cases/three_arm_cell_v1/physics_settings_reference.json'
            require(value.get('particle_physics_reference')==dict(path=str(reference),sha256=file_sha(reference))
                and value.get('source_pins',{}).get(str(reference))==file_sha(reference),
                'Particle physics must bind its explicit canonical measured reference')
        summary=value.get('cpu_planning_result', {})
        require(summary.get('status')=='CPU_GEOMETRIC_ONE_CARRY_PLANNED'
            and summary.get('completed_stages')==1 and summary.get('failure') is None
            and summary.get('source_bindings_unchanged') is True
            and summary.get('genesis_imported_modules')==[] and summary.get('genesis_initialized') is False
            and summary.get('blocked_genesis_import_attempts')==[]
            and summary.get('native_physical_steps')==summary.get('cpu_physical_steps')==0,
            'One-carry needs a successful CPU-only planning receipt')
        if value['payload']!='beads':
            supervisor=value.get('cpu_process_supervisor',{})
            if require_supervisor:
                numeric=('timeout_s','requested_wall_limit_s','worker_elapsed_including_cleanup_s',
                    'parent_wall_s_before_publication')
                require(supervisor==summary.get('cpu_process_supervisor')
                    and supervisor.get('schema_version')=='owned-cpu-worker-budget-v1'
                    and supervisor.get('status')=='COMPLETE_WITHIN_BUDGET'
                    and supervisor.get('timed_out') is False and supervisor.get('worker_exit_code')==0
                    and supervisor.get('cleanup_errors')==[] and supervisor.get('source_bindings_verified') is True
                    and supervisor.get('parent_genesis_imported_modules')==[]
                    and supervisor.get('parent_blocked_genesis_import_attempts')==[]
                    and summary.get('blocked_integration_attempts')==[]
                    and all(type(supervisor.get(k)) in (int,float) and math.isfinite(supervisor[k])
                        and supervisor[k]>=0 for k in numeric)
                    and 10<=supervisor['requested_wall_limit_s']<=3600
                    and 0<supervisor['timeout_s']<=supervisor['requested_wall_limit_s']
                    and supervisor['worker_elapsed_including_cleanup_s']<=supervisor['requested_wall_limit_s']
                    and supervisor['worker_elapsed_including_cleanup_s']<=supervisor['parent_wall_s_before_publication']
                    and supervisor['parent_wall_s_before_publication']<supervisor['requested_wall_limit_s']
                    and summary.get('process_wall_s')==value.get('cpu_planning_wall_s')
                        ==supervisor['parent_wall_s_before_publication'],
                    'Particle plan needs a completed owned-process CPU budget receipt')
                identity=supervisor.get('worker_identity',{})
                require(type(identity.get('pid')) is int and identity['pid']>0
                    and type(identity.get('windows_creation_filetime_100ns')) is int
                    and identity['windows_creation_filetime_100ns']>0,
                    'CPU budget receipt lacks the exact owned worker identity')
            else:
                require(supervisor.get('status')=='WORKER_CANDIDATE_PENDING',
                    'Only the trusted parent may admit its unpublished worker candidate')
        self.sha256 = value['bundle_sha256']
        self.post_place_policy = 'one_carry'
        self.stage_signatures = (SIGNATURE,)
        stages = value.get('stages')
        require(type(stages) is list and len(stages)==1
            and tuple(stages[0].get(k) for k in ('stage_id','kind','active_arm_id')) == SIGNATURE,
            'One-carry has exactly one real carry and no fabricated approach')
        require(value.get('post_place_resets') == [] and value.get('post_place_returns') == [],
            'One-carry stops after released rest without a donor return/reset')
        require(type(value.get('source_pins')) is dict and bool(value['source_pins']), 'CPU source pins missing')
        fixture = value.get('initial_fixture', {})
        require(fixture.get('schema_version') == FIXTURE_SCHEMA and fixture.get('scope') == FIXTURE_SCOPE
            and fixture.get('active_arm_id')=='arm_0' and fixture.get('world_sha256')==value.get('world_sha256'),
            'One-carry lacks its declared INITIAL source recipe')
        commands = value.get('initial_commands_by_arm', {})
        require(set(commands)=={'arm_0','arm_1','arm_2'}, 'One-carry INITIAL omits persistent peers')
        for command in commands.values():
            point(command['q_rad'],7);point(command['finger_q_m'],2)
        require(commands['arm_0']['q_rad']==fixture.get('initial_q_rad')
            and commands['arm_0']['finger_q_m']==fixture.get('initial_finger_q_m'),
            'One-carry INITIAL command differs from the CPU recipe')
        stage=stages[0];tr=stage['trajectory']
        trajectory=JointTrajectory(tr['waypoints'],tr['edge_durations_s'],JointLimits(**tr['limits']))
        require(trajectory.to_record()==tr and len(tr['waypoints'])>=2,
            'One-carry path/time is noncanonical or has no physical travel')
        require(tr['waypoints'][0]==commands['arm_0']['q_rad'],
            'One-carry saved first waypoint differs from its exact INITIAL command')
        require(stage['expected_peer_commands_by_arm']=={k:v for k,v in commands.items() if k!='arm_0'},
            'One-carry peer commands differ from INITIAL')
        waits=value.get('provided_wait_policy', {})
        require(all(waits.get(k)==v for k,v in saved_wait_policy().items()), 'Provided wait policy changed')
        offline=value.get('offline_planner', {})
        require(all(type(offline.get(k)) is str and offline[k] for k in
            ('module','source_sha256','plan_path_source_sha256')), 'Student planner identity is missing')
        require(all(part.isidentifier() for part in offline['module'].split('.')), 'Invalid student planner module')
        self._json=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
        self.verify_sources()

    def verify_sources(self):
        pins=self.record()['source_pins']
        local=str(Path(__file__).resolve())
        require(pins.get(local)==file_sha(local),'The executing one-carry contract is not pinned by this CPU export')
        return super().verify_sources()

    def prepare_scenario(self, world, automatic_scenario, *, design_sha256=None, task_manifest_sha256=None):
        scenario=apply_initial_fixture(world, automatic_scenario, self.record()['initial_fixture'])
        expected={r['id']:{'q_rad':[native_number(v,world) for v in r['q_rad']],
            'finger_q_m':[native_number(.04,world)]*2} for r in scenario['robots']}
        require(self.record()['initial_commands_by_arm']==expected,
            'Saved INITIAL arm/finger/peer commands differ from the declared fresh scenario')
        self.verify_bound_inputs(world,scenario,design_sha256=design_sha256,task_manifest_sha256=task_manifest_sha256)
        return scenario

    def verify_bound_inputs(self,world,scenario,*,design_sha256=None,task_manifest_sha256=None):
        value=self.record()
        if value['payload']=='beads':
            return super().verify_bound_inputs(world,scenario,design_sha256=design_sha256,
                task_manifest_sha256=task_manifest_sha256)
        w,s=unbound_particle_inputs(world,scenario)
        require(digest(w)==value['world_sha256'] and digest(s)==value['scenario_sha256'],
            'Selected particle geometry/recipe/INITIAL source differs')
        require(w.get('particle_material_request')==value['particle_material_request']
            and w['particle_material_request']['material_kind']==value['payload']
            and w.get('rigid_runtime_profile') is None,'Particle material or runtime profile differs')
        stage=value['stages'][0]
        require(stage['support_id']==w['source_support']['id']
            and stage.get('destination_support_id')==w['relay_supports'][0]['id'],
            'One-carry particle source/destination support differs')
        if design_sha256 is not None:require(design_sha256==value['design_sha256'],'Selected design differs')
        if task_manifest_sha256 is not None:require(task_manifest_sha256==value['task_manifest_sha256'],'Task manifest differs')
        return True


def initial_pick(driver, dispatcher, state):
    """Perform the actual supplied support/rest/attach gate, with no fake motion."""
    from physics.automatic_manipulation import pick
    require(driver.mode=='one_carry' and driver.index==0 and driver.current is None
        and getattr(driver,'one_carry_initial_pick',None) is None, 'Repeated or out-of-order INITIAL pickup')
    gate=pick(driver,state,dict(stage_id='arm_0__initial_pick',kind='initial_pick',
        active_arm_id='arm_0',support_id=driver.route[0]))
    driver.one_carry_initial_pick=copy.deepcopy(gate)
    return gate


def final_result(driver, observation):
    from physics.automatic_manipulation import validate_observation
    validate_observation(driver,observation)
    require(driver.mode=='one_carry' and driver.done and driver.index==1
        and driver.held is None and driver.automatic_grasp.holder is None,
        'One-carry actual release/completion is missing')
    require([(x['stage_id'],x['kind'],x['active_arm_id']) for x in driver.completed]==[SIGNATURE]
        and len(driver.plans)==len(driver.gates)==len(driver.pauses)==1,
        'One-carry requires its one exact saved motion and original gate')
    planned=driver.plans[0];saved=driver.saved_plan.stage(0)['trajectory'];pause=driver.pauses[0]
    require(planned.get('trajectory_sha256')==digest(saved)
        and planned.get('duration_s')==saved['duration_s']
        and planned.get('waypoint_count')==len(saved['waypoints'])
        and planned.get('motion_source')=='saved_cpu_plan'
        and planned.get('saved_bundle_sha256')==driver.saved_plan.sha256
        and all(planned.get(k)==0 for k in ('runtime_plan_path_invocations','runtime_ik_calls','runtime_rrt_calls'))
        and all(pause.get(k) is True for k in
            ('native_state_and_clock_unchanged','dispatcher_unchanged','source_bindings_verified')),
        'One-carry path/time, zero online planning or unchanged pause evidence differs')
    pickup=getattr(driver,'one_carry_initial_pick',None)
    require(type(pickup) is dict and pickup.get('passed') is True, 'Actual supported INITIAL pick is missing')
    gate=driver.gates[0]['gate']
    require(gate.get('passed') is True and gate.get('support_id')==driver.route[1]
        and gate.get('expected_rows')==gate.get('observed_rows')==200
        and gate.get('load_bearing_interval_s',0)>=.5,
        'Original released support/rest gate is incomplete')
    events=driver.automatic_events
    require([e['event'] for e in events]==['attach','release'], 'One-carry actual native event order differs')
    if driver.particle_mode:
        from physics.particle_student_completion import particle_context,_recorded_bulk_rest
        native=particle_context(driver.world,driver.scenario,observation,driver.initial_supply)
        require(pause.get('particle_pause_comparison',{}).get('passed') is True,
            'Complete actual particle planning-pause preservation is missing')
        _recorded_bulk_rest(gate,observation['simulation_s'])
        require(gate.get('particle_material_contract_sha256')==digest(native),
            'Actual released particle gate differs from the original full native inventory')
        material_receipt=dict(payload=native['material_kind'],native_particle_count=native['particle_count'],
            particle_material_contract_sha256=digest(native),particle_packet_sha256=observation['particle_material']['sha256'],
            original_native_particle_inventory_retained=True,rigid_bead_substitution=False)
    else:
        require(gate.get('bead_bulk_rest',{}).get('observed_rows')==201,'Full original bead rest endpoints missing')
        original={m['id']:(m['entity_idx'],m['mass_kg']) for m in driver.initial_supply}
        materials=observation['materials']
        require(len(original)==len(materials)==len({m['id'] for m in materials})==4
            and all(m['inside_carrier'] is True and original.get(m['id'])==(m['entity_idx'],m['mass_kg']) for m in materials),
            'One-carry lost an original bead identity/mass/containment')
        material_receipt=dict(payload='beads',original_four_beads_retained=True)
    return dict(status='ONE_CARRY_TRANSPORT_SAMPLED_PASS',one_carry_transport_accepted=True,
        scope='Fresh declared supported INITIAL, one exact saved carry, actual native pickup/release/rest. No physical home approach or full three-arm mission.',
        motion_source='saved_cpu_plan',saved_bundle_sha256=driver.saved_plan.sha256,
        initial_pick_gate=pickup,final_observation=copy.deepcopy(observation),student_execution=driver.record(),
        physical_motion_count=1,full_mission_passed=False,current_student_single_batch_mission_passed=False,
        material_delivery_accepted=False,physical_handoff_verified=False,physical_handoff_count=0,
        student_release_approved=False,restored_native_state=False,physical_home_approach_executed=False,
        fresh_initial_fixture=True,
        grasp_policy='automatic',grasp_model='provided_ideal',**material_receipt)

