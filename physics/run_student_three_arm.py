"""Development entry: student decisions, provided grasp/place, real three arms.

Use the existing course environment. No package installation occurs. The default
step bound is short. A complete mission is wired but not certified by a prefix.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import time
import traceback

from transport_challenge.contracts import ContractError,digest
from transport_challenge.three_arm_design import ThreeArmDesign,bind_three_arm_design
from physics.student_episode_driver import StudentEpisodeDriver
from physics.source_guard import PinnedSourceGuard

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def finalize_single_batch_result(result,*,mode,source_bindings_unchanged):
    """Fail closed after cleanup. This cannot create physical completion proof."""

    if mode=='one_carry' and result.get('execution_backend')=='device':
        from physics.device_one_carry import validate_device_completion
        healthy=(source_bindings_unchanged is True and not result.get('cleanup_errors') and not result.get('failure')
            and result.get('status')=='ONE_CARRY_DEVICE_TRANSPORT_SAMPLED_PASS'
            and result.get('one_carry_transport_accepted') is True
            and result.get('completion_scope')=='provided_device_one_carry_strict_handling_v1')
        try:
            if healthy:
                healthy=validate_device_completion(result.get('device_execution'))
                h=result['device_execution']['handling']
                if (h['prefix_guard']!=result.get('legacy_prefix_handling_enforcement',{}).get('guard')
                        or h['body_root_envelope']!=result.get('legacy_prefix_handling_enforcement',{}).get('body_root_envelope')
                        or h['native_reach_envelopes']!=result.get('native_reach_envelopes')):
                    raise ContractError('Device handling seed differs from the actual original prefix owner')
        except Exception as exc:
            healthy=False
            result.setdefault('failure',{'type':type(exc).__name__,'message':str(exc)})
        result.update(one_carry_transport_accepted=bool(healthy),full_mission_passed=False,
            current_student_single_batch_mission_passed=False,student_release_approved=False)
        if not healthy:result['status']='STUDENT_RUN_FAILED'
        return result

    completion=result.get('mission_completion_receipt')
    handling=result.get('handling_enforcement',{})
    guard=handling.get('guard') if isinstance(handling,dict) else None
    reach=result.get('native_reach_envelopes')
    steps=result.get('physical_steps')
    rigid_profile=result.get('rigid_runtime_profile')
    named16=(isinstance(handling,dict) and rigid_profile=='rigid16_bead_tc2h_v1'
        and handling.get('runtime_profile')==rigid_profile
        and handling.get('guard_dt_s')==.0025/16
        and handling.get('guard_scene_step_scope')=='actual_internal_global_substep')
    expected_observations=(16*steps if named16 and type(steps) is int else steps)
    guard_ok=(type(steps) is int and steps>0 and isinstance(guard,dict) and guard.get('passed') is True
        and (rigid_profile is None or named16)
        and type(guard.get('observations_accepted')) is int and guard['observations_accepted']==expected_observations)
    reach_ok=(isinstance(reach,dict) and reach.get('passed') is True
        and isinstance(reach.get('by_arm'),dict) and set(reach['by_arm'])=={'arm_0','arm_1','arm_2'})
    scoped=(mode=='mission' and result.get('status')=='STUDENT_THREE_ARM_SINGLE_BATCH_SAMPLED_PASS'
        and result.get('current_student_single_batch_mission_passed') is True
        and isinstance(completion,dict) and completion.get('passed') is True
        and ((result.get('grasp_policy')=='automatic' and result.get('grasp_model')=='provided_ideal'
              and completion.get('schema_version')=='current-student-automatic-single-batch-completion-v1'
              and result.get('completion_scope')=='current_student_three_arm_automatic_single_batch_v1')
             or (result.get('grasp_policy','friction')=='friction'
                 and completion.get('schema_version')=='current-student-single-batch-completion-v1'
                 and result.get('completion_scope')=='current_student_three_arm_single_batch_v1'))
        and guard_ok and reach_ok)
    empty_scoped=(mode=='mission' and result.get('status')=='EMPTY_THREE_ARM_RRT_MISSION_SAMPLED_PASS'
        and result.get('payload_mode')=='empty' and result.get('empty_container_mission_passed') is True
        and result.get('grasp_policy')=='automatic' and result.get('grasp_model')=='provided_ideal'
        and isinstance(completion,dict) and completion.get('passed') is True
        and completion.get('schema_version')=='current-student-empty-automatic-mission-completion-v1'
        and result.get('completion_scope')=='current_student_three_arm_empty_automatic_mission_v1'
        and result.get('material_delivery_accepted') is False and guard_ok and reach_ok)
    scoped=scoped or empty_scoped
    one_carry_scoped=(mode=='one_carry' and result.get('status')=='ONE_CARRY_TRANSPORT_SAMPLED_PASS'
        and result.get('one_carry_transport_accepted') is True
        and result.get('physical_motion_count')==1 and result.get('motion_source')=='saved_cpu_plan'
        and guard_ok and reach_ok)
    failed=(source_bindings_unchanged is not True or bool(result.get('cleanup_errors')) or bool(result.get('failure')))
    if result.get('status') in ('STUDENT_THREE_ARM_SINGLE_BATCH_SAMPLED_PASS','EMPTY_THREE_ARM_RRT_MISSION_SAMPLED_PASS') and not scoped:
        failed=True
        result.setdefault('failure',{'type':'ContractError','message':'Missing current student single-batch completion evidence'})
    if result.get('status')=='ONE_CARRY_TRANSPORT_SAMPLED_PASS' and not one_carry_scoped:
        failed=True
        result.setdefault('failure',{'type':'ContractError','message':'Missing complete one-carry handling/reach evidence'})
    result['full_mission_passed']=bool(scoped and not failed)
    result['current_student_single_batch_mission_passed']=bool(scoped and not failed)
    if scoped and not failed:
        completion.update(accepted_physical_steps=steps,handling_guard_sha256=digest(guard),
            native_reach_envelopes_sha256=digest(reach))
    if failed:
        result.update(status='STUDENT_RUN_FAILED',material_delivery_accepted=False,physical_handoff_verified=False)
        if 'one_carry_transport_accepted' in result:result['one_carry_transport_accepted']=False
        if 'empty_container_mission_passed' in result:result['empty_container_mission_passed']=False
        if isinstance(completion,dict):
            completion['pre_cleanup_predicate_passed']=completion.get('passed') is True
            completion['passed']=False
            completion['revoked_by_execution_cleanup_or_source_failure']=True
            completion.setdefault('revoked_reason','Execution, source, cleanup, handling or reach evidence failed')
        for key in ('delivered_material_ids','delivered_count','delivered_mass_kg','physical_handoff_count'):
            result.pop(key,None)
    return result


def prepare_pick_fixture(gs,world,scenario,output,*,attempts,seed,active_arm_id='arm_0',empty_mission_initial=False):
    """QA unit INITIAL condition, resolved by supplied IK, never a mission node.

    The isolated query has no live parent. No solved approach path is generated.
    The fresh scene starts near a grasp endpoint to bound supplied grasp cost.
    Explicit empty missions use this as their provided source initialization.
    """
    from physics.carried_body import VesselModel,WorldPose
    from physics.fixed_world_task import FixedSupportStation
    from physics.episode_joint_scene import EpisodeJointScene
    from physics.provided_manipulation import grasp_endpoint_candidates,solve_endpoint
    from transport_challenge.joint_kinematics import BoundedKinematics
    s=copy.deepcopy(scenario);model=VesselModel.from_container(s['container']);query=None
    if active_arm_id not in ('arm_0','arm_1'):raise ContractError('Private fixture actor is arm_0 or arm_1')
    robot=next(r for r in s['robots'] if r['id']==active_arm_id)
    try:
        query=EpisodeJointScene(s['case'],active_arm_id=active_arm_id,robots=s['robots'],model=model,
            body_pose=WorldPose(s['body_pose']['position_m'],s['body_pose']['quaternion_wxyz']),body_id=s['body_id'],
            asset_directory=output/'fixture_endpoint_assets',support_stations=tuple(FixedSupportStation(**{'station_id':r['id'],
                'size_m':r['size_m'],'center_world_m':r['center_world_m']}) for r in s['support_stations']),
            contents=s['contents'],render=False)
        kin=BoundedKinematics(query,query.bounds,s['case']['tool_workspace_m'],max_fk_calls=attempts,max_ik_calls=attempts)
        endpoint=solve_endpoint(kin,query,robot['q_rad'],grasp_endpoint_candidates(model,s['body_pose']),attempts=attempts,seed=seed)
        if endpoint['status']!='ENDPOINT_FOUND':raise ContractError('Unit fixture has no provided valid grasp endpoint')
        q=list(endpoint['q_rad']);q[4]+=.003
        if not query.state_valid(q):raise ContractError('Declared near-endpoint unit initial state is invalid')
        robot['q_rad']=q
        s['provenance']['unit_fixture_scope']=('Provided grasp-ready source INITIAL for the empty-container full mission. No transit pose or post-INITIAL setter.'
            if empty_mission_initial else 'Fresh INITIAL near-grasp unit state only. Not a normal material-mission initialization or intermediate route node.')
        s['provenance']['provided_endpoint_receipt_sha256']=endpoint['receipt_sha256']
        save(output/'unit_fixture_endpoint.json',{'provided_endpoint':endpoint,'initial_q_rad':q,
            'physical_steps':0,'fixture_only':not empty_mission_initial,'mission_initialization':empty_mission_initial,'active_arm_id':active_arm_id})
        return s
    finally:
        if query is not None:query.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,default=ROOT/'example/design.json')
    parser.add_argument('--task',type=Path,default=ROOT/'cases/three_arm_cell_v1')
    parser.add_argument('--strategy',default='student.joint_strategy')
    parser.add_argument('--saved-plan',type=Path,help='Execute exact stored transport paths and the declared post-placement policy without runtime IK or RRT. Kinematic returns reverse those paths without advancing physics.')
    parser.add_argument('--saved-carry',type=Path,help='One saved carry from its CPU-declared fresh supported INITIAL. No physical home approach, native IK or full-mission claim.')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('mission','one_carry','short_motions','pick_place_unit','bead_lift_unit','first_handoff','placement_probe','empty_path'),default='mission')
    parser.add_argument('--contents',choices=('beads',),default='beads')
    parser.add_argument('--joint-path',type=Path,help='Empty mode: student seven-joint waypoints and exact per-edge seconds JSON')
    parser.add_argument('--placement-policy',choices=('fixed','measured'),default=None,
        help='Private placement_probe A/B only. Normal missions always use provided measured place.')
    parser.add_argument('--grasp-policy',choices=('automatic','friction'),default=None,
        help='Provided automatic grasp by default. Friction is retained for historical diagnostics. Motion-only probes default to friction without a grasp.')
    parser.add_argument('--images',action='store_true')
    parser.add_argument('--video',action='store_true',help='Record the complete short empty path with a fixed native camera')
    parser.add_argument('--query-fixture-readback',choices=('scalar','batched'),default='scalar')
    parser.add_argument('--rigid16-readback',choices=('legacy','consolidated','verify','native','native-verify'),default='legacy',
        help='Explicit CUDA phase packet mode. verify adds same-phase original-getter parity reads.')
    parser.add_argument('--trace-mode',choices=('full','compact'),default='full',
        help='Compact rigid16 traces retain measured states and write full recent/pending evidence on failure.')
    parser.add_argument('--execution-backend',choices=('legacy','device'),default='legacy')
    parser.add_argument('--single-active-arm',choices=('arm_0','arm_1','arm_2'),help='Explicit private one-carry dynamics with other original robots retained as qualified fixed collision geometry')
    parser.add_argument('--fixed-peer-reference',type=Path,help='Source-bound original dynamic-arm INITIAL geometry reference')
    parser.add_argument('--step-limit',type=int,default=None)
    parser.add_argument('--wall-limit-s',type=float,default=300.)
    parser.add_argument('--max-queries',type=int,default=25000)
    parser.add_argument('--endpoint-attempts',type=int,default=32)
    parser.add_argument('--seed',type=int,default=71)
    parser.add_argument('--edge-timing-hook',action='store_true')
    args=parser.parse_args()
    if (args.mode=='one_carry')!=(args.saved_carry is not None) or args.saved_carry is not None and args.saved_plan is not None:
        parser.error('one_carry requires --saved-carry only. Full mission --saved-plan remains separate.')
    if args.saved_carry is not None:args.saved_plan=args.saved_carry
    particle_mode=args.contents in ('process_liquid','sand')
    if ((args.single_active_arm is None)!=(args.fixed_peer_reference is None)
            or args.single_active_arm is not None and (args.mode!='one_carry' or args.contents!='beads' or args.execution_backend!='device')):
        parser.error('Single active mode requires saved bead one_carry/device and its explicit native fixed-peer reference')
    if args.execution_backend=='device' and (particle_mode or args.mode!='one_carry' or args.saved_plan is None
            or args.images or args.video or args.rigid16_readback!='native' or args.trace_mode!='compact'
            or args.step_limit not in (None,0)):
        parser.error('Device backend is explicit headless saved bead one_carry/native/compact with no short cutoff')
    if particle_mode and args.mode!='one_carry':
        parser.error('This saved particle adapter is explicit one_carry only')
    if args.contents!='beads' and (args.rigid16_readback!='legacy' or args.trace_mode!='full'):
        parser.error('Consolidated CUDA readback and compact traces require the four-bead rigid16 task')
    if args.saved_plan is not None and (args.mode not in ('mission','one_carry')
            or not (args.contents=='beads' or particle_mode and args.mode=='one_carry')
            or args.edge_timing_hook or args.strategy!='student.joint_strategy' or args.grasp_policy=='friction'):
        parser.error('--saved-plan requires the automatic four-bead full mission, with no online strategy or timing hook')
    empty_mode=args.contents=='empty'
    if args.video and (not empty_mode or args.images):
        parser.error('--video requires --contents empty and cannot combine with --images')
    empty_path=empty_mode and args.mode=='empty_path'
    if (empty_mode and args.mode not in ('empty_path','mission') or args.mode=='empty_path' and not empty_mode
            or (args.joint_path is not None)!=empty_path):
        parser.error('Empty mission uses --contents empty --mode mission; empty_path additionally requires --joint-path')
    if empty_mode and args.grasp_policy=='friction':
        parser.error('Empty execution uses the provided automatic grasp')
    if empty_path and (args.edge_timing_hook or args.strategy!='student.joint_strategy'):
        parser.error('Empty path uses exact input timing and provided automatic grasp, with no strategy/timing hook')
    if args.step_limit is None:args.step_limit=0 if empty_mode or args.saved_plan is not None else 400
    if args.placement_policy is not None and args.mode!='placement_probe':
        parser.error('--placement-policy is private placement_probe QA only')
    if args.mode=='placement_probe' and not args.task.resolve().is_relative_to(ROOT/'qa/physics_stabilization_20261030'):
        parser.error('placement_probe requires a private explicit QA task, not the released mission task')
    if args.images and args.mode not in ('mission','first_handoff','placement_probe','bead_lift_unit','empty_path'):
        parser.error('Sparse checkpoint images support mission, first_handoff, placement/lift and empty-path modes only')
    if not 0<=args.step_limit<=500000 or not 10<=args.wall_limit_s<=14400:
        parser.error('Invalid explicit step/wall bound')
    grasp_policy=args.grasp_policy or ('friction' if args.mode=='short_motions' else 'automatic')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();gs=None;result={'status':'STUDENT_RUN_FAILED','full_mission_passed':False,'student_release_approved':False}
    paths={};snapshots={};source_errors=[];source_guard=None;setup_source_receipt=None;saved_plan=None
    def matches():
        try:
            if source_guard is not None:return source_guard()
            return all(sha(p)==h for p,h in {**paths,**snapshots}.items())
        except OSError as exc:
            source_errors.append({'type':type(exc).__name__,'message':str(exc)})
            return False
    try:
        task=args.task.resolve();manifest=read(task/'manifest.json')
        required={task/n for n in ('world.json','initialization.json','physics_settings_reference.json')}
        if (manifest.get('schema_version')!='provided-three-arm-task-inputs-v1'
                or not isinstance(manifest.get('files'),dict)
                or { (ROOT/n).resolve() for n in manifest['files']}!=required):
            raise ContractError('Task manifest must bind exactly the three provided physical recipe/calibration files')
        for name,h in manifest['files'].items():
            path=ROOT/name
            if not path.resolve().is_relative_to(task) or sha(path)!=h:raise ContractError('Provided task file changed: '+name)
        world=read(task/'world.json');scenario=read(task/'initialization.json');reference=read(task/'physics_settings_reference.json')
        design=ThreeArmDesign.from_record(read(args.design),world)
        bound=bind_three_arm_design(design,world,scenario);world,scenario=bound['world'],bound['scenario']
        if particle_mode:
            from physics.automatic_runtime_binding import prepare_automatic_scenario
            from physics.one_carry_planning import material_inputs
            scenario,_=prepare_automatic_scenario(scenario)
            world,scenario=material_inputs(world,scenario,args.contents)
            reference_path=ROOT/'cases/three_arm_cell_v1/physics_settings_reference.json'
            reference=read(reference_path)
        if args.saved_plan is not None:
            from physics.saved_motion_planning import SavedMissionPlan
            from physics.automatic_runtime_binding import prepare_automatic_scenario
            if args.mode=='one_carry':
                from physics.one_carry_planning import SavedOneCarryPlan
                saved_plan=SavedOneCarryPlan.from_file(args.saved_plan)
            else:saved_plan=SavedMissionPlan.from_file(args.saved_plan)
            effective_scenario,_=prepare_automatic_scenario(scenario)
            if args.mode=='one_carry':
                scenario=saved_plan.prepare_scenario(world,effective_scenario,design_sha256=design.sha256,
                    task_manifest_sha256=sha(task/'manifest.json'))
                effective_scenario=scenario
            saved_plan.verify_bound_inputs(world,effective_scenario,design_sha256=design.sha256,
                task_manifest_sha256=sha(task/'manifest.json'))
        empty_program=None;empty_overlay=None
        if empty_mode:
            from transport_challenge.empty_container_program import load_empty_program,empty_container_overlay
            world,scenario,empty_overlay=empty_container_overlay(world,scenario)
            empty_overlay.update(design_joint_speed_scale=bound['joint_speed_scale'],design_speed_scale_applied=not empty_path,
                timing_source='Exact edge_durations_s from student JSON; no design speed scaling' if empty_path
                    else 'RRT-generated seven-joint paths with provided parameterization and selected design speed scale')
        if empty_path:
            empty_program=load_empty_program(args.joint_path)
            strategy=None;strategy_file=None
        elif saved_plan is not None:
            strategy=None
            strategy_file=ROOT.joinpath(*saved_plan.offline_planner['module'].split('.')).with_suffix('.py')
            if sha(strategy_file)!=saved_plan.offline_planner['source_sha256']:
                raise ContractError('Saved offline planner source differs from its exact file')
        else:
            strategy=importlib.import_module(args.strategy)
            strategy_file=Path(inspect.getsourcefile(strategy)).resolve()
        sources=[*sorted((ROOT/'physics').glob('*.py')),*sorted((ROOT/'transport_challenge').glob('*.py')),
            *sorted((ROOT/'student').glob('*.py')),*sorted((ROOT/'supplied_support').glob('*.py'))]
        if strategy_file is not None:sources.append(strategy_file)
        sources+=list(task.glob('*.json'))+[args.design.resolve()]
        if args.fixed_peer_reference is not None:sources.append(args.fixed_peer_reference.resolve(strict=True))
        if saved_plan is not None:
            sources += [saved_plan.path, *[Path(name) for name in saved_plan.record()['source_pins']]]
        if particle_mode:
            from physics.particle_episode_bridge import provider_source_pins
            particle_pins=provider_source_pins(args.contents,world['particle_material_request'].get('runtime_profile','default'))
            sources.extend(Path(name) for name in particle_pins)
            sources.append(reference_path)
        empty_helper_pins={}
        if empty_mode:
            from physics.episode_runtime import empty_runtime_source_paths
            if empty_path:sources.append(args.joint_path.resolve())
            empty_helper_pins=empty_runtime_source_paths()
            sources.extend(Path(name) for name in empty_helper_pins)
        paths={str(p.resolve()):sha(p) for p in sources}
        if saved_plan is not None:
            expected_saved={**saved_plan.record()['source_pins'],str(saved_plan.path):saved_plan.file_sha256}
            for name,expected in expected_saved.items():
                resolved=str(Path(name).resolve())
                if paths.get(resolved)!=expected:
                    raise ContractError('Saved CPU source/input changed before native source binding: '+resolved)
                paths[resolved]=expected
        if any(paths[str(Path(name).resolve())]!=h for name,h in empty_helper_pins.items()):
            raise ContractError('Empty runtime helper changed during source binding')
        for p in sources:
            if p.suffix=='.py':
                relative=p.resolve().relative_to(ROOT) if p.resolve().is_relative_to(ROOT) else Path('external_strategy')/p.name
                target=output/'code'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(p.read_bytes());snapshots[str(target)]=sha(target)
        save(output/'sources.json',paths);save(output/'source_snapshots.json',snapshots)
        save(output/'selected_design.json',design.to_record());save(output/'design_binding.json',bound['receipt'])
        source_guard=PinnedSourceGuard({**paths,**snapshots},full_interval_s=1.)
        if not source_guard.before_request():raise ContractError('Full setup source verification failed')
        if particle_mode:
            from physics.episode_runtime import validate_inputs
            from physics.student_episode_driver import panda_limits
            validate_inputs(world,scenario,None,reference,student_limits=panda_limits(scenario['case'],
                precision=world['particle_material_request'].get('precision','32')))
        from physics.joint_scene import initialize_genesis
        if particle_mode and world['particle_material_request'].get('precision','32')=='64':
            from importlib.metadata import version
            versions={name:version(name) for name in ('torch','genesis-world')}
            if versions!={'torch':'2.14.0+cu126','genesis-world':'1.3.2'}:
                raise ContractError('Existing particle CUDA environment version differs')
            import genesis as gs
            gs.init(backend=gs.cuda,precision='64',logging_level='warning',seed=args.seed)
        else:gs,versions=initialize_genesis(args.seed)
        if args.mode in ('pick_place_unit','placement_probe','bead_lift_unit','empty_path') or empty_mode and args.mode=='mission':
            scenario=prepare_pick_fixture(gs,world,scenario,output,attempts=args.endpoint_attempts,seed=args.seed,
                active_arm_id='arm_1' if args.mode=='placement_probe' else 'arm_0',empty_mission_initial=empty_mode and args.mode=='mission')
        goals=None
        if args.mode=='short_motions':
            import numpy as np
            goals={r['id']:[float(np.float32(v)) for v in r['q_rad']] for r in scenario['robots']}
            for q in goals.values():q[4]+=.005
        if empty_mode:
            empty_overlay['pre_fixture_scenario_sha256']=empty_overlay['effective_scenario_sha256']
            empty_overlay['effective_scenario_sha256']=digest(scenario)
            empty_overlay['supplied_grasp_ready_INITIAL_only']=True
            save(output/'empty_payload_overlay.json',empty_overlay)
            if empty_path:save(output/'empty_program_input.json',empty_program)
        world_name='effective_live_world.json' if particle_mode else 'live_world.json'
        scenario_name='effective_live_initialization.json' if particle_mode else 'live_initialization.json'
        save(output/world_name,world);save(output/scenario_name,scenario)
        paths.update({str(output/world_name):sha(output/world_name),
            str(output/scenario_name):sha(output/scenario_name)})
        if empty_mode:
            paths[str(output/'empty_payload_overlay.json')]=sha(output/'empty_payload_overlay.json')
            if empty_path:paths[str(output/'empty_program_input.json')]=sha(output/'empty_program_input.json')
        save(output/'sources.json',paths)
        # The optional unit IK changes INITIAL before the physical scene exists.
        # End that source scope, then pin the exact effective recipes BEFORE
        # run_episode constructs the real scene. Neither guard's pins mutate.
        if not source_guard.after_request():raise ContractError('Full setup completion source verification failed')
        setup_source_receipt=source_guard.receipt();source_guard=None
        source_guard=PinnedSourceGuard({**paths,**snapshots},full_interval_s=1.)
        if not source_guard.before_request():raise ContractError('Full physical-request source verification failed')
        driver=StudentEpisodeDriver(strategy=strategy,case=scenario['case'],speed_scale=1. if empty_path else bound['joint_speed_scale'],mode=args.mode,
            probe_goals=goals,max_queries=args.max_queries,endpoint_attempts=args.endpoint_attempts,seed=args.seed,
            use_edge_timing_hook=args.edge_timing_hook,step_limit=args.step_limit or None,fixture_readback=args.query_fixture_readback,
            placement_policy=args.placement_policy,grasp_policy=grasp_policy,
            **({'empty_program':empty_program} if empty_path else {}),
            **({'saved_plan':saved_plan} if saved_plan is not None else {}))
        if particle_mode and world['particle_material_request'].get('precision','32')=='64':
            from physics.student_episode_driver import panda_limits
            driver.limits=panda_limits(scenario['case'],precision='64')
        driver.sparse_capture_only=args.images
        from physics.episode_runtime import run_episode
        bindings={'provided_task_manifest_sha256':sha(task/'manifest.json'),'design_sha256':design.sha256}
        if args.fixed_peer_reference is not None:
            bindings.update(single_active_arm=args.single_active_arm,fixed_peer_reference_sha256=sha(args.fixed_peer_reference))
        if particle_mode:bindings['source_manifest_sha256']=sha(task/'manifest.json')
        if empty_mode:
            bindings.update(payload_mode='empty',
                empty_program_module_sha256=sha(ROOT/'transport_challenge/empty_container_program.py'),
                empty_overlay_sha256=sha(output/'empty_payload_overlay.json'))
            if empty_path:bindings['joint_path_sha256']=sha(args.joint_path)
        selected_strategy_module=(saved_plan.offline_planner['module'] if saved_plan is not None else args.strategy)
        if not empty_path:bindings.update(strategy_module=selected_strategy_module,strategy_source_sha256=sha(strategy_file))
        if saved_plan is not None:
            bindings.update(motion_source='saved_cpu_plan',saved_bundle_sha256=saved_plan.sha256,
                post_place_policy=saved_plan.post_place_policy,
                saved_plan_file_sha256=saved_plan.file_sha256,runtime_plan_path_invocations=0,runtime_ik_calls=0,runtime_rrt_calls=0)
        runtime_started=time.monotonic()
        pre_runtime_s=runtime_started-started
        runtime_wall_limit_s=args.wall_limit_s
        if args.mode=='one_carry':
            runtime_wall_limit_s=args.wall_limit_s-pre_runtime_s-10.
            if runtime_wall_limit_s<=0:raise TimeoutError('One-carry startup exhausted its whole-review deadline')
        result=run_episode(gs,world=world,scenario=scenario,approach=None,physics_reference=reference,output=output,
            source_bindings=bindings,
            source_check=matches,wall_limit_s=runtime_wall_limit_s,render=args.images or args.video,record_video=args.video,
            observation_readback='batched',student_driver=driver,
            rigid16_readback=args.rigid16_readback,trace_mode=args.trace_mode,execution_backend=args.execution_backend,
            single_active_arm=args.single_active_arm,fixed_peer_reference=args.fixed_peer_reference)
        if args.video or empty_mode or saved_plan is not None:
            result['entry_timing']={'pre_runtime_setup_s':pre_runtime_s,'runtime_call_s':time.monotonic()-runtime_started}
        result.update(grasp_policy=grasp_policy,versions=versions,selected_design_sha256=design.sha256,design_binding=bound['receipt'],
            strategy_module=None if empty_path else selected_strategy_module,unit_fixture=args.mode in ('one_carry','short_motions','pick_place_unit','placement_probe','bead_lift_unit','empty_path'))
        if saved_plan is not None:
            result.update(motion_source='saved_cpu_plan',saved_bundle_sha256=saved_plan.sha256,
                post_place_policy=saved_plan.post_place_policy,
                saved_plan_file_sha256=saved_plan.file_sha256,runtime_plan_path_invocations=0,runtime_ik_calls=0,runtime_rrt_calls=0)
        if empty_mode:result.update(payload_mode='empty',provided_grasp_ready_initial=True)
        if empty_path:result.update(joint_path_sha256=sha(args.joint_path),strategy_plan_path_invocations=0)
        if hasattr(strategy,'planning_receipts'):save(output/'strategy_planning_receipts.json',strategy.planning_receipts)
    except BaseException as exc:
        result.update(status='STUDENT_RUN_FAILED',failure={'type':type(exc).__name__,'message':str(exc)},traceback=traceback.format_exc())
    finally:
        if gs is not None:
            try:gs.destroy()
            except Exception as exc:result.setdefault('cleanup_errors',[]).append('genesis: '+repr(exc))
        if source_guard is None:
            unchanged=matches()  # Preserve the pre-initialization full-hash fallback.
            source_receipt=None
        else:
            try:unchanged=source_guard.after_request()
            except Exception as exc:
                unchanged=False;source_errors.append({'type':type(exc).__name__,'message':str(exc)})
            source_receipt=source_guard.receipt()
        if setup_source_receipt is not None:
            unchanged=bool(unchanged and setup_source_receipt['request_complete_full_sha_verified'])
        result.update(source_bindings_unchanged=unchanged,process_wall_s=time.monotonic()-started,
            source_binding_check_errors=source_errors,student_release_approved=False,torch_reinstalled=False)
        if 'entry_timing' in result:
            result['entry_timing']['post_runtime_cleanup_and_source_s']=(result['process_wall_s']
                -result['entry_timing']['pre_runtime_setup_s']-result['entry_timing']['runtime_call_s'])
        result['source_guard']={'schema_version':'episode-source-boundaries-v1','setup':setup_source_receipt,
            'physical_request':source_receipt,'completed_full_sha_verified':bool(source_receipt is not None and unchanged
                and source_receipt['request_complete_full_sha_verified']),
            'preinit_full_hash_fallback_only':source_receipt is None,
            'scope':'Full initial/before/after SHA and callback-serviced periodic1s SHA. Every callback stats all paths. Same-signature content changes can evade quick checks until a full boundary. No per-call cryptographic guarantee.'}
        result.setdefault('execution_backend',args.execution_backend)
        finalize_single_batch_result(result,mode=args.mode,source_bindings_unchanged=unchanged)
        accepted=result.get('status') in ('STUDENT_EXECUTION_SHORT_CUTOFF','STUDENT_THREE_ARM_MOTION_PROBE_PASS',
            'PROVIDED_PICK_PLACE_UNIT_SAMPLED_PASS','PROVIDED_PLACEMENT_PROBE_SAMPLED_PASS','PROVIDED_BEAD_LIFT_UNIT_SAMPLED_PASS',
            'STUDENT_THREE_ARM_SINGLE_BATCH_SAMPLED_PASS','STUDENT_FIRST_HANDOFF_SAMPLED_PASS','EMPTY_CONTAINER_JOINT_PATH_SAMPLED_PASS',
            'EMPTY_THREE_ARM_RRT_MISSION_SAMPLED_PASS','ONE_CARRY_TRANSPORT_SAMPLED_PASS','ONE_CARRY_DEVICE_TRANSPORT_SAMPLED_PASS') and unchanged and not result.get('cleanup_errors')
        result['bounded_execution_accepted']=accepted
        save(output/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('status','bounded_execution_accepted','physical_steps','simulation_s','process_wall_s','failure')}))
    return 0 if result['bounded_execution_accepted'] else 1

if __name__=='__main__':raise SystemExit(main())
