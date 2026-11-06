"""Review one exact student CPU path with real GPU physics.

The default explicit scope starts from its declared supported INITIAL.
The supervisor bounds CPU-attributed planning plus this complete GPU process.
The parent samples the owned worker PID/creation handle through its exit.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
FIXED_PEER_CALIBRATION_SHA256='e532e516db9fe0c6410548f38fbc747c8344a5b7aa3e606589b6fa1908ffce16'


def single_active_options(args,plan,p):
    """Select supplied geometry calibration without asking students to capture it."""
    if args.single_active_arm is None:
        if args.fixed_peer_reference is not None:p.error('A fixed-peer calibration requires explicit single active mode')
        return []
    if (args.material!='beads' or args.execution_backend!='device'
            or args.single_active_arm!=plan.stage(0)['active_arm_id']):
        p.error('Single active mode requires the actual saved bead carry arm and device execution')
    reference=(args.fixed_peer_reference or ROOT/'physics/assets/fixed_peer_geometry_calibration_v1.json').resolve(strict=True)
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=FIXED_PEER_CALIBRATION_SHA256:
        p.error('This experimental mode accepts only its supplied source-bound geometry calibration')
    record=json.loads(reference.read_bytes())
    if (record.get('geometry_calibration_only') is not True or record.get('native_steps')!=0
            or record.get('accepted_transport') is not False):
        p.error('The supplied reference is geometry calibration, never a physical checkpoint')
    for relative,expected in record['source_sdk_relative_sha256'].items():
        package,subpath=relative.split('/',1)
        if package not in ('genesis','quadrants'):p.error('Unexpected calibrated SDK namespace')
        spec=importlib.util.find_spec(package)
        if spec is None or spec.origin is None:p.error('Calibrated native SDK is unavailable')
        source=Path(spec.origin).parent/subpath
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            p.error('Installed native source differs from the supplied fixed-peer geometry calibration')
    args.fixed_peer_reference=reference
    return ['--single-active-arm',args.single_active_arm,'--fixed-peer-reference',str(reference)]


@contextmanager
def block_online_planning():
    """Fail any actual GPU-worker attempt to replace or replan the saved path."""
    from physics import student_motion_planning as motion
    from transport_challenge import joint_planning
    from transport_challenge.joint_kinematics import BoundedKinematics
    from transport_challenge.contracts import ContractError
    calls={'plan_student_motion':0,'plan_joint_motion':0,'ik':0}
    originals=[]
    def deny(name):
        def blocked(*args,**kwargs):
            calls[name]+=1
            raise ContractError('Saved one-carry review forbids online '+name)
        return blocked
    for owner,name in ((motion,'plan_student_motion'),(joint_planning,'plan_joint_motion'),(BoundedKinematics,'ik')):
        originals.append((owner,name,getattr(owner,name)))
        setattr(owner,name,deny(name))
    try:yield calls
    finally:
        for owner,name,value in reversed(originals):setattr(owner,name,value)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--saved-carry',type=Path,required=True)
    p.add_argument('--task',type=Path,default=ROOT/'cases/three_arm_cell_bead16_v2')
    p.add_argument('--design',type=Path,default=ROOT/'student/three_arm_design.json')
    p.add_argument('--material',choices=('beads','process_liquid','sand'),default='beads')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wall-limit-s',type=float,default=1800.)
    p.add_argument('--vram-limit-gb',type=float,default=5.,help='Maximum observed sampled process memory in decimal GB, at most5. Not a hard-cap proof.')
    p.add_argument('--memory-interval-s',type=float,default=.5)
    p.add_argument('--rigid16-readback',choices=('legacy','native','native-verify'),default=None)
    p.add_argument('--trace-mode',choices=('compact','full'),default=None)
    p.add_argument('--execution-backend',choices=('legacy','device'),default='legacy')
    p.add_argument('--single-active-arm',choices=('arm_0','arm_1','arm_2'),help='Explicit experimental device mode. Other robots retain their supplied fixed collision geometry. Current qualification is arm0 only.')
    p.add_argument('--fixed-peer-reference',type=Path,help='Optional path to the identical supplied geometry-calibration file. No student capture is required.')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--gpu-wall-allowance-s',type=float,help=argparse.SUPPRESS)
    return p


def main(argv=None):
    started=time.perf_counter()
    p=parser();args=p.parse_args(argv)
    particles=args.material!='beads'
    if args.execution_backend=='device' and particles:p.error('Device execution is qualified only for the four-bead recipe')
    if args.rigid16_readback is None:args.rigid16_readback='legacy' if particles else 'native'
    if args.trace_mode is None:args.trace_mode='full' if particles else 'compact'
    if particles and (args.rigid16_readback!='legacy' or args.trace_mode!='full'):
        p.error('Native particles retain their original100-substep profile and full particle logging, not bead16 packets')
    if not 10<=args.wall_limit_s<=1800:p.error('The complete one-carry budget must be10..1800 seconds')
    if not 0<args.vram_limit_gb<=5. or not .1<=args.memory_interval_s<=3.:
        p.error('Use a positive process-memory limit at most5GB and sampling0.1..3s')
    from physics.one_carry_planning import SavedOneCarryPlan
    plan=SavedOneCarryPlan.from_file(args.saved_carry)
    if plan.record()['payload']!=args.material:p.error('Selected material differs from the saved physical recipe')
    single_args=single_active_options(args,plan,p)
    args.output=args.output.resolve()
    if args.worker:
        if args.gpu_wall_allowance_s is None or args.gpu_wall_allowance_s<=10:p.error('Worker requires its supervised remaining wall budget')
        from physics import run_student_three_arm as entry
        old=sys.argv
        sys.argv=[str(ROOT/'physics/run_student_three_arm.py'),'--mode','one_carry',
            '--saved-carry',str(args.saved_carry.resolve()),'--task',str(args.task.resolve()),
            '--design',str(args.design.resolve()),'--output',str(args.output),
            '--contents',args.material,'--grasp-policy','automatic','--query-fixture-readback','batched','--step-limit','0','--wall-limit-s',str(args.gpu_wall_allowance_s),
            '--rigid16-readback',args.rigid16_readback,'--trace-mode',args.trace_mode,'--execution-backend',args.execution_backend]
        sys.argv+=single_args
        with block_online_planning() as calls:
            try:entry_code=entry.main()
            finally:
                sys.argv=old
                args.output.mkdir(parents=True,exist_ok=True)
                with (args.output/'ONLINE_CALL_GUARD.json').open('x',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(blocked_online_calls=calls,
                        scope='Actual guarded GPU worker calls. Any attempt raises and rejects the run.'))+'\n')
        return entry_code
    summary=plan.record()['cpu_planning_result']
    cpu_wall=summary['process_wall_s']
    if type(cpu_wall) not in (int,float) or not 0<cpu_wall<args.wall_limit_s-30:
        p.error('Fresh CPU planning already exhausted the requested complete budget')
    if os.name!='nt':p.error('This resource-checked review requires exact Windows Popen-handle process-memory monitoring')
    memory_path=args.output.with_suffix('.memory_samples.jsonl')
    if args.output.exists() or args.output.with_suffix('.worker.log').exists() or memory_path.exists():
        p.error('Use a fresh output directory, worker log and memory-sample file')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    allowance=args.wall_limit_s-cpu_wall
    cleanup_reserve_s=20.
    worker_allowance=allowance-cleanup_reserve_s
    command=[sys.executable,'-B','-m','physics.review_student_transport','--worker',
        '--saved-carry',str(args.saved_carry.resolve()),'--task',str(args.task.resolve()),
        '--design',str(args.design.resolve()),'--material',args.material,'--output',str(args.output),
        '--wall-limit-s',str(args.wall_limit_s),'--gpu-wall-allowance-s',str(worker_allowance),
        '--rigid16-readback',args.rigid16_readback,'--trace-mode',args.trace_mode,'--execution-backend',args.execution_backend]
    command+=single_args
    worker_log=args.output.with_suffix('.worker.log')
    with worker_log.open('xb') as stream:
        process=subprocess.Popen(command,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        from physics.review_process_memory import ProcessMemorySampler,creation_identity,kill_owned_tree,wait_owned_worker
        identity=None
        try:
            identity=creation_identity(process)
            memory=ProcessMemorySampler(process,memory_path,started=started,interval_s=args.memory_interval_s,
                limit_gb=args.vram_limit_gb)
            memory.start()
        except BaseException:
            if identity is not None:kill_owned_tree(process,identity)
            else:
                process.kill();process.wait(timeout=3.)
            raise
        creation_s=(identity['windows_creation_filetime_100ns']-116444736000000000)/1e7
        print(json.dumps(dict(event='Applied Programming worker started',pid=process.pid,
            process_creation_unix_s=creation_s,owned_worker=identity,output=str(args.output))),flush=True)
        timed_out=False;termination=None;memory_limit_stop=False
        try:
            waited=wait_owned_worker(process,memory,identity,deadline=started+worker_allowance)
            code=waited['exit_code'];timed_out=waited['timed_out']
            memory_limit_stop=waited['memory_limit_stop'];termination=waited['termination']
        finally:
            if process.poll() is None:termination=kill_owned_tree(process,identity)
            memory_receipt=memory.finish()
    result_path=args.output/'result.json'
    result=json.loads(result_path.read_text(encoding='utf-8')) if result_path.is_file() else {}
    call_path=args.output/'ONLINE_CALL_GUARD.json'
    call_receipt=json.loads(call_path.read_text(encoding='utf-8')) if call_path.is_file() else {}
    calls=call_receipt.get('blocked_online_calls',{})
    try:sources_current=plan.verify_sources()
    except Exception:sources_current=False
    args.output.mkdir(parents=True,exist_ok=True)
    memory_receipt_path=args.output/'REVIEW_PROCESS_MEMORY.json'
    with memory_receipt_path.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(memory_receipt,indent=2,allow_nan=False)+'\n')
    wall=time.perf_counter()-started
    expected_status='ONE_CARRY_DEVICE_TRANSPORT_SAMPLED_PASS' if args.execution_backend=='device' else 'ONE_CARRY_TRANSPORT_SAMPLED_PASS'
    physical=(code==0 and not timed_out and result.get('status')==expected_status
        and result.get('one_carry_transport_accepted') is True and result.get('source_bindings_unchanged') is True
        and sources_current and result.get('saved_bundle_sha256')==plan.sha256
        and result.get('saved_plan_file_sha256')==plan.file_sha256
        and set(calls)=={'plan_student_motion','plan_joint_motion','ik'} and not any(calls.values())
        and result.get('cleanup_errors')==[] and not result.get('failure'))
    device_checks_passed=None
    if args.execution_backend=='device':
        from physics.device_one_carry import validate_device_completion
        try:
            actual_device=json.loads((args.output/'device_execution_result.json').read_text(encoding='utf-8'))
            device_checks_passed=bool(actual_device==result.get('device_execution')
                and validate_device_completion(actual_device)
                and result.get('execution_backend')=='device'
                and result.get('legacy_full_episode_coverage_claimed') is False
                and actual_device['handling']['prefix_guard']==result.get('legacy_prefix_handling_enforcement',{}).get('guard')
                and actual_device['handling']['body_root_envelope']==result.get('legacy_prefix_handling_enforcement',{}).get('body_root_envelope')
                and actual_device['handling']['native_reach_envelopes']==result.get('native_reach_envelopes'))
            active_contract=actual_device['control'].get('single_active_arm_contract')
            if args.single_active_arm is not None:
                device_checks_passed=bool(device_checks_passed and type(active_contract) is dict
                    and active_contract['active_arm_id']==args.single_active_arm
                    and active_contract.get('geometry_verified') is True
                    and actual_device['control']['arm_order']==[args.single_active_arm]
                    and actual_device['control']['controlled_dof_count']==9
                    and active_contract==result.get('single_active_arm_contract'))
            else:device_checks_passed=bool(device_checks_passed and active_contract is None)
        except Exception:device_checks_passed=False
        physical=bool(physical and device_checks_passed)
    report=dict(schema_version='student-one-carry-review-supervisor-v1',application_name='Applied Programming',
        scope='one_carry',execution_backend=args.execution_backend,device_completion_checks_passed=device_checks_passed,
        worker_pid=process.pid,worker_process_creation_unix_s=creation_s,
        exit_code=code,timed_out=timed_out,observed_memory_limit_stop=memory_limit_stop,
        physical_transport_accepted=physical,
        gpu_review_process_wall_s=wall,attributed_cpu_planning_wall_s=cpu_wall,
        additive_cpu_plus_review_wall_s=cpu_wall+wall,requested_complete_budget_s=args.wall_limit_s,
        observed_additive_cost_within_budget=physical and cpu_wall+wall<=args.wall_limit_s,
        cpu_cost_scope='Source-bound separately generated CPU plan. Additive attribution, not one uninterrupted process wall.',
        process_vram_status=memory_receipt['status'],
        observed_sampled_process_vram_within_budget=memory_receipt['observed_sampled_peak_within_limit'],
        process_vram_hard_cap_proven=False,process_memory=memory_receipt,
        cleanup_reserve_s=cleanup_reserve_s,owned_tree_termination=termination,
        timing_scope='Parent startup/precheck, worker, monitor stop, source check and memory-receipt write, before final supervisor report serialization.',
        full_mission_passed=False,student_release_approved=False,
        rigid16_readback=args.rigid16_readback,trace_mode=args.trace_mode,
        saved_bundle_sha256=plan.sha256,saved_file_sha256=plan.file_sha256)
    report['blocked_online_calls']=calls
    report['single_active_arm']=args.single_active_arm
    report['robot_dynamics_model']='single_active_fixed_peers' if args.single_active_arm is not None else 'original_dynamic_robots'
    if args.fixed_peer_reference is not None:
        report['provided_fixed_peer_geometry_calibration']={'path':str(args.fixed_peer_reference),
            'sha256':hashlib.sha256(args.fixed_peer_reference.read_bytes()).hexdigest(),
            'state_restoration_claimed':False,'same_layout_and_peer_INITIAL_only':True}
    report['review_accepted']=bool(physical and report['observed_additive_cost_within_budget']
        and report['observed_sampled_process_vram_within_budget'])
    from physics.transport_outcomes import read_device_diagnostics, review_outcome
    diagnostics=[]
    if args.execution_backend=='device':
        try:diagnostics=read_device_diagnostics(args.output)
        except Exception:
            diagnostics=[{'outcome':'INTEGRITY_ERROR'}]
        if physical and len(diagnostics)!=1+len(result['device_execution']['tail']['phases']):
            diagnostics.append({'outcome':'INTEGRITY_ERROR'})
    if (not sources_current or result.get('cleanup_errors') or any(calls.values())
            or result.get('saved_bundle_sha256')!=plan.sha256
            or result.get('saved_plan_file_sha256')!=plan.file_sha256):
        diagnostics.append({'outcome':'INTEGRITY_ERROR'})
    report['outcome']=review_outcome(accepted=report['review_accepted'],timed_out=timed_out,
        memory_limit_stop=memory_limit_stop,physical_complete=physical,sources_current=sources_current,
        diagnostic_outcomes=diagnostics)
    if report['outcome']!='ACCEPTED_TRANSPORT':report['review_accepted']=False
    report['diagnostic_check_outcomes']=diagnostics
    with (args.output/'REVIEW_SUPERVISOR.json').open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)
    return 0 if report['review_accepted'] else 2 if physical else 1


if __name__=='__main__':
    raise SystemExit(main())

