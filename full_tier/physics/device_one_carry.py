"""Supplied four-bead device execution of one unchanged CPU-saved carry.

The original runtime owns fresh INITIAL, actual pick and native static validation.
This module owns its distinct device histories, supported release and rest.
It never commits synthetic legacy dispatcher rows.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from transport_challenge.contracts import ContractError, digest
DT=.0025

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
def require(value,message):
    if not value:raise ContractError(message)


def command_table(dispatcher, first, count):
    """Sample the actual current action without preparing or committing fake steps."""
    segment = dispatcher.current_segment
    plan = segment.plan
    local_start = first.step - 1 - segment.origin_step
    actions = [a for a in plan.actions if a.start_step <= local_start < a.start_step + a.steps]
    if len(actions) != 1 or actions[0].kind != 'trajectory':
        raise ValueError('Require one actual carrying trajectory')
    action = actions[0]
    if local_start + count > action.start_step + action.steps:
        raise ValueError('Probe crosses a real action boundary')
    arms = tuple(sorted((a.arm_id for a in plan.arms)))
    start = {a.arm_id: a for a in first.arms}
    position = []
    velocity = []
    for i in range(count):
        local = local_start + i - action.start_step + 1
        now = action.duration_s if getattr(plan, '_external_grid', False) and local == action.steps else min(local * plan.dt_s, action.duration_s)
        sample = action.trajectory.sample(now)
        q = []
        qd = []
        for name in arms:
            given = start[name]
            q.extend((*sample.q_rad, *given.finger_q_m) if name == action.arm_id else (*given.q_rad, *given.finger_q_m))
            qd.extend((*sample.qd_rad_s, *given.finger_qd_m_s) if name == action.arm_id else (*given.qd_rad_s, *given.finger_qd_m_s))
        position.append(q)
        velocity.append(qd)
    original_q = [v for name in arms for v in (*start[name].q_rad, *start[name].finger_q_m)]
    original_v = [v for name in arms for v in (*start[name].qd_rad_s, *start[name].finger_qd_m_s)]
    if position[0] != original_q or velocity[0] != original_v:
        raise ValueError('Actual first command differs from the upload table')
    return ({'position': position, 'velocity': velocity}, dict(arm_order=list(arms), segment_sha256=segment.sha256, action_id=action.action_id, trajectory_sha256=action.trajectory.sha256 if hasattr(action.trajectory, 'sha256') else None, first_step=first.step, last_step=first.step + count - 1, dt_s=plan.dt_s))

def state_row(observation):
    names = sorted(observation['arms'])
    return dict(native_scene_step=observation['native_scene_step'], simulation_s=observation['simulation_s'], body_world=copy.deepcopy(observation['body_world']), q=[v for n in names for v in observation['arms'][n]['q_all_dofs']], qd=[v for n in names for v in observation['arms'][n]['qd_all_dofs']], materials=copy.deepcopy(observation.get('materials')), particle_material=copy.deepcopy(observation.get('particle_material')))

@contextmanager
def device_contact_hooks(scene, audit, mode, support=None, handling=None):
    import types
    solver = scene.rigid_solver
    constraint = solver._func_constraint_force
    substep = solver.substep
    if '_func_constraint_force' in vars(solver) or 'substep' in vars(solver):
        raise RuntimeError('Previous observation hooks must be restored before device execution')

    def wrapped_constraint(_self, *a, **kw):
        value = constraint(*a, **kw)
        g = int(scene.sim._cur_substep_global)
        if mode == 'bead16' or g % 100 == 99:
            audit.pre(g)
            if support is not None:
                support.capture_pre(g)
            if handling is not None:
                handling.pre(g)
        return value

    def wrapped_substep(_self, *a, **kw):
        value = substep(*a, **kw)
        if mode == 'bead16':
            g = int(scene.sim._cur_substep_global)
            audit.post(g)
            if handling is not None:
                handling.post(g)
        return value
    solver._func_constraint_force = types.MethodType(wrapped_constraint, solver)
    if mode == 'bead16':
        solver.substep = types.MethodType(wrapped_substep, solver)
    try:
        yield
    finally:
        delattr(solver, '_func_constraint_force')
        if mode == 'bead16':
            delattr(solver, 'substep')
        if solver._func_constraint_force.__func__ is not constraint.__func__ or solver.substep.__func__ is not substep.__func__:
            raise RuntimeError('Native methods were not restored exactly')

def _run_source_bound_device(local, command, args, *, wall_deadline):
    import quadrants as qd
    from physics.deferred_gpu_control import DeviceCarryControl
    from physics.deferred_gpu_contacts import DeviceContactAudit
    from physics.deferred_gpu_beads import DeferredPayloadAudit
    scene = local['scene']
    dispatcher = local['dispatcher']
    driver = local['student_driver']
    total = args.warmup + args.steps
    table, table_record = command_table(dispatcher, command, total)
    for name in ('rigid16_observer', 'particle_contact_copy'):
        observer = local.get(name)
        if observer is not None:
            observer.__exit__(None, None, None)
    mode = 'bead16'
    helper = local['automatic_grasp']
    snapshot = helper.snapshot(helper.holder or next(iter(local['arms'])))
    contact = DeviceContactAudit(scene, local['coverage'], owners=local['owners'], components_by_geom=local['components'], pads_by_arm=local['pads'], stations_by_geom=local['station_map'], sample_mode=mode, expected_inventory=snapshot['inventory'], model=local['model'])
    contact.set_phase('carry', helper.holder, expected_inventory=snapshot['inventory'])
    handling = None
    from physics.deferred_gpu_handling import DeviceHandlingAudit
    handling = DeviceHandlingAudit(local, contact)
    support = None
    from physics.deferred_gpu_tail import DeviceNamedSupportHistory
    support = DeviceNamedSupportHistory(contact, support_id=driver.goal['support_id'], capacity_steps=total, phase='carry', holder_arm_id=helper.holder)
    gravity_owners = local['gravity']
    if any((feed._readback != 'fused' for feed in gravity_owners.values())):
        from physics.gravity_feedforward import GravityFeedforward
        gravity_owners = {name: GravityFeedforward(arm, (0, 0, -9.81), readback='fused') for name, arm in local['arms'].items()}
        for feed in gravity_owners.values():
            feed.torques()
    control = DeviceCarryControl(scene, local['arms'], gravity_owners, local['limits_by_arm'], table, kp=local['kp'], kv=local['kv'], effort=local['effort'], finger_bounds=local['finger_bounds'], hands=local['hands'], tool_workspace_m=local['case']['tool_workspace_m'], tracking_error_rad=local['execution']['tracking_error_rad'], dt_s=DT)
    payload = DeferredPayloadAudit(scene, local['beads'], local['tray'], local['model'], local['ids'], tilt_limit_deg=driver.body_tilt_limit_deg, capacity_steps=total)
    particle = None
    begin = int(scene.t)
    initial = state_row(local['observation'])
    reference_gravity = [float(v) for name in sorted(local['arms']) for v in local['gravity'][name].torques()]
    started = None
    warm_wall = None
    qd.sync()
    all_started = time.perf_counter()
    with device_contact_hooks(scene, contact, mode, support, handling):
        for i in range(total):
            if time.perf_counter() >= wall_deadline:
                raise TimeoutError('Original complete review wall budget exceeded')
            if i == args.warmup:
                qd.sync()
                started = time.perf_counter()
                warm_wall = started - all_started
            control.apply(i)
            scene.step()
            control.capture_post(i)
            payload.capture_post(i)
        qd.sync()
        elapsed = time.perf_counter() - started
    control_result = control.finish()
    import numpy as np
    history = control_result.pop('history')
    gravity_index = control_result['history_channels'].index('gravity_tau')
    gravity_delta = np.asarray(history[0, :, gravity_index]) - np.asarray(reference_gravity)
    control_result['actual_initial_CPU_gravity_comparison'] = dict(maximum_absolute_effort_difference=float(np.max(np.abs(gravity_delta))), bitwise_equal=bool(np.array_equal(np.asarray(history[0, :, gravity_index]), np.asarray(reference_gravity))), reference_native_step=begin, reference_taken_at_boundary_only=True, original_native_motor_identity_tolerance=0.005)
    history_path = args.output / 'actual_device_control_history.npz'
    np.savez(history_path, history=history)
    control_result['history_record'] = dict(file=str(history_path), sha256=sha(history_path), shape=list(history.shape))
    payload_result = payload.finish()
    payload_history = payload_result.pop('history')
    payload_path = args.output / 'actual_device_payload_history.npz'
    np.savez(payload_path, history=payload_history)
    payload_result['history_record'] = dict(file=str(payload_path), sha256=sha(payload_path), shape=list(payload_history.shape))
    contact_result = contact.final_readback(expected_pre_samples=total * (16 if mode == 'bead16' else 1), expected_post_samples=total * 16 if mode == 'bead16' else 0)
    from physics.transport_outcomes import check_diagnostic, reject_failed_checks
    handling_result = handling.checkpoint()
    checks = check_diagnostic(control_result, payload_result, contact_result, handling_result)
    save(args.output / 'carry_checks.json', checks)
    if checks['assessment']['checks_passed'] is not True:
        import numpy as np
        support_path = args.output / 'actual_device_failed_carry_support.npz'
        raw = support.data.detach().cpu().numpy().copy()
        np.savez(support_path, history=raw)
        save(args.output / 'failed_support_history.json', dict(file=str(support_path),
            sha256=sha(support_path), shape=list(raw.shape), interpreted_as_valid_support=False))
        reject_failed_checks(checks['assessment'])
    support_result = None
    if support is not None:
        support_result = support.finish(contact_result)
        support_path = args.output / 'actual_device_carry_support.npz'
        np.savez(support_path, history=support.raw)
        support_result['history_record'] = dict(file=str(support_path), sha256=sha(support_path), shape=list(support.raw.shape))
    handling_result = None
    if handling is not None:
        handling_result = handling.checkpoint()
        save(args.output / 'actual_carry_handling.json', handling_result)
    particle_result = None
    final_particle_snapshot = None
    final = local['observe']()
    record = dict(schema_version='real-held-device-carry-prefix-v1', mode='device', actual_start_step=begin, actual_final_step=int(scene.t), actual_carry_intervals=total, actual_physical_intervals=total, initial_connection_only=False, warmup_intervals=args.warmup, timed_intervals=args.steps, timed_host_wall_s=elapsed, warmup_and_compile_wall_s=warm_wall, native_errno_checks_retained=True, full_transport_accepted=False, placement_verified=False, command_table=table_record, command_table_sha256=hashlib.sha256(json.dumps(table, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), initial=initial, final=state_row(final), control=control_result, contacts=contact_result, particles=particle_result, payload=payload_result, carry_support=support_result, handling=handling_result, actual_final_particle_snapshot=final_particle_snapshot, material='beads', actual_held_arm=helper.holder, provided_attach_events=copy.deepcopy(driver.automatic_events), no_fake_dispatcher_commits=True, scope='Measured real carrying prefix only. Deferred failure flags must pass independently. No complete transport or student release claim.')
    record['handling_checkpoint_file'] = str(args.output / 'actual_carry_handling.json')
    record['handling_checkpoint_file_sha256'] = sha(args.output / 'actual_carry_handling.json')
    action = [a for a in dispatcher.current_segment.plan.actions if a.action_id == command.action_id][0]
    if total != action.steps or begin != dispatcher.current_segment.origin_step + action.start_step:
        raise ValueError('Complete carry requires the exact entire saved action')
    if not all((record[k]['passed'] is True for k in ('control', 'payload', 'contacts'))):
        raise ValueError('A failed GPU carry cannot continue to provided release')
    save(args.output / 'ACTUAL_CARRY_RESULT.json', record)
    from physics.deferred_gpu_tail import run_supported_tail
    tail_context = dict(local)
    tail_context.update(gravity=gravity_owners, deferred_carry_result=copy.deepcopy(record), deferred_carry_support=support)
    if handling is not None:
        tail_context['deferred_handling_audit'] = handling
    tail = run_supported_tail(tail_context, {'position': table['position'][-1]}, output=args.output / 'provided_tail', wall_deadline=wall_deadline)
    record.update(schema_version='provided-deferred-one-carry-result-v1', tail=tail, actual_carry_final_step=record['actual_final_step'], actual_final_step=int(scene.t), complete_one_carry_physical_verified=tail.get('passed') is True, complete_original_handling_verified=tail.get('persistent_original_handling_verified') is True, actual_held_arm=helper.holder, scope='Entire saved carrying action plus actual supported release and final original rest gates. Resource admission is separately owned by supervisor.')
    return record


def _validate_original_handling(record, *, verify_files=True):
    """Require one exact prefix seed and persistent all-h carry/release lineage."""
    carry=record.get('handling') or {}
    tail=record.get('tail') or {}
    final=tail.get('final_handling') or {}
    seed=carry.get('prefix_guard') or {}
    start=record['actual_start_step']*16
    require(record.get('complete_original_handling_verified') is True
        and tail.get('persistent_original_handling_verified') is True
        and tail.get('carried_handling_checkpoint')==carry,
        'Original handling was not retained through the whole real tail')
    require(seed.get('passed') is True and seed.get('first_failure') is None
        and seed.get('initial_scene_step')==0 and seed.get('initial_simulation_s')==0.
        and seed.get('last_accepted_scene_step')==start
        and seed.get('observations_accepted')==start
        and abs(seed['last_accepted_simulation_s']-start*(DT/16))<=1e-9
        and seed.get('dt_s')==DT/16 and seed.get('qualified_holder_id') is None
        and seed.get('supported_pickup_count')==seed.get('release_count')==0,
        'The actual original fresh-prefix state/clock/acquisition seed differs')
    reach=carry.get('native_reach_envelopes')
    envelope=carry.get('body_root_envelope')
    arm_order=record.get('control',{}).get('arm_order',[])
    require(arm_order in (['arm_0'],['arm_0','arm_1','arm_2'])
        and set(seed.get('known_holder_ids',[]))==set(arm_order),
        'Original handling and actual controlled-arm inventory differ')
    single=record.get('control',{}).get('single_active_arm_contract')
    if len(arm_order)==1:
        from physics.fixed_peer_arms import validated_single_active_contract
        single=validated_single_active_contract(single)
        require(type(single) is dict
            and single.get('schema_version')=='single-active-arm-fixed-peers-v1'
            and single.get('active_arm_id')==arm_order[0]
            and single.get('frozen_arm_ids')==['arm_1','arm_2']
            and single.get('geometry_verified') is True
            and record['control'].get('controlled_dof_count')==9,
            'One-arm result requires its actual qualified fixed-peer scene')
    require(type(reach) is dict and reach.get('passed') is True
        and set(reach.get('by_arm',{}))==set(arm_order)
        and type(envelope) is dict and envelope.get('passed') is True
        and envelope.get('radius_limit_m')==.20,
        'Original authenticated reach/body envelope missing')
    phases=tail['phases']
    require(all(p.get('control',{}).get('arm_order')==arm_order for p in phases),
        'Tail cannot change the controlled-arm inventory')
    if single is not None:
        require(all(p['control'].get('single_active_arm_contract')==single
            and p['control'].get('controlled_dof_count')==9 for p in phases),
            'Carry and tail fixed-peer bindings differ')
    checkpoints=[(carry,record['actual_carry_final_step'],'arm_0',0)]
    for item in phases:
        checkpoints.append((item.get('handling_checkpoint') or {},item['end_native_step'],
            'arm_0' if item['phase']=='place' else None,0 if item['phase']=='place' else 1))
    checkpoints.append((final,record['actual_final_step'],None,1))
    sources=carry.get('source_hashes')
    require(type(sources) is dict and sources,'Original handling implementation source binding missing')
    for item,end,holder,releases in checkpoints:
        count=end*16-start
        require(item.get('schema_version')=='deferred-device-handling-audit-v1'
            and item.get('passed') is True and item.get('sticky_failed') is False
            and item.get('first_failure') is None and item.get('source_bindings_verified') is True
            and item.get('every_original_internal_sample_preserved') is True
            and item.get('start_global_substep')==start
            and item.get('last_accepted_global_substep')==end*16
            and item.get('PRE_samples')==item.get('POST_samples')==count
            and item.get('prefix_guard')==seed and item.get('source_hashes')==sources
            and item.get('native_reach_envelopes')==reach and item.get('body_root_envelope')==envelope
            and item.get('qualified_holder_id')==holder
            and item.get('supported_pickup_count')==1 and item.get('release_count')==releases
            and item.get('held_root_to_hand_limit_m')==.20
            and item.get('unheld_displacement_limit_m')==.005,
            'Persistent actual all-h handling state, seed, reach, source or clock failed')
    files=[(record.get('handling_checkpoint_file'),record.get('handling_checkpoint_file_sha256'),carry),
        (tail.get('final_handling_file'),tail.get('final_handling_file_sha256'),final)]
    files.extend((p.get('handling_checkpoint_file'),p.get('handling_checkpoint_file_sha256'),
                  p.get('handling_checkpoint')) for p in phases)
    if verify_files:
        require(all(type(path) is str and Path(path).is_file() and sha(path)==h
            and json.loads(Path(path).read_bytes())==item for path,h,item in files),
            'Actual persistent handling checkpoint bytes are missing or changed')
        require(all(Path(path).is_file() and sha(path)==h for path,h in sources.items()),
            'Original handling source changed')
    return True

def validate_device_completion(record, *, verify_files=True):
    """Fail closed on the distinct source-bound complete one-carry evidence."""
    require(type(record) is dict and record.get('schema_version')=='provided-deferred-one-carry-result-v1'
        and record.get('material')=='beads' and record.get('complete_one_carry_physical_verified') is True
        and record.get('no_fake_dispatcher_commits') is True
        and record.get('native_errno_checks_retained') is True,'Missing complete bead device execution')
    n=record.get('actual_carry_intervals')
    require(type(n) is int and 0<n<=20000,'Invalid complete saved carry length')
    begin=record['actual_start_step'];end=record['actual_carry_final_step']
    require(type(begin) is int and end-begin==n,'Carry clock extent changed')
    files=[]
    for label in ('control','payload','contacts'):
        require(record.get(label,{}).get('passed') is True,'Failed '+label+' device predicates')
    control,payload,contacts=record['control'],record['payload'],record['contacts']
    require(control['applied']==control['completed_post_captures']==control['expected_steps']==n
        and control['start_native_step']==begin and control['end_native_step']==end
        and control['first_failed_step'] is None and all(v is False for v in control['flags'].values()),
        'Control predicates or exact history count failed')
    require(payload['actual_POST_samples']==n and payload['bead_count']==4
        and payload['first_failed_step'] is None and all(v is False for v in payload['flags'].values()),
        'Payload predicates or all-four inventory failed')
    require(contacts['PRE_samples']==contacts['POST_samples']==16*n
        and contacts['sample_mode']=='bead16' and contacts['original_sampling_preserved'] is True
        and contacts['sticky_failed'] is False and contacts['first_bad_global_substep'] is None
        and contacts['category_counts']['UNALLOWED_CONTACT']==0,'Incomplete original16 device contact assessment')
    support=record.get('carry_support') or {}
    require(support.get('passed') is True and support['start_native_step']==begin
        and support['end_native_step']==end and support['actual_outer_intervals']==n
        and support['maximum_consecutive_unsupported_held_s']>=.5,
        'Actual carried support/load-bearing history missing')
    for part in (control,payload,support):files.append(part['history_record'])
    tail=record.get('tail') or {}
    require(tail.get('status')=='PROVIDED_DEFERRED_BEAD_TAIL_PASS' and tail.get('passed') is True
        and tail.get('actual_release_verified') is True and tail.get('native_zero_time_release') is True
        and tail.get('source_bindings_verified') is True and tail.get('original_rest_thresholds_unchanged') is True
        and tail.get('failure') is None and tail['start_native_step']==end,
        'Actual supported release/rest tail did not complete')
    cursor=end
    phases=tail['phases']
    require(len(phases)>=3 and [p['phase'] for p in phases[:3]]==['place','release','released_settle']
        and [p['actual_intervals'] for p in phases[:3]]==[200,400,400]
        and all(p['phase']=='released_settle' and p['actual_intervals']==200 for p in phases[3:]),
        'Original supplied hold/open/settle durations changed')
    for item in phases:
        count=item['actual_intervals']
        require(item['start_native_step']==cursor and item['end_native_step']==cursor+count
            and all(item[k].get('passed') is True for k in ('control','payload','contacts','support'))
            and item['contacts']['PRE_samples']==item['contacts']['POST_samples']==16*count,
            'Failed or nonconsecutive actual tail history')
        cursor+=count
        files.append(dict(file=item['history_file'],sha256=item['history_file_sha256']))
    gate=tail['final_released_gate'];final=tail['actual_final_observation']
    require(gate.get('passed') is True and gate['expected_intervals']==200
        and gate['expected_material_endpoints']==201 and gate['strict_material'] is True
        and gate['actual_unsupported_held_s']>=.5
        and all(gate[k]['passed'] is True for k in ('all_arms_rest','body_rest','bead_bulk_rest'))
        and gate['bead_bulk_rest']['quality_passed'] is True
        and final.get('provided_grasp') is None and final['native_scene_step']==cursor
        and tail['final_native_step']==record['actual_final_step']==cursor
        and tail['actual_tail_intervals']==cursor-end,
        'Original released-rest and native no-weld final boundary failed')
    require(0<=tail['additional_material_wait_s']<=5.
        and cursor-end==1000+round(tail['additional_material_wait_s']/.0025),
        'Original bounded material wait contract changed')
    if verify_files:
        require(all(type(v.get('file')) is str and Path(v['file']).is_file()
            and sha(v['file'])==v.get('sha256') for v in files),'Actual history file changed or missing')
        require(all(Path(path).is_file() and sha(path)==h
            for block in (control,payload,contacts,support,tail)
            for path,h in block.get('source_hashes',{}).items()),'Device implementation source changed')
    _validate_original_handling(record,verify_files=verify_files)
    return True

def execute_device_one_carry(context,command,*,output,wall_deadline):
    """Run only at the genuine saved carry boundary after the original pick."""
    from physics.one_carry_planning import SavedOneCarryPlan
    driver=context['student_driver'];scene=context['scene']
    require(getattr(driver,'mode',None)=='one_carry' and not driver.particle_mode
        and getattr(driver,'grasp_policy',None)=='automatic'
        and type(driver.saved_plan) is SavedOneCarryPlan
        and driver.saved_plan.record()['payload']=='beads'
        and len(context['beads'])==len(context['ids'])==4,
        'Device execution is explicit automatic four-bead saved one-carry only')
    require(context['source_check']() is True and driver.saved_plan.verify_sources() is True,
        'Saved CPU/runtime source bindings changed at actual carry boundary')
    require(command.action_role=='carry' and context['automatic_grasp'].holder==command.active_arm_id
        and scene.sim.dt==.0025 and scene.sim.substeps==16
        and int(scene.t)==command.step-1,'Actual supported pick/carry native boundary required')
    actions=[a for a in context['dispatcher'].current_segment.plan.actions if a.action_id==command.action_id]
    require(len(actions)==1 and actions[0].kind=='trajectory','Require the original actual saved trajectory action')
    count=actions[0].steps
    require(type(count) is int and 0<count<=20000,'Bound one complete saved carrying path')
    require(type(wall_deadline) in (int,float) and math.isfinite(wall_deadline)
        and time.perf_counter()<wall_deadline,'Original supervised wall budget exhausted')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    args=SimpleNamespace(output=output,warmup=min(8,max(0,count-1)),steps=count-min(8,max(0,count-1)))
    record=_run_source_bound_device(context,command,args,wall_deadline=wall_deadline)
    validate_device_completion(record)
    require(context['source_check']() is True,'Sources changed after the actual released-rest gate')
    return record
