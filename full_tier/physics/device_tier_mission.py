"""Persistent original-INITIAL bead tiers with a separate native phase scheduler.

All three robots remain real dynamic entities. Only one commanded actor moves
at a time. The cup, four beads, scene, native clock and handling owner persist.
No legacy dispatcher rows or first-carry completion flags are manufactured.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from transport_challenge.contracts import ContractError, digest
from transport_challenge.joint_planning import JointTrajectory

DT = .0025
SUBSTEPS = 16
INITIAL_TICKS = 200
MAX_BLOCK_STEPS = 8000
SCHEMA = 'provided-device-tier-mission-result-v1'


def require(condition, message):
    if not condition:
        raise ContractError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def artifact(path):
    path = Path(path).resolve()
    return dict(file=str(path), sha256=sha(path))


def trajectory_commands(trajectory, commands, arm, limits):
    """Use the existing quintic sample law and exact outer-grid endpoint rule."""
    names = tuple(sorted(commands))
    require(names == ('arm_0', 'arm_1', 'arm_2') and arm in names,
            'Require all three actual native robot commands')
    tr = JointTrajectory(trajectory['waypoints'], trajectory['edge_durations_s'], limits)
    require(list(tr.waypoints[0]) == commands[arm]['q_rad'], 'Stored motion starts at a different command')
    count = math.ceil(tr.duration_s / DT) if tr.duration_s else 200
    q = np.empty((count, 27), np.float64)
    v = np.zeros_like(q)
    initial = np.asarray([x for n in names for x in commands[n]['q_rad'] + commands[n]['finger_q_m']], np.float64)
    q[:] = initial
    if tr.duration_s:
        offset = 9 * names.index(arm)
        for index in range(count):
            sample = tr.sample(min((index + 1) * DT, tr.duration_s))
            q[index, offset:offset+7] = sample.q_rad
            v[index, offset:offset+7] = sample.qd_rad_s
    require(np.isfinite(q).all() and np.isfinite(v).all(), 'Nonfinite actual command table')
    return dict(position=q, velocity=v)


def check_completion_clock(start, end, limit):
    require(type(start) is int and start == INITIAL_TICKS and type(end) is int and end > start,
            'Transport clock must exclude exactly the supplied initial 200 ticks')
    require(type(limit) is int and 0 < end - start <= limit, 'Tier simulated transport deadline exceeded')
    return dict(schema_version='ap-first-motion-final-rest-observation-v1', native_initial_step=0,
                first_motion_start_step=start, final_rest_end_step=end, native_end_step=end)


def _host(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def free_body_state(context):
    """Read all native free-body coordinates including bead rotation and spin."""
    scene = context['scene']
    bodies = [('cup', context['tray'])] + [(item['id'], body) for item, body in zip(context['ids'], context['beads'])]
    records = []
    for name, entity in bodies:
        q, v = _host(entity.get_qpos()).reshape(-1), _host(entity.get_dofs_velocity()).reshape(-1)
        require(q.shape == (7,) and v.shape == (6,) and np.isfinite(q).all() and np.isfinite(v).all(),
                'Native free-body q7/qd6 is missing or nonfinite')
        records.append(dict(id=name, entity_idx=int(entity.idx), root_link_idx=int(entity.base_link_idx),
                            qpos=q.tolist(), qd=v.tolist()))
    return dict(native_scene_step=int(scene.t), simulation_s=float(scene.cur_t),
                global_substep=int(scene.sim.cur_substep_global), bodies=records)


class _Support:
    """Original support kernel/assessment, extended to unheld approach/settle."""
    def __init__(self, audit, support_id, count, phase, holder, permission):
        from physics.deferred_gpu_tail import support_mask, _support_kernel, SUPPORT_COLUMNS
        from genesis.utils import array_class as arrays
        require(type(count) is int and 0 < count <= MAX_BLOCK_STEPS,
                'Bound each owned native history allocation')
        require(audit.phases[-1]['phase'] == phase and audit.phases[-1]['active_holder_arm_id'] == permission,
                'Support and actual contact phase differ')
        self.audit, self.scene = audit, audit.scene
        self.support_id, self.phase, self.holder = support_id, phase, holder
        self.permission_arm_id, self.steps, self.calls = permission, count, 0
        self.start = int(self.scene.sim.cur_substep_global)
        require(self.start == int(self.scene.t)*SUBSTEPS, 'Support starts between native outer intervals')
        mask = support_mask(audit.policy['registries'][3], support_id, audit.r.n_geoms)
        self.mask = audit.torch.tensor(mask, dtype=audit.torch.int32, device=audit.gs.device)
        self.data = audit.torch.zeros((count*SUBSTEPS, len(SUPPORT_COLUMNS)), dtype=audit.torch.int64,
                                      device=audit.gs.device)
        self.collect = _support_kernel(audit.qd, arrays)
        self.sources = {str(Path(__file__).resolve()): sha(__file__)}
        from physics import deferred_gpu_tail
        self.sources[str(Path(deferred_gpu_tail.__file__).resolve())] = sha(deferred_gpu_tail.__file__)
        self.result = None

    def capture_pre(self, ordinal):
        from physics.deferred_gpu_tail import DeviceNamedSupportHistory
        return DeviceNamedSupportHistory.capture_pre(self, ordinal)

    def finish(self, receipt):
        from physics.deferred_gpu_tail import DeviceNamedSupportHistory
        return DeviceNamedSupportHistory.finish(self, receipt)


class _Mission:
    def __init__(self, context, output, deadline):
        self.c, self.scene = context, context['scene']
        self.driver, self.helper = context['student_driver'], context['automatic_grasp']
        self.plan = self.driver.saved_plan
        self.contract = self.plan.task_contract
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.deadline = deadline
        self.names = tuple(sorted(context['arms']))
        require(self.names == ('arm_0','arm_1','arm_2') and self.scene.rigid_solver.n_dofs == 57
                and getattr(self.scene, '_ap_single_active_arm_contract', None) is None,
                'Tier replay requires the persistent original 57-DOF three-native-arm scene')
        require(int(self.scene.t) == 0 and self.scene.sim.dt == DT and self.scene.sim.substeps == SUBSTEPS,
                'Tier replay must begin at unchanged native clock zero')
        require(self.helper.holder is None and self.plan.verify_sources(), 'Initial holder or source failure')
        fingerprint=context['initial_fingerprint']
        require(fingerprint['status']=='FRESH_INITIAL_OBSERVABLES_CAPTURED' and not fingerprint['missing_required_fields']
                and fingerprint['fields']['rigid.rigid_info.qpos']['shape']==[62,1]
                and fingerprint['fields']['rigid.dyn_state.dofs.pos']['shape']==[57,1],
                'Complete original native 57DOF/62q INITIAL fingerprint is required')
        self.initial = copy.deepcopy(context['initial_observation'])
        self.observation = copy.deepcopy(self.initial)
        self.commands = copy.deepcopy(self.plan.record()['initial_commands_by_arm'])
        self.q = np.asarray([v for n in self.names for v in self.commands[n]['q_rad']+self.commands[n]['finger_q_m']], np.float64)
        self.rows = []
        self.handling = None
        self.completed, self.events, self.returns = [], [], []
        self.phases, self.gates, self.handoffs = [], [], []
        self.last_release = None
        self.record = dict(schema_version=SCHEMA, passed=False, full_mission_physical_verified=False,
            status='TIER_DEVICE_EXECUTION_INCOMPLETE', task_contract=self.contract,
            task_contract_sha256=digest(self.contract), saved_bundle_sha256=self.plan.sha256,
            saved_plan_file_sha256=self.plan.file_sha256, required_arm_ids=list(self.plan.required_arm_ids),
            original_initial_observation=self.initial, original_sample_ids=copy.deepcopy(context['ids']),
            original_initial_fingerprint=copy.deepcopy(fingerprint),
            original_initial_free_bodies=free_body_state(context), model='serial-active-three-native-arm-57dof',
            native_output_directory=str(self.output),native_scene_identity=self.scene_identity(),
            native_counts=dict(dofs=int(self.scene.rigid_solver.n_dofs),q_coordinates=int(self.scene.rigid_solver.n_qs),
                               links=int(self.scene.rigid_solver.n_links),geometries=int(self.scene.rigid_solver.n_geoms)),
            phases=self.phases, boundary_events=self.events, donor_returns=self.returns,
            stage_gates=self.gates, handoffs=self.handoffs, completed_stage_ids=self.completed,
            legacy_dispatcher_commits_claimed=False, persistent_scene=True, failure=None)
        from physics.gravity_feedforward import GravityFeedforward
        if any(owner._readback != 'fused' for owner in context['gravity'].values()):
            context['gravity'] = {name: GravityFeedforward(arm, (0,0,-9.81), readback='fused')
                                  for name,arm in context['arms'].items()}
            for feed in context['gravity'].values():feed.torques()

    def check(self, next_steps=0):
        if time.perf_counter() >= self.deadline:raise TimeoutError('Complete tier review wall budget exceeded')
        require(self.c['source_check']() is True, 'Tier source bindings changed')
        require(int(self.scene.t)+next_steps <= INITIAL_TICKS+self.contract['completion_limit_ticks'],
                'Tier simulated transport deadline exceeded')
        require(self.scene_identity()==self.record['native_scene_identity'],'Native scene or dynamic state allocation was replaced')

    def scene_identity(self):
        return [id(self.scene),id(self.scene.sim),id(self.scene.rigid_solver),id(self.scene.rigid_solver.dyn_state)]

    def snapshot(self):
        return dict(observation=copy.deepcopy(self.c['observe']()),
                    free_bodies=free_body_state(self.c),
                    native_scene_identity=self.scene_identity(),
                    holder_arm_id=self.helper.holder,
                    grasp=self.helper.snapshot(self.helper.holder or self.names[0]))

    def transaction(self, receipt):
        from physics.automatic_runtime_binding import physical_observation
        fresh = self.c['observe']()
        require(physical_observation(self.observation) == physical_observation(fresh),
                'Actual weld transaction changed physical state or time')
        self.observation = copy.deepcopy(fresh)
        return fresh

    def state(self):
        now = self.observation['simulation_s']
        stable = supported = now
        for row in reversed(self.rows):
            if all(math.hypot(*m['velocity_world_m_s']) <= self.c['settings']['scenario']['stable_speed_m_s']
                   and m['inside_carrier'] for m in row['materials']):
                stable = row['simulation_s']
            else:break
        for row in reversed(self.rows):
            if row.get('named_goal_support_every_internal'):supported=row['simulation_s']
            else:break
        return dict(observation=copy.deepcopy(self.observation),recent_rows=self.rows,
                    settings=self.c['settings'],rest_policy=self.c['rest_policy'],
                    stable_since=stable,support_since=supported,observe=self.c['observe'],
                    source_check=self.c['source_check'],wall_check=self.check,output=self.output,
                    record_provided_grasp_event=self.transaction)

    def phase(self, purpose, phase, actor, support_id, commands, *, permission=None):
        from physics.deferred_gpu_control import DeviceCarryControl
        from physics.deferred_gpu_contacts import DeviceContactAudit
        from physics.deferred_gpu_beads import DeferredPayloadAudit
        from physics.deferred_gpu_handling import DeviceHandlingAudit
        from physics.device_one_carry import device_contact_hooks
        from physics.deferred_gpu_tail import project_history
        from physics.transport_outcomes import check_diagnostic, reject_failed_checks
        total = len(commands['position'])
        require(total > 0, 'Empty native phase')
        self.check(total)
        records, supports = [], []
        for offset in range(0,total,MAX_BLOCK_STEPS):
            count=min(MAX_BLOCK_STEPS,total-offset)
            table={key:np.asarray(value[offset:offset+count],np.float64) for key,value in commands.items()}
            begin=int(self.scene.t);holder=self.helper.holder
            self.progress(purpose,phase,actor,begin,total-offset)
            permission=holder if permission is None and holder is not None else permission
            snapshot=self.helper.snapshot(holder or self.names[0])
            contact=DeviceContactAudit(self.scene,self.c['coverage'],owners=self.c['owners'],
                components_by_geom=self.c['components'],pads_by_arm=self.c['pads'],stations_by_geom=self.c['station_map'],
                sample_mode='bead16',expected_inventory=snapshot['inventory'],model=self.c['model'])
            contact.set_phase(phase,permission,expected_inventory=snapshot['inventory'])
            if self.handling is None:self.handling=DeviceHandlingAudit(self.c,contact)
            else:self.handling.bind_contact(contact,phase,holder,snapshot,permission_arm_id=permission)
            support=_Support(contact,support_id,count,phase,holder,permission)
            control=DeviceCarryControl(self.scene,self.c['arms'],self.c['gravity'],self.c['limits_by_arm'],table,
                kp=self.c['kp'],kv=self.c['kv'],effort=self.c['effort'],finger_bounds=self.c['finger_bounds'],
                hands=self.c['hands'],tool_workspace_m=self.c['case']['tool_workspace_m'],
                tracking_error_rad=self.c['execution']['tracking_error_rad'],dt_s=DT)
            payload=DeferredPayloadAudit(self.scene,self.c['beads'],self.c['tray'],self.c['model'],self.c['ids'],
                tilt_limit_deg=8.,capacity_steps=count)
            with device_contact_hooks(self.scene,contact,'bead16',support,self.handling):
                for index in range(count):
                    if time.perf_counter() >= self.deadline:raise TimeoutError('Complete tier wall budget exceeded')
                    control.apply(index);self.scene.step();control.capture_post(index);payload.capture_post(index)
                    if (index+1)%200==0:
                        self.check();self.progress(purpose,phase,actor,int(self.scene.t),count-index-1)
            cr,pr=control.finish(),payload.finish()
            ct=contact.final_readback(expected_pre_samples=count*SUBSTEPS,expected_post_samples=count*SUBSTEPS)
            checkpoint=self.handling.checkpoint()
            raw=support.data.detach().cpu().numpy().copy()
            index=len(self.phases);path=self.output/f'phase_{index:03d}_actual_histories.npz'
            np.savez(path,control=cr['history'],payload=pr['history'],support=raw,
                     command_position=table['position'],command_velocity=table['velocity'])
            self.record['last_actual_history']=artifact(path)
            checks=check_diagnostic(cr,pr,ct,checkpoint)
            save(self.output/f'phase_{index:03d}_checks.json',checks)
            reject_failed_checks(checks['assessment'])
            sr=support.finish(ct)
            previous=copy.deepcopy(self.observation)
            previous.update(step=begin,post_processing_complete=True,failures=[],actual_holder_arm_id=holder)
            projected=project_history(cr,pr,sr,start_step=begin,arm_order=self.names,identities=self.c['ids'],
                phase=phase,holder=holder,previous=previous,
                payload_owner_clock=dict(start_native_step=payload.start,end_native_step=int(self.scene.t),actual_POST_samples=payload.captures))
            self.rows.extend(projected)
            self.rows=self.rows[-2048:]
            self.observation=copy.deepcopy(self.c['observe']())
            self.q=table['position'][-1].copy()
            cr.pop('history');pr.pop('history')
            hp=self.output/f'phase_{index:03d}_handling.json';save(hp,checkpoint)
            item=dict(phase_id=f'phase_{index:03d}',purpose=purpose,phase=phase,active_arm_id=actor,
                stage_id=None if self.driver.current is None else self.driver.current['stage_id'],
                native_scene_identity=self.scene_identity(),
                start_native_step=begin,end_native_step=int(self.scene.t),actual_intervals=count,
                actual_holder_arm_id=holder,contact_permission_arm_id=permission,control=cr,payload=pr,
                contacts=ct,support=sr,handling_checkpoint=checkpoint,handling_file=artifact(hp),raw_history=artifact(path))
            self.phases.append(item);records.append(item);supports.append(raw)
            save(self.output/f'phase_{index:03d}_receipt.json',item)
            self.check()
            self.progress(purpose,phase,actor,int(self.scene.t),total-offset-count)
        return records,np.concatenate(supports,axis=0)

    def progress(self,purpose,phase,actor,step,remaining):
        # Host counters only. This file is monitoring context, never a PASS receipt.
        value=dict(schema_version='tier-native-progress-v1',purpose=purpose,phase=phase,
                   active_arm_id=actor,native_scene_step=step,simulation_s=step*DT,
                   remaining_phase_intervals=remaining,physical_acceptance_claimed=False)
        (self.output/'PROGRESS.json').write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8')

    def hold(self,purpose,phase,actor,support,duration,permission=None):
        from physics.deferred_gpu_tail import tail_commands
        table=tail_commands(self.q,self.names,actor,self.c['finger_bounds'],
                            phase='release' if phase=='release' else 'released_settle',duration_s=duration)
        return self.phase(purpose,phase,actor,support,table,permission=permission)

    def validate_stage(self,index,stage,goal):
        from physics.saved_motion_planning import validate_saved_motion
        actor=stage['active_arm_id'];kind=stage['kind'];obs=copy.deepcopy(self.observation)
        request=dict(schema_version='student-motion-request-v1',stage_id=stage['stage_id'],kind=kind,
            world=self.driver.world,scenario=self.driver.scenario,observation=obs,active_arm_id=actor,
            command_start_q_rad=self.commands[actor]['q_rad'],goal=goal,
            planning=dict(seed=self.driver.seed+index,max_queries=self.driver.max_queries,speed_scale=self.driver.speed_scale,
                endpoint_attempts=self.driver.endpoint_attempts,options=self.driver.options,use_edge_timing_hook=False,query_body_tilt_limit_deg=7.5))
        if kind=='carry':request.update(held_body=self.driver.held.to_record(),placement_goal=self.driver.goal)
        before=self.snapshot()
        response=validate_saved_motion(request,self.plan,index,previous_commands=self.commands,
            output=self.output/(stage['stage_id']+'__planning'),source_check=self.c['source_check'],wall_check=self.check,
            fixture_readback=self.driver.fixture_readback)
        require(before==self.snapshot() and response.get('durable_path_validation',{}).get('passed') is True
                and response.get('source_bindings_verified') is True and response.get('physical_steps')==0
                and response.get('runtime_ik_calls')==response.get('runtime_rrt_calls')==0
                and response.get('cleanup_errors')==[] and response.get('saved_bundle_sha256')==self.plan.sha256
                and response.get('request_sha256')==digest(request)
                and response.get('provided_adapter_plan_path_invocations')==0
                and all(response.get(key) is True for key in
                        ('observed_start_valid','command_start_valid','observed_to_command_edge_valid')),
                'Saved native boundary validation changed the live scene or failed')
        return response['plan']['trajectory']

    def return_donor(self,stage,gate,release):
        from physics.provided_robot_return import perform_released_robot_return
        from physics.episode_contacts import classify_episode_live_contacts
        from physics.live_contact_coverage import live_contact_coverage_stamp
        from physics.substep_observer import _copy_contacts
        arm=stage['active_arm_id'];schedule=self.plan.kinematic_return_after(stage['stage_id'])
        require(schedule is not None,'Intermediate donor has no exact own-path reverse schedule')
        before=self.snapshot()
        def retarget(name,target):
            native=self.c['arms'][name];target=np.asarray(target,dtype=np.float32)
            tau=_host(self.c['gravity'][name].torques()).reshape(9)
            native.set_dofs_act_bias(tau,-self.c['kp'],-self.c['kv'])
            native.control_dofs_position_velocity(target,np.zeros(9,np.float32))
            motor=_host(native.get_dofs_control_force()).reshape(9)
            error=float(np.max(np.abs(motor-np.clip(tau,-self.c['effort'],self.c['effort']))))
            require(np.all(np.abs(tau)<=self.c['effort']) and np.all(np.abs(motor)<=self.c['effort']+1e-5)
                    and error<=.005,'Original donor-home gravity-PD contract failed')
            return dict(passed=True,gravity_effort=tau.tolist(),effort_identity_error=error,
                        derivative_history='Fresh native qd0 after the authorized zero-physics return.')
        def contacts():
            self.scene.rigid_solver.detect_collision()
            raw=_copy_contacts(self.scene.rigid_solver.collider);obs=self.c['observe']()
            classified,_=classify_episode_live_contacts(raw,owners=self.c['owners'],components_by_geom=self.c['components'],
                pads_by_arm=self.c['pads'],stations_by_geom=self.c['station_map'],body_pose=obs['body_world'],
                finger_poses_by_arm={n:v['finger_world'] for n,v in obs['arms'].items()},
                phase='released_settle',active_holder_arm_id=None)
            stamp=live_contact_coverage_stamp(self.scene,self.c['coverage'],automatic_grasp=self.helper)
            bad=[row for row in classified if row['category']=='UNALLOWED_CONTACT']
            support=gate['support_id']
            present=any(row['category']=='named_tray_bottom_support_top' and row.get('support_station_id')==support
                        for row in classified)
            require(stamp['policy_valid_now'] is True and stamp['native_contact_count']==len(raw['geom_a'])
                    and len(classified)==len(raw['geom_a']) and not bad and present,
                    'Original post-return endpoint contact/support proof failed')
            return dict(passed=True,fresh_detection=True,physical_steps=0,native_contact_count=len(classified),
                classified_contact_count=len(classified),unallowed_contact_count=0,support_id=support,
                named_support_contact_present=True,classified_contacts=classified,policy_stamp=stamp,
                scope='Actual source-checked endpoint only. No swept collision or physical-return claim.')
        after,receipt=perform_released_robot_return(return_schedule=schedule,
            transcript_path=self.output/(stage['stage_id']+'__kinematic_return_native.jsonl'),
            limits=self.c['limits_by_arm'][arm],allowed_initial_tracking_error_rad=self.c['execution']['final_joint_error_rad'],
            wall_check=self.check,scene=self.scene,arms=self.c['arms'],cup=self.c['tray'],beads=self.c['beads'],
            identities=self.c['ids'],helper=self.helper,arm_id=arm,after_stage_id=stage['stage_id'],
            target_command=schedule['target_command'],initial_observation=self.initial,
            before_observation=self.observation,observe=self.c['observe'],released_gate=gate,release_event=release,
            pre_reset_rest=dict(arms=gate['all_arms_rest'],body=gate['body_rest']),
            source_check=self.c['source_check'],retarget_control=retarget,check_contacts=contacts)
        self.observation=copy.deepcopy(after)
        actual=self.snapshot()
        require(before['free_bodies']==actual['free_bodies'],'Donor return changed cup/bead q7/qd6 or native time')
        self.commands[arm]=copy.deepcopy(schedule['target_command'])
        self.q=np.asarray([v for n in self.names for v in self.commands[n]['q_rad']+self.commands[n]['finger_q_m']],np.float64)
        item=dict(event='donor_return',arm_id=arm,stage_id=stage['stage_id'],before=before,after=actual,receipt=receipt)
        self.events.append(item);self.returns.append(item)
        save(self.output/(stage['stage_id']+'__actual_return.json'),item)

    def run(self):
        from physics.provided_manipulation import grasp_endpoint_candidates,placement_endpoint,assess_place_ready
        from physics.automatic_manipulation import pick
        from physics.deferred_gpu_tail import assess_support_history,released_numeric_gate
        self.hold('initial_settle','settle','arm_0',self.plan.support_route[0],.5)
        require(int(self.scene.t)==INITIAL_TICKS,'Initial stabilization must finish at tick200')
        for index,signature in enumerate(self.plan.stage_signatures):
            self.check();stage=self.plan.stage(index)
            require((stage['stage_id'],stage['kind'],stage['active_arm_id'])==signature,'Stored stage identity changed')
            actor=stage['active_arm_id'];kind=stage['kind'];leg=self.plan.required_arm_ids.index(actor)
            source=self.plan.support_route[leg];destination=self.plan.support_route[leg+1]
            op=dict(stage_id=stage['stage_id'],kind=kind,active_arm_id=actor,support_id=source,destination=destination)
            self.driver.current=op
            if kind=='approach':
                require(self.helper.holder is None and self.driver.held is None,'Receiver approached an already held cup')
                endpoint=grasp_endpoint_candidates(self.c['model'],self.observation['body_world'])[0].to_record()
                goal=dict(kind='world_hand_pose',position_m=endpoint['position_m'],quaternion_wxyz=endpoint['quaternion_wxyz'])
                self.driver.endpoint_hand_pose=endpoint
            else:
                require(self.helper.holder==actor and self.driver.held is not None,'Carry lacks the actual receiver acquisition')
                self.driver.goal=placement_endpoint(self.driver.world,self.driver.scenario,self.driver.held,destination)
                endpoint=self.driver.goal['hand_world']
                goal=dict(kind='world_hand_pose',position_m=endpoint['position_m'],quaternion_wxyz=endpoint['quaternion_wxyz'])
            tr=self.validate_stage(index,stage,goal)
            table=trajectory_commands(tr,self.commands,actor,self.c['limits_by_arm'][actor])
            begin=int(self.scene.t)
            _,raw_support=self.phase(stage['stage_id'],kind,actor,source if kind=='approach' else destination,table)
            self.commands[actor]['q_rad']=list(tr['waypoints'][-1])
            if kind=='approach':
                self.hold(stage['stage_id']+'__endpoint_settle','settle',actor,source,.5)
                before=self.snapshot()
                gate=pick(self.driver,self.state(),op)
                self.observation=copy.deepcopy(self.c['observe']())
                after=self.snapshot()
                require(before['free_bodies']==after['free_bodies'],'Actual acquisition changed native free-body state')
                event=dict(event='acquire',arm_id=actor,stage_id=stage['stage_id'],support_id=source,
                           before=before,after=after,receipt=copy.deepcopy(self.driver.automatic_attachment_receipt),gate=gate)
                self.events.append(event)
                if self.last_release is not None:
                    donor=self.last_release['arm_id']
                    require(self.last_release['support_id']==source and self.last_release['gate']['passed']
                            and after['holder_arm_id']==actor,'Actual supported handoff lineage differs')
                    self.handoffs.append(dict(from_arm=donor,to_arm=actor,support_id=source,
                        released_supported_and_resting=True,same_sample_acquired=True,
                        release_event_sha256=digest(self.last_release),acquire_event_sha256=digest(event),
                        sample_ids=[v['id'] for v in self.c['ids']]))
                self.gates.append(dict(stage_id=stage['stage_id'],kind=kind,active_arm_id=actor,gate=gate))
            else:
                support=assess_support_history(raw_support,start_ordinal=begin*SUBSTEPS,phase='carry',holder_arm_id=actor)
                load=support['maximum_consecutive_unsupported_held_s']
                require(load>=.5,'Original carried unsupported-load window is incomplete')
                self.hold(stage['stage_id']+'__supported_hold','place',actor,destination,.5)
                ready=assess_place_ready(self.observation,self.rows,self.c['settings'],self.c['rest_policy'],self.driver.goal)
                require(ready['passed'],'Actual support/body placement readiness failed')
                before=self.snapshot();snapshot=before['grasp']
                def supported(actual):
                    self.check();require(actual==snapshot,'Stale actual supported release snapshot')
                    return dict(passed=True,native_clock=actual['native_clock'],body_entity_idx=actual['body_entity_idx'],
                                support_id=destination,evidence_sha256=digest(ready))
                release=self.helper.release(actor,support_ready=supported)
                self.driver.automatic_attachment_receipt=None;self.driver.held=None
                self.observation=copy.deepcopy(self.c['observe']());after=self.snapshot()
                require(before['free_bodies']==after['free_bodies'],'Actual release changed native free-body state')
                event=dict(event='release',arm_id=actor,stage_id=stage['stage_id'],support_id=destination,
                           before=before,after=after,receipt=release,ready=ready)
                self.events.append(event)
                self.hold(stage['stage_id']+'__release','release',actor,destination,1.,permission=actor)
                self.hold(stage['stage_id']+'__released_settle','released_settle',actor,destination,1.)
                targets={n:self.q[9*j+7:9*j+9].copy() for j,n in enumerate(self.names)}
                extra=0.
                while True:
                    hard=released_numeric_gate(self.rows,settings=self.c['settings'],rest_policy=self.c['rest_policy'],
                        goal=self.driver.goal,arm_order=self.names,finger_targets=targets,load_bearing_s=load,
                        model=self.c['model'],identities=self.c['ids'],strict_material=False)
                    if hard['bead_bulk_rest']['passed']:
                        gate=released_numeric_gate(self.rows,settings=self.c['settings'],rest_policy=self.c['rest_policy'],
                            goal=self.driver.goal,arm_order=self.names,finger_targets=targets,load_bearing_s=load,
                            model=self.c['model'],identities=self.c['ids'],strict_material=True)
                        break
                    require(extra+.5<=5.,'Original additional material settling allowance exhausted')
                    self.hold(stage['stage_id']+'__material_wait','released_settle',actor,destination,.5);extra+=.5
                for j,name in enumerate(self.names):self.commands[name]['finger_q_m']=self.q[9*j+7:9*j+9].tolist()
                self.gates.append(dict(stage_id=stage['stage_id'],kind=kind,active_arm_id=actor,gate=gate,
                                       carried_support=support,additional_material_wait_s=extra))
                self.last_release=dict(arm_id=actor,support_id=destination,gate=gate,release=event)
                if leg<len(self.plan.required_arm_ids)-1:self.return_donor(stage,gate,release)
            self.completed.append(stage['stage_id'])
            self.driver.current=None
            save(self.output/(stage['stage_id']+'__device_gate.json'),self.gates[-1])
        self.check()
        final=self.c['observe']();final_handling=self.handling.finish()
        self.record.update(passed=True,full_mission_physical_verified=True,status='TIER_DEVICE_ORIGINAL_PHASES_PASS',
            actual_final_observation=final,actual_final_free_bodies=free_body_state(self.c),
            actual_terminal_goal=copy.deepcopy(self.driver.goal),
            final_handling=final_handling,final_native_step=int(self.scene.t),
            completion_clock=check_completion_clock(INITIAL_TICKS,int(self.scene.t),self.contract['completion_limit_ticks']),
            original_rest_thresholds_unchanged=True,no_online_replanning=True,
            source_hashes={str(Path(__file__).resolve()):sha(__file__)})
        self.record['completion_summary']=completion_summary(self.record)
        require(all(self.record['completion_summary']['checks'].values()),'A required full-tier predicate failed')
        return self.record


def completion_summary(record):
    """Derive protocol fields from separate source-bound native records."""
    contract=record['task_contract'];arms=list(contract['required_arm_ids'])
    phases=record['phases'];gates=record['stage_gates'];events=record['boundary_events'];returns=record['donor_returns']
    final=record['actual_final_observation'];handling=record['final_handling']
    initial_ids=[v['id'] for v in record['original_sample_ids']];final_ids=[v['id'] for v in final['materials']]
    acquisitions=[e for e in events if e['event']=='acquire'];releases=[e for e in events if e['event']=='release']
    carried=[g for g in gates if g['kind']=='carry'];last=carried[-1]['gate']
    signatures=[arm+'__'+kind for arm in arms for kind in ('approach','carry')]
    cursor=0;continuous=True
    for phase in phases:
        continuous=continuous and phase['start_native_step']==cursor and phase['end_native_step']==cursor+phase['actual_intervals']
        cursor=phase['end_native_step']
    completion=record['completion_clock'];ticks=completion['native_end_step']-completion['first_motion_start_step']
    clock=check_completion_clock(completion['first_motion_start_step'],completion['native_end_step'],contract['completion_limit_ticks'])
    checks=dict(
        collision_free=all(p['contacts']['passed'] and p['contacts']['category_counts'].get('UNALLOWED_CONTACT')==0 for p in phases),
        sample_retained=all(p['payload']['passed'] for p in phases) and initial_ids==final_ids and len(initial_ids)==4,
        tracking_passed=all(p['control']['passed'] for p in phases),
        finite_state=all(p['control']['passed'] and p['payload']['passed'] for p in phases),
        handling_passed=handling['passed'] and handling['qualified_holder_id'] is None
            and handling['supported_pickup_count']==handling['release_count']==len(arms)
            and handling['PRE_samples']==handling['POST_samples']==cursor*SUBSTEPS,
        supported_release=len(releases)==len(carried)==len(arms) and all(e['ready']['passed'] and g['gate']['passed'] for e,g in zip(releases,carried)),
        final_rest=last['passed'] and last['strict_material'] is True and last['expected_material_endpoints']==201,
        original_initial_state=record['original_initial_observation']['native_scene_step']==0
            and record['original_initial_fingerprint']['status']=='FRESH_INITIAL_OBSERVABLES_CAPTURED'
            and not record['original_initial_fingerprint']['missing_required_fields']
            and phases[0]['purpose']=='initial_settle' and phases[0]['start_native_step']==0 and phases[0]['end_native_step']==200,
        immutable_terminal=last['support_id']==contract['terminal_support_id']
            and carried[-1]['active_arm_id']==arms[-1]
            and record['actual_terminal_goal'].get('seating',{}).get('nominal_goal',record['actual_terminal_goal'])['body_world']
                ==contract['terminal_support_target']['body_world'],
        complete_native_lineage=bool(continuous and cursor==record['final_native_step']==final['native_scene_step']
            and all(e['before']['free_bodies']==e['after']['free_bodies'] for e in events)),
        all_required_phases=record['completed_stage_ids']==signatures
            and [e['arm_id'] for e in acquisitions]==[e['arm_id'] for e in releases]==arms
            and [e['arm_id'] for e in returns]==arms[:-1]
            and len(record['handoffs'])==max(0,len(arms)-1))
    return dict(completed_stage_ids=list(record['completed_stage_ids']),
        completed_arm_ids=[g['active_arm_id'] for g in carried],
        handoffs=[{key:h[key] for key in ('from_arm','to_arm','support_id','released_supported_and_resting','same_sample_acquired')}
                  for h in record['handoffs']],
        donor_returns=[dict(arm_id=e['arm_id'],policy='kinematic_return',
            original_path_reversed=e['receipt']['return_schedule_sha256']==digest(e['receipt']['return_schedule']),
            physical_steps=e['receipt']['physical_steps']) for e in returns],
        checks=checks,initial_sample_ids=initial_ids,final_sample_ids=final_ids,
        final_rest_steps=last['expected_material_endpoints'],completion_ticks=ticks,completion_clock=clock)


def validate_phase_schedule(phases,contract,plan_record=None):
    """Require every supplied physical phase without resetting the native clock."""
    require(type(phases) is list and phases,'Missing real native phase schedule')
    first=phases[0]
    require(first['purpose']=='initial_settle' and first['phase']=='settle' and first['stage_id'] is None
            and first['start_native_step']==0 and first['end_native_step']==INITIAL_TICKS
            and first['actual_intervals']==INITIAL_TICKS and first['actual_holder_arm_id'] is None,
            'The original initial 200 ticks must occur exactly once')
    signatures=[arm+'__'+kind for arm in contract['required_arm_ids'] for kind in ('approach','carry')]
    cursor=INITIAL_TICKS;position=1
    stages={stage['stage_id']:stage for stage in plan_record['stages']} if plan_record is not None else {}
    for stage_id in signatures:
        actor=stage_id.split('__')[0];kind=stage_id.split('__')[1];group=[]
        while position<len(phases) and phases[position]['stage_id']==stage_id:
            phase=phases[position]
            require(type(phase['start_native_step']) is int and type(phase['end_native_step']) is int
                    and type(phase['actual_intervals']) is int and phase['actual_intervals']>0
                    and phase['start_native_step']==cursor
                    and phase['end_native_step']==cursor+phase['actual_intervals']
                    and phase['active_arm_id']==actor,'A phase skipped time or changed its serial actor')
            group.append(phase);cursor=phase['end_native_step'];position+=1
        require(group,'A required approach/carry has no native physical phases')
        motion=[]
        while group and group[0]['phase']==kind:motion.append(group.pop(0))
        require(motion,'A required stored trajectory was omitted')
        if stages:
            tr=stages[stage_id]['trajectory']
            duration=tr['duration_s']
            expected=math.ceil(duration/DT) if duration else 200
            require(sum(p['actual_intervals'] for p in motion)==expected,'Native trajectory length differs from the exact saved table')
        if kind=='approach':
            require(len(group)==1 and group[0]['phase']=='settle' and group[0]['actual_intervals']==200
                    and all(p['actual_holder_arm_id'] is None and p['contact_permission_arm_id'] is None
                            for p in motion+group),'Original approach/settling phases differ')
        else:
            require(len(group)>=3 and [(p['phase'],p['actual_intervals']) for p in group[:3]]
                    ==[('place',200),('release',400),('released_settle',400)],
                    'Original supported placement, release or released-settle duration differs')
            require(all(p['actual_holder_arm_id']==actor and p['contact_permission_arm_id']==actor for p in motion+group[:1])
                    and group[1]['actual_holder_arm_id'] is None and group[1]['contact_permission_arm_id']==actor
                    and all(p['actual_holder_arm_id'] is None and p['contact_permission_arm_id'] is None for p in group[2:]),
                    'Actual holder/contact permission changed outside a supported release')
            extra=group[3:]
            require(len(extra)<=10 and all(p['phase']=='released_settle' and p['actual_intervals']==200 for p in extra),
                    'Original additional material settling allowance changed')
    require(position==len(phases),'Unexpected stages after the final terminal or a final return was added')
    return cursor


def validate_device_tier_completion(record,plan_record=None):
    require(type(record) is dict and record.get('schema_version')==SCHEMA and record.get('passed') is True
            and record.get('full_mission_physical_verified') is True and record.get('failure') is None,
            'Missing complete persistent native tier proof')
    require(record['task_contract_sha256']==digest(record['task_contract'])
            and record.get('persistent_scene') is True and record.get('legacy_dispatcher_commits_claimed') is False,
            'Tier model/contract identity differs')
    if plan_record is not None:
        require(record['task_contract']==plan_record['task_contract']
                and record['saved_bundle_sha256']==plan_record['bundle_sha256'],
                'Native result belongs to a different tier plan')
    from physics.tier_mission_plan import validate_task_contract
    validate_task_contract(record['task_contract'])
    require(record['model']=='serial-active-three-native-arm-57dof'
            and record['native_counts']['dofs']==57 and record['native_counts']['q_coordinates']==62,
            'Full tier requires all three actual native robots and original free bodies')
    identity=record['native_scene_identity']
    require(type(identity) is list and len(identity)==4 and all(type(v) is int and v>0 for v in identity),
            'Missing actual native scene/solver/state identity')
    phases=record['phases'];require(phases,'Missing actual phase lineage')
    end=validate_phase_schedule(phases,record['task_contract'],plan_record)
    require(end==record['final_native_step'],'Final native clock differs from the complete physical phase schedule')
    for phase in phases:
        count=phase['actual_intervals']
        require(type(count) is int and count>0 and all(phase[k].get('passed') is True for k in ('control','payload','contacts','support','handling_checkpoint')),
                'A native phase has incomplete or failed original checks')
        require(phase['contacts']['PRE_samples']==phase['contacts']['POST_samples']==count*SUBSTEPS,
                'Original actual16 PRE/POST phase coverage is incomplete')
        require(phase['native_scene_identity']==identity and phase['control']['arm_order']==['arm_0','arm_1','arm_2']
                and phase['control']['controlled_dof_count']==27
                and phase['control']['start_native_step']==phase['support']['start_native_step']==phase['start_native_step']
                and phase['control']['end_native_step']==phase['support']['end_native_step']==phase['end_native_step']
                and phase['payload']['actual_POST_samples']==count
                and phase['payload']['original_bead_ids']==record['original_sample_ids'],
                'Actual phase model, identity or native clock differs')
        require(phase['handling_checkpoint']['PRE_samples']==phase['handling_checkpoint']['POST_samples']==phase['end_native_step']*SUBSTEPS
                and phase['handling_checkpoint']['last_accepted_global_substep']==phase['end_native_step']*SUBSTEPS,
                'Persistent handling skipped or reset a native phase')
        for name in ('control','payload','contacts','support','handling_checkpoint'):
            pins=phase[name]['source_hashes']
            require(type(pins) is dict and pins and all(sha(p)==h for p,h in pins.items()),
                    'An original phase audit source binding changed')
        for key in ('raw_history','handling_file'):
            require(Path(phase[key]['file']).resolve().is_relative_to(Path(record['native_output_directory']).resolve()),
                    'A native phase history was assigned from another directory')
            require(sha(phase[key]['file'])==phase[key]['sha256'],'Native phase artifact changed')
        with np.load(phase['raw_history']['file'],allow_pickle=False) as raw:
            require(set(raw.files)=={'control','payload','support','command_position','command_velocity'}
                    and raw['control'].shape==(count,27,13) and raw['payload'].shape==(count,46)
                    and raw['support'].shape==(count*SUBSTEPS,6) and raw['support'].dtype==np.dtype('int64')
                    and raw['command_position'].shape==raw['command_velocity'].shape==(count,27)
                    and all(np.isfinite(raw[key]).all() for key in raw.files),
                    'Native phase raw array ABI is incomplete or nonfinite')
        require(phase['handling_checkpoint']['prefix_guard']==record['final_handling']['prefix_guard'],
                'Persistent handling state was reset at a handoff')
    for event in record['boundary_events']:
        require(event['before']['native_scene_identity']==event['after']['native_scene_identity']==identity
                and event['before']['free_bodies']==event['after']['free_bodies'],
                'A zero-time boundary replaced native state or changed cup/bead q7/qd6')
    for name,expected in record['source_hashes'].items():require(sha(name)==expected,'Tier implementation source changed')
    summary=completion_summary(record)
    require(summary==record['completion_summary'] and all(summary['checks'].values()),
            'Native completion predicates or transport clock failed')
    return True


def execute_device_tier_mission(context,*,output,wall_deadline):
    owner=_Mission(context,output,wall_deadline)
    started=time.perf_counter()
    try:
        result=owner.run()
        validate_device_tier_completion(result,owner.plan.record())
        return result
    except BaseException as exc:
        owner.record.update(passed=False,full_mission_physical_verified=False,
                            failure=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        owner.record.update(actual_native_end_step=int(owner.scene.t),host_wall_s=time.perf_counter()-started)
        save(Path(output)/'TIER_DEVICE_RESULT.json',owner.record)
