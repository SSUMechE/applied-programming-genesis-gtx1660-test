"""Connect student motion decisions to the existing persistent physical loop.

Only true operation endpoints are supplied to the selected student strategy.
Instructor place may make a separately logged bounded local Z correction after
the unchanged student carry path. All acceptance uses measured native contacts.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from transport_challenge.contracts import ContractError, digest, point
from transport_challenge.episode_plan import EpisodePlan
from transport_challenge.joint_planning import JointLimits, JointTrajectory
from physics.provided_manipulation import (grasp_endpoint_candidates, pick_actions, place_actions,
    placement_endpoint, assess_pick, assess_place_ready, assess_place_release, teacher_policy, supported_preload_action)

BEAD_SETTLE_HOLD_S=.5
BEAD_SETTLE_EXTRA_LIMIT_S=5.


def panda_limits(case, precision='32'):
    """Pinned Panda limits in the explicitly selected native precision."""
    if precision not in ('32', '64'):raise ContractError('Panda precision must be32 or64')
    bounds=((-2.8973,2.8973),(-1.7628,1.7628),(-2.8973,2.8973),(-3.0718,-.0698),
        (-2.8973,2.8973),(-.0175,3.7525),(-2.8973,2.8973))
    if precision == '64':
        from importlib.metadata import distribution
        import xml.etree.ElementTree as ET
        if case['robot_model'] != 'xml/franka_emika_panda/panda.xml':
            raise ContractError('Precision64 requires the pinned Panda model')
        model=Path(distribution('genesis-world').locate_file('genesis/assets'))/case['robot_model']
        if hashlib.sha256(model.read_bytes()).hexdigest() != 'e502e07fcef6219024be02dfb9cd65d5c194b4767adf381ca51f10172ff8a32d':
            raise ContractError('Pinned Panda source changed')
        xml=ET.parse(model).getroot()
        default=xml.find("./default/default[@class='panda']/joint").attrib['range']
        actual=tuple(tuple(map(float,xml.find(".//joint[@name='joint%d']" % i).attrib.get('range',default).split())) for i in range(1,8))
        if xml.find('compiler').attrib.get('angle') != 'radian' or actual != bounds:
            raise ContractError('Panda source ranges differ from the fixed model')
    dtype=np.float64 if precision == '64' else np.float32
    return JointLimits(tuple(tuple(float(dtype(v)) for v in pair) for pair in bounds),
        case['motion_limits']['velocity_rad_s'],case['motion_limits']['acceleration_rad_s2'])


SINGLE_BATCH_STAGES=(
    ('arm_0__approach','approach','arm_0'),('arm_0__carry','carry','arm_0'),
    ('arm_0__return_initial','retreat','arm_0'),('arm_1__approach','approach','arm_1'),
    ('arm_1__carry','carry','arm_1'),('arm_1__return_initial','retreat','arm_1'),
    ('arm_2__approach','approach','arm_2'),('arm_2__carry','carry','arm_2'))


def assess_current_student_single_batch(execution,observation,*,route,initial_supply,
                                      final_goal,currently_held,execution_limits):
    """Assemble one current task's completion from trusted measured evidence.

    This pure predicate does not authenticate fabricated records. The runtime
    owns each native interval, source binding and grasp/place predicate before
    this function is called. It cannot approve other materials, throughput,
    continuous collision safety, laboratory hardware, or student release.
    """
    failures=[];carry_supports=[];load_by_arm={}
    try:
        if (execution.get('mode')!='mission' or execution.get('initialized') is not True
                or execution.get('operation_sequence_complete') is not True
                or execution.get('current_held_body') is not None or currently_held is not None
                or execution.get('all_requested_student_transport_motion_from_selected_strategy',
                    execution.get('all_motion_from_selected_strategy')) is not True
                or execution.get('reference_planner_fallback') is not False
                or execution.get('supplied_intermediate_postures') is not False):
            raise ContractError('Not a completed released student mission')
        if (not isinstance(route,(list,tuple)) or len(route)!=4
                or any(type(v) is not str or not v for v in route) or len(set(route))!=4
                or execution.get('support_route')!=list(route)
                or execution.get('current_support_id')!=route[-1]):
            raise ContractError('Source, two relays and final support are not bound')
        completed=execution.get('completed',[]);plans=execution.get('planned_motions',[])
        gates=execution.get('provided_gates',[]);pauses=execution.get('planning_pauses',[])
        if any(len(rows)!=8 for rows in (completed,plans,gates,pauses)):
            raise ContractError('Completion requires eight operations, plans, gates and pauses')
        signatures=[(r.get('stage_id'),r.get('kind'),r.get('active_arm_id')) for r in completed]
        if tuple(signatures)!=SINGLE_BATCH_STAGES:
            raise ContractError('Current eight-stage order or arm identities differ')
        for rows in (plans,gates):
            if [(r.get('stage_id'),r.get('kind'),r.get('active_arm_id')) for r in rows]!=signatures:
                raise ContractError('Plan or gate identity differs from completed stages')
        if [r.get('stage_id') for r in pauses]!=[s[0] for s in SINGLE_BATCH_STAGES]:
            raise ContractError('Planning pause identities differ')
        def valid_sha(v):
            return type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)
        for plan,pause in zip(plans,pauses):
            if (plan.get('supplied_intermediate_postures') is not False
                    or not all(valid_sha(plan.get(k)) for k in
                               ('request_sha256','trajectory_sha256','plan_path_source_sha256'))
                    or not all(pause.get(k) is True for k in
                               ('native_state_and_clock_unchanged','dispatcher_unchanged','source_bindings_verified'))):
                raise ContractError('A source-bound student plan or unchanged planning pause is missing')
        if len({r['plan_path_source_sha256'] for r in plans})!=1:
            raise ContractError('Selected strategy source changes between motions')
        supply={m['id']:(m['mass_kg'],m['entity_idx']) for m in initial_supply}
        if (len(supply)!=4 or len(initial_supply)!=4
                or any(type(m['id']) is not str or not m['id']
                    or type(m['entity_idx']) is not int or m['entity_idx']<0
                    or isinstance(m['mass_kg'],bool) or not isinstance(m['mass_kg'],(int,float))
                    or not math.isfinite(m['mass_kg']) or m['mass_kg']<=0 for m in initial_supply)
                or len({m['entity_idx'] for m in initial_supply})!=4):
            raise ContractError('One current bead batch has exactly four original IDs')
        previous_step=0;previous_time=0.;last_gate=None
        for index,(done,entry) in enumerate(zip(completed,gates)):
            if (type(done.get('native_scene_step')) is not int or done['native_scene_step']<=previous_step
                    or not math.isfinite(done['simulation_s']) or done['simulation_s']<=previous_time):
                raise ContractError('Completion clocks are not strictly increasing')
            previous_step=done['native_scene_step'];previous_time=done['simulation_s']
            gate=entry.get('gate',{})
            if gate.get('passed') is not True:
                raise ContractError('A provided stage gate did not pass')
            arm=done['active_arm_id'];arm_index=int(arm[-1]);kind=done['kind']
            expected_support=route[arm_index+1] if kind in ('carry','retreat') else route[arm_index]
            if entry.get('support_id')!=expected_support:
                raise ContractError('A stage is bound to the wrong physical support')
            if kind=='approach':
                if (gate.get('source_support_observed') is not True
                        or gate.get('all_arms_rest',{}).get('passed') is not True
                        or gate.get('body_rest',{}).get('passed') is not True
                        or any(gate.get(k,0.)<=0 for k in
                               ('bilateral_window_s','stable_contents_window_s','source_support_window_s'))):
                    raise ContractError('Recipient/source supported grasp evidence is incomplete')
            if kind=='carry':
                if (gate.get('schema_version')!='sampled-released-relay-gate-v1'
                        or gate.get('support_id')!=expected_support or gate.get('active_arm_id')!=arm
                        or gate.get('failures')!=[] or gate.get('persistent_material_ids')!=sorted(supply)
                        or gate.get('all_arms_rest',{}).get('passed') is not True
                        or gate.get('body_rest',{}).get('passed') is not True
                        or type(gate.get('expected_rows')) is not int or gate['expected_rows']<1
                        or gate.get('observed_rows')!=gate['expected_rows']
                        or not math.isfinite(gate['window_s']) or gate['window_s']<=0
                        or not math.isfinite(gate['dt_s']) or gate['dt_s']<=0
                        or abs(gate['expected_rows']*gate['dt_s']-gate['window_s'])>1e-9
                        or abs(gate['ending_s']-done['simulation_s'])>1e-9
                        or not math.isfinite(gate.get('load_bearing_interval_s',-1.))
                        or gate['load_bearing_interval_s']<gate['window_s']):
                    raise ContractError('A released carrying stage lacks stable support, retention or load bearing')
                carry_supports.append(expected_support);load_by_arm[arm]=gate['load_bearing_interval_s'];last_gate=gate
        if carry_supports!=list(route[1:]) or len(load_by_arm)!=3:
            raise ContractError('Both relay placements and the final placement are required')
        if (observation.get('native_scene_step')!=previous_step
                or abs(observation['simulation_s']-previous_time)>1e-9
                or abs(previous_step*last_gate['dt_s']-previous_time)>1e-9):
            raise ContractError('Final observation does not belong to the final completed release')
        final_materials=observation['materials']
        actual={m['id']:(m['mass_kg'],m['entity_idx']) for m in final_materials}
        if len(final_materials)!=4 or actual!=supply:
            raise ContractError('Original material IDs, native entities or masses changed')
        for material in final_materials:point(material['position_world_m'],3)
        if any(m.get('inside_carrier') is not True or
               math.hypot(*point(m['velocity_world_m_s'],3))>last_gate['limits']['content_speed_m_s']
               for m in final_materials):
            raise ContractError('Final material retention or rest failed')
        arms=observation['arms']
        if set(arms)!={'arm_0','arm_1','arm_2'}:
            raise ContractError('Final named arm observations are incomplete')
        for arm in arms.values():
            point(arm['q_all_dofs'],9);v=point(arm['qd_all_dofs'],9)
            if (max(map(abs,v[:7]))>execution_limits['rest_speed_rad_s']
                    or max(map(abs,v[7:]))>.001):
                raise ContractError('Final arm/finger rest failed')
        body=observation['body_world'];nominal=final_goal.get('seating',{}).get('nominal_goal',final_goal)
        if final_goal.get('support_id')!=route[-1] or nominal.get('support_id')!=route[-1]:
            raise ContractError('Final body goal is not the fixed destination support')
        position=math.dist(point(body['position_m'],3),point(nominal['body_world']['position_m'],3))
        actual_q=point(body['quaternion_wxyz'],4);wanted_q=point(nominal['body_world']['quaternion_wxyz'],4)
        norms=math.hypot(*actual_q)*math.hypot(*wanted_q)
        if norms<=0:raise ContractError('Final orientation is undefined')
        angle=2*math.acos(min(1.,abs(sum(a*b for a,b in zip(actual_q,wanted_q)))/norms))
        body_limits=last_gate['body_rest']['limits']
        if (position>last_gate['limits']['body_position_m'] or angle>last_gate['limits']['body_orientation_rad']
                or math.hypot(*point(observation['body_linear_velocity_m_s'],3))>body_limits['maximum_body_linear_speed_m_s']
                or math.hypot(*point(observation['body_angular_velocity_rad_s'],3))>body_limits['maximum_body_angular_speed_rad_s']):
            raise ContractError('Final destination/body rest failed')
    except (ContractError,KeyError,TypeError,ValueError,OverflowError) as exc:
        failures.append(str(exc))
    result={'schema_version':'current-student-single-batch-completion-v1','passed':not failures,
        'completion_scope':'current_student_three_arm_single_batch_v1','failures':failures,
        'expected_stage_ids':[s[0] for s in SINGLE_BATCH_STAGES],'carry_support_ids':carry_supports,
        'load_bearing_interval_by_arm_s':load_by_arm,'physical_handoff_count':2 if not failures else 0,
        'final_observation_sha256':None if failures else digest(observation),'student_release_approved':False,
        'scope':'One current endpoint-planned three-arm bead batch. Trusted runtime must separately verify every native interval and unchanged sources. No throughput, other materials or laboratory approval.'}
    return result


class StudentEpisodeDriver:
    """Serial three-arm single-batch driver. Short modes are labeled unit probes.

    probe_goals is instructor QA input of true terminal goals for short_motions
    only. It is never accepted in the student design or normal mission.
    """
    def __init__(self, *, strategy, case, speed_scale, mode='mission', probe_goals=None,
                 max_queries=25000, endpoint_attempts=32, seed=71, options=None,
                 use_edge_timing_hook=False, simulation_deadline_s=1200.,step_limit=None,fixture_readback='scalar',
                 placement_policy=None,provided_bead_settle=False,private_contact_profile=False,grasp_policy='friction',empty_program=None,
                 saved_plan=None):
        if grasp_policy not in ('friction','automatic') or grasp_policy=='automatic' and mode=='short_motions':
            raise ContractError('Explicit friction/automatic policy required; automatic needs a manipulation mode')
        if grasp_policy=='automatic' and (provided_bead_settle or private_contact_profile):
            raise ContractError('Legacy friction fixture/settling options cannot certify automatic manipulation')
        self.grasp_policy=grasp_policy;self.automatic_grasp=None;self.automatic_attachment_receipt=None;self.automatic_events=[]
        if mode not in ('mission','tier_mission','one_carry','short_motions','pick_place_unit','bead_lift_unit','first_handoff','placement_probe','source_lift_probe','empty_path'):
            raise ContractError('Unknown student execution/probe mode')
        if (placement_policy is not None and (mode!='placement_probe' or placement_policy not in ('fixed','measured'))):
            raise ContractError('Fixed/measured placement policy selection is private placement_probe QA only')
        if (mode=='empty_path')!=(empty_program is not None):
            raise ContractError('Only empty_path accepts an explicit external empty-container program')
        if mode=='empty_path':
            from transport_challenge.empty_container_program import validate_empty_program
            if strategy is not None or grasp_policy!='automatic' or use_edge_timing_hook or speed_scale!=1.:
                raise ContractError('Empty external paths use automatic grasp and exact supplied timing, without a strategy or speed rescaling')
            empty_program=validate_empty_program(empty_program)
        elif saved_plan is not None:
            from physics.saved_motion_planning import SavedMissionPlan
            from physics.one_carry_planning import SavedOneCarryPlan
            from physics.tier_mission_plan import TierMissionPlan
            valid_saved=(type(saved_plan) is SavedMissionPlan and mode=='mission'
                or type(saved_plan) is TierMissionPlan and mode=='tier_mission'
                or type(saved_plan) is SavedOneCarryPlan and mode=='one_carry')
            if (not valid_saved or grasp_policy!='automatic'
                    or strategy is not None or use_edge_timing_hook or provided_bead_settle or private_contact_profile):
                raise ContractError('Saved CPU replay requires automatic full mission with no runtime planner or timing hook')
        elif mode=='one_carry':
            raise ContractError('One-carry requires its explicit saved CPU path and INITIAL recipe')
        elif not callable(getattr(strategy,'plan_path',None)):
            raise ContractError('Selected strategy must implement plan_path')
        if type(speed_scale) not in (int,float) or not math.isfinite(speed_scale) or not 0<speed_scale<=1:
            raise ContractError('Use a finite joint speed scale in (0,1]')
        if any(type(v) is not int or v<1 for v in (max_queries,endpoint_attempts)):
            raise ContractError('Positive planning/endpoint budgets required')
        if type(seed) is not int or type(use_edge_timing_hook) is not bool:
            raise ContractError('Explicit seed and timing selection required')
        if step_limit is not None and (type(step_limit) is not int or step_limit<1):
            raise ContractError('Short step bound must be a positive integer')
        if (mode=='short_motions')!=(probe_goals is not None):
            raise ContractError('Only short_motions accepts declared instructor probe terminal goals')
        if (type(provided_bead_settle) is not bool or provided_bead_settle and mode not in ('mission','first_handoff','placement_probe')):
            raise ContractError('Explicit bead settling requires a provided placement operation')
        self.strategy=strategy;self.saved_plan=saved_plan;self.limits=panda_limits(case);self.speed_scale=float(speed_scale)
        self.post_place_policy='return_home' if saved_plan is None else saved_plan.post_place_policy
        if self.post_place_policy!='return_home' and (grasp_policy!='automatic'
                 or not (mode in ('mission','tier_mission') or mode=='one_carry' and self.post_place_policy=='one_carry')):
            raise ContractError('Changed post-placement policy requires an automatic saved full mission')
        self.post_place_resets=[]
        self.post_place_returns=[]
        self.mode=mode;self.probe_goals=copy.deepcopy(probe_goals);self.max_queries=max_queries
        self.endpoint_attempts=endpoint_attempts;self.seed=seed;self.options=dict(options or {})
        self.use_edge_timing_hook=use_edge_timing_hook;self.simulation_deadline_s=simulation_deadline_s
        self.step_limit=step_limit;self.body_tilt_limit_deg=8.;self.current=None;self.index=0
        self.empty_program=copy.deepcopy(empty_program);self.empty_phase=None;self.empty_resolution=None
        if mode=='empty_path':self.body_tilt_limit_deg=180.
        self.completed=[];self.plans=[];self.gates=[];self.pauses=[];self.held=None;self.initialized=False
        self.support_id=None;self.endpoint_hand_pose=None;self.done=False
        if fixture_readback not in ('scalar','batched'):raise ContractError('Unknown query fixture readback')
        self.fixture_readback=fixture_readback;self.image_sink=None;self.image_captures=[];self.sparse_capture_only=False
        if mode=='bead_lift_unit' and grasp_policy!='automatic':
            raise ContractError('Private bead lift unit requires the unchanged provided automatic grasp')
        self.placement_policy=(placement_policy or 'measured') if mode in ('mission','first_handoff','placement_probe') else 'fixed'
        self.placement_phase=None;self.placement_events=[];self.placement_attempt=0;self.placement_translation_m=0.;self.placement_rotation_rad=0.
        self.placement_anchor=None;self.placement_execution_max={};self.placement_execution_last_step=None
        self.provided_preload_events=[];self.provided_preload_anchors={};self.provided_preload_intervals={}
        self.provided_native_target_rows=0
        self.particle_mode=False;self._particle_runtime=None
        self.provided_bead_settle=provided_bead_settle;self.bead_settle_events=[];self._bead_settle_wait=None
        if type(private_contact_profile) is not bool or private_contact_profile and (mode!='source_lift_probe' or step_limit!=3):
            raise ContractError('Private contact profile requires source_lift_probe and step_limit3')
        self.private_contact_profile=private_contact_profile;self._private_contact_observer=None

    def bind_automatic_grasp(self,helper):
        from physics.provided_automatic_grasp import ProvidedAutomaticGrasp
        if (self.grasp_policy!='automatic' or self.initialized or self.automatic_grasp is not None
                or type(helper) is not ProvidedAutomaticGrasp or helper.holder is not None):
            raise ContractError('Bind the actual unheld automatic helper once before INITIAL')
        self.automatic_grasp=helper

    def require_private_contact_profile(self,world):
        from physics.particle_episode_bridge import request,CACHE_RUNTIME_PROFILE
        if (self.private_contact_profile is not True or self.mode!='source_lift_probe' or self.step_limit!=3
                or world.get('particle_material_request')!=request('sand','64',CACHE_RUNTIME_PROFILE)):
            raise ContractError('Private contact fixture requires exact Sand64/cache source_lift_probe/step_limit3')

    def bind_private_contact_profile(self,observer):
        raise ContractError('This hardware package supports the fixed four-bead tiers only')
        if (self.private_contact_profile is not True or self.initialized or self._particle_runtime is None
                or self._private_contact_observer is not None or type(observer) is not ContactProfileObserver):
            raise ContractError('Bind the source-owned private observer after native particle authentication')
        observer.validate_source_binding()
        observer.bind_snapshot_callback(self._particle_runtime[2])
        self._private_contact_observer=observer

    def _private_contact_initial_plan(self,observation):
        self.require_private_contact_profile(self.world)
        if self._private_contact_observer is None or not self.initialized:
            raise ContractError('Private fixture observer or authenticated INITIAL is missing')
        if (observation['native_scene_step']!=0 or observation['simulation_s']!=0.
                or observation['arms']['arm_0']['q_all_dofs'][7:]!=[.020,.020]):
            raise ContractError('Private fixture requires the geometry-touch fresh INITIAL')
        self._private_contact_observer.capture_initial(observation,self._particle_runtime[2])
        commands=copy.deepcopy(self.initial_commands)
        commands['arm_0']['finger_q_m']=[.010,.010]
        return self._plan(observation,commands,[dict(action_id='private_contact_profile_hold',
            arm_id='arm_0',role='final_hold',kind='hold',duration_s=.0075)],'private_contact_profile_hold')

    def capture_private_contact_profile(self,state,dispatcher_record):
        if (self.private_contact_profile is not True or self._private_contact_observer is None
                or self.provided_native_target_rows!=state['observation']['native_scene_step']):
            raise ContractError('Private endpoint capture requires original native target/observe checks')
        self._private_contact_observer.capture_committed(state,dispatcher_record)

    def bind_particle_runtime(self,contract,coverage_validator,pause_snapshot_reader,pause_bindings_factory):
        """Bind root-owned actual native hooks once, before initial_plan.

        The snapshot reader receives an actual body observation. The factory
        receives the INITIAL observation and full snapshot. Neither callback is
        executed by this binder. Source authentication remains runtime-owned.
        """
        from physics.particle_episode_state import _contract
        if self.initialized or self._particle_runtime is not None:
            raise ContractError('Particle runtime binds exactly once before initialization')
        _contract(contract)
        if not all(callable(f) for f in (coverage_validator,pause_snapshot_reader,pause_bindings_factory)):
            raise ContractError('Actual particle coverage, snapshot and pause-binding hooks required')
        self._particle_runtime=(copy.deepcopy(contract),coverage_validator,pause_snapshot_reader,pause_bindings_factory)

    def _particle_pause_before(self,observation,dispatcher):
        from physics.episode_continuation_protocol import rest_receipt
        raise ContractError('This hardware package supports the fixed four-bead tiers only')
        snapshot=self._particle_runtime[2](observation)
        boundary=rest_receipt(dispatcher.record(),observation,list(self.arm_ids))
        receipt=pause.begin_pause(observation,snapshot,bindings=self._particle_pause_bindings,
            dispatcher_record=dispatcher.record(),boundary_receipt=boundary)
        return dict(receipt=receipt,boundary=boundary)

    def _particle_pause_after(self,context,observation,dispatcher):
        raise ContractError('This hardware package supports the fixed four-bead tiers only')
        snapshot=self._particle_runtime[2](observation)
        return pause.verify_pause(context['receipt'],observation,snapshot,bindings=self._particle_pause_bindings,
            dispatcher_record=dispatcher.record(),boundary_receipt=context['boundary'])

    def initial_plan(self,world,scenario,model,observation,limits_by_arm,finger_bounds):
        from physics.episode_runtime import PROTOCOL
        if self.initialized:raise ContractError('Driver cannot own two live scenes')
        if self.grasp_policy=='automatic' and self.automatic_grasp is None:
            raise ContractError('Automatic manipulation requires the bound actual native helper')
        self.world=copy.deepcopy(world);self.scenario=copy.deepcopy(scenario);self.model=model
        if world.get('payload_mode')=='empty':self.body_tilt_limit_deg=180.
        self.particle_mode='particle_material_request' in world
        if self.particle_mode and self.provided_bead_settle:
            raise ContractError('The bead settling policy cannot alter a native-particle episode')
        if self.particle_mode:
            from physics.particle_student_completion import particle_context
            raise ContractError('This hardware package supports the fixed four-bead tiers only')
            if self._particle_runtime is None:raise ContractError('Bind the actual particle runtime before initial_plan')
            self.initial_supply=particle_context(world,scenario,observation,self._particle_runtime[0])
            if (observation['native_scene_step']!=0 or observation['simulation_s']!=0.
                    or self.initial_supply['initial_native_clock']!=[0,0.,0,0]):
                raise ContractError('Particle driver binds the same fresh native INITIAL at clock zero')
            snapshot=self._particle_runtime[2](observation)
            self._particle_pause_bindings=copy.deepcopy(self._particle_runtime[3](observation,snapshot))
            if (self._particle_pause_bindings['world_sha256']!=digest(world)
                    or self._particle_pause_bindings['body']['id']!=world['object_id']
                    or self._particle_pause_bindings['body']['model_sha256']!=model.geometry_sha256):
                raise ContractError('Particle pause factory differs from the bound initial world/body')
            pause.project_paused_state(observation,snapshot,bindings=self._particle_pause_bindings)
        elif self._particle_runtime is not None or self.mode=='source_lift_probe':
            raise ContractError('Particle runtime and private source lift require explicit world opt-in')
        self.limits_by_arm=limits_by_arm;self.finger_bounds=finger_bounds
        from physics.single_active_boundary import contract_from_observation,robot_coordinates
        self.single_active_arm_contract=contract_from_observation(observation,scenario)
        original_arm_ids=tuple(r['id'] for r in scenario['robots'])
        if original_arm_ids!=('arm_0','arm_1','arm_2'):
            raise ContractError('Use the exact three persistent arm identities')
        if self.single_active_arm_contract is not None and (self.mode!='one_carry' or self.particle_mode or self.saved_plan is None):
            raise ContractError('Single active execution currently requires the explicit saved four-bead one-carry scope')
        self.arm_ids=((self.single_active_arm_contract['active_arm_id'],) if self.single_active_arm_contract else original_arm_ids)
        self.initial_commands={r['id']:{'q_rad':(list(observation['arms'][r['id']]['q_all_dofs'][:7])
                if self.particle_mode and world['particle_material_request'].get('precision','32') == '64'
                else [float(np.float32(v)) for v in r['q_rad']]),
            'finger_q_m':robot_coordinates(observation,r['id'])[7:]} for r in scenario['robots']}
        if self.saved_plan is not None:
            self.saved_plan.verify_initial(world,scenario,self.initial_commands,limits_by_arm)
        if not self.particle_mode:self.initial_supply=copy.deepcopy(observation['materials'])
        self.route=[world['source_support']['id'],*[s['id'] for s in world['relay_supports']],world['goal_support']['id']]
        if self.mode=='tier_mission':
            self.route=list(self.saved_plan.support_route)
        self.support_id=self.route[0];self.endpoint_hand_pose=scenario['endpoint_hand_pose']
        self.queue=[]
        # Private QA endpoints only. No transit q is supplied to the student.
        if self.mode=='tier_mission':
            self.queue=[dict(stage_id=stage_id,kind=kind,active_arm_id=arm,
                support_id=self.route[int(arm[-1])],
                **({'destination':self.route[int(arm[-1])+1]} if kind=='carry' else {}))
                for stage_id,kind,arm in self.saved_plan.stage_signatures]
        elif self.mode=='one_carry':
            self.one_carry_initial_pick=None
            self.queue=[dict(stage_id='arm_0__carry',kind='carry',active_arm_id='arm_0',
                support_id=self.route[0],destination=self.route[1])]
        elif self.mode=='empty_path':
            from physics.episode_runtime import empty_payload_mode
            if not empty_payload_mode(world,scenario) or self.empty_program['arm_id']!=scenario['active_arm_id']:
                raise ContractError('Empty path must bind the explicit empty world and initialized active arm')
            self.queue=[dict(stage_id='empty_external_joint_path',kind='carry',active_arm_id=self.empty_program['arm_id'],support_id=self.route[0])]
            self.empty_phase='initial_settle'
        elif self.mode=='bead_lift_unit':
            if (self.particle_mode or world.get('rigid_runtime_profile')!='rigid16_bead_tc2h_v1'
                    or scenario['active_arm_id']!='arm_0' or self.route[0]!='source_support'):
                raise ContractError('Private bead lift unit requires unchanged bead16 arm0/source case')
            self.queue=[dict(stage_id='arm_0__approach',kind='approach',active_arm_id='arm_0',support_id=self.route[0]),
                dict(stage_id='arm_0__probe_lift',kind='carry',active_arm_id='arm_0',support_id=self.route[0],
                    probe_lift=True,bead_lift_unit=True),
                dict(stage_id='arm_0__carry',kind='carry',active_arm_id='arm_0',support_id=self.route[0],destination=self.route[0])]
        elif self.mode=='source_lift_probe':
            if scenario['active_arm_id']!='arm_0':raise ContractError('Private source lift requires actual arm_0')
            self.queue=[dict(stage_id='arm_0__approach',kind='approach',active_arm_id='arm_0',support_id=self.route[0]),
                dict(stage_id='arm_0__source_lift',kind='carry',active_arm_id='arm_0',support_id=self.route[0],
                    probe_lift=True,particle_source_lift=True)]
        elif self.mode=='placement_probe':
            if scenario['active_arm_id']!='arm_1' or world['source_support']['id']!='relay_2_support':
                raise ContractError('Private placement_probe requires actual arm_1 and relay2 as fixture source')
            self.queue=[dict(stage_id='arm_1__approach',kind='approach',active_arm_id='arm_1',support_id=self.route[0]),
                dict(stage_id='arm_1__probe_lift',kind='carry',active_arm_id='arm_1',support_id=self.route[0],probe_lift=True),
                dict(stage_id='arm_1__carry',kind='carry',active_arm_id='arm_1',support_id=self.route[0],destination=self.route[0])]
        elif self.mode=='short_motions':
            if set(self.probe_goals)!=set(self.arm_ids):raise ContractError('Probe needs one terminal goal per actual arm')
            for arm in self.arm_ids:
                JointTrajectory((self.probe_goals[arm],),(),limits_by_arm[arm])
                self.queue.append(dict(stage_id=arm+'__probe_return',kind='retreat',active_arm_id=arm,
                    support_id=self.route[0],goal_q=list(self.probe_goals[arm])))
        else:
            for i,arm in enumerate(self.arm_ids if self.mode in ('mission','first_handoff') else self.arm_ids[:1]):
                source=self.route[i];destination=self.route[i+1] if self.mode in ('mission','first_handoff') else source
                self.queue.extend([dict(stage_id=arm+'__approach',kind='approach',active_arm_id=arm,support_id=source),
                    dict(stage_id=arm+'__carry',kind='carry',active_arm_id=arm,support_id=source,destination=destination)])
                if self.mode in ('mission','first_handoff') and i<2 and self.post_place_policy not in ('instant_reset','kinematic_return'):
                    if self.post_place_policy=='clear_handoff':
                        stage=self.saved_plan.stage(len(self.queue))
                        self.queue.append(dict(stage_id=arm+'__clear_handoff',kind='retreat',active_arm_id=arm,
                            support_id=destination,goal_q=stage['trajectory']['waypoints'][-1]))
                    else:
                        self.queue.append(dict(stage_id=arm+'__return_initial',kind='retreat',active_arm_id=arm,
                            support_id=destination,goal_q=self.initial_commands[arm]['q_rad']))
            if self.mode=='first_handoff':self.queue=self.queue[:4]
        self.initialized=True
        if self.private_contact_profile:
            return self._private_contact_initial_plan(observation)
        initial_settle_s=(world['particle_material_request']['initial_settle_s'] if self.particle_mode else PROTOCOL['initial_settle_s'])
        if self.mode=='one_carry':
            from physics.episode_runtime import DT
            # First support/material row is at dt. Obtain 201 actual endpoints
            # for the unchanged 200-interval/0.5s rest window. No fake t0 row.
            initial_settle_s=max(initial_settle_s, .5+DT)
        if self.world.get('payload_mode')=='empty':
            from physics.episode_runtime import DT
            # The first support observation is at dt, not time zero. Keep the
            # existing required support window in full before automatic grasp.
            initial_settle_s=max(initial_settle_s,world['physics_settings']['scenario']['bilateral_window_s']+DT)
        return self._plan(observation,self.initial_commands,[{'action_id':'student_initial_settle',
            'arm_id':'arm_0','role':'initial_settle','kind':'hold','duration_s':initial_settle_s}],
            'student_initial_settle')

    def _plan(self,observation,commands,actions,stage_id,*,external_grid=False):
        from physics.episode_runtime import DT
        arms=[{'arm_id':arm,'role':'manipulator','initial_q_rad':commands[arm]['q_rad'],
            'initial_finger_m':commands[arm]['finger_q_m'],'finger_bounds_m':[list(v) for v in self.finger_bounds[arm]],
            'joint_limits':self.limits_by_arm[arm].to_record()} for arm in self.arm_ids]
        steps=sum((round(a['trajectory']['duration_s']/DT) if external_grid else math.ceil(a['trajectory']['duration_s']/DT)) if a['kind']=='trajectory'
            else round(a['duration_s']/DT) for a in actions)
        schema=('external-grid-single-active-joint-episode-plan-v1' if external_grid else 'serial-single-active-joint-episode-plan-v1') if self.single_active_arm_contract else ('external-grid-joint-episode-plan-v1' if external_grid else 'provided-instructor-supported-preload-plan-v1' if any(a['kind']=='supported_preload' for a in actions) else 'serial-joint-episode-plan-v1')
        return EpisodePlan.from_record({'schema_version':schema,'episode_id':stage_id,
            'initial_state_sha256':digest(observation),'dt_s':DT,'deadline_s':steps*DT,'arms':arms,'actions':actions},
            limits_by_arm=self.limits_by_arm)

    def _save(self,state,name,value):
        (Path(state['output'])/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

    def _capture(self,label,state,detail):
        if self.image_sink is not None and not any(r['checkpoint']==label for r in self.image_captures):
            self.image_captures.extend(self.image_sink.capture(label,state['observation'],detail))

    def observe_step(self,state):
        if self.grasp_policy=='automatic':
            from physics.automatic_manipulation import validate_observation,observe_released_wait,observe_owned_open
            validate_observation(self,state['observation'])
            observe_released_wait(self,state)
            observe_owned_open(self,state)
        target=state.get('provided_native_control_targets')
        if (not isinstance(target,dict) or target.get('step')!=state['observation']['native_scene_step']
                or target.get('exact_native_dtype_target_match') is not True or set(target.get('by_arm',{}))!=set(self.arm_ids)):
            raise ContractError('Provided operation requires actual current native target evidence')
        self.provided_native_target_rows+=1
        if self.provided_bead_settle and self.placement_phase=='bead_material_wait':
            self._guard_bead_material_settle(state)
        row=state['recent_rows'][-1];command=row['command']
        if command['kind']=='supported_preload':
            action=command['action_id']
            if action not in self.provided_preload_anchors:raise ContractError('Supported action lacks actual entry authorization')
            for arm in command['arms']:
                if (tuple(arm['q_rad'])!=self.provided_preload_anchors[action][arm['arm_id']]
                        or any(arm['qd_rad_s']) or any(arm['qdd_rad_s2'])):
                    raise ContractError('Provided supported adjustment must keep every7q command constant')
            self.provided_preload_intervals[action]+=1
            if self.provided_preload_intervals[action]>1400:raise ContractError('Supported action exceeded its declared1400intervals')
            if self.provided_preload_intervals[action]==1400:
                self.provided_preload_events.append(dict(schema_version='provided-supported-preload-completed-v1',
                    action_id=action,actor=command['active_arm_id'],native_scene_step=row['native_scene_step'],
                    simulation_s=row['simulation_s'],observation_sha256=digest(state['observation']),
                    physical_intervals=1400,declared_final_target_m=.015,actual_native_targets_verified=True))
                self._save(state,'provided_supported_preload_events.json',self.provided_preload_events)
        if self.placement_phase=='corrected_hold':
            from physics.provided_placement_plan import pose_displacements,displacement_failures,_angle
            obs=state['observation'];arm=self.current['active_arm_id'];step=obs['native_scene_step']
            values=pose_displacements(obs['arms'][arm]['hand_world'],obs['body_world'],self.placement_anchor)
            failures=displacement_failures(values)
            if self.placement_execution_last_step is not None and step!=self.placement_execution_last_step+1:
                failures.append('nonconsecutive_provided_execution_observation')
            self.placement_execution_last_step=step
            for key,value in values.items():self.placement_execution_max[key]=max(self.placement_execution_max.get(key,0.),value)
            nominal=self.goal.get('seating',{}).get('nominal_goal',self.goal)['body_world']
            if math.dist(obs['body_world']['position_m'],nominal['position_m'])>state['settings']['execution_limits']['final_body_position_error_m']:
                failures.append('original_destination_position_error')
            if _angle(obs['body_world']['quaternion_wxyz'],nominal['quaternion_wxyz'])>state['settings']['execution_limits']['final_body_orientation_error_rad']:
                failures.append('original_destination_orientation_error')
            try:source_ok=state['source_check']() is True
            except Exception:source_ok=False
            if not source_ok:failures.append('source_bindings_changed')
            record={'schema_version':'provided-placement-native-envelope-observation-v1','stage_id':self.current['stage_id'],
                'attempt':self.placement_attempt,'native_scene_step':step,'simulation_s':obs['simulation_s'],
                'observation_sha256':digest(obs),'source_bindings_verified':source_ok,
                'hand_world':copy.deepcopy(obs['arms'][arm]['hand_world']),'body_world':copy.deepcopy(obs['body_world']),
                'first_correction_anchor':copy.deepcopy(self.placement_anchor),'displacements':values,
                'maximum_displacements':copy.deepcopy(self.placement_execution_max),'failures':failures,'passed':not failures,
                'scope':'Actual post-step hand/body maximum displacement from the first provided correction anchor. Not arc length.'}
            with (Path(state['output'])/'provided_placement_execution_checks.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
            if failures:raise ContractError('Provided placement native envelope failed with fingers closed: '+json.dumps(record))
        if (self.mode not in ('first_handoff','source_lift_probe') or self.current is None or self.current['kind']!='carry'
                or self.current['active_arm_id']!='arm_0' or self.image_sink is None
                or any(r['checkpoint']=='lifted' for r in self.image_captures)
                or state.get('unsupported_provided_grasp_s' if self.grasp_policy=='automatic' else 'unsupported_bilateral_s',0.)<.5):return
        from physics.episode_transport_inputs import _bottom_corners
        from physics.carried_body import WorldPose
        p=state['observation']['body_world'];body=WorldPose(p['position_m'],p['quaternion_wxyz'])
        support=self.world['source_support'];top=support['center_world_m'][2]+support['size_m'][2]/2
        clearance=min(c[2] for c in _bottom_corners(self.model,body))-top
        if clearance>=.015:
            if self.grasp_policy=='automatic':
                self._capture('lifted',state,{'bottom_clearance_m':clearance,'source_support_top_m':top,
                    'unsupported_provided_grasp_s':state['unsupported_provided_grasp_s'],'source_support_id':support['id'],'grasp_model':'provided_ideal'})
            else:
                self._capture('lifted',state,{'bottom_clearance_m':clearance,'source_support_top_m':top,
                    'unsupported_bilateral_s':state['unsupported_bilateral_s'],'source_support_id':support['id']})

    def check_boundary(self,request,state):
        """Generic measured joint-rest/source verification is owned by runtime."""
        if self.grasp_policy=='automatic':
            from physics.automatic_manipulation import check_boundary
            return check_boundary(self,request,state)
        result={'provided_policy':teacher_policy()['schema_version'],'action_role':request['action_role']}
        if request['action_kind']=='supported_preload':
            operation=self.current;obs=state['observation'];row=state['recent_rows'][-1];action=request['action_id']
            if operation is None or operation['kind']!='carry':raise ContractError('Supported adjustment requires the actual current carry')
            actor=operation['active_arm_id'];destination=operation['destination']
            if (self.goal['support_id']!=destination or request['active_arm_id']!=actor or self.held is None
                    or request['measured_rest_required'] is not False
                    or action!=operation['stage_id']+'__instructor_supported_preload' or action in self.provided_preload_anchors):
                raise ContractError('Supported adjustment actor/destination/held/unique-action binding differs')
            ready=assess_place_ready(obs,state['recent_rows'],state['settings'],state['rest_policy'],self.goal)
            if self.particle_mode:
                from physics.particle_student_completion import checked_bulk_rest
                ready['particle_bulk_rest']=checked_bulk_rest(state['recent_rows'],obs,self.initial_supply)
            bilateral=row['bilateral_by_arm'].get(actor) is True
            support=any(c['category']=='named_tray_bottom_support_top' and c.get('support_station_id')==destination for c in row['classified_contacts'])
            source_ok=state['source_check']() is True
            event=dict(schema_version='provided-supported-preload-entry-v1',action_id=action,actor=actor,
                destination_support_id=destination,native_scene_step=obs['native_scene_step'],simulation_s=obs['simulation_s'],
                observation_sha256=digest(obs),actual_bilateral=bilateral,actual_named_support=support,
                original_support_body_ready=ready,source_bindings_verified=source_ok,
                authorized=bool(ready['passed'] and bilateral and support and source_ok))
            self.provided_preload_events.append(event);self._save(state,'provided_supported_preload_events.json',self.provided_preload_events)
            if not event['authorized']:raise ContractError('Provided supported-entry physical evidence rejected')
            anchors={a['arm_id']:tuple(a['q_rad']) for a in request['expected_arm_commands']}
            if set(anchors)!=set(self.arm_ids) or any(len(q)!=7 for q in anchors.values()):
                raise ContractError('Supported adjustment requires all exact dispatcher7q anchors')
            self.provided_preload_anchors[action]=anchors;self.provided_preload_intervals[action]=0
            result['provided_supported_preload']=event
        if request['action_role']=='release':
            if self.current is None or self.current['kind']!='carry':raise ContractError('Release without an active placement')
            gate=assess_place_ready(state['observation'],state['recent_rows'],state['settings'],state['rest_policy'],self.goal)
            if self.particle_mode:
                from physics.particle_student_completion import checked_bulk_rest
                gate['particle_bulk_rest']=checked_bulk_rest(state['recent_rows'],state['observation'],self.initial_supply)
            result['place_ready']=gate
            self._save(state,self.current['stage_id']+'__before_release.json',gate)
            if not gate['passed']:raise ContractError('Provided placement gate rejected opening: '+json.dumps(gate))
        return result

    def _finish_current(self,state):
        if self.grasp_policy=='automatic':
            from physics.automatic_manipulation import finish_current
            return finish_current(self,state)
        if self.current is None:return
        op=self.current;obs=state['observation'];arm=op['active_arm_id'];kind=op['kind']
        if kind=='approach':
            gate,held=assess_pick(obs,state['recent_rows'],state['settings'],self.arm_ids,self.scenario,
                state['rest_policy'],state['contact_rows'],state['bilateral_since'],state['stable_since'],
                state['support_since'],self.model,active_arm_id=arm,support_id=op['support_id'],
                endpoint_hand_pose=self.endpoint_hand_pose)
            if self.particle_mode:
                from physics.particle_student_completion import particle_context
                from physics.particle_episode_state import window_rest
                particle_context(self.world,self.scenario,obs,self.initial_supply)
                gate['particle_bulk_rest']=window_rest(state['recent_rows'],obs['simulation_s'],self.initial_supply)
                gate['passed']=bool(gate['passed'] and gate['particle_bulk_rest']['passed'])
            self._save(state,op['stage_id']+'__provided_pick.json',{'gate':gate,'held_body':held.to_record(),
                'observation_sha256':digest(obs)})
            if not gate['passed']:raise ContractError('Provided supported grasp failed: '+json.dumps(gate))
            self.held=held;self.held_observation_sha256=digest(obs)
            if self.mode=='first_handoff' and arm=='arm_1':
                self._capture('recipient_regrasp',state,{'support_id':op['support_id'],'gate_passed':True,
                    'held_body':held.to_record(),'held_binding_sha256':held.binding_sha256})
        elif kind=='carry' and op.get('particle_source_lift'):
            from physics.particle_student_completion import source_lift_gate
            from physics.provided_placement_plan import refresh_measured_held
            gate=source_lift_gate(state,arm=arm,model=self.model,source_support=self.world['source_support'],
                start_body=self.probe_lift_start_body,contract=self.initial_supply)
            if not gate['passed']:raise ContractError('Private particle source lift failed: '+json.dumps(gate))
            self.held=refresh_measured_held(self.world,self.scenario,self.model,obs,state['recent_rows'],self.held,self.initial_supply,arm)
            self.held_observation_sha256=digest(obs)
            self._save(state,op['stage_id']+'__provided_lift.json',{'gate':gate,'held_body':self.held.to_record(),'observation_sha256':digest(obs)})
        elif kind=='carry' and op.get('probe_lift'):
            from physics.episode_runtime import measured_rest,body_rest
            from physics.provided_placement_plan import refresh_measured_held
            from physics.episode_transport_inputs import _bottom_corners
            from physics.carried_body import WorldPose
            load=state['maximum_unsupported_bilateral_by_arm_s'][arm]
            rest=measured_rest(state['recent_rows'],obs['simulation_s'],state['settings']['execution_limits'],self.arm_ids)
            body_rest_gate=body_rest(state['recent_rows'],obs['simulation_s'],state['settings']['execution_limits']['rest_window_s'],state['rest_policy'])
            lifted=obs['body_world']['position_m'][2]-self.probe_lift_start_body['position_m'][2]
            support=next(s for s in self.scenario['support_stations'] if s['id']==op['support_id'])
            top=support['center_world_m'][2]+support['size_m'][2]/2
            actual=WorldPose(obs['body_world']['position_m'],obs['body_world']['quaternion_wxyz'])
            clearance=min(c[2]-top for c in _bottom_corners(self.model,actual))
            gate={'schema_version':'provided-placement-probe-lift-v1','load_bearing_interval_s':load,
                'requested_lift_m':.005,'measured_lift_m':lifted,'minimum_bottom_clearance_m':clearance,
                'all_arms_rest':rest,'body_rest':body_rest_gate,
                'passed':bool(load>=.5 and abs(lifted-.005)<=.0005 and clearance>=.003 and rest['passed'] and body_rest_gate['passed'])}
            if not gate['passed']:raise ContractError('Private probe did not complete a real stable unsupported 5mm lift: '+json.dumps(gate))
            self.held=refresh_measured_held(self.world,self.scenario,self.model,obs,state['recent_rows'],self.held,self.initial_supply,arm)
            self.held_observation_sha256=digest(obs)
            self._save(state,op['stage_id']+'__provided_lift.json',{'gate':gate,'held_body':self.held.to_record(),'observation_sha256':digest(obs)})
        elif kind=='carry':
            from physics.episode_runtime import DT,TRANSPORT_PROTOCOL
            nominal=self.goal.get('seating',{}).get('nominal_goal',self.goal)
            release=assess_place_release
            if self.particle_mode:
                from physics.particle_episode_assessment import released_particle_relay_gate
                release=released_particle_relay_gate
            gate=release(state['recent_rows'],obs['simulation_s'],state['settings'],self.arm_ids,
                state['rest_policy'],nominal,self.finger_bounds,self.initial_supply,dt=DT,
                window_s=TRANSPORT_PROTOCOL['acceptance_window_s'],active_arm_id=arm,
                **({'coverage_validator':self._particle_runtime[1]} if self.particle_mode else {}))
            load=state['maximum_unsupported_bilateral_by_arm_s'][arm]
            gate['load_bearing_interval_s']=load
            if self.provided_bead_settle:
                from physics.episode_transport_assessment import bead_material_rest_gate
                gate['bead_bulk_rest']=bead_material_rest_gate(state['recent_rows'],obs['simulation_s'],self.initial_supply,self.model)
                gate['passed']=bool(gate['passed'] and gate['bead_bulk_rest']['passed'])
            if self.mode in ('mission','first_handoff') and load<TRANSPORT_PROTOCOL['acceptance_window_s']:
                gate['passed']=False;gate['missing_load_bearing_interval']=True
            self._save(state,op['stage_id']+'__provided_place.json',gate)
            if not gate['passed']:raise ContractError('Provided released placement failed: '+json.dumps(gate))
            if self.provided_bead_settle and self.bead_settle_events:
                event=self.bead_settle_events[-1]
                if event['stage_id']==op['stage_id'] and event['native_scene_step']==obs['native_scene_step']:
                    event['accepted']=True;self._save(state,'provided_bead_material_settle.json',self.bead_settle_events)
            if self.mode=='first_handoff':
                self._capture('relay_release',state,{'support_id':self.goal['support_id'],
                    'gate_passed':True,'load_bearing_interval_s':load,'placement_goal':self.goal})
            elif self.mode=='mission':
                checkpoint='mission_'+arm+('_goal_release' if arm=='arm_2' else '_relay_release')
                self._capture(checkpoint,state,{'support_id':self.goal['support_id'],
                    'gate_passed':True,'load_bearing_interval_s':load,'placement_goal':self.goal})
            self.held=None
        else:
            from physics.episode_runtime import measured_rest
            gate=measured_rest(state['recent_rows'],obs['simulation_s'],state['settings']['execution_limits'],self.arm_ids)
            if not gate['passed']:raise ContractError('Student return endpoint has not settled')
        self.gates.append({'stage_id':op['stage_id'],'kind':kind,'active_arm_id':arm,
            'support_id':self.goal['support_id'] if kind=='carry' and not op.get('probe_lift') else op['support_id'],'gate':gate})
        self.completed.append({'stage_id':op['stage_id'],'kind':kind,'active_arm_id':arm,
            'native_scene_step':obs['native_scene_step'],'simulation_s':obs['simulation_s']})
        self.current=None
        self.placement_phase=None
        self._bead_settle_wait=None

    def _guard_bead_material_settle(self,state):
        """Each extra interval keeps the same supported, released native task."""
        from physics.episode_runtime import measured_rest
        from physics.episode_transport_assessment import _materials,_bead_material_sample
        wait=self._bead_settle_wait;obs=state['observation'];row=state['recent_rows'][-1]
        if (self.particle_mode or wait is None or self.current is None
                or wait['stage_id']!=self.current['stage_id'] or obs['native_scene_step']!=wait['last_step']+1
                or row['native_scene_step']!=obs['native_scene_step'] or row.get('phase')!='released_settle'):
            raise ContractError('Provided bead wait needs the next actual released native interval')
        command=row['command'];scheduled=command['arms']
        if (command['kind']!='hold' or command['action_role']!='released_settle'
                or len(scheduled)!=len(self.arm_ids) or {c['arm_id'] for c in scheduled}!=set(self.arm_ids)):
            raise ContractError('Provided bead wait must remain a released hold for every arm')
        for c in scheduled:
            anchor=wait['commands'][c['arm_id']]
            if (c['q_rad']!=anchor['q_rad'] or c['finger_q_m']!=anchor['finger_q_m']
                    or any(c['qd_rad_s']) or any(c['qdd_rad_s2']) or any(c['finger_qd_m_s'])):
                raise ContractError('Provided bead wait cannot change any joint/finger command')
        _bead_material_sample(row,_materials(self.initial_supply),self.model.geometry_record())
        ready=assess_place_ready(obs,state['recent_rows'],state['settings'],state['rest_policy'],self.goal)
        rest=measured_rest(state['recent_rows'],obs['simulation_s'],state['settings']['execution_limits'],self.arm_ids)
        arm=self.current['active_arm_id'];q=obs['arms'][arm]['q_all_dofs']
        opened=max(abs(q[7+i]-self.finger_bounds[arm][i][1]) for i in range(2))
        if (not ready['passed'] or not rest['passed'] or opened>state['settings']['closure_match_tolerances']['finger_m']
                or state['source_check']() is not True):
            raise ContractError('Provided bead wait lost its original support/body/arm/open-finger/source guard')
        wait['last_step']=obs['native_scene_step']

    def _advance_bead_material_settle(self,dispatcher,state):
        """Append genuine .5s holds only for an otherwise valid unsettled bead load."""
        if (not self.provided_bead_settle or self.current is None or self.current['kind']!='carry'
                or self.current.get('probe_lift') or self.placement_phase not in ('released_tail','bead_material_wait')):
            return False
        if self.particle_mode:raise ContractError('Bead settle cannot consume a particle observation')
        from physics.episode_runtime import DT,TRANSPORT_PROTOCOL
        from physics.episode_continuation_protocol import rest_receipt
        from physics.episode_transport_assessment import bead_material_rest_gate
        state['wall_check']()
        if state['source_check']() is not True:raise ContractError('Provided bead wait requires unchanged sources')
        obs=state['observation'];now=obs['simulation_s'];stage=self.current['stage_id'];arm=self.current['active_arm_id']
        if now>self.simulation_deadline_s+1e-9:raise ContractError('Provided bead wait cannot approve after the simulation deadline')
        if self.bead_settle_events and self.bead_settle_events[-1]['stage_id']==stage and self.bead_settle_events[-1]['native_scene_step']>=obs['native_scene_step']:
            raise ContractError('Provided bead wait requires newly completed physical intervals')
        nominal=self.goal.get('seating',{}).get('nominal_goal',self.goal)
        hard=assess_place_release(state['recent_rows'],now,state['settings'],self.arm_ids,state['rest_policy'],nominal,
            self.finger_bounds,self.initial_supply,dt=DT,window_s=TRANSPORT_PROTOCOL['acceptance_window_s'],active_arm_id=arm)
        load=state['maximum_unsupported_bilateral_by_arm_s'][arm]
        if (hard['passed'] is not True or self.mode in ('mission','first_handoff')
                and load<TRANSPORT_PROTOCOL['acceptance_window_s']):
            raise ContractError('Only bulk-unsettled placements with every original release/load guard may wait')
        bulk=bead_material_rest_gate(state['recent_rows'],now,self.initial_supply,self.model)
        wait=self._bead_settle_wait;elapsed=0. if wait is None else now-wait['start_s']
        event=dict(stage_id=stage,native_scene_step=obs['native_scene_step'],simulation_s=now,
            hard_release_gate_passed=True,bead_bulk_rest=bulk,additional_simulation_s=elapsed,
            accepted=False,extra_hold_s=0.,timeout=False,observation_sha256=digest(obs))
        self.bead_settle_events.append(event)
        self._save(state,'provided_bead_material_settle.json',self.bead_settle_events)
        if bulk['quality_passed'] is not True:raise ContractError('Invalid bead data is not a settling retry')
        if bulk['passed'] is True:return False
        if elapsed+BEAD_SETTLE_HOLD_S>BEAD_SETTLE_EXTRA_LIMIT_S+1e-9 or now+BEAD_SETTLE_HOLD_S>self.simulation_deadline_s+1e-9:
            event['timeout']=True;self._save(state,'provided_bead_material_settle.json',self.bead_settle_events)
            raise ContractError('Provided bead settling exhausted its fixed5s or remaining simulation budget')
        commands=dispatcher.record()['last_command_by_arm']
        if wait is None:
            wait=dict(stage_id=stage,start_s=now,start_step=obs['native_scene_step'],last_step=obs['native_scene_step'],segments=0,commands=copy.deepcopy(commands))
            self._bead_settle_wait=wait
        if wait['stage_id']!=stage or commands!=wait['commands'] or wait['last_step']!=obs['native_scene_step']:
            raise ContractError('Provided bead wait changed its held command or skipped native interval guards')
        wait['segments']+=1
        actions=[dict(action_id=stage+f'__provided_bead_settle_{wait["segments"]:02d}',arm_id=arm,
            role='released_settle',kind='hold',duration_s=BEAD_SETTLE_HOLD_S)]
        plan=self._plan(obs,commands,actions,actions[0]['action_id'])
        boundary=rest_receipt(dispatcher.record(),obs,list(self.arm_ids))
        dispatcher.extend(plan,model_sha256_by_arm={n:self.world['robot_model_sha256'] for n in self.arm_ids},boundary=boundary)
        self.placement_phase='bead_material_wait'
        event.update(extra_hold_s=BEAD_SETTLE_HOLD_S,episode_sha256=digest(plan.to_record()),commands_sha256=digest(commands))
        self._save(state,'provided_bead_material_settle.json',self.bead_settle_events)
        return True

    def _advance_provided_place(self,dispatcher,state):
        """Extend just the instructor place phase, preserving student plan records."""
        if self.grasp_policy=='automatic':
            from physics.automatic_manipulation import advance_place
            return advance_place(self,dispatcher,state)
        from physics.episode_runtime import measured_rest,body_rest
        from physics.episode_continuation_protocol import rest_receipt
        from physics.provided_placement_plan import plan_measured_seating,MAX_ATTEMPTS
        if self.placement_phase not in ('closed_hold','corrected_hold'):return False
        obs=state['observation'];arm=self.current['active_arm_id'];stage=self.current['stage_id']
        execution=state['settings']['execution_limits']
        rest=measured_rest(state['recent_rows'],obs['simulation_s'],execution,self.arm_ids)
        body=body_rest(state['recent_rows'],obs['simulation_s'],execution['rest_window_s'],state['rest_policy'])
        if not rest['passed'] or not body['passed'] or not state['source_check']():
            raise ContractError('Provided placement continuation requires unchanged sources and measured arm/body rest')
        if self.particle_mode:
            from physics.particle_student_completion import particle_context,checked_bulk_rest
            particle_context(self.world,self.scenario,obs,self.initial_supply)
            checked_bulk_rest(state['recent_rows'],obs,self.initial_supply)
        if self.placement_phase=='corrected_hold':
            self.placement_events[-1].update(physical_segment_completed=True,
                completed_step=obs['native_scene_step'],completed_simulation_s=obs['simulation_s'],
                completed_observation_sha256=digest(obs),actual_maximum_displacements=copy.deepcopy(self.placement_execution_max))
            self._save(state,'provided_placement_segments.json',self.placement_events)
        ready=assess_place_ready(obs,state['recent_rows'],state['settings'],state['rest_policy'],self.goal)
        self._save(state,stage+f'__provided_closed_hold_{self.placement_attempt:02d}.json',ready)
        commands=dispatcher.record()['last_command_by_arm']
        if ready['passed']:
            actions=place_actions(arm,self.finger_bounds[arm],prefix=stage+'__provided_place')[1:]
            phase='released_tail'
            event={'schema_version':'provided-placement-ready-boundary-v1','stage_id':stage,
                'active_arm_id':arm,'support_id':self.goal['support_id'],'ready_gate':ready,
                'native_scene_step':obs['native_scene_step'],'simulation_s':obs['simulation_s'],
                'observation_sha256':digest(obs),'correction_required':False,'physical_release_verified':False}
        else:
            if self.placement_policy=='fixed':
                self._save(state,stage+'__before_release.json',ready)
                raise ContractError('Fixed QA placement policy failed the unchanged support/rest gate: '+json.dumps(ready))
            if self.placement_attempt>=MAX_ATTEMPTS:
                raise ContractError('Provided measured placement attempts exhausted with fingers closed: '+json.dumps(ready))
            before=state['observe']();dispatch_before=dispatcher.record()
            particle_pause=self._particle_pause_before(obs,dispatcher) if self.particle_mode else None
            if self.placement_anchor is None:
                self.placement_anchor={'hand_world':copy.deepcopy(obs['arms'][arm]['hand_world']),
                    'body_world':copy.deepcopy(obs['body_world'])}
                self.placement_execution_max={};self.placement_execution_last_step=obs['native_scene_step']
            self.placement_attempt+=1
            folder=Path(state['output'])/(stage+f'__provided_seating_{self.placement_attempt:02d}')
            trajectory,receipt=plan_measured_seating(world=self.world,scenario=self.scenario,model=self.model,
                observation=obs,recent_rows=state['recent_rows'],previous_held=self.held,initial_supply=self.initial_supply,
                arm=arm,command_start_q=commands[arm]['q_rad'],original_goal=self.goal,limits=self.limits_by_arm[arm],
                speed_scale=self.speed_scale,attempt=self.placement_attempt,used_translation_m=self.placement_translation_m,
                output=folder,source_check=state['source_check'],wall_check=state['wall_check'],
                fixture_readback=self.fixture_readback,endpoint_attempts=min(8,self.endpoint_attempts),
                seed=self.seed+self.index+self.placement_attempt,used_rotation_rad=self.placement_rotation_rad,
                anchor=self.placement_anchor)
            after=state['observe']()
            if before!=after or dispatch_before!=dispatcher.record() or not state['source_check']():
                raise ContractError('Live native state, clock, dispatcher or sources changed while provided place planned')
            particle_comparison=self._particle_pause_after(particle_pause,after,dispatcher) if self.particle_mode else None
            if self.particle_mode:
                self._save(state,str(folder.relative_to(state['output'])/'particle_pause_before.json'),particle_pause['receipt'])
                self._save(state,str(folder.relative_to(state['output'])/'particle_pause_after.json'),particle_comparison)
            self.placement_translation_m=receipt['cumulative_translation_m']
            self.placement_rotation_rad=receipt['cumulative_rotation_rad']
            prefix=stage+f'__provided_seat_{self.placement_attempt:02d}'
            actions=[{'action_id':prefix+'__motion','arm_id':arm,'role':'place','kind':'trajectory','trajectory':trajectory.to_record()},
                {**place_actions(arm,self.finger_bounds[arm],prefix=prefix)[0]}]
            phase='corrected_hold'
            event={'schema_version':'provided-placement-local-segment-v1','stage_id':stage,'active_arm_id':arm,
                'support_id':self.goal['support_id'],'plan_receipt':str(folder.relative_to(state['output']))+'/provided_placement_plan.json',
                'plan_receipt_sha256':digest(receipt),'trajectory_sha256':digest(trajectory.to_record()),
                'start_step':obs['native_scene_step'],'start_simulation_s':obs['simulation_s'],
                'observation_sha256':digest(obs),'physical_segment_completed':False,
                'native_state_and_clock_unchanged_during_query':True,'dispatcher_unchanged_during_query':True,
                'source_bindings_verified':True,'cumulative_translation_m':self.placement_translation_m}
            if self.particle_mode:event['particle_pause_comparison']=particle_comparison
        next_plan=self._plan(obs,commands,actions,stage+'__provided_'+phase+f'_{self.placement_attempt:02d}')
        boundary=rest_receipt(dispatcher.record(),obs,list(self.arm_ids))
        dispatcher.extend(next_plan,model_sha256_by_arm={n:self.world['robot_model_sha256'] for n in self.arm_ids},boundary=boundary)
        event.update(episode_sha256=digest(next_plan.to_record()),segment_index=dispatcher.record()['segments'][-1]['index'])
        self.placement_events.append(event);self.placement_phase=phase
        self._save(state,next_plan.episode_id+'__episode.json',next_plan.to_record())
        self._save(state,'provided_placement_segments.json',self.placement_events)
        return True

    def advance(self,dispatcher,state):
        """Plan from a paused measured boundary and extend exact commanded state."""
        if self.mode=='empty_path':
            from physics.empty_motion import advance
            return advance(self,dispatcher,state)
        from physics.episode_runtime import measured_rest,body_rest
        from physics.episode_continuation_protocol import rest_receipt
        from physics.student_motion_planning import plan_student_motion
        state['wall_check']()
        if not state['source_check']():raise ContractError('Student completion requires unchanged source bindings')
        if dispatcher.record().get('current_segment_complete') is not True:
            raise ContractError('Global deadline or incomplete physical segment cannot complete a student operation')
        if self.current is not None and self._advance_provided_place(dispatcher,state):return True
        if self.current is not None and self._advance_bead_material_settle(dispatcher,state):return True
        completed_operation=copy.deepcopy(self.current)
        self._finish_current(state)
        reset_extension_boundary=None
        if (self.post_place_policy=='instant_reset' and completed_operation is not None
                and completed_operation['kind']=='carry' and completed_operation['active_arm_id']!='arm_2'):
            from physics.provided_robot_reset import validate_robot_reset_receipt
            expected=self.saved_plan.reset_after(completed_operation['stage_id'])
            callback=state.get('reset_completed_robot')
            if expected is None or not callable(callback):
                raise ContractError('Instant-reset replay requires its actual native reset callback and saved schedule')
            transaction=callback(self,dispatcher,state,completed_operation)
            if type(transaction) is not dict or set(transaction)!={'state','receipt','extension_boundary'}:
                raise ContractError('Native robot reset returned an invalid transaction')
            receipt=transaction['receipt'];state=transaction['state']
            validate_robot_reset_receipt(receipt,after_observation=state['observation'],
                expected_command=expected['target_command'],expected_arm_id=expected['arm_id'],
                expected_stage_id=expected['after_stage_id'])
            reset_info=state.get('instant_reset_boundary',{})
            if (reset_info.get('verified') is not True or reset_info.get('receipt_sha256')!=digest(receipt)
                    or reset_info.get('extension_boundary')!=transaction['extension_boundary']):
                raise ContractError('Fresh reset state lacks its exact authenticated extension boundary')
            self.post_place_resets.append(copy.deepcopy(receipt))
            reset_extension_boundary=transaction['extension_boundary']
        elif (self.post_place_policy=='kinematic_return' and completed_operation is not None
                and completed_operation['kind']=='carry' and completed_operation['active_arm_id']!='arm_2'):
            from physics.provided_robot_return import validate_kinematic_return_receipt
            expected=self.saved_plan.kinematic_return_after(completed_operation['stage_id'])
            callback=state.get('return_completed_robot_kinematically')
            if expected is None or not callable(callback):
                raise ContractError('Kinematic return requires its actual native callback and exact saved reverse schedule')
            transaction=callback(self,dispatcher,state,completed_operation)
            if type(transaction) is not dict or set(transaction)!={'state','receipt','extension_boundary'}:
                raise ContractError('Native kinematic return produced an invalid transaction')
            receipt=transaction['receipt'];state=transaction['state']
            validate_kinematic_return_receipt(receipt,after_observation=state['observation'],
                expected_schedule=expected,expected_command=expected['target_command'],
                expected_arm_id=expected['arm_id'],expected_stage_id=expected['after_stage_id'])
            return_info=state.get('kinematic_return_boundary',{})
            if (return_info.get('verified') is not True or return_info.get('receipt_sha256')!=digest(receipt)
                    or return_info.get('extension_boundary')!=transaction['extension_boundary']):
                raise ContractError('Fresh return state lacks its exact authenticated extension boundary')
            self.post_place_returns.append(copy.deepcopy(receipt))
            reset_extension_boundary=transaction['extension_boundary']
        if self.index==len(self.queue):self.done=True;return False
        obs=state['observation'];execution=state['settings']['execution_limits']
        if reset_extension_boundary is None:
            rest=measured_rest(state['recent_rows'],obs['simulation_s'],execution,self.arm_ids)
            b_rest=body_rest(state['recent_rows'],obs['simulation_s'],execution['rest_window_s'],state['rest_policy'])
            if not rest['passed'] or not b_rest['passed']:
                raise ContractError('A real all-arm/body rest boundary is required for student planning')
        # An explicit nonphysical event uses its authenticated pre-event rest
        # plus fresh zero-velocity readback. No old rows become home history.
        if not state['source_check']():
            raise ContractError('Student planning requires unchanged source bindings')
        if self.particle_mode:
            from physics.particle_student_completion import particle_context,checked_bulk_rest
            particle_context(self.world,self.scenario,obs,self.initial_supply)
            checked_bulk_rest(state['recent_rows'],obs,self.initial_supply)
        if self.mode=='one_carry' and self.index==0 and self.held is None:
            from physics.one_carry_planning import initial_pick
            initial_pick(self,dispatcher,state)
            obs=state['observation']
        op=copy.deepcopy(self.queue[self.index]);arm=op['active_arm_id'];kind=op['kind']
        commands=dispatcher.record()['last_command_by_arm']
        if self.single_active_arm_contract is not None:
            from physics.single_active_boundary import with_static_commands
            commands=with_static_commands(commands,obs)
        if kind=='approach':
            endpoint=grasp_endpoint_candidates(self.model,obs['body_world'])[0].to_record()
            goal={'kind':'world_hand_pose','position_m':endpoint['position_m'],'quaternion_wxyz':endpoint['quaternion_wxyz']}
        elif kind=='retreat':
            endpoint=None;goal={'kind':('declared_probe_endpoint' if self.mode=='short_motions' else
                'clear_handoff' if self.post_place_policy=='clear_handoff' else 'return_initial'),'q_rad':op['goal_q']}
        elif op.get('probe_lift'):
            if self.held is None or self.held_observation_sha256!=digest(obs):
                raise ContractError('Private lift starts only from the exact fresh supported grasp')
            self.probe_lift_start_body=copy.deepcopy(obs['body_world'])
            endpoint=copy.deepcopy(obs['arms'][arm]['hand_world'])
            # 55mm endpoint. Gate separately requires >=50mm actual body rise.
            # Original endpoint/rest tolerances remain unchanged.
            endpoint['position_m'][2]+=.055 if op.get('bead_lift_unit') else .03 if op.get('particle_source_lift') else .005
            goal={'kind':'world_hand_pose','position_m':endpoint['position_m'],'quaternion_wxyz':endpoint['quaternion_wxyz']}
        else:
            if self.held is None or self.held_observation_sha256!=digest(obs):
                raise ContractError('Carry must begin at the exact freshly verified provided grasp')
            self.goal=placement_endpoint(self.world,self.scenario,self.held,op['destination'])
            endpoint=self.goal['hand_world'];goal={'kind':'world_hand_pose','position_m':endpoint['position_m'],
                'quaternion_wxyz':endpoint['quaternion_wxyz']}
        request={'schema_version':'student-motion-request-v1','stage_id':op['stage_id'],'kind':kind,
            'world':self.world,'scenario':self.scenario,'observation':copy.deepcopy(obs),'active_arm_id':arm,
            'command_start_q_rad':commands[arm]['q_rad'],'goal':goal,
            'planning':{'seed':self.seed if self.world.get('payload_mode')=='empty' else self.seed+self.index,'max_queries':self.max_queries,'speed_scale':self.speed_scale,
                'endpoint_attempts':self.endpoint_attempts,'options':self.options,'use_edge_timing_hook':self.use_edge_timing_hook,
                'query_body_tilt_limit_deg':180. if self.world.get('payload_mode')=='empty' else 7.5}}
        if kind=='carry':
            request['held_body']=self.held.to_record()
            if not op.get('probe_lift'):request['placement_goal']=self.goal
        if self.mode=='short_motions':request['probe_scope']='three_arm_short_motion_unit'
        raw_before=state['observe']();dispatch_before=dispatcher.record();state['wall_check']()
        particle_pause=self._particle_pause_before(obs,dispatcher) if self.particle_mode else None
        folder=Path(state['output'])/(op['stage_id']+'__planning');folder.mkdir(exist_ok=False)
        if self.saved_plan is None:
            response=plan_student_motion(request,self.strategy,output=folder,source_check=state['source_check'],wall_check=state['wall_check'],
                **({'fixture_readback':self.fixture_readback} if self.fixture_readback!='scalar' else {}))
        else:
            from physics.saved_motion_planning import validate_saved_motion
            if (self.particle_mode and self.mode=='one_carry'
                    and self.saved_plan.record().get('particle_recheck_mode')=='cpu_geometry_actual_boundary'):
                from physics.particle_saved_motion import validate_particle_saved_motion_cpu as validate_saved_motion
            response=validate_saved_motion(request,self.saved_plan,self.index,previous_commands=commands,
                output=folder,source_check=state['source_check'],wall_check=state['wall_check'],fixture_readback=self.fixture_readback)
        state['wall_check']();raw_after=state['observe']()
        if raw_before!=raw_after or dispatch_before!=dispatcher.record() or not state['source_check']():
            raise ContractError('Live scene/clock/dispatcher/sources changed while the student planned')
        particle_comparison=self._particle_pause_after(particle_pause,raw_after,dispatcher) if self.particle_mode else None
        if self.particle_mode:
            self._save(state,str(folder.relative_to(state['output'])/'particle_pause_before.json'),particle_pause['receipt'])
            self._save(state,str(folder.relative_to(state['output'])/'particle_pause_after.json'),particle_comparison)
        if (response.get('schema_version')!=('student-motion-response-v1' if self.saved_plan is None else 'saved-motion-response-v1')
                or any(response.get(k)!=request[k] for k in ('stage_id','kind','active_arm_id'))
                or response.get('request_sha256')!=digest(request)
                or type(response.get('physical_steps')) is not int or response.get('physical_steps')!=0
                or response.get('cleanup_errors')!=[] or response.get('source_bindings_verified') is not True
                or response.get('durable_path_validation',{}).get('passed') is not True
                or type(response.get('provided_adapter_plan_path_invocations')) is not int
                or response.get('provided_adapter_plan_path_invocations')!=(1 if self.saved_plan is None else 0)
                or (self.saved_plan is not None and (response.get('saved_bundle_sha256')!=self.saved_plan.sha256
                    or response.get('runtime_ik_calls')!=0 or response.get('runtime_rrt_calls')!=0))
                or (self.particle_mode and (response.get('contents_collision_queried') is not False
                    or type(response.get('contents_query_scope')) is not str or not response['contents_query_scope']))
                or (self.particle_mode and self.saved_plan is not None and response.get('validation_backend')!=
                    ('cpu_geometry_actual_boundary' if self.saved_plan.record().get('particle_recheck_mode')==
                        'cpu_geometry_actual_boundary' else 'native_full_geometry_reference'))
                or any(response.get(k) is not True for k in
                    ('observed_start_valid','command_start_valid','observed_to_command_edge_valid'))):
            raise ContractError('Student query response is not source-bound and fully validated')
        plan_record=response['plan'];tr=plan_record['trajectory']
        trajectory=JointTrajectory(tr['waypoints'],tr['edge_durations_s'],self.limits_by_arm[arm])
        if list(trajectory.waypoints[0])!=commands[arm]['q_rad'] or digest(tr)!=plan_record['evidence']['trajectory_sha256']:
            raise ContractError('Student trajectory differs from verified command/time evidence')
        actions=[];role='carry' if kind=='carry' else 'approach' if kind=='approach' else 'retreat'
        if trajectory.duration_s:
            actions.append({'action_id':op['stage_id']+'__student_motion','arm_id':arm,'role':role,
                'kind':'trajectory','trajectory':trajectory.to_record()})
        else:
            actions.append({'action_id':op['stage_id']+'__student_equal_endpoint','arm_id':arm,'role':role,
                'kind':'hold','duration_s':.5})
        if kind=='approach' and self.grasp_policy=='automatic':
            from physics.episode_runtime import PROTOCOL
            actions.append(dict(action_id=op['stage_id']+'__automatic_endpoint_settle',arm_id=arm,role='approach_settle',kind='hold',duration_s=PROTOCOL['approach_settle_s']))
        elif kind=='approach':actions+=pick_actions(arm,prefix=op['stage_id']+'__provided_pick',transport_close=self.mode in ('mission','first_handoff','source_lift_probe'))
        elif op.get('probe_lift'):
            actions.append({'action_id':op['stage_id']+'__load_hold','arm_id':arm,'role':'carry','kind':'hold','duration_s':.6})
        elif kind=='carry' and self.grasp_policy=='automatic':
            from physics.episode_runtime import TRANSPORT_PROTOCOL
            actions.append(dict(action_id=op['stage_id']+'__automatic_supported_hold',arm_id=arm,role='place',kind='hold',duration_s=TRANSPORT_PROTOCOL['placement_hold_s']))
            self.placement_phase='automatic_ready'
        elif kind=='carry':
            provided=place_actions(arm,self.finger_bounds[arm],prefix=op['stage_id']+'__provided_place')
            if self.mode in ('mission','first_handoff','placement_probe'):
                actions+=provided[:1]
                if self.mode in ('mission','first_handoff'):
                    actions.append(supported_preload_action(arm,prefix=op['stage_id']))
                    actions.append({'action_id':op['stage_id']+'__instructor_post_preload_settle',
                        'arm_id':arm,'role':'place','kind':'hold','duration_s':.5})
                self.placement_phase='closed_hold';self.placement_attempt=0;self.placement_translation_m=0.;self.placement_rotation_rad=0.
                self.placement_anchor=None;self.placement_execution_max={};self.placement_execution_last_step=None
            else:actions+=provided
        else:actions.append({'action_id':op['stage_id']+'__settle','arm_id':arm,'role':'approach_settle','kind':'hold','duration_s':.5})
        next_plan=self._plan(obs,commands,actions,op['stage_id'])
        receipt=(rest_receipt(dispatcher.record(),obs,list(self.arm_ids))
                 if reset_extension_boundary is None else reset_extension_boundary)
        dispatcher.extend(next_plan,model_sha256_by_arm={n:self.world['robot_model_sha256'] for n in self.arm_ids},boundary=receipt)
        self.current=op;self.index+=1;self.support_id=op.get('destination',op['support_id']) if kind=='carry' else op['support_id']
        self.endpoint_hand_pose=response['endpoint_hand_pose']
        self.plans.append({'stage_id':op['stage_id'],'kind':kind,'active_arm_id':arm,'request_sha256':digest(request),
            'trajectory_sha256':digest(tr),'plan_path_source_sha256':plan_record['evidence']['plan_path_source_sha256'],
            'speed_scale':self.speed_scale,'duration_s':trajectory.duration_s,'waypoint_count':len(trajectory.waypoints),
            'supplied_intermediate_postures':False,'planning_evidence':str(folder.relative_to(state['output']))})
        if self.saved_plan is not None:
            self.plans[-1].update(motion_source='saved_cpu_plan',saved_bundle_sha256=self.saved_plan.sha256,
                runtime_plan_path_invocations=0,runtime_ik_calls=0,runtime_rrt_calls=0,
                offline_planner=copy.deepcopy(self.saved_plan.offline_planner))
        self.pauses.append({'stage_id':op['stage_id'],'native_state_and_clock_unchanged':True,'dispatcher_unchanged':True,
            'measured_observation_sha256':digest(obs),'source_bindings_verified':True})
        if self.particle_mode:self.pauses[-1]['particle_pause_comparison']=particle_comparison
        self._save(state,op['stage_id']+'__episode.json',next_plan.to_record())
        return True

    def record(self):
        if self.mode=='empty_path':
            from physics.empty_motion import record
            return record(self)
        record={'schema_version':'student-three-arm-driver-v1','mode':self.mode,'initialized':self.initialized,
            'post_place_policy':self.post_place_policy,
            'post_place_resets':copy.deepcopy(self.post_place_resets),
            'post_place_returns':copy.deepcopy(self.post_place_returns),
            'completed':copy.deepcopy(self.completed),'planned_motions':copy.deepcopy(self.plans),'provided_gates':copy.deepcopy(self.gates),
            'planning_pauses':copy.deepcopy(self.pauses),'operation_sequence_complete':self.done,
            'image_captures':copy.deepcopy(self.image_captures),'query_fixture_readback':self.fixture_readback,
            'current_held_body':None if self.held is None else self.held.to_record(),
            'current_held_binding_sha256':None if self.held is None else self.held.binding_sha256,
            'current_held_observation_sha256':getattr(self,'held_observation_sha256',None),
            'support_route':list(getattr(self,'route',())), 'current_support_id':self.support_id,
            'all_requested_student_transport_motion_from_selected_strategy':True,
            'all_motion_from_selected_strategy':not any('trajectory_sha256' in event for event in self.placement_events),
            'legacy_all_motion_scope':'False when teacher-owned local seven-joint seating motion exists. All requested student routes remain separately source-bound.',
            'reference_planner_fallback':False,
            'supplied_intermediate_postures':False,'provided_manipulation_policy':teacher_policy(),
            'provided_control':{'transport_policy_applied':self.mode in ('mission','first_handoff','source_lift_probe'),
                'transport_close_target_m':.010 if self.mode in ('mission','first_handoff','source_lift_probe') else None,'events':copy.deepcopy(self.provided_preload_events),
                'intervals_by_action':copy.deepcopy(self.provided_preload_intervals),'native_target_rows':self.provided_native_target_rows},
            'provided_placement_policy':self.placement_policy,'provided_placement_segments':copy.deepcopy(self.placement_events),
            'provided_placement_actual_maximum_displacements':copy.deepcopy(self.placement_execution_max),
            'student_release_approved':False}
        if self.saved_plan is not None:
            record.update(motion_source='saved_cpu_plan',saved_bundle_sha256=self.saved_plan.sha256,
                saved_plan_file_sha256=self.saved_plan.file_sha256,runtime_plan_path_invocations=0,
                runtime_ik_calls=0,runtime_rrt_calls=0,offline_planner=copy.deepcopy(self.saved_plan.offline_planner))
        if self.private_contact_profile:
            record['private_contact_profile']=dict(fixture_only=True,pre_settled=False,
                raw_endpoint_captures=0 if self._private_contact_observer is None else len(self._private_contact_observer.raw_captures),
                supplied_intermediate_postures=False,full_mission_passed=False)
        if self.particle_mode:
            inventory=getattr(self,'initial_supply',None)
            bound=self._particle_runtime[0] if self._particle_runtime is not None else None
            contract=inventory if isinstance(inventory,dict) else bound
            record.update(particle_material_contract_sha256=None if contract is None else digest(contract),
                initial_inventory_accepted=bool(self.initialized and isinstance(inventory,dict)),
                contents_collision_queried=False,contents_query_scope='Robot/vessel/map geometry only. Native particle safety is assessed by actual physics.')
        if self.provided_bead_settle:
            record['provided_bead_material_settle']=dict(hold_s=BEAD_SETTLE_HOLD_S,maximum_extra_s=BEAD_SETTLE_EXTRA_LIMIT_S,
                events=copy.deepcopy(self.bead_settle_events),all_extra_simulation_time_counted=True,
                scope='Provided open-finger constant-command physical holds. No extra student plan, waypoint or native state replay.')
        if self.grasp_policy=='automatic':
            from physics.automatic_manipulation import record as automatic_record
            return automatic_record(self,record)
        return record

    def final_result(self,observation):
        if self.mode=='empty_path':
            from physics.empty_motion import final_result
            return final_result(self,observation)
        if self.grasp_policy=='automatic':
            from physics.automatic_manipulation import final_result
            return final_result(self,observation)
        if not self.done or len(self.completed)!=len(self.queue):raise ContractError('Student operation sequence is incomplete')
        if self.mode=='first_handoff' and (self.held is None or self.held_observation_sha256!=digest(observation)
                or self.completed[-1]['active_arm_id']!='arm_1' or self.support_id!=self.route[1]):
            raise ContractError('First-handoff completion requires the exact freshly verified recipient grasp')
        if self.mode in ('mission','first_handoff'):
            expected={op['stage_id']+'__instructor_supported_preload' for op in self.queue if op['kind']=='carry'}
            if (set(self.provided_preload_intervals)!=expected or any(v!=1400 for v in self.provided_preload_intervals.values())
                    or len(self.provided_preload_events)!=2*len(expected)
                    or self.provided_native_target_rows!=observation['native_scene_step']):
                raise ContractError('Provided supported actions and native target evidence must all complete')
        if self.mode=='source_lift_probe' and (not self.particle_mode or self.held is None
                or self.held_observation_sha256!=digest(observation) or self.provided_native_target_rows!=observation['native_scene_step']):
            raise ContractError('Private source lift needs the current retained particle grasp and all actual native target rows')
        if self.mode=='source_lift_probe':
            if (len(self.gates)!=2 or len(self.plans)!=2 or len(self.pauses)!=2
                    or any(entry.get('gate',{}).get('passed') is not True for entry in self.gates)
                    or self.gates[-1]['gate'].get('schema_version')!='particle-source-lift-probe-gate-v1'
                    or self.gates[-1]['gate'].get('particle_material_contract_sha256')!=digest(self.initial_supply)
                    or any(p.get('particle_pause_comparison',{}).get('passed') is not True for p in self.pauses)):
                raise ContractError('Private source lift requires both planned/paused operations and their physical gates')
        status={'mission':'STUDENT_THREE_ARM_SINGLE_BATCH_SAMPLED_PASS','short_motions':'STUDENT_THREE_ARM_MOTION_PROBE_PASS',
            'placement_probe':'PROVIDED_PLACEMENT_PROBE_SAMPLED_PASS','source_lift_probe':'PARTICLE_SOURCE_LIFT_PROBE_SAMPLED_PASS',
            'pick_place_unit':'PROVIDED_PICK_PLACE_UNIT_SAMPLED_PASS','first_handoff':'STUDENT_FIRST_HANDOFF_SAMPLED_PASS'}[self.mode]
        execution=self.record();completion=None;passed=False
        if self.mode=='mission':
            assess=assess_current_student_single_batch
            if self.particle_mode:
                from physics.particle_student_completion import assess_particle_student_single_batch
                assess=assess_particle_student_single_batch
            completion=assess(execution,observation,route=self.route,
                initial_supply=self.initial_supply,final_goal=self.goal,currently_held=self.held,
                execution_limits=self.world['physics_settings']['execution_limits'])
            if completion['passed'] is not True:
                raise ContractError('Current student single-batch completion rejected: '+json.dumps(completion))
            passed=True
        result={'status':status,'final_observation':observation,'student_execution':execution,
            'material_delivery_accepted':passed,'physical_handoff_verified':passed or self.mode=='first_handoff',
            'full_mission_passed':passed,'current_student_single_batch_mission_passed':passed,
            'completion_scope':'current_student_three_arm_single_batch_v1' if passed else None,
            'mission_completion_receipt':completion,'student_release_approved':False,
            'scope':('Actual first relay placement and recipient supported grasp in one retained scene. No second relay or final delivery.'
                if self.mode=='first_handoff' else 'One current student endpoint-planned three-arm bead-batch mission. No repeated throughput, other materials or student-release approval.'
                if passed else 'One real physical scene. Short probe modes are unit integration evidence, not full handoff/delivery or student release.')}
        if passed:
            result.update(physical_handoff_count=2,load_bearing_grasp_verified=True)
            if self.particle_mode:
                result.update(status='STUDENT_THREE_ARM_PARTICLE_SINGLE_BATCH_SAMPLED_PASS',
                    completion_scope='current_student_three_arm_particle_single_batch_v1',
                    delivered_count=self.initial_supply['particle_count'],delivered_mass_kg=self.initial_supply['initial_total_mass_kg'],
                    delivered_native_particle_ids_sha256=observation['particle_material']['native_particle_ids_sha256'],
                    scope='One current three-arm native-particle batch. No repeated model series, throughput, hardware or student-release approval.')
            else:
                result.update(delivered_material_ids=sorted(m['id'] for m in observation['materials']),
                    delivered_count=len(observation['materials']),delivered_mass_kg=math.fsum(m['mass_kg'] for m in observation['materials']))
        if self.mode=='placement_probe':
            result.update(placement_probe_verified=True,unit_fixture=True,
                scope='Private reconstructed INITIAL supported relay2 fixture: fresh arm1 grasp, endpoint-planned 5mm lift and local place. Not the failed mission continuation or full mission approval.')
        if self.mode=='source_lift_probe':
            result.update(source_lift_probe_verified=True,unit_fixture=True,physical_handoff_verified=False,
                physical_handoff_count=0,full_mission_passed=False,material_delivery_accepted=False,
                scope='Private particle source approach and endpoint-planned 30mm lift with unsupported hold only. No handoff, release or full-mission approval.')
        return result
