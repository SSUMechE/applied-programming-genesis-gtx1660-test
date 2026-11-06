"""Bind native interval contacts and measured post-step roots to handling limits.

The contact classifier authenticates contacts at the pre-integration pose.
This adapter states both clocks, enforces the post-step hand/root reach bound,
and advances one persistent UnheldMotionGuard. It never changes native state.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from transport_challenge.contracts import ContractError
from physics.episode_handling_guard import UnheldMotionGuard, check_body_root_envelope

MAX_HELD_ROOT_TO_HAND_M = .20
REACH_NUMERIC_ALLOWANCE_M = .000020
RIGID16_PROFILE = 'rigid16_bead_tc2h_v1'


def authenticate_native_reach(arms, scenario, *, model_file, model_sha256,
                              revolute_type, array_value):
    """Bind the analytic Panda radius to native fixed bases and actual link poses.

    The continuous-state enforcement below also checks every accepted native
    post-step hand/root against this radius. Initial pose agreement alone does
    not replace that per-step check.
    """
    path = Path(model_file)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=model_sha256:
        raise ContractError('Reach envelope model differs from the authored robot')
    xml = ET.parse(path).getroot()
    bodies = {b.attrib['name']:b for b in xml.iter('body') if 'name' in b.attrib}
    chain = ('link1','link2','link3','link4','link5','link6','link7','hand')
    parents = {b.attrib['name']:parent.attrib.get('name') for parent in xml.iter('body')
               for b in parent.findall('body')}
    if any(parents[name]!=parent for name,parent in zip(chain[1:],chain[:-1])):
        raise ContractError('Reach proof requires the exact Panda shoulder/hand ancestry')
    offset = tuple(float(v) for v in bodies['link1'].attrib.get('pos','0 0 0').split())
    if offset!=(0.,0.,.333):raise ContractError('Panda shoulder offset is not the certified yaw-invariant offset')
    lengths = [math.hypot(*(float(v) for v in bodies[name].attrib.get('pos','0 0 0').split())) for name in chain[1:]]
    nominal_radius = sum(lengths)
    records = {}
    for record in scenario['robots']:
        name = record['id'];arm = arms[name]
        root = arm.get_link('link0');links = [arm.get_link(n) for n in chain]
        if not root.is_fixed:raise ContractError('Analytic reach proof requires an actually fixed native robot base')
        if any(arm.get_joint('joint'+str(i)).type!=revolute_type for i in range(1,8)):
            raise ContractError('Native Panda main joints differ from the seven revolute joints')
        if any(link.parent_idx!=parent.idx for link,parent in zip(links,[root,*links[:-1]])):
            raise ContractError('Native parent chain differs from the exact source model')
        positions = [list(map(float,array_value(link.get_pos(relative=False)).reshape(3))) for link in links]
        native_lengths = [math.dist(a,b) for a,b in zip(positions[1:],positions[:-1])]
        if any(abs(a-b)>.000002 for a,b in zip(native_lengths,lengths)):
            raise ContractError('Native link separations do not match the analytic source lengths')
        base = list(map(float,array_value(root.get_pos(relative=False)).reshape(3)))
        declared_base = [record['base_placement']['x_m'],record['base_placement']['y_m'],
                         record.get('base_z_m',scenario['case']['base_z_m'])]
        if (not all(math.isfinite(v) for v in [*base,*declared_base,*sum(positions,[])])
                or math.dist(base,declared_base)>.000002):
            raise ContractError('Actual native base is not the declared fixed installation position')
        shoulder = [base[0],base[1],base[2]+offset[2]]
        if math.dist(positions[0],shoulder)>.000002:
            raise ContractError('Actual native shoulder differs from the fixed-base shoulder')
        records[name] = {'passed':True,'fixed_base_world_m':base,'declared_base_world_m':declared_base,
                         'base_and_shoulder_parity_allowance_m':.000004,'shoulder_world_m':positions[0],
                         'source_link_lengths_m':lengths,'native_initial_link_lengths_m':native_lengths,
                         'source_shoulder_to_hand_radius_m':nominal_radius,
                         'numeric_allowance_m':REACH_NUMERIC_ALLOWANCE_M,
                         'accepted_hand_radius_m':nominal_radius+REACH_NUMERIC_ALLOWANCE_M}
    return {'schema_version':'native-panda-reach-envelope-v1','passed':True,
            'robot_model_sha256':model_sha256,'by_arm':records,
            'scope':'Source chain upper bound authenticated at build and enforced on accepted post-step hand/root observations.'}


class SampledHandlingEnforcer:
    """One persistent native vessel, fixed supports, and contiguous native steps."""

    def __init__(self, *, body_identity, model_geometry, holder_ids, support_ids,
                 initial_observation, dt_s, reach_envelopes=None, automatic_grasp=None,
                 rigid_profile=None):
        if rigid_profile not in (None,RIGID16_PROFILE) or (rigid_profile is not None and dt_s != .0025):
            raise ContractError('Handling profile must be the exact named rigid16 clock')
        self.rigid_profile = rigid_profile
        envelope = check_body_root_envelope(model_geometry['component_boxes'])
        if not envelope['passed']:
            raise ContractError('Authored vessel exceeds the declared root envelope')
        self.identity = copy.deepcopy(body_identity)
        self.holders = tuple(holder_ids)
        self.supports = tuple(support_ids)
        self.dt = float(dt_s)
        self.envelope = envelope
        self.reach_envelopes = copy.deepcopy(reach_envelopes)
        self.guard = UnheldMotionGuard(body_identity=body_identity, known_holder_ids=self.holders,
            allowed_support_ids=self.supports,
            initial_root_world_m=initial_observation['body_world']['position_m'],
            initial_scene_step=initial_observation['native_scene_step']*(16 if rigid_profile else 1),
            initial_simulation_s=initial_observation['simulation_s'], dt_s=dt_s/(16 if rigid_profile else 1))
        self.automatic_grasp = automatic_grasp
        self._automatic_pre = None
        self.last = None
        self.maximum_held_root_to_hand_m = 0.

    def capture_automatic_grasp(self, pre):
        """Call before scene.step, after any between-interval attach/release."""
        if self.automatic_grasp is None:
            return None
        from physics.live_contact_coverage import automatic_grasp_witness
        proof = automatic_grasp_witness(self.automatic_grasp)
        self._bind_automatic_observation(proof, pre)
        self._automatic_pre = proof
        return copy.deepcopy(proof)

    def _bind_automatic_observation(self, proof, observation):
        snapshot = proof['snapshot']
        if (snapshot['body_entity_idx'] != self.identity['entity_idx']
                or snapshot['native_clock'][:2] != [observation['native_scene_step'],observation['simulation_s']]
                or snapshot['body_world'] != observation['body_world']
                or snapshot['hand_world'] != observation['arms'][next(n for n,i in proof['hand_links_by_arm'].items() if i==snapshot['hand_link_idx'])]['hand_world']):
            raise ContractError('Automatic holder is not this same-clock body/hand observation')

    def _automatic_interval(self, pre, post):
        from physics.live_contact_coverage import automatic_grasp_witness
        before = self._automatic_pre
        if before is None:
            raise ContractError('Automatic handling requires a fresh pre-step membership witness')
        self._bind_automatic_observation(before, pre)
        after = automatic_grasp_witness(self.automatic_grasp)
        self._bind_automatic_observation(after, post)
        keys = ('holder_arm_id','owned_weld_row','baseline_inventory','hand_links_by_arm')
        if (any(before[k] != after[k] for k in keys)
                or before['snapshot']['inventory'] != after['snapshot']['inventory']
                or before['snapshot']['cup_mass_kg'] != after['snapshot']['cup_mass_kg']):
            raise ContractError('Automatic attach/release may occur only between completed intervals')
        self._automatic_pre = None
        return before,after

    def observe_interval(self, *, pre, post, classified_contacts, bilateral_by_arm,
                         contacts_complete, active_contact_permission_arm, contact_detection_clock=None,
                         rigid_substep_samples=None, rigid_profile=None):
        """Consume one completed native interval, not a command-based grip claim.

        pre/post are the immutable runtime observations. classified_contacts and
        bilateral_by_arm are the complete native classifier output at pre by
        default. Coupled100 mode supplies its actual last-internal clock. The
        actor argument only grants contact permission. It cannot grant a holder
        without that actor's bilateral witness and a valid post-root bound.
        Explicit automatic mode instead requires native pre/post weld membership;
        bilateral contact remains the unchanged measured diagnostic.
        """
        if self.rigid_profile == RIGID16_PROFILE:
            if rigid_profile not in (None,RIGID16_PROFILE) or contact_detection_clock is not None:
                raise ContractError('Rigid16 cannot reuse a final coupled100 contact clock')
            return self._observe_rigid16(pre=pre,post=post,samples=rigid_substep_samples,
                contacts_complete=contacts_complete,active_arm=active_contact_permission_arm)
        if rigid_profile is not None or rigid_substep_samples is not None:
            raise ContractError('Rigid16 captures require an explicitly constructed named16 enforcer')
        pre_step, post_step = pre['native_scene_step'], post['native_scene_step']
        pre_time, post_time = pre['simulation_s'], post['simulation_s']
        if (type(pre_step) is not int or type(post_step) is not int or post_step != pre_step+1
                or not math.isfinite(pre_time) or not math.isfinite(post_time)
                or abs(pre_time-pre_step*self.dt)>1e-9 or abs(post_time-post_step*self.dt)>1e-9):
            raise ContractError('Handling evidence requires exact native pre/post interval clocks')
        if type(contacts_complete) is not bool or not contacts_complete:
            raise ContractError('Handling evidence requires complete native contact classification')
        witness_time=pre_time
        if contact_detection_clock is not None:
            clock=contact_detection_clock
            if (type(clock) is not list or len(clock)!=4 or any(type(v) is not int for v in clock[:3])
                    or clock[:3]!=[(pre_step+1)*100-1,0,pre_step]
                    or type(clock[3]) not in (int,float) or not math.isfinite(clock[3])
                    or abs(clock[3]-clock[0]*(self.dt/100))>1e-12
                    or not pre_time<=clock[3]<post_time):
                raise ContractError('Handling contact clock must be the actual final coupled100 substep')
            witness_time=clock[3]
        automatic_interval = None
        if self.automatic_grasp is None:
            witnessed = [name for name in self.holders if bilateral_by_arm.get(name) is True]
            if len(witnessed)>1:
                raise ContractError('Supported handoff cannot directly switch between simultaneous holders')
            qualified = witnessed[0] if witnessed else None
            if qualified is not None and qualified != active_contact_permission_arm:
                raise ContractError('Bilateral contact belongs to an unauthorized manipulation actor')
        else:
            automatic_interval = self._automatic_interval(pre, post)
            qualified = automatic_interval[1]['holder_arm_id']
            if qualified is not None and (qualified not in self.holders or qualified != active_contact_permission_arm):
                raise ContractError('Automatic holder belongs to an unauthorized manipulation actor')
        root = post['body_world']['position_m']
        distance = None
        if qualified is not None:
            hand = post['arms'][qualified]['hand_world']['position_m']
            distance = math.dist(root, hand)
            if not math.isfinite(distance) or distance>MAX_HELD_ROOT_TO_HAND_M:
                raise ContractError('Post-step body root exceeds the bilateral grasp reach envelope')
            self.maximum_held_root_to_hand_m = max(self.maximum_held_root_to_hand_m, distance)
            if self.reach_envelopes is not None:
                arm_reach = self.reach_envelopes['by_arm'][qualified]
                shoulder = arm_reach['shoulder_world_m'];radius = arm_reach['accepted_hand_radius_m']
                if math.dist(hand,shoulder)>radius or math.dist(root,shoulder)>radius+MAX_HELD_ROOT_TO_HAND_M:
                    raise ContractError('Accepted post-step grasp exceeds the native authenticated robot reach envelope')
        supports = sorted({row['support_station_id'] for row in classified_contacts
                           if row['category']=='named_tray_bottom_support_top'})
        allowed = all(row['category']!='UNALLOWED_CONTACT' for row in classified_contacts)
        facts = {'schema_version':'episode-handling-observation-v1',
                 'body_identity':self.identity, 'scene_step':post_step, 'simulation_s':post_time,
                 'body_root_world_m':list(root), 'qualified_holder_id':qualified,
                 'support_ids':supports, 'manipulation_contacts_allowed':allowed}
        verdict = self.guard.observe(facts)
        self.last = {'passed':verdict['passed'], 'contact_witness_scene_step':pre_step,
                     'contact_witness_simulation_s':witness_time,
                     'root_observation_scene_step':post_step, 'root_observation_simulation_s':post_time,
                     'contact_scope':'Native pre-pose contact witness for the completed interval; not a post-pose contact recomputation.',
                     'qualified_holder_id':qualified, 'support_ids':supports,
                     'post_root_to_hand_m':distance, 'post_root_to_hand_limit_m':MAX_HELD_ROOT_TO_HAND_M,
                     'unheld_displacement_m':verdict['current_unheld_displacement_m'],
                     'first_failure':verdict['first_failure']}
        if contact_detection_clock is not None:
            self.last.update(contact_witness_native_clock=list(contact_detection_clock),
                contact_scope='Actual final25us preintegration contact sample of this2.5ms outer interval. Root/hand bounds use the outer post observation. No continuous/all100-subinterval certificate.')
        if automatic_interval is not None:
            self.last.update(schema_version='automatic-grasp-native-interval-handling-v1',
                automatic_grasp_pre=automatic_interval[0], automatic_grasp_post=automatic_interval[1],
                bilateral_by_arm_diagnostic=copy.deepcopy(bilateral_by_arm),
                holder_model='provided-ideal-native-weld-v1')
        return copy.deepcopy(self.last)

    def _observe_rigid16(self, *, pre, post, samples, contacts_complete, active_arm):
        from physics.episode_transport_assessment import rigid16_interval_facts
        pre_step,step = pre['native_scene_step'],post['native_scene_step']
        if (type(pre_step) is not int or type(step) is not int or step != pre_step+1
                or abs(pre['simulation_s']-pre_step*self.dt)>1e-9
                or abs(post['simulation_s']-step*self.dt)>1e-9 or contacts_complete is not True):
            raise ContractError('Rigid16 handling needs contiguous actual completed outer clocks')
        facts = rigid16_interval_facts(samples,step=step,dt=self.dt)
        for actual,outer in ((samples[0]['detection']['rigid_state'],pre),
                             (samples[-1]['post_integration_state'],post)):
            if actual['body_world'] != outer['body_world'] or set(actual['arms']) != set(self.holders):
                raise ContractError('Rigid16 first/last actual state differs from the outer body/arms')
            for name in self.holders:
                outer_fingers=outer['arms'][name]['finger_world']
                if set(outer_fingers)=={'left_finger','right_finger'}:
                    outer_fingers={'left':outer_fingers['left_finger'],'right':outer_fingers['right_finger']}
                if (actual['arms'][name]['hand_world'] != outer['arms'][name]['hand_world']
                        or actual['arms'][name]['finger_world'] != outer_fingers):
                    raise ContractError('Rigid16 actual hand/finger endpoint differs from the outer observation')
        automatic = self._automatic_interval(pre,post) if self.automatic_grasp is not None else None
        verdict = None; distance = None; all_passed = True; holder = None
        for sample,timing in zip(samples,facts['timeline']):
            measured = sample['post_integration_state']; bilateral = timing['bilateral_by_arm']
            if automatic is None:
                if sample['policy_stamp'].get('automatic_grasp') is not None:
                    raise ContractError('Nonautomatic rigid16 handling cannot borrow a weld-qualified holder')
                witnessed = [name for name in self.holders if bilateral.get(name) is True]
                if len(witnessed)>1: raise ContractError('Rigid16 cannot directly switch simultaneous holders')
                holder = witnessed[0] if witnessed else None
            else:
                holder = automatic[1]['holder_arm_id']
                proof = sample['policy_stamp'].get('automatic_grasp')
                if (proof is None or proof['holder_arm_id'] != holder
                        or proof['snapshot']['inventory'] != automatic[1]['snapshot']['inventory']
                        or proof['baseline_inventory'] != automatic[1]['baseline_inventory']
                        or proof['snapshot']['cup_mass_kg'] != automatic[1]['snapshot']['cup_mass_kg']):
                    raise ContractError('Rigid16 actual internal owned-weld inventory/mass differs from outer witnesses')
            if holder is not None and (holder not in self.holders or holder != active_arm):
                raise ContractError('Rigid16 holder is not the authorized actual manipulation actor')
            root = measured['body_world']['position_m']; distance = None
            if holder is not None:
                hand = measured['arms'][holder]['hand_world']['position_m'];distance = math.dist(root,hand)
                if not math.isfinite(distance) or distance>MAX_HELD_ROOT_TO_HAND_M:
                    raise ContractError('Actual rigid16 post root exceeds the unchanged held hand bound')
                self.maximum_held_root_to_hand_m=max(self.maximum_held_root_to_hand_m,distance)
                if self.reach_envelopes is not None:
                    reach=self.reach_envelopes['by_arm'][holder];shoulder=reach['shoulder_world_m'];radius=reach['accepted_hand_radius_m']
                    if math.dist(hand,shoulder)>radius or math.dist(root,shoulder)>radius+MAX_HELD_ROOT_TO_HAND_M:
                        raise ContractError('Actual rigid16 post hand/root exceeds native authenticated robot reach')
            native=dict(schema_version='episode-handling-observation-v1',body_identity=self.identity,
                scene_step=timing['global_substep']+1,simulation_s=timing['end_s'],body_root_world_m=list(root),
                qualified_holder_id=holder,support_ids=timing['support_ids'],
                manipulation_contacts_allowed=timing['contacts_allowed'])
            verdict=self.guard.observe(native);all_passed &= verdict['passed']
        common_supports=set(facts['timeline'][0]['support_ids'])
        for timing in facts['timeline'][1:]:common_supports.intersection_update(timing['support_ids'])
        self.last=dict(schema_version='rigid16-native-interval-handling-v1',runtime_profile=RIGID16_PROFILE,
            passed=bool(all_passed),contact_witness_scene_step=pre_step,root_observation_scene_step=step,
            root_observation_simulation_s=post['simulation_s'],qualified_holder_id=holder,
            support_ids=sorted(common_supports),post_root_to_hand_m=distance,
            post_root_to_hand_limit_m=MAX_HELD_ROOT_TO_HAND_M,
            unheld_displacement_m=verdict['current_unheld_displacement_m'],first_failure=verdict['first_failure'],
            interval_facts=facts,contact_scope='All actual16 detection contacts plus16 actual produced post roots, each h. No last-sample outerDT or continuous-time claim.')
        if automatic is not None:
            self.last.update(automatic_grasp_pre=automatic[0],automatic_grasp_post=automatic[1],
                             holder_model='provided-ideal-native-weld-v1')
        return copy.deepcopy(self.last)

    def record(self):
        result = {'schema_version':'native-interval-handling-enforcement-v1',
                'body_root_envelope':copy.deepcopy(self.envelope),
                'maximum_held_root_to_hand_m':self.maximum_held_root_to_hand_m,
                'held_root_to_hand_limit_m':MAX_HELD_ROOT_TO_HAND_M,
                'guard':self.guard.status(), 'last_interval':copy.deepcopy(self.last),
                'continuous_time_collision_proof':False}
        if self.automatic_grasp is not None:
            result.update(schema_version='automatic-grasp-native-interval-handling-enforcement-v1',
                holder_model='provided-ideal-native-weld-v1')
        if self.rigid_profile == RIGID16_PROFILE:
            result.update(runtime_profile=RIGID16_PROFILE,guard_dt_s=self.dt/16,
                          guard_scene_step_scope='actual_internal_global_substep')
        return result
