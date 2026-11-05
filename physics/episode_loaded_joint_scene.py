"""Peer-aware, isolated held-body geometry queries. Never a live attachment.

The active arm keeps seven independent native joint coordinates. Its observed
fingers are fixed. The tray and measured body-local sphere proxies follow the
observed hand/body relation only in this zero-step query scene. Other arms,
supports, obstacles and bases stay fixed. This hypothesis proves neither a
friction grasp nor dynamic contents retention, placement or a physical handoff.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

from physics.carried_body import HeldBodySpec, WorldPose, validate_provided_grasp_binding
from physics.episode_contacts import classify_episode_live_contacts
from physics.episode_joint_scene import (EpisodeJointScene, validate_episode_snapshot,
    _grounded_obstacle_floor_contact)
from physics.gravity_feedforward import rotation_matrix_wxyz
from physics.handle_contacts import (SupportedHandleContactPolicy, MAX_PENETRATION_M,
    query_witness_snapshot, verify_contact_sources)
from physics.loaded_joint_scene import POSE_TOLERANCE, _same_quaternion
from physics.joint_scene import array
from transport_challenge.contracts import ContractError, digest

IK_FK_RESTORE_SOURCES={
    'engine/solvers/rigid/abd/inverse_kinematics.py':'ddc847e984c1368ed9c410e19bcd53772c8bad38c2d148d3ff3baaf73a9ae1e4',
    'engine/solvers/rigid/abd/forward_kinematics.py':'c78116d05f8d7d2eb69d4b728f4369527f0530e1b61f6f367564262fd59e4497'}


class EpisodeLoadedJointScene(EpisodeJointScene):
    """Full-scene held query with explicitly named active-arm pad permission.

    ``robots`` and ``contents`` have the existing EpisodeJointScene schemas.
    Their q/fingers and sphere centres must be the same observation as held_body.
    The caller authenticates that recorded observation and its source bundle.
    Optional ``observed_snapshot`` has the same exact measured-state binding
    as EpisodeJointScene. It permits only that initializer's existing native
    readback residual, never a relaxed planning-state or edge bound.
    No target, route or intermediate pose is supplied by this constructor.
    Without an explicit bound policy, no held state can be accepted.
    """
    def __init__(self, case, *, active_arm_id, robots, held_body,
                 asset_directory, support_stations=(), contents=(),
                 contact_policy=None, maximum_body_tilt_deg=None, render=False,
                 observed_snapshot=None, attachment_binding=None):
        if type(held_body) is not HeldBodySpec:
            raise ContractError('Use an immutable observed HeldBodySpec')
        self._provided_attachment=None
        if held_body.grasp_model=='provided_ideal':
            if contact_policy is not None or type(observed_snapshot) is not dict:
                raise ContractError('Provided ideal query requires current attachment binding, not friction policy')
            self._provided_attachment=validate_provided_grasp_binding(held_body,attachment_binding,
                active_arm_id=active_arm_id,observation=observed_snapshot['observation'])
        elif attachment_binding is not None:
            raise ContractError('Friction query cannot claim an automatic attachment binding')
        if contact_policy is not None:
            if type(contact_policy) is not SupportedHandleContactPolicy:
                raise ContractError('Unsupported held contact policy')
            contact_policy.validate_binding(held_body)
        if (maximum_body_tilt_deg is not None and
                (type(maximum_body_tilt_deg) not in (int,float) or not math.isfinite(maximum_body_tilt_deg)
                 or not 0.<=maximum_body_tilt_deg<=180.)):
            raise ContractError('Optional body tilt is a finite angle in [0,180] degrees')
        snapshot=validate_episode_snapshot(case,active_arm_id,robots,held_body.model,
            held_body.measured_body_world,held_body.object_id,support_stations,contents,
            observed_snapshot=observed_snapshot)
        active=next(r for r in snapshot['robots'] if r['id']==active_arm_id)
        if tuple(active['finger_q_m'])!=held_body.finger_dof_positions_m:
            raise ContractError('Active fingers differ from the exact held observation')
        self.held_body=held_body;self.contact_policy=contact_policy
        self.maximum_body_tilt_deg=maximum_body_tilt_deg
        source_root,_=verify_contact_sources()
        for name,h in IK_FK_RESTORE_SOURCES.items():
            if hashlib.sha256((source_root/name).read_bytes()).hexdigest()!=h:
                raise ContractError('Reviewed native IK/FK restore source changed: '+name)
        self._ik_readback_pending=False;self._ik_fk_refreshes=0
        self._proxy_updates=0;self._proxy_readback_checks=0;self._last_proxy=None
        body=held_body.measured_body_world
        inverse_rotation=rotation_matrix_wxyz(body.quaternion_wxyz).T
        self._content_local=tuple({**copy.deepcopy(s),
            'center_body_m':(inverse_rotation@(np.asarray(s['center_world_m'])-body.position_m)).tolist()}
            for s in snapshot['contents'])
        self._held_snapshot_sha256=digest({'episode_snapshot':snapshot,'held_body':held_body.to_record(),
            'content_body_local_proxies':self._content_local})
        super().__init__(case,active_arm_id=active_arm_id,robots=robots,model=held_body.model,
            body_pose=body,body_id=held_body.object_id,asset_directory=asset_directory,
            support_stations=support_stations,contents=contents,render=render,
            observed_snapshot=observed_snapshot)
        try:
            # The native start configuration must reproduce the observed pose.
            # This is the existing query numerical readback tolerance, not a
            # relaxed live grasp/drift or collision acceptance threshold.
            self._source_reproduction={
                'hand_position_error_m':math.dist(self._last_proxy['hand_world']['position_m'],held_body.measured_hand_world.position_m),
                'body_position_error_m':math.dist(self._last_proxy['body_world']['position_m'],body.position_m),
                'query_pose_readback_tolerance_m':POSE_TOLERANCE}
            if (self._source_reproduction['hand_position_error_m']>POSE_TOLERANCE or
                    self._source_reproduction['body_position_error_m']>POSE_TOLERANCE or
                    not _same_quaternion(self._last_proxy['hand_world']['quaternion_wxyz'],held_body.measured_hand_world.quaternion_wxyz) or
                    not _same_quaternion(self._last_proxy['body_world']['quaternion_wxyz'],body.quaternion_wxyz)):
                raise ContractError('Initial native q/FK does not reproduce the bound held observation')
            self.initial_state_validation['scope']='Held snapshot geometry, not a load-bearing or retention result'
        except BaseException:
            self.close()
            raise

    def _read_frozen_state(self):
        state=super()._read_frozen_state()
        # Only these declared proxies may move. Every remaining original field
        # is compared exactly, including peer q, active fingers and robot roots.
        for name in (self.tray.name,*(entity.name for entity in self.contents)):
            del state['bodies'][name]
        return state

    def _check_immutable_world(self):
        if self._baseline is not None and self._read_frozen_state()!=self._baseline:
            raise ContractError('A peer, finger, robot base, obstacle or support changed during held query')
        self._fixed_checks+=1

    def _after_arm_q_placement(self):
        self._check_immutable_world()
        if self._ik_readback_pending:
            # Native IK restores the original q using entity FK. Reissue those
            # exact values to obtain the ordinary global FK also used by the
            # later proxy setters. This is query-only cache canonicalization,
            # never a new configuration, tolerance or live command.
            unchanged_q=array(self.arm.get_qpos()).copy()
            self.arm.set_qpos(unchanged_q)
            if not np.array_equal(array(self.arm.get_qpos()),unchanged_q):
                raise ContractError('Same-q query FK refresh changed active joint state')
            self._ik_fk_refreshes+=1
        hand_raw=self._pose(self.hand)
        q_at_composition=array(self.arm.get_qpos()).copy().tolist()
        hand=WorldPose(hand_raw['position_m'],hand_raw['quaternion_wxyz'])
        body=self.held_body.body_pose_at(hand)
        self.tray.set_pos(body.position_m,relative=False,zero_velocity=True)
        self.tray.set_quat(body.quaternion_wxyz,relative=False,zero_velocity=True)
        targets={}
        for entity,sphere in zip(self.contents,self._content_local):
            target=body.transform_point(sphere['center_body_m']);targets[entity.name]=list(target)
            entity.set_pos(target,relative=False,zero_velocity=True)
        self._proxy_updates+=1
        self._last_proxy={'hand_world':hand_raw,'expected_body_world':body.to_record(),
            'expected_content_centers_world_m':targets,'query_pose_update_index':self._proxy_updates,
            'active_q_at_composition':q_at_composition}
        self._check_proxy_readbacks()

    def _check_proxy_readbacks(self):
        """Read only. Never move proxies after copying contact witnesses."""
        self._check_immutable_world()
        if self._last_proxy is None:raise ContractError('Held query lacks a composed proxy pose')
        current_hand=self._pose(self.hand)
        if current_hand!=self._last_proxy['hand_world']:
            raise ContractError('Active hand changed after the composed query pose: '+repr({
                'before':self._last_proxy['hand_world'],'after':current_hand,
                'active_q_at_composition':self._last_proxy['active_q_at_composition'],
                'active_q_now':array(self.arm.get_qpos()).copy().tolist(),
                'query_pose_update_index':self._proxy_updates}))
        body=self._pose(self.tray);expected=self._last_proxy['expected_body_world']
        if (math.dist(body['position_m'],expected['position_m'])>POSE_TOLERANCE or
                not _same_quaternion(body['quaternion_wxyz'],expected['quaternion_wxyz'])):
            raise ContractError('Query tray does not match measured-frame composition')
        actual={}
        if len(self.contents)!=len(self._content_local):raise ContractError('Content proxy inventory changed')
        for entity,sphere in zip(self.contents,self._content_local):
            if entity.name!=sphere['id']:raise ContractError('Content proxy identity/order changed')
            center=self._pose(entity)['position_m'];actual[entity.name]=center
            if math.dist(center,self._last_proxy['expected_content_centers_world_m'][entity.name])>POSE_TOLERANCE:
                raise ContractError('Query sphere does not match its measured body-local position')
        self._last_proxy.update(body_world=body,content_centers_world_m=actual)
        self._proxy_readback_checks+=1

    def current_body_pose(self):
        self._require_open();self._check_proxy_readbacks()
        return copy.deepcopy(self._last_proxy['body_world'])

    def ik(self,position_m,quaternion_wxyz,seed_q,*,position_tolerance_m=None,orientation_tolerance_rad=None):
        if self._ik_readback_pending:raise ContractError('Nested held-query IK is unsupported')
        self._ik_readback_pending=True
        precision={}
        if position_tolerance_m is not None:precision['position_tolerance_m']=position_tolerance_m
        if orientation_tolerance_rad is not None:precision['orientation_tolerance_rad']=orientation_tolerance_rad
        try:return super().ik(position_m,quaternion_wxyz,seed_q,**precision)
        finally:self._ik_readback_pending=False

    def _additional_state_valid(self):
        if self.maximum_body_tilt_deg is None:return True
        z=rotation_matrix_wxyz(self._last_proxy['body_world']['quaternion_wxyz'])[:,2]
        tilt=math.degrees(math.acos(float(np.clip(z[2],-1.,1.))))
        return True if tilt<=self.maximum_body_tilt_deg else self._reject('body_tilt_constraint',
            body_tilt_deg=tilt,maximum_body_tilt_deg=self.maximum_body_tilt_deg)

    def _query_collision_pairs(self):
        self._check_proxy_readbacks()
        finger_poses={name:{finger:self._pose(arm.get_link(finger)) for finger in ('left_finger','right_finger')}
                      for name,arm in self.arms.items()}
        pairs=self.scene.rigid_solver.detect_collision()
        raw=query_witness_snapshot(self.scene.rigid_solver.collider,pairs,set(self._owners),self._owners)
        self.scene.rigid_solver.check_errno();self.collision_queries+=1
        self._check_proxy_readbacks()
        # No setter occurs between the query, raw copy and exact-pose checks.
        if any(self._pose(self.arms[name].get_link(f))!=p for name,v in finger_poses.items() for f,p in v.items()):
            raise ContractError('Finger pose changed during same-query contact copy')
        rows,bilateral=classify_episode_live_contacts(raw,owners=self._owners,
            components_by_geom={v['geom_id']:v for v in self._components},
            pads_by_arm={name:{v['geom_id']:v for v in pads} for name,pads in self._pads.items()},
            body_pose=self._last_proxy['body_world'],finger_poses_by_arm=finger_poses,
            stations_by_geom=self._station_map,phase='carry',
            active_holder_arm_id=self.active_arm_id if self.contact_policy is not None else None)
        for row in rows:
            kinds={self._owners[g]['kind'] for g in row['geom_ids']}
            if (row['category']=='UNALLOWED_CONTACT' and kinds=={'obstacle','floor'} and
                    0.<=row['penetration_m']<=MAX_PENETRATION_M and row['normal_unit_error']<=.001 and
                    _grounded_obstacle_floor_contact(row,self._owners,self._obstacle_map)):
                row['category']='named_grounded_obstacle_bottom_floor'
        forbidden=sorted({tuple(sorted(r['geom_ids'])) for r in rows if r['category']=='UNALLOWED_CONTACT'})
        self._last_detection={'public_pair_count':len(pairs),'public_pairs':np.asarray(pairs).copy().tolist(),
            'raw_contacts':{k:np.asarray(v).copy().tolist() for k,v in raw.items()},
            'copied_contact_count':len(rows),'pair_witness_multiset_agrees':True,'rows':rows,
            'forbidden_pairs':[list(p) for p in forbidden],'active_arm_id':self.active_arm_id,
            'bilateral_by_arm':bilateral,'same_query_finger_poses_world':finger_poses,
            'same_query_proxy_poses':copy.deepcopy(self._last_proxy),'forces_used':False}
        return np.asarray(forbidden,dtype=np.int32).reshape((-1,2))

    def state_valid(self,q):
        valid=super().state_valid(q)
        if self.held_body.grasp_model=='provided_ideal':
            # The validated current native attachment replaces the holder
            # witness only. All queried contacts still use the original
            # classifier, with no hand/cup or finger pair exemption.
            return valid
        if valid and (self.contact_policy is None or not self._last_detection['bilateral_by_arm'][self.active_arm_id]):
            return self._reject('missing_active_bilateral_handle_contact')
        return valid

    def evidence(self):
        record=super().evidence()
        scope=('held_geometry_with_measured_body_local_sphere_proxies' if self.contents else 'held_geometry_without_content_proxies')
        record['snapshot']['motion_scope']=scope
        record['snapshot_sha256']=digest(record['snapshot'])
        record.update(schema_version='episode-loaded-joint-query-evidence-v1',
            motion_scope=scope,held_carried_query_supported=True,
            loaded_payload_checked=self.collision_queries>0,held_body=self.held_body.to_record(),
            held_body_binding_sha256=self.held_body.binding_sha256,held_snapshot_sha256=self._held_snapshot_sha256,
            source_observation_reproduction=copy.deepcopy(self._source_reproduction),
            content_body_local_proxies=copy.deepcopy(self._content_local),last_composed_proxy=copy.deepcopy(self._last_proxy),
            body_pose_setters=self._proxy_updates*2,content_position_setters=self._proxy_updates*len(self.contents),
            isolated_proxy_pose_updates=self._proxy_updates,proxy_readback_checks=self._proxy_readback_checks,
            query_pose_readback_tolerance_m=POSE_TOLERANCE,maximum_body_tilt_deg=self.maximum_body_tilt_deg,
            ik_same_q_global_fk_refreshes=self._ik_fk_refreshes,ik_fk_restore_source_sha256=copy.deepcopy(IK_FK_RESTORE_SOURCES),
            contact_policy={'explicit_opt_in':self.contact_policy is not None,'held_binding_sha256':self.held_body.binding_sha256,
                'permission_arm_id':self.active_arm_id if self.contact_policy else None,
                'accepted_state_requires_active_bilateral_witnesses':True,'one_global_query_per_state':True,
                'cross_arm_contact_allowances':[],'unknown_contacts':'reject'},
            proxy_motion_scope='Isolated zero-step scene only. No live pose commands or welds.',
            contents_model_scope=('Measured body-local sphere proxies. No dynamic motion or retention proof.' if self.contents else
                'No free-particle/content geometry in this query. Only robot, vessel and map geometry. Actual native physics must verify particle safety/retention.'),
            physical_grasp_verified=False,physical_carry_verified=False,physical_handoff_verified=False,
            material_retention_verified=False,continuous_collision_certificate=False)
        if self.held_body.grasp_model=='provided_ideal':
            record.update(provided_attachment_binding=copy.deepcopy(self._provided_attachment),
                contact_policy={'explicit_opt_in':True,'grasp_model':'provided_ideal',
                    'held_binding_sha256':self.held_body.binding_sha256,
                    'attachment_receipt_sha256':self.held_body.attachment_receipt_sha256,
                    'accepted_state_requires_active_bilateral_witnesses':False,
                    'one_global_query_per_state':True,'unknown_contacts':'reject',
                    'hand_cup_query_pair_exemptions':[],'cross_arm_contact_allowances':[],
                    'native_policy':'Only the actual attached hand-link/cup-link pair is excluded by the native weld.',
                    'query_policy':'All geometry pairs remain checked. This is conservative relative to the native weld exclusion.'})
        return record
