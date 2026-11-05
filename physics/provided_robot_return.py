"""Reverse frozen outbound commands without advancing physical simulation.

Only a released donor's native joints move. This explicitly authorized display
operation is not dynamic motion or a swept collision certificate. Real payload
states, peers, bases and the original native clock remain unchanged.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import time

from transport_challenge.contracts import ContractError,digest
from physics import provided_robot_reset as reset

SCHEMA='provided-kinematic-robot-return-v1'
MODEL='provided-post-release-reverse-command-return-v1'


def _source_hash(schedule,limits=None):
    from physics.saved_motion_planning import iter_kinematic_return_samples
    hasher=hashlib.sha256();count=0
    for sample in iter_kinematic_return_samples(schedule,limits=limits):
        hasher.update((json.dumps(sample,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8'))
        count+=1
    return hasher.hexdigest(),count


def validate_kinematic_return_receipt(receipt, *, before_observation=None,after_observation=None,
        expected_schedule=None,expected_command=None,expected_arm_id=None,expected_stage_id=None):
    reset.require(type(receipt) is dict and receipt.get('schema_version')==SCHEMA
        and receipt.get('model')==MODEL,'Exact supplied physics-free reverse-return receipt required')
    projected=dict(receipt,schema_version=reset.SCHEMA,model=reset.MODEL)
    reset.validate_robot_reset_receipt(projected,before_observation=before_observation,
        after_observation=after_observation,expected_command=expected_command,
        expected_arm_id=expected_arm_id,expected_stage_id=expected_stage_id)
    schedule=receipt.get('return_schedule')
    reset.require(type(schedule) is dict and (expected_schedule is None or schedule==expected_schedule),
        'Kinematic return differs from its exact saved outbound trajectories')
    reset.require(schedule.get('arm_id')==receipt['arm_id'] and schedule.get('after_stage_id')==receipt['after_stage_id']
        and schedule.get('target_command')==receipt['target_command']
        and [leg['source_stage_id'] for leg in schedule['legs']]==[receipt['arm_id']+'__carry',receipt['arm_id']+'__approach'],
        'Kinematic return donor, target or ordered source stages differ')
    checksum,count=_source_hash(schedule)
    reset.require(receipt.get('source_samples_sha256')==checksum and receipt.get('applied_sample_count')==count
        and receipt.get('pose_setter_count')==count and receipt.get('actual_q9_verified_samples')==count
        and receipt.get('display_duration_s')==schedule['display_duration_s']
        and receipt.get('return_schedule_sha256')==digest(schedule)
        and receipt.get('physical_steps')==0 and receipt.get('simulation_duration_s')==0.,
        'Kinematic return lacks exact reverse sample, display-time or setter accounting')
    reset.require(receipt.get('trajectory_order')==[leg['source_stage_id'] for leg in schedule['legs']]
        and receipt.get('clock_checked_samples')==count and receipt.get('swept_collision_certified') is False
        and receipt.get('physics_steps_during_return')==0,'Return order or frozen native-clock proof differs')
    transcript=receipt.get('native_transcript',{})
    reset.require(transcript.get('rows')==count+1 and transcript.get('includes_initial_actual_state') is True
        and type(transcript.get('sha256')) is str and len(transcript['sha256'])==64,
        'Every native return sample and its initial actual state must be recorded')
    frames=receipt.get('recorded_frames')
    reset.require(type(frames) is list and len(frames)>=2 and receipt.get('recorded_frames_fps')==15
        and receipt.get('recorded_frames_sha256')==digest(frames), 'Return display frames lack actual-state binding')
    clock=receipt['native_before']['native_clock']
    reset.require(all(frame['step']==clock[0] and frame['simulation_s']==clock[1]
        and frame['native_clock']==clock for frame in frames)
        and frames[0]['display_s']==0. and frames[-1]['display_s']==receipt['display_duration_s']
        and all(a['display_s']<=b['display_s'] for a,b in zip(frames,frames[1:])),
        'Return display time must be separate from its fixed physical clock')
    return True


def perform_released_robot_return(*,return_schedule,transcript_path,limits,allowed_initial_tracking_error_rad,
        wall_check,frame_sink=None,**kwargs):
    """Apply every exact reversed command, read it back, then validate home."""
    began=time.perf_counter()
    import numpy as np
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity
    from physics.provided_automatic_grasp import ProvidedAutomaticGrasp,_clock
    from physics.automatic_runtime_binding import physical_observation
    from physics.saved_motion_planning import iter_kinematic_return_samples
    scene=kwargs['scene'];arms=kwargs['arms'];arm_id=kwargs['arm_id'];arm=arms[arm_id]
    helper=kwargs['helper'];cup=kwargs['cup'];beads=kwargs['beads'];identities=kwargs['identities']
    before_observation=copy.deepcopy(kwargs['before_observation']);observe=kwargs['observe'];source_check=kwargs['source_check']
    reset.require(type(helper) is ProvidedAutomaticGrasp and helper.scene is scene and helper.holder is None
        and helper.cup is cup and type(arm) is RigidEntity and arm.base_link.is_fixed is True
        and arm.n_qs==arm.n_dofs==9,'Kinematic return requires the exact released fixed-base native donor')
    source=Path(__import__('inspect').getsourcefile(RigidEntity)).resolve()
    reset.require(hashlib.sha256(source.read_bytes()).hexdigest()==reset.ENTITY_SOURCE_SHA256,
        'Unreviewed native robot setter implementation')
    reset.require(return_schedule['arm_id']==arm_id and return_schedule['after_stage_id']==kwargs['after_stage_id']
        and return_schedule['target_command']==kwargs['target_command'] and source_check() is True,
        'Return schedule does not match this source-verified completed donor')
    reset.require(physical_observation(observe())==physical_observation(before_observation),
        'Kinematic return started from a stale physical observation')
    before=reset._snapshot(scene,arms,cup,beads,identities,helper,arm_id);clock=before['native_clock']
    reset.require(before['provided_snapshot']['inventory']['weld_pairs']==[]
        and kwargs['released_gate'].get('passed') is True
        and kwargs['release_event'].get('event')=='release'
        and kwargs['release_event'].get('arm_id')==arm_id
        and kwargs['release_event']['after']['inventory']==before['provided_snapshot']['inventory'],
        'Kinematic return lacks its actual native supported detach')
    target=kwargs['target_command']['q_rad']+kwargs['target_command']['finger_q_m']
    reset.require(target==kwargs['initial_observation']['arms'][arm_id]['q_all_dofs'],
        'Kinematic return must end at exact declared native INITIAL')
    first=next(iter(iter_kinematic_return_samples(return_schedule,limits=limits)))
    tracking=max(abs(a-b) for a,b in zip(before['arms'][arm_id]['q_all_dofs'][:7],first['q_rad']))
    reset.require(math.isfinite(tracking) and tracking<=allowed_initial_tracking_error_rad,
        'Actual released joints differ from the stored carry endpoint beyond original tracking tolerance')
    open_fingers=kwargs['target_command']['finger_q_m']
    reset.require(before['control_targets'][arm_id]['ctrl_pos'][7:]==open_fingers,
        'Physics-free return requires the already-open declared native finger command')
    path=Path(transcript_path);reset.require(not path.exists(),'Preserve existing native return transcript')
    frames=[];next_frame=1./15;count=0;transcript_hash=hashlib.sha256();samples_hash=hashlib.sha256()
    def record(stream,row):
        encoded=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
        stream.write(encoded);transcript_hash.update(encoded)
    def capture(display_s,role):
        native=reset._snapshot(scene,arms,cup,beads,identities,helper,arm_id)
        reset.require(native['native_clock']==clock and native['payload']==before['payload']
            and native['arms'][arm_id]['base_world']==before['arms'][arm_id]['base_world']
            and all(native['arms'][name]==before['arms'][name] for name in arms if name!=arm_id),
            'Kinematic display capture changed payload, peer, base or native clock')
        actual=observe()
        frame=dict(step=clock[0],simulation_s=clock[1],native_clock=copy.deepcopy(clock),display_s=display_s,
            arms={name:dict(q=copy.deepcopy(value['q_all_dofs']),hand_position_m=copy.deepcopy(value['hand_world']['position_m']))
                for name,value in actual['arms'].items()},body=copy.deepcopy(actual['body_world']),
            materials=[dict(id=m['id'],position_m=copy.deepcopy(m['position_world_m']),inside_carrier=m['inside_carrier']) for m in actual['materials']],
            phase='kinematic_return',active_arm=arm_id,action_role=role,playback_speed=1.)
        frames.append(frame)
        if frame_sink is not None:frame_sink(copy.deepcopy(frame))
        reset.require(_clock(scene)==clock,'A return frame sink advanced physical simulation')
    with path.open('xb') as stream:
        record(stream,dict(kind='initial_actual_released_state',sample_index=-1,display_s=0.,native_clock=clock,
            q_all_dofs=before['arms'][arm_id]['q_all_dofs'],qd_all_dofs=before['arms'][arm_id]['qd_all_dofs'],
            initial_tracking_error_rad=tracking))
        capture(0.,'actual_released_state')
        for sample in iter_kinematic_return_samples(return_schedule,limits=limits):
            wall_check()
            reset.require(_clock(scene)==clock,'Physical clock advanced before a kinematic sample')
            command=np.asarray(sample['q_rad']+open_fingers,dtype=np.float32)
            callback=kwargs.get('record_setter')
            if callback is not None:callback('attempt')
            arm.set_qpos(command,zero_velocity=True)
            if callback is not None:callback('executed')
            actual_q=reset._vector(arm.get_qpos(),9);actual_qd=reset._vector(arm.get_dofs_velocity(),9)
            reset.require(actual_q==command.tolist() and actual_qd==[0.]*9 and _clock(scene)==clock,
                'Native kinematic q9/qd0 or fixed clock differs from exact reversed command')
            samples_hash.update((json.dumps(sample,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8'))
            record(stream,dict(kind='reverse_command_sample',sample_index=count,**sample,native_clock=copy.deepcopy(clock),
                q_all_dofs=actual_q,qd_all_dofs=actual_qd))
            count+=1
            if sample['display_s']>=next_frame or sample['display_s']==return_schedule['display_duration_s']:
                capture(sample['display_s'],'reverse_carry' if sample['leg_index']==0 else 'reverse_approach')
                while next_frame<=sample['display_s']:next_frame+=1./15
            if count%128==0:
                stream.flush();reset.require(source_check() is True,'Sources changed during kinematic return')
        stream.flush()
    # Reuse the strict supported endpoint transaction with no extra pose setter.
    end_kwargs={**kwargs,'before_observation':observe(),'set_pose':False}
    finalized,endpoint=reset.perform_released_robot_reset(**end_kwargs)
    after_observation=copy.deepcopy(before_observation)
    after_observation['arms'][arm_id]=copy.deepcopy(finalized['arms'][arm_id])
    reset.require(physical_observation(after_observation)==physical_observation(observe()),
        'Kinematic return changed a physical observation outside its donor')
    receipt=dict(endpoint,schema_version=SCHEMA,model=MODEL,native_before=before,
        before_observation_sha256=digest(before_observation),after_observation_sha256=digest(after_observation),
        return_schedule=copy.deepcopy(return_schedule),return_schedule_sha256=digest(return_schedule),
        trajectory_order=[leg['source_stage_id'] for leg in return_schedule['legs']],
        display_duration_s=return_schedule['display_duration_s'],source_samples_sha256=samples_hash.hexdigest(),
        initial_actual_q9=before['arms'][arm_id]['q_all_dofs'],initial_tracking_error_rad=tracking,
        applied_sample_count=count,actual_q9_verified_samples=count,clock_checked_samples=count,pose_setter_count=count,
        physics_steps_during_return=0,swept_collision_certified=False,
        native_transcript=dict(path=str(path.resolve()),sha256=transcript_hash.hexdigest(),rows=count+1,includes_initial_actual_state=True),
        recorded_frames=frames,recorded_frames_sha256=digest(frames),recorded_frames_fps=15,
        scope='Actual native reverse carry then reverse approach commands. Display time advances, physical time does not. Payload and peers remain fixed. Endpoint-only contact proof.')
    receipt.pop('pose_setter_performed',None)
    receipt['endpoint_pose_setter_performed']=False
    validate_kinematic_return_receipt(receipt,before_observation=before_observation,after_observation=after_observation,
        expected_schedule=return_schedule,expected_command=kwargs['target_command'],expected_arm_id=arm_id,
        expected_stage_id=kwargs['after_stage_id'])
    receipt['return_host_wall_s']=time.perf_counter()-began
    receipt['return_host_wall_scope']='Helper including reverse samples, readbacks, transcript, frame capture, endpoint checks and validation. Runtime event publication is separate.'
    return after_observation,receipt
