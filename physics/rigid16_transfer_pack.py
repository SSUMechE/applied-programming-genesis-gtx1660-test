"""Phase-owned CUDA contact/pose/policy packets for pinned Genesis 1.3.2.

PRE is transferred and validated before integration. POST is a separate fresh
transfer before the existing internal sink. This never batches across internal
intervals. Provided grasp snapshots and outer observations remain fresh reads.
"""
from __future__ import annotations

import math
import time
import numpy as np

from physics.rigid_pose_pack import RigidPosePack
from physics.substep_observer import CONTACT_FIELDS, ObservationError, _validate_contacts, _detection_clock
from transport_challenge.contracts import ContractError


def _need(ok, message):
    if not ok:
        raise ContractError(message)


class Rigid16TransferPack:
    def __init__(self, pose_reader, *, verify_legacy=False):
        from physics.live_contact_coverage import _READBACKS, _BatchedReadback
        _need(type(pose_reader) is RigidPosePack and pose_reader.mode=='packed'
              and not pose_reader.verify_scalar and type(verify_legacy) is bool,
              'Consolidated readback requires the exact packed ten-pose reader')
        self.pose_reader=pose_reader;self.scene=pose_reader.scene;self.solver=pose_reader.solver
        self.torch,self.qd=pose_reader.torch,pose_reader.qd
        self.policy_reader=_READBACKS.get(self.scene)
        _need(type(self.policy_reader) is _BatchedReadback
              and all(view.device.type=='cuda' for view in self.policy_reader.views),
              'Consolidated readback requires the original CUDA batched policy views')
        import genesis as gs
        _need(gs.use_zerocopy and self.solver.collider._contact_data is not None,
              'Consolidated contacts require the pinned default CUDA zero-copy layout')
        self.verify_legacy=verify_legacy;self.failed=False
        self._pending=None;self._policy=None;self._last_ordinal=None
        self._failure_packet=None
        self.qd.sync()
        native_capacity=self.policy_reader.views[self.policy_reader.names.index('contact_capacity')].item()
        _need(type(native_capacity) is int and native_capacity>=0,'Invalid native contact capacity at packet setup')
        self.pre_contact_capacity=native_capacity
        policy_values=len(self.policy_reader.names)
        self.device_buffer_values={'pre':14*native_capacity+70+policy_values,'post':70+policy_values}
        self._device_buffers={phase:self.torch.empty(values,dtype=self.torch.float64,
            device=self.policy_reader.views[0].device) for phase,values in self.device_buffer_values.items()}
        self.setup_capacity_scalar_reads=1;self.out_cat_calls=0
        self.pre_packets=self.post_packets=self.policy_claims=0
        self.host_transfer_attempts=self.completed_host_transfers=self.completed_host_bytes=0
        self.public_contact_calls=self.source_modeled_count_scalar_reads=self.legacy_parity_checks=0
        self.wall_s={'pre':0.,'post':0.}

    def _clock(self, expected):
        _need(not self.failed and _detection_clock(self.solver)==expected,
              'Combined capture left its actual native phase clock')

    def _host_packet(self, pieces, *, phase):
        pieces=tuple(piece.reshape(-1) for piece in pieces)
        required=sum(piece.numel() for piece in pieces)
        buffer=self._device_buffers[phase]
        _need(required<=buffer.numel(),'Live combined packet exceeds its original native capacity')
        view=buffer[:required]
        self.out_cat_calls+=1
        values=self.torch.cat(pieces,out=view)
        _need(values.device.type=='cuda' and values.dtype==self.torch.float64,
              'Combined CUDA packet dtype/device changed')
        _need(values.numel()==required and values.data_ptr()==buffer.data_ptr(),
              'Combined CUDA packet resized or replaced its owned output buffer')
        self.host_transfer_attempts+=1
        host=np.asarray(values.detach().cpu().numpy()).copy()
        _need(host.ndim==1 and host.dtype==np.dtype('float64'),'Invalid owned combined host packet')
        self.completed_host_transfers+=1;self.completed_host_bytes+=host.nbytes
        return host

    def capture_pre(self, clock):
        started=time.perf_counter()
        try:
            self._clock(clock)
            _need(self._pending is None and self._policy is None
                  and (self._last_ordinal is None or clock[0]==self._last_ordinal+1),
                  'Missing, repeated or unconsumed combined internal packet')
            collider=self.solver.collider
            # device_capture synchronizes the native producer before the
            # public contact-count .item() and every borrowed CUDA field.
            pose_clock,poses=self.pose_reader.device_capture()
            cached=bool(collider._contact_data_cache.get((True,True)))
            self.public_contact_calls+=1
            contacts=collider.get_contacts(as_tensor=True,to_torch=True)
            if not cached:self.source_modeled_count_scalar_reads+=1
            _need(set(contacts)==set(CONTACT_FIELDS),'Pinned public contact fields changed')
            keys=tuple(contacts);n=int(contacts['geom_a'].shape[0])
            shapes={key:((n,) if key in CONTACT_FIELDS[:5] else (n,3)) for key in keys}
            device=self.policy_reader.views[0].device
            for key,value in contacts.items():
                _need(isinstance(value,self.torch.Tensor) and tuple(value.shape)==shapes[key]
                      and value.device==device
                      and (value.dtype==self.torch.int32 if key in CONTACT_FIELDS[:4]
                           else value.dtype in (self.torch.float32,self.torch.float64)),
                      'Pinned public CUDA contact tensor changed')
            policy=self.policy_reader.device_values(self.scene,synchronized=True)
            host=self._host_packet([*(contacts[key] for key in keys),poses,policy],phase='pre')
            offset=0;raw={}
            for key in keys:
                size=int(np.prod(shapes[key]));part=host[offset:offset+size].reshape(shapes[key]);offset+=size
                if key in CONTACT_FIELDS[:4]:
                    if not np.isfinite(part).all() or not np.equal(part,np.trunc(part)).all():
                        raise ObservationError('Invalid contact identity')
                    raw[key]=part.astype(np.int64).tolist()
                else:raw[key]=part.tolist()
            # Retain original PRE validation order before any integration.
            _validate_contacts(raw)
            state=self.pose_reader.decode_device_capture(pose_clock,host[offset:offset+70]);offset+=70
            header=dict(zip(self.policy_reader.names,host[offset:].tolist()))
            fields=[]
            for name in ('contact_count','contact_capacity','errno'):
                value=header[name]
                if not math.isfinite(value) or value!=int(value) or value<0:
                    raise ObservationError('Native integer required: '+name)
                fields.append(int(value))
            count,capacity,errno=fields
            if count!=n or count>capacity or errno!=0:
                raise ObservationError('Detection buffer/count/errno invalid')
            self._clock(clock)
            if self.verify_legacy:
                from physics.substep_observer import _copy_contacts, _capture_state, _field_integer
                _need(raw==_copy_contacts(collider) and state==_capture_state(self.pose_reader.read),
                      'PRE combined contact/pose differs from same-phase original getters')
                _need(fields==[_field_integer(collider._collider_state.n_contacts,'contact count'),
                    _field_integer(collider._collider_info.max_contacts,'contact capacity'),
                    _field_integer(self.solver._errno,'errno')], 'PRE combined native headers differ')
                self.legacy_parity_checks+=1
            self._pending={'clock':list(clock),'state':state}
            self.pre_packets+=1
            return raw,state,count,capacity,errno
        except BaseException:
            self.failed=True;self.pose_reader.failed=True
            if self._failure_packet is None:self._failure_packet={'phase':'pre','native_clock':list(clock),
                'host_packet_available':'host' in locals(),
                **({'host_packet_values':host.tolist(),'contact_fields':list(keys),
                    'contact_shapes':{key:list(value) for key,value in shapes.items()},
                    'pose_values':70,'policy_field_names':list(self.policy_reader.names)} if 'host' in locals() else {})}
            raise
        finally:self.wall_s['pre']+=time.perf_counter()-started

    def capture_post(self, clock):
        from physics.live_contact_coverage import _decode_native_pack
        started=time.perf_counter()
        try:
            self._clock(clock)
            _need(self._pending is not None and self._pending['clock']==clock and self._policy is None,
                  'POST combined capture lacks this exact PRE phase')
            pose_clock,poses=self.pose_reader.device_capture()
            policy=self.policy_reader.device_values(self.scene,synchronized=True)
            host=self._host_packet((poses,policy),phase='post')
            state=self.pose_reader.decode_device_capture(pose_clock,host[:70])
            native=_decode_native_pack(host[70:].tolist(),self.policy_reader.names)
            self._clock(clock)
            if self.verify_legacy:
                from physics.substep_observer import _capture_state
                _need(state==_capture_state(self.pose_reader.read)
                      and native==self.policy_reader.read(self.scene),
                      'POST combined pose/policy differs from same-phase original getters')
                self.legacy_parity_checks+=1
            self._policy={'clock':list(clock),'pre_state':self._pending['state'],
                          'post_state':state,'native':native}
            self._pending=None;self._last_ordinal=clock[0];self.post_packets+=1
            return state
        except BaseException:
            self.failed=True;self.pose_reader.failed=True
            if self._failure_packet is None:self._failure_packet={'phase':'post','native_clock':list(clock),
                'actual_pre_state':None if self._pending is None else self._pending['state'],
                'host_packet_available':'host' in locals(),
                **({'host_packet_values':host.tolist(),'pose_values':70,
                    'policy_field_names':list(self.policy_reader.names)} if 'host' in locals() else {})}
            raise
        finally:self.wall_s['post']+=time.perf_counter()-started

    def claim_policy(self, scene, capture):
        from physics.live_contact_coverage import _READBACKS
        try:
            packet=self._policy
            _need(scene is self.scene and _READBACKS.get(scene) is self.policy_reader
                  and packet is not None,'Missing, reused or foreign combined policy packet')
            self._clock(packet['clock'])
            _need(capture.get('global_substep_ordinal')==packet['clock'][0]
                  and capture.get('post_read_native_clock')==packet['clock']
                  and capture.get('post_integration_state') is packet['post_state']
                  and capture.get('detection',{}).get('rigid_state') is packet['pre_state'],
                  'Combined policy is not bound to the actual PRE/POST capture')
            self._policy=None;self.policy_claims+=1
            return dict(packet['native'])
        except BaseException:
            self.failed=True;self.pose_reader.failed=True
            if self._failure_packet is None:self._failure_packet={'phase':'policy_claim','actual_owned_packet':self._policy}
            raise

    def failure_data(self):
        from physics.run_joint_path import _json_safe
        return _json_safe(self._failure_packet)

    def metadata(self):
        return dict(schema_version='rigid16-phase-cuda-transfer-v1',mode='verify' if self.verify_legacy else 'consolidated',initialized=True,
            pre_packets=self.pre_packets,post_packets=self.post_packets,policy_claims=self.policy_claims,
            host_transfer_attempts=self.host_transfer_attempts,completed_host_transfers=self.completed_host_transfers,
            setup_capacity_scalar_reads=self.setup_capacity_scalar_reads,pre_contact_capacity=self.pre_contact_capacity,
            device_buffer_values=dict(self.device_buffer_values),device_buffer_allocations=2,out_cat_calls=self.out_cat_calls,
            completed_host_bytes=self.completed_host_bytes,public_contact_calls=self.public_contact_calls,
            source_modeled_count_scalar_reads=self.source_modeled_count_scalar_reads,
            legacy_parity_checks=self.legacy_parity_checks,measured_host_wall_s=dict(self.wall_s),failed=self.failed,
            pending_pre=self._pending is not None,pending_policy=self._policy is not None,
            failed_phase_packet=self.failure_data(),
            scope='Two phase-owned float64 payload transfers per internal interval plus the pinned public contact-count scalar read. PRE checks finish before integration. POST checks finish before the next interval. Public grasp snapshots, outer observations and controller reads remain separate. Counters describe explicit project API copies and source-modeled public count reads, not measured PCIe transactions or kernel timing.')
