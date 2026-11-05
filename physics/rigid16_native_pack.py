"""Read-only native CUDA collection and finite/identity/header predicates.

PRE is accepted before integration and POST before the next internal interval.
The native collector writes owned scratch only. No Genesis import at module load.
No future annotations: Quadrants decorators require actual native ABI types.
"""
import hashlib
import importlib.metadata
from pathlib import Path
import time
import numpy as np

from physics.rigid16_transfer_pack import Rigid16TransferPack, _need
from physics.substep_observer import CONTACT_FIELDS, ObservationError
from transport_challenge.contracts import ContractError

_KERNELS={}


def _collector(qd, arrays):
    key=(id(qd),id(arrays))
    if key not in _KERNELS:
        @qd.kernel
        def collect(ids:qd.types.ndarray(), output:qd.types.ndarray(),
                    ds:arrays.DynState, di:arrays.DynInfo,
                    cs:arrays.ColliderState, ci:arrays.ColliderInfo,
                    ri:arrays.RigidInfo, constraints:arrays.ConstraintState, errno:qd.Tensor,
                    capacity:qd.template(), eq_count:qd.template(),
                    permuted:qd.template(), pre:qd.template()):
            pose_offset=0
            policy_offset=70
            flag_offset=70+6+3*eq_count
            pose_flags=flag_offset
            if qd.static(pre):
                pose_offset=14*capacity
                policy_offset=pose_offset+70
                flag_offset=policy_offset+6+3*eq_count
                pose_flags=flag_offset+capacity
                for i in range(capacity):
                    for j in qd.static(range(14)):
                        output[14*i+j]=0.0
                    output[flag_offset+i]=0.0
                    output[pose_flags+11+i]=i
                    live=qd.max(0,qd.min(cs.n_contacts[0],capacity))
                    if i<live:
                        slot=i
                        if qd.static(permuted):slot=cs.contact_sort_idx[i,0]
                        output[pose_flags+11+i]=slot
                        bad=0
                        if slot<0 or slot>=capacity:
                            bad=1
                        else:
                            a=cs.contact_data.link_a[slot,0]
                            b=cs.contact_data.link_b[slot,0]
                            ga=cs.contact_data.geom_a[slot,0]
                            gb=cs.contact_data.geom_b[slot,0]
                            output[14*i]=qd.cast(a,qd.f64)
                            output[14*i+1]=qd.cast(b,qd.f64)
                            output[14*i+2]=qd.cast(ga,qd.f64)
                            output[14*i+3]=qd.cast(gb,qd.f64)
                            if a<0 or b<0 or ga<0 or gb<0:bad=1
                            penetration=cs.contact_data.penetration[slot,0]
                            output[14*i+4]=qd.cast(penetration,qd.f64)
                            if (qd.bit_cast(penetration,qd.i32)&2139095040)==2139095040:
                                if bad==0:bad=5
                            for j in qd.static(range(3)):
                                p=cs.contact_data.pos[slot,0][j]
                                n=cs.contact_data.normal[slot,0][j]
                                f=cs.contact_data.force[slot,0][j]
                                output[14*i+5+j]=qd.cast(p,qd.f64)
                                output[14*i+8+j]=qd.cast(n,qd.f64)
                                output[14*i+11+j]=qd.cast(f,qd.f64)
                                if (qd.bit_cast(p,qd.i32)&2139095040)==2139095040:
                                    if bad==0 or bad>2:bad=2
                                if (qd.bit_cast(n,qd.i32)&2139095040)==2139095040:
                                    if bad==0 or bad>3:bad=3
                                if (qd.bit_cast(f,qd.i32)&2139095040)==2139095040:
                                    if bad==0 or bad>4:bad=4
                        output[flag_offset+i]=bad
            for k in range(10):
                bad=0
                link=ids[k]
                for j in qd.static(range(3)):
                    value=ds.links.pos[link,0][j]
                    output[pose_offset+3*k+j]=qd.cast(value,qd.f64)
                    if (qd.bit_cast(value,qd.i32)&2139095040)==2139095040:bad=6
                for j in qd.static(range(4)):
                    value=ds.links.quat[link,0][j]
                    output[pose_offset+30+4*k+j]=qd.cast(value,qd.f64)
                    if (qd.bit_cast(value,qd.i32)&2139095040)==2139095040:bad=6
                output[pose_flags+k]=bad
            pruning=ci.contact_pruning_tolerance[None]
            count=cs.n_contacts[0]
            limit=ci.max_contacts[None]
            output[policy_offset]=qd.cast(pruning,qd.f64)
            output[policy_offset+1]=qd.cast(ri.n_equalities[None],qd.f64)
            output[policy_offset+2]=qd.cast(constraints.qd_n_equalities[0],qd.f64)
            output[policy_offset+3]=qd.cast(errno[0],qd.f64)
            for e in qd.static(range(eq_count)):
                output[policy_offset+4+3*e]=qd.cast(di.equalities.eq_type[e,0],qd.f64)
                output[policy_offset+5+3*e]=qd.cast(di.equalities.eq_obj1id[e,0],qd.f64)
                output[policy_offset+6+3*e]=qd.cast(di.equalities.eq_obj2id[e,0],qd.f64)
            output[policy_offset+4+3*eq_count]=qd.cast(count,qd.f64)
            output[policy_offset+5+3*eq_count]=qd.cast(limit,qd.f64)
            bad=0
            if qd.static(pre):
                if count<0 or count>limit or limit!=capacity or errno[0]!=0:bad=7
            else:
                if (qd.bit_cast(pruning,qd.i32)&2139095040)==2139095040:bad=8
                if bad==0 and (limit<0 or count<0 or count>limit):bad=9
            output[pose_flags+10]=bad
        _KERNELS[key]=collect
    return _KERNELS[key]


def packet_layout(capacity, policy_values, pre):
    pose=14*capacity if pre else 0
    policy=pose+70
    flags=policy+policy_values
    flag_count=capacity+11 if pre else 11
    return dict(pose=pose,policy=policy,flags=flags,flag_count=flag_count,
                slots=flags+flag_count,total=flags+flag_count+(capacity if pre else 0))


def reference_flags(host, capacity, policy_values, pre):
    """CPU reference for every GPU flag, used by native-verify and CPU tests."""
    layout=packet_layout(capacity,policy_values,pre)
    policy=host[layout['policy']:layout['flags']]
    flags=np.zeros(layout['flag_count'],dtype=np.int64)
    if pre:
        rows=host[:14*capacity].reshape(capacity,14)
        slots=host[layout['slots']:]
        live=max(0,min(int(policy[-2]),capacity))
        for i in range(live):
            row=rows[i]
            if not 0<=slots[i]<capacity or np.any(row[:4]<0):flags[i]=1
            elif not np.isfinite(row[5:8]).all():flags[i]=2
            elif not np.isfinite(row[8:11]).all():flags[i]=3
            elif not np.isfinite(row[11:14]).all():flags[i]=4
            elif not np.isfinite(row[4]):flags[i]=5
        pose_flags=capacity
        if policy[-2]<0 or policy[-2]>policy[-1] or policy[-1]!=capacity or policy[3]!=0:
            flags[-1]=7
    else:
        pose_flags=0
        if not np.isfinite(policy[0]):flags[-1]=8
        elif policy[-1]<0 or policy[-2]<0 or policy[-2]>policy[-1]:flags[-1]=9
    poses=host[layout['pose']:layout['policy']]
    for k in range(10):
        if not np.isfinite(poses[3*k:3*k+3]).all() or not np.isfinite(poses[30+4*k:34+4*k]).all():
            flags[pose_flags+k]=6
    return flags


def first_flag(host, layout):
    flags=host[layout['flags']:layout['flags']+layout['flag_count']]
    _need(np.isfinite(flags).all() and np.equal(flags,np.trunc(flags)).all()
          and np.all((flags>=0)&(flags<=9)),'Malformed native CUDA predicate flags')
    bad=flags[flags>0]
    return int(bad.min()) if bad.size else 0


def raise_flag(code):
    if code in (1,2,3,4,5):
        label={1:'identity or physical index',2:'position',3:'normal',4:'force',5:'penetration'}[code]
        raise ObservationError('GPU detected invalid contact '+label)
    if code==6:raise ContractError('GPU detected nonfinite native pose')
    if code==7:raise ObservationError('Detection buffer/count/errno invalid')
    if code in (8,9):raise ContractError('GPU detected invalid native policy value')


class NativeRigid16TransferPack(Rigid16TransferPack):
    def __init__(self, pose_reader, *, verify_legacy=False):
        super().__init__(pose_reader,verify_legacy=verify_legacy)
        from physics.gravity_jacobian_batch import (_load_dependencies,_verify_sources,
            GENESIS_SOURCE_PINS,QUADRANTS_SOURCE_PINS,EXPECTED_VERSIONS)
        from physics.substep_observer import verify_installed_sources
        torch,qd,gs,arrays,misc=_load_dependencies()
        root,_=verify_installed_sources()
        _verify_sources(root,GENESIS_SOURCE_PINS,'native collector')
        _verify_sources(Path(qd.__file__).resolve().parent,QUADRANTS_SOURCE_PINS,'native collector')
        _need({name:importlib.metadata.version(name) for name in EXPECTED_VERSIONS}==EXPECTED_VERSIONS
              and gs.tc_float==torch.float32 and gs.qd_float==qd.f32,
              'Native collector requires the pinned float32 CUDA ABI')
        self._arrays=arrays;self._kernel=_collector(qd,arrays)
        self.eq_count=len(self.solver.equalities)
        expected=['pruning','static_equalities','active_equalities','errno']
        expected.extend('equality_'+str(e)+'_'+key for e in range(self.eq_count) for key in ('eq_type','eq_obj1id','eq_obj2id'))
        expected.extend(('contact_count','contact_capacity'))
        _need(self.policy_reader.names==expected,'Native policy row order differs from the pinned decoder')
        collider=self.solver.collider
        self.permuted=bool(collider._collider_static_config.has_prunable_contacts
                           or collider._collider_static_config.spatial_sort_supported)
        self._validate_native_views(misc)
        self._field_signature=self._fields()
        self._ids=torch.tensor(pose_reader.pose_ids,dtype=torch.int32,device=gs.device)
        self.layouts={phase:packet_layout(self.pre_contact_capacity,len(expected),phase=='pre') for phase in ('pre','post')}
        self.device_buffer_values={phase:layout['total'] for phase,layout in self.layouts.items()}
        self._device_buffers={phase:torch.empty(size,dtype=torch.float64,device=gs.device)
                              for phase,size in self.device_buffer_values.items()}
        self.collector_invocations=self.gpu_predicate_checks=self.native_pose_captures=0

    def _fields(self):
        from physics.live_contact_coverage import _native_buffer
        r=self.solver;c=r.collider
        fields=[r.dyn_state.links.pos,r.dyn_state.links.quat,c._collider_state.n_contacts,
                c._collider_state.contact_sort_idx,c._collider_info.max_contacts,
                c._collider_info.contact_pruning_tolerance,r.rigid_info.n_equalities,
                r.constraint_solver.constraint_state.qd_n_equalities,r._errno]
        fields.extend(getattr(c._collider_state.contact_data,key) for key in
                      ('link_a','link_b','geom_a','geom_b','penetration','pos','normal','force'))
        fields.extend(getattr(r.dyn_info.equalities,key) for key in ('eq_type','eq_obj1id','eq_obj2id'))
        return tuple((id(_native_buffer(field)),tuple(field.shape),str(getattr(field,'dtype',None))) for field in fields)

    def _validate_native_views(self,misc):
        r=self.solver;c=r.collider;cap=self.pre_contact_capacity;t=self.torch
        specs=[('link_positions',r.dyn_state.links.pos,(r.n_links,1,3),t.float32),
               ('link_quaternions',r.dyn_state.links.quat,(r.n_links,1,4),t.float32),
               ('contact_count',c._collider_state.n_contacts,(1,),t.int32),
               ('contact_permutation',c._collider_state.contact_sort_idx,(cap,1),t.int32),
               ('contact_capacity',c._collider_info.max_contacts,(),t.int32),
               ('contact_pruning',c._collider_info.contact_pruning_tolerance,(),t.float32),
               ('static_equalities',r.rigid_info.n_equalities,(),t.int32),
               ('active_equalities',r.constraint_solver.constraint_state.qd_n_equalities,(1,),t.int32),
               ('errno',r._errno,(1,),t.int32)]
        for key in ('link_a','link_b','geom_a','geom_b','penetration','pos','normal','force'):
            specs.append((key,getattr(c._collider_state.contact_data,key),
                (cap,1,3) if key in ('pos','normal','force') else (cap,1),
                t.int32 if key in ('link_a','link_b','geom_a','geom_b') else t.float32))
        self._native_views=[];self.native_field_layout={}
        for name,field,shape,dtype in specs:
            view=misc.qd_to_torch(field,copy=False)
            _need(tuple(view.shape)==shape and view.dtype==dtype and view.device==self.policy_reader.views[0].device,
                  'Native collector field dtype/shape/device differs: '+name)
            self._native_views.append(view);self.native_field_layout[name]={'shape':list(shape),'dtype':str(dtype)}
        eq_shapes=[]
        for key in ('eq_type','eq_obj1id','eq_obj2id'):
            view=misc.qd_to_torch(getattr(r.dyn_info.equalities,key),copy=False)
            _need(view.ndim==2 and view.shape[1]==1 and view.shape[0]>=self.eq_count and view.dtype==t.int32
                  and view.device==self.policy_reader.views[0].device,'Native equality layout differs: '+key)
            eq_shapes.append(tuple(view.shape));self._native_views.append(view)
            self.native_field_layout[key]={'shape':list(view.shape),'dtype':str(view.dtype)}
        _need(len(set(eq_shapes))==1,'Native equality column capacities differ')

    def _collect(self, clock, pre):
        self._clock(clock)
        _need(self._fields()==self._field_signature and self.pose_reader._topology()==self.pose_reader._signature,
              'Native collector field allocation or pose topology changed')
        pose_clock=self.pose_reader._clock()
        self.pose_reader.capture_calls+=1
        phase='pre' if pre else 'post';output=self._device_buffers[phase];r=self.solver;c=r.collider
        self.collector_invocations+=1
        self._kernel(self._ids,output,r.dyn_state,r.dyn_info,c._collider_state,c._collider_info,
            r.rigid_info,r.constraint_solver.constraint_state,r._errno,
            self.pre_contact_capacity,self.eq_count,self.permuted,pre)
        self.qd.sync()
        self.host_transfer_attempts+=1
        host=np.asarray(output.detach().cpu().numpy()).copy()
        self.completed_host_transfers+=1;self.completed_host_bytes+=host.nbytes
        try:
            layout=self.layouts[phase]
            _need(host.dtype==np.dtype('float64') and host.shape==(layout['total'],),'Native collector packet layout changed')
            self._clock(clock)
            _need(self.pose_reader._clock()==pose_clock,'Native collector changed actual pose clock')
            if self.verify_legacy:
                actual=host[layout['flags']:layout['flags']+layout['flag_count']]
                _need(np.array_equal(actual,reference_flags(host,self.pre_contact_capacity,len(self.policy_reader.names),pre)),
                      'GPU predicate flags differ from the complete CPU reference')
            self.gpu_predicate_checks+=layout['flag_count']
            return host,pose_clock
        except BaseException:self._failed(phase,clock,host);raise

    def _state(self,host,phase,pose_clock):
        layout=self.layouts[phase];values=host[layout['pose']:layout['policy']]
        state=self.pose_reader._record(values[:30].reshape(10,3),values[30:].reshape(10,4))
        self.pose_reader.completed_captures+=1;self.pose_reader.last_native_clock=pose_clock
        self.native_pose_captures+=1
        return state

    def _failed(self,phase,clock,host):
        self.failed=True;self.pose_reader.failed=True
        if self._failure_packet is None:
            self._failure_packet=dict(phase=phase,native_clock=list(clock),layout=self.layouts[phase],
                capacity=self.pre_contact_capacity,policy_field_names=list(self.policy_reader.names),
                host_packet_available=host is not None,
                host_packet_values=None if host is None else host.tolist(),
                scope='Actual owned native collector scratch. Inactive or invalid physical slots are zero-filled and never decoded as accepted contacts.')

    def capture_pre(self,clock):
        started=time.perf_counter();host=None
        try:
            _need(self._pending is None and self._policy is None
                  and (self._last_ordinal is None or clock[0]==self._last_ordinal+1),
                  'Missing, repeated or unconsumed native internal packet')
            host,pose_clock=self._collect(clock,True)
            layout=self.layouts['pre'];raise_flag(first_flag(host,layout))
            policy=host[layout['policy']:layout['flags']]
            count=int(policy[-2]);capacity=int(policy[-1]);errno=int(policy[3])
            rows=host[:14*self.pre_contact_capacity].reshape(self.pre_contact_capacity,14)[:count]
            raw={key:(rows[:,i].astype(np.int64).tolist() if i<4 else rows[:,4].tolist() if i==4
                 else rows[:,5+3*(i-5):8+3*(i-5)].tolist()) for i,key in enumerate(CONTACT_FIELDS)}
            state=self._state(host,'pre',pose_clock)
            if self.verify_legacy:
                from physics.substep_observer import _copy_contacts,_capture_state,_field_integer
                c=self.solver.collider
                _need(raw==_copy_contacts(c) and state==_capture_state(self.pose_reader.read)
                      and [count,capacity,errno]==[_field_integer(c._collider_state.n_contacts,'count'),
                          _field_integer(c._collider_info.max_contacts,'capacity'),_field_integer(self.solver._errno,'errno')],
                      'Native PRE contacts/poses/headers differ from original getters')
                self.legacy_parity_checks+=1
            self._pending={'clock':list(clock),'state':state};self.pre_packets+=1
            return raw,state,count,capacity,errno
        except BaseException:self._failed('pre',clock,host);raise
        finally:self.wall_s['pre']+=time.perf_counter()-started

    def capture_post(self,clock):
        from physics.live_contact_coverage import _decode_native_pack
        started=time.perf_counter();host=None
        try:
            _need(self._pending is not None and self._pending['clock']==clock and self._policy is None,
                  'Native POST capture lacks its exact PRE phase')
            host,pose_clock=self._collect(clock,False)
            layout=self.layouts['post'];raise_flag(first_flag(host,layout))
            state=self._state(host,'post',pose_clock)
            native=_decode_native_pack(host[layout['policy']:layout['flags']].tolist(),self.policy_reader.names)
            if self.verify_legacy:
                from physics.substep_observer import _capture_state
                _need(state==_capture_state(self.pose_reader.read) and native==self.policy_reader.read(self.scene),
                      'Native POST poses/policy differ from original getters')
                self.legacy_parity_checks+=1
            self._policy={'clock':list(clock),'pre_state':self._pending['state'],'post_state':state,'native':native}
            self._pending=None;self._last_ordinal=clock[0];self.post_packets+=1
            return state
        except BaseException:self._failed('post',clock,host);raise
        finally:self.wall_s['post']+=time.perf_counter()-started

    def metadata(self):
        result=super().metadata()
        result.update(schema_version='rigid16-native-cuda-collector-v1',mode='native-verify' if self.verify_legacy else 'native',
            collector_invocations=self.collector_invocations,gpu_predicate_checks=self.gpu_predicate_checks,
            native_pose_captures=self.native_pose_captures,contact_permutation_applied=self.permuted,layouts=self.layouts,
            native_field_layout=self.native_field_layout,device_buffer_allocations=4,active_device_buffers=2,
            scope='Read-only native CUDA collector calls, not actual GPU kernel/offload counts. gpu_predicate_checks counts output flag slots, including inactive zero slots. Full-capacity PRE and fixed POST scratch are copied once each. GPU flags preserve PRE versus POST stopping; CPU selects first predicate-priority code. Classifier, handling and grasp checks remain unchanged. Legacy parity adds diagnostic reads.')
        return result
