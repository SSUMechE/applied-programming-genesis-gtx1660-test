"""Opt-in device contact audit for one fixed path. No native imports at load.

Native physics is never written here. A serial device witness loop retains the
first bad sample deterministically. Continuing after a sticky failure is an
explicit deferred audit, not the original immediate CPU stop or a full log.
"""
import copy
import hashlib
from pathlib import Path
import numpy as np

from physics.episode_contacts import _validate_static, _validate_dynamic, _PHASES, _NO_HOLDER_PHASES
from physics.handle_contacts import validate_contact_snapshot
from physics.gravity_feedforward import rotation_matrix_wxyz
from transport_challenge.contracts import ContractError, digest

CATEGORIES=('UNALLOWED_CONTACT','named_tray_bottom_support_top','exact_finger_pad_handle',
            'contents_contact','named_support_floor')
KINDS={'arm':0,'tray':1,'bead':2,'support':3,'floor':4,'obstacle':5}
_KERNELS={}


def require(ok,message):
    if not ok:raise ContractError(message)


def compile_policy(*,owners,components_by_geom,pads_by_arm,stations_by_geom,canonical_geometry=None,
                   single_active_contract=None):
    """Validate original registries and detach one dense immutable GPU layout."""
    o,c,p,s,arms=_validate_static(*copy.deepcopy((owners,components_by_geom,pads_by_arm,stations_by_geom)),
                                single_active_contract=single_active_contract)
    arms=tuple(sorted(arms));n=max(o)+1
    ids=np.full((n,6),-1,np.int32);values=np.zeros((n,12),np.float64)
    body_links={v['link_id'] for v in o.values() if v['kind']=='tray'}
    require(len(body_links)==1,'Device policy requires the original one-link nine-box vessel')
    pose_ids=[next(iter(body_links))]
    for arm in arms:
        hand={v['link_id'] for v in o.values() if v['kind']=='arm' and v['entity']==arm and v['link']=='hand'}
        require(len(hand)==1,'Every registered Panda needs its actual hand link')
        pose_ids.append(next(iter(hand)))
    for arm in arms:
        for finger in ('left_finger','right_finger'):
            links={v['link_id'] for v in p[arm].values() if v['link']==finger}
            require(len(links)==1,'Each matched finger must retain exactly one native link')
            pose_ids.append(next(iter(links)))
    for g,v in o.items():
        ids[g,0]=KINDS[v['kind']];ids[g,1]=v['link_id']
        if v['kind']=='arm':ids[g,2]=arms.index(v['entity'])
        if g in c:
            ids[g,3]=0 if c[g]['component_id']=='solid_0' else 5 if c[g]['component_id']=='solid_5' else 9
            values[g,:6]=[*c[g]['center_body_m'],*c[g]['size_m']]
        if g in s:ids[g,5]=1;values[g,6:]=[*s[g]['center_world_m'],*s[g]['size_m']]
    for ai,arm in enumerate(arms):
        for g,pad in p[arm].items():
            ids[g,4]=1+len(arms)+2*ai+(pad['link']=='right_finger')
            values[g,:6]=[*pad['center_link_m'],*pad['size_m']]
    if canonical_geometry is not None:
        from physics.particle_episode_contacts import _axis_aligned
        boxes=np.asarray(canonical_geometry['component_boxes'],float)
        by_name={v['component_id']:v for v in c.values()}
        require(boxes.shape==(9,2,3) and np.isfinite(boxes).all() and np.all(boxes[:,0]>0)
            and len(by_name)==9 and set(by_name)=={'solid_'+str(i) for i in range(9)},'Explicit canonical/native nine-box geometry required')
        size,center=boxes[0];floor=by_name['solid_0']
        require(_axis_aligned(floor) and np.max(np.abs(np.asarray(floor['size_m'])-size))<=1e-12
            and np.max(np.abs(np.asarray(floor['center_body_m'])-center))<=1e-12,'Canonical/native floor representation differs')
        bottom=center[2]-size[2]/2
        for g,component in c.items():
            if (component['component_id'] in {'solid_1','solid_2','solid_3','solid_4'} and _axis_aligned(component)
                and abs(component['center_body_m'][2]-component['size_m'][2]/2-bottom)<=1e-12):
                ids[g,3]=1;values[g,6:]=[*center,*size]
    geometry_hash=None if canonical_geometry is None else digest(canonical_geometry)
    binding=dict(arm_ids=list(arms),integer_policy=ids.tolist(),geometry_policy=values.tolist(),pose_link_ids=pose_ids,canonical_geometry_sha256=geometry_hash)
    if single_active_contract is not None:binding['single_active_contract_sha256']=single_active_contract['contract_sha256']
    signature=digest(binding)
    for a in (ids,values):a.setflags(write=False)
    return dict(arm_ids=arms,integer_policy=ids,geometry_policy=values,
        pose_link_ids=tuple(pose_ids),sha256=signature,
        canonical_geometry=copy.deepcopy(canonical_geometry),canonical_geometry_sha256=geometry_hash,
        registries=(o,c,p,s),single_active_contract=copy.deepcopy(single_active_contract))


def classify_cpu(policy,raw,*,body_pose,finger_poses_by_arm,phase,active_holder_arm_id=None):
    """Independent scalar port for CPU parity tests, never native authority."""
    require(phase in _PHASES,'Unknown original contact phase')
    arms=policy['arm_ids'];_validate_dynamic(body_pose,finger_poses_by_arm,set(arms))
    require(active_holder_arm_id is None or active_holder_arm_id in arms,'Unknown holder')
    require(active_holder_arm_id is None or phase not in _NO_HOLDER_PHASES,'Holder forbidden in this phase')
    holder=-1 if active_holder_arm_id is None else arms.index(active_holder_arm_id)
    x=validate_contact_snapshot(raw);ip=policy['integer_policy'];fp=policy['geometry_policy']
    bodyR=rotation_matrix_wxyz(body_pose['quaternion_wxyz']);bodyP=np.asarray(body_pose['position_m'])
    categories=[];bilateral=0
    for row,(a,b,position,normal,depth) in enumerate(zip(x['geom_a'],x['geom_b'],x['position'],x['normal'],x['penetration'])):
        a,b=int(a),int(b);require(0<=a<len(ip) and 0<=b<len(ip) and ip[a,0]>=0 and ip[b,0]>=0,'Unknown native owner')
        if 'link_a'in raw:
            require(int(raw['link_a'][row])==ip[a,1] and int(raw['link_b'][row])==ip[b,1],'Native link differs from geometry owner')
        err=abs(float(np.linalg.norm(normal))-1.);valid=0<=depth<=.001 and err<=.001
        tolerance=.000020+max(0.,float(depth));category=0
        for tray,other in ((a,b),(b,a)):
            if ip[tray,0]!=1:continue
            local=bodyR.T@(position-bodyP);bn=bodyR.T@normal
            center,size=fp[tray,:3],fp[tray,3:6]
            inside=bool(np.all(np.abs(local-center)<=size/2+.000020))
            if ip[other,5]==1:
                support=position-fp[other,6:9];ss=fp[other,9:12]
                bottom=center[2]-size[2]/2;floor_inside=True
                if ip[tray,3]==1:
                    bottom=fp[tray,8]-fp[tray,11]/2
                    floor_inside=bool(np.all(np.abs(local[:2]-fp[tray,6:8])<=fp[tray,9:11]/2+.000020))
                if (ip[tray,3] in (0,1) and valid and inside and floor_inside and np.all(np.abs(support)<=ss/2+.000020)
                    and abs(local[2]-bottom)<=tolerance
                    and abs(support[2]-ss[2]/2)<=tolerance and abs(bn[2])>=.95 and abs(normal[2])>=.95):category=1
            elif ip[other,4]>=0 and ip[other,2]==holder:
                finger='left_finger' if (ip[other,4]-(1+len(arms)))%2==0 else 'right_finger'
                pose=finger_poses_by_arm[arms[holder]][finger];R=rotation_matrix_wxyz(pose['quaternion_wxyz'])
                fl=R.T@(position-np.asarray(pose['position_m']));fn=R.T@normal
                pc,ps=fp[other,:3],fp[other,3:6]
                if (phase not in _NO_HOLDER_PHASES and ip[tray,3]==5 and valid
                    and abs(abs(local[1]-center[1])-size[1]/2)<=tolerance
                    and abs(fl[1]-(pc[1]-ps[1]/2))<=tolerance
                    and abs(bn[1])>=.95 and abs(fn[1])>=.95 and inside
                    and np.all(np.abs(fl-pc)<=ps/2+.000020)):
                    category=2;bilateral|=1 if finger=='left_finger' else 2
            break
        if category==0:
            kinds={int(ip[a,0]),int(ip[b,0])}
            if kinds in ({2},{1,2}) and -.000001<=depth<=.001 and 0<=err<=.001:category=3
            elif valid and kinds=={3,4}:category=4
        categories.append(CATEGORIES[category])
    return categories,bilateral==3


def _kernels(qd,arrays):
    key=(id(qd),id(arrays))
    if key in _KERNELS:return _KERNELS[key]

    @qd.func
    def finite(x:qd.f64):
        return (qd.bit_cast(x,qd.i64)&qd.i64(0x7ff0000000000000))!=qd.i64(0x7ff0000000000000)

    @qd.func
    def inverse_rotate(q:qd.types.vector(4,qd.f64),v:qd.types.vector(3,qd.f64)):
        norm=qd.sqrt(q.dot(q));w=q[0]/norm;x=q[1]/norm;y=q[2]/norm;z=q[3]/norm
        return qd.Vector([(1-2*(y*y+z*z))*v[0]+2*(x*y+w*z)*v[1]+2*(x*z-w*y)*v[2],
            2*(x*y-w*z)*v[0]+(1-2*(x*x+z*z))*v[1]+2*(y*z+w*x)*v[2],
            2*(x*z+w*y)*v[0]+2*(y*z-w*x)*v[1]+(1-2*(x*x+y*y))*v[2]])

    @qd.kernel
    def initialize(state:qd.types.ndarray(),witness:qd.types.ndarray(),cats:qd.types.ndarray(),capacity:qd.template()):
        for i in range(32):state[i]=0;witness[i]=qd.f64(0.0)
        state[1]=-1;state[2]=-1;state[4]=-1;state[5]=-1;state[6]=-1
        state[15]=-1;state[16]=-1
        for i in range(capacity):cats[i]=-1

    @qd.kernel
    def audit(ip:qd.types.ndarray(),fp:qd.types.ndarray(),links:qd.types.ndarray(),config:qd.types.ndarray(),
        expected_eq:qd.types.ndarray(),state:qd.types.ndarray(),witness:qd.types.ndarray(),cats:qd.types.ndarray(),
        ds:arrays.DynState,di:arrays.DynInfo,cs:arrays.ColliderState,ci:arrays.ColliderInfo,
        ri:arrays.RigidInfo,constraints:arrays.ConstraintState,errno:qd.Tensor,
        ordinal:qd.i32,pre:qd.template(),capacity:qd.template(),eq_capacity:qd.template(),
        n_geom:qd.template(),n_pose:qd.template(),permuted:qd.template(),stride:qd.template()):
        # One device thread owns each whole witness batch. Native queues order
        # calls, so first-bad selection needs no racing multi-field atomics.
        for singleton in range(1):
            bad=0;badrow=-1;bada=-1;badb=-1
            badp=qd.Vector.zero(qd.f64,3);badn=qd.Vector.zero(qd.f64,3);badd=qd.f64(0.0)
            count=cs.n_contacts[0];limit=ci.max_contacts[None]
            if count<0 or count>limit or limit!=capacity:bad=1
            if bad==0 and errno[0]!=0:bad=2
            if bad==0 and (not finite(qd.cast(ci.contact_pruning_tolerance[None],qd.f64)) or ci.contact_pruning_tolerance[None]!=0):bad=3
            if bad==0 and (constraints.qd_n_equalities[0]!=config[2] or ri.n_candidate_equalities[None]!=eq_capacity or ri.n_equalities[None]!=config[3]):bad=4
            for e in range(eq_capacity):
                if e<config[2] and (di.equalities.eq_type[e,0]!=expected_eq[e,0]
                    or di.equalities.eq_obj1id[e,0]!=expected_eq[e,1] or di.equalities.eq_obj2id[e,0]!=expected_eq[e,2]):
                    if bad==0:bad=5
            positions=qd.Matrix.zero(qd.f64,n_pose,3);quats=qd.Matrix.zero(qd.f64,n_pose,4)
            for i in range(n_pose):
                for j in qd.static(range(3)):positions[i,j]=qd.cast(ds.links.pos[links[i],0][j],qd.f64)
                for j in qd.static(range(4)):quats[i,j]=qd.cast(ds.links.quat[links[i],0][j],qd.f64)
                norm=qd.f64(0.0);valid=1
                for j in qd.static(range(3)):
                    if not finite(positions[i,j]):valid=0
                for j in qd.static(range(4)):
                    if not finite(quats[i,j]):valid=0
                    norm+=quats[i,j]*quats[i,j]
                if valid==0 or not finite(norm) or qd.abs(qd.sqrt(norm)-qd.f64(1.0))>qd.f64(.0001):
                    if bad==0:bad=6
            if qd.static(pre):
                if state[15]>=0 and ordinal!=state[15]+stride:
                    if bad==0:bad=10
                if stride==100 and ordinal%100!=99:
                    if bad==0:bad=10
                if stride==1 and state[15]>=0 and state[16]!=state[15]:
                    if bad==0:bad=10
                state[7]+=1;state[15]=ordinal
                for i in range(capacity):cats[i]=-1
                if count>=0 and count<=capacity and limit==capacity and bad==0:
                    for i in range(count):
                        slot=i
                        if qd.static(permuted):slot=cs.contact_sort_idx[i,0]
                        code=0;category=0;a=-1;b=-1
                        point=qd.Vector.zero(qd.f64,3);normal=qd.Vector.zero(qd.f64,3);depth=qd.f64(0.)
                        if slot<0 or slot>=capacity:code=7
                        else:
                            a=cs.contact_data.geom_a[slot,0];b=cs.contact_data.geom_b[slot,0]
                            if a<0 or b<0 or a>=n_geom or b>=n_geom or a==b:code=7
                            else:
                                if ip[a,0]<0 or ip[b,0]<0 or cs.contact_data.link_a[slot,0]!=ip[a,1] or cs.contact_data.link_b[slot,0]!=ip[b,1]:code=7
                            depth=qd.cast(cs.contact_data.penetration[slot,0],qd.f64)
                            if not finite(depth) and code==0:code=8
                            for j in qd.static(range(3)):
                                point[j]=qd.cast(cs.contact_data.pos[slot,0][j],qd.f64)
                                normal[j]=qd.cast(cs.contact_data.normal[slot,0][j],qd.f64)
                                if (not finite(point[j]) or not finite(normal[j]) or not finite(qd.cast(cs.contact_data.force[slot,0][j],qd.f64))) and code==0:code=8
                        if code==0:
                            error=qd.abs(qd.sqrt(normal.dot(normal))-qd.f64(1.0))
                            valid=depth>=qd.f64(0.0) and depth<=qd.f64(.001) and error<=qd.f64(.001)
                            tol=qd.f64(.000020)+qd.max(qd.f64(0.0),depth)
                            body_q=qd.Vector([quats[0,0],quats[0,1],quats[0,2],quats[0,3]])
                            origin=qd.Vector([positions[0,0],positions[0,1],positions[0,2]])
                            body_local=inverse_rotate(body_q,point-origin);body_normal=inverse_rotate(body_q,normal)
                            tray=-1;other=-1
                            if ip[a,0]==1:tray=a;other=b
                            elif ip[b,0]==1:tray=b;other=a
                            if tray>=0:
                                inside=1
                                for j in qd.static(range(3)):
                                    if qd.abs(body_local[j]-fp[tray,j])>fp[tray,3+j]/2+qd.f64(.000020):inside=0
                                if ip[other,5]==1:
                                    support=point-qd.Vector([fp[other,6],fp[other,7],fp[other,8]])
                                    support_inside=1
                                    for j in qd.static(range(3)):
                                        if qd.abs(support[j])>fp[other,9+j]/2+qd.f64(.000020):support_inside=0
                                    bottom=fp[tray,2]-fp[tray,5]/2;floor_inside=1
                                    if ip[tray,3]==1:
                                        bottom=fp[tray,8]-fp[tray,11]/2
                                        for j in qd.static(range(2)):
                                            if qd.abs(body_local[j]-fp[tray,6+j])>fp[tray,9+j]/2+qd.f64(.000020):floor_inside=0
                                    if ((ip[tray,3]==0 or ip[tray,3]==1) and valid and inside==1 and floor_inside==1 and support_inside==1
                                        and qd.abs(body_local[2]-bottom)<=tol
                                        and qd.abs(support[2]-fp[other,11]/2)<=tol
                                        and qd.abs(body_normal[2])>=qd.f64(.95) and qd.abs(normal[2])>=qd.f64(.95)):category=1
                                elif ip[other,4]>=0 and ip[other,2]==config[1]:
                                    f=ip[other,4];fq=qd.Vector([quats[f,0],quats[f,1],quats[f,2],quats[f,3]])
                                    fo=qd.Vector([positions[f,0],positions[f,1],positions[f,2]])
                                    fl=inverse_rotate(fq,point-fo);fn=inverse_rotate(fq,normal);finger_inside=1
                                    for j in qd.static(range(3)):
                                        if qd.abs(fl[j]-fp[other,j])>fp[other,3+j]/2+qd.f64(.000020):finger_inside=0
                                    if (config[0]==1 and ip[tray,3]==5 and valid and inside==1 and finger_inside==1
                                        and qd.abs(qd.abs(body_local[1]-fp[tray,1])-fp[tray,4]/2)<=tol
                                        and qd.abs(fl[1]-(fp[other,1]-fp[other,4]/2))<=tol
                                        and qd.abs(body_normal[1])>=qd.f64(.95) and qd.abs(fn[1])>=qd.f64(.95)):category=2
                            if category==0:
                                ka=ip[a,0];kb=ip[b,0]
                                contents=(ka==2 and kb==2) or (ka==2 and kb==1) or (ka==1 and kb==2)
                                if contents and depth>=-qd.f64(.000001) and depth<=qd.f64(.001) and error>=0 and error<=qd.f64(.001):category=3
                                elif valid and ((ka==3 and kb==4) or (ka==4 and kb==3)):category=4
                            if category==0:code=9
                        cats[i]=category
                        state[9]+=1;state[10+category]+=1
                        if code!=0 and bad==0:
                            bad=code;badrow=i;bada=a;badb=b;badp=point;badn=normal;badd=depth
            else:
                if state[15]!=ordinal or state[16]==ordinal:
                    if bad==0:bad=10
                state[8]+=1;state[16]=ordinal
            state[17]=count;state[18]=limit
            if bad!=0:
                state[21]+=1
                if state[0]==0:
                    state[0]=1;state[1]=ordinal;state[2]=0 if qd.static(pre) else 1
                    state[3]=bad;state[4]=badrow;state[5]=bada;state[6]=badb
                    for j in qd.static(range(3)):witness[j]=badp[j];witness[3+j]=badn[j]
                    witness[6]=badd
    _KERNELS[key]=(initialize,audit)
    return _KERNELS[key]


class DeviceContactAudit:
    def __init__(self,scene,coverage,*,owners,components_by_geom,pads_by_arm,stations_by_geom,
                 sample_mode,expected_inventory,model=None,canonical_geometry=None,single_active_contract=None):
        require(sample_mode=='bead16','This public fixed example supports original bead16 contact sampling only')
        from physics.handle_contacts import verify_contact_sources
        from physics.gravity_jacobian_batch import _load_dependencies
        from physics import live_contact_coverage as rigid_coverage
        rigid_coverage.require_live_contact_coverage(coverage)
        root,pins=verify_contact_sources();self.torch,self.qd,self.gs,arrays,misc=_load_dependencies()
        require(self.gs.backend==self.gs.cuda and self.gs.device.type=='cuda','Deferred contact audit requires actual CUDA')
        self.scene=scene;self.r=scene.rigid_solver;self.sample_mode=sample_mode;self.stride=1 if sample_mode=='bead16' else 100
        self.single_active_contract=None
        native_mode=getattr(scene,'_ap_single_active_arm_contract',None)
        if single_active_contract is not None or native_mode is not None:
            from physics.fixed_peer_arms import validated_single_active_contract,validate_scene_contract
            proof=validated_single_active_contract(native_mode)
            actual={e.name:e for e in self.r.entities if e.name==proof['active_arm_id']}
            actual_proof=validate_scene_contract(scene,actual)
            require(single_active_contract is None or single_active_contract==actual_proof,'Foreign single-active scene contract')
            require(sample_mode=='bead16','Single-active operation is scoped to rigid16 beads')
            self.single_active_contract=actual_proof
        require(scene.is_built and scene.n_envs==0 and self.r._B==1,'Original built single-environment scene required')
        require(scene.sim.substeps==(16 if sample_mode=='bead16' else 100),'Native internal sampling count differs')
        require(scene.sim.dt==.0025 and scene.sim.substeps_local==1 and not scene.sim.requires_grad,
                'Original forward dt/local-clock settings required')
        if model is not None:
            geometry=model.geometry_record()
            require(canonical_geometry is None or digest(canonical_geometry)==digest(geometry),'Explicit/model canonical geometry differs')
            canonical_geometry=geometry
        require(sample_mode!='particle100_last' or canonical_geometry is not None,
                'Particle contacts require the source-bound canonical vessel geometry')
        self.policy=compile_policy(owners=owners,components_by_geom=components_by_geom,pads_by_arm=pads_by_arm,
            stations_by_geom=stations_by_geom,canonical_geometry=canonical_geometry if sample_mode=='particle100_last' else None,
            single_active_contract=self.single_active_contract)
        require(set(owners)==set(range(self.r.n_geoms)),'Coverage must account for every native geometry')
        self.capacity=int(self.r.collider._collider_info.max_contacts[None]);self.eq_capacity=int(self.r.rigid_info.n_candidate_equalities[None])
        require(self.capacity>0 and self.eq_capacity>0,'Positive exact native capacities required')
        self._validate_fields(misc)
        self.static_equalities=int(self.r.n_equalities)
        self.coverage_sha256=coverage['coverage_policy_sha256'];self.sources={str(root/n):h for n,h in pins.items()}
        import physics.episode_contacts as ep,physics.handle_contacts as hc,physics.gravity_feedforward as gf,physics.particle_episode_contacts as pc
        for p in (__file__,ep.__file__,hc.__file__,gf.__file__,pc.__file__):self.sources[str(Path(p).resolve())]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if self.single_active_contract is not None:
            import physics.fixed_peer_arms as fixed
            self.sources[str(Path(fixed.__file__).resolve())]=hashlib.sha256(Path(fixed.__file__).read_bytes()).hexdigest()
        self._binding=self._signature();device=self.gs.device;t=self.torch
        self.ip=t.tensor(self.policy['integer_policy'].copy(),dtype=t.int32,device=device)
        self.fp=t.tensor(self.policy['geometry_policy'].copy(),dtype=t.float64,device=device)
        self.links=t.tensor(self.policy['pose_link_ids'],dtype=t.int32,device=device)
        self.config=t.zeros(8,dtype=t.int32,device=device);self.eq=t.zeros((self.eq_capacity,3),dtype=t.int32,device=device)
        self.state=t.zeros(32,dtype=t.int64,device=device);self.witness=t.zeros(32,dtype=t.float64,device=device)
        self.categories=t.full((self.capacity,),-1,dtype=t.int64,device=device)
        self._initialize,self._audit=_kernels(self.qd,arrays);self._initialize(self.state,self.witness,self.categories,self.capacity)
        self.permuted=bool(self.r.collider._collider_static_config.has_prunable_contacts or self.r.collider._collider_static_config.spatial_sort_supported)
        self.phases=[];self.calls=dict(pre=0,post=0);self.finalized=False
        self.set_phase('settle',expected_inventory=expected_inventory)

    def _signature(self):
        if self.single_active_contract is not None:
            require(getattr(self.scene,'_ap_single_active_arm_contract',None)==self.single_active_contract,
                    'Qualified compact fixed-peer contract changed')
        from physics.live_contact_coverage import _native_buffer
        r=self.r;c=r.collider
        fields=[r.dyn_state.links.pos,r.dyn_state.links.quat,c._collider_state.n_contacts,
            c._collider_state.contact_sort_idx,c._collider_info.max_contacts,
            c._collider_info.contact_pruning_tolerance,r.rigid_info.n_equalities,
            r.rigid_info.n_candidate_equalities,r.constraint_solver.constraint_state.qd_n_equalities,r._errno]
        fields.extend(getattr(c._collider_state.contact_data,k) for k in ('link_a','link_b','geom_a','geom_b','penetration','pos','normal','force'))
        fields.extend(getattr(r.dyn_info.equalities,k) for k in ('eq_type','eq_obj1id','eq_obj2id'))
        return (id(self.scene),id(r),id(r.dyn_state),id(r.dyn_info),id(c._collider_state),id(c._collider_info),
            id(r.constraint_solver.constraint_state),tuple(map(id,r.geoms)),tuple(map(id,r.links)),
            tuple((id(_native_buffer(f)),tuple(f.shape),str(f.dtype)) for f in fields))

    def _validate_fields(self,misc):
        r=self.r;c=r.collider;t=self.torch;fp=self.gs.tc_float;cap=self.capacity
        require(fp in (t.float32,t.float64),'Only the explicit original native float32/64 ABI is supported')
        specs=[(r.dyn_state.links.pos,(r.n_links,1,3),fp),(r.dyn_state.links.quat,(r.n_links,1,4),fp),
            (c._collider_state.n_contacts,(1,),t.int32),(c._collider_state.contact_sort_idx,(cap,1),t.int32),
            (c._collider_info.max_contacts,(),t.int32),(c._collider_info.contact_pruning_tolerance,(),fp),
            (r.rigid_info.n_equalities,(),t.int32),(r.rigid_info.n_candidate_equalities,(),t.int32),
            (r.constraint_solver.constraint_state.qd_n_equalities,(1,),t.int32),(r._errno,(1,),t.int32)]
        for key in ('link_a','link_b','geom_a','geom_b','penetration','pos','normal','force'):
            specs.append((getattr(c._collider_state.contact_data,key),(cap,1)+((3,) if key in ('pos','normal','force') else ()),
                t.int32 if key in ('link_a','link_b','geom_a','geom_b') else fp))
        for key in ('eq_type','eq_obj1id','eq_obj2id'):specs.append((getattr(r.dyn_info.equalities,key),(self.eq_capacity,1),t.int32))
        self._views=[]
        for field,shape,dtype in specs:
            view=misc.qd_to_torch(field,copy=False)
            require(tuple(view.shape)==shape and view.dtype==dtype and view.device.type=='cuda','Native contact field ABI differs')
            self._views.append(view)

    def set_phase(self,phase,active_holder_arm_id=None,*,expected_inventory):
        require(not self.finalized and phase in _PHASES,'Invalid deferred contact boundary phase')
        arms=self.policy['arm_ids'];require(active_holder_arm_id is None or active_holder_arm_id in arms,'Unknown active arm')
        require(active_holder_arm_id is None or phase not in _NO_HOLDER_PHASES,'This phase grants no holder/pad permission')
        rows=expected_inventory.get('rows');n=expected_inventory.get('count')
        require(type(n) is int and 0<=n<=self.eq_capacity and expected_inventory.get('capacity')==self.eq_capacity
            and type(rows) is list and len(rows)==n and all(type(v) is list and len(v)==3
            and all(type(x) is int and x>=0 for x in v) for v in rows),'Actual boundary equality inventory required')
        values=np.zeros((self.eq_capacity,3),np.int32)
        if n:values[:n]=rows
        config=[int(phase not in _NO_HOLDER_PHASES),-1 if active_holder_arm_id is None else arms.index(active_holder_arm_id),n,self.static_equalities,0,0,0,0]
        self.eq.copy_(self.torch.tensor(values,dtype=self.torch.int32,device=self.gs.device))
        self.config.copy_(self.torch.tensor(config,dtype=self.torch.int32,device=self.gs.device))
        self.phases.append(dict(phase=phase,active_holder_arm_id=active_holder_arm_id,expected_inventory=copy.deepcopy(expected_inventory)))

    def _observe(self,ordinal,pre):
        require(not self.finalized and type(ordinal) is int and ordinal>=0,'Exact current Python native ordinal required')
        require(ordinal==self.scene.sim.cur_substep_global,'Supplied ordinal differs from the actual native host clock')
        require(self._signature()==self._binding,'Native object allocation/topology changed')
        r=self.r;c=r.collider
        self._audit(self.ip,self.fp,self.links,self.config,self.eq,self.state,self.witness,self.categories,
            r.dyn_state,r.dyn_info,c._collider_state,c._collider_info,r.rigid_info,
            r.constraint_solver.constraint_state,r._errno,ordinal,pre,self.capacity,self.eq_capacity,
            r.n_geoms,len(self.policy['pose_link_ids']),self.permuted,self.stride)
        self.calls['pre' if pre else 'post']+=1

    def pre(self,global_substep):self._observe(global_substep,True)
    def post(self,global_substep):self._observe(global_substep,False)
    observe_pre_internal=pre
    observe_post_internal=post

    def final_readback(self,*,expected_pre_samples=None,expected_post_samples=None):
        require(not self.finalized,'Deferred contact audit finalizes once')
        require(self._signature()==self._binding,'Native object topology changed before final audit')
        for p,h in self.sources.items():require(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'Contact source changed')
        self.qd.sync()
        state=self.state.detach().cpu().numpy().copy();witness=self.witness.detach().cpu().numpy().copy()
        categories=self.categories.detach().cpu().numpy().copy();self.finalized=True
        require(int(state[7])==self.calls['pre'] and int(state[8])==self.calls['post'],'Device/host audit submission counts differ')
        require(state[7]>0,'No actual contact sample was observed')
        if self.sample_mode=='bead16':require(state[7]==state[8] and state[15]==state[16],'Final bead internal PRE/POST pair is incomplete')
        if expected_pre_samples is not None:require(int(state[7])==expected_pre_samples,'Missing original PRE samples')
        if expected_post_samples is not None:require(int(state[8])==expected_post_samples,'Missing original POST samples')
        result=dict(schema_version='deferred-device-contact-audit-v1',passed=bool(state[0]==0),
            sticky_failed=bool(state[0]),first_bad_global_substep=None if state[1]<0 else int(state[1]),
            first_bad_phase=None if state[2]<0 else ('PRE' if state[2]==0 else 'POST'),first_bad_code=int(state[3]),
            first_bad_raw_row=int(state[4]),first_bad_geom_ids=[int(state[5]),int(state[6])],
            first_bad_position_world_m=witness[:3].tolist(),first_bad_normal_world=witness[3:6].tolist(),first_bad_penetration_m=float(witness[6]),
            PRE_samples=int(state[7]),POST_samples=int(state[8]),retained_contact_rows=int(state[9]),
            category_counts={name:int(state[10+i]) for i,name in enumerate(CATEGORIES)},
            last_contact_count=int(state[17]),last_capacity=int(state[18]),last_category_codes=categories[:max(0,min(int(state[17]),self.capacity))].tolist(),
            sample_mode=self.sample_mode,original_sampling_preserved=True,phase_boundaries=copy.deepcopy(self.phases),
            source_hashes=dict(self.sources),coverage_policy_sha256=self.coverage_sha256,compiled_policy_sha256=self.policy['sha256'],
            per_sample_host_transfers=0,per_sample_explicit_syncs=0,final_host_transfers=3,
            immediate_original_CPU_stop_preserved=False,full_legacy_raw_log_claimed=False,
            scope='Device native contacts/poses/equalities at the declared original sample phases. Sticky failure never clears. Not a continuous-collision or complete handling/material certificate.')
        if self.single_active_contract is not None:
            result.update(schema_version='deferred-single-active-device-contact-audit-v1',
                single_active_contract_sha256=self.single_active_contract['contract_sha256'],
                fixed_peer_ledger_sha256=self.single_active_contract['fixed_peer_ledger_sha256'],
                active_dynamic_arm_ids=[self.single_active_contract['active_arm_id']],
                frozen_arm_ids=list(self.single_active_contract['frozen_arm_ids']),
                original_three_dynamic_arm_model_preserved=False)
        return result
