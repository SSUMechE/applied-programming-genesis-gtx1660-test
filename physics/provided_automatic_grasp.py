"""Supplied ideal grasp: current-pose native weld, explicit supported release.

Root's trusted caller owns source/geometry/contact/support authentication and
contents checks. This helper never steps, snaps, resets or writes material state.
Native welds suppress ALL hand-link/cup-link collision pairs. Finger contacts,
other links/arms/world and cup/material physics are not globally filtered.
"""
from __future__ import annotations
import math
import numpy as np
from physics.carried_body import VesselModel,WorldPose
from physics.provided_manipulation import grasp_endpoint_candidates
from transport_challenge.contracts import ContractError

# Existing BoundedKinematics endpoint verification defaults, not friction gates.
POSITION_TOLERANCE_M=.003
ORIENTATION_TOLERANCE_RAD=.02

def _require(value,message):
    if not value:raise ContractError(message)
def _host(value):return np.asarray(value.detach().cpu().numpy() if hasattr(value,'detach') else value).copy()
def _vector(value,n):
    a=_host(value);_require(a.shape==(n,) and a.dtype.kind=='f' and np.isfinite(a).all(),'Invalid native pose/velocity vector');return a.tolist()
def _pose(entity):
    p=dict(frame='world',position_m=_vector(entity.get_pos(relative=False),3),quaternion_wxyz=_vector(entity.get_quat(relative=False),4))
    WorldPose(p['position_m'],p['quaternion_wxyz']);return p
def _clock(scene):return [int(scene.t),float(scene.cur_t),int(scene.sim.cur_substep_global),int(scene.sim.cur_substep_local)]

def _native_bindings(scene,cup,arms):
    import genesis as gs
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity
    r=scene.rigid_solver
    _require(type(scene) is Scene and type(r) is RigidSolver and scene.is_built and r.n_envs==0
        and not scene.sim.requires_grad,'Built unbatched forward native scene required')
    _require(type(cup) is RigidEntity and cup.solver is r and cup.n_links==1 and len(cup.geoms)==9
        and cup.morph.fixed is False,'Actual free nine-box cup required')
    _require(type(arms) is dict and bool(arms) and all(type(a) is RigidEntity and a.solver is r and a.name==n for n,a in arms.items()),'Actual named native arms required')
    hand={n:a.get_link('hand') for n,a in arms.items()};body=cup.get_link('payload')
    _require(not body.is_fixed and body.entity is cup and all(h.entity is arms[n] for n,h in hand.items()),'Actual hand/payload ownership differs')
    return hand,body,int(gs.EQUALITY_TYPE.WELD)

class ProvidedAutomaticGrasp:
    def __init__(self,scene,*,cup,model,arms,readback='scalar',verify_scalar=False):
        _require(readback in ('scalar','packed') and type(verify_scalar) is bool
            and (not verify_scalar or readback=='packed'),'Explicit scalar/packed grasp readback and Boolean packed parity required')
        _require(type(model) is VesselModel,'Immutable supplied vessel model required')
        self.hands,self.body,self.weld_type=_native_bindings(scene,cup,arms)
        self.scene,self.cup,self.model,self.arms=scene,cup,model,dict(arms)
        self.r=scene.rigid_solver;self._holder=None;self._owned_row=None;self._faulted=False
        self._readback=readback;self._verify_scalar=verify_scalar;self._readback_checks=0;self._inventory_checks=0
        self._snapshot_counts={'scalar':0,'packed':0};self._inventory_counts={'scalar':0,'packed':0}
        self._readback_pack=None
        if readback=='packed':
            from physics.grasp_snapshot_pack import GraspSnapshotPack
            self._readback_pack=GraspSnapshotPack(scene,cup=cup,hands=self.hands,body=self.body)
        self._baseline=self._inventory()
        _require(not any(self.body.idx in p for p in self._baseline['weld_pairs']),'Cup already has a native holder')

    @property
    def holder(self):return self._holder

    def _old_inventory(self):
        count=_host(self.r.constraint_solver.constraint_state.qd_n_equalities.to_numpy())
        capacity=_host(self.r.rigid_info.n_candidate_equalities.to_numpy())
        _require(count.shape==(1,) and count.dtype.kind in 'iu' and capacity.shape==() and capacity.dtype.kind in 'iu','Native equality counter layout differs')
        n,cap=int(count[0]),int(capacity);_require(0<=self.r.n_equalities<=n<=cap,'Native equality capacity invalid')
        eq=self.r.get_equality_constraints(as_tensor=True,to_torch=False)
        arrays=[_host(eq[k]) for k in ('type','obj_a','obj_b')]
        _require(all(a.shape==(1,n) and a.dtype.kind in 'iu' for a in arrays),'Public equality inventory differs')
        rows=np.stack(arrays,axis=-1)[0].tolist()
        weld=self.r.get_weld_constraints(as_tensor=True,to_torch=False);a,b=(_host(weld[k]) for k in ('link_a','link_b'))
        pairs=[row[1:] for row in rows if row[0]==self.weld_type]
        _require(a.shape==b.shape==(1,len(pairs)) and a.dtype.kind in 'iu' and b.dtype.kind in 'iu'
            and np.stack((a,b),axis=-1)[0].tolist()==pairs,'Public weld pair/count readback differs')
        return dict(count=n,capacity=cap,rows=rows,weld_pairs=pairs)

    def _inventory(self):
        self._inventory_counts[self._readback]+=1
        if self._readback=='scalar':return self._old_inventory()
        try:
            clock=_clock(self.scene);result=self._readback_pack.inventory()
            if self._verify_scalar:
                _require(result==self._old_inventory(),'Packed grasp equality inventory differs from exact same-clock scalar readback')
                self._inventory_checks+=1
            _require(clock==_clock(self.scene),'Native clock advanced during provided-grasp inventory read');return result
        except BaseException:self._faulted=True;raise

    def _old_snapshot(self,arm_id):
        clock=_clock(self.scene);hand=self.hands[arm_id];mass=_host(self.cup.get_mass())
        _require(mass.size==1 and mass.dtype.kind=='f' and np.isfinite(mass).all() and mass.item()>0,'Invalid cup mass')
        result=dict(native_clock=clock,body_entity_idx=int(self.cup.idx),cup_link_idx=int(self.body.idx),hand_link_idx=int(hand.idx),cup_mass_kg=float(mass.item()),
            body_world=_pose(self.cup),hand_world=_pose(hand),body_linear_velocity_m_s=_vector(self.cup.get_vel(),3),body_angular_velocity_rad_s=_vector(self.cup.get_ang(),3),
            hand_linear_velocity_m_s=_vector(hand.get_vel(),3),hand_angular_velocity_rad_s=_vector(hand.get_ang(),3),inventory=self._old_inventory())
        _require(clock==_clock(self.scene),'Native clock advanced during provided-grasp read');return result

    def snapshot(self,arm_id):
        _require(not self._faulted and arm_id in self.arms and self.scene.is_built and self.scene.rigid_solver is self.r
            and self.cup.solver is self.r and self.cup.get_link('payload') is self.body
            and self.arms[arm_id].get_link('hand') is self.hands[arm_id],'Bound scene/arm/cup identity changed or failed transaction')
        self._snapshot_counts[self._readback]+=1
        if self._readback=='scalar':return self._old_snapshot(arm_id)
        try:
            clock=_clock(self.scene);result=self._readback_pack.snapshot(arm_id)
            if self._verify_scalar:
                _require(result==self._old_snapshot(arm_id),'Packed provided-grasp snapshot differs from exact same-clock scalar readback')
                self._readback_checks+=1
            _require(clock==_clock(self.scene),'Native clock advanced during provided-grasp read');return result
        except BaseException:self._faulted=True;raise

    def readback_metadata(self):
        return dict(mode=self._readback,verify_scalar=self._verify_scalar,
            exact_same_clock_snapshot_parity_checks=self._readback_checks,
            exact_same_clock_inventory_parity_checks=self._inventory_checks,
            snapshot_calls_by_mode=dict(self._snapshot_counts),inventory_calls_by_mode=dict(self._inventory_counts),
            packed=None if self._readback_pack is None else self._readback_pack.metadata(),
            scope='Fresh readback only. No cached pose, mass, inventory, holder or support verdict. Native grasp transaction guards are unchanged.')

    def set_diagnostic_readback(self,mode):
        _require(mode in ('scalar','packed') and self._readback_pack is not None and not self._faulted,
            'Diagnostic grasp readback needs scalar/packed mode and an already constructed pack')
        self._readback=mode

    def _check_inventory(self,snapshot):
        expected=self._baseline['rows']+([] if self._owned_row is None else [self._owned_row])
        inv=snapshot['inventory'];_require(inv['capacity']==self._baseline['capacity'] and inv['rows']==expected and inv['count']==len(expected),'Native equality/holder inventory changed')

    @staticmethod
    def _unchanged(before,after):
        _require({k:v for k,v in before.items() if k!='inventory'}=={k:v for k,v in after.items() if k!='inventory'},'Attach/release or support callback changed pose/velocity/mass/clock')

    def attach(self,arm_id):
        _require(self._holder is None,'Double attachment is not allowed')
        before=self.snapshot(arm_id);self._check_inventory(before)
        target=grasp_endpoint_candidates(self.model,before['body_world'])[0]
        measured=WorldPose(before['hand_world']['position_m'],before['hand_world']['quaternion_wxyz'])
        position=math.dist(target.position_m,measured.position_m)
        dot=abs(sum(a*b for a,b in zip(target.quaternion_wxyz,measured.quaternion_wxyz)))
        angle=2*math.acos(min(1.,dot))
        _require(position<=POSITION_TOLERANCE_M and angle<=ORIENTATION_TOLERANCE_RAD,'Actual hand has not reached the supplied grasp endpoint')
        inv=before['inventory'];_require(inv['count']<inv['capacity'],'No reserved native dynamic-weld capacity')
        pair=[before['hand_link_idx'],before['cup_link_idx']]
        _require(all(0<=i<self.r.n_links for i in pair),'Native weld link index outside this scene')
        self.r.add_weld_constraint(*pair)
        try:
            after=self.snapshot(arm_id);self._unchanged(before,after)
            _require(after['inventory']['rows']==inv['rows']+[[self.weld_type,*pair]] and after['inventory']['count']==inv['count']+1,'Native weld add was ignored or changed another equality')
        except BaseException:self._faulted=True;raise
        self._holder=arm_id;self._owned_row=[self.weld_type,*pair]
        return dict(model='provided-ideal-native-weld-v1',event='attach',arm_id=arm_id,pair=pair,position_error_m=position,orientation_error_rad=angle,before=before,after=after,
            scope='Native attachment membership only; finite solver compliance and actual carried motion require physical observation. Standard entire hand-link/cup-link collision exclusion applies.')

    def release(self,arm_id,*,support_ready):
        _require(self._holder==arm_id and arm_id in self.arms,'Only the actual provided holder may release')
        _require(callable(support_ready),'Actual support-ready callback is required')
        before=self.snapshot(arm_id);self._check_inventory(before);proof=support_ready(before)
        checked=self.snapshot(arm_id);self._unchanged(before,checked);self._check_inventory(checked)
        _require(type(proof) is dict and proof.get('passed') is True and proof.get('native_clock')==before['native_clock']
            and proof.get('body_entity_idx')==before['body_entity_idx'] and type(proof.get('support_id')) is str and bool(proof['support_id'].strip())
            and type(proof.get('evidence_sha256')) is str and len(proof['evidence_sha256'])==64 and all(c in '0123456789abcdef' for c in proof['evidence_sha256']),
            'Actual same-clock/body support-ready evidence is missing')
        pair=self._owned_row[1:];self.r.delete_weld_constraint(*pair)
        try:
            after=self.snapshot(arm_id);self._unchanged(before,after)
            _require(after['inventory']==self._baseline,'Native weld release was ignored or changed another equality')
        except BaseException:self._faulted=True;raise
        self._holder=None;self._owned_row=None
        return dict(model='provided-ideal-native-weld-v1',event='release',arm_id=arm_id,pair=pair,support_ready=proof,before=before,after=after,
            scope='Trusted runtime callback owns actual support/contact/rest authentication. No material/delivery success is assigned here; existing velocities and contents continue physically.')
