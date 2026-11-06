"""Read-only public-getter observation pack for two or three Panda arms.

No controller, sensor, contact, collision query or simulator step is called.
Scalar and packed reads expose the same fresh state at the current native clock.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path
import numpy as np

from transport_challenge.contracts import ContractError, digest
from physics.particle_material import ParticleMaterialReader
_particle_pause_hook = None
_PARTICLE_PAUSE_READ = None

OBSERVATION_SOURCE_PINS = {
    'engine/entities/rigid_entity/rigid_entity.py': 'b0f86f9c9ebdf62b9df2e5aadc0719c28e06cd52e0ce1bef452b1a269d9b2092',
    'engine/entities/rigid_entity/rigid_link.py': 'c192087859143ffc931e36db40306d456adb92b412ad0ca2704119747ffae690',
    'engine/solvers/rigid/rigid_solver.py': 'e076854af17262323041544ff692d366402a700357053618c8d89fed1e86b19f',
    'engine/solvers/kinematic_solver.py': 'c663445fd2dfbbf2286cec08166b1c45d34dd7e1bb9850d881c7b0b2d408ed20',
    'utils/misc.py': 'ba5949408015aa7148dc1518977a958bca15e665b40b44243bfa5d5b2fa62454',
}


def observation_source_paths():
    """Verified absolute source paths for parent freeze manifests."""
    from physics.handle_contacts import verify_contact_sources
    from physics.live_contact_coverage import live_contact_readback_source_paths
    root,_=verify_contact_sources()
    result=live_contact_readback_source_paths()
    for name,expected in OBSERVATION_SOURCE_PINS.items():
        path=Path(root)/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ContractError('Reviewed observation getter changed: '+name)
        result[str(path.resolve())]=expected
    return result


def _dependencies(scene):
    sources=observation_source_paths()
    import torch
    import quadrants as qd
    import genesis as gs
    solver=scene.rigid_solver
    path=Path(inspect.getsourcefile(type(solver))).resolve()
    if (str(path) not in sources or type(solver).__name__!='RigidSolver'
            or type(solver).__module__!='genesis.engine.solvers.rigid.rigid_solver'
            or gs.use_zerocopy is not True):
        raise ContractError('Unreviewed public-getter backend. No scalar fallback.')
    return torch,qd,sources


def _clock(scene):
    step=scene.t;time=scene.cur_t
    if (isinstance(step,(bool,np.bool_)) or not isinstance(step,(int,np.integer)) or step<0
            or isinstance(time,(bool,np.bool_)) or not isinstance(time,(int,float,np.number))
            or not math.isfinite(float(time)) or time<0):
        raise ContractError('Invalid native observation clock')
    return int(step),float(time)


def _host(value,shape):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    a=np.asarray(value,dtype=float).copy()
    if a.shape!=shape or not np.isfinite(a).all():
        raise ContractError('Nonfinite/wrong-shaped native observation')
    return a


def _pose(position,quaternion):
    return {'frame':'world','position_m':position.tolist(),'quaternion_wxyz':quaternion.tolist()}


def require_equal_observations(first,second):
    """Exact same-clock comparison, without a physical tolerance exemption."""
    if first!=second:raise ContractError('Same-time scalar/packed observation mismatch')


class EpisodeObservationReader:
    def __init__(self,scene,*,arms,hands,fingers,tray,beads,particle_material=None,payload_mode=None):
        if payload_mode not in (None,'empty') or payload_mode=='empty' and (particle_material is not None or len(beads)!=0):
            raise ContractError('Empty observation requires explicit empty mode and no content reader/entities')
        self.payload_mode=payload_mode
        self.particle_material=particle_material
        self._particle_binding=self._bind_particle_material(scene,beads,particle_material)
        self.single_active_contract=None
        if len(arms)==1:
            from physics.fixed_peer_arms import validate_scene_contract
            self.single_active_contract=validate_scene_contract(scene,arms)
        if (len(arms) not in (2,3) and self.single_active_contract is None or (len(beads)!=(0 if payload_mode=='empty' else 4) if particle_material is None else bool(beads)) or set(hands)!=set(arms) or set(fingers)!=set(arms)
                or any(not isinstance(n,str) or not n.strip() for n in arms)
                or any(len(fingers[n])!=2 for n in arms) or scene.rigid_solver.n_envs!=0):
            raise ContractError('Observation pack requires unbatched two/three-Panda and its explicit payload inventory')
        self.scene=scene;self.solver=scene.rigid_solver
        self.arms=dict(arms);self.hands=dict(hands);self.fingers={n:dict(v) for n,v in fingers.items()}
        self.tray=tray;self.beads=tuple(beads);self.names=tuple(arms)
        self.torch,self.qd,self.sources=_dependencies(scene)
        if any(a.n_qs!=9 or a.n_dofs!=9 for a in self.arms.values()):
            raise ContractError('Expected seven native joints and two native fingers per arm')
        self.q_indices=[int(a.q_start)+i for a in self.arms.values() for i in range(9)]
        self.dof_indices=[int(a.dof_start)+i for a in self.arms.values() for i in range(9)]
        self.pose_links=[link for n in self.names for link in (self.hands[n],*self.fingers[n].values())]
        self.pose_ids=[int(link.idx) for link in self.pose_links]+[int(tray.base_link_idx)]
        self.position_ids=self.pose_ids+[int(b.base_link_idx) for b in self.beads]
        self.velocity_ids=[int(tray.base_link_idx)]+[int(b.base_link_idx) for b in self.beads]
        entity_ids=[a.idx for a in (*self.arms.values(),self.tray,*self.beads)]
        if (any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in entity_ids)
                or any(len(v)!=len(set(v)) or min(v)<0 for v in (entity_ids,self.q_indices,self.dof_indices,self.position_ids))):
            raise ContractError('Overlapping/negative native observation indices')
        self._body_slot=len(self.pose_links)
        self._shapes=((len(self.q_indices),),(len(self.dof_indices),),
            (len(self.position_ids),3),(len(self.pose_ids),4),(len(self.velocity_ids),3),(1,3))
        self._packed_values=sum(math.prod(shape) for shape in self._shapes)
        self._signature=self._topology();self._dtype_device=None
        self.batched_reads=0;self.scalar_reads=0;self.parity_checks=0

    def _topology(self):
        if self.single_active_contract is not None and getattr(self.scene,'_ap_single_active_arm_contract',None)!=self.single_active_contract:
            raise ContractError('Qualified static peer contract changed during observation')
        rigid=(id(self.scene),id(self.scene.rigid_solver),self.scene.rigid_solver.n_envs,
            tuple((n,id(a),a.idx,a.q_start,a.dof_start,a.n_qs,a.n_dofs) for n,a in self.arms.items()),
            tuple((id(x),x.idx) for x in self.pose_links),
            tuple((id(x),x.idx,x.base_link_idx) for x in (self.tray,*self.beads)),
            self.names,tuple((n,id(x),x.idx) for n,x in self.hands.items()),
            tuple((n,tuple((k,id(x),x.idx) for k,x in links.items())) for n,links in self.fingers.items()),
            tuple(self.q_indices),tuple(self.dof_indices),tuple(self.pose_ids),
            tuple(self.position_ids),tuple(self.velocity_ids),self._body_slot,self._shapes,self._packed_values)

        return rigid if self.particle_material is None else rigid+(self._particle_topology(),)

    def _begin(self):
        if self._topology()!=self._signature:raise ContractError('Observation entity/index binding changed')
        if self.particle_material is not None:self._particle_read_clock=self._particle_native_clock()
        return _clock(self.scene)

    def _finish(self,clock):
        if _clock(self.scene)!=clock:raise ContractError('Scene clock changed during observation')
        if self._topology()!=self._signature:raise ContractError('Observation topology changed while reading')
        if self.particle_material is not None and self._particle_native_clock()!=self._particle_read_clock:
            raise ContractError('Native internal clock changed during composite particle observation')

    def _record(self,clock,q,qd,positions,quaternions,velocities,angular,*,_particle_pending=False):
        arms={};index=0
        for arm_index,name in enumerate(self.names):
            hand=_pose(positions[index],quaternions[index]);index+=1;finger={}
            for key in self.fingers[name]:
                finger[key]=_pose(positions[index],quaternions[index]);index+=1
            arms[name]={'q_all_dofs':q[9*arm_index:9*(arm_index+1)].tolist(),
                'qd_all_dofs':qd[9*arm_index:9*(arm_index+1)].tolist(),
                'hand_world':hand,'finger_world':finger}
        record={'simulation_s':clock[1],'native_scene_step':clock[0],'arms':arms,
            'body_world':_pose(positions[self._body_slot],quaternions[self._body_slot]),
            'body_linear_velocity_m_s':velocities[0].tolist(),
            'body_angular_velocity_rad_s':angular[0].tolist(),
            'material_positions_world_m':positions[self._body_slot+1:].tolist(),
            'material_velocities_world_m_s':velocities[1:].tolist()}
        if self.single_active_contract is not None:
            import copy
            record['single_active_arm_contract']=copy.deepcopy(self.single_active_contract)
        if _particle_pending:
            if self.particle_material is None:raise ContractError('Rigid pending is particle-only')
            del record['material_positions_world_m'];del record['material_velocities_world_m_s']
            self._finish(clock)
            return {'schema_version':'episode-rigid-pending-particle-read-v1','body_observation':record}
        return record if self.particle_material is None else self._compose_particles(record,clock)

    def read_batched(self,*,_particle_pending=False):
        if type(_particle_pending) is not bool or (_particle_pending and self.particle_material is None):
            raise ContractError('Explicit particle-only pending mode required')
        clock=self._begin();self.qd.sync();s=self.solver
        tensors=(s.get_qpos(qs_idx=self.q_indices),s.get_dofs_velocity(dofs_idx=self.dof_indices),
            s.get_links_pos(self.position_ids,ref='link_origin',relative=False),
            s.get_links_quat(self.pose_ids,relative=False),
            s.get_links_vel(self.velocity_ids,ref='link_origin'),s.get_links_ang([self.tray.base_link_idx]))
        shapes=self._shapes
        for value,shape in zip(tensors,shapes):
            if not isinstance(value,self.torch.Tensor) or tuple(value.shape)!=shape or value.dtype not in (self.torch.float32,self.torch.float64):
                raise ContractError('Unexpected public getter tensor shape/dtype')
        identities={(v.dtype,v.device) for v in tensors}
        if len(identities)!=1:raise ContractError('Public getter tensor dtype/device disagreement')
        identity=next(iter(identities))
        if self._dtype_device is not None and identity!=self._dtype_device:
            raise ContractError('Observation tensor dtype/device changed')
        self._dtype_device=identity
        packed=self.torch.cat([v.reshape(-1) for v in tensors])
        host=_host(packed,(self._packed_values,));arrays=[];offset=0
        for shape in shapes:
            count=math.prod(shape);arrays.append(host[offset:offset+count].reshape(shape));offset+=count
        self._finish(clock);self.batched_reads+=1
        return self._record(clock,*arrays,_particle_pending=_particle_pending)

    def read_batched_particle_bundle(self):
        """Owned same-clock read. No caller-supplied arrays or prior state."""
        if (type(self) is not EpisodeObservationReader or self.particle_material is None
                or any(k in vars(self) for k in ('read_batched', '_record',
                    'read_batched_particle_bundle', '_particle_topology', '_begin', '_finish'))
                or _particle_pause_hook.read_particle_pause_snapshot is not _PARTICLE_PAUSE_READ):
            raise ContractError('Use the exact owned particle observation reader/hook')
        self._particle_topology()
        pending=self.read_batched(_particle_pending=True)
        if (type(pending) is not dict or set(pending)!={'schema_version','body_observation'}
                or pending['schema_version']!='episode-rigid-pending-particle-read-v1'):
            raise ContractError('Actual rigid pending fragment required')
        record=pending['body_observation']
        clock=(record['native_scene_step'],record['simulation_s'])
        snapshot,pack=_PARTICLE_PAUSE_READ(self.particle_material,record,return_reader_result=True)
        if (type(pack) is not dict or set(pack)!={'arrays','summary'}
                or pack['summary'].get('native_scene_step')!=clock[0]
                or pack['summary'].get('simulation_s')!=clock[1]
                or pack['summary'].get('persistent_id_count')!=self.particle_material.n
                or pack['summary'].get('native_entity_idx')!=self.particle_material.entity_idx
                or pack['summary'].get('particle_material_contract_sha256')!=self._particle_binding[4]):
            raise ContractError('Particle bundle lacks current-clock inventory binding')
        arrays=pack['arrays'];n=self.particle_material.n
        for name,shape in (('pos',(n,3)),('vel',(n,3)),('active',(n,)),('mass_kg',(n,)),('native_particle_id',(n,))):
            if not isinstance(arrays.get(name),np.ndarray) or arrays[name].shape!=shape:
                raise ContractError('Particle bundle omitted native inventory array: '+name)
        self._finish(clock)
        record['observation_material_mode']='native_particles'
        return {'schema_version':'episode-native-particle-observation-bundle-v1',
            'body_observation':record,'reader_result':pack,'pause_snapshot':snapshot}

    def read_scalar(self):
        """Independent entity-level getter sequence for the native parity probe."""
        clock=self._begin();q=[];qd=[];positions=[];quats=[]
        body_pos=_host(self.tray.get_pos(relative=False),(3,));body_q=_host(self.tray.get_quat(relative=False),(4,))
        bead_pos=[_host(b.get_pos(relative=False),(3,)) for b in self.beads]
        bead_vel=[_host(b.get_vel(),(3,)) for b in self.beads]
        for name in self.names:
            arm=self.arms[name];q.extend(_host(arm.get_qpos(),(9,)));qd.extend(_host(arm.get_dofs_velocity(),(9,)))
            for link in (self.hands[name],*self.fingers[name].values()):
                positions.append(_host(link.get_pos(relative=False),(3,)));quats.append(_host(link.get_quat(relative=False),(4,)))
        positions.extend((body_pos,*bead_pos));quats.append(body_q)
        velocities=[_host(self.tray.get_vel(),(3,)),*bead_vel];angular=_host(self.tray.get_ang(),(3,))[None,:]
        self._finish(clock);self.scalar_reads+=1
        return self._record(clock,np.asarray(q),np.asarray(qd),np.asarray(positions),np.asarray(quats),np.asarray(velocities),angular)

    def verify(self):
        first=self.read_scalar();second=self.read_batched()
        if self.particle_material is None:require_equal_observations(first,second)
        else:self._require_equal_particle_observations(first,second)
        self.parity_checks+=1
        return first

    def metadata(self):
        record={'schema_version':'episode-public-observation-pack-v1','packed_values':self._packed_values,
            'q_indices':self.q_indices.copy(),'dof_indices':self.dof_indices.copy(),
            'position_link_indices':self.position_ids.copy(),'pose_link_indices':self.pose_ids.copy(),
            'velocity_link_indices':self.velocity_ids.copy(),'source_hashes':dict(self.sources),
            'modeled_cpu_transfers_per_batched_read':1,'public_getters_per_batched_read':6,
            'scalar_fallback':False,'batched_reads':self.batched_reads,'scalar_reads':self.scalar_reads,
            'same_time_parity_checks':self.parity_checks,
            'native_dtype_device':None if self._dtype_device is None else [str(x) for x in self._dtype_device]}
        if self.payload_mode=='empty':
            record.update(payload_mode='empty',material_count=0,material_rest='NOT_APPLICABLE')
        return record if self.particle_material is None else self._particle_metadata(record)

    @staticmethod
    def _bind_particle_material(scene,beads,reader):
        if reader is None:return None
        if type(reader) is not ParticleMaterialReader or reader.scene is not scene or len(beads)!=0:
            raise ContractError('Particle mode requires this scene\'s exact ParticleMaterialReader and no rigid beads')
        methods=('read','_topology','verify_source_bindings')
        if any(name in vars(reader) for name in methods):
            raise ContractError('Do not replace particle reader methods or install a fallback')
        reader.verify_source_bindings()
        contract=reader.contract
        if digest(contract)!=reader.contract_sha256:
            raise ContractError('Particle reader contract digest differs')
        return (id(reader),id(reader.scene),id(reader.entity),id(reader.solver),reader.contract_sha256,
            tuple(sorted(reader.source_pins.items())),tuple(getattr(type(reader),name) for name in methods))

    def _particle_topology(self):
        reader=self.particle_material
        if type(reader) is not ParticleMaterialReader or reader.scene is not self.scene:
            raise ContractError('Particle material reader/scene binding changed')
        methods=('read','_topology','verify_source_bindings')
        if any(name in vars(reader) for name in methods):
            raise ContractError('Particle reader method was replaced')
        reader._topology()  # Read-only native identity/settings check, no getter or source-file I/O.
        actual=(id(reader),id(reader.scene),id(reader.entity),id(reader.solver),reader.contract_sha256,
            tuple(sorted(reader.source_pins.items())),tuple(getattr(type(reader),name) for name in methods))
        if actual!=self._particle_binding:
            raise ContractError('Particle reader entity/solver/contract/source binding changed')
        if self.beads:
            raise ContractError('Particle observation cannot mix rigid bead proxies')
        return actual

    def _particle_native_clock(self):
        sim=self.scene.sim
        values=(sim.cur_substep_global,sim.cur_substep_local)
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<0 for v in values):
            raise ContractError('Invalid native particle internal clock')
        return (id(sim),*(int(v) for v in values))

    def _compose_particles(self,record,clock):
        self._particle_topology()
        # Omit the bead schema completely. A legacy rigid consumer must explicitly
        # branch, rather than receiving an empty supply that could pass all([]).
        del record['material_positions_world_m']
        del record['material_velocities_world_m_s']
        pack=self.particle_material.read(record,include_rows=False)
        if (not isinstance(pack,dict) or set(pack)!={'arrays','summary'}
                or pack['summary'].get('native_scene_step')!=clock[0]
                or pack['summary'].get('simulation_s')!=clock[1]
                or pack['summary'].get('persistent_id_count')!=self.particle_material.n
                or pack['summary'].get('native_entity_idx')!=self.particle_material.entity_idx
                or pack['summary'].get('particle_material_contract_sha256')!=self._particle_binding[4]):
            raise ContractError('Particle readback lacks this exact current-clock inventory binding')
        arrays=pack['arrays'];n=self.particle_material.n
        for name,shape in (('pos',(n,3)),('vel',(n,3)),('active',(n,)),('mass_kg',(n,)),('native_particle_id',(n,))):
            if not isinstance(arrays.get(name),np.ndarray) or arrays[name].shape!=shape:
                raise ContractError('Particle readback omitted native inventory array: '+name)
        record['observation_material_mode']='native_particles'
        record['native_particle_material']=pack
        self._finish(clock)
        return record

    @staticmethod
    def _require_equal_particle_observations(first,second):
        if set(first)!=set(second):raise ContractError('Particle observation schema mismatch')
        first_rigid={k:v for k,v in first.items() if k!='native_particle_material'}
        second_rigid={k:v for k,v in second.items() if k!='native_particle_material'}
        require_equal_observations(first_rigid,second_rigid)
        a=first['native_particle_material'];b=second['native_particle_material']
        if a['summary']!=b['summary'] or set(a['arrays'])!=set(b['arrays']):
            raise ContractError('Same-time native material summary/array schema mismatch')
        for name,value in a['arrays'].items():
            other=b['arrays'][name]
            if (not isinstance(value,np.ndarray) or not isinstance(other,np.ndarray)
                    or value.dtype!=other.dtype or value.shape!=other.shape
                    or not np.array_equal(value,other)):
                raise ContractError('Same-time native particle array mismatch: '+name)

    def _particle_metadata(self,record):
        self._particle_topology()
        reader=self.particle_material
        sources=dict(record['source_hashes'])
        for path,sha in reader.source_pins.items():
            if path in sources and sources[path]!=sha:
                raise ContractError('Rigid and particle source bindings conflict')
            sources[path]=sha
        # Three public particle getters. SPH exports mass/rho. MPM exports
        # mass/material_idx/free/current-ring F. The initial MPM public mass call
        # belongs to reader construction and is not a per-step transfer.
        fields=2 if reader.kind=='process_liquid' else 4
        extra=3+fields
        record.update(schema_version='episode-public-observation-pack-native-particles-v1',
            observation_material_mode='native_particles',source_hashes=sources,
            native_particle_material_contract_sha256=reader.contract_sha256,
            native_particle_material_count=reader.n,rigid_content_proxy_count=0,
            modeled_cpu_transfers_per_batched_read=1+extra,
            native_particle_readback={'public_particle_getters_per_read':3,
                'native_field_exports_per_read':fields,'modeled_extra_cpu_transfers_per_read':extra,
                'rigid_pack_cpu_transfers_per_batched_read':1,'per_particle_row_dicts_created':False,
                'native_clock_and_topology_verified':True,
                'source_hashes_verified_at_binding_and_parent_freeze':True,
                'source_files_rehashed_per_step':False,
                'public_mass_status':reader.contract['public_mass_status'],
                'scope':'Logical host-transfer count from public getters and native field exports. Native timing and bytes are not measured here. Scalar/packed parity reads material twice.'})
        return record
