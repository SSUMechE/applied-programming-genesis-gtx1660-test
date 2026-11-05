"""Provided deferred CUDA control. Native qualification is a separate gate.

Only construction and finish read host data. apply/capture_post enqueue QD work.
The native PD controller, force range and outer-cadence Jacobian gravity formula
remain in place. This module does not advance physics or certify contacts.
"""
import copy
import hashlib
import math
from pathlib import Path
import threading

import numpy as np

from transport_challenge.contracts import ContractError

DT = .0025
HEADER = 32
WIDTH = 13
FLAGS = ('nonfinite_model_or_gravity', 'model_changed', 'quaternion_invalid',
         'motor_identity', 'motor_force_bound', 'effort_saturation',
         'position_limit', 'velocity_limit', 'acceleration_limit',
         'tracking_limit', 'nonfinite_native_state', 'target_or_mode_changed', 'tool_workspace')
FIRST_FAILURE_SLOT = len(FLAGS)
CHANNELS = ('gravity_tau', 'requested_effort', 'native_bounded_effort',
            'pre_q', 'pre_qd', 'reference_q', 'reference_qd', 'post_q',
            'post_qd', 'post_qdd_fd', 'native_ctrl_pos', 'native_ctrl_vel',
            'native_gravity_bias')
PROVIDED_PINS = {
    'gravity_feedforward.py':'808e764bd403247bd38e4197b5e40c1049e56482d5406ffbbcb55075db42b001',
    'gravity_jacobian_batch.py':'4cc4a616d5e88135304c777fa6fa2b5b39ca6216ed35f8a71926c2145886ff59',
}
EXTRA_NATIVE_PINS = {
    'engine/solvers/rigid/abd/accessor.py':'64250f83e393119cdec38f6bfdac243a68fc8701e3a6109e1775c78c3513c54d',
    'engine/solvers/rigid/abd/forward_dynamics.py':'464f6deae2e8c61bc43d7ed483e07dd4bd71fc6cdbec150a2af7461ba92a5710',
}
_KERNELS = {}


def require(value, message):
    if not value:raise ContractError(message)


def numeric(value, shape, label):
    a = np.asarray(value)
    require(a.shape == tuple(shape) and a.dtype.kind in 'iuf' and np.isfinite(a).all(),
            'Finite supplied numeric array required: '+label)
    return np.asarray(a, dtype=np.float64).copy()


def command_tables(commands, names, *, native_dtype):
    """Normalize exact provided commands once. Never fit or generate a path."""
    if type(commands) is dict and set(commands) == {'position', 'velocity'}:
        position = np.asarray(commands['position'])
        require(position.ndim == 2 and position.shape[1] == 9*len(names)
                and position.shape[0] > 0, 'Position table needs T by the actual controlled DOFs')
        q = numeric(position, position.shape, 'reference positions')
        v = numeric(commands['velocity'], q.shape, 'reference velocities')
    else:
        rows = list(commands);require(bool(rows), 'An exact saved command grid is required')
        positions = [];velocities = []
        for row in rows:
            arms = row.get('arms') if type(row) is dict else row.arms
            require(len(arms)==len(names), 'Do not duplicate or drop persistent arm commands')
            if type(arms) is dict:by_name = arms
            else:by_name = {(c.get('arm_id') if type(c) is dict else c.arm_id):c for c in arms}
            require(set(by_name) == set(names), 'Every command must retain all persistent arms')
            p = [];v = []
            for name in names:
                c = by_name[name]
                get = c.__getitem__ if type(c) is dict else lambda k, c=c:getattr(c, k)
                p.extend(numeric(get('q_rad'), (7,), 'q').tolist())
                p.extend(numeric(get('finger_q_m'), (2,), 'fingers').tolist())
                v.extend(numeric(get('qd_rad_s'), (7,), 'qd').tolist())
                v.extend(numeric(get('finger_qd_m_s'), (2,), 'finger qd').tolist())
            positions.append(p);velocities.append(v)
        q = np.asarray(positions, dtype=np.float64);v = np.asarray(velocities, dtype=np.float64)
    reference = np.ascontiguousarray(np.stack((q, v), axis=-1))
    native = np.ascontiguousarray(reference.astype(native_dtype))
    require(np.isfinite(native).all(), 'Native command cast overflowed')
    return reference, native


def gravity_reference(jacobians, masses, gravity):
    """Original serial-link NumPy formula, supplied for CPU/native parity."""
    j = np.asarray(jacobians);m = numeric(masses, (len(j),), 'masses')
    g = numeric(gravity, (3,), 'gravity')
    require(j.ndim == 3 and j.shape[1] == 6 and np.isfinite(j).all(), 'Finite COM Jacobians required')
    result = np.zeros(j.shape[2], dtype=np.float64)
    for link, mass in zip(j, m):result -= link[:3].astype(np.float64).T @ (mass*g)
    return result


def source_pins(native_root):
    from physics.gravity_jacobian_batch import GENESIS_SOURCE_PINS, QUADRANTS_SOURCE_PINS
    import importlib.util
    pins = {str(Path(__file__).with_name(p).resolve()):h for p,h in PROVIDED_PINS.items()}
    pins.update({str((Path(native_root)/p).resolve()):h for p,h in {**GENESIS_SOURCE_PINS, **EXTRA_NATIVE_PINS}.items()})
    spec = importlib.util.find_spec('quadrants')
    require(spec is not None and spec.origin is not None, 'Reviewed Quadrants source must exist')
    pins.update({str((Path(spec.origin).parent/p).resolve()):h for p,h in QUADRANTS_SOURCE_PINS.items()})
    for p,h in pins.items():require(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, 'Control source changed: '+p)
    return pins


def _kernels(qd, gs, arrays):
    key = (id(qd), id(gs), id(arrays))
    if key in _KERNELS:return _KERNELS[key]

    @qd.func
    def finite(value):
        # Bit-pattern detection remains effective with native fast-math flags.
        return (qd.bit_cast(qd.cast(value, qd.f64), qd.u64) & qd.u64(0x7ff0000000000000)) != qd.u64(0x7ff0000000000000)

    @qd.func
    def flag(output:qd.template(), reason, step):
        qd.atomic_max(output[reason], 1.0)
        qd.atomic_min(output[FIRST_FAILURE_SLOT], qd.cast(step, qd.f64))

    @qd.kernel
    def prepare(arm:qd.template(), model:qd.types.ndarray(), points:qd.types.ndarray(),
                masses:qd.types.ndarray(), output:qd.types.ndarray(), step:qd.i32,
                state:arrays.DynState, info:arrays.DynInfo, config:qd.template()):
        for local in range(arm.n_links):
            link = arm._link_start+local
            I = [link, 0] if qd.static(config.batch_links_info) else link
            mass = qd.cast(info.links.inertial_mass[I], qd.f64)
            # Match the public link_com getter: native addition, then FP64 copy.
            com_native = state.links.i_pos[link,0]+state.links.root_COM[link,0]
            delta = qd.Vector.zero(qd.f64,3)
            for d in qd.static(range(3)):delta[d] = qd.cast(com_native[d],qd.f64)-qd.cast(state.links.pos[link,0][d],qd.f64)
            q = qd.Vector.zero(qd.f64,4)
            for d in qd.static(range(4)):q[d] = qd.cast(state.links.quat[link,0][d],qd.f64)
            norm = qd.sqrt(q.dot(q))
            if not finite(norm) or qd.abs(norm-1.0)>qd.f64(1e-4) or norm<=0.0:
                flag(output,2,step)
                norm = 1.0
            q = q/norm
            w,x,y,z = q[0],q[1],q[2],q[3]
            R = qd.Matrix([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                           [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                           [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]],dt=qd.f64)
            local_com = R.transpose()@delta
            changed = qd.abs(mass-model[local,0])>qd.f64(1e-8)+qd.f64(1e-6)*qd.abs(model[local,0])
            change2 = qd.f64(0.0)
            for d in qd.static(range(3)):
                change2 += (local_com[d]-model[local,d+1])**2
                if not finite(local_com[d]):flag(output,0,step)
                points[local,d] = qd.cast(local_com[d],gs.qd_float)
            if changed or qd.sqrt(change2)>qd.f64(1e-5):flag(output,1,step)
            if not finite(mass) or mass<0.0:flag(output,0,step)
            masses[local] = mass

    @qd.kernel
    def write(arm:qd.template(), jac:qd.types.ndarray(), masses:qd.types.ndarray(),
              reference:qd.types.ndarray(), native:qd.types.ndarray(), parameters:qd.types.ndarray(),
              gravity:qd.types.ndarray(), output:qd.types.ndarray(), index:qd.i32, offset:qd.i32,
              global_step:qd.i32, state:arrays.DynState, info:arrays.DynInfo, config:qd.template()):
        for d in range(arm.n_dofs):
            column = offset+d;dof = arm._dof_start+d
            I = [dof,0] if qd.static(config.batch_dofs_info) else dof
            tau = qd.f64(0.0)
            # The link loop is serial, matching the original accumulation order.
            for link in range(arm.n_links):
                part = qd.f64(0.0)
                for axis in qd.static(range(3)):
                    part += qd.cast(jac[link,axis,d,0],qd.f64)*(masses[link]*gravity[axis])
                tau -= part
            if not finite(tau):flag(output,0,global_step)
            kp,kv,effort = parameters[column,0],parameters[column,1],parameters[column,2]
            q = qd.cast(state.dofs.pos[dof,0],qd.f64);v = qd.cast(state.dofs.vel[dof,0],qd.f64)
            requested = tau+kp*(reference[index,column,0]-q)+kv*(reference[index,column,1]-v)
            state.dofs.ctrl_mode[dof,0] = gs.CTRL_MODE.POSITION
            state.dofs.ctrl_pos[dof,0] = native[index,column,0]
            state.dofs.ctrl_vel[dof,0] = native[index,column,1]
            info.dofs.act_bias[I][0] = qd.cast(tau,gs.qd_float)
            info.dofs.act_bias[I][1] = -qd.cast(kp,gs.qd_float)
            info.dofs.act_bias[I][2] = -qd.cast(kv,gs.qd_float)
            motor = (info.dofs.act_gain[I]*(state.dofs.ctrl_pos[dof,0]-state.dofs.pos[dof,0])
                +info.dofs.act_bias[I][0]+(info.dofs.act_gain[I]+info.dofs.act_bias[I][1])*state.dofs.pos[dof,0]
                +info.dofs.act_bias[I][2]*(state.dofs.vel[dof,0]-state.dofs.ctrl_vel[dof,0]))
            motor = qd.math.clamp(motor,info.dofs.force_range[I][0],info.dofs.force_range[I][1])
            if not finite(q) or not finite(v) or not finite(requested) or not finite(motor):flag(output,10,global_step)
            if qd.abs(qd.cast(motor,qd.f64)-qd.math.clamp(requested,-effort,effort))>qd.f64(.005):flag(output,3,global_step)
            if qd.abs(qd.cast(motor,qd.f64))>effort+qd.f64(1e-5):flag(output,4,global_step)
            if qd.abs(requested)>effort:flag(output,5,global_step)
            base = HEADER+(index*parameters.shape[0]+column)*WIDTH
            output[base+0]=tau;output[base+1]=requested;output[base+2]=qd.cast(motor,qd.f64)
            output[base+3]=q;output[base+4]=v
            output[base+5]=reference[index,column,0];output[base+6]=reference[index,column,1]
            output[base+10]=qd.cast(state.dofs.ctrl_pos[dof,0],qd.f64)
            output[base+11]=qd.cast(state.dofs.ctrl_vel[dof,0],qd.f64)
            output[base+12]=qd.cast(info.dofs.act_bias[I][0],qd.f64)

    @qd.kernel
    def post(arm:qd.template(), parameters:qd.types.ndarray(), native:qd.types.ndarray(), output:qd.types.ndarray(),
             workspace:qd.types.ndarray(), hand_link:qd.i32,
             index:qd.i32, offset:qd.i32, global_step:qd.i32, state:arrays.DynState):
        for d in range(arm.n_dofs):
            column=offset+d;dof=arm._dof_start+d;base=HEADER+(index*parameters.shape[0]+column)*WIDTH
            q=qd.cast(state.dofs.pos[dof,0],qd.f64);v=qd.cast(state.dofs.vel[dof,0],qd.f64)
            acceleration=(v-output[base+4])/qd.f64(.0025)
            output[base+7]=q;output[base+8]=v;output[base+9]=acceleration
            if not finite(q) or not finite(v) or not finite(acceleration):flag(output,10,global_step)
            if q<parameters[column,3]-qd.f64(1e-5) or q>parameters[column,4]+qd.f64(1e-5):flag(output,6,global_step)
            if d<7:
                if qd.abs(v)>parameters[column,5]+qd.f64(1e-6):flag(output,7,global_step)
                if qd.abs(acceleration)>parameters[column,6]+qd.f64(1e-5):flag(output,8,global_step)
                if qd.abs(q-output[base+5])>parameters[column,7]:flag(output,9,global_step)
            if (state.dofs.ctrl_mode[dof,0]!=gs.CTRL_MODE.POSITION
                or state.dofs.ctrl_pos[dof,0]!=native[index,column,0]
                or state.dofs.ctrl_vel[dof,0]!=native[index,column,1]):flag(output,11,global_step)
            if d==0:
                for axis in qd.static(range(3)):
                    position=qd.cast(state.links.pos[hand_link,0][axis],qd.f64)
                    if not finite(position):flag(output,10,global_step)
                    if position<workspace[axis,0] or position>workspace[axis,1]:flag(output,12,global_step)

    _KERNELS[key] = (prepare,write,post)
    return _KERNELS[key]


class DeviceCarryControl:
    def __init__(self,scene,arms,gravity,limits,commands,*,kp,kv,effort,finger_bounds,tracking_error_rad,hands,tool_workspace_m,dt_s=DT):
        from physics.gravity_feedforward import GravityFeedforward
        from physics.gravity_jacobian_batch import NativeLinkJacobianBatch
        import torch
        import quadrants as qd
        import genesis as gs
        from genesis.utils import array_class as arrays
        names=tuple(sorted(arms))
        single=getattr(scene,'_ap_single_active_arm_contract',None)
        if names!=('arm_0','arm_1','arm_2'):
            from physics.fixed_peer_arms import validate_scene_contract
            single=validate_scene_contract(scene,arms)
            require(type(single) is dict
                    and single.get('schema_version')=='single-active-arm-fixed-peers-v1'
                    and names==(single.get('active_arm_id'),)
                    and set(single.get('frozen_arm_ids',()))=={'arm_0','arm_1','arm_2'}-set(names)
                    and single.get('geometry_verified') is True
                    and scene.rigid_solver.n_dofs==39,
                    'One controlled arm requires verified fixed-peer geometry and actual reduced native DOFs')
        else:
            require(single is None, 'A single-active scene cannot claim three dynamic controller histories')
        require(dt_s==DT, 'Use the original physical time interval')
        require(gs.backend==gs.cuda and gs.use_zerocopy and gs.device.type=='cuda', 'Deferred control requires original CUDA zero-copy')
        self.scene=scene;self.solver=scene.rigid_solver;self.arms=dict(arms);self.gravity=dict(gravity)
        require(not self.solver._use_hibernation, 'First deferred control prototype requires the unchanged non-hibernating scene')
        self.names=names;self.controlled_dofs=9*len(names);self.qd=qd;self.torch=torch;self.gs=gs
        self.single_active_contract=copy.deepcopy(single)
        self.owner=threading.get_ident();self.start_step=int(scene.t);self.applies=0;self.posts=0;self.closed=False
        self.state=self.solver.dyn_state;self.info=self.solver.dyn_info;self.config=self.solver.rigid_config
        self.dtype=gs.tc_float;self.device=gs.device
        workspace=numeric(tool_workspace_m,(3,2),'original tool workspace')
        require(np.all(workspace[:,0]<workspace[:,1]) and set(hands)==set(self.names), 'Exact supplied workspace/hand identities required')
        self.hand_indices={name:int(hands[name].idx) for name in self.names}
        require(all(hands[name].entity is arms[name] for name in self.names), 'Tool links must belong to the corresponding persistent arms')
        self.workspace=torch.as_tensor(workspace,dtype=torch.float64,device=self.device)
        self.pins=source_pins(Path(gs.__file__).resolve().parent)
        self.pins[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if single is not None:
            p=Path(__file__).with_name('fixed_peer_arms.py').resolve()
            self.pins[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        reference,native=command_tables(commands,self.names,native_dtype=np.float32 if self.dtype==torch.float32 else np.float64)
        self.steps=len(reference);self.reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest()
        self.native_commands_sha256=hashlib.sha256(native.tobytes()).hexdigest()
        kp=numeric(kp,(9,),'kp');kv=numeric(kv,(9,),'kv');effort=numeric(effort,(9,),'effort')
        require((kp>=0).all() and (kv>=0).all() and (effort>0).all() and type(tracking_error_rad) in (int,float)
                and math.isfinite(tracking_error_rad) and tracking_error_rad>0, 'Original finite controller bounds required')
        parameters=[];self.backends={};self.models={};self.points={};self.masses={};self.g_vectors={}
        for name in self.names:
            arm=arms[name];feed=gravity[name];backend=feed._fused_backend
            require(type(feed) is GravityFeedforward and feed._arm is arm and feed._readback=='fused'
                    and type(backend) is NativeLinkJacobianBatch and arm.n_dofs==9
                    and arm.scene is scene and arm._solver is self.solver, 'Use exact bound fused gravity owners')
            backend._context();self.backends[name]=backend
            model=np.column_stack((feed._masses,feed._local_coms));require(model.shape==(arm.n_links,4), 'Gravity model shape changed')
            self.models[name]=torch.as_tensor(model,dtype=torch.float64,device=self.device)
            self.points[name]=torch.empty((arm.n_links,3),dtype=self.dtype,device=self.device)
            self.masses[name]=torch.empty(arm.n_links,dtype=torch.float64,device=self.device)
            self.g_vectors[name]=torch.as_tensor(feed._gravity,dtype=torch.float64,device=self.device)
            bound=limits[name];position=numeric([*bound.position_rad,*finger_bounds[name]],(9,2),'native limits')
            velocity=numeric(bound.velocity_rad_s,(7,),'velocity limits');acceleration=numeric(bound.acceleration_rad_s2,(7,),'acceleration limits')
            for d in range(9):parameters.append([kp[d],kv[d],effort[d],*position[d],velocity[d] if d<7 else 0.,acceleration[d] if d<7 else 0.,tracking_error_rad])
        parameters=np.asarray(parameters)
        require(np.all(reference[:,:,0]>=parameters[:,3]-1e-12) and np.all(reference[:,:,0]<=parameters[:,4]+1e-12), 'Saved target leaves provided position bounds')
        self.parameters=torch.as_tensor(parameters,dtype=torch.float64,device=self.device)
        self.reference=torch.as_tensor(reference,dtype=torch.float64,device=self.device)
        self.native=torch.as_tensor(native,dtype=self.dtype,device=self.device)
        initial=np.zeros(HEADER+self.steps*self.controlled_dofs*WIDTH,dtype=np.float64);initial[FIRST_FAILURE_SLOT]=self.start_step+self.steps+1
        self.output=torch.as_tensor(initial,dtype=torch.float64,device=self.device)
        self.prepare_kernel,self.write_kernel,self.post_kernel=_kernels(qd,gs,arrays)
        qd.sync()  # Setup-only boundary, never the advancing control path.
        self._context()

    def _context(self):
        require(not self.closed and threading.get_ident()==self.owner and self.scene.rigid_solver is self.solver
                and self.solver.dyn_state is self.state and self.solver.dyn_info is self.info
                and self.scene.n_envs==0 and self.solver._B==1, 'Deferred control owner/scene changed')
        require(getattr(self.scene,'_ap_single_active_arm_contract',None)==self.single_active_contract,
                'Fixed-peer scene binding changed during dynamic control')
        for backend in self.backends.values():backend._context()

    def apply(self,step_index):
        self._context()
        require(type(step_index) is int and step_index==self.applies==self.posts and step_index<self.steps
                and int(self.scene.t)==self.start_step+step_index, 'Apply each exact command before one physical interval')
        self.solver.update_forward_pos()
        for index,name in enumerate(self.names):
            arm=self.arms[name];backend=self.backends[name]
            self.prepare_kernel(arm,self.models[name],self.points[name],self.masses[name],self.output,
                self.start_step+step_index+1,self.state,self.info,self.config)
            backend._kernel(arm,self.points[name],backend._output,self.state,self.info)
            self.write_kernel(arm,backend._output,self.masses[name],self.reference,self.native,self.parameters,
                self.g_vectors[name],self.output,step_index,9*index,self.start_step+step_index+1,self.state,self.info,self.config)
        self.applies+=1

    def capture_post(self,step_index):
        self._context()
        require(type(step_index) is int and step_index==self.posts and self.applies==step_index+1
                and int(self.scene.t)==self.start_step+step_index+1, 'Capture exactly one actual completed physical interval')
        for index,name in enumerate(self.names):
            self.post_kernel(self.arms[name],self.parameters,self.native,self.output,self.workspace,self.hand_indices[name],step_index,9*index,
                self.start_step+step_index+1,self.state)
        self.posts+=1

    def finish(self):
        self._context()
        self.qd.sync()  # The single final device-completion boundary.
        raw=self.output.detach().cpu().numpy().copy()
        unchanged=all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in self.pins.items())
        flags={name:bool(raw[i]) for i,name in enumerate(FLAGS)}
        self.closed=True
        return dict(schema_version='provided-deferred-device-control-v1',source_bindings_unchanged=unchanged,
            passed=bool(unchanged and not any(flags.values()) and self.applies==self.posts==self.steps),
            applied=self.applies,completed_post_captures=self.posts,expected_steps=self.steps,
            start_native_step=self.start_step,end_native_step=int(self.scene.t),flags=flags,
            first_failed_step=int(raw[FIRST_FAILURE_SLOT]) if any(flags.values()) else None,
            history=raw[HEADER:].reshape(self.steps,self.controlled_dofs,WIDTH),history_channels=list(CHANNELS),arm_order=list(self.names),
            single_active_arm_contract=copy.deepcopy(self.single_active_contract),controlled_dof_count=self.controlled_dofs,
            reference_commands_sha256=self.reference_sha256,native_commands_sha256=self.native_commands_sha256,
            final_host_payload_copies=1,per_step_host_state_copies=0,per_step_explicit_syncs=0,
            source_hashes=copy.deepcopy(self.pins),controller='Original outer-cadence serial-link Jacobian gravity plus native bounded PD',
            bitwise_CPU_GPU_gravity_parity_claimed=False,physical_transport_passed=False,full_mission_passed=False)

    readback_once=finish
