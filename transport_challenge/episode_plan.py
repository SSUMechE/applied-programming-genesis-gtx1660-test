"""Pure serial command contract for externally planned two/three-arm episodes.

No solver, planner, IK, scene, material, ownership or contact action is called.
Every approach/carry/retreat trajectory is supplied by the external caller.
The module schedules only the caller's ordered actions and gripper commands.

Planned command continuity is checked here. Actual physical continuity is a
separate responsibility of trusted instructor integration. Its boundary hook
must inspect the latest measured state and declared tolerances. A boolean ack
is an integration assertion, not physical evidence or a student acceptance API.
If actual state fails that check, stop and externally plan a continuation from
the observation. This module never resets, clips, teleports or inserts a path.
"""
from dataclasses import dataclass
import json
import math

from .contracts import ContractError, digest, number, point
from .joint_planning import JointLimits, JointTrajectory


def _name(value, label):
    if type(value) is not str or not value or value.strip()!=value:
        raise ContractError(label+' must be an explicit nonempty name')
    return value


def _hash(value, label):
    if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ContractError(label+' must be a lowercase SHA256')
    return value


def _fields(record, keys, label):
    if not isinstance(record,dict) or set(record)!=set(keys):
        raise ContractError(label+' fields must be explicit and exact')


def _detached(record):
    try:return json.loads(json.dumps(record,allow_nan=False,sort_keys=True))
    except (TypeError,ValueError,OverflowError,RecursionError) as exc:
        raise ContractError('Episode records must be finite JSON data') from exc


def _grid(seconds, dt, label):
    seconds=number(seconds,label,minimum=0.)
    ratio=seconds/dt
    if seconds<=0 or not math.isfinite(ratio) or not 1<=ratio<=2**53-1 or abs(ratio-round(ratio))>1e-8:
        raise ContractError(label+' must be positive and lie on the dt grid')
    return round(ratio)


@dataclass(frozen=True)
class EpisodeArm:
    arm_id: str
    role: str
    initial_q_rad: tuple
    initial_finger_m: tuple
    finger_bounds_m: tuple
    limits: JointLimits

    def to_record(self):
        return dict(arm_id=self.arm_id,role=self.role,initial_q_rad=list(self.initial_q_rad),
            initial_finger_m=list(self.initial_finger_m),finger_bounds_m=[list(x) for x in self.finger_bounds_m],joint_limits=self.limits.to_record())


@dataclass(frozen=True)
class EpisodeAction:
    action_id: str
    arm_id: str
    role: str
    kind: str
    duration_s: float
    trajectory: object
    target_finger_m: object
    start_step: int
    steps: int
    start_q_rad: tuple
    start_finger_m: tuple
    preload_ramp_s: object = None
    instructor_only: bool = False

    def to_record(self):
        record=dict(action_id=self.action_id,arm_id=self.arm_id,role=self.role,kind=self.kind)
        if self.kind=='trajectory':record['trajectory']=self.trajectory.to_record()
        else:
            record['duration_s']=self.duration_s
            if self.kind in ('close','open','supported_preload'):record['target_finger_m']=list(self.target_finger_m)
            if self.kind=='supported_preload':
                record.update(ramp_duration_s=self.preload_ramp_s,instructor_only=True)
        return record


@dataclass(frozen=True,init=False)
class EpisodePlan:
    """Validated external serial actions. Use from_record, not direct construction.

Required top-level fields: schema_version='serial-joint-episode-plan-v1',
episode_id, initial_state_sha256, dt_s, deadline_s, arms and actions.
arms has two or three unique arm_id records with role, initial_q_rad[7],
initial_finger_m[2], finger_bounds_m[2][2] and joint_limits. limits_by_arm must supply those exact
IDs and seven native-q JointLimits per arm. Initial state SHA binds an actual
snapshot owned by the caller, not a snapshot manufactured by this module.

Every action has unique action_id, arm_id, role and kind. Kind trajectory needs
the complete canonical JointTrajectory record, with limits matching that arm.
Kind close/open needs positive duration_s and target_finger_m[2]. Kind hold
needs duration_s only. Names do not impose a relay order or confer ownership.
The two Panda fingers are mechanically coupled, not independent choices.
Both commanded target openings must be equal. The two initial measured finger
positions may differ and remain recorded, with the ramp beginning at each.

Positive trajectory duration occupies ceil(duration/dt) intervals. The last
sample is its exact endpoint and its fractional tail is disclosed. No extra
waypoint is generated. A zero-duration path must be omitted or represented by
an explicit hold. Close/open/hold durations and the deadline lie on the dt
grid. The deadline may truncate an action and never means plan completion.

The explicit external-grid-joint-episode-plan-v1 schema instead requires every
edge on the dt grid and schedules that exact integer count. It avoids binary
floating point ceil adding an interval to a declared grid-aligned duration.
"""
    episode_id: str
    initial_state_sha256: str
    dt_s: float
    deadline_s: float
    deadline_steps: int
    arms: tuple
    actions: tuple

    @classmethod
    def from_record(cls,record,*,limits_by_arm):
        try:return cls._parse(_detached(record),limits_by_arm)
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            if isinstance(exc,ContractError):raise
            raise ContractError('Invalid external episode: '+str(exc)) from exc

    @classmethod
    def _parse(cls,r,limits_by_arm):
        _fields(r,('schema_version','episode_id','initial_state_sha256','dt_s','deadline_s','arms','actions'),'Episode')
        single_active=r['schema_version'] in ('serial-single-active-joint-episode-plan-v1','external-grid-single-active-joint-episode-plan-v1')
        if r['schema_version'] not in ('serial-joint-episode-plan-v1','provided-instructor-supported-preload-plan-v1','external-grid-joint-episode-plan-v1','serial-single-active-joint-episode-plan-v1','external-grid-single-active-joint-episode-plan-v1'):raise ContractError('Unknown episode schema')
        external_grid=r['schema_version'] in ('external-grid-joint-episode-plan-v1','external-grid-single-active-joint-episode-plan-v1')
        episode=_name(r['episode_id'],'episode_id');initial_hash=_hash(r['initial_state_sha256'],'initial_state_sha256')
        dt=number(r['dt_s'],'dt_s',minimum=0.)
        if dt<=0:raise ContractError('dt_s must be positive')
        deadline_steps=_grid(r['deadline_s'],dt,'deadline_s')
        if (not isinstance(r['arms'],list) or (len(r['arms'])!=1 if single_active else len(r['arms']) not in (2,3))):raise ContractError('Declare one active arm only in the explicit single-active schema, otherwise two or three physical arms')
        if single_active and (not isinstance(r['arms'][0],dict) or r['arms'][0].get('arm_id')!='arm_0'):raise ContractError('Single-active operation selects arm_0 explicitly')
        if not isinstance(limits_by_arm,dict):raise ContractError('Provide native limits by exact arm ID')
        arms=[];by_id={};q_by_id={};f_by_id={}
        for entry in r['arms']:
            _fields(entry,('arm_id','role','initial_q_rad','initial_finger_m','finger_bounds_m','joint_limits'),'Arm')
            name=_name(entry['arm_id'],'arm_id');role=_name(entry['role'],'arm role')
            if name in by_id:raise ContractError('Duplicate physical arm ID')
            limits=limits_by_arm.get(name)
            if not isinstance(limits,JointLimits) or limits.dimension!=7:raise ContractError('Every physical arm needs seven native limits')
            if entry['joint_limits']!=limits.to_record():raise ContractError('Declared native limits differ from supplied arm limits')
            q=point(entry['initial_q_rad'],7);f=point(entry['initial_finger_m'],2)
            if any(not lo<=x<=hi for x,(lo,hi) in zip(q,limits.position_rad)):raise ContractError('Initial arm command outside native bounds')
            if not isinstance(entry['finger_bounds_m'],list) or len(entry['finger_bounds_m'])!=2:raise ContractError('Two native finger intervals required')
            fb=tuple(point(x,2) for x in entry['finger_bounds_m'])
            if any(not 0<=lo<hi or not lo<=x<=hi for x,(lo,hi) in zip(f,fb)):raise ContractError('Invalid native finger bounds or initial command')
            arm=EpisodeArm(name,role,q,f,fb,limits);arms.append(arm);by_id[name]=arm;q_by_id[name]=q;f_by_id[name]=f
        if set(limits_by_arm)!=set(by_id):raise ContractError('Native-limit IDs must exactly match the declared arms')
        if not isinstance(r['actions'],list) or not r['actions']:raise ContractError('External action sequence must be nonempty')
        actions=[];ids=set();elapsed=0
        for entry in r['actions']:
            kind=entry.get('kind') if isinstance(entry,dict) else None
            common=('action_id','arm_id','role','kind')
            extra={'trajectory':('trajectory',),'close':('duration_s','target_finger_m'),'open':('duration_s','target_finger_m'),'hold':('duration_s',),'supported_preload':('duration_s','target_finger_m','ramp_duration_s','instructor_only')}
            if kind not in extra:raise ContractError('Use trajectory, close, open or hold actions')
            _fields(entry,common+extra[kind],'Action')
            aid=_name(entry['action_id'],'action_id');arm_id=_name(entry['arm_id'],'action arm_id');role=_name(entry['role'],'action role')
            if aid in ids or arm_id not in by_id:raise ContractError('Action ID duplicated or arm unknown')
            ids.add(aid);arm=by_id[arm_id];start_q=q_by_id[arm_id];start_f=f_by_id[arm_id];trajectory=None;target=None;ramp=None;instructor=False
            if kind=='trajectory':
                t=entry['trajectory'];trajectory=JointTrajectory(t['waypoints'],t['edge_durations_s'],arm.limits)
                if trajectory.to_record()!=t:raise ContractError('External trajectory must be the complete canonical record with exact native limits')
                if trajectory.waypoints[0]!=start_q:raise ContractError('Next trajectory must begin at this arm\'s preceding commanded endpoint')
                duration=trajectory.duration_s
                if duration<=0:raise ContractError('Omit a zero-duration path or supply an explicit positive hold')
                ratio=duration/dt
                if not math.isfinite(ratio) or not 0<ratio<=2**53-1:raise ContractError('Action duration is not representable on this grid')
                if external_grid:
                    for edge_duration in trajectory.edge_durations_s:_grid(edge_duration,dt,'external edge duration')
                    steps=_grid(duration,dt,'external trajectory duration')
                else:steps=math.ceil(ratio)
                q_by_id[arm_id]=trajectory.waypoints[-1]
            else:
                steps=_grid(entry['duration_s'],dt,'action duration');duration=number(entry['duration_s'],'action duration')
                if kind in ('close','open','supported_preload'):
                    target=point(entry['target_finger_m'],2)
                    if target[0]!=target[1]:raise ContractError('Coupled Panda fingers require one equal commanded target opening')
                    if any(not lo<=x<=hi for x,(lo,hi) in zip(target,arm.finger_bounds_m)):raise ContractError('Gripper target outside native bounds')
                    if kind=='supported_preload':
                        ramp=number(entry['ramp_duration_s'],'preload ramp duration',minimum=0.)
                        _grid(ramp,dt,'preload ramp duration')
                        instructor=entry['instructor_only'] is True
                        if (not instructor or role!='place' or r['schema_version']!='provided-instructor-supported-preload-plan-v1'
                                or start_f!=(.010,.010) or target!=(.015,.015) or ramp!=2.5 or duration!=3.5):
                            raise ContractError('Provided instructor preload is exactly .010 to .015, 2.5s ramp plus1s settle at constant7q')
                    if kind=='close' and any(b>a for a,b in zip(start_f,target)):raise ContractError('Close may not increase either finger opening')
                    if kind=='open' and any(b<a for a,b in zip(start_f,target)):raise ContractError('Open may not decrease either finger opening')
                    if target==start_f:raise ContractError('Use hold for unchanged gripper commands')
                    f_by_id[arm_id]=target
            if elapsed+steps>2**53-1 or not math.isfinite((elapsed+steps)*dt):raise ContractError('Episode time is not representable on this grid')
            actions.append(EpisodeAction(aid,arm_id,role,kind,duration,trajectory,target,elapsed,steps,start_q,start_f,ramp,instructor));elapsed+=steps
        if (r['schema_version']=='provided-instructor-supported-preload-plan-v1')!=any(a.kind=='supported_preload' for a in actions):
            raise ContractError('The explicit instructor schema is required exactly for instructor supported-preload actions')
        instance=object.__new__(cls)
        object.__setattr__(instance,'_external_grid',external_grid)
        object.__setattr__(instance,'_single_active',single_active)
        for k,v in dict(episode_id=episode,initial_state_sha256=initial_hash,dt_s=dt,deadline_s=number(r['deadline_s'],'deadline_s'),
                        deadline_steps=deadline_steps,arms=tuple(arms),actions=tuple(actions)).items():object.__setattr__(instance,k,v)
        return instance

    def to_record(self):
        schema=('external-grid-single-active-joint-episode-plan-v1' if getattr(self,'_external_grid',False) else 'serial-single-active-joint-episode-plan-v1') if getattr(self,'_single_active',False) else ('external-grid-joint-episode-plan-v1' if getattr(self,'_external_grid',False) else 'provided-instructor-supported-preload-plan-v1' if any(a.kind=='supported_preload' for a in self.actions) else 'serial-joint-episode-plan-v1')
        return dict(schema_version=schema,episode_id=self.episode_id,initial_state_sha256=self.initial_state_sha256,
                    dt_s=self.dt_s,deadline_s=self.deadline_s,arms=[a.to_record() for a in self.arms],actions=[a.to_record() for a in self.actions])

    @property
    def sha256(self):return digest(self.to_record())

    @property
    def full_steps(self):return sum(a.steps for a in self.actions)

    @property
    def scheduled_steps(self):return min(self.full_steps,self.deadline_steps)

    def timing_record(self):
        return dict(full_steps=self.full_steps,scheduled_steps=self.scheduled_steps,deadline_steps=self.deadline_steps,
            deadline_truncates_plan=self.full_steps>self.deadline_steps,actions=[dict(action_id=a.action_id,arm_id=a.arm_id,role=a.role,
            start_step=a.start_step,end_step=a.start_step+a.steps,start_s=a.start_step*self.dt_s,end_s=(a.start_step+a.steps)*self.dt_s,
            external_duration_s=a.duration_s,grid_tail_s=max(0.,a.steps*self.dt_s-a.duration_s)) for a in self.actions])


@dataclass(frozen=True)
class ArmCommand:
    arm_id: str
    role: str
    q_rad: tuple
    qd_rad_s: tuple
    qdd_rad_s2: tuple
    finger_q_m: tuple
    finger_qd_m_s: tuple

    def to_record(self):
        return dict(arm_id=self.arm_id,role=self.role,q_rad=list(self.q_rad),qd_rad_s=list(self.qd_rad_s),
                    qdd_rad_s2=list(self.qdd_rad_s2),finger_q_m=list(self.finger_q_m),finger_qd_m_s=list(self.finger_qd_m_s))


@dataclass(frozen=True)
class EpisodeCommand:
    step: int
    pre_time_s: float
    post_time_s: float
    action_id: str
    action_role: str
    active_arm_id: str
    kind: str
    arms: tuple

    def to_record(self):
        return dict(step=self.step,pre_time_s=self.pre_time_s,post_time_s=self.post_time_s,action_id=self.action_id,
            action_role=self.action_role,active_arm_id=self.active_arm_id,kind=self.kind,arms=[a.to_record() for a in self.arms])


class EpisodeDispatcher:
    """Two-phase pure dispatch. Preparing never advances actual-observation time.

prepare_next(verify_boundary=trusted_callback) returns the pending command,
or None after every requested interval or the declared deadline. Before each
external action, callback receives the detached boundary_request(). It must
verify ALL current arms/source bindings using the last real observation,
including declared measured endpoint tolerances. Motion, close and open require
measured rest. An explicit hold may begin with measured_rest_verified=False
so residual motion can settle while every q/finger command stays unchanged.
The false rest assertion is retained. Hold does not waive the measured-state
or source checks, and the next motion/open/close again requires rest.
The callback returns the exact ack
schema below. A rejected hook leaves the clock unchanged and issues no command.

Ack fields: schema_version='episode-boundary-ack-v1', plan_sha256,
initial_state_sha256, action_id, at_step, simulation_s,
measured_state_sha256, verified_arm_ids, actual_state_verified,
measured_rest_verified, source_bindings_verified. The hash must equal the last
committed physical observation (initial snapshot for the first action).

After actual integration, commit_step requires the pending step, observed
simulation_s, observation_sha256 and physical_step_completed=True. Never pass
predicted timestamps or a digest of only commanded values as measurements.
This pure module cannot authenticate a dishonest integration assertion.
Missing/interrupted steps must not be committed. Completion is COMMAND/OBSERVATION
SEQUENCE completion only, never material delivery, handoff or physical safety.
"""
    def __init__(self,plan):
        if type(plan) is not EpisodePlan:raise ContractError('Use a validated EpisodePlan')
        self._plan=plan;self._plan_hash=plan.sha256;self._step=0;self._action_index=0;self._pending=None;self._acked=None;self._observed_time=0.
        self._last_observation=plan.initial_state_sha256;self._acks=[]
        self._q={a.arm_id:a.initial_q_rad for a in plan.arms};self._f={a.arm_id:a.initial_finger_m for a in plan.arms}

    @property
    def committed_steps(self):return self._step

    @property
    def plan(self):return self._plan

    def _action(self):
        if self._step>=self.plan.scheduled_steps:return None
        return self.plan.actions[self._action_index]

    def boundary_request(self):
        action=self._action()
        if action is None or self._acked==self._action_index:return None
        return dict(schema_version='episode-boundary-request-v1',plan_sha256=self._plan_hash,
            initial_state_sha256=self.plan.initial_state_sha256,action_id=action.action_id,action_role=action.role,
            active_arm_id=action.arm_id,action_kind=action.kind,measured_rest_required=action.kind not in ('hold','supported_preload'),
            at_step=self._step,simulation_s=self._observed_time,command_grid_time_s=self._step*self.plan.dt_s,
            measured_state_sha256=self._last_observation,expected_arm_commands=[dict(arm_id=a.arm_id,q_rad=list(self._q[a.arm_id]),
            finger_q_m=list(self._f[a.arm_id])) for a in self.plan.arms],
            meaning='Expected command endpoints only. Trusted runtime must independently verify actual state, source bindings and the stated rest requirement.')

    def _acknowledge(self,callback):
        request=self.boundary_request()
        if request is None:return
        if not callable(callback):raise ContractError('A trusted measured-state boundary verifier is mandatory')
        # Compare against the internal request, never the callback-owned copy.
        ack=_detached(callback(_detached(request)))
        fields=('schema_version','plan_sha256','initial_state_sha256','action_id','at_step','simulation_s','measured_state_sha256',
                'verified_arm_ids','actual_state_verified','measured_rest_verified','source_bindings_verified')
        _fields(ack,fields,'Measured boundary acknowledgement')
        if ack['schema_version']!='episode-boundary-ack-v1':raise ContractError('Unknown measured boundary ack')
        for key in ('plan_sha256','initial_state_sha256','action_id','measured_state_sha256'):
            if ack[key]!=request[key]:raise ContractError('Boundary identity differs: '+key)
        if type(ack['at_step']) is not int or ack['at_step']!=self._step or number(ack['simulation_s'],'observed boundary time')!=request['simulation_s']:
            raise ContractError('Boundary acknowledgement uses a stale or predicted clock')
        if (not isinstance(ack['verified_arm_ids'],list) or any(type(v) is not str for v in ack['verified_arm_ids'])
                or sorted(ack['verified_arm_ids'])!=sorted(self._q)):raise ContractError('Verify every arm exactly once at each boundary')
        if (any(ack[key] is not True for key in ('actual_state_verified','source_bindings_verified'))
                or type(ack['measured_rest_verified']) is not bool
                or (request['measured_rest_required'] and not ack['measured_rest_verified'])):
            raise ContractError('Actual boundary failed. Require externally planned continuation, never a pose reset or inserted route')
        self._acks.append(ack);self._acked=self._action_index

    def prepare_next(self,*,verify_boundary=None):
        if self._pending is not None:return self._pending
        action=self._action()
        if action is None:return None
        self._acknowledge(verify_boundary)
        local_step=self._step-action.start_step+1;dt=self.plan.dt_s;zeros7=(0.,)*7;zeros2=(0.,)*2
        commands=[]
        for arm in self.plan.arms:
            q=self._q[arm.arm_id];f=self._f[arm.arm_id];qd=zeros7;qdd=zeros7;fv=zeros2
            if arm.arm_id==action.arm_id:
                if action.kind=='trajectory':
                    sample_time=(action.duration_s if getattr(self.plan,'_external_grid',False) and local_step==action.steps
                        else min(local_step*dt,action.duration_s))
                    sample=action.trajectory.sample(sample_time);q,qd,qdd=sample.q_rad,sample.qd_rad_s,sample.qdd_rad_s2
                elif action.kind in ('close','open'):
                    if local_step==action.steps:f=action.target_finger_m
                    else:
                        u=local_step/action.steps;blend=u*u*(3.-2.*u)
                        f=tuple(a+(b-a)*blend for a,b in zip(action.start_finger_m,action.target_finger_m))
                        fv=tuple((b-a)*6.*u*(1.-u)/action.duration_s for a,b in zip(action.start_finger_m,action.target_finger_m))
                elif action.kind=='supported_preload':
                    ramp_steps=round(action.preload_ramp_s/dt)
                    if local_step>=ramp_steps:f=action.target_finger_m
                    else:
                        u=local_step/ramp_steps;blend=u*u*(3.-2.*u)
                        f=tuple(a+(b-a)*blend for a,b in zip(action.start_finger_m,action.target_finger_m))
                        fv=tuple((b-a)*6.*u*(1.-u)/action.preload_ramp_s for a,b in zip(action.start_finger_m,action.target_finger_m))
            commands.append(ArmCommand(arm.arm_id,arm.role,q,qd,qdd,f,fv))
        self._pending=EpisodeCommand(self._step+1,self._step*dt,(self._step+1)*dt,action.action_id,action.role,action.arm_id,action.kind,tuple(commands))
        return self._pending

    def commit_step(self,*,step,simulation_s,observation_sha256,physical_step_completed):
        command=self._pending
        if command is None:raise ContractError('There is no issued interval to commit')
        if physical_step_completed is not True:raise ContractError('Interrupted physics interval cannot advance the episode')
        observed_time=number(simulation_s,'measured post-step time')
        if type(step) is not int or step!=command.step or abs(observed_time-command.post_time_s)>1e-9 or observed_time<=self._observed_time:
            raise ContractError('Commit must identify exactly the issued physical interval')
        checksum=_hash(observation_sha256,'measured observation SHA256')
        for arm in command.arms:self._q[arm.arm_id]=arm.q_rad;self._f[arm.arm_id]=arm.finger_q_m
        self._last_observation=checksum;self._observed_time=observed_time;self._step=command.step;self._pending=None
        action=self.plan.actions[self._action_index]
        if self._step==action.start_step+action.steps:self._action_index+=1;self._acked=None

    def record(self):
        completed=self._step==self.plan.full_steps
        deadline=self._step==self.plan.deadline_steps and not completed
        return dict(schema_version='serial-joint-episode-dispatch-v1',plan_sha256=self._plan_hash,
            initial_state_sha256=self.plan.initial_state_sha256,committed_steps=self._step,simulation_s=self._observed_time,command_grid_time_s=self._step*self.plan.dt_s,
            last_observation_sha256=self._last_observation,pending_step=self._pending.step if self._pending else None,
            command_sequence_completed=completed,deadline_reached=deadline,
            status='COMMAND_SEQUENCE_COMPLETED' if completed else 'DEADLINE_REACHED' if deadline else 'AWAITING_PHYSICAL_OBSERVATION' if self._pending else 'READY',
            boundary_acknowledgements=_detached(self._acks),physical_evidence_verified_by_dispatcher=False,
            material_delivery_accepted=False,handoff_accepted=False,continuous_collision_certificate=False,
            scope='Serial external command schedule plus trusted integration acknowledgements. No physical mission or gripper dynamics acceptance.')
