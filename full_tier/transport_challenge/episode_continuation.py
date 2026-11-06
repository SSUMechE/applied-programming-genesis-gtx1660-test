"""Supplied pure continuation dispatcher. Native evidence remains caller-owned.

EpisodePlan action times remain local. Every command, boundary acknowledgement
and commit exposed here uses the original episode's absolute native step/time.
The old dispatcher receives explicitly translated elapsed times internally.
No observation hash, command endpoint, or physical entity is synthesized.
"""
from dataclasses import dataclass
import copy
import math

from transport_challenge.contracts import ContractError, digest, number
from transport_challenge.episode_plan import EpisodePlan, EpisodeDispatcher


def _hash(value):
    if type(value) is not str or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError("Expected lowercase SHA256")
    return value


def _models(value, arms):
    if not isinstance(value,dict) or set(value)!={a.arm_id for a in arms}:
        raise ContractError("Model bindings must identify every exact arm")
    return tuple(sorted((name,_hash(checksum)) for name,checksum in value.items()))


def _fields(record, expected, label):
    if type(record) is not dict or set(record)!=set(expected):
        raise ContractError(label+" requires exactly the declared fields")


def _clock(value, expected, label):
    value=number(value,label,minimum=0.)
    if abs(value-expected)>1e-9:
        raise ContractError(label+" differs from committed native global clock")
    return value


@dataclass(frozen=True)
class SegmentEnvelope:
    plan: EpisodePlan
    index: int
    origin_step: int
    origin_time_s: float
    action_offset: int
    previous_segment_sha256: object
    boundary_observation_sha256: str
    models: tuple

    def to_record(self):
        return dict(schema_version="episode-continuation-segment-v1",
            index=self.index,origin_step=self.origin_step,origin_time_s=self.origin_time_s,
            action_offset=self.action_offset,previous_segment_sha256=self.previous_segment_sha256,
            boundary_observation_sha256=self.boundary_observation_sha256,
            model_sha256_by_arm=dict(self.models),plan=self.plan.to_record())

    @property
    def sha256(self):
        return digest(self.to_record())


@dataclass(frozen=True)
class GlobalEpisodeCommand:
    step: int
    pre_time_s: float
    post_time_s: float
    action_id: str
    action_role: str
    active_arm_id: str
    kind: str
    arms: tuple
    segment_sha256: str
    segment_index: int
    global_action_index: int

    def to_record(self):
        return dict(step=self.step,pre_time_s=self.pre_time_s,post_time_s=self.post_time_s,
            action_id=self.action_id,action_role=self.action_role,active_arm_id=self.active_arm_id,
            kind=self.kind,arms=[a.to_record() for a in self.arms],
            segment_sha256=self.segment_sha256,segment_index=self.segment_index,
            global_action_index=self.global_action_index)


class ContinuationDispatcher:
    """Supplied continuation dispatcher. Trusted runtime still owns physical proof.

    prepare_next(verify_boundary=callback) exposes a global boundary request.
    The callback returns the existing episode-boundary acknowledgement fields
    with schema_version='episode-continuation-boundary-ack-v1', plus exact
    segment_sha256, segment_index and global_action_index from that request.

    extend(next_plan, model_sha256_by_arm=..., boundary=...) requires an exact
    episode-continuation-rest-boundary-v1 receipt: previous_segment_sha256,
    step, simulation_s, observation_sha256, verified_arm_ids,
    actual_state_verified, measured_rest_verified, source_bindings_verified.
    All three assertions must be True. The receipt binds the latest committed
    actual observation. It is not a student self-certification or native proof.

    All segments use the same model IDs, arm IDs/roles, native limits, finger
    bounds and dt. The new plan's initial command equals every preceding
    commanded endpoint exactly, even when a contacted finger's measured value
    differs. The immutable plan initial-state hash binds the real observation.
    The fixed global deadline never restarts when another segment is added.
    """
    def __init__(self, initial_plan, *, model_sha256_by_arm, global_deadline_s):
        if type(initial_plan) is not EpisodePlan:
            raise ContractError("Use validated EpisodePlan")
        self._dt=initial_plan.dt_s
        deadline=number(global_deadline_s,"global deadline",minimum=0.)
        count=deadline/self._dt
        if (deadline<=0 or not math.isfinite(count) or not 1<=count<=2**53-1
                or abs(count-round(count))>1e-8):
            raise ContractError("Global deadline must be positive on the dt grid")
        self._deadline_steps=round(count)
        self._models=_models(model_sha256_by_arm,initial_plan.arms)
        self._arm_contract={a.arm_id:dict(role=a.role,finger_bounds=a.finger_bounds_m,
            limits=a.limits.to_record()) for a in initial_plan.arms}
        self._step=0;self._time=0.;self._last_hash=initial_plan.initial_state_sha256
        self._last_commands={a.arm_id:(a.initial_q_rad,a.initial_finger_m) for a in initial_plan.arms}
        self._segments=[SegmentEnvelope(initial_plan,0,0,0.,0,None,self._last_hash,self._models)]
        self._dispatcher=EpisodeDispatcher(initial_plan)
        self._pending=None;self._global_acks=[];self._extension_receipts=[]
        self._action_ids={a.action_id for a in initial_plan.actions}

    @property
    def current_segment(self):
        return self._segments[-1]

    def _action_index(self):
        n=self._dispatcher.committed_steps
        for i,a in enumerate(self.current_segment.plan.actions):
            if a.start_step<=n<a.start_step+a.steps:
                return i
        raise ContractError("No current segment action")

    def _global_request(self, local):
        s=self.current_segment
        return dict(local,schema_version="episode-continuation-boundary-request-v1",
            at_step=self._step,simulation_s=self._time,command_grid_time_s=self._step*self._dt,
            segment_sha256=s.sha256,segment_index=s.index,
            global_action_index=s.action_offset+self._action_index(),
            meaning="Absolute native episode clock. Internal local time is only elapsed segment time.")

    def boundary_request(self):
        if self._step>=self._deadline_steps:return None
        local=self._dispatcher.boundary_request()
        return None if local is None else self._global_request(local)

    def _verify(self, local, callback):
        if not callable(callback):raise ContractError("Trusted global boundary callback required")
        request=self._global_request(local)
        ack=copy.deepcopy(callback(copy.deepcopy(request)))
        core=("schema_version","plan_sha256","initial_state_sha256","action_id","at_step",
              "simulation_s","measured_state_sha256","verified_arm_ids","actual_state_verified",
              "measured_rest_verified","source_bindings_verified")
        extra=("segment_sha256","segment_index","global_action_index")
        _fields(ack,core+extra,"Global boundary acknowledgement")
        if ack["schema_version"]!="episode-continuation-boundary-ack-v1":
            raise ContractError("Unknown global boundary acknowledgement")
        for name in ("plan_sha256","initial_state_sha256","action_id","measured_state_sha256",*extra):
            if ack[name]!=request[name] or (name in ("segment_index","global_action_index") and type(ack[name]) is not int):
                raise ContractError("Global boundary identity changed: "+name)
        if type(ack["at_step"]) is not int or ack["at_step"]!=self._step:
            raise ContractError("Stale global boundary step")
        if number(ack["simulation_s"],"global boundary time",minimum=0.)!=self._time:
            raise ContractError("Global boundary time is not the exact committed observation")
        # Preserve real observation hashes. Only the explicitly local clock and
        # step are mapped for the existing internal dispatcher contract.
        translated={name:copy.deepcopy(ack[name]) for name in core}
        translated.update(schema_version="episode-boundary-ack-v1",at_step=local["at_step"],
                          simulation_s=local["simulation_s"])
        return translated,ack

    def prepare_next(self, *, verify_boundary=None):
        if self._pending is not None:return self._pending
        if self._step>=self._deadline_steps:return None
        receipts=[]
        def mapped(local):
            translated,ack=self._verify(local,verify_boundary)
            receipts.append(ack)
            return translated
        local=self._dispatcher.prepare_next(verify_boundary=mapped)
        if local is None:return None
        self._global_acks.extend(receipts)
        s=self.current_segment
        self._pending=GlobalEpisodeCommand(s.origin_step+local.step,
            (s.origin_step+local.step-1)*self._dt,(s.origin_step+local.step)*self._dt,
            local.action_id,local.action_role,local.active_arm_id,local.kind,local.arms,
            s.sha256,s.index,s.action_offset+self._action_index())
        return self._pending

    def commit_step(self, *, step, simulation_s, observation_sha256, physical_step_completed):
        command=self._pending
        if command is None:raise ContractError("No pending global command")
        if type(step) is not int or step!=command.step:
            raise ContractError("Commit must identify the issued global step")
        now=_clock(simulation_s,command.post_time_s,"global post-step time")
        if now<=self._time:raise ContractError("Global native time must advance")
        checksum=_hash(observation_sha256)
        s=self.current_segment
        self._dispatcher.commit_step(step=step-s.origin_step,
            simulation_s=now-s.origin_time_s,observation_sha256=checksum,
            physical_step_completed=physical_step_completed)
        self._step=step;self._time=now;self._last_hash=checksum
        self._last_commands={a.arm_id:(a.q_rad,a.finger_q_m) for a in command.arms}
        self._pending=None

    def accept_provided_grasp_event(self, before_observation, after_observation, event,
                                    source_bindings_verified=True, *, runtime_profile=None):
        """A trusted zero-step native weld event at a completed segment boundary.

        Only provided_grasp may change in the committed observation. The caller
        authenticates the exact helper/native scene and body/link ownership.
        This pure check cannot authenticate caller-fabricated native receipts.
        It never issues or commits a physics step or changes plan/command clocks.
        """
        if runtime_profile is not None and (type(runtime_profile) is not str
                or runtime_profile != "rigid16_bead_tc2h_v1"):
            raise ContractError("Unknown provided-grasp boundary runtime profile")
        if (self.record()["status"] != "SEGMENT_COMPLETED" or self._pending is not None
                or self._dispatcher._pending is not None or source_bindings_verified is not True):
            raise ContractError("Provided grasp event needs a source-verified completed segment boundary")
        before, after, native = copy.deepcopy((before_observation,after_observation,event))
        before_hash, after_hash, event_hash = digest(before),digest(after),digest(native)
        if (before_hash != self._last_hash or self._dispatcher._last_observation != self._last_hash
                or any(r["event_sha256"] == event_hash for r in getattr(self,"_native_boundary_events",()))):
            raise ContractError("Provided grasp event has a foreign/stale observation or duplicate receipt")
        for observation in (before,after):
            if (type(observation) is not dict or type(observation.get("native_scene_step")) is not int
                    or observation["native_scene_step"] != self._step
                    or number(observation.get("simulation_s"),"native event observation time",minimum=0.) != self._time
                    or not {"body_world","body_linear_velocity_m_s","body_angular_velocity_rad_s"} <= set(observation)
                    or set(observation.get("arms",{})) != set(self._arm_contract)):
                raise ContractError("Provided grasp event is not this exact committed native observation")
        physical_before={k:v for k,v in before.items() if k!="provided_grasp"}
        physical_after={k:v for k,v in after.items() if k!="provided_grasp"}
        if physical_before != physical_after or before_hash == after_hash:
            raise ContractError("Provided grasp event may change only the actual provided_grasp binding")
        if (type(native) is not dict or native.get("model") != "provided-ideal-native-weld-v1"
                or native.get("event") not in ("attach","release") or native.get("arm_id") not in self._arm_contract):
            raise ContractError("Exact provided helper model/event/arm required")
        arm,pair,old,new=native["arm_id"],native.get("pair"),native.get("before"),native.get("after")
        keys={"native_clock","body_entity_idx","cup_link_idx","hand_link_idx","cup_mass_kg","body_world","hand_world",
              "body_linear_velocity_m_s","body_angular_velocity_rad_s","hand_linear_velocity_m_s","hand_angular_velocity_rad_s","inventory"}
        if (type(pair) is not list or len(pair)!=2 or any(type(i) is not int or i<0 for i in pair) or pair[0]==pair[1]
                or type(old) is not dict or type(new) is not dict or set(old)!=keys or set(new)!=keys
                or {k:v for k,v in old.items() if k!="inventory"}!={k:v for k,v in new.items() if k!="inventory"}):
            raise ContractError("Native grasp event changed pose/velocity/mass/clock or omitted actual snapshot fields")
        if (type(old["body_entity_idx"]) is not int or old["body_entity_idx"]<0
                or any(type(old[k]) is not int or old[k]<0 for k in ("hand_link_idx","cup_link_idx"))
                or [old["hand_link_idx"],old["cup_link_idx"]]!=pair
                or number(old["cup_mass_kg"],"native cup mass",minimum=0.)<=0):
            raise ContractError("Native grasp hand/cup/body identity or mass differs")
        clock=old["native_clock"]
        allowed_global_steps = ((self._step,self._step*100) if runtime_profile is None
                                else (self._step*16,))
        if (type(clock) is not list or len(clock)!=4 or any(type(clock[i]) is not int for i in (0,2,3))
                or type(clock[1]) not in (int,float) or not math.isfinite(clock[1])
                or clock[:2]!=[self._step,self._time] or clock[3]!=0
                or clock[2] not in allowed_global_steps):
            raise ContractError("Native grasp event clock differs from the actual outer boundary")
        if "particle_material" in before and before["particle_material"]["native_clock"]!=clock:
            raise ContractError("Native particle and grasp boundary clocks differ")
        for key in ("body_world","body_linear_velocity_m_s","body_angular_velocity_rad_s"):
            if old[key]!=before[key]:raise ContractError("Native grasp body snapshot differs from observation: "+key)
        if old["hand_world"]!=before["arms"][arm]["hand_world"]:
            raise ContractError("Native grasp hand snapshot differs from this actual arm")
        history=getattr(self,"_native_boundary_events",())
        if history:
            first=history[0]["event"]["before"]
            if any(old[k]!=first[k] for k in ("body_entity_idx","cup_link_idx","cup_mass_kg")):
                raise ContractError("Persistent native cup identity or mass changed across boundary events")
        for inv in (old["inventory"],new["inventory"]):
            if (type(inv) is not dict or set(inv)!={"count","capacity","rows","weld_pairs"}
                    or type(inv["count"]) is not int or type(inv["capacity"]) is not int
                    or not 0<=inv["count"]<=inv["capacity"] or inv["count"]!=len(inv["rows"])
                    or any(type(r) is not list or len(r)!=3 or any(type(v) is not int or v<0 for v in r) for r in inv["rows"])
                    or inv["weld_pairs"]!=[r[1:] for r in inv["rows"] if r[0]==1]):
                raise ContractError("Native grasp equality inventory is malformed")
        released,attached=(old,new) if native["event"]=="attach" else (new,old)
        baseline,owned=released["inventory"],attached["inventory"]
        if (baseline["weld_pairs"] or any(r[0]!=2 for r in baseline["rows"])
                or owned["rows"]!=baseline["rows"]+[[1,*pair]] or owned["weld_pairs"]!=[pair]
                or owned["count"]!=baseline["count"]+1 or owned["capacity"]!=baseline["capacity"]):
            raise ContractError("Native grasp event must add/delete exactly its one owned hand/cup weld")
        binding=after.get("provided_grasp") if native["event"]=="attach" else before.get("provided_grasp")
        if (type(binding) is not dict or set(binding)!={"schema_version","model","holder_arm_id","helper_source_sha256","attach_receipt","current_snapshot"}
                or binding["schema_version"]!="provided-automatic-grasp-binding-v1" or binding["model"]!="provided_ideal"
                or binding["holder_arm_id"]!=arm or binding["helper_source_sha256"]!="7458d4f339da20d91d819f981cc8e30c3b018dd5cce375e9d61f32eb55e3c291"
                or binding["current_snapshot"]!=attached):
            raise ContractError("Actual provided-grasp source/holder/snapshot binding differs")
        if native["event"]=="attach":
            if ((history and history[-1]["event"]["event"]!="release")
                    or before.get("provided_grasp") is not None or binding["attach_receipt"]!=native
                    or number(native.get("position_error_m"),"provided endpoint position",minimum=0.)>.003
                    or number(native.get("orientation_error_rad"),"provided endpoint angle",minimum=0.)>.02):
                raise ContractError("Provided attach lacks its exact endpoint receipt or already has a holder")
        else:
            attach=binding["attach_receipt"];proof=native.get("support_ready")
            if (not history or history[-1]["event"]!=attach
                    or after.get("provided_grasp") is not None or type(attach) is not dict
                    or attach.get("model")!="provided-ideal-native-weld-v1" or attach.get("event")!="attach"
                    or attach.get("arm_id")!=arm or attach.get("pair")!=pair
                    or attach["after"]["inventory"]!=owned
                    or any(attach["after"][k]!=old[k] for k in ("body_entity_idx","cup_link_idx","hand_link_idx","cup_mass_kg"))
                    or type(proof) is not dict or proof.get("passed") is not True
                    or proof.get("native_clock")!=clock or proof.get("body_entity_idx")!=old["body_entity_idx"]
                    or type(proof.get("support_id")) is not str or not proof["support_id"].strip()):
                raise ContractError("Provided release lacks actual owned attach/support-ready binding")
            _hash(proof.get("evidence_sha256"))
        receipt=dict(schema_version="provided-grasp-native-boundary-event-v1",event=copy.deepcopy(native),
            event_sha256=event_hash,before_observation_sha256=before_hash,after_observation_sha256=after_hash,
            segment_sha256=self.current_segment.sha256,step=self._step,simulation_s=self._time,
            source_bindings_verified=True,physical_step_completed=False,
            scope="Actual zero-step provided native weld membership event. No physical interval, plan or command advancement.")
        # No mutation before all checks succeed. The completed inner dispatcher's
        # recorded last-observation hash must agree with its global wrapper.
        self._last_hash=after_hash;self._dispatcher._last_observation=after_hash
        self._native_boundary_events=[*history,receipt]
        return copy.deepcopy(receipt)

    def accept_provided_robot_reset_event(self, before_observation, after_observation, event, *, source_bindings_verified):
        return self._accept_released_robot_joint_event(before_observation,after_observation,event,
            source_bindings_verified=source_bindings_verified,kinematic=False)

    def accept_provided_kinematic_return_event(self, before_observation, after_observation, event, *, source_bindings_verified):
        return self._accept_released_robot_joint_event(before_observation,after_observation,event,
            source_bindings_verified=source_bindings_verified,kinematic=True)

    def _accept_released_robot_joint_event(self,before_observation,after_observation,event,*,source_bindings_verified,kinematic):
        """Narrow source-verified released joint operation. No physical commit."""
        if kinematic:
            from physics.provided_robot_return import validate_kinematic_return_receipt as validate
        else:
            from physics.provided_robot_reset import validate_robot_reset_receipt as validate
        before, after, native = copy.deepcopy((before_observation, after_observation, event))
        validate(native, before_observation=before, after_observation=after)
        history_key='_native_kinematic_return_events' if kinematic else '_native_robot_reset_events'
        history = getattr(self, history_key, [])
        require_arm = ('arm_0','arm_1')[len(history)] if len(history) < 2 else None
        if (source_bindings_verified is not True or self.record()['status'] != 'SEGMENT_COMPLETED'
                or self._pending is not None or self._dispatcher._pending is not None
                or native['arm_id'] != require_arm or digest(before) != self._last_hash
                or self._dispatcher._last_observation != self._last_hash
                or native['native_scene_step'] != self._step or native['simulation_s'] != self._time
                or getattr(self, '_pending_robot_reset_extension', None) is not None):
            raise ContractError('Robot reset requires its exact source-verified completed donor boundary')
        weld_history = getattr(self, '_native_boundary_events', [])
        release = weld_history[-1]['event'] if weld_history else None
        if (type(release) is not dict or release.get('event') != 'release'
                or release.get('arm_id') != native['arm_id'] or digest(release) != native['release_event_sha256']):
            raise ContractError('Robot reset must follow its authenticated native supported release')
        arm = native['arm_id']
        if (before.get('provided_grasp') is not None or after.get('provided_grasp') is not None
                or {k:v for k,v in before.items() if k!='arms'} != {k:v for k,v in after.items() if k!='arms'}
                or set(before['arms']) != set(self._arm_contract) or set(after['arms']) != set(self._arm_contract)
                or any(before['arms'][n] != after['arms'][n] for n in self._arm_contract if n != arm)):
            raise ContractError('Robot reset changed the payload, another robot, clock or provided grasp')
        for obs, snapshot in ((before,native['native_before']), (after,native['native_after'])):
            for name in self._arm_contract:
                if any(obs['arms'][name][key] != snapshot['arms'][name][key] for key in ('q_all_dofs','qd_all_dofs')):
                    raise ContractError('Reset native joint snapshot differs from its bound observation')
        receipt = dict(schema_version='provided-kinematic-return-native-boundary-event-v1' if kinematic else 'provided-robot-reset-native-boundary-event-v1',
            event=copy.deepcopy(native), event_sha256=digest(native),
            before_observation_sha256=digest(before), after_observation_sha256=digest(after),
            segment_sha256=self.current_segment.sha256, step=self._step, simulation_s=self._time,
            physical_step_completed=False, source_bindings_verified=True)
        target = native['target_command']
        self._last_hash = digest(after)
        self._dispatcher._last_observation = self._last_hash
        self._last_commands[arm] = (tuple(target['q_rad']), tuple(target['finger_q_m']))
        setattr(self,history_key,[*history,receipt])
        self._pending_robot_reset_extension = copy.deepcopy(receipt)
        return copy.deepcopy(receipt)

    def extend(self, next_plan, *, model_sha256_by_arm, boundary):
        if self._pending is not None:raise ContractError("Cannot extend with an uncommitted command")
        previous=self.current_segment
        if not self._dispatcher.record()["command_sequence_completed"]:
            raise ContractError("Previous segment must be fully physically committed")
        if self._step>=self._deadline_steps:raise ContractError("Global episode deadline reached")
        if type(next_plan) is not EpisodePlan:raise ContractError("Use validated next EpisodePlan")
        expected=("schema_version","previous_segment_sha256","step","simulation_s","observation_sha256",
                  "verified_arm_ids","actual_state_verified","measured_rest_verified","source_bindings_verified")
        reset_boundary = boundary.get('schema_version') in ('episode-continuation-reset-boundary-v1','episode-continuation-kinematic-return-boundary-v1')
        event_key=('native_robot_return_receipt_sha256' if boundary.get('schema_version')=='episode-continuation-kinematic-return-boundary-v1' else 'native_robot_reset_receipt_sha256')
        if reset_boundary:
            expected += (event_key,'rest_semantics')
        _fields(boundary,expected,"Measured extension boundary")
        receipt=copy.deepcopy(boundary)
        if receipt["schema_version"] not in ("episode-continuation-rest-boundary-v1",'episode-continuation-reset-boundary-v1','episode-continuation-kinematic-return-boundary-v1'):
            raise ContractError("Unknown extension receipt")
        reset = getattr(self, '_pending_robot_reset_extension', None)
        if reset_boundary:
            expected_event_kind=('provided-kinematic-return-native-boundary-event-v1'
                if receipt['schema_version']=='episode-continuation-kinematic-return-boundary-v1'
                else 'provided-robot-reset-native-boundary-event-v1')
            if (reset is None or reset.get('schema_version')!=expected_event_kind
                    or receipt[event_key] != reset['event_sha256']
                    or receipt['observation_sha256'] != reset['after_observation_sha256']
                    or receipt['rest_semantics'] != ('Actual pre-return rest plus native home zero velocity. No physical return interval or home rest window.' if receipt['schema_version']=='episode-continuation-kinematic-return-boundary-v1' else 'Actual pre-reset rest plus fresh post-reset native zero velocity. No post-reset pose history.')):
                raise ContractError('Released robot extension lacks its exact one-use native event')
        elif reset is not None:
            raise ContractError('A released robot anchor must use its explicit operation extension receipt')
        if (receipt["previous_segment_sha256"]!=previous.sha256 or type(receipt["step"]) is not int
                or receipt["step"]!=self._step or receipt["observation_sha256"]!=self._last_hash):
            raise ContractError("Extension is not bound to the last committed observation")
        if number(receipt["simulation_s"],"extension time",minimum=0.)!=self._time:
            raise ContractError("Extension time is not the exact committed observation")
        if (type(receipt["verified_arm_ids"]) is not list
                or any(type(n) is not str for n in receipt["verified_arm_ids"])
                or sorted(receipt["verified_arm_ids"])!=sorted(self._arm_contract)
                or any(receipt[n] is not True for n in
                       ("actual_state_verified","measured_rest_verified","source_bindings_verified"))):
            raise ContractError("Every exact arm, measured rest and source must be verified")
        if next_plan.dt_s!=self._dt:raise ContractError("Continuation dt changed")
        models=_models(model_sha256_by_arm,next_plan.arms)
        if models!=self._models:raise ContractError("Continuation arm/model bindings changed")
        if next_plan.initial_state_sha256!=self._last_hash:
            raise ContractError("Next segment initial state must be the actual boundary hash")
        for arm in next_plan.arms:
            current=dict(role=arm.role,finger_bounds=arm.finger_bounds_m,limits=arm.limits.to_record())
            if current!=self._arm_contract.get(arm.arm_id):
                raise ContractError("Continuation arm role, native limits or finger bounds changed")
            if (arm.initial_q_rad,arm.initial_finger_m)!=self._last_commands[arm.arm_id]:
                raise ContractError("All-arm commanded endpoint continuity is mandatory")
        if self._action_ids.intersection(a.action_id for a in next_plan.actions):
            raise ContractError("Global action IDs must remain unique")
        if self._step+next_plan.full_steps>2**53-1:
            raise ContractError("Global continuation step not representable")
        segment=SegmentEnvelope(next_plan,previous.index+1,self._step,self._time,
            previous.action_offset+len(previous.plan.actions),previous.sha256,self._last_hash,models)
        # No mutation occurs before every extension check succeeds.
        self._segments.append(segment);self._dispatcher=EpisodeDispatcher(next_plan)
        self._action_ids.update(a.action_id for a in next_plan.actions)
        self._extension_receipts.append(receipt)
        if reset_boundary:self._pending_robot_reset_extension=None
        return segment

    def record(self):
        local=self._dispatcher.record()
        status=("AWAITING_PHYSICAL_OBSERVATION" if self._pending is not None else
                "GLOBAL_DEADLINE_REACHED" if self._step>=self._deadline_steps else
                "SEGMENT_COMPLETED" if local["command_sequence_completed"] else
                "SEGMENT_DEADLINE_REACHED" if local["deadline_reached"] else "READY")
        result = dict(schema_version="episode-continuation-dispatch-v1",status=status,
            global_step=self._step,simulation_s=self._time,global_deadline_s=self._deadline_steps*self._dt,
            last_observation_sha256=self._last_hash,pending_step=self._pending.step if self._pending else None,
            current_segment_sha256=self.current_segment.sha256,
            segments=[s.to_record() for s in self._segments],
            global_boundary_acknowledgements=copy.deepcopy(self._global_acks),
            extension_receipts=copy.deepcopy(self._extension_receipts),
            last_command_by_arm={n:dict(q_rad=list(q),finger_q_m=list(f)) for n,(q,f) in self._last_commands.items()},
            current_segment_complete=local["command_sequence_completed"],
            local_dispatch_clock_meaning="Internal elapsed segment time only. Not a reset native clock.",
            physical_evidence_verified=False,physical_handoff_verified=False,full_mission_passed=False,
            native_integration_status="REQUIRES_SEPARATE_NATIVE_EVIDENCE")
        if getattr(self,"_native_boundary_events",None):
            result["native_boundary_events"]=copy.deepcopy(self._native_boundary_events)
        if getattr(self,'_native_kinematic_return_events',None):
            result['native_kinematic_return_events']=copy.deepcopy(self._native_kinematic_return_events)
        if getattr(self,'_native_robot_reset_events',None):
            result['native_robot_reset_events']=copy.deepcopy(self._native_robot_reset_events)
        return result

