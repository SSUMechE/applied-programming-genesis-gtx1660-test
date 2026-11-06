"""Supplied sampled released-delivery assessment from one actual episode.

This pure module controls no simulator and writes no file. The runtime freezes
``frozen_inputs()`` before its first step and supplies native observations using
the local contracts below. Hashes bind records, not the truth of an untrusted
adapter. The adapter must verify source files, measured identities and coverage.

Only a released delivery stage is assessed. Approach, handoff, retreat, multiple
robots, continuous safety and grade calibration are outside this contract.
Legacy fixed-transform fidelity is reported separately, without a new allowance.
"""
from dataclasses import dataclass
import hashlib
import json
import math

from physics.handle_contacts import contents_contact_witness_valid


class MissionInputError(ValueError):
    """Missing, malformed or chronologically inconsistent physical evidence."""


def _copy(value):
    try:
        return json.loads(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False))
    except (TypeError,ValueError,OverflowError) as exc:
        raise MissionInputError('Evidence must be finite JSON data') from exc


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _number(value,label,minimum=None):
    try:finite=math.isfinite(value) if type(value) in (int,float) else False
    except OverflowError:finite=False
    if not finite:
        raise MissionInputError(label+' must be a finite number')
    value=float(value)
    if minimum is not None and value<minimum:raise MissionInputError(label+' is below its minimum')
    return value


def _int(value,label,minimum=0):
    if type(value) is not int or value<minimum:raise MissionInputError(label+' must be a nonnegative exact integer')
    return value


def _bool(value,label):
    if type(value) is not bool:raise MissionInputError(label+' must be a measured boolean')
    return value


def _vector(value,n,label):
    if not isinstance(value,(list,tuple)) or len(value)!=n:raise MissionInputError(label+' has the wrong shape')
    return [_number(x,label) for x in value]


def _name(value,label):
    if type(value) is not str or not value.strip():raise MissionInputError(label+' must be a nonempty string')
    return value


def _hash(value,label):
    if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise MissionInputError(label+' must be a lowercase SHA256')
    return value


def _pose(value):
    if not isinstance(value,dict) or set(value)!={'frame','position_m','quaternion_wxyz'} or value['frame']!='world':
        raise MissionInputError('A measured world pose needs frame, position_m and quaternion_wxyz')
    _vector(value['position_m'],3,'world position');q=_vector(value['quaternion_wxyz'],4,'world quaternion')
    if abs(math.fsum(v*v for v in q)-1.)>1e-6:raise MissionInputError('Measured quaternion must be unit')
    return value


def _norm(value):return math.sqrt(math.fsum(v*v for v in value))


def _angle(a,b):
    return 2*math.acos(min(1.,abs(math.fsum(x*y for x,y in zip(a,b)))/(_norm(a)*_norm(b))))


_FIXED = dict(goal_position_tolerance_m=.006,goal_orientation_tolerance_rad=.04,stability_s=.5,
    maximum_content_speed_m_s=.1,maximum_body_linear_speed_m_s=.01,maximum_body_angular_speed_rad_s=.1,
    release_open_minimum_m=.035,maximum_tracking_error_rad=.03,effort_identity_tolerance=.005,
    position_numeric_tolerance=1e-5,speed_numeric_tolerance=1e-6,acceleration_numeric_tolerance=1e-5)
_POLICY_KEYS=set(_FIXED)|{'schema_version','dt_s','deadline_s','carrier_id','goal_support_id','goal_body_pose',
    'arm_position_bounds_rad','finger_position_bounds_m','arm_speed_limits_rad_s','arm_acceleration_limits_rad_s2','effort_limits'}
_ACTIONS=('post_initialization_robot_pose_setters','runtime_tray_pose_setters','particle_resets','runtime_welds','external_body_trajectory_commands')
_BINDINGS=('world_sha256','design_sha256','configuration_sha256','grasp_sha256','source_manifest_sha256','coverage_policy_sha256')
_PHASES=('settle','close','hold','carry','place','release','released_settle')
_CATEGORIES={'exact_finger_pad_handle','named_tray_bottom_support_top','contents_contact','named_support_floor','UNALLOWED_CONTACT'}


@dataclass(frozen=True,init=False)
class SampledMissionPolicy:
    """Immutable explicit teacher-development policy, with no inferred defaults.

``from_record`` requires every field listed in ``policy_fields()``. Seven arm
position/speed/acceleration limits and both finger position bounds use native
coordinate order. Nine effort limits cover every actuated DOF. Positions are m,
arm angles rad and all times simulation seconds. The fixed thresholds identify
this prototype, not grading or a changed fixed-transform tolerance.
"""
    _json: str

    @classmethod
    def from_record(cls,record):
        data=_copy(record)
        if not isinstance(data,dict) or set(data)!=_POLICY_KEYS:raise MissionInputError('Policy fields must be explicit and exact')
        if data['schema_version']!='sampled-released-delivery-policy-v1':raise MissionInputError('Unknown mission policy schema')
        for key,wanted in _FIXED.items():
            if _number(data[key],key)!=wanted:raise MissionInputError('This prototype requires its explicit uncalibrated '+key)
        dt=_number(data['dt_s'],'dt_s',0.);deadline=_number(data['deadline_s'],'deadline_s',0.)
        if dt<=0 or deadline<=0 or not math.isfinite(deadline/dt) or abs(deadline/dt-round(deadline/dt))>1e-8:
            raise MissionInputError('Deadline must be positive and on the positive dt grid')
        _name(data['carrier_id'],'carrier ID');_name(data['goal_support_id'],'goal support');_pose(data['goal_body_pose'])
        for key,n in (('arm_position_bounds_rad',7),('finger_position_bounds_m',2)):
            if not isinstance(data[key],list) or len(data[key])!=n:raise MissionInputError('Native position bounds have wrong dimension')
            for pair in data[key]:
                lo,hi=_vector(pair,2,key)
                if not lo<hi:raise MissionInputError('Native position bounds must be ordered')
        for lo,hi in data['finger_position_bounds_m']:
            if not lo<data['release_open_minimum_m']<=hi:raise MissionInputError('Release opening must fit both native finger bounds')
        for key,n in (('arm_speed_limits_rad_s',7),('arm_acceleration_limits_rad_s2',7),('effort_limits',9)):
            if any(x<=0 for x in _vector(data[key],n,key)):raise MissionInputError('Native motion/effort limits must be positive')
        instance=object.__new__(cls);object.__setattr__(instance,'_json',json.dumps(data,sort_keys=True,separators=(',',':')))
        return instance

    @staticmethod
    def policy_fields():return tuple(sorted(_POLICY_KEYS))

    def to_record(self):return json.loads(self._json)

    @property
    def sha256(self):return _digest(self.to_record())


class SampledMissionAssessment:
    """Accumulate complete same-episode native samples and unique delivery IDs.

Constructor inputs are detached/frozen. ``initial_material`` is a nonempty list
of unique ``{id,mass_kg}`` records. ``initial_state`` contains time0, measured nine
DOF q/qd, world body pose/velocities, complete ``materials``, verified source and
runtime-action records. A material sample contains id,mass_kg,active,
inside_carrier,world_position_m,velocity_world_m_s for every original ID.

``observe`` uses existing loaded trace names, plus input_binding_sha256,
source_bindings_verified, runtime_action_counts, phase_constraints_passed,
workspace_constraints_passed, materials and coverage. Coverage contains the
live policy stamp, exact step/pre/post times, native/copied/classified counts
and contacts_complete. The static policy ledger alone is insufficient.

Detection-time support/no-handle witnesses and both pre/post state endpoints
must satisfy each consecutive released-settle interval. A malformed row raises
MissionInputError and permanently marks measurement evidence incomplete. An
observed safety violation is retained even if a later row recovers. Final source
verification is mandatory. Release interruption never earns delivery credit.
"""
    def __init__(self,policy,*,bindings,initial_material,initial_state,contact_coverage):
        if type(policy) is not SampledMissionPolicy:raise MissionInputError('Use an explicit frozen SampledMissionPolicy')
        self._p=policy.to_record();self._bindings=_copy(bindings)
        if not isinstance(self._bindings,dict) or set(self._bindings)!=set(_BINDINGS)|{'episode_id'}:raise MissionInputError('Episode bindings are incomplete')
        _name(self._bindings['episode_id'],'episode ID')
        for key in _BINDINGS:_hash(self._bindings[key],key)
        supply=_copy(initial_material)
        if not isinstance(supply,list) or not supply:raise MissionInputError('Finite persistent supply must be nonempty')
        self._weights={}
        for row in supply:
            if not isinstance(row,dict) or set(row)!={'id','mass_kg'}:raise MissionInputError('Initial supply needs exact ID/mass pairs')
            name=_name(row['id'],'material ID');mass=_number(row['mass_kg'],'material mass',0.)
            if name in self._weights or mass<=0:raise MissionInputError('Supply IDs must be unique and masses positive')
            self._weights[name]=mass
        coverage=_copy(contact_coverage)
        if coverage.get('schema_version')!='live-contact-coverage-v1' or coverage.get('coverage_policy_sha256')!=self._bindings['coverage_policy_sha256']:
            raise MissionInputError('Static contact policy is not bound to this episode')
        _bool(coverage['eligible_for_sampled_contact_assessment'],'static coverage eligibility')
        if not isinstance(coverage['failures'],list):raise MissionInputError('Static coverage failures must be explicit')
        initial=_copy(initial_state)
        if _number(initial['simulation_s'],'initial time')!=0.:raise MissionInputError('Start with a measured time-zero state')
        _vector(initial['q_all_dofs'],9,'initial q');_vector(initial['qd_all_dofs'],9,'initial qd');_pose(initial['body_authored_world'])
        _vector(initial['body_linear_velocity_m_s'],3,'initial body velocity');_vector(initial['body_angular_velocity_rad_s'],3,'initial body angular velocity')
        self._materials(initial['materials']);self._actions(initial['runtime_action_counts'])
        _bool(initial['source_bindings_verified'],'initial source verification')
        self._inputs=_copy(dict(schema_version='sampled-mission-inputs-v1',policy=self._p,bindings=self._bindings,
            initial_material=supply,initial_state=initial,contact_coverage=coverage,
            contact_numeric_guard=dict(maximum_penetration_m=.001,maximum_normal_unit_error=.001),
            scope='Uncalibrated teacher-development single released-delivery stage'))
        self._input_binding_sha256=_digest(self._inputs)
        self._last=initial;self._step=0;self._phase=None;self._seen=set();self._release_command=None;self._opened=False
        self._since={name:None for name in self._weights};self._qualified=set();self._ever={};self._violations={};self._missing=[];self._missing_count=0
        self._rows_hash=self.input_binding_sha256;self._final=None;self._terminal_hash=None;self._last_actions=initial['runtime_action_counts']
        self._fidelity=dict(observed_pre_release_intervals=0,maximum_translation_m=0.,maximum_orientation_rad=0.,legacy_diagnostic_exceedance_intervals=0)
        self._effort_peaks={k:[0.]*9 for k in ('requested_motor','bounded_motor_control','transmitted_joint_output')}
        if not initial['source_bindings_verified']:self._miss('Initial source bindings not verified')
        if not coverage['eligible_for_sampled_contact_assessment'] or coverage['failures']:self._miss('Static contact coverage incomplete')
        for key,count in self._last_actions.items():
            if count:self._violate('prohibited_'+key,0)

    def frozen_inputs(self):return _copy(self._inputs)

    @property
    def input_binding_sha256(self):return self._input_binding_sha256

    def _miss(self,message):
        self._missing_count+=1
        if len(self._missing)<8:self._missing.append(message)

    def _materials(self,rows):
        if not isinstance(rows,list):raise MissionInputError('Material observations must be a complete list')
        out={}
        for row in rows:
            name=_name(row['id'],'observed material ID')
            if name in out or name not in self._weights:raise MissionInputError('Duplicate or unknown persistent material ID')
            if _number(row['mass_kg'],'observed mass')!=self._weights[name]:raise MissionInputError('Persistent material mass changed')
            _bool(row['active'],'material active');_bool(row['inside_carrier'],'inside carrier')
            if not row['active'] and row['inside_carrier']:raise MissionInputError('Inactive material cannot be inside carrier')
            _vector(row['world_position_m'],3,'material world position');_vector(row['velocity_world_m_s'],3,'material world velocity')
            out[name]=row
        if set(out)!=set(self._weights):raise MissionInputError('Missing persistent material measurement')
        return out

    def _actions(self,record):
        if not isinstance(record,dict) or set(record)!=set(_ACTIONS):raise MissionInputError('Runtime action counters are incomplete')
        for key in _ACTIONS:_int(record[key],key)
        return record

    def _violate(self,kind,step):
        value=self._violations.setdefault(kind,dict(count=0,first_step=step,last_step=step));value['count']+=1;value['last_step']=step

    def _goal(self,pose):
        goal=self._p['goal_body_pose']
        return math.dist(pose['position_m'],goal['position_m'])<=self._p['goal_position_tolerance_m'] and _angle(pose['quaternion_wxyz'],goal['quaternion_wxyz'])<=self._p['goal_orientation_tolerance_rad']

    def _coverage(self,row,contacts):
        value=row['coverage'];stamp=value['policy_stamp'];step=row['step'];dt=self._p['dt_s']
        if stamp['schema_version']!='live-contact-policy-stamp-v1' or stamp['coverage_policy_sha256']!=self._bindings['coverage_policy_sha256']:
            raise MissionInputError('Live contact stamp identity differs')
        for key in ('policy_valid_now','pruning_disabled','single_substep'):_bool(stamp[key],key)
        if not isinstance(stamp['failures'],list):raise MissionInputError('Live policy failures must be explicit')
        if type(value['step']) is not int or value['step']!=step or abs(_number(value['pre_time_s'],'coverage pre time')-(step-1)*dt)>1e-9 or abs(_number(value['post_time_s'],'coverage post time')-step*dt)>1e-9:
            raise MissionInputError('Coverage stamp is stale or has the wrong interval')
        counts=[_int(value[k],k) for k in ('native_contact_count','copied_contact_count','classified_contact_count')]
        complete=_bool(value['contacts_complete'],'contact buffer completeness')
        return complete and counts==[len(contacts)]*3 and all(stamp[k] for k in ('policy_valid_now','pruning_disabled','single_substep')) and not stamp['failures']

    def observe(self,observation):
        if self._final is not None:raise MissionInputError('Episode already finalized')
        try:return self._observe(_copy(observation))
        except (MissionInputError,KeyError,TypeError,ValueError,OverflowError) as exc:
            self._miss('Rejected observation: '+str(exc))
            self._qualified.clear()
            if isinstance(exc,MissionInputError):raise
            raise MissionInputError('Incomplete or malformed observation: '+str(exc)) from exc

    def _observe(self,row):
        p=self._p;dt=p['dt_s'];step=_int(row['step'],'step',1)
        if step!=self._step+1 or abs(_number(row['contact_detection_time_s'],'detection time')-(step-1)*dt)>1e-9 or abs(_number(row['simulation_s'],'post time')-step*dt)>1e-9:
            raise MissionInputError('Observations must cover every contiguous dt interval')
        if row['input_binding_sha256']!=self.input_binding_sha256:raise MissionInputError('Observation belongs to another episode/input binding')
        if not _bool(row['post_processing_complete'],'post processing'):raise MissionInputError('Partial integration row cannot be assessed')
        for key in ('source_bindings_verified','phase_constraints_passed','workspace_constraints_passed'):_bool(row[key],key)
        phase=row['phase']
        if phase not in _PHASES:raise MissionInputError('Unknown measured phase')
        allowed={'settle':('settle','close'),'close':('close','hold'),'hold':('hold','carry'),'carry':('carry','place','release'),
            'place':('place','release'),'release':('release','released_settle'),'released_settle':('released_settle',)}
        if (self._phase is None and phase!='settle') or (self._phase is not None and phase not in allowed[self._phase]):raise MissionInputError('Missing or reversed physical phase')
        pre=_vector(row['pre_q_all_dofs'],9,'pre q');pre_v=_vector(row['pre_qd_all_dofs'],9,'pre qd')
        q=_vector(row['post_q_all_dofs'],9,'post q');v=_vector(row['post_qd_all_dofs'],9,'post qd')
        command=_vector(row['command_q_all_dofs'],9,'command q');command_v=_vector(row['command_qd_all_dofs'],9,'command qd')
        if pre!=self._last['q_all_dofs'] or pre_v!=self._last['qd_all_dofs']:raise MissionInputError('Missing or discontinuous measured q/qd state')
        pre_body=_pose(row['pre_body_authored_world']);body=_pose(row['post_body_authored_world'])
        if pre_body!=self._last['body_authored_world']:raise MissionInputError('Missing or discontinuous measured body pose')
        body_v=_vector(row['body_linear_velocity_m_s'],3,'body velocity');body_w=_vector(row['body_angular_velocity_rad_s'],3,'body angular velocity')
        acceleration=[(v[i]-pre_v[i])/dt for i in range(7)]
        supplied=_vector(row['measured_qdd_fd_rad_s2'],7,'measured qdd')
        if any(abs(a-b)>1e-10 for a,b in zip(acceleration,supplied)):raise MissionInputError('Measured acceleration does not cover this interval')
        requested=_vector(row['requested_unclipped_effort'],9,'requested effort');control=_vector(row['pre_step_controller_effort'],9,'pre controller effort')
        measured=_vector(row['post_step_gearbox_effort'],9,'measured post effort')
        actions=self._actions(row['runtime_action_counts'])
        if any(actions[k]<self._last_actions[k] for k in _ACTIONS):raise MissionInputError('Runtime action counter decreased')
        materials=self._materials(row['materials']);prior_materials=self._materials(self._last['materials'])
        contacts=row['classified_contacts']
        if not isinstance(contacts,list):raise MissionInputError('Classified contact list must be measured, even when empty')
        pad_count=0;goal_support=0;forbidden=False
        for i,c in enumerate(contacts):
            if type(c['raw_row']) is not int or c['raw_row']!=i:raise MissionInputError('Classified rows must cover the copied buffer exactly')
            pair=c['geom_ids']
            if not isinstance(pair,list) or len(pair)!=2 or any(type(x) is not int or x<0 for x in pair) or pair[0]==pair[1]:raise MissionInputError('Invalid measured geometry pair')
            category=_name(c['category'],'contact category')
            if category=='contents_contact' and _number(c['penetration_m'],'contact penetration')<0.:
                valid_contact=contents_contact_witness_valid(c['penetration_m'],c['normal_unit_error'],c.get('owners'))
            else:
                depth=_number(c['penetration_m'],'contact penetration',0.);normal_error=_number(c['normal_unit_error'],'contact normal error',0.)
                valid_contact=depth<=.001 and normal_error<=.001
            forbidden |= category not in _CATEGORIES or category=='UNALLOWED_CONTACT' or not valid_contact
            pad_count+=category=='exact_finger_pad_handle'
            if category=='named_tray_bottom_support_top':goal_support+=_name(c['support_station_id'],'named support')==p['goal_support_id']
        coverage=self._coverage(row,contacts)
        translation=_number(row['relative_transform_position_error_m'],'fixed-transform position diagnostic',0.)
        orientation=_number(row['relative_transform_orientation_error_rad'],'fixed-transform orientation diagnostic',0.)
        # All evidence shape/clock checks precede updates to delivery eligibility.
        if not coverage:self._miss('Incomplete contacts at step '+str(step))
        if not row['source_bindings_verified']:self._miss('Unverified source bindings at step '+str(step))
        for key,count in actions.items():
            if count:self._violate('prohibited_'+key,step)
        if row['simulation_s']>p['deadline_s']+1e-9:self._violate('after_deadline',step)
        for key in ('phase_constraints_passed','workspace_constraints_passed'):
            if not row[key]:self._violate(key,step)
        bounds=p['arm_position_bounds_rad']+p['finger_position_bounds_m'];tol=p['position_numeric_tolerance']
        if any(not lo-tol<=x<=hi+tol for values in (pre,q) for x,(lo,hi) in zip(values,bounds)):self._violate('measured_native_joint_bounds',step)
        if any(abs(x)>limit+p['speed_numeric_tolerance'] for values in (pre_v[:7],v[:7]) for x,limit in zip(values,p['arm_speed_limits_rad_s'])):self._violate('measured_joint_speed',step)
        if any(abs(x)>limit+p['acceleration_numeric_tolerance'] for x,limit in zip(acceleration,p['arm_acceleration_limits_rad_s2'])):self._violate('measured_joint_acceleration',step)
        if max(abs(x-y) for x,y in zip(q[:7],command[:7]))>p['maximum_tracking_error_rad']:self._violate('measured_joint_tracking',step)
        efforts=p['effort_limits'];etol=p['effort_identity_tolerance']
        if any(abs(x)>limit for x,limit in zip(requested,efforts)):self._violate('actuator_saturation',step)
        if any(abs(x)>limit+etol for x,limit in zip(control,efforts)):self._violate('measured_effort_limit',step)
        if any(abs(x-max(-limit,min(limit,y)))>etol for x,y,limit in zip(control,requested,efforts)):self._violate('effort_identity',step)
        # Native force ranges limit the motor control, not transmitted shaft
        # output. The latter includes armature acceleration and passive losses.
        # Its nine finite measurements remain required, without inventing a
        # separate unprovided output-load limit.
        for key,values in (('requested_motor',requested),('bounded_motor_control',control),('transmitted_joint_output',measured)):
            self._effort_peaks[key]=[max(old,abs(value)) for old,value in zip(self._effort_peaks[key],values)]
        if forbidden:self._violate('forbidden_contact',step)
        if phase in ('hold','carry','place'):
            f=self._fidelity;f['observed_pre_release_intervals']+=1;f['maximum_translation_m']=max(f['maximum_translation_m'],translation);f['maximum_orientation_rad']=max(f['maximum_orientation_rad'],orientation)
            f['legacy_diagnostic_exceedance_intervals']+=translation>.0001 or orientation>.001
        if phase=='release' and self._phase!='release':
            self._release_command=self._last.get('command_q_all_dofs',command)[:7]
            if not self._goal(pre_body) or not all(x<p['release_open_minimum_m'] for x in pre[7:]):self._violate('invalid_measured_release_entry',step)
        if phase in ('release','released_settle'):
            if command[:7]!=self._release_command or command_v[:7]!=[0.]*7:self._violate('release_arm_command_changed',step)
            if all(x>=p['release_open_minimum_m'] for x in q[7:]):self._opened=True
        global_good=(phase=='released_settle' and self._opened and coverage and row['source_bindings_verified']
            and not self._violations and not self._missing and row['simulation_s']<=p['deadline_s']+1e-9 and pad_count==0 and goal_support>0
            and all(x>=p['release_open_minimum_m'] for values in (pre[7:],q[7:]) for x in values)
            and self._goal(pre_body) and self._goal(body)
            and all(_norm(x)<=p['maximum_body_linear_speed_m_s'] for x in (self._last['body_linear_velocity_m_s'],body_v))
            and all(_norm(x)<=p['maximum_body_angular_speed_rad_s'] for x in (self._last['body_angular_velocity_rad_s'],body_w)))
        qualified=set()
        for name,value in materials.items():
            old=prior_materials[name]
            good=global_good and all(x['active'] and x['inside_carrier'] and _norm(x['velocity_world_m_s'])<=p['maximum_content_speed_m_s'] for x in (old,value))
            self._since[name]=(row['contact_detection_time_s'] if self._since[name] is None else self._since[name]) if good else None
            if good and row['simulation_s']-self._since[name]>=p['stability_s']-1e-9:
                qualified.add(name);self._ever.setdefault(name,row['simulation_s'])
        self._qualified=qualified;self._step=step;self._phase=phase;self._seen.add(phase);self._last_actions=actions
        self._last=dict(simulation_s=row['simulation_s'],q_all_dofs=q,qd_all_dofs=v,body_authored_world=body,
            body_linear_velocity_m_s=body_v,body_angular_velocity_rad_s=body_w,materials=row['materials'],command_q_all_dofs=command)
        self._rows_hash=_digest(dict(previous=self._rows_hash,observation=row))
        return self._summary(final=False)

    def _summary(self,final):
        materials=self._materials(self._last['materials']);allowed=not self._violations and not self._missing
        delivered=set(self._qualified) if allowed else set();buckets={k:[] for k in ('delivered','remaining','in_transit','outside','inactive')}
        for name,value in materials.items():
            bucket='inactive' if not value['active'] else 'delivered' if name in delivered else 'outside' if not value['inside_carrier'] else 'remaining' if 'carry' not in self._seen else 'in_transit'
            buckets[bucket].append(name)
        counts={k:len(v) for k,v in buckets.items()};masses={k:math.fsum(self._weights[name] for name in ids) for k,ids in buckets.items()}
        return _copy(dict(schema_version='sampled-mission-assessment-v1',episode_id=self._bindings['episode_id'],input_binding_sha256=self.input_binding_sha256,
            observed_steps=self._step,simulation_s=self._last['simulation_s'],observation_hash_chain=self._rows_hash,
            policy_sha256=_digest(self._p),phase=self._phase,observed_phases=[p for p in _PHASES if p in self._seen],
            delivered_unique_ids=sorted(delivered),first_qualified_delivery_times_s=self._ever,material_buckets=buckets,counts=counts,masses_kg=masses,
            initial_count=len(self._weights),initial_mass_kg=math.fsum(self._weights.values()),count_conserved=sum(counts.values())==len(self._weights),
            mass_conserved=math.isclose(math.fsum(masses.values()),math.fsum(self._weights.values()),rel_tol=1e-12,abs_tol=1e-14),
            original_mass_kg=math.fsum(self._weights.values()),
            active_mass_kg=math.fsum(masses[k] for k in masses if k!='inactive'),conserved_mass_kg=math.fsum(masses.values()),
            retained_not_delivered_ids=sorted(buckets['remaining']+buckets['in_transit']),
            retained_not_delivered_mass_kg=masses['remaining']+masses['in_transit'],
            delivered_mass_kg=masses['delivered'],outside_ids=sorted(buckets['outside']),outside_mass_kg=masses['outside'],
            violations=self._violations,missing_measurements=self._missing,missing_measurement_count=self._missing_count,planning_model_fidelity=dict(**self._fidelity,
                legacy_translation_limit_m=.0001,legacy_orientation_limit_rad=.001,copied_to_mission_failure=False),
            peak_absolute_efforts=self._effort_peaks,
            effort_scope='Nine native motor limits apply to requested/control effort. Nine measured transmitted joint-output efforts include armature and passive losses and have no supplied separate output-load bound.',
            provisional=not final,delivery_stage_only=True,grade_calibrated=False,continuous_collision_certificate=False,
            full_project_mission_passed=False,student_release_approved=False,
            outside_scope='Strict measured carrier containment. Outside/inactive accounting does not by itself prove a ground spill.',
            coverage_scope='Complete supplied sampled policy and copied-buffer evidence. Not continuous contact safety.'))

    def finalize(self,terminal):
        """Finish with exact last step/time, reason and final source/action audit.

Reasons completed/deadline may qualify. Interrupted/error never count release.
Repeated identical finalization is idempotent. Earlier qualified IDs are not
double-counted and are revoked if they are no longer stable at the final sample.
"""
        data=_copy(terminal);checksum=_digest(data)
        if self._final is not None:
            if checksum!=self._terminal_hash:raise MissionInputError('Finalized terminal record changed')
            return _copy(self._final)
        try:
            if data['termination_reason'] not in ('completed','deadline','interrupted','error'):raise MissionInputError('Unknown terminal reason')
            if type(data['step']) is not int or data['step']!=self._step or _number(data['simulation_s'],'terminal clock')!=self._last['simulation_s'] or data['input_binding_sha256']!=self.input_binding_sha256:
                raise MissionInputError('Terminal step/time/input binding differs from observations')
            if not _bool(data['source_bindings_verified'],'terminal source verification'):self._miss('Final source bindings not verified')
            actions=self._actions(data['runtime_action_counts'])
            if any(actions[k]<self._last_actions[k] for k in _ACTIONS):raise MissionInputError('Terminal action counter decreased')
            for key,count in actions.items():
                if count:self._violate('prohibited_'+key,self._step)
            if data['termination_reason']=='deadline' and abs(data['simulation_s']-self._p['deadline_s'])>1e-9:raise MissionInputError('Deadline termination must use the common deadline')
            if data['termination_reason'] in ('interrupted','error'):self._miss('Physical release/episode interrupted')
            if self._last['simulation_s']>self._p['deadline_s']+1e-9:self._violate('after_deadline',self._step)
            if self._phase!='released_settle' or not self._opened:self._miss('No completed measured release and settling phase')
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            self._miss('Invalid terminal evidence: '+str(exc))
        result=self._summary(final=True)
        accepted=bool(result['delivered_unique_ids']) and not result['violations'] and not result['missing_measurements']
        result.update(status='SAMPLED_MISSION_FAIL' if result['violations'] else 'SAMPLED_RELEASED_DELIVERY' if accepted else 'SAMPLED_DELIVERY_INCOMPLETE',
            delivery_stage_accepted=accepted,terminal=data,
            interpretation='Prospective actual sampled release/delivery stage only. Legacy fixed-transform fidelity remains a separate diagnostic.')
        self._terminal_hash=checksum;self._final=_copy(result)
        return _copy(result)
