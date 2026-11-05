"""Pure immutable world data for a future joint-base evaluator.

The source body, contents, supports and goal body do not follow a robot base.
Only endpoints are described. This module performs no IO, FK, IK, simulation,
physical initialization or evaluation. Existing diagnostic tasks are untouched.
Declared robot/physics identities still require native verification by a later
initializer. A measured HeldBodySpec is geometry evidence, not grip approval.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math

from physics.carried_body import HeldBodySpec, VesselModel, WorldPose
from transport_challenge.contracts import ContractError, digest, identifier, number, point
from transport_challenge.joint_context import JointInstallationZone, JointMapObstacle
from transport_challenge.joint_design import JointDesign


def _sha(value, label):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ContractError(label+' requires a lowercase SHA256')
    return value


def _reject_solution(value):
    forbidden={'waypoints','joint_waypoints','pose_nodes','nodes','edges','graph','route','path','trajectory',
               'intermediate_states','intermediate_poses','q_rad','start_q_rad','goal_q_rad'}
    if isinstance(value, Mapping):
        if forbidden.intersection(value):
            raise ContractError('A fixed world cannot supply joint states, a graph or an intermediate route')
        for child in value.values():_reject_solution(child)
    elif isinstance(value,(list,tuple)):
        for child in value:_reject_solution(child)


def _json_mapping(value, label):
    if not isinstance(value,Mapping):raise ContractError(label+' must be a JSON mapping')
    try:
        encoded=json.dumps(dict(value),sort_keys=True,separators=(',',':'),allow_nan=False)
        _reject_solution(value)
    except (TypeError,ValueError,OverflowError,RecursionError) as exc:
        raise ContractError('Invalid '+label+': '+str(exc)) from exc
    return encoded


def _world_pose(value, label):
    if type(value) is not WorldPose:
        raise ContractError(label+' must be an immutable WorldPose in metres/wxyz, not a base/local pose or raw list')
    return value


@dataclass(frozen=True)
class FixedSupportStation:
    """One axis-aligned authored box in world metres."""
    station_id: str
    size_m: tuple
    center_world_m: tuple

    def __post_init__(self):
        identifier(self.station_id,'support station ID')
        size=point(self.size_m);center=point(self.center_world_m)
        if min(size)<=0:raise ContractError('Support size must be positive in every axis')
        if any(not math.isfinite(c+s/2) or not math.isfinite(c-s/2) for c,s in zip(center,size)):
            raise ContractError('Support extents must remain finite')
        object.__setattr__(self,'size_m',size);object.__setattr__(self,'center_world_m',center)

    def to_record(self):
        return {'id':self.station_id,'frame':'world','units':'m','axis_aligned':True,
                'size_m':list(self.size_m),'center_world_m':list(self.center_world_m)}


@dataclass(frozen=True)
class FixedSphereContent:
    """Declared persistent sphere supply. Native fit/material checks remain required."""
    content_id: str
    mass_kg: float
    radius_m: float
    center_body_m: tuple

    def __post_init__(self):
        identifier(self.content_id,'content ID')
        mass=number(self.mass_kg,'content mass',minimum=0.)
        radius=number(self.radius_m,'content radius',minimum=0.)
        if mass==0 or radius==0:raise ContractError('Content mass and radius must be positive')
        object.__setattr__(self,'mass_kg',mass);object.__setattr__(self,'radius_m',radius)
        object.__setattr__(self,'center_body_m',point(self.center_body_m))

    def to_record(self):
        return {'id':self.content_id,'shape':'sphere','frame':'body','position_units':'m','mass_units':'kg',
                'mass_kg':self.mass_kg,'radius_m':self.radius_m,'center_body_m':list(self.center_body_m)}


@dataclass(frozen=True)
class FixedWorldJointTask:
    task_id: str
    object_id: str
    case_json: str = field(repr=False)
    robot_model_sha256: str
    model: VesselModel
    source_body_world: WorldPose
    source_support: FixedSupportStation
    goal_body_world: WorldPose
    goal_support: FixedSupportStation
    nominal_source_hand_world: WorldPose
    initial_contents: tuple
    physics_settings_json: str = field(repr=False)
    world_sha256: str = field(init=False)

    def __post_init__(self):
        identifier(self.task_id,'fixed world task ID');identifier(self.object_id,'body object ID')
        _sha(self.robot_model_sha256,'Declared robot model')
        if type(self.model) is not VesselModel:raise ContractError('An exact immutable VesselModel is required')
        try:
            case=json.loads(self.case_json);settings=json.loads(self.physics_settings_json)
            case_json=_json_mapping(case,'case');settings_json=_json_mapping(settings,'physics settings')
            if case.get('schema_version')!='joint-map-v1':raise ContractError('Use the explicit joint-map-v1 case')
            identifier(case['case_id'],'case ID');identifier(case['tool_link'],'tool link')
            if type(case['robot_model']) is not str or not case['robot_model'].strip():raise ContractError('Robot model path is missing')
            names=case['arm_joint_names']
            if not isinstance(names,(list,tuple)) or len(names)!=7 or len(set(names))!=7:
                raise ContractError('The task requires seven distinct native arm joint names')
            for name in names:identifier(name,'native joint name')
            number(case['base_z_m'],'task-fixed base Z',minimum=0.)
            zones=tuple(JointInstallationZone(**z) for z in case['installation_zones'])
            if not zones:raise ContractError('At least one installation zone is required')
            workspace=tuple(point(v,2) for v in case['tool_workspace_m'])
            if len(workspace)!=3 or any(lo>=hi or not math.isfinite(hi-lo) for lo,hi in workspace):
                raise ContractError('Tool workspace needs three bounded world XYZ intervals')
            obstacles=tuple(JointMapObstacle(o['id'],o['lower_m'],o['upper_m']) for o in case['obstacles'])
            if len({o.obstacle_id for o in obstacles})!=len(obstacles):raise ContractError('Obstacle IDs must be distinct')
        except (KeyError,TypeError,ValueError,OverflowError,RecursionError) as exc:
            if isinstance(exc,ContractError):raise
            raise ContractError('Malformed fixed-world map/configuration: '+str(exc)) from exc
        for name in ('source_body_world','goal_body_world','nominal_source_hand_world'):
            _world_pose(getattr(self,name),name)
        if any(not lo<=x<=hi for x,(lo,hi) in zip(self.nominal_source_hand_world.position_m,workspace)):
            raise ContractError('Nominal source hand endpoint is outside the world tool workspace')
        if type(self.source_support) is not FixedSupportStation or type(self.goal_support) is not FixedSupportStation:
            raise ContractError('Source and goal supports require fixed world boxes')
        if self.source_support.station_id==self.goal_support.station_id:
            raise ContractError('Source and goal supports must have distinct IDs')
        if not isinstance(self.initial_contents,(list,tuple)):
            raise ContractError('Initial contents require a sequence of immutable sphere records')
        contents=tuple(self.initial_contents)
        if any(type(p) is not FixedSphereContent for p in contents) or len({p.content_id for p in contents})!=len(contents):
            raise ContractError('Persistent content IDs must be unique and explicitly described')
        entity_ids=['arm_0','floor',self.object_id,self.source_support.station_id,self.goal_support.station_id,
                    *(o.obstacle_id for o in obstacles),*(p.content_id for p in contents)]
        if len(set(entity_ids))!=len(entity_ids):
            raise ContractError('Robot, floor, body, support, obstacle and content IDs share one nonaliased namespace')
        # This also rejects unrepresentable transformed positions without moving anything.
        for particle in contents:self.source_body_world.transform_point(particle.center_body_m)
        object.__setattr__(self,'case_json',case_json)
        object.__setattr__(self,'physics_settings_json',settings_json)
        object.__setattr__(self,'initial_contents',contents)
        object.__setattr__(self,'world_sha256',digest(self.to_record()))

    @classmethod
    def from_inputs(cls, *, task_id, object_id, case, robot_model_sha256, model,
                    source_body_world, source_support, goal_body_world, goal_support,
                    nominal_source_hand_world, initial_contents, physics_settings):
        """Copy declared inputs. No existing diagnostic fixture is silently adopted."""
        return cls(task_id,object_id,_json_mapping(case,'case'),robot_model_sha256,model,
                   source_body_world,source_support,goal_body_world,goal_support,nominal_source_hand_world,
                   initial_contents,_json_mapping(physics_settings,'physics settings'))

    def case_record(self):return json.loads(self.case_json)

    @classmethod
    def from_record(cls, record):
        """Restore the exact declared world. Reject extra fields and stale hashes.

        The eight-ULP unit check below only restores serialized binary64 data.
        It is not a simulator pose or physical orientation tolerance.
        """
        try:
            if not isinstance(record, dict):raise ContractError('Fixed-world record must be a mapping')
            model=VesselModel.from_container(record['vessel_model']['container'])
            def pose(raw):
                if set(raw)!={'frame','position_m','quaternion_wxyz'} or raw['frame']!='world':
                    raise ContractError('Fixed-world pose schema/units changed')
                q=point(raw['quaternion_wxyz'],4)
                if (max(map(abs,q))>1. or abs(math.fsum(v*v for v in q)-1.)>8*math.ulp(1.)
                        or not any(q) or next(v for v in q if v!=0.)<0.
                        or any(v==0. and math.copysign(1.,v)<0. for v in q)):
                    raise ContractError('Serialized world quaternion must already be unit and sign-canonical')
                value=WorldPose(raw['position_m'],q)
                object.__setattr__(value,'quaternion_wxyz',q)
                return value
            def support(raw):
                return FixedSupportStation(raw['id'],raw['size_m'],raw['center_world_m'])
            result=cls.from_inputs(task_id=record['task_id'],object_id=record['object_id'],case=record['case'],
                robot_model_sha256=record['robot_model_sha256'],model=model,
                source_body_world=pose(record['source_body_world']),source_support=support(record['source_support']),
                goal_body_world=pose(record['goal_body_world']),goal_support=support(record['goal_support']),
                nominal_source_hand_world=pose(record['nominal_source_hand_world']),
                initial_contents=tuple(FixedSphereContent(p['id'],p['mass_kg'],p['radius_m'],p['center_body_m'])
                                       for p in record['initial_contents']),physics_settings=record['physics_settings'])
            if digest(result.to_record())!=digest(record):
                raise ContractError('Fixed-world serialized identity, units or derived hashes changed')
            return result
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            if isinstance(exc,ContractError):raise
            raise ContractError('Malformed serialized fixed world: '+str(exc)) from exc

    def physics_settings_record(self):return json.loads(self.physics_settings_json)

    def bind_design(self,design):
        return FixedWorldCandidate(self,design)

    def to_record(self):
        case=self.case_record();settings=self.physics_settings_record()
        return {'schema_version':'fixed-world-joint-task-v1','task_id':self.task_id,'object_id':self.object_id,
            'units':{'world_position':'m','body_position':'m','quaternion_order':'wxyz','base_yaw':'rad','mass':'kg','time':'s'},
            'case':case,'case_sha256':digest(case),'robot_model_sha256':self.robot_model_sha256,
            'vessel_model':{'container':self.model.container_record(),'configuration_sha256':digest(self.model.container_record()),
                            'geometry_sha256':self.model.geometry_sha256},
            'source_body_world':self.source_body_world.to_record(),'source_support':self.source_support.to_record(),
            'goal_body_world':self.goal_body_world.to_record(),'goal_support':self.goal_support.to_record(),
            'nominal_source_hand_world':self.nominal_source_hand_world.to_record(),
            'initial_contents':[p.to_record() for p in self.initial_contents],
            'initial_content_world_centers_m':[list(self.source_body_world.transform_point(p.center_body_m)) for p in self.initial_contents],
            'physics_settings':settings,'physics_settings_sha256':digest(settings),
            'endpoint_authority':'Fixed nominal source hand and fixed goal body. Legacy case hand endpoints do not move the world fixture.',
            'candidate_variables':['named robot base X/Y/yaw','joint command timing'],
            'intermediate_states_supplied':False,'integration_status':'PURE_FIXED_WORLD_SUPPORT_NOT_A_PHYSICAL_EVALUATOR',
            'native_model_identity_verified':False,'physical_initialization_performed':False,
            'physical_settings_approved':False,'material_mission_passed':False,'student_release_approved':False}


def fixed_world_fixture(world, *, pad_primitives=()):
    """Copy world fixture coordinates. Pad observations never position the tray."""
    if type(world) is not FixedWorldJointTask:raise ContractError('Use a validated fixed world')
    pads=json.loads(json.dumps(list(pad_primitives),allow_nan=False))
    return {'tray_initial_authored_root_world_m':list(world.source_body_world.position_m),
        'tray_initial_quaternion_wxyz':list(world.source_body_world.quaternion_wxyz),
        'source_support_size_m':list(world.source_support.size_m),
        'source_support_center_world_m':list(world.source_support.center_world_m),
        'goal_support_size_m':list(world.goal_support.size_m),
        'goal_support_center_world_m':list(world.goal_support.center_world_m),
        'pad_primitives':pads,'source':'Exact fixed-world coordinates. Candidate pad FK does not place any world object.',
        'fixed_world_sha256':world.world_sha256,'measured_grasp_transform':False}


@dataclass(frozen=True)
class FixedWorldCandidate:
    world: FixedWorldJointTask
    design: JointDesign

    def __post_init__(self):
        if type(self.world) is not FixedWorldJointTask or type(self.design) is not JointDesign:
            raise ContractError('Bind an immutable fixed world and explicit JointDesign')
        validated=JointDesign.from_record(self.design.to_record(),self.world.case_record())
        object.__setattr__(self,'design',validated)

    def to_record(self):
        design=self.design.to_record();base=self.design.base_placement
        return {'schema_version':'fixed-world-joint-candidate-v1','world_sha256':self.world.world_sha256,
            'design':design,'design_sha256':digest(design),
            'robot_base_world_position_m':[base.x_m,base.y_m,self.world.case_record()['base_z_m']],
            'robot_base_world_yaw_rad':base.yaw_rad,'requires_fresh_initialization_and_grasp':True,
            'physical_evaluation_performed':False}


@dataclass(frozen=True)
class DerivedGoalHandEndpoint:
    hand_world: WorldPose
    reconstructed_body_world: WorldPose
    fixed_world_sha256: str
    observed_held_binding_sha256: str
    roundtrip_position_error_m: float
    roundtrip_quaternion_component_error: float
    arithmetic_position_bound_m: float
    arithmetic_quaternion_component_bound: float

    def to_record(self):
        return {'hand_world':self.hand_world.to_record(),'reconstructed_body_world':self.reconstructed_body_world.to_record(),
            'fixed_world_sha256':self.fixed_world_sha256,'observed_held_binding_sha256':self.observed_held_binding_sha256,
            'roundtrip_position_error_m':self.roundtrip_position_error_m,
            'roundtrip_quaternion_component_error':self.roundtrip_quaternion_component_error,
            'arithmetic_position_bound_m':self.arithmetic_position_bound_m,
            'arithmetic_quaternion_component_bound':self.arithmetic_quaternion_component_bound,
            'bound_scope':'Binary64 composition roundoff only, not a physical drift tolerance.',
            'observed_candidate_base_binding_verified':False,'ik_or_collision_verified':False,
            'physical_grasp_verified':False,'world_fixture_moved':False}


def derive_goal_hand_endpoint(world,observed):
    """T_WH_goal = fixed T_WC_goal @ inverse(measured T_HC), no base substitution."""
    if type(world) is not FixedWorldJointTask or type(observed) is not HeldBodySpec:
        raise ContractError('Use a fixed world and a measured immutable HeldBodySpec')
    if (observed.object_id!=world.object_id or observed.model.geometry_sha256!=world.model.geometry_sha256
            or observed.model.container_json!=world.model.container_json):
        raise ContractError('Observed body/model differs from the fixed-world object')
    # The base/episode identity of this observation is deliberately not inferred.
    # A future initializer must bind it to this exact candidate separately.
    w,x,y,z=world.goal_body_world.quaternion_wxyz
    a,b,c,d=observed.hand_to_body.quaternion_hand_body_wxyz;b,c,d=-b,-c,-d
    q=(w*a-x*b-y*c-z*d,w*b+x*a+y*d-z*c,w*c-x*d+y*a+z*b,w*d+x*c-y*b+z*a)
    orientation=WorldPose((0.,0.,0.),q)
    offset=orientation.transform_point(observed.hand_to_body.translation_hand_m)
    position=tuple(v-o for v,o in zip(world.goal_body_world.position_m,offset))
    hand=WorldPose(position,orientation.quaternion_wxyz)
    reconstructed=observed.body_pose_at(hand)
    position_error=max(abs(a-b) for a,b in zip(reconstructed.position_m,world.goal_body_world.position_m))
    qa,qb=reconstructed.quaternion_wxyz,world.goal_body_world.quaternion_wxyz
    orientation_error=min(max(abs(a-b) for a,b in zip(qa,qb)),max(abs(a+b) for a,b in zip(qa,qb)))
    scale=max(1.,*(abs(v) for v in (*position,*offset,*world.goal_body_world.position_m)))
    position_bound=128*math.ulp(scale);orientation_bound=128*math.ulp(1.)
    if (not math.isfinite(position_bound) or position_error>position_bound or orientation_error>orientation_bound):
        raise ContractError('Goal-hand inverse composition failed its binary64 roundtrip check')
    workspace=world.case_record()['tool_workspace_m']
    if any(not lo<=x<=hi for x,(lo,hi) in zip(hand.position_m,workspace)):
        raise ContractError('Derived goal hand endpoint is outside the nominal world tool workspace')
    return DerivedGoalHandEndpoint(hand,reconstructed,world.world_sha256,observed.binding_sha256,
        position_error,orientation_error,position_bound,orientation_bound)
