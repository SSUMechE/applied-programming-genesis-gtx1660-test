"""Algorithm-neutral endpoint planning for the current loaded-carry checkpoint."""
import copy
from .contracts import (ContractError,PlanningBudgetExceeded,PlanningRequest,digest,point,validate_path_document)
from .design import SPEED_PROFILES
from .free_planning import CarryStateSpace,_CarryOracle
from .geometry import distance,quaternion_angle
from .planning import _source_fingerprint
from .query_diagnostics import bind_query_input, completed_query_diagnostic, diagnostic_record


class QueryBudgetOracle:
    """One charge per public state/edge call. Internal geometry is not recharged."""
    def __init__(self,oracle,max_queries):
        if type(max_queries) is not int or max_queries<0:
            raise ContractError("Query budget must be a nonnegative integer")
        self._oracle,self._limit=oracle,max_queries
        self._states=self._edges=self._exhausted=0
        self._last_diagnostic=diagnostic_record()
        self._last_diagnostic.update(query_ordinal=None,attempt_ordinal=0,remaining_queries=max_queries,
                                     query_input=None,query_input_sha256=None)

    @property
    def queries_used(self): return self._states+self._edges

    @property
    def remaining_queries(self): return self._limit-self.queries_used

    def _charge(self,kind):
        if self.remaining_queries==0:
            self._exhausted+=1
            raise PlanningBudgetExceeded("Shared endpoint validity-query budget exhausted")
        if kind=="state": self._states+=1
        else: self._edges+=1

    def state_valid(self,state):
        return self._call("state",state)

    def edge_valid(self,first,second):
        return self._call("edge",first,second)

    def _call(self,kind,*arguments):
        try:
            self._charge(kind)
        except PlanningBudgetExceeded:
            self._last_diagnostic=diagnostic_record("budget_exhausted",reason="query_budget_exhausted",query_kind=kind)
            self._bind_diagnostic(arguments,charged=False)
            raise
        try:
            answer=getattr(self._oracle,kind+"_valid")(*arguments)
        except Exception as error:
            self._last_diagnostic=diagnostic_record("query_error",reason=type(error).__name__,query_kind=kind)
            self._last_diagnostic["details"]={"message":str(error)}
            self._bind_diagnostic(arguments)
            raise
        self._last_diagnostic=completed_query_diagnostic(self._oracle,kind,bool(answer))
        self._bind_diagnostic(arguments)
        return answer

    def _bind_diagnostic(self,arguments,*,charged=True):
        bind_query_input(self._last_diagnostic,arguments)
        self._last_diagnostic.update(query_ordinal=self.queries_used if charged else None,
            attempt_ordinal=self.queries_used+self._exhausted,remaining_queries=self.remaining_queries)

    def last_query_diagnostic(self):
        """Read the last public-call snapshot without another query or charge."""
        return copy.deepcopy(self._last_diagnostic)

    def record(self):
        return {"max_queries":self._limit,"queries_used":self.queries_used,"state_queries":self._states,
                "edge_queries":self._edges,"exhausted_call_attempts":self._exhausted}


def _checked(path,request,oracle,limit):
    if path is None:
        raise ContractError("Student planner found no path within its declared search and query budget")
    if not isinstance(path,(tuple,list)) or not 2<=len(path)<=limit:
        raise ContractError(f"Physical carry path needs 2..{limit} states")
    states=tuple(point(p,len(request.bounds)) for p in path)
    if states[0]!=request.start or states[-1]!=request.goal:
        raise ContractError("Planner/refinement must preserve exact endpoint states")
    if not all(oracle.state_valid(p) for p in states) or not all(oracle.edge_valid(a,b) for a,b in zip(states,states[1:])):
        raise ContractError("Planner/refinement returned an invalid whole-edge carry path")
    return states


def endpoint_route_document(case,strategy,*,speed_profile="careful",algorithm_hint=None):
    """Use the actual plan_path hook. Neither A* graphs nor RRT trees are supplied."""
    if case.get("planning_mode")!="endpoint_planning" or case.get("motion_scope")!="carry_only_development":
        raise ContractError("Use the current endpoint-only carry development contract")
    for key in ("pose_nodes","nodes","graph","candidate_carry_heights_m","deposit_segment","deposit_segments"):
        if key in case: raise ContractError("No intermediate solution states or graph may be supplied")
    if algorithm_hint is not None and algorithm_hint not in getattr(strategy,"supported_algorithms",()):
        raise ContractError("This strategy does not declare support for the optional algorithm hint. Omit --algorithm and implement one plan_path, or choose a supporting comparison module.")
    space=CarryStateSpace(case["planning_state_space"],case["workspace_m"])
    oracle=_CarryOracle(case,space)
    transfers=case["transfers"]
    budget=case["planning_budget"]
    ids=[task["id"] for task in transfers]
    maximum=budget["queries_per_transfer"]
    if (type(maximum) is not int or maximum<1 or len(set(ids))!=len(ids)
            or budget["total_queries"]!=maximum*len(transfers) or set(budget["seeds_by_segment"])!=set(ids)):
        raise ContractError("Teacher query budget must bind every carry and its fixed seed")
    options=copy.deepcopy(case.get("selected_planner_options",case.get("planner_defaults",{})))
    if algorithm_hint is not None: options["algorithm"]=algorithm_hint
    profiles=case.get("selected_segment_speed_profiles",{})
    if speed_profile not in SPEED_PROFILES or not set(profiles)<=set(ids):
        raise ContractError("Unknown speed profile or target segment")
    segments,calls=[],[]
    refine=getattr(strategy,"refine_path",None)
    for transfer in transfers:
        if any(key in transfer for key in ("pose_nodes","extra_nodes","nodes","graph","waypoints","quaternion_waypoints")):
            raise ContractError("Transfer must expose endpoints without an intermediate solution route")
        request=PlanningRequest(space.endpoint(transfer,"start"),space.endpoint(transfer,"goal"),space.bounds,
                                maximum,budget["seeds_by_segment"][transfer["id"]],options)
        metered=QueryBudgetOracle(oracle,maximum)
        try:
            raw=strategy.plan_path(request,metered)
        except PlanningBudgetExceeded as error:
            raise ContractError("Planning exhausted the shared query budget without returning a route") from error
        raw=_checked(raw,request,oracle,10000)
        planning_queries=metered.queries_used
        status="not_requested"
        refined=raw
        if refine is not None:
            try:
                refined=refine(raw,metered.edge_valid)
                status="returned"
            except PlanningBudgetExceeded:
                refined=raw
                status="query_budget_exhausted_retained_student_raw_path"
        path=_checked(refined,request,oracle,200)
        poses=[space.pose(state) for state in path]
        profile=profiles.get(transfer["id"],speed_profile)
        if profile not in SPEED_PROFILES: raise ContractError("Unknown segment speed profile")
        timing=sum(distance(a["xyz_m"],b["xyz_m"])+.10*quaternion_angle(a["quaternion_wxyz"],b["quaternion_wxyz"])
                   for a,b in zip(poses,poses[1:]))
        duration=max(1.0,timing/SPEED_PROFILES[profile]["nominal_path_progress_speed_m_s"])
        if duration>30: raise ContractError("Carry exceeds the supplied30-s limit. Improve its path or choose a permitted speed.")
        segment={"id":transfer["id"],"robot_id":transfer["robot_id"],"object_id":transfer["object_id"],
            "phase":"transfer","entry_phase":transfer.get("entry_phase","after_lift"),
            "waypoints":[pose["xyz_m"] for pose in poses],"duration_s":duration,"speed_profile":profile}
        if space.dimension==4: segment["quaternion_waypoints"]=[pose["quaternion_wxyz"] for pose in poses]
        segments.append(segment)
        calls.append({"segment_id":transfer["id"],"request":request.to_record(),
            "raw_states":[list(p) for p in raw],"raw_states_sha256":digest(raw),
            "refined_states":[list(p) for p in path],"refined_states_sha256":digest(path),
            "exported_waypoints":copy.deepcopy(segment["waypoints"]),
            "exported_quaternion_waypoints":copy.deepcopy(segment.get("quaternion_waypoints")),
            "query_meter":metered.record(),"planning_queries":planning_queries,"refinement_status":status,
            "direct_edge_proxy_valid":oracle.edge_valid(request.start,request.goal)})
    document={"schema_version":"transport-path-v1","coordinate_frame":"world_xyz_m","case_id":case["case_id"],
        "scenario_content_sha256":case.get("source_scenario_content_sha256",digest(case)),
        "algorithm":"endpoint_planning","method_label":algorithm_hint or getattr(strategy,"planner_label","student_supplied"),
        "motion_scope":"carry_only_development","segments":segments,"speed_profile":speed_profile,
        "planner_provenance":{"schema_version":"endpoint-planner-provenance-v1","state_space":space.specification,
            "workspace_m":case["workspace_m"],"budget":copy.deepcopy(budget),
            "plan_path_source_sha256":_source_fingerprint(strategy.plan_path),
            "refine_path_source_sha256":_source_fingerprint(refine) if refine is not None else None,"calls":calls},
        "validation_scope":"Endpoint-only student planner for loaded carries. Method labels are descriptive, not algorithm proof. Full robot/IK and all non-carry motions remain separate."}
    if "design_content_sha256" in case: document["design_content_sha256"]=case["design_content_sha256"]
    return validate_path_document(document,require_planner_provenance=True)


def validate_planner_provenance(document):
    provenance=document["planner_provenance"]
    if (provenance.get("schema_version")!="endpoint-planner-provenance-v1"
            or document.get("algorithm")!="endpoint_planning" or document.get("motion_scope")!="carry_only_development"):
        raise ContractError("Unsupported endpoint planner provenance/scope")
    space=CarryStateSpace(provenance["state_space"],provenance["workspace_m"])
    calls=provenance.get("calls")
    segments=document["segments"]
    budget=provenance["budget"]
    if not isinstance(calls,list) or len(calls)!=len(segments) or not calls:
        raise ContractError("Planner provenance must cover all executed carries exactly once")
    if budget["total_queries"]!=budget["queries_per_transfer"]*len(segments):
        raise ContractError("Planner total query budget differs from its carry budgets")
    for call,segment in zip(calls,segments):
        if call.get("segment_id")!=segment["id"]: raise ContractError("Planner trace carry identity/order mismatch")
        request=PlanningRequest(**call["request"])
        if (request.bounds!=space.bounds or request.max_queries!=budget["queries_per_transfer"]
                or request.seed!=budget["seeds_by_segment"][segment["id"]]):
            raise ContractError("Planner trace changed teacher bounds, budget or seed")
        meter=call["query_meter"]
        names=("max_queries","queries_used","state_queries","edge_queries","exhausted_call_attempts")
        if any(type(meter.get(name)) is not int or meter[name]<0 for name in names):
            raise ContractError("Planner query counters must be nonnegative integers")
        if type(call.get("planning_queries")) is not int or type(call.get("direct_edge_proxy_valid")) is not bool:
            raise ContractError("Planner query split and direct-edge result must have explicit types")
        if (meter["max_queries"]!=request.max_queries or meter["queries_used"]!=meter["state_queries"]+meter["edge_queries"]
                or not 0<=call["planning_queries"]<=meter["queries_used"]<=request.max_queries):
            raise ContractError("Planner query accounting is inconsistent")
        for name,limit in (("raw_states",10000),("refined_states",200)):
            states=call.get(name)
            if not isinstance(states,(list,tuple)) or not 2<=len(states)<=limit:
                raise ContractError("Planner trace path size is invalid")
            checked=[point(state,space.dimension) for state in states]
            if checked[0]!=request.start or checked[-1]!=request.goal or call.get(name+"_sha256")!=digest(states):
                raise ContractError("Planner path endpoints/content hash mismatch")
            for state in checked: space.pose(state)
        status=call.get("refinement_status")
        if status not in ("not_requested","returned","query_budget_exhausted_retained_student_raw_path"):
            raise ContractError("Unknown refinement completion status")
        if status=="not_requested" and call["raw_states"]!=call["refined_states"]:
            raise ContractError("No-refinement trace must retain the original student path")
        if status=="query_budget_exhausted_retained_student_raw_path" and (
                call["raw_states"]!=call["refined_states"] or meter["queries_used"]!=request.max_queries
                or meter["exhausted_call_attempts"]<1):
            raise ContractError("Refinement budget interruption must retain the actual raw student route")
        poses=[space.pose(state) for state in call["refined_states"]]
        xyz=[pose["xyz_m"] for pose in poses]
        quats=[pose["quaternion_wxyz"] for pose in poses] if space.dimension==4 else None
        if (segment["waypoints"]!=xyz or call.get("exported_waypoints")!=xyz
                or segment.get("quaternion_waypoints")!=quats or call.get("exported_quaternion_waypoints")!=quats):
            raise ContractError("Planner states do not map to exact executed XYZ/orientation points")
