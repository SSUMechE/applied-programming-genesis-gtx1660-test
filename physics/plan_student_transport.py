"""Plan student seven-joint transport paths on CPU and export frozen GPU-review inputs."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.abc
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PLANNING_OPTIONS = {"ik_sample_cap": 10000}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


class NoGenesis(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.blocked_attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "genesis" or fullname.startswith("genesis."):
            self.blocked_attempts.append(fullname)
            raise RuntimeError("Genesis import is forbidden in CPU planning: " + fullname)



def load_strategy(module_name, *, internal_teacher=False):
    """Select the requested project callback, never a replacement strategy."""
    from transport_challenge.contracts import ContractError
    if not isinstance(module_name, str) or not module_name:
        raise ContractError("A student strategy module is required")
    if not module_name.startswith("student.") and not internal_teacher:
        raise ContractError("Nonstudent strategies require explicit --internal-teacher-strategy")
    module = importlib.import_module(module_name)
    source = Path(inspect.getsourcefile(module)).resolve()
    allowed = ROOT if internal_teacher else ROOT / "student"
    if not source.is_relative_to(allowed) or not callable(getattr(module, "plan_path", None)):
        raise ContractError("Selected strategy must provide a project-owned plan_path")
    function_source = Path(inspect.getsourcefile(module.plan_path)).resolve()
    if not function_source.is_relative_to(allowed):
        raise ContractError("Strategy callback source escaped its declared project namespace")
    return module


def optional_strategy_receipt(strategy):
    """Optional caller-supplied telemetry is not an acceptance predicate."""
    rows = getattr(strategy, "planning_receipts", None)
    if not isinstance(rows, list) or not rows or not isinstance(rows[-1], dict):
        return None
    try:
        json.dumps(rows[-1], allow_nan=False)
    except (TypeError, ValueError):
        return None
    return {"scope": "Optional selected-strategy telemetry. Supplied path validation is authoritative.",
            "value": copy.deepcopy(rows[-1])}


def block_integration(mujoco):
    """Forbid CPU physical stepping while retaining geometry and kinematics."""
    state = {"originals": {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2")},
             "attempts": []}
    def blocked(name):
        def forbidden_step(*unused_args, **unused_kwargs):
            state["attempts"].append(name)
            raise RuntimeError("CPU planning forbids physical integration: " + name)
        return forbidden_step
    for name in state["originals"]:
        setattr(mujoco, name, blocked(name))
    return state


def restore_integration(mujoco, state):
    for name, original in state["originals"].items():
        setattr(mujoco, name, original)


def native_geometry_precision(geometry, scenario, precision):
    """Use the selected native command representation before saving any path."""
    from transport_challenge.contracts import ContractError
    if precision not in ("32", "64"):
        raise ContractError("Require an explicit supported native precision")
    if precision == "64":
        # CpuGeometry defaults to the bead float32 recipe. Sand64 uses the
        # original double XML ranges and declared commands instead.
        names = scenario["case"]["arm_joint_names"]
        ids = [geometry.model.joint("arm_0__" + name).id for name in names]
        geometry.bounds = tuple(tuple(float(value) for value in row)
                                for row in geometry.model.jnt_range[ids])
        geometry.initial_q_by_arm = {
            robot["id"]: [float(value) for value in
                          (*robot["q_rad"], *robot["finger_q_m"])]
            for robot in scenario["robots"]}
    return geometry


def cpu_geometry_evidence(geometry, material, precision):
    """Separate selected container geometry from uncomputed particle physics."""
    record = geometry.evidence()
    record["selected_material"] = material
    record["native_command_precision"] = precision
    record["particle_dynamics_computed"] = False
    record["particle_retention_verified"] = False
    if material != "beads":
        record["contents_geometry_scope"] = (
            "No rigid bead substitutes or simulated particle states. All robot, "
            "map and selected container collision geometry is checked. Actual "
            "native particle quantity, deformation and retention need GPU review.")
    return record


def _worker_main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--strategy", default="student.fixed_example")
    parser.add_argument("--material", choices=("beads",), default="beads")
    parser.add_argument("--precision", choices=("32", "64"), default=None,
                        help="Native recipe precision. Defaults to Sand64 and otherwise32.")
    parser.add_argument("--runtime-profile", choices=("default", "cpic_cache_64_tol12"), default=None,
                        help="Explicit particle recipe. Sand defaults to the existing FP64 cache profile.")
    parser.add_argument("--particle-recheck-mode", choices=("cpu_geometry_actual_boundary", "native_full"), default=None,
                        help="Particle saved-path recheck backend. Default uses CPU geometry at the actual paused boundary.")
    parser.add_argument("--scope", choices=("mission", "one-carry"), default="mission")
    parser.add_argument("--internal-teacher-strategy", action="store_true", help="Explicit internal reference trial only. No student fallback.")
    parser.add_argument("--edge-timing-hook", action="store_true",
                        help="Call the selected strategy choose_edge_durations_s after geometric validation. Explicit times require unit joint_speed_scale.")
    parser.add_argument("--task", type=Path, default=ROOT / "cases/three_arm_cell_bead16_v2")
    parser.add_argument("--design", type=Path, default=ROOT / "example/design.json")
    parser.add_argument("--max-stages", type=int, default=None, help="Optional incomplete CPU debug prefix. Default is the complete policy sequence.")
    parser.add_argument("--wall-limit-s", type=float, default=1800.)
    parser.add_argument("--post-place-policy", choices=("kinematic_return", "instant_reset", "clear_handoff", "return_home"), default="kinematic_return")
    args = parser.parse_args(argv)
    if args.material != "beads" or args.scope != "one-carry":
        parser.error("This public hardware test requires --material beads --scope one-carry")
    if args.material == "beads" and args.particle_recheck_mode is not None:
        parser.error("Particle recheck selection does not apply to the rigid-bead recipe")
    if args.scope == "one-carry" and args.post_place_policy != "kinematic_return":
        parser.error("One-carry has no donor return. Do not select a full-mission post-place policy.")
    motion_count = 6 if args.post_place_policy in ("instant_reset", "kinematic_return") else 8
    if args.max_stages is None:
        args.max_stages = motion_count
    out = args.output.resolve()
    if out.exists() or out == ROOT or not 1 <= args.max_stages <= motion_count:
        parser.error("Use a fresh output and a prefix within the selected policy motion count")
    if not 10 <= args.wall_limit_s <= 3600:
        parser.error("CPU wall budget must be10..3600 seconds")
    if args.material != "beads" and args.scope != "one-carry":
        out.mkdir(parents=True, exist_ok=False)
        summary = {"status": "CPU_MATERIAL_SCOPE_NOT_IMPLEMENTED", "material": args.material,
            "planning_scope": args.scope, "completed_stages": 0,
            "failure": {"type": "UnsupportedMaterialScope", "message": "Particle export requires the explicit one-carry scope. Full particle missions are not implemented."},
            "genesis_initialized": False, "native_physical_steps": 0, "cpu_physical_steps": 0,
            "replayable_path_exported": False, "full_mission_passed": False,
            "physical_transport_verified": False, "student_release_approved": False}
        save(out / "CPU_PLANNING_RESULT.json", summary)
        print(json.dumps(summary), flush=True)
        return 2
    blocker = NoGenesis()
    if any(name == "genesis" or name.startswith("genesis.") for name in sys.modules):
        parser.error("CPU planning requires a fresh process without Genesis imports")
    sys.meta_path.insert(0, blocker)
    import mujoco
    original_step = block_integration(mujoco)
    from dataclasses import replace
    import numpy as np
    from physics.automatic_runtime_binding import prepare_automatic_scenario
    from physics.carried_body import VesselModel, WorldPose, measure_hand_to_body
    from physics.provided_manipulation import grasp_endpoint_candidates, solve_endpoint
    from physics.source_guard import PinnedSourceGuard
    from physics.saved_motion_planning import kinematic_return_schedule
    from physics.one_carry_planning import material_inputs
    from transport_challenge.contracts import digest, ContractError
    from transport_challenge.joint_context import JointBasePlacement, JointTaskContext
    from transport_challenge.joint_kinematics import BoundedKinematics
    from transport_challenge.joint_planning import JointLimits, make_joint_request, plan_joint_motion
    from transport_challenge.three_arm_design import ThreeArmDesign, bind_three_arm_design
    from supplied_support.cpu_geometry import CpuGeometry, _rotation, _quat
    from supplied_support.clearance_goal import select_clearance_goal

    out.mkdir(parents=True, exist_ok=False)
    task = args.task.resolve()
    manifest = read(task / "manifest.json")
    for relative, expected in manifest["files"].items():
        if sha(ROOT / relative) != expected:
            raise ContractError("Task source differs: " + relative)
    original_world, original_scenario = read(task / "world.json"), read(task / "initialization.json")
    design = ThreeArmDesign.from_record(read(args.design), original_world)
    bound = bind_three_arm_design(design, original_world, original_scenario)
    world = bound["world"]
    scenario, automatic_overlay = prepare_automatic_scenario(bound["scenario"])
    world, scenario = material_inputs(world, scenario, args.material,
                                     precision=args.precision, runtime_profile=args.runtime_profile)
    particle_request = copy.deepcopy(world.get("particle_material_request"))
    native_precision = particle_request.get("precision", "32") if particle_request else "32"
    particle_reference = ROOT / "cases/three_arm_cell_v1/physics_settings_reference.json" if particle_request else None
    geometry = native_geometry_precision(CpuGeometry(world, scenario), scenario, native_precision)
    sources = [*sorted((ROOT / "physics").glob("*.py")), *sorted((ROOT / "transport_challenge").glob("*.py")),
               Path(__file__).resolve(), Path(inspect.getsourcefile(CpuGeometry)).resolve(),
               *sorted((ROOT / "supplied_support").glob("*.py")),
               *task.glob("*.json"), args.design.resolve()]
    if particle_reference is not None:
        sources.append(particle_reference)
    try:
        strategy = load_strategy(args.strategy, internal_teacher=args.internal_teacher_strategy)
    except BaseException as exc:
        restore_integration(mujoco, original_step)
        summary = {"status": "CPU_PLAN_FAILED", "planning_scope": args.scope, "material": args.material,
            "completed_stages": 0, "failure": {"type": type(exc).__name__, "message": str(exc)},
            "genesis_imported_modules": [name for name in sys.modules if name == "genesis" or name.startswith("genesis.")],
            "blocked_genesis_import_attempts": blocker.blocked_attempts,
            "blocked_integration_attempts": list(original_step["attempts"]),
            "genesis_initialized": False, "native_physical_steps": 0, "cpu_physical_steps": 0,
            "source_bindings_unchanged": None, "process_wall_s": time.perf_counter()-started,
            "full_mission_passed": False, "physical_transport_verified": False, "student_release_approved": False}
        save(out / "CPU_PLANNING_RESULT.json", summary)
        print(json.dumps(summary), flush=True)
        return 1
    strategy_file = Path(inspect.getsourcefile(strategy)).resolve()
    sources.append(strategy_file)
    sources.extend(sorted((ROOT / "student").rglob("*.py")))
    sources.extend(Path(module.__file__).resolve() for module in list(sys.modules.values())
        if getattr(module, "__file__", None) and Path(module.__file__).suffix == ".py"
        and Path(module.__file__).resolve().is_relative_to(ROOT))
    sources.append(Path(inspect.getsourcefile(select_clearance_goal)).resolve())
    pins = {str(path.resolve()): sha(path) for path in sources}
    pins.update(geometry.source_pins)
    if particle_request:
        # This reads source/recipe provenance only. Missing factory inputs fail
        # closed instead of exporting an incompletely bound material recipe.
        from physics.particle_episode_bridge import provider_source_pins
        pins.update(provider_source_pins(args.material, particle_request.get("runtime_profile", "default")))
    guard = PinnedSourceGuard(pins, full_interval_s=1.)
    if not guard.before_request():
        raise ContractError("CPU source boundary failed")
    def check():
        if time.perf_counter() - started > args.wall_limit_s:
            raise TimeoutError("CPU complete planning wall budget exceeded")
        if guard() is not True:
            raise ContractError("CPU planning source changed")

    class GuardedOracle:
        def __init__(self, cpu):
            self.cpu = cpu
        def state_valid(self, q):
            check(); answer = self.cpu.state_valid(q); check()
            return answer
        def edge_valid(self, a, b):
            check(); answer = self.cpu.edge_valid(a, b); check()
            return answer
        def last_query_diagnostic(self):
            return self.cpu.last_query_diagnostic()

    route = [world["source_support"]["id"], *[r["id"] for r in world["relay_supports"]], world["goal_support"]["id"]]
    commands = {name: {"q_rad": values[:7], "finger_q_m": values[7:]} for name, values in geometry.initial_q_by_arm.items()}
    initial_commands = copy.deepcopy(commands)
    body_pose = WorldPose(**{key: scenario["body_pose"][key] for key in ("position_m", "quaternion_wxyz")})
    model = VesselModel.from_container(scenario["container"])
    stages = []
    queue = [(arm, kind) for arm in ("arm_0", "arm_1", "arm_2")
             for kind in (("approach", "carry", "retreat") if arm != "arm_2" and args.post_place_policy not in ("instant_reset", "kinematic_return") else ("approach", "carry"))]
    resets = ([{"after_stage_id": arm+"__carry", "arm_id": arm,
        "target_command": copy.deepcopy(initial_commands[arm]), "zero_joint_velocity": True,
        "physical_steps": 0, "simulation_duration_s": 0.} for arm in ("arm_0", "arm_1")]
        if args.post_place_policy == "instant_reset" else [])
    returns = []
    bundle = {"schema_version": "saved-three-arm-mission-v1", "post_place_policy": args.post_place_policy,
        "world_sha256": digest(world),
        "scenario_sha256": digest(scenario), "design_sha256": design.sha256, "task_manifest_sha256": sha(task / "manifest.json"),
        "payload": args.material, "rigid_runtime_profile": world.get("rigid_runtime_profile"), "source_pins": pins,
        "initial_commands_by_arm": initial_commands, "stages": stages, "post_place_resets": resets,
        "post_place_returns": returns,
        "initial_representation": (
            "Declared seven-joint commands, open fingers and original XML limits retain native float64 values. No later path/time snapping."
            if native_precision == "64" else
            "Each declared7q and provided .04m fingers converted once to the existing native float32 representation. No later path/time snapping."),
        "automatic_initialization_overlay": automatic_overlay,
        "offline_planner": {"module": strategy.__name__, "source_sha256": sha(strategy_file),
            "plan_path_source_sha256": hashlib.sha256(inspect.getsource(strategy.plan_path).encode("utf-8")).hexdigest(),
            "cpu_pipeline_module": "physics.plan_student_transport", "seed": 71,
            "strategy_role": "internal_teacher_reference" if args.internal_teacher_strategy else "student",
            "post_place_policy": args.post_place_policy,
            "max_queries": 50000, "endpoint_attempts": 32, "joint_speed_scale": bound["joint_speed_scale"],
            "planning_options": copy.deepcopy(PLANNING_OPTIONS)},
        "nominal_only": True, "physical_grasp_verified": False, "physical_transport_verified": False,
        "full_mission_passed": False, "student_release_approved": False}
    if particle_request:
        bundle["particle_material_request"] = particle_request
        bundle["particle_recheck_mode"] = args.particle_recheck_mode or "cpu_geometry_actual_boundary"
        bundle["cpu_process_supervisor"] = {"status": "WORKER_CANDIDATE_PENDING"}
        bundle["particle_physics_reference"] = {"path":str(particle_reference.resolve()),
                                                 "sha256":sha(particle_reference)}
    constants = {}
    for node in ast.parse((ROOT / "physics/episode_runtime.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("PROTOCOL", "TRANSPORT_PROTOCOL"):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    bundle["provided_wait_policy"] = {**constants, "additional_material_wait_s": .5, "maximum_additional_material_wait_s": 5.,
        "policy": "Original supplied actual support/rest gates decide whether the source-pinned conditional material wait is needed. No student route/timing change."}
    save(out / "source_pins.json", pins)
    failure = None
    try:
        edge_timing_hook = None
        if args.edge_timing_hook:
            edge_timing_hook = getattr(strategy, "choose_edge_durations_s", None)
            if not callable(edge_timing_hook):
                raise ContractError("Timing opt-in requires the selected strategy choose_edge_durations_s callback")
        nominal_transform = None
        fixed_endpoints = None
        prepared_recipient_terminals = {}
        if args.scope == "one-carry":
            from physics.one_carry_planning import make_initial_fixture, apply_initial_fixture
            args.post_place_policy = "one_carry"
            queue = [("arm_0", "carry")]
            old_cpu = geometry.configure("arm_0", commands, body_pose)
            endpoint_limits = old_cpu.bounds
            endpoint_kin = BoundedKinematics(old_cpu, endpoint_limits, scenario["case"]["tool_workspace_m"],
                max_fk_calls=32, max_ik_calls=32, wall_check=check)
            grasp = grasp_endpoint_candidates(model, body_pose.to_record())[0].to_record()
            if strategy.__name__ == "student.fixed_example":
                from supplied_support.fixed_example_endpoints import bind_fixed_example_endpoints, validate_fixed_endpoint
                fixed_endpoints = bind_fixed_example_endpoints(world, scenario, design.to_record())
                selected = validate_fixed_endpoint(endpoint_kin, GuardedOracle(old_cpu),
                    fixed_endpoints[0], grasp, wall_check=check)
                bundle["offline_planner"].update(endpoint_input_mode="supplied_fixed_joint_coordinates",
                    endpoint_attempts=0, fixed_source_endpoint_validation=selected)
            else:
                selected = solve_endpoint(endpoint_kin, GuardedOracle(old_cpu), tuple(commands["arm_0"]["q_rad"]),
                    (grasp,), attempts=32, seed=71, wall_check=check)
            if selected["status"] != "ENDPOINT_FOUND":
                raise ContractError("CPU supplied source grasp endpoint not found")
            # Canonicalize declared INITIAL before RRT. Frozen output is never rounded during replay.
            q_initial = [float(value) if native_precision == "64" else float(np.float32(value))
                         for value in selected["q_rad"]]
            fixture = make_initial_fixture(world, scenario, q_initial)
            scenario = apply_initial_fixture(world, scenario, fixture)
            geometry = native_geometry_precision(CpuGeometry(world, scenario), scenario, native_precision)
            commands = {name: {"q_rad": values[:7], "finger_q_m": values[7:]}
                for name, values in geometry.initial_q_by_arm.items()}
            initial_commands = copy.deepcopy(commands)
            if particle_request:
                from physics.episode_runtime import validate_inputs
                validate_inputs(world, scenario, None, read(particle_reference),
                    student_limits=JointLimits(geometry.bounds,
                        scenario["case"]["motion_limits"]["velocity_rad_s"],
                        scenario["case"]["motion_limits"]["acceleration_rad_s2"]))
            cpu_start = geometry.configure("arm_0", commands, body_pose)
            if not GuardedOracle(cpu_start).state_valid(tuple(q_initial)):
                raise ContractError("Declared native-precision source grasp is not geometrically valid")
            hand = cpu_start.fk(q_initial)
            nominal_transform = measure_hand_to_body(WorldPose(hand.position_m, hand.quaternion_wxyz), body_pose).to_record()
            bundle.update(schema_version="saved-one-carry-v1", scope="one_carry", post_place_policy="one_carry",
                scenario_sha256=digest(scenario), initial_fixture=fixture, initial_commands_by_arm=initial_commands,
                post_place_resets=[], post_place_returns=[])
            bundle["offline_planner"]["post_place_policy"] = "one_carry"
        save(out / "input_identity.json", {"world": world, "scenario": scenario, "design": design.to_record()})
        for index, (arm, kind) in enumerate(queue[:args.max_stages]):
            check()
            stage_started = time.perf_counter()
            arm_index = int(arm[-1])
            stage_id = arm + (("__clear_handoff" if args.post_place_policy == "clear_handoff" else "__return_initial")
                              if kind == "retreat" else "__" + kind)
            nominal_start_body = body_pose.to_record()
            destination = route[arm_index+1] if kind == "carry" else None
            support = route[arm_index+1] if kind == "retreat" else route[arm_index]
            cpu = geometry.configure(arm, commands, body_pose, holder_arm_id=arm if kind == "carry" else None,
                                     hand_to_body=nominal_transform if kind == "carry" else None)
            oracle = GuardedOracle(cpu)
            start_q = tuple(commands[arm]["q_rad"])
            limits = JointLimits(cpu.bounds, scenario["case"]["motion_limits"]["velocity_rad_s"], scenario["case"]["motion_limits"]["acceleration_rad_s2"])
            if not oracle.state_valid(start_q):
                raise ContractError("CPU stage start invalid: " + json.dumps(cpu.last_failure))
            nominal_goal_body = nominal_command_body = nominal_hand = None
            terminal_started = time.perf_counter()
            if kind == "retreat" and args.post_place_policy == "clear_handoff":
                next_arm = "arm_" + str(arm_index+1)
                terminal = select_clearance_goal(geometry, commands, body_pose, model, arm, next_arm,
                    recipient_seed=71+index+1, check=check)
                goal_q = tuple(terminal["q_rad"])
                nominal_hand = copy.deepcopy(terminal["nominal_goal_hand_world"])
                prepared_recipient_terminals[next_arm] = copy.deepcopy(terminal["recipient_terminal"])
                # The search temporarily configured the next robot. Restore an
                # owned donor snapshot before invoking the original RRT.
                cpu = geometry.configure(arm, commands, body_pose)
                oracle = GuardedOracle(cpu)
            elif kind == "retreat":
                goal_q = tuple(initial_commands[arm]["q_rad"])
                terminal = {"schema_version": "cpu-nominal-return-endpoint-v1", "q_rad": list(goal_q), "native_ik_performed": False}
            else:
                if kind == "approach":
                    nominal_hand = grasp_endpoint_candidates(model, body_pose.to_record())[0].to_record()
                else:
                    nominal_goal_body = copy.deepcopy(scenario["supported_body_poses"][destination])
                    nominal_command_body = copy.deepcopy(nominal_goal_body)
                    nominal_command_body["position_m"][2] -= .0003
                    rotation = _rotation(nominal_command_body["quaternion_wxyz"]) @ _rotation(nominal_transform["quaternion_hand_body_wxyz"]).T
                    position = np.asarray(nominal_command_body["position_m"]) - rotation @ nominal_transform["translation_hand_m"]
                    nominal_hand = WorldPose(position.tolist(), _quat(rotation).tolist()).to_record()
                endpoint_kinematics = BoundedKinematics(cpu, cpu.bounds, scenario["case"]["tool_workspace_m"],
                    max_fk_calls=32, max_ik_calls=32, wall_check=check)
                if kind == "approach" and arm in prepared_recipient_terminals:
                    raw_terminal = prepared_recipient_terminals.pop(arm)
                    if not oracle.state_valid(raw_terminal["q_rad"]):
                        raise ContractError("Computed parked peers no longer permit the saved next-grasp endpoint")
                elif fixed_endpoints is not None:
                    raw_terminal = validate_fixed_endpoint(endpoint_kinematics, oracle,
                        fixed_endpoints[1], nominal_hand, wall_check=check)
                else:
                    raw_terminal = solve_endpoint(endpoint_kinematics, oracle, start_q, (nominal_hand,), attempts=32, seed=71+index, wall_check=check)
                terminal = {"schema_version": "cpu-nominal-terminal-endpoint-v1", "cpu_solver_result": raw_terminal,
                            "physical_grasp_verified": False, "native_ik_performed": False}
                if raw_terminal["status"] != "ENDPOINT_FOUND":
                    raise ContractError("CPU nominal endpoint not found for " + stage_id + ": " + json.dumps(cpu.last_failure))
                goal_q = tuple(raw_terminal["q_rad"])
            terminal_wall_s = time.perf_counter() - terminal_started
            recipe = next(r for r in scenario["robots"] if r["id"] == arm)
            context_case = copy.deepcopy(scenario["case"])
            context_case["planning"]["tracking_clearance_m"] = geometry.clearance
            context_case["planning"]["edge_max_joint_delta_rad"] = geometry.resolution
            context = JointTaskContext.from_case(context_case, base_placement=JointBasePlacement(**recipe["base_placement"]), limits=limits,
                                                start_tool_pose=cpu.fk(start_q), goal_tool_pose=cpu.fk(goal_q))
            context = replace(context, fixed_base_z_m=recipe.get("base_z_m", context_case["base_z_m"]))
            kin = BoundedKinematics(cpu, cpu.bounds, context_case["tool_workspace_m"], max_fk_calls=50000, max_ik_calls=10000, wall_check=check)
            request = make_joint_request(start_q, goal_q, limits, max_queries=50000, seed=71+index, options=PLANNING_OPTIONS)
            planning_started = time.perf_counter()
            plan = plan_joint_motion(request, oracle, strategy, limits, speed_scale=bound["joint_speed_scale"],
                kinematics=kin, context=context, edge_timing_hook=edge_timing_hook, oracle_description=(
                    "CPU full-source three-arm/map/selected-container geometry. "
                    + ("Nominal rigid bead geometry. " if args.material == "beads" else
                       "No particle dynamics or rigid bead substitutes. ")
                    + "No native scene or actual grasp receipt. Actual Genesis replay remains mandatory."))
            planning_wall_s = time.perf_counter() - planning_started
            tr = plan.trajectory.to_record()
            stage = {"stage_id": stage_id, "kind": kind, "active_arm_id": arm, "support_id": support,
                "destination_support_id": destination, "trajectory": tr, "nominal_body_world": nominal_start_body,
                "nominal_goal_body_world": nominal_goal_body, "nominal_command_body_world": nominal_command_body,
                "nominal_goal_hand_world": nominal_hand, "nominal_hand_to_body": copy.deepcopy(nominal_transform) if kind == "carry" else None,
                "expected_peer_commands_by_arm": {name: copy.deepcopy(command) for name, command in commands.items() if name != arm},
                "offline_planning": {"terminal": terminal, "terminal_wall_s": terminal_wall_s, "planning_wall_s": planning_wall_s,
                    "planning_request": request.to_record(),
                    "stage_wall_s": time.perf_counter() - stage_started, "plan_evidence": plan.evidence, "cpu_geometry": cpu.evidence(),
                    "strategy_receipt": optional_strategy_receipt(strategy)}}
            stages.append(stage)
            save(out / (stage_id + ".json"), stage)
            commands[arm]["q_rad"] = tr["waypoints"][-1]
            if kind == "approach":
                actual_cpu_hand = cpu.fk(tr["waypoints"][-1])
                nominal_transform = measure_hand_to_body(WorldPose(actual_cpu_hand.position_m, actual_cpu_hand.quaternion_wxyz), body_pose).to_record()
            elif kind == "carry":
                body_pose = WorldPose(nominal_goal_body["position_m"], nominal_goal_body["quaternion_wxyz"])
                nominal_transform = None
                if args.post_place_policy in ("instant_reset", "kinematic_return") and arm != "arm_2":
                    commands[arm] = copy.deepcopy(initial_commands[arm])
                if args.post_place_policy == "kinematic_return":
                    returns[:] = kinematic_return_schedule(stages, initial_commands)
            print(json.dumps({"event": "cpu_stage_completed", "stage_id": stage_id, "stage_wall_s": stage["offline_planning"]["stage_wall_s"],
                "trajectory_simulation_s": tr["duration_s"], "waypoints": len(tr["waypoints"])}), flush=True)
        check()
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        source_ok = guard.after_request()
        restore_integration(mujoco, original_step)
        imported = [name for name in sys.modules if name == "genesis" or name.startswith("genesis.")]
        if failure is None and (imported or blocker.blocked_attempts or original_step["attempts"]):
            failure = {"type": "ForbiddenNativeOperation", "message": "Student planning attempted a forbidden native import or CPU integration API"}
        success_status = "CPU_GEOMETRIC_ONE_CARRY_PLANNED" if args.scope == "one-carry" else "CPU_GEOMETRIC_MISSION_PLANNED"
        summary = {"status": success_status if failure is None and len(stages) == len(queue) and source_ok else "CPU_PLAN_PREFIX_ONLY" if failure is None and source_ok else "CPU_PLAN_FAILED",
            "planning_scope": args.scope, "material": args.material,
            "expected_physical_motion_count": len(queue), "expected_zero_time_reset_count": len(resets),
            "expected_nonphysical_return_count": len(returns),
            "post_place_policy": args.post_place_policy,
            "completed_stages": len(stages), "failure": failure, "source_bindings_unchanged": source_ok, "source_guard": guard.receipt(),
            "genesis_imported_modules": imported, "blocked_genesis_import_attempts": blocker.blocked_attempts,
            "blocked_integration_attempts": list(original_step["attempts"]),
            "genesis_initialized": False, "native_physical_steps": 0, "cpu_physical_steps": 0,
            "cpu_model": cpu_geometry_evidence(geometry, args.material, native_precision),
            "particle_material_request": particle_request,
            "process_wall_s": time.perf_counter()-started,
            "full_mission_passed": False, "physical_transport_verified": False, "student_release_approved": False}
        save(out / "CPU_PLANNING_RESULT.json", summary)
        if summary["status"] == success_status and not imported and not blocker.blocked_attempts:
            bundle["cpu_planning_wall_s"] = summary["process_wall_s"]
            bundle["stored_motion_duration_s"] = sum(stage["trajectory"]["duration_s"] for stage in stages)
            bundle["cpu_planning_result"] = summary
            if args.scope == "one-carry":
                from physics.one_carry_planning import seal_one_carry_bundle
                bundle = seal_one_carry_bundle(bundle)
                save(out / "saved_one_carry.json", bundle)
            else:
                bundle["bundle_sha256"] = digest(bundle)
                save(out / "saved_mission.json", bundle)
    print(json.dumps({key: summary[key] for key in ("status", "completed_stages", "process_wall_s", "failure")}), flush=True)
    return 0 if summary["status"] == success_status else 1


def supervise_cpu_worker(argv, *, output, timeout_s):
    """Bound even a student callback that never returns or calls its oracle."""
    from physics.review_process_memory import creation_identity, kill_owned_tree
    started = time.perf_counter()
    output = Path(output)
    hidden = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    receipt = {'schema_version': 'owned-cpu-worker-budget-v1', 'timeout_s': timeout_s,
               'timed_out': False, 'worker_exit_code': None, 'cleanup_errors': []}
    process = identity = None
    with (output/'worker_stdout.log').open('xb') as stream:
        try:
            process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, creationflags=hidden)
            identity = creation_identity(process)
            receipt['worker_identity'] = identity
            try:
                process.wait(timeout=max(.01, timeout_s-(time.perf_counter()-started)))
            except subprocess.TimeoutExpired:
                receipt['timed_out'] = True
                receipt['tree_termination'] = kill_owned_tree(process, identity)
        except Exception as exc:
            receipt['failure'] = {'type':type(exc).__name__, 'message':str(exc)}
        finally:
            if process is not None and process.poll() is None:
                try:
                    if identity is None:
                        # Still the exact Popen handle owned by this parent.
                        process.kill(); process.wait(timeout=3.)
                    else:
                        receipt['final_tree_termination'] = kill_owned_tree(process, identity)
                except Exception as exc:
                    receipt['cleanup_errors'].append(repr(exc))
            if process is not None: receipt['worker_exit_code'] = process.poll()
            receipt['worker_elapsed_including_cleanup_s'] = time.perf_counter()-started
    return receipt


def main(argv=None):
    started = time.perf_counter()
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--help' in argv or '-h' in argv:
        return _worker_main(argv)
    admission = argparse.ArgumentParser(add_help=False)
    admission.add_argument('--worker', action='store_true')
    admission.add_argument('--output', type=Path, required=True)
    admission.add_argument('--wall-limit-s', type=float, default=1800.)
    admission.add_argument('--scope', default='mission')
    admission.add_argument('--material', default='beads')
    args, _ = admission.parse_known_args(argv)
    if args.worker or (args.material != 'beads' and args.scope != 'one-carry'):
        return _worker_main(argv)
    if any(name == 'genesis' or name.startswith('genesis.') for name in sys.modules):
        admission.error('CPU supervision requires a fresh process without Genesis imports')
    parent_blocker = NoGenesis()
    sys.meta_path.insert(0, parent_blocker)
    if not 10 <= args.wall_limit_s <= 3600:
        admission.error('CPU whole-process budget must be10..3600 seconds')
    out = args.output.resolve()
    if out.exists() or out == ROOT:
        admission.error('Use a fresh CPU export output')
    out.mkdir(parents=True, exist_ok=False)
    worker_args = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == '--output': index += 2; continue
        if value.startswith('--output='): index += 1; continue
        worker_args.append(value); index += 1
    worker = out/'worker'
    command = [sys.executable, '-B', '-m', 'physics.plan_student_transport',
               *worker_args, '--worker', '--output', str(worker)]
    # Reserve bounded trusted publication time. Student execution cannot consume
    # the final second and then publish a late success candidate.
    allowance = max(.01, args.wall_limit_s-(time.perf_counter()-started)-1.)
    receipt = supervise_cpu_worker(command, output=out, timeout_s=allowance)
    receipt.update(requested_wall_limit_s=args.wall_limit_s,
                   scope='Owned CPU child startup through exit. Timeout kills its owned process tree. No Genesis import in the parent.')
    summary_path = worker/'CPU_PLANNING_RESULT.json'
    missing_summary = {
        'status':'CPU_PLAN_FAILED', 'planning_scope':args.scope, 'material':args.material,
        'completed_stages':0, 'failure':{'type':'CpuWorkerDidNotComplete',
            'message':'No completed worker receipt. Consult worker_stdout.log and CPU_PROCESS_RESULT.json.'},
        'replayable_path_exported':False, 'physical_transport_verified':False}
    try:
        summary = read(summary_path) if summary_path.exists() else missing_summary
        if not isinstance(summary, dict): raise ValueError('CPU worker summary must be a JSON object')
    except Exception as exc:
        receipt['failure'] = {'type':type(exc).__name__, 'message':str(exc)}
        summary = missing_summary
    success = (not receipt['timed_out'] and not receipt['cleanup_errors']
        and not receipt.get('failure')
        and receipt['worker_exit_code'] == 0
        and summary.get('status') in ('CPU_GEOMETRIC_ONE_CARRY_PLANNED','CPU_GEOMETRIC_MISSION_PLANNED'))
    try:
        if success:
            candidates = [path for path in (worker/'saved_one_carry.json', worker/'saved_mission.json') if path.is_file()]
            if len(candidates) != 1: raise RuntimeError('Successful CPU worker lacks exactly one candidate bundle')
            value = read(candidates[0])
            from transport_challenge.contracts import digest
            if candidates[0].name == 'saved_one_carry.json':
                from physics.one_carry_planning import SavedOneCarryPlan
                SavedOneCarryPlan(value, path=candidates[0], require_supervisor=False).verify_sources()
            else:
                from physics.saved_motion_planning import SavedMissionPlan
                SavedMissionPlan(value, path=candidates[0]).verify_sources()
            if value['bundle_sha256'] != digest({k:v for k,v in value.items() if k!='bundle_sha256'}):
                raise RuntimeError('CPU candidate canonical digest differs')
            for name, expected in value['source_pins'].items():
                if sha(name) != expected: raise RuntimeError('CPU source changed before public export: '+name)
            if summary.get('source_bindings_unchanged') is not True:
                raise RuntimeError('Worker CPU source boundary failed')
            for path in worker.glob('*.json'):
                if path.name in ('CPU_PLANNING_RESULT.json','saved_one_carry.json','saved_mission.json'): continue
                with (out/path.name).open('xb') as stream: stream.write(path.read_bytes())
            elapsed = time.perf_counter()-started
            if elapsed >= args.wall_limit_s: raise TimeoutError('Whole CPU process budget exhausted before publication')
            parent_imports = [name for name in sys.modules if name == 'genesis' or name.startswith('genesis.')]
            if parent_imports or parent_blocker.blocked_attempts:
                raise RuntimeError('CPU parent attempted a forbidden Genesis import')
            receipt.update(status='COMPLETE_WITHIN_BUDGET', parent_wall_s_before_publication=elapsed,
                           source_bindings_verified=True, worker_result_path=str(summary_path),
                           worker_result_sha256=sha(summary_path), parent_genesis_imported_modules=parent_imports,
                           parent_blocked_genesis_import_attempts=list(parent_blocker.blocked_attempts))
            summary['worker_process_wall_s'] = summary['process_wall_s']
            summary['process_wall_s'] = elapsed
            summary['process_wall_scope'] = 'Parent entry through child exit and source admission before final bundle encoding/publication validation/output writes. Parent exit is excluded.'
            summary['cpu_process_supervisor'] = copy.deepcopy(receipt)
            value['cpu_process_supervisor'] = copy.deepcopy(receipt)
            value['cpu_planning_result'] = copy.deepcopy(summary)
            value['cpu_planning_wall_s'] = elapsed
            value.pop('bundle_sha256', None); value['bundle_sha256'] = digest(value)
            pending = out/('.pending_'+candidates[0].name)
            save(pending, value)
            if candidates[0].name == 'saved_one_carry.json':
                SavedOneCarryPlan(value, path=pending).verify_sources()
            else:
                SavedMissionPlan(value, path=pending).verify_sources()
            if time.perf_counter()-started >= args.wall_limit_s:
                raise TimeoutError('Whole CPU budget exhausted during publication validation')
            pending.rename(out/candidates[0].name)
        else:
            receipt['status'] = 'CPU_WORKER_TIMEOUT' if receipt['timed_out'] else 'CPU_WORKER_FAILED'
    except Exception as exc:
        success = False
        receipt.update(status='CPU_PUBLICATION_REJECTED', failure={'type':type(exc).__name__, 'message':str(exc)})
    if not success:
        summary.update(status='CPU_PLAN_FAILED', replayable_path_exported=False,
                       cpu_process_supervisor=copy.deepcopy(receipt))
    save(out/'CPU_PLANNING_RESULT.json', summary)
    receipt['parent_wall_s_before_receipt_write'] = time.perf_counter()-started
    save(out/'CPU_PROCESS_RESULT.json', receipt)
    print(json.dumps({'status':summary['status'], 'output':str(out),
        'worker_process_wall_s':summary.get('worker_process_wall_s'),
        'cpu_attributed_parent_wall_s':summary.get('process_wall_s'),
        'timed_out':receipt['timed_out'], 'public_bundle_exported':success,
        'failure':summary.get('failure') or receipt.get('failure'),
        'worker_log':str(out/'worker_stdout.log')}), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
