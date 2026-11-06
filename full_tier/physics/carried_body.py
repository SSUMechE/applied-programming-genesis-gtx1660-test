"""Pure teacher support for an observed hand/body relation in a query scene.

Positions are metres. Quaternions are (w, x, y, z), normalized and sign-canonical.
WorldPose always names a world-frame pose. HandToBodyTransform maps body-local
points into hand-local coordinates. No fixed grasp offset is assumed.

This module performs no simulator calls. Composition describes an isolated
geometric query model, not a weld, runtime pose command or verified friction grip.
The caller must supply actual observed hand/body poses and finger coordinates.

Future exact contact-classifier input (NOT implemented here): active phase,
object and robot IDs, built geom/link ownership, identified finger-pad geoms,
identified handle/support component geoms, contact world point/normal/penetration,
and body/hand poses from the SAME detection time. A support contact additionally
needs its named support and permitted phase. Raw pair IDs or a later body pose
alone cannot authorize a contact. The default policy grants no exemptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

from transport_challenge.contracts import ContractError, digest, identifier, number, point
from transport_challenge.vessel import build_container_geometry


def _quaternion(value):
    q = point(value, 4)
    scale = max(map(abs, q))
    if scale == 0.:
        raise ContractError("A zero quaternion has no orientation")
    scaled = tuple(v/scale for v in q)
    norm = math.sqrt(sum(v*v for v in scaled))
    normalized = tuple(v/norm for v in scaled)
    sign = 1. if next(v for v in normalized if v != 0.) > 0. else -1.
    return tuple(0. if v == 0. else sign*v for v in normalized)


def _multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return _quaternion((w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                        w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v))


def _inverse(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def _rotate(q, vector):
    w, x, y, z = q
    a, b, c = point(vector)
    tx, ty, tz = 2*(y*c-z*b), 2*(z*a-x*c), 2*(x*b-y*a)
    return point((a+w*tx+y*tz-z*ty, b+w*ty+z*tx-x*tz, c+w*tz+x*ty-y*tx))


@dataclass(frozen=True)
class WorldPose:
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "position_m", point(self.position_m))
        object.__setattr__(self, "quaternion_wxyz", _quaternion(self.quaternion_wxyz))

    def transform_point(self, body_local_m):
        rotated = _rotate(self.quaternion_wxyz, body_local_m)
        return point(tuple(p+r for p, r in zip(self.position_m, rotated)))

    def to_record(self):
        return {"frame": "world", "position_m": list(self.position_m),
                "quaternion_wxyz": list(self.quaternion_wxyz)}


@dataclass(frozen=True)
class HandToBodyTransform:
    """Body-to-hand mapping measured from two simultaneous world poses."""
    translation_hand_m: tuple[float, float, float]
    quaternion_hand_body_wxyz: tuple[float, float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "translation_hand_m", point(self.translation_hand_m))
        object.__setattr__(self, "quaternion_hand_body_wxyz", _quaternion(self.quaternion_hand_body_wxyz))

    def to_record(self):
        return {"parent_frame": "hand", "child_frame": "body",
                "translation_hand_m": list(self.translation_hand_m),
                "quaternion_hand_body_wxyz": list(self.quaternion_hand_body_wxyz)}


def measure_hand_to_body(hand_world: WorldPose, body_world: WorldPose):
    """Compute inverse(T_world_hand) @ T_world_body, without moving either."""
    if not isinstance(hand_world, WorldPose) or not isinstance(body_world, WorldPose):
        raise ContractError("Measurement requires two explicitly world-frame poses")
    inverse = _inverse(hand_world.quaternion_wxyz)
    displacement = tuple(b-h for b, h in zip(body_world.position_m, hand_world.position_m))
    return HandToBodyTransform(_rotate(inverse, displacement),
                               _multiply(inverse, body_world.quaternion_wxyz))


def compose_body_world(hand_world: WorldPose, measured: HandToBodyTransform):
    """Compose a hypothetical query pose. This does not command a real body."""
    if not isinstance(hand_world, WorldPose) or not isinstance(measured, HandToBodyTransform):
        raise ContractError("Composition requires WorldPose and HandToBodyTransform")
    return WorldPose(hand_world.transform_point(measured.translation_hand_m),
                     _multiply(hand_world.quaternion_wxyz, measured.quaternion_hand_body_wxyz))


@dataclass(frozen=True)
class PrimitiveBox:
    size_m: tuple[float, float, float]
    center_body_m: tuple[float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "size_m", point(self.size_m))
        object.__setattr__(self, "center_body_m", point(self.center_body_m))
        if min(self.size_m) <= 0.:
            raise ContractError("Primitive box size must be positive")


@dataclass(frozen=True)
class VesselModel:
    """An immutable snapshot derived by the existing exact geometry function."""
    container_json: str
    geometry_json: str = field(init=False, repr=False)
    geometry_sha256: str = field(init=False)
    component_boxes: tuple[PrimitiveBox, ...] = field(init=False)
    center_of_mass_body_m: tuple[float, float, float] = field(init=False)
    empty_mass_kg: float = field(init=False)

    def __post_init__(self):
        try:
            container = json.loads(self.container_json)
            if not isinstance(container, dict):
                raise ContractError("Container must be a mapping")
            canonical = json.dumps(container, sort_keys=True, separators=(",", ":"), allow_nan=False)
            geometry = build_container_geometry(container)
        except (TypeError, ValueError, KeyError) as exc:
            raise ContractError(f"Invalid vessel model: {exc}") from exc
        if not geometry["material_policy_compatible"]:
            raise ContractError("Held-body support requires the fixed-density vessel model")
        encoded = json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "container_json", canonical)
        object.__setattr__(self, "geometry_json", encoded)
        object.__setattr__(self, "geometry_sha256", digest(geometry))
        object.__setattr__(self, "component_boxes", tuple(PrimitiveBox(size, center)
                           for size, center in geometry["component_boxes"]))
        object.__setattr__(self, "center_of_mass_body_m", point(geometry["center_of_mass_local_m"]))
        object.__setattr__(self, "empty_mass_kg", geometry["empty_mass_kg"])

    @classmethod
    def from_container(cls, container):
        try:
            return cls(json.dumps(container, sort_keys=True, separators=(",", ":"), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"Invalid vessel configuration: {exc}") from exc

    def container_record(self):
        return json.loads(self.container_json)

    def geometry_record(self):
        return json.loads(self.geometry_json)


def default_contact_policy():
    """Fresh description, not an implementation of the required classifier."""
    return {"default": "reject", "hand_body_pair_exemptions": [], "support_body_pair_exemptions": [],
            "exact_phase_contact_classifier_required": True, "physical_grasp_verified": False}


PROVIDED_AUTOMATIC_GRASP_SHA256 = "7458d4f339da20d91d819f981cc8e30c3b018dd5cce375e9d61f32eb55e3c291"


def validate_provided_grasp_binding(spec, binding, *, active_arm_id, observation):
    """Match trusted native receipts to this exact current query observation.

    This pure structural binding cannot authenticate a fabricated receipt or
    inspect a scene. The runtime owns helper-source and actual-holder checks.
    No hash alone grants a geometric exemption or a native grasp verdict.
    """
    def require(value, message):
        if not value: raise ContractError(message)
    require(type(spec) is HeldBodySpec and spec.grasp_model == "provided_ideal",
            "Explicit provided ideal HeldBodySpec required")
    require(type(binding) is dict and set(binding) == {"schema_version", "model", "holder_arm_id",
            "helper_source_sha256", "attach_receipt", "current_snapshot"}, "Exact provided-grasp binding required")
    digest(binding)  # Detached finite JSON only, not live simulator objects.
    require(binding["schema_version"] == "provided-automatic-grasp-binding-v1"
            and binding["model"] == "provided_ideal" and binding["holder_arm_id"] == active_arm_id
            and binding["helper_source_sha256"] == PROVIDED_AUTOMATIC_GRASP_SHA256,
            "Provided-grasp policy, source or active holder differs")
    receipt, current = binding["attach_receipt"], binding["current_snapshot"]
    require(type(receipt) is dict and digest(receipt) == spec.attachment_receipt_sha256
            and receipt.get("model") == "provided-ideal-native-weld-v1"
            and receipt.get("event") == "attach" and receipt.get("arm_id") == active_arm_id,
            "Actual owned attachment receipt differs")
    before, after, pair = receipt["before"], receipt["after"], receipt["pair"]
    require(type(pair) is list and len(pair) == 2 and all(type(i) is int and i >= 0 for i in pair)
            and pair[0] != pair[1], "Native attachment link pair differs")
    require({k:v for k,v in before.items() if k != "inventory"}
            == {k:v for k,v in after.items() if k != "inventory"},
            "Attach receipt changed pose, velocity, mass or clock")
    require(type(current) is dict and current["body_entity_idx"] == before["body_entity_idx"]
            and [current["hand_link_idx"], current["cup_link_idx"]] == pair
            and [before["hand_link_idx"], before["cup_link_idx"]] == pair,
            "Current native hand/cup identity differs from attachment")
    b, a, c = (s["inventory"] for s in (before, after, current))
    owned = [1, *pair]  # Installed Genesis 1.3.2 EQUALITY_TYPE.WELD.
    require(a["rows"] == b["rows"] + [owned]
            and a["weld_pairs"] == b["weld_pairs"] + [pair]
            and all(pair[1] not in p for p in b["weld_pairs"])
            and b["count"] == len(b["rows"])
            and a["count"] == b["count"] + 1 == len(a["rows"])
            and a["capacity"] == b["capacity"] >= a["count"] and c == a,
            "Current owned native weld inventory differs")
    clock = current["native_clock"]
    require(type(clock) is list and len(clock) == 4 and type(clock[0]) is int
            and type(clock[2]) is int and type(clock[3]) is int
            and clock[0] >= before["native_clock"][0] and clock[2] >= before["native_clock"][2]
            and clock[3] == 0 and clock[:2] == [observation["native_scene_step"], observation["simulation_s"]],
            "Provided grasp is not this exact current outer-boundary observation")
    for key in ("body_world", "body_linear_velocity_m_s", "body_angular_velocity_rad_s"):
        require(current[key] == observation[key], "Provided grasp current body differs: " + key)
    require(current["hand_world"] == observation["arms"][active_arm_id]["hand_world"]
            and WorldPose(**{k:current["body_world"][k] for k in ("position_m", "quaternion_wxyz")}) == spec.measured_body_world
            and WorldPose(**{k:current["hand_world"][k] for k in ("position_m", "quaternion_wxyz")}) == spec.measured_hand_world
            and tuple(observation["arms"][active_arm_id]["q_all_dofs"][7:]) == spec.finger_dof_positions_m,
            "Provided grasp does not bind the measured hand/body/fingers")
    require(type(current["cup_mass_kg"]) in (int, float) and current["cup_mass_kg"] > 0
            and current["cup_mass_kg"] == before["cup_mass_kg"], "Native cup mass changed since attachment")
    if "particle_material" in observation:
        require(clock == observation["particle_material"]["native_clock"], "Particle/attachment clock differs")
    return {"schema_version": binding["schema_version"], "attachment_receipt_sha256": spec.attachment_receipt_sha256,
            "helper_source_sha256": binding["helper_source_sha256"], "holder_arm_id": active_arm_id,
            "native_clock": list(clock), "pair": list(pair),
            "scope": "Current trusted-runtime receipt binding only. Not independent native/source authentication."}


@dataclass(frozen=True)
class HeldBodySpec:
    object_id: str
    model: VesselModel
    measured_hand_world: WorldPose
    measured_body_world: WorldPose
    finger_dof_positions_m: tuple[float, float]
    observation_time_s: float
    observation_id: str
    grasp_model: str = field(default="friction", kw_only=True)
    attachment_receipt_sha256: str | None = field(default=None, kw_only=True)
    hand_to_body: HandToBodyTransform = field(init=False)
    binding_sha256: str = field(init=False)

    def __post_init__(self):
        identifier(self.object_id, "object_id")
        identifier(self.observation_id, "observation_id")
        if not isinstance(self.model, VesselModel):
            raise ContractError("Held body requires an immutable VesselModel")
        if self.grasp_model not in ("friction", "provided_ideal"):
            raise ContractError("Unknown held-body grasp model")
        receipt = self.attachment_receipt_sha256
        if self.grasp_model == "provided_ideal":
            if (type(receipt) is not str or len(receipt) != 64
                    or any(c not in "0123456789abcdef" for c in receipt)):
                raise ContractError("Provided ideal grasp requires its actual attachment receipt SHA256")
        elif receipt is not None:
            raise ContractError("Friction grasp cannot claim an automatic attachment receipt")
        # Store actual measured coordinates without clipping to nominal limits.
        # The later built-model query adapter must separately enforce its limits.
        object.__setattr__(self, "finger_dof_positions_m", point(self.finger_dof_positions_m, 2))
        object.__setattr__(self, "observation_time_s", number(self.observation_time_s, "observation time", minimum=0.))
        object.__setattr__(self, "hand_to_body", measure_hand_to_body(self.measured_hand_world, self.measured_body_world))
        object.__setattr__(self, "binding_sha256", digest(self.to_record()))

    def body_pose_at(self, queried_hand_world: WorldPose):
        return compose_body_world(queried_hand_world, self.hand_to_body)

    def center_of_mass_world_at(self, queried_hand_world: WorldPose):
        return self.body_pose_at(queried_hand_world).transform_point(self.model.center_of_mass_body_m)

    def component_poses_world_at(self, queried_hand_world: WorldPose):
        body = self.body_pose_at(queried_hand_world)
        return tuple(WorldPose(body.transform_point(box.center_body_m), body.quaternion_wxyz)
                     for box in self.model.component_boxes)

    def to_record(self):
        record = {"schema_version": "observed-held-body-v1", "object_id": self.object_id,
                "model_geometry_sha256": self.model.geometry_sha256,
                "model_configuration_sha256": digest(self.model.container_record()),
                "observed_hand_world": self.measured_hand_world.to_record(),
                "observed_body_world": self.measured_body_world.to_record(),
                "observed_finger_dof_positions_m": list(self.finger_dof_positions_m),
                "observation_time_s": self.observation_time_s, "observation_id": self.observation_id,
                "measured_hand_to_body": self.hand_to_body.to_record(),
                "contact_policy": default_contact_policy(), "physical_grasp_certified_by_this_module": False,
                "runtime_body_pose_commands_permitted": False}
        if self.grasp_model == "provided_ideal":
            record.update(schema_version="observed-held-body-provided-ideal-v1",
                grasp_model=self.grasp_model, attachment_receipt_sha256=self.attachment_receipt_sha256,
                contact_policy={**default_contact_policy(),
                    "grasp_model": "provided_ideal", "active_bilateral_witnesses_required": False,
                    "native_weld_hand_cup_link_pair_excluded": True,
                    "query_hand_cup_geometry_policy": "checked_without_pair_exemption",
                    "attachment_authentication": "trusted_runtime_owns_native_holder_and_source_verification"})
        return record


@dataclass(frozen=True)
class VesselAsset:
    path: Path
    asset_sha256: str
    model_geometry_sha256: str
    empty_mass_kg: float


def write_carried_vessel_urdf(directory, model: VesselModel):
    """Explicit asset write through the existing writer. No simulator work."""
    if not isinstance(model, VesselModel):
        raise ContractError("Asset generation requires an immutable VesselModel")
    from .scene_assets import write_tray_urdf
    path, mass, contract = write_tray_urdf(directory, model.container_record())
    expected = model.geometry_record()
    if digest({key: contract[key] for key in expected}) != model.geometry_sha256 or mass != model.empty_mass_kg:
        raise ContractError("URDF writer geometry/mass differs from the held-body model")
    return VesselAsset(path.resolve(), hashlib.sha256(path.read_bytes()).hexdigest(), model.geometry_sha256, mass)
