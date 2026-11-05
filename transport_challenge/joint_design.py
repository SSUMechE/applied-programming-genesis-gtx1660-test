"""Explicit joint-project design, separate from legacy XYZ/Cartesian designs.

Parsing and installation checks are pure. A loaded replay may use only its
already observed base. This module does not create a fresh grasp or evaluate
physical design quality. World task endpoints and obstacles never follow base.
"""
from dataclasses import dataclass
from collections.abc import Mapping

from .contracts import ContractError, number, point
from .joint_context import JointBasePlacement, JointInstallationZone, JointTaskContext


@dataclass(frozen=True)
class JointDesign:
    base_placement: JointBasePlacement
    joint_speed_scale: float

    def __post_init__(self):
        if not isinstance(self.base_placement, JointBasePlacement):
            raise ContractError("Use named base X/Y metres and yaw radians, never an XYZ list")
        scale = number(self.joint_speed_scale, "joint_speed_scale", minimum=0., maximum=1.)
        if scale == 0.:
            raise ContractError("joint_speed_scale must be in (0,1]")
        object.__setattr__(self, "joint_speed_scale", scale)

    @classmethod
    def from_record(cls, record, case, *, observed_base=None):
        if (not isinstance(record, Mapping) or set(record) !=
                {"schema_version", "base_placement", "joint_speed_scale"}
                or record["schema_version"] != "joint-design-v1"):
            raise ContractError("joint-design-v1 requires exactly base_placement and joint_speed_scale")
        base = record["base_placement"]
        if not isinstance(base, Mapping) or set(base) != {"x_m", "y_m", "yaw_rad"}:
            raise ContractError("base_placement requires named x_m, y_m and yaw_rad. Z is task-fixed")
        result = cls(JointBasePlacement(**base), record["joint_speed_scale"])
        try:
            if case.get("schema_version") != "joint-map-v1":
                raise ContractError("Joint designs need a joint-map-v1 case")
            zones = tuple(JointInstallationZone(**zone) for zone in case["installation_zones"])
            if not any(zone.contains(result.base_placement) for zone in zones):
                raise ContractError("Selected base lies outside the declared installation zones")
        except (KeyError, TypeError) as error:
            raise ContractError("Invalid joint installation-zone schema") from error
        if observed_base is not None and result.scene_base != point(observed_base, 3):
            raise ContractError("Loaded base must exactly match the observed task. A new base needs a fresh physical initialization/grasp")
        return result

    @property
    def scene_base(self):
        """Native scene adapter's explicit X,Y,yaw order, not legacy XYZ."""
        base = self.base_placement
        return base.x_m, base.y_m, base.yaw_rad

    def context(self, case, limits, **endpoint_overrides):
        return JointTaskContext.from_case(case, base_placement=self.base_placement,
                                         limits=limits, **endpoint_overrides)

    def to_record(self):
        return {"schema_version": "joint-design-v1", "base_placement": self.base_placement.to_record(),
                "joint_speed_scale": self.joint_speed_scale}


def planning_support(design, case, limits, strategy, *, use_edge_timing_hook=False, **endpoint_overrides):
    """Return the actual opt-in keywords consumed by plan_joint_motion.

    Explicit durations and a nonunit scale remain incompatible. A missing hook
    rejects instead of silently choosing another timing implementation.
    """
    if not isinstance(design, JointDesign) or not isinstance(use_edge_timing_hook, bool):
        raise ContractError("Use a validated joint design and a boolean timing-hook opt-in")
    support = {"context": design.context(case, limits, **endpoint_overrides)}
    if use_edge_timing_hook:
        hook = getattr(strategy, "choose_edge_durations_s", None)
        if not callable(hook):
            raise ContractError("Timing opt-in requires choose_edge_durations_s(path,limits,*,context)")
        support["edge_timing_hook"] = hook
    return support
