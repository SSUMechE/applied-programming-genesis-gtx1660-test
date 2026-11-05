"""Provided conservative Cartesian proxy geometry. Not a robot collision oracle."""

from dataclasses import dataclass
import math
from .contracts import ContractError, number, point


def distance(a, b):
    if len(a) != len(b):
        raise ContractError("Coordinate dimensions differ")
    return math.dist(a, b)


def path_length(path):
    return sum(distance(a, b) for a, b in zip(path, path[1:]))


def steer(start, target, step_size):
    separation = distance(start, target)
    if separation <= step_size:
        return tuple(target)
    fraction = step_size / separation
    return tuple(a + fraction * (b - a) for a, b in zip(start, target))


@dataclass(frozen=True)
class Box:
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self):
        if len(self.lower) != len(self.upper) or not self.lower:
            raise ContractError("Box bounds must have the same positive dimension")
        point(self.lower, len(self.lower))
        point(self.upper, len(self.upper))
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ContractError("Box lower bounds must be below upper bounds")

    def contains(self, position):
        point(position, len(self.lower))
        return all(lo <= x <= hi for lo, x, hi in zip(self.lower, position, self.upper))

    def intersects_segment(self, start, end, clearance=0.0):
        """Exact slab intersection with a closed, clearance-inflated axis box."""
        point(start, len(self.lower))
        point(end, len(self.lower))
        number(clearance, "clearance", minimum=0)
        near, far = 0.0, 1.0
        for a, b, lo, hi in zip(start, end, self.lower, self.upper):
            lo, hi = lo - clearance, hi + clearance
            delta = b - a
            if abs(delta) < 1e-15:
                if a < lo or a > hi:
                    return False
                continue
            first, last = sorted(((lo - a) / delta, (hi - a) / delta))
            near, far = max(near, first), min(far, last)
            if near > far:
                return False
        return True


class CartesianOracle:
    """Tool-centre proxy, inflated obstacles, exact straight-segment checking.

    Full payload swept volume, IK, joint limits, robot links and other arms must
    ALSO pass the physical runner. A proxy-clear path can fail those checks.
    """
    def __init__(self, workspace, obstacles=(), clearance_m=0, physical_constraints=None):
        self.workspace = workspace
        self.obstacles = tuple(obstacles)
        self.clearance_m = number(clearance_m, "clearance_m", minimum=0)
        self.carry_constraints = CarryConstraints(physical_constraints) if physical_constraints else None

    def state_valid(self, position):
        return (self.workspace.contains(position)
                and (self.carry_constraints is None or self.carry_constraints.state_valid(position))) and not any(
            box.intersects_segment(position, position, self.clearance_m)
            for box in self.obstacles)

    def edge_valid(self, start, end):
        return (self.state_valid(start) and self.state_valid(end)
                and (self.carry_constraints is None or self.carry_constraints.edge_valid(start, end))) and not any(
            box.intersects_segment(start, end, self.clearance_m)
            for box in self.obstacles)


class ToolEnvelopeOracle(CartesianOracle):
    """Conservative fixed-orientation payload box relative to the hand origin.

    The supplied payload box must enclose the actual payload for every allowed
    orientation. For changing orientations use a rotation-enclosing box or the
    provided joint-space collision oracle. This still omits the robot links.
    """
    def __init__(self, workspace, obstacles, envelope, clearance_m=0, physical_constraints=None):
        self.envelope = envelope
        inflated = [Box(tuple(lo - offset for lo, offset in zip(box.lower, envelope.upper)),
                        tuple(hi - offset for hi, offset in zip(box.upper, envelope.lower)))
                    for box in obstacles]
        super().__init__(workspace, inflated, clearance_m, physical_constraints)


def recover_path(parents, goal):
    """Provided parent-chain utility used by the reference and alternative solvers."""
    result, seen = [], set()
    node = goal
    while node is not None:
        if node in seen or node not in parents:
            raise ContractError("Parent chain is cyclic or incomplete")
        seen.add(node)
        result.append(node)
        node = parents[node]
    return tuple(reversed(result))


def quaternion_angle(first, second):
    return 2 * math.acos(min(1.0, abs(sum(a*b for a, b in zip(first, second)))))


def quaternion_slerp(first, second, fraction):
    dot = sum(a*b for a, b in zip(first, second))
    if dot < 0:
        second, dot = tuple(-v for v in second), -dot
    if dot > 0.9995:
        result = tuple(a + fraction * (b - a) for a, b in zip(first, second))
        norm = math.sqrt(sum(v*v for v in result))
        return tuple(v / norm for v in result)
    theta = math.acos(max(-1.0, min(1.0, dot)))
    return tuple((math.sin((1-fraction)*theta)*a + math.sin(fraction*theta)*b) / math.sin(theta)
                 for a, b in zip(first, second))


def rotation_axes(quaternion):
    w, x, y, z = point(quaternion, 4)
    if not math.isclose(w*w+x*x+y*y+z*z, 1, abs_tol=1e-6):
        raise ContractError("Rotation requires normalized wxyz quaternion")
    return ((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)),
            (2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)),
            (2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))


def oriented_box_intersects(center, axes, half_size, obstacle, inflation=0):
    """Provided separating-axis test, including all nine cross-product axes."""
    world_axes = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    obstacle_center = tuple((a+b)/2 for a, b in zip(obstacle.lower, obstacle.upper))
    obstacle_half = tuple((b-a)/2 + inflation for a, b in zip(obstacle.lower, obstacle.upper))
    delta = tuple(a-b for a, b in zip(center, obstacle_center))
    cross_axes = []
    for a in axes:
        for b in world_axes:
            cross_axes.append((a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]))
    for separating_axis in tuple(axes) + world_axes + tuple(cross_axes):
        if sum(v*v for v in separating_axis) < 1e-20:
            continue
        projection = abs(sum(a*b for a, b in zip(delta, separating_axis)))
        first_radius = sum(h * abs(sum(a*b for a, b in zip(axis, separating_axis)))
                           for h, axis in zip(half_size, axes))
        second_radius = sum(h * abs(value) for h, value in zip(obstacle_half, separating_axis))
        if projection > first_radius + second_radius + 1e-12:
            return False
    return True


class CarryConstraints:
    """Instructor task limits on the carried hand, checked over whole XYZ edges."""
    def __init__(self, specification=None):
        specification = specification or {}
        bounds = specification.get("carry_tool_workspace_m")
        self.workspace = Box(*(point(v) for v in bounds)) if bounds is not None else None
        self.height_limits = []
        for item in specification.get("crossing_height_limits", []):
            if item["axis"] not in ("x", "y"):
                raise ContractError("Crossing-height axis must be x or y")
            lo, hi = point(item["interval_m"], 2)
            if not lo < hi:
                raise ContractError("Crossing interval must have positive width")
            maximum = number(item["max_tool_z_m"], "crossing height")
            self.height_limits.append((("x", "y").index(item["axis"]), lo, hi, maximum))

    def state_valid(self, position):
        position = point(position)
        return (self.workspace is None or self.workspace.contains(position)) and all(
            not lo <= position[axis] <= hi or position[2] <= maximum
            for axis, lo, hi, maximum in self.height_limits)

    def edge_valid(self, start, end):
        if not self.state_valid(start) or not self.state_valid(end):
            return False
        for axis, lo, hi, maximum in self.height_limits:
            delta = end[axis]-start[axis]
            if abs(delta) < 1e-15:
                if lo <= start[axis] <= hi and max(start[2], end[2]) > maximum:
                    return False
                continue
            first, last = sorted(((lo-start[axis])/delta, (hi-start[axis])/delta))
            first, last = max(first, 0), min(last, 1)
            if first <= last and max(start[2]+t*(end[2]-start[2]) for t in (first, last)) > maximum:
                return False
        return True


class PayloadPoseOracle:
    """Provided conservative swept rod-box geometry, excluding robot links.

    A pose is {'xyz_m': [x,y,z], 'quaternion_wxyz': [w,x,y,z]}. Each edge follows
    linear hand translation and shortest quaternion SLERP, matching the physical
    adapter's interpolation. Midpoint OBB checks inflate each obstacle enough to
    enclose movement throughout that subinterval. This bounds the entire swept
    rod, not just its sampled centres. Full robot verification remains separate.
    """
    def __init__(self, workspace, obstacles, half_size, center_in_hand, clearance_m=0.005, physical_constraints=None):
        self.workspace, self.obstacles = workspace, tuple(obstacles)
        self.half_size, self.center_in_hand = point(half_size), point(center_in_hand)
        self.clearance_m = number(clearance_m, "clearance", minimum=0)
        self.radius = distance((0, 0, 0), self.center_in_hand) + distance((0, 0, 0), self.half_size)
        self.carry_constraints = CarryConstraints(physical_constraints)

    def _clear(self, position, quaternion, extra=0):
        axes = rotation_axes(quaternion)
        center = tuple(position[i] + sum(self.center_in_hand[j] * axes[j][i] for j in range(3))
                       for i in range(3))
        return not any(oriented_box_intersects(center, axes, self.half_size, obstacle,
                                              self.clearance_m + extra) for obstacle in self.obstacles)

    def state_valid(self, pose):
        return (self.workspace.contains(pose["xyz_m"]) and self.carry_constraints.state_valid(pose["xyz_m"])
                and self._clear(pose["xyz_m"], pose["quaternion_wxyz"]))

    def edge_valid(self, first, second):
        if not self.state_valid(first) or not self.state_valid(second):
            return False
        a, b = first["xyz_m"], second["xyz_m"]
        if not self.carry_constraints.edge_valid(a, b):
            return False
        qa, qb = first["quaternion_wxyz"], second["quaternion_wxyz"]
        travel, angle = distance(a, b), quaternion_angle(qa, qb)
        steps = max(1, math.ceil(travel / 0.01), math.ceil(angle / math.radians(5)))
        movement_bound = (travel + self.radius * angle) / (2 * steps)
        for index in range(steps):
            fraction = (index + 0.5) / steps
            position = tuple(x + fraction*(y-x) for x, y in zip(a, b))
            quaternion = quaternion_slerp(qa, qb, fraction)
            if not self._clear(position, quaternion, movement_bound):
                return False
        return True


class FixedOrientationPoseOracle:
    """Supplied XYZ interface to the payload oracle at one fixed quaternion.

    Students plan in metres. Pose interpolation and payload geometry remain
    supplied. This proxy still does not certify IK or full robot-link clearance.
    """
    def __init__(self, pose_oracle, quaternion):
        self.pose_oracle = pose_oracle
        self.quaternion = point(quaternion, 4)
        rotation_axes(self.quaternion)

    def pose(self, position):
        return {"xyz_m": point(position), "quaternion_wxyz": self.quaternion}

    def state_valid(self, position):
        return self.pose_oracle.state_valid(self.pose(position))

    def edge_valid(self, start, end):
        return self.pose_oracle.edge_valid(self.pose(start), self.pose(end))
