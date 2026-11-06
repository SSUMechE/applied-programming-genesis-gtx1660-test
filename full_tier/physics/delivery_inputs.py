"""Small adapters for the explicit instructor sampled-release prototype."""
from __future__ import annotations


def prototype_policy(*, dt_s, goal_body_pose, goal_support_id, arm_limits,
                     finger_limits, effort_limits):
    from physics.sampled_mission import SampledMissionPolicy
    return SampledMissionPolicy.from_record({
        "schema_version": "sampled-released-delivery-policy-v1",
        "dt_s": dt_s, "deadline_s": 60., "carrier_id": "tray_0",
        "goal_support_id": goal_support_id, "goal_body_pose": goal_body_pose,
        "goal_position_tolerance_m": .006, "goal_orientation_tolerance_rad": .04,
        "stability_s": .5, "maximum_content_speed_m_s": .1,
        "maximum_body_linear_speed_m_s": .01,
        "maximum_body_angular_speed_rad_s": .1,
        "release_open_minimum_m": .035,
        "arm_position_bounds_rad": [list(v) for v in arm_limits.position_rad],
        "finger_position_bounds_m": [list(v) for v in finger_limits],
        "arm_speed_limits_rad_s": list(arm_limits.velocity_rad_s),
        "arm_acceleration_limits_rad_s2": list(arm_limits.acceleration_rad_s2),
        "effort_limits": list(effort_limits), "maximum_tracking_error_rad": .03,
        "effort_identity_tolerance": .005, "position_numeric_tolerance": 1e-5,
        "speed_numeric_tolerance": 1e-6, "acceleration_numeric_tolerance": 1e-5})


def material_observations(ids, positions, velocities, inside):
    if not (len(ids) == len(positions) == len(velocities) == len(inside)):
        raise ValueError("Persistent material observation lengths disagree")
    return [{"id": entry["id"], "mass_kg": entry["mass_kg"], "active": True,
             "inside_carrier": bool(contained), "world_position_m": list(position),
             "velocity_world_m_s": list(velocity)}
            for entry, position, velocity, contained in zip(ids, positions, velocities, inside)]


def initial_runtime_action_counts():
    # These are owned counters for this executor's code paths. They do not claim
    # to instrument arbitrary student callbacks, which this runner never invokes.
    return {"post_initialization_robot_pose_setters": 0, "runtime_tray_pose_setters": 0,
            "particle_resets": 0, "runtime_welds": 0,
            "external_body_trajectory_commands": 0}
