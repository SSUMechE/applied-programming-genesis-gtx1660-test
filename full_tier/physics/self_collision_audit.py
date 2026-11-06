"""Read-only built-model ledger for an opt-in, macro-step contact diagnostic.

No simulator import, state change or extra collision detection is performed.
The ledger enumerates the installed collider's actual pairs, not XML guesses.
"""


def adjacent_by_installed_rule(first, second, links):
    """Mirror Genesis1.3.2's parent/fixed-chain rule using built link metadata."""
    low, high = sorted((first, second), key=lambda link: link.idx)
    while high.parent_idx != -1:
        if high.parent_idx == low.idx:
            return True
        if not all(joint.n_dofs == 0 for joint in high.joints):
            break
        high = links[high.parent_idx]
    return False


def built_self_pair_ledger(solver, arms):
    """Capture actual geometry metadata and enabled/disabled same-arm pairs."""
    valid = {tuple(sorted((int(a), int(b)))) for a, b in solver.collider._valid_collision_pairs}
    ledger = {}
    for index, arm in enumerate(arms):
        geoms = list(arm.geoms)
        geometry = [{"geom_id": int(g.idx), "link_id": int(g.link.idx), "link_name": g.link.name,
                     "parent_link_id": int(g.link.parent_idx), "root_link_id": int(g.link.root_idx),
                     "fixed_relative_to_world": bool(g.link.is_fixed), "contype": int(g.contype),
                     "conaffinity": int(g.conaffinity)} for g in geoms]
        pairs = []
        for i, first in enumerate(geoms):
            for second in geoms[i+1:]:
                ids = tuple(sorted((int(first.idx), int(second.idx))))
                enabled = ids in valid
                reasons = []
                if not enabled:
                    if first.link.idx == second.link.idx:
                        reasons.append("same_link")
                    if first.link.is_fixed and second.link.is_fixed:
                        reasons.append("both_fixed_relative_to_world")
                    if not ((int(first.contype) & int(second.conaffinity)) or
                            (int(second.contype) & int(first.conaffinity))):
                        reasons.append("collision_mask")
                    if (not solver._enable_adjacent_collision and first.link.idx != second.link.idx and
                            adjacent_by_installed_rule(first.link, second.link, solver.links)):
                        reasons.append("adjacent_or_fixed_chain")
                    if not reasons:
                        reasons.append("unexplained_filter_requires_review")
                pairs.append({"geom_ids": list(ids), "link_ids": [int(first.link.idx), int(second.link.idx)],
                              "link_names": [first.link.name, second.link.name],
                              "enabled": enabled, "excluded_reasons": reasons})
        ledger[f"arm_{index}"] = {
            "entity_idx": int(arm.idx), "geom_range": [int(arm.geom_start), int(arm.geom_end)],
            "geometries": geometry, "same_arm_geometry_pairs": pairs,
            "enabled_pair_count": sum(pair["enabled"] for pair in pairs),
            "excluded_pair_count": sum(not pair["enabled"] for pair in pairs),
            "unexplained_excluded_pair_count": sum("unexplained_filter_requires_review" in pair["excluded_reasons"] for pair in pairs)}
    return {"status": "RECORDED_BUILT_COLLIDER_PAIRS", "robots": ledger,
            "source": "read-only Genesis1.3.2 collider._valid_collision_pairs and built entity geometry metadata",
            "installed_rigid_options": {"enable_self_collision": bool(solver._enable_self_collision),
                                        "enable_neutral_collision": bool(solver._enable_neutral_collision),
                                        "enable_adjacent_collision": bool(solver._enable_adjacent_collision)},
            "approved_physical_filter_audit": False}
