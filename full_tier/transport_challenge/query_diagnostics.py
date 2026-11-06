"""Supplied, simulator-free snapshots of already completed validity queries.

This module never queries geometry or suggests a replacement state/route.
Diagnostic failures cannot change a validity answer. Missing or malformed
observations are explicitly unavailable, never reconstructed as contacts.
"""
import copy
import math

from .contracts import digest


SCHEMA = "last-query-diagnostic-v1"


def diagnostic_record(status="not_queried", *, reason=None, query_kind=None):
    return {"schema_version": SCHEMA, "status": status, "reason": reason,
            "query_kind": query_kind, "contacts": [], "details": {},
            "witness_status": "unavailable", "diagnostic_issues": [],
            "failed_edge_sample": None,
            "failed_edge_sample_scope": "Not recorded. Contacts describe a rejected sampled state, not every point of an edge.",
            "scope": "Historical query snapshot only. No new query, route, contact force or continuous-safety certificate."}


def read_query_diagnostic(oracle):
    """Read only an explicit supplied API, without inspecting simulator state."""
    try:
        method = getattr(oracle, "last_query_diagnostic", None)
        if not callable(method):
            return diagnostic_record("unavailable", reason="backend_diagnostic_not_supported")
        result = method()
        if not isinstance(result, dict) or result.get("schema_version") != SCHEMA:
            raise ValueError("Unsupported diagnostic record")
        return copy.deepcopy(result)
    except Exception as error:
        result = diagnostic_record("unavailable", reason="backend_diagnostic_unavailable")
        result["diagnostic_issues"] = [type(error).__name__ + ": " + str(error)]
        return result


def completed_query_diagnostic(oracle, kind, accepted):
    if accepted:
        result = diagnostic_record("accepted", query_kind=kind)
        result["witness_status"] = "not_applicable"
        return result
    result = read_query_diagnostic(oracle)
    result.update(status="rejected", query_kind=kind)
    return result


def bind_query_input(record, arguments):
    """Bind finite student-supplied endpoints, not an internally chosen sample."""
    try:
        values = []
        for arg in arguments:
            if not isinstance(arg, (tuple, list)):
                raise ValueError("Input is not a coordinate sequence")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in arg):
                raise ValueError("Input is not finite numeric coordinates")
            values.append(list(arg))
        record["query_input"] = values
        record["query_input_sha256"] = digest(values)
    except (TypeError, ValueError, OverflowError):
        record["query_input"] = None
        record["query_input_sha256"] = None
    return record


def _ids(values):
    return (isinstance(values, (list, tuple)) and len(values) == 2
            and all(type(v) is int and v >= 0 for v in values) and values[0] != values[1])


def _vector(values):
    return (isinstance(values, (list, tuple)) and len(values) == 3
            and all(not isinstance(v, bool) and isinstance(v, (int, float))
                    and math.isfinite(v) for v in values))


def _owner(value):
    if not isinstance(value, dict):
        return None
    keys = ("kind", "entity", "link", "link_id", "component_id", "assembly_id")
    return copy.deepcopy({key: value[key] for key in keys if key in value})


def native_failure_diagnostic(failure, owners, detection=None, *, geometry=None):
    """Format retained failure/pair/witness records. No native calls occur here.

    A collision witness is eligible only for a named failed pair, an explicitly
    verified same-query buffer, and an UNALLOWED_CONTACT row. Early rejections
    never expose a previous detection. Raw normal orientation is preserved.
    """
    if not isinstance(failure, dict):
        return diagnostic_record("no_failure_recorded")
    result = diagnostic_record("rejected", reason=failure.get("reason"))
    result["query_geometry"] = copy.deepcopy(geometry)
    result["details"] = copy.deepcopy({key: value for key, value in failure.items() if key != "pairs"})
    if failure.get("reason") != "full_model_collision":
        result["witness_status"] = "not_applicable_to_rejection"
        return result
    contacts = {}
    for pair in failure.get("pairs", ()):
        if not isinstance(pair, dict) or not _ids(pair.get("geom_ids")):
            result["diagnostic_issues"].append("Malformed failed geometry pair")
            continue
        a, b = pair["geom_ids"]
        contacts[tuple(sorted((a, b)))] = {"geom_ids": [a, b],
            "a": _owner(owners.get(a, pair.get("a"))), "b": _owner(owners.get(b, pair.get("b"))),
            "witnesses": []}
    sources = []
    if isinstance(detection, dict):
        if detection.get("pair_witness_multiset_agrees") is True:
            sources.append(("global", detection.get("rows", ())))
        for label, snapshot in detection.get("same_query_witnesses_by_body", {}).items():
            if isinstance(snapshot, dict) and snapshot.get("pair_witness_multiset_agrees") is True:
                sources.append((label, snapshot.get("rows", ())))
    for label, rows in sources:
        for row in rows:
            if not isinstance(row, dict) or not _ids(row.get("geom_ids")):
                result["diagnostic_issues"].append("Malformed retained witness geometry IDs")
                continue
            key = tuple(sorted(row["geom_ids"]))
            if key not in contacts or row.get("category") != "UNALLOWED_CONTACT":
                continue
            depth = row.get("penetration_m")
            if (not _vector(row.get("position_world_m")) or not _vector(row.get("normal_world"))
                    or isinstance(depth, bool) or not isinstance(depth, (int, float)) or not math.isfinite(depth)):
                result["diagnostic_issues"].append("Malformed retained contact coordinates/depth")
                continue
            contacts[key]["witnesses"].append({"source_body": label,
                "raw_row": row.get("raw_row"), "geom_ids": list(row["geom_ids"]),
                "position_world_m": list(row["position_world_m"]),
                "normal_world": list(row["normal_world"]), "penetration_m": depth,
                "normal_convention": "Unmodified native witness normal in its recorded geom_ids order. No force inferred."})
    result["contacts"] = list(contacts.values())
    witnessed = sum(bool(row["witnesses"]) for row in result["contacts"])
    if witnessed:
        result["witness_status"] = "available" if witnessed == len(contacts) else "partially_available"
    return result
