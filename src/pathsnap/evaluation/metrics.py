import math
from typing import Dict, List, Optional

from src.pathsnap.decoding.viterbi import TransitionDistFn
from src.pathsnap.schemas import DecodedPoint


def collapse_consecutive_arcs(arc_seq: List[Optional[int]]) -> List[int]:
    """Collapse consecutive duplicate arc IDs while preserving legitimate revisits."""
    collapsed: List[int] = []
    for arc in arc_seq:
        if arc is None:
            continue
        if not collapsed or collapsed[-1] != arc:
            collapsed.append(arc)
    return collapsed


def sequence_edit_similarity(pred_arcs: List[int], ref_arcs: List[int]) -> float:
    """Normalized Levenshtein sequence similarity in [0, 1] preserving arc order."""
    n, m = len(pred_arcs), len(ref_arcs)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if pred_arcs[i - 1] == ref_arcs[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp

    dist = dp[m]
    return round(1.0 - (dist / max(n, m)), 4)


def compute_route_continuity(
    points: List[DecodedPoint],
    transition_dist_fn: TransitionDistFn,
    max_speed_mps: float = 45.0,
) -> Dict[str, float]:
    """Fraction of consecutive matched predictions connected by a feasible network path."""
    matched = [p for p in points if p.arc_id is not None]
    if len(matched) < 2:
        return {"route_continuity": 1.0 if matched else 0.0, "gap_resets": 0}

    connected = 0
    total_pairs = 0
    gap_resets = sum(1 for p in points if p.status == "gap_reset")

    for prev_p, curr_p in zip(matched[:-1], matched[1:]):
        total_pairs += 1
        dt = max(curr_p.timestamp - prev_p.timestamp, 1e-3)
        d_net = transition_dist_fn(
            prev_p.arc_id,
            prev_p.proj_fraction or 0.5,
            curr_p.arc_id,
            curr_p.proj_fraction or 0.5,
        )
        if d_net is not None and not math.isinf(d_net) and (d_net / dt) <= max_speed_mps:
            connected += 1

    return {
        "route_continuity": round(connected / total_pairs, 4) if total_pairs else 0.0,
        "gap_resets": gap_resets,
    }


def evaluate_trip(
    decoded_points: List[DecodedPoint],
    transition_dist_fn: TransitionDistFn,
    aligned_ref_arcs: Optional[List[Optional[int]]] = None,
    raw_ref_route: Optional[List[int]] = None,
) -> Dict[str, object]:
    """
    Isolated server-side evaluation function.
    Reference labels enter ONLY here, never inside prediction or UI.
    """
    continuity_stats = compute_route_continuity(decoded_points, transition_dist_fn)
    unmatched_count = sum(1 for p in decoded_points if p.arc_id is None)

    metrics: Dict[str, object] = {
        "total_observations": len(decoded_points),
        "unmatched_observations": unmatched_count,
        "route_continuity": continuity_stats["route_continuity"],
        "gap_resets": continuity_stats["gap_resets"],
    }

    if aligned_ref_arcs is not None:
        eligible = 0
        correct = 0
        for pred_p, ref_arc in zip(decoded_points, aligned_ref_arcs):
            if ref_arc is None:
                continue
            eligible += 1
            if pred_p.arc_id == ref_arc:
                correct += 1
        metrics["observation_top1_accuracy"] = (
            round(correct / eligible, 4) if eligible > 0 else None
        )

    if raw_ref_route is not None:
        pred_seq = collapse_consecutive_arcs([p.arc_id for p in decoded_points])
        metrics["sequence_similarity"] = sequence_edit_similarity(pred_seq, raw_ref_route)

    return metrics