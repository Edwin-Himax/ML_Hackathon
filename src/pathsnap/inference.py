import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.pathsnap.decoding.viterbi import (
    HMMConfig,
    TransitionDistFn,
    viterbi_decode,
)
from src.pathsnap.schemas import CandidateObservation, DecodedPoint, PredictionResult


# ==============================================================================
# MEMBER 2 FROZEN CLASSIFIER INTEGRATION (LAZY-IMPORTED FOR WINDOWS .PYD SAFETY)
# ==============================================================================

_BUNDLE_PATH = Path("artifacts/member2/final_model.joblib")
_MODEL_BUNDLE = None


def get_member2_bundle():
    """Load Member 2's frozen classifier artifact once and cache it in memory."""
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        from pathsnap_ml.inference import load_bundle

        _MODEL_BUNDLE = load_bundle(_BUNDLE_PATH)
    return _MODEL_BUNDLE


def score_with_member2(
    trip: Any,
    geometry: Any,
    candidates_df: Any,
) -> Dict[Tuple[int, int], float]:
    """
    Run Member 2's frozen classifier on Member 1's trip, geometry, and candidates
    DataFrame, returning {(observation_id, arc_id): compatibility_probability}.
    """
    from pathsnap_ml.inference import predict_from_geometry

    bundle = get_member2_bundle()
    scored_df = predict_from_geometry(bundle, trip, geometry, candidates_df)
    return {
        (int(row.observation_id), int(row.arc_id)): float(row.compatibility_probability)
        for row in scored_df.itertuples(index=False)
    }


# ==============================================================================
# INTEGRATION HOOKS FOR MEMBER 1 & MEMBER 2
# ==============================================================================

def load_candidates_and_scores(
    trip_id: str,
) -> Tuple[List[CandidateObservation], TransitionDistFn, Dict[int, List[Tuple[float, float]]]]:
    """
    TODO (Replace mock block below once Member 1 provides their loader):
      1. Call Member 1's loader to get `trip`, `geometry`, `candidates_df`,
         `shortest_network_distance`, and `arc_geometries`.
      2. Call `score_map = score_with_member2(trip, geometry, candidates_df)`.
      3. Build `List[CandidateObservation]` with `cand["score"] = score_map[(obs_id, arc_id)]`.
    """
    # Realistic mock trip over Bengaluru roads (Cubbon Park -> MG Road)
    observations = [
        CandidateObservation(trip_id, 0, 0.0, 77.5940, 12.9710, [
            {"candidate_id": 1, "arc_id": 101, "proj_fraction": 0.1, "score": 0.92},
            {"candidate_id": 2, "arc_id": 901, "proj_fraction": 0.2, "score": 0.15},
        ]),
        CandidateObservation(trip_id, 1, 5.0, 77.5948, 12.9714, [
            # Parallel disconnected side-street has higher raw classifier score (0.78 vs 0.35)
            {"candidate_id": 3, "arc_id": 902, "proj_fraction": 0.5, "score": 0.78},
            {"candidate_id": 4, "arc_id": 102, "proj_fraction": 0.3, "score": 0.35},
        ]),
        CandidateObservation(trip_id, 2, 10.0, 77.5956, 12.9718, [
            {"candidate_id": 5, "arc_id": 103, "proj_fraction": 0.6, "score": 0.89},
        ]),
        CandidateObservation(trip_id, 3, 15.0, 77.5964, 12.9722, [
            {"candidate_id": 6, "arc_id": 104, "proj_fraction": 0.9, "score": 0.94},
        ]),
    ]

    def mock_transition_dist(a1: int, p1: float, a2: int, p2: float) -> Optional[float]:
        # Main corridor: 101 -> 102 -> 103 -> 104 (each 110m)
        if a1 == a2 and a1 in (101, 102, 103, 104):
            return abs(p2 - p1) * 110.0
        if a1 in (101, 102, 103) and a2 == a1 + 1:
            return (1.0 - p1) * 110.0 + p2 * 110.0
        # Parallel street (901, 902) has no valid connection to main corridor
        return None

    arc_geometries: Dict[int, List[Tuple[float, float]]] = {
        101: [(77.5939, 12.9709), (77.5944, 12.9712)],
        102: [(77.5944, 12.9712), (77.5952, 12.9716)],
        103: [(77.5952, 12.9716), (77.5960, 12.9720)],
        104: [(77.5960, 12.9720), (77.5966, 12.9723)],
        901: [(77.5938, 12.9715), (77.5945, 12.9719)],
        902: [(77.5945, 12.9719), (77.5953, 12.9723)],
    }

    return observations, mock_transition_dist, arc_geometries


def load_reference_labels(
    trip_id: str,
) -> Tuple[Optional[List[Optional[int]]], Optional[List[int]], Optional[List[List[Tuple[float, float]]]]]:
    """
    TODO (Member 1 Server-Side Reference Loader):
      Load aligned observation-level reference arc IDs and raw reference route
      strictly on Laptop 1 for server-side evaluation.
    """
    aligned_ref_arcs: List[Optional[int]] = [101, 102, 103, 104]
    raw_ref_route: List[int] = [101, 102, 103, 104]
    ref_segments = [[
        (77.5939, 12.9709),
        (77.5944, 12.9712),
        (77.5952, 12.9716),
        (77.5960, 12.9720),
        (77.5966, 12.9723),
    ]]
    return aligned_ref_arcs, raw_ref_route, ref_segments


# ==============================================================================
# CORE ROUTE RECONSTRUCTION PIPELINE
# ==============================================================================

def build_route_segments(
    decoded_points: List[DecodedPoint],
    arc_geometries: Dict[int, List[Tuple[float, float]]],
) -> List[List[Tuple[float, float]]]:
    """
    Assemble connected polylines from decoded arcs, splitting into separate
    segments whenever an observation is 'unmatched' or triggers a 'gap_reset'.
    """
    segments: List[List[Tuple[float, float]]] = []
    current_seg: List[Tuple[float, float]] = []
    last_arc: Optional[int] = None

    for pt in decoded_points:
        if pt.arc_id is None:
            if current_seg:
                segments.append(current_seg)
                current_seg = []
                last_arc = None
            continue

        if pt.status == "gap_reset" and current_seg:
            segments.append(current_seg)
            current_seg = []
            last_arc = None

        if pt.arc_id != last_arc:
            coords = arc_geometries.get(pt.arc_id, [])
            if coords:
                if current_seg and current_seg[-1] == coords[0]:
                    current_seg.extend(coords[1:])
                else:
                    current_seg.extend(coords)
            last_arc = pt.arc_id

    if current_seg:
        segments.append(current_seg)
    return segments
def run_trip_inference(
    trip_id: str,
    artifact_version: str = "v0.3.0-adapter",
    hmm_config: Optional[HMMConfig] = None,
) -> Tuple[PredictionResult, TransitionDistFn]:
    """
    End-to-end prediction wrapper returning PredictionResult (with zero reference labels).
    """
    t0 = time.perf_counter()

    t_cand = time.perf_counter()
    observations, transition_dist_fn, arc_geometries = load_candidates_and_scores(trip_id)
    cand_cls_ms = round((time.perf_counter() - t_cand) * 1000, 2)

    t_hmm = time.perf_counter()
    decoded_points, warnings = viterbi_decode(
        observations, transition_dist_fn, config=hmm_config or HMMConfig()
    )
    hmm_ms = round((time.perf_counter() - t_hmm) * 1000, 2)

    route_segments = build_route_segments(decoded_points, arc_geometries)
    total_ms = round((time.perf_counter() - t0) * 1000, 2)

    result = PredictionResult(
        trip_id=trip_id,
        artifact_version=artifact_version,
        points=decoded_points,
        route_segments=route_segments,
        warnings=warnings,
        timings_ms={
            "candidate_and_classifier": cand_cls_ms,
            "viterbi_decoding": hmm_ms,
            "total": total_ms,
        },
    )
    return result, transition_dist_fn