import json
from pathlib import Path
from typing import Dict, List

from src.pathsnap.decoding.viterbi import decode_classifier_only
from src.pathsnap.evaluation.metrics import evaluate_trip
from src.pathsnap.inference import (
    build_route_segments,
    load_candidates_and_scores,
    load_reference_labels,
    run_trip_inference,
)
from visualizations.maps import build_trip_folium_map


def run_batch_ablation(trip_ids: List[str]) -> None:
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("visualizations").mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    print(f"{'Trip ID':<18} | {'Method':<18} | {'Top-1 Acc':<10} | {'Continuity':<10} | {'Seq Sim':<10}")
    print("-" * 78)

    for trip_id in trip_ids:
        observations, transition_dist_fn, arc_geometries = load_candidates_and_scores(trip_id)
        aligned_ref, raw_ref, ref_segments = load_reference_labels(trip_id)

        # 1. Baseline: Classifier-Only (no HMM)
        clf_points = decode_classifier_only(observations)
        clf_metrics = evaluate_trip(
            clf_points, transition_dist_fn, aligned_ref_arcs=aligned_ref, raw_ref_route=raw_ref
        )

        # 2. Full Pipeline: Classifier + HMM/Viterbi
        hmm_result, _ = run_trip_inference(trip_id)
        hmm_metrics = evaluate_trip(
            hmm_result.points, transition_dist_fn, aligned_ref_arcs=aligned_ref, raw_ref_route=raw_ref
        )

        # Export interactive HTML map (includes Reference Route layer for report screenshots)
        road_net = list(arc_geometries.values())
        html_out = f"visualizations/{trip_id}_hmm_comparison.html"
        build_trip_folium_map(
            points=hmm_result.points,
            predicted_segments=hmm_result.route_segments,
            road_network_arcs=road_net,
            reference_segments=ref_segments,
            save_html_path=html_out,
        )

        for method, m in [("Classifier-Only", clf_metrics), ("Classifier + HMM", hmm_metrics)]:
            row = {
                "trip_id": trip_id,
                "method": method,
                "observation_top1_accuracy": m.get("observation_top1_accuracy"),
                "route_continuity": m.get("route_continuity"),
                "sequence_similarity": m.get("sequence_similarity"),
                "unmatched_observations": m.get("unmatched_observations"),
                "gap_resets": m.get("gap_resets"),
            }
            summary_rows.append(row)
            print(
                f"{trip_id:<18} | {method:<18} | "
                f"{str(row['observation_top1_accuracy']):<10} | "
                f"{str(row['route_continuity']):<10} | "
                f"{str(row['sequence_similarity']):<10}"
            )

    out_json = Path("results/ablation_summary.json")
    out_json.write_text(json.dumps(summary_rows, indent=2))
    print(f"\nSaved ablation table to {out_json} and exported HTML maps to visualizations/.")


if __name__ == "__main__":
    sample_trips = [f"student_trip_{i:02d}" for i in range(1, 4)]
    run_batch_ablation(sample_trips)