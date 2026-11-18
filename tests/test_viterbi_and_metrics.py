from src.pathsnap.decoding.viterbi import (
    HMMConfig,
    decode_classifier_only,
    viterbi_decode,
)
from src.pathsnap.evaluation.metrics import evaluate_trip
from src.pathsnap.schemas import CandidateObservation


def mock_transition_dist(arc_from: int, proj_from: float, arc_to: int, proj_to: float):
    if arc_from == arc_to:
        return abs(proj_to - proj_from) * 100.0
    if (arc_from, arc_to) in ((10, 11), (11, 12)):
        return (1.0 - proj_from) * 100.0 + proj_to * 100.0
    return None  # Disconnected parallel road


def test_pipeline():
    obs = [
        CandidateObservation("t1", 0, 0.0, 77.5000, 12.9000, [
            {"candidate_id": 1, "arc_id": 10, "proj_fraction": 0.2, "score": 0.90},
        ]),
        # Noisy GPS gives disconnected parallel arc 99 a higher classifier score
        CandidateObservation("t1", 1, 5.0, 77.5008, 12.9000, [
            {"candidate_id": 2, "arc_id": 99, "proj_fraction": 0.5, "score": 0.75},
            {"candidate_id": 3, "arc_id": 11, "proj_fraction": 0.1, "score": 0.25},
        ]),
        # Empty candidate set -> unmatched
        CandidateObservation("t1", 2, 10.0, 77.5015, 12.9000, []),
        # 45s GPS gap -> gap_reset
        CandidateObservation("t1", 3, 55.0, 77.5025, 12.9000, [
            {"candidate_id": 4, "arc_id": 12, "proj_fraction": 0.8, "score": 0.85},
        ]),
    ]

    clf_only = decode_classifier_only(obs)
    hmm_out, warnings = viterbi_decode(obs, mock_transition_dist, HMMConfig(max_gap_sec=30.0))

    clf_eval = evaluate_trip(clf_only, mock_transition_dist, aligned_ref_arcs=[10, 11, 11, 12], raw_ref_route=[10, 11, 12])
    hmm_eval = evaluate_trip(hmm_out, mock_transition_dist, aligned_ref_arcs=[10, 11, 11, 12], raw_ref_route=[10, 11, 12])

    print("Classifier-only metrics:", clf_eval)
    print("Classifier+HMM metrics :", hmm_eval)
    print("Warnings:", warnings)

    assert [p.arc_id for p in hmm_out] == [10, 11, None, 12]
    assert hmm_eval["observation_top1_accuracy"] > clf_eval["observation_top1_accuracy"]
    print("ALL MEMBER 3 UNIT TESTS PASSED!")


if __name__ == "__main__":
    test_pipeline()