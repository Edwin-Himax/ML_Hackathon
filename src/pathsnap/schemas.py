from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CandidateObservation:
    """Input to HMM/Viterbi from Member 1 (geo) + Member 2 (classifier scores)."""
    trip_id: str
    observation_id: int
    timestamp: float
    gps_lon: float
    gps_lat: float
    # Each candidate dict:
    # {"candidate_id": int, "arc_id": int, "proj_fraction": float,
    #  "dist_to_road_m": float, "score": float}
    candidates: List[Dict] = field(default_factory=list)


@dataclass
class DecodedPoint:
    """Per-observation output from Member 3's Viterbi decoder."""
    observation_id: int
    timestamp: float
    gps_lon: float
    gps_lat: float
    arc_id: Optional[int]          # None if observation is unmatched
    proj_fraction: Optional[float]
    status: str                    # "matched" | "unmatched" | "gap_reset"


@dataclass
class PredictionResult:
    """Output of predict_trip() — strictly isolated from ground-truth labels."""
    trip_id: str
    artifact_version: str
    points: List[DecodedPoint]
    route_segments: List[List[Tuple[float, float]]]  # Connected [lon, lat] polylines
    warnings: List[str]
    timings_ms: Dict[str, float]