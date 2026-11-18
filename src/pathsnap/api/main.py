import logging
import os
import uuid
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pathsnap.evaluation.metrics import evaluate_trip
from src.pathsnap.inference import load_reference_labels, run_trip_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap Inference API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

APPROVED_TRIPS = {
    "student_testing": [f"student_trip_{i:02d}" for i in range(1, 11)],
    "evaluator_testing": [f"eval_trip_{i:02d}" for i in range(1, 11)],
}
ACTIVE_PARTITION = os.getenv("PATHSNAP_PARTITION", "student_testing")
ARTIFACT_VERSION = os.getenv("PATHSNAP_ARTIFACT_VERSION", "v0.3.0-adapter")


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "artifact_version": ARTIFACT_VERSION,
        "partition": ACTIVE_PARTITION,
    }


@app.get("/trips")
def list_trips() -> Dict[str, object]:
    return {
        "partition": ACTIVE_PARTITION,
        "trips": APPROVED_TRIPS[ACTIVE_PARTITION],
    }


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, object]:
    request_id = str(uuid.uuid4())[:8]

    if req.trip_id not in APPROVED_TRIPS[ACTIVE_PARTITION]:
        logger.warning(f"[{request_id}] Rejected unknown trip_id={req.trip_id}")
        raise HTTPException(status_code=404, detail=f"Unknown trip_id: {req.trip_id}")

    pred_result, transition_dist_fn = run_trip_inference(
        trip_id=req.trip_id,
        artifact_version=ARTIFACT_VERSION,
    )

    aligned_ref_arcs, raw_ref_route, _ = load_reference_labels(req.trip_id)
    metrics = evaluate_trip(
        decoded_points=pred_result.points,
        transition_dist_fn=transition_dist_fn,
        aligned_ref_arcs=aligned_ref_arcs,
        raw_ref_route=raw_ref_route,
    )

    logger.info(
        f"[{request_id}] trip_id={req.trip_id} | obs={len(pred_result.points)} "
        f"| timings={pred_result.timings_ms} | warnings={len(pred_result.warnings)}"
    )

    return {
        "request_id": request_id,
        "trip_id": pred_result.trip_id,
        "artifact_version": pred_result.artifact_version,
        "points": [
            {
                "observation_id": p.observation_id,
                "timestamp": p.timestamp,
                "gps_lon": p.gps_lon,
                "gps_lat": p.gps_lat,
                "arc_id": p.arc_id,
                "proj_fraction": p.proj_fraction,
                "status": p.status,
            }
            for p in pred_result.points
        ],
        "predicted_arcs": [p.arc_id for p in pred_result.points],
        "route_geometry": pred_result.route_segments,
        "metrics": metrics,
        "warnings": pred_result.warnings,
        "timings_ms": pred_result.timings_ms,
    }