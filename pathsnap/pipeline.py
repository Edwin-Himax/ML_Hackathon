"""Fresh prediction from raw GPS/network; deliberately no reference imports."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from pathsnap_ml.geometry import load_trip, Geometry
from pathsnap_ml.inference import load_bundle, predict_from_geometry
from .graph import RoadGraph
from .decoding import prepare_lattice, viterbi, classifier_decoded, reconstruct


@dataclass
class Prediction:
    trip: object
    geometry: object
    graph: object
    candidates: object
    scores: object
    classifier: object
    decoded: object
    classifier_route: dict
    decoded_route: dict
    timings: dict
    model_version: str
    decoder_config: dict


class Predictor:
    def __init__(self, bundle_path, decoder_config):
        self.bundle = load_bundle(bundle_path)
        self.decoder_config = dict(decoder_config)
        self.model_version = hashlib.sha256(Path(bundle_path).read_bytes()).hexdigest()[:12]

    def predict(self, folder, decoder_config=None):
        config = dict(decoder_config or self.decoder_config)
        timings, total = {}, time.perf_counter()
        start = time.perf_counter()
        trip = load_trip(folder)
        geometry = Geometry(trip, self.bundle['config']['candidate']['index_spacing_m'])
        graph = RoadGraph(trip, geometry, config['distance_cache_entries'])
        timings['load_graph_s'] = time.perf_counter() - start
        start = time.perf_counter()
        candidates = geometry.candidates(trip, self.bundle['config']['candidate']).reset_index(drop=True)
        timings['candidates_s'] = time.perf_counter() - start
        start = time.perf_counter()
        with threadpool_limits(limits=self.bundle['config']['threads']):
            scores = predict_from_geometry(self.bundle, trip, geometry, candidates)
        classifier = classifier_decoded(trip, candidates, scores)
        timings['features_classifier_s'] = time.perf_counter() - start
        start = time.perf_counter()
        lattice = prepare_lattice(trip, geometry, candidates, graph, config)
        timings['transitions_s'] = time.perf_counter() - start
        start = time.perf_counter()
        decoded = viterbi(trip, candidates, scores, lattice, config)
        timings['viterbi_s'] = time.perf_counter() - start
        start = time.perf_counter()
        classifier_route = reconstruct(trip, geometry, candidates, graph, classifier, config)
        decoded_route = reconstruct(trip, geometry, candidates, graph, decoded, config)
        timings['route_geometry_s'] = time.perf_counter() - start
        timings['total_prediction_s'] = time.perf_counter() - total
        return Prediction(trip, geometry, graph, candidates, scores, classifier, decoded, classifier_route, decoded_route, timings, self.model_version, config)


def public_prediction(prediction):
    """Explicit response whitelist: never serializes any scorer/reference object."""
    p, trip = prediction, prediction.trip
    lo = trip.track[:, :2].min(axis=0) - .003
    hi = trip.track[:, :2].max(axis=0) + .003
    starts, ends = trip.nodes[trip.arcs[:, 0]], trip.nodes[trip.arcs[:, 1]]
    # Segment bounding boxes intersect the display extent.
    keep = ((np.maximum(starts, ends) >= lo) & (np.minimum(starts, ends) <= hi)).all(axis=1)
    road_ids = np.flatnonzero(keep)
    full_count = len(road_ids)
    if len(road_ids) > 18000:
        road_ids = road_ids[np.linspace(0, len(road_ids) - 1, 18000).astype(int)]
    selected = np.unique(p.decoded.assignments[p.decoded.assignments >= 0])
    road_ids = np.union1d(road_ids, selected)
    def rounded_pieces(route):
        return [np.round(piece, 6).tolist() for piece in route['pieces']]
    def snapped(route):
        return [[round(float(v), 6) for v in row] if np.isfinite(row).all() else None for row in route['snapped_coordinates']]
    return {
        'trip_id': trip.trip_id, 'model_version': p.model_version,
        'fresh_inference': True, 'observation_count': len(trip.track),
        'gps': np.round(trip.track[:, :2], 6).tolist(),
        'timestamps_s': trip.track[:, 2].tolist(),
        'road_segments': np.round(trip.nodes[trip.arcs[road_ids]], 6).tolist(),
        'road_geometry_total_in_extent': full_count,
        'road_geometry_displayed': len(road_ids),
        'classifier': {'assignments': p.classifier.assignments.tolist(), 'pieces': rounded_pieces(p.classifier_route), 'summary': p.classifier_route['summary']},
        'hmm': {'assignments': p.decoded.assignments.tolist(), 'pieces': rounded_pieces(p.decoded_route), 'snapped': snapped(p.decoded_route), 'summary': p.decoded_route['summary'], 'resets': [r for r in p.decoded.resets if r['reason'] != 'initial']},
        'network': {'nodes': len(trip.nodes), 'arcs': len(trip.arcs), 'weak_components': int(p.graph.n_components)},
        'timings': p.timings,
    }
