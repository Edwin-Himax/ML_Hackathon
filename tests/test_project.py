from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from pathsnap_ml.geometry import Trip, Geometry
from pathsnap.graph import RoadGraph
from pathsnap.decoding import prepare_lattice, viterbi, reconstruct
from pathsnap.pipeline import Predictor, public_prediction
from pathsnap.evaluation import Scorer, edit_similarity, collapse
from pathsnap.settings import Settings, ROOT
from api.app import create_app

CFG = json.loads((ROOT / 'configs/decoder.json').read_text())


def tiny_graph(arcs=None):
    trip = Trip('tiny', np.array([[.00025, 0., 0.], [.00175, 0., 2.]]),
                np.array([[0., 0.], [.001, 0.], [.002, 0.], [.01, 0.], [.011, 0.]]),
                np.array(arcs if arcs is not None else [[0, 1], [1, 2], [3, 4]], dtype=int))
    geometry = Geometry(trip)
    return trip, geometry, RoadGraph(trip, geometry, cache_entries=2)


class GraphTests(unittest.TestCase):
    def test_same_arc_partial_distance(self):
        _, g, graph = tiny_graph()
        distance, path = graph.connection(0, .2, 0, .8, 1000, with_path=True)
        self.assertAlmostEqual(distance, .6 * g.lengths[0])
        self.assertEqual(path, [0])

    def test_adjacent_arcs_include_both_partials(self):
        _, g, graph = tiny_graph()
        distance, path = graph.connection(0, .75, 1, .25, 1000, with_path=True)
        self.assertAlmostEqual(distance, .25 * (g.lengths[0] + g.lengths[1]))
        self.assertEqual(path, [0, 1])

    def test_direction_and_disconnection(self):
        _, _, graph = tiny_graph()
        self.assertFalse(np.isfinite(graph.connection(1, .2, 0, .8, 10000)[0]))
        self.assertFalse(np.isfinite(graph.connection(0, .2, 2, .8, 10000)[0]))

    def test_exact_movement_bound(self):
        _, _, graph = tiny_graph()
        self.assertFalse(np.isfinite(graph.connection(0, .1, 1, .9, 30)[0]))

    def test_small_jitter_allowed_but_long_backward_move_needs_cycle(self):
        _, _, graph = tiny_graph()
        self.assertEqual(graph.connection(0, .5, 0, .49, 200)[0], 0.)
        self.assertFalse(np.isfinite(graph.connection(0, .9, 0, .1, 200)[0]))

    def test_parallel_arcs_are_not_summed_and_paths_keep_ids(self):
        _, g, graph = tiny_graph([[0, 1], [0, 1], [1, 2], [2, 0]])
        distances, parents = graph.search(0, 500)
        self.assertAlmostEqual(distances[1], g.lengths[0])
        self.assertIn(parents[1], [0, 1])

    def test_zero_length_arcs_and_cache_bound(self):
        _, _, graph = tiny_graph([[0, 0], [0, 1], [1, 2], [2, 0]])
        for source in [0, 1, 2]:
            distances, _ = graph.search(source, 500)
            self.assertEqual(distances[source], 0)
        self.assertLessEqual(len(graph.cache), 2)


class DecoderTests(unittest.TestCase):
    def build(self, observations, arcs, fractions, probabilities, timestamps=None):
        trip, geo, graph = tiny_graph()
        if timestamps is not None:
            trip.track[:, 2] = timestamps
        candidates = pd.DataFrame({'observation_id': observations, 'arc_id': arcs, 'projection_fraction': fractions})
        scores = candidates[['observation_id', 'arc_id']].copy()
        scores['log_emission_potential'] = np.log(probabilities)
        lattice = prepare_lattice(trip, geo, candidates, graph, CFG)
        decoded = viterbi(trip, candidates, scores, lattice, CFG)
        return trip, geo, graph, candidates, decoded

    def test_disconnected_high_score_loses_to_connected_candidate(self):
        *_, result = self.build([0, 1, 1], [0, 1, 2], [.25, .75, .75], [1., .3, .7])
        self.assertEqual(result.assignments.tolist(), [0, 1])

    def test_impossible_transition_restarts_instead_of_faking_path(self):
        trip, geo, graph, candidates, result = self.build([0, 1], [0, 2], [.25, .75], [1., 1.])
        self.assertTrue(result.break_before[1])
        self.assertEqual(result.resets[-1]['reason'], 'infeasible_transition')
        route = reconstruct(trip, geo, candidates, graph, result, CFG)
        self.assertEqual(len(route['pieces']), 2)
        self.assertEqual(route['summary']['continuity'], 0)

    def test_large_gap_resets_even_when_states_are_connected(self):
        *_, result = self.build([0, 1], [0, 1], [.25, .75], [1., 1.], [0., 100.])
        self.assertEqual(result.resets[-1]['reason'], 'gps_gap')

    def test_no_candidate_retains_unmatched_observation(self):
        *_, result = self.build([0], [0], [.25], [1.])
        self.assertEqual(result.assignments.tolist(), [0, -1])
        self.assertEqual(result.resets[-1]['reason'], 'no_candidates')

    def test_empty_candidate_trip_is_supported(self):
        *_, result = self.build([], [], [], [])
        self.assertEqual(result.assignments.tolist(), [-1, -1])


class SequenceTests(unittest.TestCase):
    def test_revisits_survive_collapse(self):
        self.assertEqual(collapse([1, 1, 2, 1]), [1, 2, 1])

    def test_known_edit_distances(self):
        self.assertEqual(edit_similarity([], []), 1)
        self.assertEqual(edit_similarity([1, 2], []), 0)
        self.assertAlmostEqual(edit_similarity([1, 2, 3], [1, 4, 3]), 2 / 3)
        self.assertAlmostEqual(edit_similarity([1, 2], [1, 9, 2]), 2 / 3)

    def test_vectorized_edit_distance_matches_scalar_reference(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            a, b = rng.integers(0, 4, size=rng.integers(1, 12)).tolist(), rng.integers(0, 4, size=rng.integers(1, 12)).tolist()
            dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
            dp[:, 0] = np.arange(len(a) + 1); dp[0] = np.arange(len(b) + 1)
            for i in range(1, len(a) + 1):
                for j in range(1, len(b) + 1):
                    dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1,j-1] + (a[i-1] != b[j-1]))
            self.assertAlmostEqual(edit_similarity(a, b), 1 - dp[-1, -1] / max(len(a), len(b)))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.settings = Settings(log_path=cls.folder / 'requests.jsonl')
        cls.client_context = TestClient(create_app(cls.settings))
        cls.client = cls.client_context.__enter__()
        cls.predictor = cls.client.app.state.predictor

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        cls.temp.cleanup()

    def test_health_and_ten_trips(self):
        self.assertEqual(self.client.get('/health').json()['status'], 'ready')
        self.assertEqual(len(self.client.get('/trips').json()['trips']), 10)

    def test_trip_list_does_not_open_references(self):
        with patch.object(Scorer, 'load_reference', side_effect=AssertionError('reference access')):
            self.assertEqual(self.client.get('/trips').status_code, 200)

    def test_invalid_and_unknown_trip_ids(self):
        self.assertEqual(self.client.post('/predict', json={'trip_id': '../train'}).status_code, 422)
        self.assertEqual(self.client.post('/predict', json={'trip_id': '99999999'}).status_code, 404)
        self.assertEqual(self.client.post('/predict', json={'trip_id': '00000052', 'reference': [1]}).status_code, 422)

    def test_actual_prediction_fresh_ids_and_reference_whitelist(self):
        first = self.client.post('/predict', json={'trip_id': '00000052'})
        second = self.client.post('/predict', json={'trip_id': '00000052'})
        self.assertEqual(first.status_code, 200)
        body = first.json()
        self.assertNotEqual(body['request_id'], second.json()['request_id'])
        self.assertTrue(body['fresh_inference'])
        self.assertEqual(len(body['hmm']['assignments']), body['observation_count'])
        def keys(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    yield k
                    yield from keys(v)
            elif isinstance(value, list):
                for item in value:
                    yield from keys(item)
        forbidden = {'reference_arc_id', 'reference_route', 'reference_geometry', 'correctness', 'route_occurrence', 'ground_truth'}
        self.assertFalse(forbidden & set(keys(body)))

    def test_raw_prediction_never_loads_route(self):
        original = np.loadtxt
        def guarded(path, *args, **kwargs):
            if str(path).endswith('.route'):
                raise AssertionError('Reference leaked into predictor')
            return original(path, *args, **kwargs)
        with patch('numpy.loadtxt', side_effect=guarded):
            p = self.predictor.predict(self.settings.partition_root / '00000052')
            self.assertEqual(len(p.decoded.assignments), 262)

    def test_missing_reference_still_returns_prediction(self):
        scorer = self.client.app.state.scorer
        self.client.app.state.scorer = Scorer(None, None, provisional=False)
        try:
            response = self.client.post('/predict', json={'trip_id': '00000052'})
        finally:
            self.client.app.state.scorer = scorer
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()['evaluation']['hmm']['top1_accuracy'])

    def test_cors_and_static_paths(self):
        result = self.client.options('/predict', headers={'Origin': 'http://localhost:8001', 'Access-Control-Request-Method': 'POST'})
        self.assertEqual(result.headers['access-control-allow-origin'], 'http://localhost:8001')
        self.assertEqual(self.client.get('/ui/').status_code, 200)
        self.assertEqual(self.client.get('/ui/styles.css').status_code, 200)
        self.assertEqual(self.client.get('/Student%20folder/Student%20folder/student_test/00000052/00000052.route').status_code, 404)

    def test_official_pointwise_csv_replaces_provisional_labels(self):
        p = self.predictor.predict(self.settings.partition_root / '00000052')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pd.DataFrame({'observation_id': [0, 1], 'arc_id': p.decoded.assignments[:2]}).to_csv(root / '00000052.csv', index=False)
            metrics = Scorer(None, root).score(p, self.predictor.bundle)
            self.assertEqual(metrics['status'], 'official_pointwise')
            self.assertEqual(metrics['eligible_observations'], 2)
            self.assertEqual(metrics['hmm']['top1_accuracy'], 1.)

    def test_evaluator_settings_never_auto_read_development_references(self):
        with patch.dict('os.environ', {}, clear=True):
            settings = Settings(partition='evaluator_test')
        self.assertIsNone(settings.reference_root)


if __name__ == '__main__':
    unittest.main()
