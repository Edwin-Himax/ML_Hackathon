import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from pathsnap_ml.geometry import Trip, Geometry, point_segments, load_trip, project
from pathsnap_ml.features import make_features, FEATURE_NAMES
from pathsnap_ml.alignment import align_reference
from pathsnap_ml.dataset import make_folds
from pathsnap_ml.inference import score_candidates, classifier_assignments
from pathsnap_ml.metrics import evaluate_trip
from pathsnap_ml.bayesian import propose, decode, SPACES


CFG = {'radius_m': 50., 'fallback_radii_m': [100., 250.], 'max_candidates': 8, 'index_spacing_m': 30.}
ALIGN = {'nearest_occurrences': 8, 'max_label_distance_m': 50., 'min_emission_margin': .15, 'gap_reset_s': 60., 'break_exclusion_m': 15.}


class DummyModel:
    classes_ = np.array([0, 1])
    def predict_proba(self, x):
        p = np.clip(x[:, 0], 0, 1)
        return np.column_stack((1 - p, p))


class Member2Tests(unittest.TestCase):
    def trip(self):
        return Trip('demo', np.array([[0.0002, 0, 0], [.0004, 0, 1], [.0006, 0, 2]]), np.array([[0., 0.], [.001, 0.], [.002, 0.]]), np.array([[0, 1], [1, 0], [1, 2], [1, 1]]))

    def test_projection_origin_and_distance(self):
        p = project(np.array([[0, 0], [.001, 0]]), (0, 0))
        np.testing.assert_allclose(p[0], [0, 0], atol=1e-6)
        self.assertAlmostEqual(p[1, 0], 111.19508, places=3)

    def test_segment_broadcast_and_zero_length(self):
        d, f, _ = point_segments(np.array([[[1., 1.]], [[2., 0.]]]), np.array([[[0., 0.], [0., 0.]]]), np.array([[[2., 0.], [0., 0.]]]))
        self.assertEqual(d.shape, (2, 2))
        self.assertTrue(np.isfinite(f).all())
        self.assertAlmostEqual(d[0, 0], 1.)

    def test_spatial_index_finds_middle_of_long_edge_and_preserves_reverse(self):
        t = Trip('long', np.array([[.05, 0, 0]]), np.array([[0., 0.], [.1, 0.]]), np.array([[0, 1], [1, 0]]))
        g = Geometry(t)
        c = g.candidates(t, CFG)
        self.assertEqual(set(c.arc_id), {0, 1})
        self.assertLess(c.distance_m.max(), 1e-6)

    def test_features_have_no_absolute_ids_and_are_finite(self):
        t = self.trip(); g = Geometry(t); c = g.candidates(t, CFG)
        x = make_features(t, g, c)
        self.assertEqual(x.shape, (len(c), len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(x).all())
        self.assertNotIn('arc_id', FEATURE_NAMES)
        moved = Trip('other', t.track.copy(), t.nodes.copy(), t.arcs.copy())
        np.testing.assert_allclose(x, make_features(moved, g, c))

    def test_loader_and_features_work_without_reference_file(self):
        t = self.trip()
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder) / 'demo'; f.mkdir()
            for ext, a in [('track', t.track), ('nodes', t.nodes), ('arcs', t.arcs)]: np.savetxt(f / f'demo.{ext}', a)
            loaded = load_trip(f)
            geo = Geometry(loaded)
            self.assertGreater(len(make_features(loaded, geo, geo.candidates(loaded, CFG))), 0)

    def test_alignment_has_original_observations_and_forward_arc(self):
        t = self.trip(); g = Geometry(t)
        labels = align_reference(t, g, np.array([0, 2]), ALIGN)
        self.assertEqual(labels.observation_id.tolist(), [0, 1, 2])
        self.assertEqual(labels.reference_arc_id.tolist(), [0, 0, 0])

    def test_group_folds_have_no_overlap_and_cover_all_trips(self):
        trips = [{'trip_id': f'{i:08d}'} for i in range(80)]
        folds = make_folds(trips, 5, 42)
        ids = []
        for f in folds:
            self.assertFalse(set(f['train_trip_ids']) & set(f['validation_trip_ids']))
            ids.extend(f['validation_trip_ids'])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 80)

    def test_emissions_normalize_per_observation_keep_keys_and_unmatched(self):
        c = pd.DataFrame({'observation_id': [0, 0, 2], 'arc_id': [7, 9, 11]})
        x = np.zeros((3, len(FEATURE_NAMES))); x[:, 0] = [.2, .8, 0]
        scores = score_candidates({'model': DummyModel(), 'feature_names': FEATURE_NAMES}, x, c)
        np.testing.assert_allclose(scores.groupby('observation_id').emission_potential.sum(), [1, 1])
        np.testing.assert_array_equal(classifier_assignments(scores, 3), [9, -1, 11])
        self.assertEqual(scores.arc_id.tolist(), [7, 9, 11])

    def test_top1_counts_candidate_misses_and_no_candidate_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'test.npz'
            x = np.array([[.8], [.1]])
            np.savez(p, x=x, y=[1, 0], observation_id=[0, 1], arc_id=[5, 7], reference_arc_id=[5, 8, 9], eligible=[True, True, True])
            metrics = evaluate_trip(DummyModel(), {'path': p, 'trip_id': 'test', 'partition': 'student_test'})
            self.assertEqual(metrics['evaluated_observations'], 3)
            self.assertEqual(metrics['candidate_misses'], 2)
            self.assertEqual(metrics['unmatched_observations'], 1)
            self.assertAlmostEqual(metrics['top1_accuracy'], 1 / 3)

    def test_gp_acquisition_is_used_after_random_initialization(self):
        history = []
        for i, u in enumerate([.1, .3, .6, .9]):
            history.append({'status': 'complete', 'unit_parameters': [u], 'parameters': decode([u], SPACES['logistic_regression']), 'objective': 1 - (u - .65) ** 2})
        unit, info = propose(history, 'logistic_regression', {'initial_random_trials': 4, 'acquisition_candidates': 100}, np.random.default_rng(42))
        self.assertEqual(info['strategy'], 'gp_expected_improvement')
        self.assertGreaterEqual(unit[0], 0)
        self.assertLessEqual(unit[0], 1)


if __name__ == '__main__':
    unittest.main()
