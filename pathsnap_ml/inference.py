"""Deployment handoff to Member 3. No imports of offline alignment/evaluation.

Candidate scores are binary compatibility probabilities, not calibrated HMM
likelihoods. Normalized values are potentials for a documented hybrid decoder.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .features import FEATURE_NAMES, make_features
from .models import positive_scores


def load_bundle(path):
    """Load only your team's trusted joblib artifact; pickle can execute code."""
    bundle = joblib.load(Path(path))
    if bundle['feature_names'] != FEATURE_NAMES:
        raise ValueError('Artifact feature schema does not match code')
    return bundle


def score_candidates(bundle, features, candidates):
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(bundle['feature_names']) or len(x) != len(candidates):
        raise ValueError('Feature/candidate shape mismatch')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite inference features')
    if candidates.duplicated(['observation_id', 'arc_id']).any():
        raise ValueError('Duplicate candidate keys')
    result = candidates[['observation_id', 'arc_id']].copy()
    result['compatibility_probability'] = positive_scores(bundle['model'], x)
    positive = np.clip(result.compatibility_probability.to_numpy(), 1e-9, 1)
    totals = pd.Series(positive, index=result.index).groupby(result.observation_id).transform('sum').to_numpy()
    result['emission_potential'] = positive / totals
    result['log_emission_potential'] = np.log(result.emission_potential)
    return result


def predict_from_geometry(bundle, trip, geometry, candidates):
    """Member 1's geometry + candidate frame -> Member 3's score table."""
    features = make_features(trip, geometry, candidates)
    result = score_candidates(bundle, features, candidates)
    result.insert(0, 'trip_id', trip.trip_id)
    return result


def classifier_assignments(scores, n_observations):
    """One arc per original observation, -1 for no candidate. No route labels."""
    assignments = np.full(n_observations, -1, dtype=np.int64)
    if len(scores):
        ranked = scores.sort_values(['observation_id', 'compatibility_probability', 'arc_id'], ascending=[True, False, True], kind='stable')
        chosen = ranked.drop_duplicates('observation_id')
        obs = chosen.observation_id.to_numpy(dtype=int)
        if obs.min() < 0 or obs.max() >= n_observations:
            raise ValueError('Observation ID outside supplied trip')
        assignments[obs] = chosen.arc_id.to_numpy(dtype=int)
    return assignments
