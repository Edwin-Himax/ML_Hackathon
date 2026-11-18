"""Server/offline scoring ONLY, invoked after a complete prediction.

Reference geometry, IDs and pointwise correctness never enter public output.
Optional official pointwise CSV labels replace the provisional alignment.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pathsnap_ml.alignment import align_reference


def collapse(sequence):
    output = []
    for value in sequence:
        value = int(value)
        if value >= 0 and (not output or output[-1] != value):
            output.append(value)
    return output


def edit_similarity(a, b):
    """1 - Levenshtein distance / max length; visits retain their order."""
    if not a and not b:
        return 1.
    if len(a) > len(b):
        a, b = b, a
    previous = np.arange(len(a) + 1)
    left_values, indices = np.asarray(a), np.arange(len(a) + 1)
    # Vectorized insertion closure of the ordinary Levenshtein recurrence.
    # min_k(base[k] + i-k) = i + prefix_min(base[k]-k).
    for j, right in enumerate(b, 1):
        base = np.r_[j, np.minimum(previous[1:] + 1, previous[:-1] + (left_values != right))]
        previous = np.minimum.accumulate(base - indices) + indices
    return float(1 - previous[-1] / max(len(a), len(b)))


class Scorer:
    def __init__(self, reference_root=None, pointwise_root=None, provisional=True):
        self.reference_root = Path(reference_root) if reference_root else None
        self.pointwise_root = Path(pointwise_root) if pointwise_root else None
        self.provisional = provisional

    def reference_path(self, tid, suffix):
        root = self.reference_root if suffix == '.route' else self.pointwise_root
        if root is None:
            return None
        for path in [root / f'{tid}{suffix}', root / tid / f'{tid}{suffix}']:
            if path.is_file():
                return path
        return None

    def load_reference(self, tid, n_arcs):
        path = self.reference_path(tid, '.route')
        if path is None:
            return None
        route = np.loadtxt(path, ndmin=1)
        if route.ndim != 1 or not len(route) or not np.isfinite(route).all() or not np.equal(route, np.floor(route)).all() or (route < 0).any() or (route >= n_arcs).any():
            raise ValueError('Reference route has invalid arc identifiers')
        return route.astype(int)

    def labels(self, prediction, bundle, route):
        p = prediction
        official = self.reference_path(p.trip.trip_id, '.csv')
        if official:
            df = pd.read_csv(official)
            if not {'observation_id', 'arc_id'} <= set(df):
                raise ValueError('Pointwise CSV requires observation_id and arc_id')
            raw = df[['observation_id', 'arc_id']].to_numpy(dtype=float)
            if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
                raise ValueError('Pointwise identifiers must be finite integers')
            ids, arcs = raw.astype(int).T
            if len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= len(p.trip.track)).any() or (arcs < 0).any() or (arcs >= len(p.trip.arcs)).any():
                raise ValueError('Invalid or duplicate pointwise identifiers')
            refs = np.full(len(p.trip.track), -1, dtype=int)
            eligible = np.zeros(len(refs), dtype=bool)
            refs[ids] = arcs
            eligible[ids] = True
            return refs, eligible, 'official_pointwise'
        if self.provisional and route is not None:
            labels = align_reference(p.trip, p.geometry, route, bundle['config']['alignment'])
            return labels.reference_arc_id.to_numpy(), labels.eligible.to_numpy(), 'provisional_alignment'
        return None, None, 'unavailable'

    def score(self, prediction, bundle):
        p = prediction
        route = self.load_reference(p.trip.trip_id, len(p.trip.arcs))
        refs, eligible, status = self.labels(p, bundle, route)
        result = {'status': status, 'eligible_observations': int(eligible.sum()) if eligible is not None else 0,
                  'excluded_observations': int((~eligible).sum()) if eligible is not None else len(p.trip.track),
                  'label_notice': 'Agreement with derived route alignment; not independently verified GPS-point truth.' if status == 'provisional_alignment' else ('Official pointwise labels supplied to the isolated scorer.' if status == 'official_pointwise' else 'No usable pointwise reference; accuracy is unavailable.'),
                  'classifier': {}, 'hmm': {}}
        for name, assigned, reconstructed in [('classifier', p.classifier.assignments, p.classifier_route), ('hmm', p.decoded.assignments, p.decoded_route)]:
            metrics = {'top1_accuracy': None, 'arc_macro_f1': None, 'sequence_similarity': None}
            if refs is not None and eligible.any():
                metrics['top1_accuracy'] = float((assigned[eligible] == refs[eligible]).mean())
                # Local arc classification, reference classes only; unmatched predictions
                # still contribute false negatives. Report separately from binary candidate F1.
                metrics['arc_macro_f1'] = float(f1_score(refs[eligible], assigned[eligible], labels=np.unique(refs[eligible]), average='macro', zero_division=0))
            if route is not None:
                predicted = [arc for seq in reconstructed['arc_sequences'] for arc in seq]
                metrics['sequence_similarity'] = edit_similarity(collapse(route), collapse(predicted))
            result[name] = metrics
        if refs is not None and eligible.any():
            hit = np.zeros(len(refs), dtype=bool)
            obs = p.candidates.observation_id.to_numpy(dtype=int)
            arc = p.candidates.arc_id.to_numpy(dtype=int)
            hit[obs[arc == refs[obs]]] = True
            result['candidate_miss_rate'] = float((~hit[eligible]).mean())
        else:
            result['candidate_miss_rate'] = None
        return result
