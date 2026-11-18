"""Member 2 classifier diagnostics; Member 3 owns route/sequence evaluation.

All reference-based numbers refer to eligible provisional alignment labels.
Candidate misses count against observation Top-1 accuracy; excluded alignments
are counted separately, never silently treated as correct observations.
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, log_loss, confusion_matrix
from .models import positive_scores
from .dataset import observation_sample


def evaluate_trip(model, trip, maximum=None, threshold=0.5):
    with np.load(trip['path']) as d:
        eligible = d['eligible']
        selected = observation_sample(eligible, maximum)
        obs_all = d['observation_id']
        keep = np.isin(obs_all, selected)
        x, y, obs, arcs = d['x'][keep], d['y'][keep], obs_all[keep], d['arc_id'][keep]
        references = d['reference_arc_id']
        start = time.perf_counter()
        probability = positive_scores(model, x)
        inference_s = time.perf_counter() - start
        prediction = probability >= threshold
        if len(y):
            precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, labels=[0, 1], average='macro', zero_division=0)
            loss = log_loss(y, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1])
            cm = confusion_matrix(y, prediction, labels=[0, 1])
        else:
            precision = recall = f1 = loss = float('nan')
            cm = np.zeros((2, 2), dtype=int)
        assigned = np.full(len(eligible), -1, dtype=np.int32)
        hit = np.zeros(len(eligible), dtype=bool)
        if len(obs):
            # Group ID primary, descending score secondary; candidate input order resolves ties.
            order = np.lexsort((np.arange(len(obs)), -probability, obs))
            _, first = np.unique(obs[order], return_index=True)
            chosen = order[first]
            assigned[obs[chosen]] = arcs[chosen]
            hit[obs[y == 1]] = True
        n = len(selected)
        result = {
            'trip_id': trip['trip_id'], 'partition': trip['partition'],
            'all_observations': len(eligible), 'alignment_eligible_observations': int(eligible.sum()),
            'alignment_excluded_observations': int((~eligible).sum()),
            'evaluated_observations': n, 'candidate_rows': len(y),
            'positive_candidate_rows': int(y.sum()),
            'candidate_macro_precision': float(precision), 'candidate_macro_recall': float(recall),
            'candidate_macro_f1': float(f1), 'candidate_log_loss': float(loss),
            'top1_correct': int((assigned[selected] == references[selected]).sum()),
            'top1_accuracy': float((assigned[selected] == references[selected]).mean()) if n else float('nan'),
            'candidate_misses': int((~hit[selected]).sum()),
            'candidate_miss_rate': float((~hit[selected]).mean()) if n else float('nan'),
            'unmatched_observations': int((assigned[selected] == -1).sum()),
            'model_inference_s': inference_s,
            'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0]), 'tp': int(cm[1, 1]),
        }
    return result


def aggregate(rows):
    df = pd.DataFrame(rows)
    denominator = int(df.evaluated_observations.sum())
    # Trip-macro candidate metrics ensure long trips do not dominate selection.
    out = {k: float(df[k].mean()) for k in ['candidate_macro_precision', 'candidate_macro_recall', 'candidate_macro_f1', 'candidate_log_loss']}
    out.update({
        'trips': len(df), 'evaluated_observations': denominator,
        'candidate_rows': int(df.candidate_rows.sum()),
        'top1_accuracy': float(df.top1_correct.sum() / denominator) if denominator else float('nan'),
        'candidate_miss_rate': float(df.candidate_misses.sum() / denominator) if denominator else float('nan'),
        'alignment_eligible_observations': int(df.alignment_eligible_observations.sum()),
        'alignment_excluded_observations': int(df.alignment_excluded_observations.sum()),
        'model_inference_s': float(df.model_inference_s.sum()),
    })
    return out
