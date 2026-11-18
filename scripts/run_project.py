"""Full project reproducible development experiments, maps and report."""
from pathlib import Path
import argparse
import json
import sys
import time
from dataclasses import replace
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.pipeline import Predictor
from pathsnap.evaluation import Scorer
from pathsnap.decoding import prepare_lattice, viterbi, reconstruct


def classifier_routes(args):
    from pathsnap_ml.models import MODEL_NAMES
    from pathsnap_ml.inference import load_bundle, predict_from_geometry
    from pathsnap.decoding import classifier_decoded
    from threadpoolctl import threadpool_limits
    bundles = {name: load_bundle(args.bundle.parent / f'{name}.joblib') for name in MODEL_NAMES}
    config = json.loads(args.decoder.read_text())
    predictor = Predictor(args.bundle, config)
    rows = []
    for folder in sorted((args.data / 'student_test').iterdir()):
        if not folder.is_dir():
            continue
        p = predictor.predict(folder)
        for name, bundle in bundles.items():
            with threadpool_limits(limits=bundle['config']['threads']):
                scores = predict_from_geometry(bundle, p.trip, p.geometry, p.candidates)
            assigned = classifier_decoded(p.trip, p.candidates, scores)
            route = reconstruct(p.trip, p.geometry, p.candidates, p.graph, assigned, config)
            rows.append({'model': name, 'trip_id': folder.name, **route['summary']})
        print('Five-model route continuity:', folder.name, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.results / 'classifier_route_per_trip.csv', index=False)
    aggregate = []
    for name, part in frame.groupby('model'):
        aggregate.append({'model': name, 'continuity': float(part.feasible_pairs.sum() / part.comparable_pairs.sum()), 'feasible_pairs': int(part.feasible_pairs.sum()), 'comparable_pairs': int(part.comparable_pairs.sum())})
    pd.DataFrame(aggregate).to_csv(args.results / 'classifier_continuity_comparison.csv', index=False)


def tuning(args):
    base = json.loads(args.decoder.read_text())
    predictor = Predictor(args.bundle, base)
    train = sorted(p for p in (args.data / 'train').iterdir() if p.is_dir())
    # Seeded, declared development budget; no student-test/evaluator inputs select settings.
    pool = []
    for folder in train:
        with (folder / f'{folder.name}.track').open() as f:
            count = sum(bool(s.strip()) for s in f)
        if count <= 1500:
            pool.append(folder)
    indices = np.random.default_rng(42).choice(len(pool), size=min(8, len(pool)), replace=False)
    selected = sorted([pool[i] for i in indices])
    trials = []
    for beta in [15., 30., 60.]:
        for weight in [1., 2.]:
            trials.append({'transition_beta_m': beta, 'emission_weight': weight, 'trips': []})
    scorer = Scorer(args.data / 'train')
    start = time.perf_counter()
    for folder in selected:
        print('Decoder development trip:', folder.name, flush=True)
        p = predictor.predict(folder)
        route = scorer.load_reference(folder.name, len(p.trip.arcs))
        refs, eligible, status = scorer.labels(p, predictor.bundle, route)
        lattice = prepare_lattice(p.trip, p.geometry, p.candidates, p.graph, base)
        for trial in trials:
            cfg = {**base, 'transition_beta_m': trial['transition_beta_m'], 'emission_weight': trial['emission_weight']}
            decoded = viterbi(p.trip, p.candidates, p.scores, lattice, cfg)
            reconstructed = reconstruct(p.trip, p.geometry, p.candidates, p.graph, decoded, cfg)
            trial['trips'].append({'trip_id': folder.name, 'eligible': int(eligible.sum()), 'top1_accuracy': float((decoded.assignments[eligible] == refs[eligible]).mean()), 'continuity': reconstructed['summary']['continuity'], 'label_status': status})
    for trial in trials:
        trial['objective'] = float(np.mean([r['top1_accuracy'] for r in trial['trips']]))
        trial['continuity'] = float(np.mean([r['continuity'] for r in trial['trips']]))
    best = max(trials, key=lambda t: (t['objective'], t['continuity']))
    final = {**base, 'transition_beta_m': best['transition_beta_m'], 'emission_weight': best['emission_weight']}
    args.decoder.write_text(json.dumps(final, indent=2) + '\n')
    summary = {'method': 'six fixed decoder configurations; classifier frozen', 'objective': 'mean pointwise agreement on eligible provisional labels of eight seeded short training trips', 'bias_notice': 'Classifier trained on these trips; this is optimistic decoder development, not an unbiased generalisation estimate.', 'training_trip_ids': [p.name for p in selected], 'student_test_used': False, 'trials': trials, 'selected': final, 'duration_s': time.perf_counter() - start}
    (args.results / 'decoder_search.json').write_text(json.dumps(summary, indent=2))
    print('Saved decoder settings:', final, flush=True)


def evaluate(args, make_maps=True):
    from pathsnap.maps import create_map
    config = json.loads(args.decoder.read_text())
    predictor = Predictor(args.bundle, config)
    scorer = Scorer(args.data / 'student_test')
    rows = []
    for folder in sorted((args.data / 'student_test').iterdir()):
        if not folder.is_dir():
            continue
        p = predictor.predict(folder)
        metrics = scorer.score(p, predictor.bundle)
        row = {'trip_id': folder.name, 'observations': len(p.trip.track), 'eligible_observations': metrics['eligible_observations'], 'excluded_observations': metrics['excluded_observations'], 'candidate_miss_rate': metrics['candidate_miss_rate'], 'label_status': metrics['status'], **p.timings}
        for name, reconstructed in [('classifier', p.classifier_route), ('hmm', p.decoded_route)]:
            row.update({f'{name}_{key}': value for key, value in metrics[name].items()})
            row.update({f'{name}_{key}': value for key, value in reconstructed['summary'].items()})
        row['hmm_resets'] = len(p.decoded.resets) - 1
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.results / 'decoder_per_trip.csv', index=False)
        destination = args.results / 'predictions'
        destination.mkdir(exist_ok=True)
        pd.DataFrame({'observation_id': np.arange(len(p.trip.track)), 'classifier_arc_id': p.classifier.assignments, 'hmm_arc_id': p.decoded.assignments, 'decoder_break_before': p.decoded.break_before}).to_csv(destination / f'{folder.name}.csv', index=False)
        if make_maps:
            create_map(p, args.results / 'maps' / f'{folder.name}.html', scorer.load_reference(folder.name, len(p.trip.arcs)))
        print(f"Evaluated {folder.name}: classifier={metrics['classifier']['top1_accuracy']:.4f}, HMM={metrics['hmm']['top1_accuracy']:.4f}, prediction={p.timings['total_prediction_s']:.2f}s", flush=True)
    frame = pd.DataFrame(rows)
    summary = {'trips': len(frame), 'observations': int(frame.observations.sum()), 'eligible_observations': int(frame.eligible_observations.sum()), 'excluded_observations': int(frame.excluded_observations.sum()), 'label_status': 'provisional_alignment', 'decoder': config, 'model_version': predictor.model_version, 'model_name': predictor.bundle['model_name'], 'classifier': {}, 'hmm': {}}
    for name in ['classifier', 'hmm']:
        summary[name] = {
            'top1_accuracy': float(np.average(frame[f'{name}_top1_accuracy'], weights=frame.eligible_observations)),
            'arc_macro_f1_trip_mean': float(frame[f'{name}_arc_macro_f1'].mean()),
            'sequence_similarity_trip_mean': float(frame[f'{name}_sequence_similarity'].mean()),
            'continuity': float(frame[f'{name}_feasible_pairs'].sum() / frame[f'{name}_comparable_pairs'].sum()),
            'unmatched_observations': int(frame[f'{name}_unmatched_observations'].sum()),
        }
    summary['prediction_seconds'] = {'min': float(frame.total_prediction_s.min()), 'median': float(frame.total_prediction_s.median()), 'max': float(frame.total_prediction_s.max()), 'total': float(frame.total_prediction_s.sum())}
    summary['trips_pointwise_improved'] = int((frame.hmm_top1_accuracy > frame.classifier_top1_accuracy).sum())
    summary['trips_pointwise_worse'] = int((frame.hmm_top1_accuracy < frame.classifier_top1_accuracy).sum())
    (args.results / 'decoder_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['eda', 'classifier-routes', 'tune-decoder', 'evaluate', 'report', 'all'])
    p.add_argument('--data', type=Path, default=ROOT / 'Student folder' / 'Student folder')
    p.add_argument('--bundle', type=Path, default=ROOT / 'artifacts/member2/final_model.joblib')
    p.add_argument('--decoder', type=Path, default=ROOT / 'configs/decoder.json')
    p.add_argument('--results', type=Path, default=ROOT / 'results/full')
    args = p.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    if args.stage in ['eda', 'all']:
        from pathsnap.reporting import eda
        eda(args)
    if args.stage in ['classifier-routes', 'all']:
        classifier_routes(args)
    if args.stage in ['tune-decoder', 'all']:
        tuning(args)
    if args.stage in ['evaluate', 'all']:
        evaluate(args)
    if args.stage in ['report', 'all']:
        from pathsnap.reporting import report
        report(args)


if __name__ == '__main__':
    main()
