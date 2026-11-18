"""Compare LightGBM with the saved HGB benchmark using the same training protocol."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import lightgbm
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap_ml.bayesian import SPACES, decode, propose
from pathsnap_ml.dataset import fingerprint, training_matrix
from pathsnap_ml.experiments import context, environment, fit, write_json
from pathsnap_ml.features import FEATURE_NAMES
from pathsnap_ml.inference import load_bundle
from pathsnap_ml.metrics import aggregate, evaluate_trip
from run_project import evaluate


BASELINE = {
    'learning_rate': 0.08,
    'max_leaf_nodes': 15,
    'min_samples_leaf': 30,
    'l2_regularization': 1.0,
    'max_iter': 120,
}
NATIVE_NAMES = {
    'max_leaf_nodes': 'num_leaves',
    'min_samples_leaf': 'min_child_samples',
    'l2_regularization': 'reg_lambda',
    'max_iter': 'n_estimators',
}


def model_parameters(settings, config):
    return {
        **{NATIVE_NAMES.get(key, key): value for key, value in settings.items()},
        'boosting_type': 'gbdt', 'objective': 'binary', 'max_bin': 255,
        'random_state': config['seed'], 'n_jobs': config['threads'],
        'deterministic': True, 'force_col_wise': True, 'verbosity': -1,
    }


def train(settings, config, x, y):
    model = make_pipeline(
        FunctionTransformer(pd.DataFrame, kw_args={'columns': FEATURE_NAMES}),
        lightgbm.LGBMClassifier(**model_parameters(settings, config)),
    )
    seconds, warnings = fit(model, x, y, config['threads'])
    return model, seconds, warnings


def score(model, trips, config, maximum=None):
    with threadpool_limits(limits=config['threads']):
        return [evaluate_trip(model, trip, maximum, config['training']['candidate_threshold']) for trip in trips]


def validate(settings, config, folds):
    rows = []
    for fold, x, y, validation in folds:
        model, seconds, warnings = train(settings, config, x, y)
        trips = score(model, validation, config, config['training']['cv_validation_observations_per_trip'])
        rows.append({**aggregate(trips), 'fold': fold, 'training_s': seconds, 'warnings': warnings})
    return {
        'objective': float(np.mean([row['candidate_macro_f1'] for row in rows])),
        'cv_top1_mean': float(np.mean([row['top1_accuracy'] for row in rows])),
        'folds': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'Student folder' / 'Student folder')
    parser.add_argument('--benchmark', type=Path, default=ROOT / 'results/member2')
    parser.add_argument('--results', type=Path, default=ROOT / 'results/lightgbm')
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts/lightgbm')
    parser.add_argument('--hgb-bundle', type=Path, default=ROOT / 'artifacts/member2/final_model.joblib')
    parser.add_argument('--decoder', type=Path, default=ROOT / 'configs/decoder.json')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/member2.json').read_text())
    manifest, signature = context(args.benchmark, config)
    hgb = load_bundle(args.hgb_bundle)
    comparison = pd.read_csv(args.benchmark / 'model_comparison.csv')
    if hgb['run_signature'] != signature or hgb['model_name'] != 'hist_gradient_boosting':
        raise ValueError('This experiment requires the original five-model benchmark and final HGB bundle.')
    for partition, trips in manifest.items():
        for trip in trips:
            if not Path(trip['path']).is_file() or fingerprint(args.data / partition / trip['trip_id'], config) != trip['fingerprint']:
                raise ValueError(f"Stale or missing cache for {partition}/{trip['trip_id']}. Run Member 2 preparation.")

    # Load the persisted folds exactly as originally evaluated.
    original_folds = json.loads((args.benchmark / 'folds.json').read_text())
    by_id = {trip['trip_id']: trip for trip in manifest['train']}
    folds = []
    for fold in original_folds:
        training = [by_id[tid] for tid in fold['train_trip_ids']]
        validation = [by_id[tid] for tid in fold['validation_trip_ids']]
        x, y, _ = training_matrix(training, config['training']['max_observations_per_trip'])
        folds.append((fold['fold'], x, y, validation))
    write_json(args.results / 'folds.json', original_folds)

    # Reuse the HGB search bounds on the five equivalent LightGBM parameters.
    search_family = 'hist_gradient_boosting'
    space = SPACES[search_family]
    env = {**environment(), 'lightgbm': lightgbm.__version__}
    study_signature = hashlib.sha256(
        signature.encode() + lightgbm.__version__.encode() + Path(__file__).read_bytes()
    ).hexdigest()
    write_json(args.results / 'protocol.json', {
        'study_signature': study_signature, 'benchmark_signature': signature, 'environment': env,
        'config': config, 'search_space': [[NATIVE_NAMES.get(name, name), *bounds] for name, *bounds in space],
        'selection': 'highest mean training-fold candidate Macro F1; baseline retained on ties',
        'decoder': json.loads(args.decoder.read_text()),
    })

    baseline = {'trial': 'baseline', 'parameters': BASELINE, **validate(BASELINE, config, folds)}
    write_json(args.results / 'baseline_cv.json', baseline)
    print(f"LightGBM baseline CV F1: {baseline['objective']:.6f}", flush=True)
    history = []
    for index in range(config['bayesian']['trials']):
        unit, acquisition = propose(history, search_family, config['bayesian'], np.random.default_rng(config['seed'] + index))
        settings = decode(unit, space)
        trial = {
            'trial': index + 1, 'status': 'complete', 'unit_parameters': unit.tolist(),
            'parameters': settings, 'lightgbm_parameters': model_parameters(settings, config),
            **acquisition, **validate(settings, config, folds),
        }
        history.append(trial)
        write_json(args.results / 'bayesian_trials.json', history)
        print(f"Trial {trial['trial']:02d} | {trial['strategy']} | CV F1 {trial['objective']:.6f}", flush=True)

    selected = max([baseline, *history], key=lambda record: record['objective'])
    x, y, keys = training_matrix(manifest['train'], config['training']['max_observations_per_trip'])
    model, seconds, warnings = train(selected['parameters'], config, x, y)
    metadata = {
        'model_name': 'lightgbm', 'parameters': model_parameters(selected['parameters'], config),
        'feature_names': FEATURE_NAMES, 'config': config, 'run_signature': study_signature,
        'training_trip_ids': list(by_id), 'training_s': seconds, 'warnings': warnings,
        'training_candidate_rows': len(y), 'training_observations': len(np.unique(keys, axis=0)),
        'label_status': hgb['label_status'], 'probability_status': hgb['probability_status'],
        'score_mapping': hgb['score_mapping'], 'environment': env,
    }
    args.artifacts.mkdir(parents=True, exist_ok=True)
    bundle = args.artifacts / 'lightgbm.joblib'
    joblib.dump({'model': model, **metadata}, bundle, compress=3)
    write_json(bundle.with_suffix('.metadata.json'), {
        **metadata, 'artifact_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
    })

    student = manifest['student_test']
    student_rows = score(model, student, config)
    pd.DataFrame(student_rows).to_csv(args.results / 'student_classifier_per_trip.csv', index=False)
    metrics = {
        'model': 'lightgbm', 'cv_macro_f1_mean': selected['objective'],
        'cv_macro_f1_std': float(np.std([row['candidate_macro_f1'] for row in selected['folds']])),
        'cv_top1_mean': selected['cv_top1_mean'],
        'cv_training_s': sum(row['training_s'] for row in selected['folds']),
        'final_training_s': seconds, 'training_candidate_rows': len(y),
        'training_observations': metadata['training_observations'],
        **{'student_' + key: value for key, value in aggregate(student_rows).items()},
    }
    write_json(args.results / 'selected.json', {'selected': selected, 'comparison': metrics, 'metadata': metadata})
    pd.concat([comparison, pd.DataFrame([metrics])], ignore_index=True).to_csv(
        args.results / 'model_comparison.csv', index=False,
    )

    # The current decoder is frozen for both models, including its HGB-developed settings.
    routes = {}
    for name, artifact in [('hgb', args.hgb_bundle), ('lightgbm', bundle)]:
        output = args.results / name
        output.mkdir(exist_ok=True)
        evaluate(SimpleNamespace(bundle=artifact, decoder=args.decoder, data=args.data, results=output), make_maps=False)
        routes[name] = json.loads((output / 'decoder_summary.json').read_text())
    write_json(args.results / 'route_comparison.json', routes)


if __name__ == '__main__':
    main()
