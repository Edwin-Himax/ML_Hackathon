"""Measured five-model comparison, Bayesian search and reproducible exports."""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits
from . import __version__
from .features import FEATURE_NAMES
from .models import MODEL_NAMES, DEFAULTS, build_model
from .dataset import make_folds, training_matrix
from .metrics import evaluate_trip, aggregate
from .bayesian import SPACES, decode, propose


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def context(result_root, config):
    manifest = json.loads((result_root / 'cache_manifest.json').read_text())
    if len(manifest['train']) != 80 or len(manifest['student_test']) != 10:
        raise ValueError('Full experiments require 80 train and 10 student-test trips; smoke preparation is not a comparison')
    identity = {'config': config, 'data': {k: [(t['trip_id'], t['fingerprint']) for t in v] for k, v in manifest.items()}}
    h = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for name in ['models.py', 'dataset.py', 'metrics.py', 'experiments.py', 'bayesian.py', 'features.py', 'inference.py']:
        h.update((Path(__file__).parent / name).read_bytes())
    signature = h.hexdigest()
    return manifest, signature


def environment():
    return {'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__}


def fit(model, x, y, threads):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=threads):
        warnings.simplefilter('always')
        model.fit(x, y)
    return time.perf_counter() - start, [str(w.message) for w in caught]


def cross_validate(name, overrides, manifest, folds, config):
    trip_rows, fold_rows = [], []
    for fold in folds:
        train = [t for t in manifest['train'] if t['trip_id'] in fold['train_trip_ids']]
        validation = [t for t in manifest['train'] if t['trip_id'] in fold['validation_trip_ids']]
        x, y, _ = training_matrix(train, config['training']['max_observations_per_trip'])
        model = build_model(name, config['seed'], config['threads'], overrides)
        seconds, caught = fit(model, x, y, config['threads'])
        with threadpool_limits(limits=config['threads']):
            rows = [evaluate_trip(model, t, config['training']['cv_validation_observations_per_trip'], config['training']['candidate_threshold']) for t in validation]
        metrics = aggregate(rows)
        metrics.update({'fold': fold['fold'], 'training_s': seconds, 'training_candidate_rows': len(y), 'warnings': caught})
        fold_rows.append(metrics)
        for row in rows:
            row.update({'model': name, 'fold': fold['fold']})
        trip_rows.extend(rows)
        print(f"  {name} fold {fold['fold'] + 1}: F1={metrics['candidate_macro_f1']:.4f}, Top1={metrics['top1_accuracy']:.4f}, fit={seconds:.1f}s", flush=True)
    return fold_rows, trip_rows


def export_model(name, overrides, manifest, config, signature, artifacts, filename):
    x, y, keys = training_matrix(manifest['train'], config['training']['max_observations_per_trip'])
    model = build_model(name, config['seed'], config['threads'], overrides)
    seconds, caught = fit(model, x, y, config['threads'])
    metadata = {
        'package_version': __version__, 'model_name': name,
        'parameters': {**DEFAULTS[name], **(overrides or {})},
        'feature_names': FEATURE_NAMES, 'config': config, 'run_signature': signature,
        'training_trip_ids': [t['trip_id'] for t in manifest['train']],
        'training_candidate_rows': len(y), 'training_observations': len(np.unique(keys, axis=0)),
        'label_status': 'provisional_ordered_reference_alignment_pending_team_validation',
        'probability_status': 'uncalibrated_binary_compatibility',
        'score_mapping': 'clip(p,1e-9,1) then normalize within observation; potentials, not calibrated HMM likelihoods',
        'training_s': seconds, 'warnings': caught, 'environment': environment(),
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, **metadata}, artifacts / filename, compress=3)
    metadata['artifact_sha256'] = hashlib.sha256((artifacts / filename).read_bytes()).hexdigest()
    write_json(artifacts / filename.replace('.joblib', '.metadata.json'), metadata)
    with threadpool_limits(limits=config['threads']):
        student_rows = [evaluate_trip(model, t, config['training']['student_test_observations_per_trip'], config['training']['candidate_threshold']) for t in manifest['student_test']]
    return metadata, student_rows


def compare(result_root, artifacts, config):
    result_root, artifacts = Path(result_root), Path(artifacts)
    manifest, signature = context(result_root, config)
    folds = make_folds(manifest['train'], config['folds'], config['seed'])
    write_json(result_root / 'folds.json', folds)
    write_json(result_root / 'run_metadata.json', {'signature': signature, 'config': config, 'environment': environment(), 'label_status': 'provisional; not independently verified observation truth', 'selection_rule': 'highest mean grouped-CV trip-macro candidate F1; no student-test results used for selection'})
    comparisons, all_cv, all_student = [], [], []
    for name in MODEL_NAMES:
        record = result_root / 'models' / f'{name}.json'
        if record.exists() and json.loads(record.read_text()).get('signature') == signature and (artifacts / f'{name}.joblib').exists():
            saved = json.loads(record.read_text())
            print('Reusing completed model:', name, flush=True)
        else:
            print('Training model:', name, flush=True)
            start = time.perf_counter()
            fold_rows, trip_rows = cross_validate(name, {}, manifest, folds, config)
            metadata, student_rows = export_model(name, {}, manifest, config, signature, artifacts, f'{name}.joblib')
            summary = aggregate(student_rows)
            comparison = {'model': name, 'cv_macro_f1_mean': float(np.mean([r['candidate_macro_f1'] for r in fold_rows])), 'cv_macro_f1_std': float(np.std([r['candidate_macro_f1'] for r in fold_rows])), 'cv_top1_mean': float(np.mean([r['top1_accuracy'] for r in fold_rows])), 'cv_training_s': sum(r['training_s'] for r in fold_rows), 'final_training_s': metadata['training_s'], 'training_candidate_rows': metadata['training_candidate_rows'], 'training_observations': metadata['training_observations'], 'total_experiment_s': time.perf_counter() - start}
            comparison.update({'student_' + k: v for k, v in summary.items()})
            saved = {'signature': signature, 'comparison': comparison, 'cv_folds': fold_rows, 'cv_trips': trip_rows, 'student_trips': student_rows, 'metadata': metadata}
            write_json(record, saved)
        comparisons.append(saved['comparison'])
        all_cv.extend(saved['cv_trips'])
        for r in saved['student_trips']:
            all_student.append({**r, 'model': name})
        pd.DataFrame(comparisons).to_csv(result_root / 'model_comparison.csv', index=False)
        pd.DataFrame(all_cv).to_csv(result_root / 'cv_per_trip.csv', index=False)
        pd.DataFrame(all_student).to_csv(result_root / 'student_test_per_trip.csv', index=False)
    selected = max(comparisons, key=lambda r: r['cv_macro_f1_mean'])
    write_json(result_root / 'selected_model.json', {'signature': signature, 'model': selected['model'], 'selection_metric': 'mean grouped-CV trip-macro candidate F1', 'value': selected['cv_macro_f1_mean'], 'label_status': 'provisional derived labels; validate before final submission'})
    print('Selected by training CV:', selected['model'], flush=True)


def tune(result_root, artifacts, config):
    result_root, artifacts = Path(result_root), Path(artifacts)
    manifest, signature = context(result_root, config)
    selection = json.loads((result_root / 'selected_model.json').read_text())
    if selection['signature'] != signature:
        raise ValueError('Comparison is stale; rerun compare with the current code/config')
    name = selection['model']
    folds = json.loads((result_root / 'folds.json').read_text())
    search = {'model': name, 'signature': signature, 'space': SPACES[name], 'settings': config['bayesian'], 'folds': folds, 'sampling': config['training'], 'objective': 'mean of fold trip-macro candidate Macro F1; maximize', 'student_test_used_in_objective': False}
    write_json(result_root / 'bayesian_search.json', search)
    history_path = result_root / 'bayesian_trials.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    if history and any(h['signature'] != signature for h in history):
        raise ValueError('Existing Bayesian trials belong to a different run; select another results directory')
    for trial in range(len(history), config['bayesian']['trials']):
        rng = np.random.default_rng(config['seed'] + trial)
        unit, acquisition = propose(history, name, config['bayesian'], rng)
        params = decode(unit, SPACES[name])
        print(f'Bayesian trial {trial + 1}/{config["bayesian"]["trials"]}: {params}', flush=True)
        start = time.perf_counter()
        record = {'trial': trial, 'signature': signature, 'model': name, 'parameters': params, 'unit_parameters': unit.tolist(), **acquisition}
        try:
            fold_rows, _ = cross_validate(name, params, manifest, folds, config)
            record.update({'status': 'complete', 'objective': float(np.mean([r['candidate_macro_f1'] for r in fold_rows])), 'folds': fold_rows})
        except Exception as exc:
            record.update({'status': 'failed', 'error': repr(exc)})
        record['duration_s'] = time.perf_counter() - start
        history.append(record)
        write_json(history_path, history)
        pd.DataFrame([{k: v for k, v in h.items() if k not in ['folds', 'surrogate_warnings']} for h in history]).to_csv(result_root / 'bayesian_trials.csv', index=False)
    complete = [h for h in history if h['status'] == 'complete']
    if not complete or not any(h['strategy'] == 'gp_expected_improvement' for h in complete):
        raise RuntimeError('Bayesian search requires successful GP-proposed trials, not just random initialization')
    best = max(complete, key=lambda h: h['objective'])
    metadata, rows = export_model(name, best['parameters'], manifest, config, signature, artifacts, 'tuned_model.joblib')
    pd.DataFrame(rows).to_csv(result_root / 'tuned_student_test_per_trip.csv', index=False)
    summary = {'signature': signature, 'model': name, 'trial': best['trial'], 'parameters': best['parameters'], 'cv_macro_f1_mean': best['objective'], 'cv_macro_f1_std': float(np.std([f['candidate_macro_f1'] for f in best['folds']])), 'student': aggregate(rows), 'training_s': metadata['training_s'], 'total_search_s': sum(h['duration_s'] for h in history), 'successful_trials': len(complete), 'gp_proposed_trials': sum(h['strategy'] == 'gp_expected_improvement' for h in complete), 'metadata': metadata}
    baseline = json.loads((result_root / 'models' / f'{name}.json').read_text())
    # Do not automatically deploy a tuned model if its training-CV objective regresses.
    final_name = 'tuned_model.joblib' if best['objective'] > baseline['comparison']['cv_macro_f1_mean'] else f'{name}.joblib'
    final = joblib.load(artifacts / final_name)
    joblib.dump(final, artifacts / 'final_model.joblib', compress=3)
    write_json(artifacts / 'final_model.metadata.json', {k: v for k, v in final.items() if k != 'model'})
    write_json(result_root / 'tuning_summary.json', {**summary, 'final_artifact_source': final_name, 'final_selection_rule': 'higher training-CV objective, baseline retained on a tie; student testing not used'})
    print('Final bundle:', artifacts / 'final_model.joblib', flush=True)
