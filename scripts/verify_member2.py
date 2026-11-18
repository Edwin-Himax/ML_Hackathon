"""Re-run actual prediction for every student-test observation without route input."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap_ml.geometry import load_trip, Geometry
from pathsnap_ml.inference import load_bundle, predict_from_geometry, classifier_assignments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'Student folder' / 'Student folder')
    p.add_argument('--results', type=Path, default=ROOT / 'results' / 'member2')
    p.add_argument('--bundle', type=Path, default=ROOT / 'artifacts' / 'member2' / 'final_model.joblib')
    args = p.parse_args()
    bundle = load_bundle(args.bundle)
    manifest = json.loads((args.results / 'cache_manifest.json').read_text())
    dest = args.results / 'handoff_predictions'
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for cached in manifest['student_test']:
        tid = cached['trip_id']
        start = time.perf_counter()
        trip = load_trip(args.data / 'student_test' / tid)
        config = bundle['config']['candidate']
        geo = Geometry(trip, config['index_spacing_m'])
        candidates = geo.candidates(trip, config)
        with threadpool_limits(limits=bundle['config']['threads']):
            scores = predict_from_geometry(bundle, trip, geo, candidates)
        assigned = classifier_assignments(scores, len(trip.track))
        seconds = time.perf_counter() - start
        assert len(assigned) == len(trip.track)
        assert np.all((assigned >= -1) & (assigned < len(trip.arcs)))
        assert np.isfinite(scores.log_emission_potential).all()
        np.testing.assert_allclose(scores.groupby('observation_id').emission_potential.sum(), 1, atol=1e-7)
        with np.load(cached['path']) as d:
            np.testing.assert_array_equal(candidates.observation_id, d['observation_id'])
            np.testing.assert_array_equal(candidates.arc_id, d['arc_id'])
            cached_scores = bundle['model'].predict_proba(d['x'])[:, list(bundle['model'].classes_).index(1)]
            np.testing.assert_allclose(scores.compatibility_probability, cached_scores, atol=1e-8)
        scores.to_csv(dest / f'{tid}_scores.csv', index=False)
        pd.DataFrame({'trip_id': tid, 'observation_id': np.arange(len(assigned)), 'predicted_arc_id': assigned, 'status': np.where(assigned >= 0, 'matched', 'unmatched')}).to_csv(dest / f'{tid}_classifier_assignments.csv', index=False)
        rows.append({'trip_id': tid, 'observations': len(assigned), 'candidate_rows': len(scores), 'unmatched': int((assigned == -1).sum()), 'end_to_end_inference_s': seconds, 'raw_vs_cached_scores_match': True})
        print(f'Verified {tid}: {len(assigned)} observations, {seconds:.3f}s', flush=True)
    report = {'model': bundle['model_name'], 'run_signature': bundle['run_signature'], 'artifact_sha256': hashlib.sha256(args.bundle.read_bytes()).hexdigest(), 'verified_trips': len(rows), 'prediction_path_reads_reference_routes': False, 'trips': rows}
    (args.results / 'bundle_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('All ten student-testing trips verified', flush=True)


if __name__ == '__main__':
    main()
