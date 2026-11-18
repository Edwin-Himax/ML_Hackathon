"""Run all six frozen classifiers on the ten evaluator trips through FastAPI."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.model_catalog import MODELS


def trip_metrics(result):
    evaluation = result['evaluation']
    row = {
        'model': result['model'],
        'model_version': result['model_version'],
        'trip_id': result['trip_id'],
        'request_id': result['request_id'],
        'observations': result['observation_count'],
        'label_status': evaluation['status'],
        'eligible_observations': evaluation['eligible_observations'],
        'candidate_miss_rate': evaluation['candidate_miss_rate'],
        'prediction_s': result['timings']['total_prediction_s'],
        'request_s': result['timings']['request_s'],
    }
    for method in ('classifier', 'hmm'):
        for key in ('top1_accuracy', 'arc_macro_f1', 'sequence_similarity'):
            row[f'{method}_{key}'] = evaluation[method].get(key)
        summary = result[method]['summary']
        for key in ('feasible_pairs', 'comparable_pairs', 'unmatched_observations', 'route_length_m', 'route_pieces'):
            row[f'{method}_{key}'] = summary[key]
    return row


def aggregate(rows):
    result = {
        'model': rows[0]['model'],
        'model_version': rows[0]['model_version'],
        'trips': len(rows),
        'observations': sum(row['observations'] for row in rows),
        'eligible_observations': sum(row['eligible_observations'] for row in rows),
        'label_status_counts': {status: sum(row['label_status'] == status for row in rows)
                                for status in sorted({row['label_status'] for row in rows})},
        'sequence_scored_trips': sum(row['hmm_sequence_similarity'] is not None for row in rows),
        'median_prediction_s': median(row['prediction_s'] for row in rows),
        'median_request_s': median(row['request_s'] for row in rows),
    }
    for method in ('classifier', 'hmm'):
        point_rows = [row for row in rows if row[f'{method}_top1_accuracy'] is not None]
        eligible = sum(row['eligible_observations'] for row in point_rows)
        feasible = sum(row[f'{method}_feasible_pairs'] for row in rows)
        comparable = sum(row[f'{method}_comparable_pairs'] for row in rows)
        result[method] = {
            'top1_accuracy': sum(row[f'{method}_top1_accuracy'] * row['eligible_observations']
                                 for row in point_rows) / eligible if eligible else None,
            'continuity': feasible / comparable if comparable else None,
        }
        for key in ('arc_macro_f1', 'sequence_similarity'):
            values = [row[f'{method}_{key}'] for row in rows if row[f'{method}_{key}'] is not None]
            result[method][f'{key}_trip_mean'] = mean(values) if values else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', help='Call a running API over HTTP. By default, run the same FastAPI app in this process.')
    parser.add_argument('--data-root', type=Path, default=ROOT, help='Parent of evaluator_test for an in-process run.')
    parser.add_argument('--reference-root', type=Path, help='Ordered route reference directory for an in-process run.')
    parser.add_argument('--pointwise-root', type=Path, help='Optional official label directory for an in-process run.')
    parser.add_argument('--per-trip', type=Path, default=ROOT / 'results/evaluator_test/per_trip.csv')
    parser.add_argument('--summary', type=Path, default=ROOT / 'results/full/evaluator_summary.json')
    args = parser.parse_args()
    rows = []
    if args.url:
        connection = httpx.Client(base_url=args.url, timeout=180, trust_env=False)
    else:
        from fastapi.testclient import TestClient
        from api.app import create_app
        from pathsnap.settings import Settings

        settings = Settings(
            data_root=args.data_root,
            partition='evaluator_test',
            reference_root=args.reference_root or args.data_root / 'evaluator_test',
            pointwise_root=args.pointwise_root,
            log_path=ROOT / 'runtime/evaluator_requests.jsonl',
        )
        connection = TestClient(create_app(settings))

    with connection as client:
        health = client.get('/health').raise_for_status().json()
        available = {entry['id']: entry['version'] for entry in health['models']}
        if health['partition'] != 'evaluator_test' or set(MODELS) - available.keys():
            raise ValueError('Start the API in evaluator_test with all six saved model bundles.')
        data = client.get('/trips').raise_for_status().json()
        trips = {trip['trip_id']: trip['observations'] for trip in data['trips']}
        if data['partition'] != 'evaluator_test' or len(trips) != 10:
            raise ValueError(f'Expected ten distinct evaluator trips; received {len(trips)}.')

        for model in MODELS:
            for trip_id, observations in trips.items():
                response = client.post('/predict', json={'trip_id': trip_id, 'model': model})
                if response.status_code != 200:
                    raise RuntimeError(f'{model}/{trip_id}: HTTP {response.status_code}: {response.text}')
                result = response.json()
                if (not result['fresh_inference'] or result['trip_id'] != trip_id
                        or result['model'] != model or result['partition'] != 'evaluator_test'
                        or result['model_version'] != available[model]
                        or result['observation_count'] != observations
                        or any(len(result[method]['assignments']) != observations for method in ('classifier', 'hmm'))):
                    raise ValueError(f'Unexpected prediction response for {model}/{trip_id}.')
                row = trip_metrics(result)
                rows.append(row)
                print(f"HTTP 200 {model} trip={trip_id} request={row['request_id']} scoring={row['label_status']}", flush=True)

    if len({row['request_id'] for row in rows}) != len(rows):
        raise ValueError('Expected a distinct request ID for each model and trip.')
    summary = {
        'partition': health['partition'],
        'transport': 'http' if args.url else 'in_process',
        'trips': len(trips),
        'observations': sum(trips.values()),
        'successful_requests': len(rows),
        'default_model': health['model'],
        'decoder': health['decoder'],
        'models': [aggregate([row for row in rows if row['model'] == model]) for model in MODELS],
    }
    args.per_trip.parent.mkdir(parents=True, exist_ok=True)
    with args.per_trip.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f"Completed {len(rows)} prediction requests. Aggregate results: {args.summary}", flush=True)


if __name__ == '__main__':
    main()
