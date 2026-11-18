"""Exercise the live HTTP API on all ten student-testing trips, saving evidence."""
from pathlib import Path
import argparse
import json
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/full/api_smoke_summary.json')
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=180, trust_env=False) as client:
        health = client.get('/health').raise_for_status().json()
        if health['partition'] != 'student_test':
            raise ValueError('Development smoke checks only run on student_test; use the evaluator UI for authorised final evaluation.')
        trips = client.get('/trips').raise_for_status().json()['trips']
        assert len(trips) == 10
        rows = []
        for trip in trips:
            start = time.perf_counter()
            response = client.post('/predict', json={'trip_id': trip['trip_id']})
            response.raise_for_status()
            body = response.json()
            assert body['fresh_inference']
            assert len(body['hmm']['assignments']) == trip['observations']
            assert len(body['classifier']['assignments']) == trip['observations']
            rows.append({'trip_id': body['trip_id'], 'request_id': body['request_id'], 'status': response.status_code,
                         'observations': body['observation_count'], 'label_status': body['evaluation']['status'],
                         'classifier_top1': body['evaluation']['classifier']['top1_accuracy'], 'hmm_top1': body['evaluation']['hmm']['top1_accuracy'],
                         'client_elapsed_s': time.perf_counter() - start, 'server_timings': body['timings']})
            print(f"HTTP {response.status_code} trip={body['trip_id']} request={body['request_id']} observations={body['observation_count']}", flush=True)
        assert len({r['request_id'] for r in rows}) == 10
        unknown_status = client.post('/predict', json={'trip_id': '99999999'}).status_code
        malformed_status = client.post('/predict', json={'trip_id': '../train'}).status_code
        assert unknown_status == 404 and malformed_status == 422
        summary = {'base_url': args.url, 'health': health, 'passed_trips': len(rows), 'observations': sum(r['observations'] for r in rows), 'unknown_trip_status': unknown_status, 'malformed_id_status': malformed_status, 'trips': rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
