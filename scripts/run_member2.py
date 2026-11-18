"""Member 2 CLI. Run from the project root; no third-party downloads required."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'compare', 'tune', 'report', 'all'])
    parser.add_argument('--data', type=Path, default=ROOT / 'Student folder' / 'Student folder')
    parser.add_argument('--results', type=Path, default=ROOT / 'results' / 'member2')
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts' / 'member2')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'member2.json')
    parser.add_argument('--limit', type=int, help='Smoke preparation only; full run uses all trips')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    args.results.mkdir(parents=True, exist_ok=True)
    if args.stage in ['prepare', 'all']:
        from pathsnap_ml.dataset import prepare
        prepare(args.data, args.results, config, args.limit)
    if args.stage in ['compare', 'all']:
        from pathsnap_ml.experiments import compare
        compare(args.results, args.artifacts, config)
    if args.stage in ['tune', 'all']:
        from pathsnap_ml.experiments import tune
        tune(args.results, args.artifacts, config)
    if args.stage in ['report', 'all']:
        from pathsnap_ml.reporting import generate_report
        generate_report(args.results, args.artifacts, config)


if __name__ == '__main__':
    main()
