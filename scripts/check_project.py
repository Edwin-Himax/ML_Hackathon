"""Run the meaningful unit/integration checks and persist their actual outcome."""
from pathlib import Path
import io
import json
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
start = time.perf_counter()
suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
out = ROOT / 'results' / 'full'
out.mkdir(parents=True, exist_ok=True)
(out / 'test_results.txt').write_text(stream.getvalue(), encoding='utf-8')
summary = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped), 'successful': result.wasSuccessful(), 'duration_s': time.perf_counter() - start}
(out / 'test_summary.json').write_text(json.dumps(summary, indent=2))
print(stream.getvalue())
print(json.dumps(summary))
raise SystemExit(0 if result.wasSuccessful() else 1)
