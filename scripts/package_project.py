"""Package source, trained models, actual results and the report; exclude raw data."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = ROOT / 'output/pdf/PathSnap_Technical_Report.pdf'
    if not report.exists():
        raise SystemExit('Build the report PDF before packaging')
    log = ROOT / 'runtime/api_requests.jsonl'
    if log.exists():
        shutil.copyfile(log, ROOT / 'results/full/api_request_log.jsonl')
    files = []
    for name in ['pathsnap', 'pathsnap_ml', 'api', 'ui', 'configs', 'tests', 'scripts', 'artifacts/member2', 'results/member2', 'results/full', 'results/dataset_audit']:
        for path in (ROOT / name).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            rel = path.relative_to(ROOT)
            if rel.parts[:3] == ('results', 'member2', 'cache'):
                continue
            files.append(path)
    for name in ['README.md', 'DEPLOYMENT.md', 'MEMBER2_README.md', 'PROJECT_WORKFLOW.md', 'requirements.txt', 'requirements-member2.txt', 'requirements-lock.txt', '.gitignore']:
        files.append(ROOT / name)
    files.append(report)
    files = sorted(set(files))
    manifest = {'format': 'pathsnap-project-v1', 'contains_raw_dataset': False, 'contains_virtual_environment': False,
                'data_location_expected': 'Student folder/Student folder/{train,student_test}/<trip_id>',
                'entry_point': 'scripts/start_api.ps1', 'deployment_instructions': 'DEPLOYMENT.md',
                'files': [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    manifest_path = ROOT / 'output/PathSnap_Full_Project.manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = ROOT / 'output/PathSnap_Full_Project.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files:
            z.write(path, path.relative_to(ROOT).as_posix())
        z.write(manifest_path, 'PROJECT_MANIFEST.json')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError('Archive integrity check failed')
        required = ['api/app.py', 'ui/index.html', 'artifacts/member2/final_model.joblib', 'configs/decoder.json', 'output/pdf/PathSnap_Technical_Report.pdf']
        assert all(name in z.namelist() for name in required)
        assert not any(name.startswith(('Student folder/', '.venv/', 'results/member2/cache/')) for name in z.namelist())
    print(archive)
    print(f'{len(files) + 1} files; {archive.stat().st_size / 1024**2:.2f} MB')
    print('SHA256:', hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
