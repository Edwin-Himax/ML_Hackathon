"""Package source, trained models, actual results and the report; exclude raw data."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = ROOT / 'output/pdf/PathSnap_Technical_Report.pdf'
    if not report.exists():
        raise SystemExit('Build the report PDF before packaging')
    files = []
    for name in ['pathsnap', 'pathsnap_ml', 'api', 'ui', 'configs', 'tests', 'scripts', 'artifacts/member2', 'artifacts/lightgbm', 'results/member2', 'results/lightgbm', 'results/full', 'results/dataset_audit']:
        for path in (ROOT / name).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            rel = path.relative_to(ROOT)
            if rel.parts[:3] == ('results', 'member2', 'cache') or path.suffix == '.jsonl':
                continue
            files.append(path)
    for name in ['README.md', 'DEPLOYMENT.md', 'MEMBER2_README.md', 'PROJECT_WORKFLOW.md', 'requirements.txt', 'requirements-member2.txt', 'requirements-lock.txt', '.gitignore']:
        files.append(ROOT / name)
    files.append(report)
    files.append(ROOT / 'output/pdf/PathSnap_From_Scratch_Explanation.pdf')
    files = sorted(set(files))
    manifest = {'format': 'pathsnap-project-v1', 'contains_raw_dataset': False, 'contains_virtual_environment': False,
                'data_location_expected': 'Student folder/Student folder/{train,student_test}/<trip_id>',
                'evaluator_data_location_expected': 'evaluator_test/<trip_id>',
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
        required = ['api/app.py', 'ui/index.html', 'artifacts/member2/final_model.joblib', 'artifacts/lightgbm/lightgbm.joblib', 'results/lightgbm/model_comparison.csv', 'configs/decoder.json', 'output/pdf/PathSnap_Technical_Report.pdf', 'scripts/run_evaluator.py']
        if (ROOT / 'results/full/evaluator_summary.json').is_file():
            required.append('results/full/evaluator_summary.json')
        assert all(name in z.namelist() for name in required)
        assert not any(name.startswith(('Student folder/', 'evaluator_test/', 'results/evaluator_test/',
                                        '.venv/', 'runtime/', 'results/member2/cache/')) for name in z.namelist())
    print(archive)
    print(f'{len(files) + 1} files; {archive.stat().st_size / 1024**2:.2f} MB')
    print('SHA256:', hashlib.sha256(archive.read_bytes()).hexdigest())
    member_prefixes = ('pathsnap_ml/', 'artifacts/member2/', 'artifacts/lightgbm/', 'results/member2/', 'results/lightgbm/')
    member_names = {
        'MEMBER2_README.md', 'PROJECT_WORKFLOW.md', 'requirements-member2.txt', 'requirements.txt',
        'requirements-lock.txt', 'configs/member2.json', 'configs/decoder.json',
        'scripts/run_member2.py', 'scripts/run_lightgbm.py', 'scripts/run_project.py',
        'scripts/verify_member2.py', 'output/pdf/PathSnap_From_Scratch_Explanation.pdf',
    }
    member_archive = ROOT / 'output/Member2_Handoff.zip'
    with zipfile.ZipFile(member_archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith(member_prefixes) or relative in member_names or relative.startswith('pathsnap/'):
                z.write(path, relative)
    with zipfile.ZipFile(member_archive) as z:
        if z.testzip() is not None:
            raise RuntimeError('Member 2 archive integrity check failed')
    print(member_archive)


if __name__ == '__main__':
    main()
