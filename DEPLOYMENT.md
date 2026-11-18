# Running PathSnap

## One laptop

Use Python 3.11. From the project folder in PowerShell:

```powershell
.\scripts\setup.ps1
.\scripts\start_api.ps1
```

Open [the route explorer](http://127.0.0.1:8000/ui/). The API also serves this UI for convenient local use. The first student-testing trip runs automatically; selecting another trip performs a new request. The trained bundle is included under `artifacts/member2/`; retraining is not required to try the app.

If PowerShell blocks scripts, use the equivalent Python command from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

## Two laptops

**Laptop 1 hosts the model, road data, decoder, FastAPI and isolated scorer.**

1. Copy the project, saved model and permitted dataset to Laptop 1. Keep `Student folder/Student folder/train` and `student_test` under the project, or set the data-root argument below.
2. Run setup, then start the API for LAN access:

```powershell
.\scripts\start_api.ps1 -Lan -Partition student_test
```

3. Find Laptop 1's Wi-Fi IPv4 address using `ipconfig`. Both laptops must be on a network that permits peer connections; some guest Wi-Fi networks isolate clients. If Windows requests network access for Python, use the team's permitted private-network configuration.
4. Keep the server terminal visible during the demo. Every actual inference request prints a `PATHSNAP` JSON log with request/trip/model IDs and timings. The same records are saved in `runtime/api_requests.jsonl`.

**Laptop 2 hosts the UI.** It does not need model files or reference data.

Copy `ui/` and `scripts/start_ui.ps1` while preserving their relative layout (or copy the full project). Python alone is sufficient for this static server:

```powershell
.\scripts\start_ui.ps1
```

Open [the Laptop 2 UI](http://localhost:8001). Use the top-right connection settings and enter Laptop 1's address, for example `http://192.168.1.10:8000`. `localhost` on Laptop 2 points to Laptop 2, so it is not Laptop 1's API address.

The default CORS origins are `http://localhost:8001` and `http://127.0.0.1:8001`. If the UI uses another origin, pass the exact origin(s) to Laptop 1:

```powershell
.\scripts\start_api.ps1 -Lan -AllowedOrigins 'http://localhost:8010,http://127.0.0.1:8010'
```

Choose several trips, rerun one to observe a new request ID, test map layers/replay and verify that results appear on Laptop 2 while logs appear on Laptop 1. Capture the actual receiving terminal and UI for your submission. Stop each server with Ctrl+C.

## Final evaluator mode

Only training/student-testing data was supplied in this workspace. The app does not invent ten reserved trips or score against training references in evaluator mode.

When organisers provide final inputs, use this input layout:

```text
<data-root>/
  evaluator_test/
    00000100/
      00000100.track
      00000100.nodes
      00000100.arcs
    ...
```

The eight-digit IDs are examples of the naming format, not hardcoded evaluation answers. The UI lists whichever valid IDs actually exist in this active partition. Scoring references must be configured explicitly and stay on Laptop 1 outside the public repository:

```text
<private-reference-root>/
  00000100.route
  ...
<private-pointwise-root>/
  00000100.csv
  ...
```

Each official pointwise CSV must contain integer `observation_id,arc_id`, with unique original zero-based observation IDs and valid local arc IDs. It may list a subset if only some observations have verified labels. This interface is an implementation convention; adapt it to the organiser's official scorer if their protocol differs. A nested `<root>/<trip_id>/<trip_id>.route` or `.csv` layout is also accepted.

```powershell
.\scripts\start_api.ps1 -Lan -Partition evaluator_test -DataRoot 'D:\PathSnapData' -ReferenceRoot 'D:\PrivateReferences' -PointwiseRoot 'D:\OfficialPointwise'
```

Official pointwise CSVs take precedence over derived alignment. If only ordered routes are available, pointwise results remain marked provisional. With no usable pointwise references, predictions still run and accuracy is unavailable. Sequence similarity may still be calculated when a reference route exists. Prediction functions do not open reference files; the scorer runs only after predictions are complete and returns aggregate metrics.

Do not use final evaluator outputs to retune features, model, candidate thresholds or decoder settings. The development smoke script intentionally only accepts the student-testing partition.

## API contract

| Endpoint | Purpose |
|---|---|
| `GET /health` | Loaded model/version, active partition and available trip count |
| `GET /trips` | Active partition trip IDs and observation counts |
| `POST /predict` | Full fresh inference and subsequent isolated aggregate scoring |
| `GET /docs` | Generated interactive API documentation |

Example PowerShell request:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -ContentType 'application/json' -Body '{"trip_id":"00000052"}'
```

Responses include original-order assignments, raw GPS, predicted connected route pieces, bounded display road geometry, timing stages, reset warnings and aggregate evaluation. They do not include reference arc IDs, reference geometry or pointwise correctness. `-1` denotes an unmatched observation. The API serves only `ui/` as static content; result/audit/reference directories are not HTTP downloads.

## Reproduce experiments and checks

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py
.\.venv\Scripts\python.exe scripts/smoke_api.py
.\.venv\Scripts\python.exe scripts/run_member2.py all
.\.venv\Scripts\python.exe scripts/verify_member2.py
.\.venv\Scripts\python.exe scripts/run_member2.py report
.\.venv\Scripts\python.exe scripts/run_project.py all
.\.venv\Scripts\python.exe scripts/build_report_pdf.py
```

The live smoke check needs the API running. Other experiment stages run offline. Preparation caches and compatible completed classifier experiments are reused; use a new `--results` directory for an independent rerun. Read `MEMBER2_README.md` for the declared 200-observation-per-training-trip and 400-observation-per-CV-trip budget, feature definitions, metric aggregation and label limitations.

`run_project.py all` performs EDA, decoder development on eight seeded training trips, all-ten student comparison, Folium map generation and consolidated report-source generation. A report rebuild incorporates screenshot evidence already present under `results/full/screenshots/`; scripts do not invent screenshots. Folium HTML contains allowed development references and standard CDN library links. The evaluator UI uses local assets and no external map tiles.

## Configuration and artifacts

Environment settings: `PATHSNAP_DATA_ROOT`, `PATHSNAP_PARTITION`, `PATHSNAP_MODEL`, `PATHSNAP_DECODER`, `PATHSNAP_REFERENCE_ROOT`, `PATHSNAP_POINTWISE_ROOT`, `PATHSNAP_ALLOWED_ORIGINS`, `PATHSNAP_LOG`. Restart the API after changing configuration or replacing the model. Use only your team's trusted joblib files and the matching versions in `requirements-lock.txt`.

The public submission should include source, requirements, model artifacts, configs, saved trial/fold results, plot/map code, README and one consolidated PDF. Raw data is excluded from the handoff archive; retain the dataset locally. Do not publish private evaluator references. Review AI assistance and actual experiment claims with all three teammates before submission. Repository publication and the physical two-laptop evaluation remain team actions.
