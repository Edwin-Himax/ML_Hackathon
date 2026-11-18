# Member 2 implementation

The current ownership is defined in `PROJECT_WORKFLOW.md`, based on the team's new three-member distribution. This package implements **features, provisional reference alignment, six-model experiments, grouped CV, Bayesian optimisation, model export and the ML report sections**. It does not implement Member 3's route decoder or application.

## Run

From the project root, with Python and the packages in `requirements-member2.txt`:

```powershell
python -m unittest discover -s tests -v
python scripts/run_member2.py prepare
python scripts/run_member2.py compare
python scripts/run_member2.py tune
python scripts/verify_member2.py
python scripts/run_member2.py report
```

`python scripts/run_member2.py all` runs preparation, comparison, tuning and reporting. Run the verification script afterwards. Install dependencies into your own environment using `python -m pip install -r requirements-member2.txt` if needed. The exact environment used for experiments is saved in `results/member2/run_metadata.json`.

The CLI accepts `--data`, `--results`, `--artifacts` and `--config`. A preparation smoke run supports `--limit 1 --results results/member2_smoke`; it cannot be used as the required five-model experiment. Results directories with an incompatible completed Bayesian study must not be reused; select a new directory to preserve history.

## Important status

The dataset has ordered reference arcs, not direct GPS-point labels. `alignment.py` creates a **provisional ordered alignment**, excluding distant, ambiguous, locally inconsistent and reference-break observations. This is a development implementation awaiting Member 1's validation or an official alignment. It uses reference geometry and heading, which overlap the prediction feature family, so agreement can overstate real-world accuracy. Metrics must be described as agreement with eligible derived labels, not independently verified ground-truth accuracy.

The pipeline makes predictions for all observations, including those excluded from reference-based metrics. Candidate misses remain errors in eligible-observation Top-1 accuracy. They are not inserted into candidate sets or silently removed from validation. Training excludes observation groups with no positive candidate.

## Experiment protocol

- All 80 training trips participate in the five fixed trip-grouped CV folds. A trip occurs in exactly one held-out fold.
- To keep all five algorithms comparable on a laptop, fit uses up to **200 eligible, candidate-covered observations per training trip**, selected at evenly spaced positions. All generated candidates for each selected observation are retained. This is deterministic observation subsampling, not training on every GPS row.
- CV evaluates up to **400 eligible observations per validation trip**, including candidate misses. Student testing evaluates **every eligible observation across all ten trips**.
- The same folds, candidate rules, feature definitions and sampling policy apply to every model and tuning trial. Fitted scaling exists inside the Logistic Regression and KNN pipelines and is learned separately in each fold.
- Candidate binary Macro F1, precision, recall and binary log loss are computed per trip and averaged. Observation Top-1 and candidate miss rate use pooled observation counts. The selection objective is the mean of the five fold trip-macro candidate F1 scores.
- Histogram Gradient Boosting's internal row-based early stopping is disabled. No student-testing data fits the classifier, scaler or Bayesian objective.
- Model selection uses training CV. The ten student-testing trips provide the required development comparison, and are not an untouched final test set.
- The Bayesian optimiser fits a Matérn Gaussian-process surrogate to measured CV objectives and proposes new configurations using expected improvement after random initialization. Every actual trial, parameter, duration, fold result and surrogate warning is saved.
- An untuned baseline is retained if the best tuning trial fails to improve its CV objective. The result is reported honestly even when tuning does not help.
- Classifier times exclude candidate generation and HMM decoding; end-to-end inference timings are measured separately by the verification script.
- No route-continuity or HMM quality claim is made here. Those metrics depend on Member 3's integrated decoder.

## Handoff to Member 1

`geometry.py` is a temporary raw-file adapter so Member 2 can run before the shared geospatial pipeline arrives. Replace it with your reviewed projection/candidate implementation through the following contract; a change requires rebuilding caches and rerunning comparisons.

Trip input:

- `trip_id`: string; local arc IDs are meaningful only within this trip.
- `track`: `N x 3`, longitude, latitude, timestamp; original row order retained.
- `nodes`: longitude/latitude array.
- `arcs`: original directed integer endpoint pairs.

Candidate DataFrame, contiguous by observation:

| Column | Meaning |
|---|---|
| observation_id | Original zero-based GPS row |
| arc_id | Original local arc row |
| distance_m | Point-to-segment distance in the trip projection |
| projection_fraction | Clipped segment fraction in [0, 1] |
| distance_rank | Zero-based nearest-candidate rank |
| search_radius_m | Radius used for that observation |

Geometry provides `gps_xy`, `vectors`, `lengths`, `in_degree`, `out_degree` indexed by original IDs. Features are generated by `make_features(trip, geometry, candidates)`. The temporary index uses sampled edge points plus exact segment filtering; it handles long-edge interiors and preserves reverse-direction arcs. The projection is a per-trip **spherical azimuthal-equidistant approximation**, not an ellipsoidal CRS, and rejects points more than ten angular degrees from its origin.

Review `results/member2/alignment_audits/` together. Each row includes observation ID, reference arc, route occurrence, distance, confidence diagnostics and exclusion flags. Focus on loops, stationary segments, directed reverse arcs and reference breaks. Do not approve labels based only on a high model score. Obtain the organiser's pointwise alignment/scoring convention if available. If verified per-observation labels become available, replace `align_reference` and invalidate all derived results.

## Handoff to Member 3

Use the final model bundle only with the matching package and environment. Only load trusted joblib files from your team.

```python
from pathsnap_ml.inference import (
    load_bundle, predict_from_geometry, classifier_assignments,
)

bundle = load_bundle("artifacts/member2/final_model.joblib")
# trip, geometry and candidates come from Member 1; no reference route input.
scores = predict_from_geometry(bundle, trip, geometry, candidates)
raw_assignments = classifier_assignments(scores, len(trip.track))
```

The score table preserves `trip_id`, `observation_id` and `arc_id`, plus:

- `compatibility_probability`: binary candidate probability, uncalibrated.
- `emission_potential`: positive clipped score divided by the within-observation sum.
- `log_emission_potential`: log of that normalized potential, ready for your documented hybrid decoder.

These are not calibrated generative HMM likelihoods or verified route confidence. Never reorder scores without their keys. Use `-1` for unmatched observations and show a visible status. The baseline helper does not connect road paths or run Viterbi.

`verify_member2.py` runs the frozen model from raw GPS/nodes/arcs across the ten student-testing trips without opening `.route` in the prediction path. It exports all-observation score tables and classifier assignments in `results/member2/handoff_predictions/`. These are development checks, not hardcoded outputs for the application. `/predict` must compute actual scores on each request.

Do not include the offline alignment module, reference audits or references in evaluator API responses. Apply graph transitions from Member 1, then Viterbi. Compare against these same classifier assignments and declare your eligible-observation denominator, gap behavior and sequence metric.

## Main outputs

- `results/member2/model_comparison.csv`: all five untuned models.
- `results/member2/cv_per_trip.csv`, `student_test_per_trip.csv`: reproducible per-trip diagnostics.
- `results/member2/bayesian_search.json`, `bayesian_trials.json`, `bayesian_trials.csv`: saved search specification and actual history.
- `results/member2/tuning_summary.json`: tuned/untuned selection and search cost.
- `results/member2/member2_report.md`: your evidence-based feature/model/tuning report sections.
- `results/member2/plots/`: real-result figures for the team report.
- `artifacts/member2/final_model.joblib` and `.metadata.json`: model, scaler when applicable, feature order, candidate/label policy, versions and provenance.

Raw data and preparation caches are ignored by Git. The current trained models are small enough for normal Git inclusion; the full-project task removed their former ignore rule and includes them in the handoff archive. Repository publication remains a team submission action.

## Classifier results

| Model | Grouped-CV candidate F1 | Student candidate F1 | Student point agreement |
|---|---:|---:|---:|
| Logistic Regression | 0.887401 | 0.904692 | 85.13% |
| Distance-weighted KNN | 0.891221 | 0.911373 | 84.53% |
| Decision Tree | 0.895788 | 0.912632 | 85.11% |
| Random Forest | 0.901464 | 0.917506 | 85.36% |
| Histogram Gradient Boosting | 0.905646 | 0.920494 | 85.91% |
| LightGBM | 0.905305 | 0.920337 | 85.96% |

## Boosted classifier comparison

LightGBM 4.6.0 used the shared feature cache, five saved training folds, sampling, and evaluation code. The experiment compared one starting configuration and 12 Bayesian trials on training CV. The winning LightGBM trial used 91 trees, a learning rate of 0.118846, 25 leaves, a minimum of 70 samples per leaf, and L2 regularization of 0.615846. The final fit used 96,698 candidate rows from 16,000 observations and all 80 training trips.

| Metric | Current HGB | LightGBM |
| --- | ---: | ---: |
| Training grouped-CV candidate Macro F1 | 0.905646 | 0.905305 |
| Student candidate Macro F1 | 0.920494 | 0.920337 |
| Student classifier point agreement | 85.91% | 85.96% |
| Student HMM point agreement | 90.36% | 90.42% |
| Student HMM route continuity | 99.75% | 99.75% |
| Student HMM route sequence similarity | 0.756493 | 0.753936 |

Both bundles ran through the same frozen decoder on the ten supplied student trips. Point agreement uses 15,068 provisional labels from 19,160 GPS observations. LightGBM agreed with 10 more points after decoding. The default remains HGB because it scored higher on the training CV selection metric.

With the original benchmark and its prepared caches available, reproduce the comparison from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install lightgbm==4.6.0
.\.venv\Scripts\python.exe scripts/run_lightgbm.py
```

The six-model table, trial log, and side-by-side route summary are under `results/lightgbm/`. The LightGBM bundle is `artifacts/lightgbm/lightgbm.joblib`. The baseline model records are stored under `results/member2/`.

## Reserved evaluator results

All six frozen classifiers ran across the ten reserved trips. These are classifier-only scores.
Point agreement uses 14,831 of 19,674 GPS observations per model and is provisional because the supplied ordered routes had to be aligned to individual observations.
The default remains Histogram Gradient Boosting based on training cross-validation.

| Classifier | Provisional point agreement | Mean arc Macro F1 |
|---|---:|---:|
| Logistic Regression | 77.34% | 86.14% |
| Distance-weighted KNN | 77.63% | 86.33% |
| Decision Tree | 77.14% | 86.35% |
| Random Forest | 77.99% | 86.52% |
| Histogram Gradient Boosting | 77.86% | 87.03% |
| LightGBM | 78.09% | 87.85% |
