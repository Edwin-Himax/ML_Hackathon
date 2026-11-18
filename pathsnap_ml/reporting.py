"""Generate Member 2 report sections and figures from saved actual results."""
from pathlib import Path
import json
import platform
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from .models import MODEL_NAMES, DEFAULTS, positive_scores
from .features import FEATURE_NAMES
from .dataset import training_matrix

DISPLAY = {'logistic_regression': 'Logistic Regression', 'distance_weighted_knn': 'Distance-weighted KNN', 'decision_tree': 'Decision Tree', 'random_forest': 'Random Forest', 'hist_gradient_boosting': 'Histogram Gradient Boosting'}
FAMILY = {'logistic_regression': 'Linear probabilistic', 'distance_weighted_knn': 'Instance based', 'decision_tree': 'Tree', 'random_forest': 'Bagged tree ensemble', 'hist_gradient_boosting': 'Boosted tree ensemble'}
OBJECTIVE = {'logistic_regression': 'L2-regularised binary logistic loss.', 'distance_weighted_knn': 'No iterative training loss; inverse-distance neighbour voting in scaled feature space.', 'decision_tree': 'Greedy Gini impurity reduction with depth and leaf-size constraints.', 'random_forest': 'Bootstrap-trained trees with Gini splits, averaging class probabilities.', 'hist_gradient_boosting': 'Stagewise minimisation of binary log loss, with L2 regularisation and a fixed iteration budget.'}
RATIONALE = {
    'logistic_regression': 'A small probabilistic baseline tests whether geometric relationships are adequately linear. Standardisation makes the numerical feature scales comparable. Its linear boundary can miss interactions between heading reliability, distance and road structure.',
    'distance_weighted_knn': 'Neighbour voting tests whether locally similar candidate configurations share compatibility labels. Standardisation is necessary for meaningful distance. Prediction must search retained training examples, and correlated geometric features can distort neighbour distance.',
    'decision_tree': 'A bounded tree can learn readable nonlinear rules over distance, heading and candidate context. A single tree is sensitive to the training sample and can produce abrupt probability changes; the depth/leaf constraints limit memorisation.',
    'random_forest': 'Averaged bootstrapped trees reduce the instability of a single tree while modelling feature interactions. More trees increase storage and prediction cost; their averaged probabilities are not automatically calibrated.',
    'hist_gradient_boosting': 'Histogram-based boosting combines nonlinear feature rules and handles continuous geometric inputs without scaling. Its iteration count, learning rate and leaf settings require validation. Internal row-split early stopping is disabled to preserve the external trip-group protocol.'
}


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def generate_report(result_root, artifacts, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result_root, artifacts = Path(result_root), Path(artifacts)
    comparison = pd.read_csv(result_root / 'model_comparison.csv').set_index('model').loc[MODEL_NAMES]
    lightgbm_root = Path(__file__).resolve().parents[1] / 'results/lightgbm'
    has_lightgbm = (lightgbm_root / 'selected.json').exists()
    parameters = dict(DEFAULTS)
    if has_lightgbm:
        comparison = pd.read_csv(lightgbm_root / 'model_comparison.csv').set_index('model')
        parameters['lightgbm'] = json.loads((lightgbm_root / 'selected.json').read_text())['metadata']['parameters']
        DISPLAY['lightgbm'] = 'LightGBM'
        FAMILY['lightgbm'] = 'Boosted tree ensemble'
        OBJECTIVE['lightgbm'] = 'Binary log loss with leaf-wise gradient boosting and L2 regularisation.'
        RATIONALE['lightgbm'] = 'LightGBM models nonlinear interactions among distance, heading and road context with histogram-based, leaf-wise boosted trees. Leaf count, minimum leaf size and regularisation constrain complexity. Its candidate compatibility scores supply emissions to the shared sequence decoder.'
    names = list(comparison.index)
    selected = json.loads((result_root / 'selected_model.json').read_text())
    tuning = json.loads((result_root / 'tuning_summary.json').read_text())
    metadata = json.loads((result_root / 'run_metadata.json').read_text())
    preparation = pd.read_csv(result_root / 'preparation_summary.csv', dtype={'trip_id': str})
    manifest = json.loads((result_root / 'cache_manifest.json').read_text())
    plots = result_root / 'plots'
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 140})
    labels = ['Logistic\nRegression', 'Weighted\nKNN', 'Decision\nTree', 'Random\nForest', 'Histogram\nBoosting']
    if has_lightgbm:
        labels.append('LightGBM')
    xs = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(xs, comparison.cv_macro_f1_mean, yerr=comparison.cv_macro_f1_std, color='#27798c', capsize=4)
    ax.set_xticks(xs, labels); ax.set_ylim(0, 1); ax.set_ylabel('Candidate Macro F1'); ax.set_title('Classifier comparison on trip-grouped cross-validation\nMean +/- fold standard deviation; provisional labels')
    fig.tight_layout(); fig.savefig(plots / 'cv_macro_f1.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(xs - .18, comparison.student_candidate_macro_f1, width=.36, label='Candidate Macro F1', color='#27798c')
    ax.bar(xs + .18, comparison.student_top1_accuracy, width=.36, label='Observation Top-1', color='#efb759')
    ax.set_xticks(xs, labels); ax.set_ylim(0, 1); ax.legend(); ax.set_title('Student-testing agreement with eligible derived labels')
    fig.tight_layout(); fig.savefig(plots / 'student_test_comparison.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(xs, comparison.student_candidate_log_loss, color='#27798c'); axes[0].set_title('Trip-mean binary candidate log loss'); axes[0].set_ylabel('Lower is better')
    axes[1].bar(xs, comparison.student_model_inference_s, color='#a25c3e'); axes[1].set_title('Total student-test classifier time'); axes[1].set_ylabel('Seconds; features / HMM excluded')
    for ax in axes: ax.set_xticks(xs, labels)
    fig.tight_layout(); fig.savefig(plots / 'log_loss_and_inference.png'); plt.close(fig)
    history = json.loads((result_root / 'bayesian_trials.json').read_text())
    good = [h for h in history if h['status'] == 'complete']
    fig, ax = plt.subplots(figsize=(9, 4.5))
    numbers = [h['trial'] + 1 for h in good]; values = [h['objective'] for h in good]
    ax.plot(numbers, values, 'o-', label='Measured trial CV objective', color='#27798c')
    ax.plot(numbers, np.maximum.accumulate(values), '--', label='Best trial so far', color='#a25c3e')
    ax.axhline(selected['value'], linestyle=':', color='black', label='Untuned baseline')
    ax.set_xlabel('Actual trial number'); ax.set_ylabel('Candidate Macro F1'); ax.set_title('Bayesian search on training trip folds'); ax.legend()
    fig.tight_layout(); fig.savefig(plots / 'bayesian_search.png'); plt.close(fig)
    student = pd.read_csv(result_root / 'student_test_per_trip.csv')
    if has_lightgbm:
        lightgbm_student = pd.read_csv(lightgbm_root / 'student_classifier_per_trip.csv')
        lightgbm_student['model'] = 'lightgbm'
        student = pd.concat([student, lightgbm_student], ignore_index=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, name in zip(axes.flat, names):
        rr = student[student.model == name]
        matrix = np.array([[rr.tn.sum(), rr.fp.sum()], [rr.fn.sum(), rr.tp.sum()]], dtype=int)
        ax.imshow(matrix, cmap='Blues')
        for i in range(2):
            for j in range(2): ax.text(j, i, f'{matrix[i,j]:,}', ha='center', va='center', color='white' if matrix[i,j] > matrix.max() / 2 else 'black')
        ax.set_title(DISPLAY[name], fontsize=9); ax.set_xticks([0, 1], ['Nonmatch', 'Match'], fontsize=8); ax.set_yticks([0, 1], ['Nonmatch', 'Match'], fontsize=8); ax.set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Derived label'); fig.suptitle('Student-testing candidate confusion matrices (threshold 0.5)'); fig.tight_layout(); fig.savefig(plots / 'confusion_matrices.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(xs - .18, comparison.cv_training_s, width=.36, label='Five CV fits combined', color='#27798c')
    ax.bar(xs + .18, comparison.final_training_s, width=.36, label='Final evaluated fit', color='#efb759')
    ax.set_xticks(xs, labels); ax.set_ylabel('Seconds'); ax.set_title('Measured estimator fitting time'); ax.legend()
    fig.tight_layout(); fig.savefig(plots / 'training_times.png'); plt.close(fig)

    # Explicitly sampled resubstitution diagnostic: not a validation score.
    diagnostic_path = result_root / 'training_diagnostics.json'
    diagnostics = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {}
    if diagnostics.get('signature') != selected['signature'] or any(name not in diagnostics.get('models', {}) for name in names):
        x, y, _ = training_matrix(manifest['train'], config['training']['max_observations_per_trip'])
        index = np.unique(np.linspace(0, len(y) - 1, min(10000, len(y))).astype(int))
        diagnostics = {'signature': selected['signature'], 'candidate_rows_sampled': len(index), 'metric': 'pooled binary Macro F1 on evenly spaced rows of the exact fitting matrix; resubstitution, not held-out', 'models': {}}
        for name in names:
            bundle_path = artifacts.parent / 'lightgbm/lightgbm.joblib' if name == 'lightgbm' else artifacts / f'{name}.joblib'
            bundle = joblib.load(bundle_path)
            with threadpool_limits(limits=config['threads']):
                probability = positive_scores(bundle['model'], x[index])
            diagnostics['models'][name] = float(f1_score(y[index], probability >= .5, labels=[0, 1], average='macro', zero_division=0))
        diagnostic_path.write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')

    train_prep = preparation[preparation.partition == 'train']
    test_prep = preparation[preparation.partition == 'student_test']
    baseline = comparison.loc[tuning['model']]
    tuned = tuning['student']
    rows = []
    for name, r in comparison.iterrows():
        rows.append([DISPLAY[name], FAMILY[name], f'{r.cv_macro_f1_mean:.4f} +/- {r.cv_macro_f1_std:.4f}', f'{r.student_candidate_macro_precision:.4f}', f'{r.student_candidate_macro_recall:.4f}', f'{r.student_candidate_macro_f1:.4f}', f'{r.student_top1_accuracy:.4f}', f'{r.student_candidate_log_loss:.4f}', f'{r.student_model_inference_s:.4f}'])
    sections = [
        '# Member 2 feature engineering model experiments and optimisation',
        'These sections document the implementation and actual runs performed in this workspace. They are ready for team review and adaptation into the consolidated report; they do not cover Member 3\'s decoder, route metrics or deployed application.',
        '## Status and interpretation',
        '**All reference-based results below measure agreement with eligible provisional labels derived from the ordered supplied routes. They are not independently verified GPS-point accuracy.** No official pointwise mapping was supplied during this work. The same distance/heading geometry influences derived labels and input features, which can inflate measured agreement. Member 1 must review alignment quality; replace labels and rerun if an official mapping becomes available. Excluding ambiguous or distant observations also biases these results toward the more confidently aligned subset.',
        f'All 80 training trips and ten student-testing trips were processed. Training alignment accepted {int(train_prep.eligible_aligned_observations.sum()):,} of {int(train_prep.observations.sum()):,} observations; student-testing alignment accepted {int(test_prep.eligible_aligned_observations.sum()):,} of {int(test_prep.observations.sum()):,}. Excluded points remain in inference outputs. Student-testing candidate miss rate on the eligible subset is {100*baseline.student_candidate_miss_rate:.3f}%. Candidate policy was fixed before model comparison; no reference arc was added to validation candidates.',
        '## Feature engineering',
        'We model each observation-candidate pair as a binary compatibility problem. Arc IDs and trip IDs remain metadata. This supports road networks whose IDs are local to each trip. Features use GPS observations and road geometry only; no reference-route membership or reference-derived feature enters prediction.',
        ', '.join(f'`{n}`' for n in FEATURE_NAMES) + '.',
        'Distances and fractions describe spatial fit. Directed heading difference disambiguates opposing arcs; a separate validity flag suppresses unreliable heading during low movement, anomalous speed or long gaps. Length and endpoint degrees describe road context. Speed and time gap describe sampling/motion conditions. Rank, relative distance, candidate count and search radius describe local competition. Zero-length arcs are flagged and projected without division by zero. The first point uses no future observation for its motion features.',
        'The development geospatial adapter uses per-trip spherical azimuthal-equidistant coordinates and a sampled-segment spatial index with exact planar filtering. Candidate radius starts at 50 m and expands to 100/250 m only when no candidate exists; at most eight directed arcs are retained. This is a temporary Member 1 adapter pending shared geometry review, not an ellipsoidal projection or a guarantee of reference coverage.',
        '## Reference label preparation',
        'The offline aligner considers up to eight route occurrences per observation, scores geometric/heading compatibility and backtracks an ordered path through reference progress. Repeated arc occurrences are preserved. Large GPS gaps and infeasible progress cause explicit resets; reference discontinuities are kept as separate chunks. Eligibility requires a reference distance at most 50 m, an emission margin at least 0.15, local consistency and distance from reference-break endpoints. Post-initial resets are excluded. These are fixed provisional heuristics, not validated confidence probabilities. Per-point decisions are saved in `alignment_audits/` for manual review.',
        '## Training and evaluation protocol',
        f'The five fixed group folds each hold out 16 complete training trips and fit on the remaining 64. Up to {config["training"]["max_observations_per_trip"]} eligible, candidate-covered observations per training trip are selected deterministically; all their candidates are retained. Each final untuned model fits {int(baseline.training_observations):,} observations and {int(baseline.training_candidate_rows):,} candidate rows drawn from all 80 trips. This is a laptop-budget experiment, not fitting every available GPS observation. CV evaluates up to {config["training"]["cv_validation_observations_per_trip"]} eligible observations per validation trip; student testing evaluates all eligible observations. Candidate misses remain observation-level errors. There is no observation weighting or class resampling after this group sampling.',
        'StandardScaler is fit only inside each training fold for Logistic Regression and KNN. The tree estimators use the same unscaled feature matrix. Histogram Gradient Boosting internal early stopping is disabled. Candidate binary metrics use labels [nonmatch, match] and threshold 0.5; candidate metrics are averaged by trip. Top-1 accuracy is pooled across eligible observations. Folds and all sampling rules are identical across models. Model selection and the Bayesian objective use only training CV; student testing is the permitted development comparison.',
        '## Six-model comparison',
        md_table(['Model', 'Family', 'CV candidate F1', 'Student precision', 'Student recall', 'Student F1', 'Student Top-1', 'Binary log loss', 'Classifier seconds'], rows),
        'CV values are fold means with standard deviation. Precision, recall and F1 are binary candidate Macro metrics; they are not arc-class Macro F1. Classifier seconds cover the ten student-testing trips\' eligible candidate matrices, excluding raw-file loading, candidates, features and decoding. Log loss is binary candidate loss, not normalized observation loss. Route continuity is pending Member 3\'s integration and is not estimated here.',
        '![Cross-validation comparison](plots/cv_macro_f1.png)',
        '![Student-testing comparison](plots/student_test_comparison.png)',
        '![Log loss and inference](plots/log_loss_and_inference.png)',
    ]
    for name in names:
        r = comparison.loc[name]
        sections.extend([
            '## ' + DISPLAY[name],
            RATIONALE[name],
            '**Evaluated configuration:** `' + json.dumps(parameters[name], sort_keys=True) + '`. Random seed 42 applies where supported; compute threads are limited to four.',
            '**Objective:** ' + OBJECTIVE[name],
            f'**Measured results:** CV candidate Macro F1 {r.cv_macro_f1_mean:.4f} +/- {r.cv_macro_f1_std:.4f}; student-testing precision {r.student_candidate_macro_precision:.4f}, recall {r.student_candidate_macro_recall:.4f}, candidate Macro F1 {r.student_candidate_macro_f1:.4f}, Top-1 {r.student_top1_accuracy:.4f} and binary log loss {r.student_candidate_log_loss:.4f}. Combined five-fold fitting took {r.cv_training_s:.3f} s; final fitting took {r.final_training_s:.3f} s; student classifier scoring took {r.student_model_inference_s:.4f} s on this machine.',
            f'**Fit/generalisation diagnostic:** pooled candidate Macro F1 on {diagnostics["candidate_rows_sampled"]:,} sampled rows of the exact training matrix is {diagnostics["models"][name]:.4f}. This resubstitution statistic is optimistic and has a different aggregation from the trip-macro held-out scores. The fold standard deviation is {r.cv_macro_f1_std:.4f}; inspect per-trip results rather than attributing the difference solely to overfitting. These diagnostics do not establish whether the model has exhausted its capacity or generalises to independently labelled routes.' + (' KNN queries include the fitted examples themselves in this diagnostic, so near-perfect training agreement can reflect interpolation.' if name == 'distance_weighted_knn' else ''),
        ])
    sections.extend([
        '## Model selection',
        f'{DISPLAY[selected["model"]]} achieved the highest mean training-CV candidate Macro F1 ({selected["value"]:.4f}) and was selected for optimisation using that predeclared objective. The decision did not depend on the student-testing ranking. This is selection for provisional-label compatibility, not proof of the best decoded route quality.',
        '## Bayesian hyperparameter optimisation',
        'The optimiser uses a Gaussian-process surrogate with a Matérn kernel (nu=2.5), observation noise and expected improvement. It starts with four random trials; later trials maximize expected improvement over 1,024 sampled search points. This is an actual sequential Bayesian search; every objective below was measured by refitting all five training folds.',
        md_table(['Parameter', 'Lower', 'Upper', 'Distribution'], [[n, lo, hi, kind] for n, lo, hi, kind in json.loads((result_root / 'bayesian_search.json').read_text())['space']]),
        f'{len(history)} trials were attempted, {tuning["successful_trials"]} completed, and {tuning["gp_proposed_trials"]} completed trials were GP-proposed. Total measured search wall time was {tuning["total_search_s"]:.2f} seconds. Best actual trial: {tuning["trial"] + 1}. Best settings: `{json.dumps(tuning["parameters"], sort_keys=True)}`. Search settings, proposal details, warnings, durations and each fold objective are saved in `bayesian_search.json` and `bayesian_trials.json`.',
        md_table(['Measure', 'Untuned', 'Best tuning trial'], [
            ['CV candidate Macro F1', f'{baseline.cv_macro_f1_mean:.4f}', f'{tuning["cv_macro_f1_mean"]:.4f}'],
            ['Student candidate Macro F1', f'{baseline.student_candidate_macro_f1:.4f}', f'{tuned["candidate_macro_f1"]:.4f}'],
            ['Student observation Top-1', f'{baseline.student_top1_accuracy:.4f}', f'{tuned["top1_accuracy"]:.4f}'],
            ['Student binary log loss', f'{baseline.student_candidate_log_loss:.4f}', f'{tuned["candidate_log_loss"]:.4f}'],
            ['Student classifier time (s)', f'{baseline.student_model_inference_s:.4f}', f'{tuned["model_inference_s"]:.4f}'],
        ]),
        f'The exported final bundle uses `{tuning["final_artifact_source"]}` according to the higher training-CV objective, retaining the baseline on a tie. CV changed by {tuning["cv_macro_f1_mean"]-baseline.cv_macro_f1_mean:+.4f}; student Top-1 changed by {tuned["top1_accuracy"]-baseline.student_top1_accuracy:+.4f}. These small differences are descriptive; no statistical significance claim is made.',
        '![Measured Bayesian search](plots/bayesian_search.png)',
        '## Diagnostics and figures',
        '![Candidate confusion matrices](plots/confusion_matrices.png)',
        '![Measured training times](plots/training_times.png)',
        'No epoch training/validation curves were fabricated. The selected estimators were run without a recorded epoch-wise validation history. Confusion matrices use actual final-model student candidate predictions and the fixed 0.5 threshold.',
        '## Integration and limitations',
        'The bundle includes the classifier/pipeline, ordered feature schema, configuration, training trip IDs, software versions and run signature. The inference API accepts GPS/network-derived candidate features only, preserves keys and returns binary compatibility probabilities plus candidate-normalized positive emission potentials. These potentials are not calibrated generative HMM likelihoods. Member 3 must document their use, add Member 1\'s graph transitions, and measure decoder effects.',
        'Remaining team work: validate/reference alignment with Member 1 or organisers; review geometry/projection assumptions; integrate Member 3\'s HMM; report route continuity and ordered sequence performance; run the actual two-laptop application; collect deployment/Folium evidence. The current ML outputs alone do not establish completion of those components.',
        '## Reproducibility and evidence',
        f'Run signature: `{selected["signature"]}`. Python and library versions are recorded in `run_metadata.json`; this run used scikit-learn {metadata["environment"]["scikit_learn"]}, NumPy {metadata["environment"]["numpy"]} and {config["threads"]} computational threads on {metadata["environment"]["platform"]}. Timings are machine-specific.',
        'Reproduce with the commands in `MEMBER2_README.md`. Sources of evidence: model_comparison.csv, cv_per_trip.csv, student_test_per_trip.csv, folds.json, preparation_summary.csv, per-model JSON records, Bayesian trial history, artifact metadata and alignment audits. The report text and code were prepared with AI assistance; the team must review the implementation and explain the actual results, and disclose assistance as required by event policy.',
        'Technical references used for implementation: [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html), [GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html), and [PROJ azimuthal-equidistant projection documentation](https://proj.org/en/stable/operations/projections/aeqd.html). The development spherical projection is implemented locally and does not call PROJ.',
    ])
    verification_path = result_root / 'bundle_verification.json'
    if verification_path.exists():
        verified = json.loads(verification_path.read_text())
        timings = [r['end_to_end_inference_s'] for r in verified['trips']]
        sections.extend(['## Saved bundle verification', f'The saved final bundle was loaded and run from raw GPS/nodes/arcs on all {verified["verified_trips"]} student-testing trips, covering {sum(r["observations"] for r in verified["trips"]):,} observations. Raw-input scores matched cached-feature scores. Measured per-trip end-to-end classifier pipeline latency ranged from {min(timings):.3f} to {max(timings):.3f} seconds, excluding HMM and the network/UI. The raw prediction path did not read reference routes. Full score and assignment exports are in `handoff_predictions/`; these are development diagnostics, not application lookup outputs.'])
    from .lightgbm_reporting import generate
    boosted_comparison = generate(Path(__file__).resolve().parents[1])
    if boosted_comparison:
        sections.insert(sections.index('## Integration and limitations'), boosted_comparison)
    (result_root / 'member2_report.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    print('Generated report and six measured-result figures:', result_root / 'member2_report.md', flush=True)
