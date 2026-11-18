"""Dataset EDA and consolidated report from actual saved experiments."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pathsnap_ml.geometry import load_trip, project


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 140})
    return plt


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def eda(args):
    rows, delta = [], {}
    for partition in ['train', 'student_test']:
        steps = []
        for folder in sorted((args.data / partition).iterdir()):
            if not folder.is_dir():
                continue
            trip = load_trip(folder)
            xy = project(trip.nodes, tuple(np.median(trip.track[:, :2], axis=0)))
            lengths = np.linalg.norm(xy[trip.arcs[:, 1]] - xy[trip.arcs[:, 0]], axis=1)
            dt = np.diff(trip.track[:, 2]); steps.extend(dt.tolist())
            graph = coo_matrix((np.ones(len(trip.arcs)), tuple(trip.arcs.T)), shape=(len(trip.nodes), len(trip.nodes))).tocsr()
            components, _ = connected_components(graph, directed=True, connection='weak')
            rows.append({'partition': partition, 'trip_id': folder.name, 'gps_points': len(trip.track), 'nodes': len(trip.nodes), 'arcs': len(trip.arcs), 'max_gap_s': float(dt.max()) if len(dt) else 0, 'time_gaps_gt1': int((dt > 1).sum()), 'weak_components': components, 'zero_length_arcs': int((lengths < 1e-9).sum()), 'missing_track_rows': int((~np.isfinite(trip.track)).any(axis=1).sum()), 'invalid_coordinate_rows': int(((np.abs(trip.track[:, 0]) > 180) | (np.abs(trip.track[:, 1]) > 90)).sum()), 'non_increasing_timestamps': int((dt <= 0).sum())})
        delta[partition] = np.asarray(steps)
    df = pd.DataFrame(rows)
    df.to_csv(args.results / 'dataset_inventory.csv', index=False)
    summary = {'partitions': {}, 'one_second_fraction': float(sum((a == 1).sum() for a in delta.values()) / sum(len(a) for a in delta.values()))}
    for name, part in df.groupby('partition'):
        summary['partitions'][name] = {'trips': len(part), 'observations': int(part.gps_points.sum()), 'missing_rows': int(part.missing_track_rows.sum()), 'invalid_coordinates': int(part.invalid_coordinate_rows.sum()), 'non_increasing_timestamps': int(part.non_increasing_timestamps.sum()), 'zero_length_arcs': int(part.zero_length_arcs.sum()), 'maximum_gap_s': float(part.max_gap_s.max()), 'gps_points_min': int(part.gps_points.min()), 'gps_points_median': float(part.gps_points.median()), 'gps_points_max': int(part.gps_points.max())}
    (args.results / 'data_quality_summary.json').write_text(json.dumps(summary, indent=2))
    plt = plotting(); plots = args.results / 'plots'; plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for part, color in [('train', '#27798c'), ('student_test', '#d19b4c')]:
        subset = df[df.partition == part]
        axes[0, 0].hist(subset.gps_points, bins=np.geomspace(200, 22000, 16), alpha=.7, label=part, color=color)
        axes[0, 1].hist(delta[part], bins=np.geomspace(.9, 200000, 24), alpha=.7, label=part, color=color)
        axes[1, 0].hist(subset.arcs, bins=np.geomspace(500, 350000, 17), alpha=.7, label=part, color=color)
        axes[1, 1].hist(subset.weak_components, bins=np.geomspace(.9, 500, 15), alpha=.7, label=part, color=color)
    for ax, title in zip(axes.flat, ['GPS observations per trip', 'Sampling interval in seconds', 'Directed road arcs per trip', 'Weakly connected components per trip']):
        ax.set_xscale('log'); ax.set_title(title); ax.set_ylabel('Count'); ax.legend(fontsize=8)
    axes[0, 1].set_yscale('log')
    fig.suptitle('Dataset exploration: 80 training + 10 student-testing trips')
    fig.tight_layout(); fig.savefig(plots / 'dataset_exploration.png'); plt.close(fig)
    print('EDA inventory and plot generated for', len(df), 'trips', flush=True)


def report(args):
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(args.results / 'decoder_per_trip.csv', dtype={'trip_id': str})
    summary = json.loads((args.results / 'decoder_summary.json').read_text())
    quality = json.loads((args.results / 'data_quality_summary.json').read_text())
    search = json.loads((args.results / 'decoder_search.json').read_text())
    plt = plotting(); plots = args.results / 'plots'; plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ['Classifier', '+ HMM']
    for ax, key, title in zip(axes, ['top1_accuracy', 'continuity', 'sequence_similarity_trip_mean'], ['Eligible point agreement', 'Feasible route continuity', 'Ordered route edit similarity']):
        ax.bar(labels, [summary['classifier'][key], summary['hmm'][key]], color=['#cd9b54', '#27798c'])
        ax.set_ylim(0, 1.08); ax.set_title(title, fontsize=10)
        for i, name in enumerate(['classifier', 'hmm']): ax.text(i, summary[name][key] + .025, f'{summary[name][key]*100:.2f}%', ha='center', fontsize=9)
    fig.suptitle('Classifier versus HMM / Viterbi: actual student-testing results\nPoint agreement uses eligible provisional alignment labels', fontsize=11)
    fig.tight_layout(); fig.savefig(plots / 'decoder_comparison.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(df)); ax.bar(x - .18, df.classifier_top1_accuracy, width=.36, label='Classifier', color='#cd9b54'); ax.bar(x + .18, df.hmm_top1_accuracy, width=.36, label='+ HMM', color='#27798c')
    ax.set_xticks(x, [str(int(t)) for t in df.trip_id]); ax.set_ylim(0, 1.08); ax.set_xlabel('Student-testing trip ID'); ax.set_ylabel('Eligible derived-label agreement'); ax.legend(); ax.set_title('Pointwise comparison by trip')
    fig.tight_layout(); fig.savefig(plots / 'decoder_per_trip.png'); plt.close(fig)
    corrected = harmed = 0
    changes = []
    for _, row in df.iterrows():
        labels_path = root / 'results/member2/alignment_audits' / f'student_test_{row.trip_id}.csv'
        if labels_path.exists():
            labels_df = pd.read_csv(labels_path)
            predicted = pd.read_csv(args.results / 'predictions' / f'{row.trip_id}.csv')
            merged = labels_df.merge(predicted, on='observation_id', validate='one_to_one')
            use = merged.eligible.astype(bool)
            a = merged.classifier_arc_id == merged.reference_arc_id
            b = merged.hmm_arc_id == merged.reference_arc_id
            fix, hurt = int((use & ~a & b).sum()), int((use & a & ~b).sum())
            corrected += fix; harmed += hurt
            changes.append({'trip_id': row.trip_id, 'corrected_by_hmm': fix, 'harmed_by_hmm': hurt})
    pd.DataFrame(changes).to_csv(args.results / 'decoder_changed_points.csv', index=False)
    trial_rows = [[t['transition_beta_m'], t['emission_weight'], f"{t['objective']:.4f}", f"{t['continuity']:.4f}"] for t in search['trials']]
    per_trip = [[r.trip_id, int(r.observations), int(r.eligible_observations), f'{r.classifier_top1_accuracy:.4f}', f'{r.hmm_top1_accuracy:.4f}', f'{r.hmm_continuity:.4f}', f'{r.hmm_sequence_similarity:.4f}'] for _, r in df.iterrows()]
    test_summary = json.loads((args.results / 'test_summary.json').read_text()) if (args.results / 'test_summary.json').exists() else None
    smoke = json.loads((args.results / 'api_smoke_summary.json').read_text()) if (args.results / 'api_smoke_summary.json').exists() else None
    md = [
        '# PathSnap Technical Report',
        'Distributed map matching and route reconstruction. This consolidated report documents the implemented data pipeline, five classifier experiments, Bayesian optimisation, directed-graph HMM/Viterbi reconstruction, local API and evaluator interface. It reports actual saved experiments and identifies the remaining evaluation setup checks.',
        '## 1 Project outcome and evaluation status',
        f'The complete application processed all ten supplied student-testing trips ({summary["observations"]:,} observations). On {summary["eligible_observations"]:,} eligible derived labels, classifier-only agreement was {summary["classifier"]["top1_accuracy"]:.4f} and HMM agreement was {summary["hmm"]["top1_accuracy"]:.4f}. Feasible continuity across comparable adjacent observations increased from {summary["classifier"]["continuity"]:.4f} to {summary["hmm"]["continuity"]:.4f}. All {summary["trips_pointwise_improved"]} trips improved overall point agreement.',
        '**Reference limitation:** supplied route files list traversed arcs, not one label per GPS point. Pointwise results use a provisional ordered reference alignment and exclude ambiguous/distant points. They are agreement with derived labels, not independently verified ground-truth accuracy. Model features and derived labels share geometric cues, which can inflate agreement. Official pointwise labels can be configured in the isolated scorer. No organiser-only evaluator dataset was provided or used.',
        '## 2 Dataset understanding and exploratory data analysis',
        table(['Partition', 'Trips', 'GPS observations', 'Missing / invalid coordinates', 'Largest time gap (s)'], [[name, v['trips'], f"{v['observations']:,}", f"{v['missing_rows']} / {v['invalid_coordinates']}", int(v['maximum_gap_s'])] for name, v in quality['partitions'].items()]),
        'The data is grouped by trip. Track files contain longitude, latitude and timestamp; nodes contain longitude and latitude; arcs contain directed node pairs. Original row indices identify observations, nodes and arcs. Routes contain chronological local arc IDs. IDs are never pooled as universal road classes, and trips stay in their declared partitions.',
        f'Coordinates span several geographical regions. The pipeline projects each trip into spherical azimuthal-equidistant metre coordinates around its GPS median, retaining the original longitude/latitude for display. This is an explicit spherical approximation rather than an ellipsoidal CRS; extents above ten degrees from the origin are rejected for manual projection selection. {100*quality["one_second_fraction"]:.2f}% of time steps are one second. There are no non-increasing timestamps or missing GPS rows in these supplied partitions. The largest training gap is 153,064 seconds; long gaps are handled as sequence breaks. Zero-length arcs are preserved by original ID and guarded against division by zero.',
        'The road graph preserves directed and parallel arcs, weights each by projected length, computes weak components and uses a bounded shortest-path cache. Candidate retrieval samples complete segments, queries a spatial index, then applies exact projected point-to-segment distances. This includes long-edge interiors, not just endpoints. Up to eight nearby candidates are retained; radius expands from 50 m to 100/250 m only if the smaller radius returns none. Neither candidates nor features use reference-route membership.',
        '![Dataset EDA](plots/dataset_exploration.png)',
        '## 3 System architecture',
        'Laptop 1 loads the frozen classifier, raw GPS/network, graph transition policy and decoder. The inference sequence is validation -> per-trip projection -> candidates -> shared feature pipeline -> classifier probabilities -> directed transition distances -> log-space Viterbi -> connected geometry pieces. Only after prediction finishes does a separate scorer open references and compute aggregate metrics. Laptop 2 selects a trip ID, sends a real HTTP request and displays predicted geometry, aggregate metrics, timings and route breaks. No evaluator reference geometry, arc sequence or pointwise correctness is returned.',
        '## 4 Feature engineering classifier experiments and Bayesian optimisation',
    ]
    member = (root / 'results/member2/member2_report.md').read_text(encoding='utf-8')
    member = member.replace('](plots/', '](../member2/plots/')
    member = member.replace('Route continuity is pending Member 3\'s integration and is not estimated here.', 'Route continuity and decoder comparisons are reported in Section 5.')
    continuity_path = args.results / 'classifier_continuity_comparison.csv'
    if continuity_path.exists():
        continuity = pd.read_csv(continuity_path).set_index('model')
        comparison = pd.read_csv(root / 'results/member2/model_comparison.csv')
        from pathsnap_ml.reporting import DISPLAY, FAMILY
        brief = {'logistic_regression': 'C=1; scaled', 'distance_weighted_knn': 'k=15; distance weights; scaled', 'decision_tree': 'depth=10; leaf=20', 'random_forest': '100 trees; depth=14; leaf=10', 'hist_gradient_boosting': '120 rounds; rate=.08; 15 leaves'}
        combined = table(['Model', 'Family', 'Key configuration', 'CV candidate F1', 'Student Top-1', 'Route continuity', 'Student candidate F1', 'Log loss', 'Inference s'], [[DISPLAY[r.model], FAMILY[r.model], brief[r.model], f'{r.cv_macro_f1_mean:.4f}', f'{r.student_top1_accuracy:.4f}', f'{continuity.loc[r.model, "continuity"]:.4f}', f'{r.student_candidate_macro_f1:.4f}', f'{r.student_candidate_log_loss:.4f}', f'{r.student_model_inference_s:.4f}'] for _, r in comparison.iterrows()])
        md.append(combined)
        md.append('The consolidated table combines measured classifier CV/student diagnostics with classifier-only graph continuity on all student GPS observations. Candidate F1 and observation Top-1 use the eligible-label protocol; continuity uses all comparable adjacent observations. Details of precision, recall, timings and each model follow.')
    # Keep measured ML content, but exclude now-superseded application handoff text.
    member = member[member.index('## Status and interpretation'):member.index('## Integration and limitations')]
    member = member.replace('\n## ', '\n### ')
    md.append(member)
    md += [
        '## 5 HMM and Viterbi reconstruction',
        'A hidden state is an original directed road arc and its projected candidate position for an observation. The selected classifier provides uncalibrated binary compatibility probabilities. Clipped positive scores are normalized within each observation and used as emission potentials, not claimed as calibrated generative observation likelihoods.',
        'Transition distance includes the unused part of the previous arc, a directed shortest path from its endpoint to the next arc start, and the used part of the next arc. Same-arc forward travel is handled directly. Backward projection jitter up to eight metres is treated as zero travel; larger backward moves require a legal directed cycle. Parallel roads and opposite directions remain distinct states. Bounds are min(5,000 m, 70 m/s * elapsed time + 40 m). Zero-length edges remain traversable without causing predecessor cycles.',
        'For feasible transitions, the unnormalized log potential is -abs(network_distance - GPS_displacement) / beta, minus a 1.5 penalty for directly reversing an arc. Outgoing feasible transition potentials are normalized by log-sum-exp. Viterbi uses V[t,j] = emission_weight * log_emission[t,j] + max_i(V[t-1,i] + log_transition[i,j]); backpointers reconstruct the globally best candidate chain within each uninterrupted segment.',
        'No candidates preserve an unmatched observation (-1). Gaps above 60 seconds restart the sequence. If no finite transition exists from the active predecessor states, the previous chain is finalized and a new chain begins. Route rendering adds actual shortest-path road segments between matched positions and shows disconnected pieces separately; it never draws a straight bridge across an impossible connection.',
        '### Decoder parameter development',
        'The classifier remained frozen. Six configurations varied beta in {15,30,60} metres and emission weight in {1,2}. Eight training trips with at most 1,500 observations were chosen with seed 42. Their IDs were ' + ', '.join(search['training_trip_ids']) + '. Selection maximized mean eligible pointwise agreement, with continuity as tie-breaker. This uses training trips already seen by the classifier and is an optimistic development objective, not an independent generalisation estimate. Student testing and evaluator testing were not used to choose decoder parameters.',
        table(['Beta (m)', 'Emission weight', 'Training point agreement', 'Training continuity'], trial_rows),
        'Selected decoder configuration: `' + json.dumps(search['selected'], sort_keys=True) + '`.',
        '### Classifier and decoder comparison',
        table(['Measure', 'Classifier only', 'Classifier + HMM'], [
            ['Eligible observation Top-1', f"{summary['classifier']['top1_accuracy']:.4f}", f"{summary['hmm']['top1_accuracy']:.4f}"],
            ['Trip-mean local arc Macro F1', f"{summary['classifier']['arc_macro_f1_trip_mean']:.4f}", f"{summary['hmm']['arc_macro_f1_trip_mean']:.4f}"],
            ['Feasible route continuity', f"{summary['classifier']['continuity']:.4f}", f"{summary['hmm']['continuity']:.4f}"],
            ['Trip-mean ordered edit similarity', f"{summary['classifier']['sequence_similarity_trip_mean']:.4f}", f"{summary['hmm']['sequence_similarity_trip_mean']:.4f}"],
            ['Unmatched observations', summary['classifier']['unmatched_observations'], summary['hmm']['unmatched_observations']],
        ]),
        'Pointwise scores use the same eligible subset for both methods; candidate misses count as errors. Arc Macro F1 uses reference arc classes within each trip, then averages trips; it is distinct from binary candidate Macro F1 in the classifier table. Continuity divides finite, movement-bounded transitions by all adjacent observation pairs with elapsed time at most 60 seconds; unmatched or disconnected positions fail and long gaps are reported separately. Sequence similarity is 1 - Levenshtein distance / maximum sequence length, comparing ordered arc visits and preserving nonconsecutive revisits. Consecutive duplicates are collapsed. Connected intermediate arcs enter the predicted sequence. Disconnected pieces are flattened for this declared sequence metric; their breaks remain visible and are penalized by continuity.',
        '![Decoder comparison](plots/decoder_comparison.png)',
        '![Per-trip decoder comparison](plots/decoder_per_trip.png)',
        table(['Trip', 'GPS points', 'Eligible', 'Classifier Top-1', 'HMM Top-1', 'HMM continuity', 'HMM sequence'], per_trip),
        f'Within the eligible subset, HMM corrected {corrected:,} previously wrong point assignments and changed {harmed:,} previously correct ones to incorrect assignments. Overall improvement therefore does not mean every decoding change helped. The lowest student HMM point agreement was on trip {df.loc[df.hmm_top1_accuracy.idxmin(), "trip_id"]}, at {df.hmm_top1_accuracy.min():.4f}. Ambiguous geometry, reference alignment limitations, local candidate misses and imperfect road connectivity remain failure modes. These counts use the saved provisional alignment audits.',
        '## 6 Geographical visualisation with Folium',
        'Interactive Folium HTML maps were generated from actual inference for all ten student-testing trips under results/full/maps. Layers distinguish raw GPS (blue), road geometry (gray), supplied development reference (purple), classifier (amber) and HMM (teal), with start/end markers, legend, layer controls and fitted bounds. References are permitted development comparisons only; this directory is not mounted by FastAPI. The optional OpenStreetMap layer is off by default. Folium viewer libraries use their standard CDN assets, so opening the HTML may require Internet even though vectors are embedded. The evaluator UI itself uses an offline canvas map with local assets.',
    ]
    screenshots = args.results / 'screenshots'
    for name, caption in [('folium_map.jpg', 'Actual Folium development map'), ('ui_desktop.jpg', 'Working local evaluator interface'), ('ui_mobile.jpg', 'Responsive evaluator interface'), ('api_response.jpg', 'Actual local API response')]:
        if (screenshots / name).exists():
            md.append(f'![{caption}](screenshots/{name})')
    md += [
        '## 7 API UI deployment and reproducibility',
        'FastAPI implements GET /health, GET /trips and POST /predict with an eight-digit trip ID. The model loads once at startup; every prediction request runs fresh GPS/network inference. Request IDs, model versions, timings, outcome and trip ID are logged on the receiving server and in runtime/api_requests.jsonl. Unknown IDs return 404, malformed inputs return 422 and extra prediction fields are rejected. Static serving is restricted to ui/. Reference failures leave predictions available and mark scoring unavailable; no fabricated metric replaces missing references.',
        'The UI supports all trips in the configured partition, separate API connection settings, real request loading/error states, raw GPS/classifier/HMM layers, keyboard and pointer map navigation, trajectory replay, metrics and JSON export. Returned data includes no reference routes or pointwise correctness. Missing official labels are clearly marked as provisional alignment agreement.',
        'Run scripts/start_api.ps1 on Laptop 1 (add -Lan for LAN binding); on Laptop 2 run scripts/start_ui.ps1, open localhost:8001 and set the API address to Laptop 1. Set PATHSNAP_ALLOWED_ORIGINS if using a different UI origin. Use scripts/setup.ps1 and requirements-lock.txt for dependencies. Deployment instructions and evaluator data/reference conventions are in DEPLOYMENT.md.',
        (f'Automated checks: {test_summary["tests_run"]} tests executed, {test_summary["failures"]} failures, {test_summary["errors"]} errors. The tests cover graph direction, partial distances, parallel/zero-length edges, decoder gaps/unmatched states, sequence metrics, API validation/CORS, reference separation, and optional official label CSVs.' if test_summary else 'Automated check results have not yet been recorded.'),
        (f'Live HTTP smoke check: all {smoke["passed_trips"]} student-testing trips returned successful predictions, covering {smoke["observations"]:,} observations. Saved request IDs and real timings are in api_smoke_summary.json. Unknown-trip and malformed-ID checks returned 404 and 422 respectively.' if smoke else 'The live HTTP smoke check has not yet been recorded.'),
        f'Pure prediction time across ten student trips ranged from {summary["prediction_seconds"]["min"]:.3f} to {summary["prediction_seconds"]["max"]:.3f} seconds (median {summary["prediction_seconds"]["median"]:.3f}) in this run; scoring, JSON transport and UI are excluded. These are machine-specific timings. Browser UI checks and localhost API/UI communication were performed on this machine. A physical two-laptop LAN rehearsal, receiving-laptop terminal screenshot and final reserved-trip evaluation must still be completed on the actual event setup. Reserved evaluator inputs were not supplied; no results are claimed for them.',
        '## 8 Conclusions limitations and team review',
        'Directed graph transitions and sequence decoding improved both continuity and provisional-label agreement in the measured student-development comparison. The implementation is reproducible through saved source, model/configuration bundles, trip-grouped folds, Bayesian trial logs, decoder trials, per-trip outputs and plot-generation scripts. Label validation remains the most important prerequisite for a definitive accuracy claim. Spherical projection, limited candidate counts, sampled classifier training and imperfect references are practical limitations; official scoring conventions may change the measured results.',
        'Member 1 reviews dataset interpretation, geometry, candidates and graph transitions; Member 2 reviews features, alignment, five models and Bayesian optimisation; Member 3 reviews HMM, scoring, UI/API and deployment. All members must understand and defend the complete system. AI assistance was used to prepare code, analysis and report text; disclose it as required and review every claim against actual outputs. The public GitHub submission and organiser-specific submission actions are not performed by these local build scripts.',
        '## 9 Technical references',
        '[FastAPI static files](https://fastapi.tiangolo.com/tutorial/static-files/) and [CORS](https://fastapi.tiangolo.com/tutorial/cors/); [scikit-learn grouped CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) and [Gaussian process regression](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html); [PROJ azimuthal-equidistant projection](https://proj.org/en/stable/operations/projections/aeqd.html); [Folium layer control](https://python-visualization.github.io/folium/v0.20.0/user_guide/ui_elements/layer_control.html). The directed graph search is implemented locally with a heap and bounded source cache.',
    ]
    (args.results / 'technical_report.md').write_text('\n\n'.join(md) + '\n', encoding='utf-8')
    print('Consolidated technical report and decoder plots generated', flush=True)
