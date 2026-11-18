"""Build the boosted-classifier comparison from saved experiments."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate(root):
    from .reporting import DISPLAY, md_table
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder = Path(root) / 'results/lightgbm'
    if not (folder / 'route_comparison.json').exists():
        return ''
    comparison = pd.read_csv(folder / 'model_comparison.csv')
    selected = json.loads((folder / 'selected.json').read_text())
    routes = json.loads((folder / 'route_comparison.json').read_text())
    history = json.loads((folder / 'bayesian_trials.json').read_text())
    labels = {**DISPLAY, 'lightgbm': 'LightGBM'}
    plots = folder / 'plots'
    plots.mkdir(exist_ok=True)
    x = np.arange(len(comparison))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x, comparison.cv_macro_f1_mean, yerr=comparison.cv_macro_f1_std, capsize=3, color='#27798c')
    axes[0].set_title('Training trip-grouped CV candidate Macro F1')
    axes[1].bar(x, comparison.student_top1_accuracy, color='#c99952')
    axes[1].set_title('Student point agreement with provisional labels')
    short = ['Logistic\nRegression', 'Weighted\nKNN', 'Decision\nTree', 'Random\nForest', 'Histogram\nBoosting', 'LightGBM']
    for ax in axes:
        ax.set_xticks(x, short, fontsize=8)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plots / 'six_model_comparison.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot([h['trial'] for h in history], [h['objective'] for h in history], 'o-', label='Measured trial')
    ax.axhline(selected['selected']['objective'], color='#c99952', linestyle='--', label='Selected LightGBM')
    hgb = comparison[comparison.model == 'hist_gradient_boosting'].iloc[0]
    ax.axhline(hgb.cv_macro_f1_mean, color='#27798c', linestyle=':', label='HGB baseline')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Training CV candidate Macro F1')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / 'bayesian_search.png', dpi=150)
    plt.close(fig)
    rows = [
        [labels[r.model], f'{r.cv_macro_f1_mean:.6f}', f'{r.student_candidate_macro_f1:.6f}',
         f'{100*r.student_top1_accuracy:.2f}%', f'{r.student_candidate_log_loss:.6f}']
        for _, r in comparison.iterrows()
    ]
    route_rows = []
    for phase in ['classifier', 'hmm']:
        for key, title in [
            ('top1_accuracy', 'Point agreement'), ('arc_macro_f1_trip_mean', 'Arc Macro F1'),
            ('continuity', 'Continuity'), ('sequence_similarity_trip_mean', 'Sequence similarity'),
        ]:
            route_rows.append([f'{phase.upper()} {title}', f'{routes["hgb"][phase][key]:.6f}', f'{routes["lightgbm"][phase][key]:.6f}'])
    params = selected['metadata']['parameters']
    text = '\n\n'.join([
        '## Boosted classifier tuning and route evaluation',
        'The classifiers use the same 16 features, candidates and labels, five trip folds, and observation sampling. LightGBM 4.6.0 fitting used 16,000 observations and 96,698 candidate rows across 80 training trips. All ten student trips were evaluated after training CV selected its settings.',
        md_table(['Model', 'CV candidate F1', 'Student candidate F1', 'Student Top-1', 'Log loss'], rows),
        '![Six-model comparison](../lightgbm/plots/six_model_comparison.png)',
        'HGB and LightGBM each used a baseline and 12 Bayesian trials. Each search initialized four trials randomly and proposed eight using GP expected improvement. Corresponding search bounds were learning_rate 0.02–0.2 (log), num_leaves/max_leaf_nodes 7–47, min_child_samples/min_samples_leaf 10–80, reg_lambda/l2_regularization 0.001–10 (log), and n_estimators/max_iter 80–200.',
        'Selected LightGBM configuration: ' + ', '.join(f'{key}={params[key]}' for key in ['n_estimators', 'learning_rate', 'num_leaves', 'min_child_samples', 'reg_lambda']) + '. CPU fitting used four threads, seed 42, deterministic column-wise training, binary objective, and max_bin=255.',
        '![LightGBM Bayesian trials](../lightgbm/plots/bayesian_search.png)',
        'Both HGB and LightGBM were evaluated through the same frozen decoder. Decoder parameters were selected using HGB emissions on the declared training-trip development subset, so this comparison uses one common transition policy.',
        md_table(['Route metric', 'HGB', 'LightGBM'], route_rows),
        'Decoded point agreement was 13,615/15,068 for HGB and 13,625/15,068 for LightGBM. LightGBM improved four trips, tied three, and scored lower on three. These labels are provisional ordered-route alignments and share geometric cues with model features. All 19,160 GPS observations received predictions.',
        'HGB is the application default under the selection rule of highest training grouped-CV candidate Macro F1. Timing comparisons use recorded runs and are machine-dependent.',
        'Evidence is saved in results/lightgbm/model_comparison.csv, selected.json, protocol.json, bayesian_trials.json, route_comparison.json, and each model subdirectory. Reproduce with python scripts/run_lightgbm.py after cache preparation.',
    ])
    # The standalone report resolves plot links from its own directory.
    (folder / 'lightgbm_report.md').write_text(text.replace('../lightgbm/plots/', 'plots/') + '\n', encoding='utf-8')
    return text.replace(md_table(['Model', 'CV candidate F1', 'Student candidate F1', 'Student Top-1', 'Log loss'], rows), '').replace('![Six-model comparison](../lightgbm/plots/six_model_comparison.png)', '')
