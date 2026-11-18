"""Gaussian-process Bayesian optimisation with expected improvement.

Uses existing scipy/sklearn dependencies. Following random initial trials,
every proposal maximizes GP expected improvement over a fresh random pool.
The model objective is measured on the same fixed trip-grouped CV folds.
"""
import warnings
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

SPACES = {
    'logistic_regression': [('C', 1e-3, 100.0, 'log')],
    'distance_weighted_knn': [('n_neighbors', 3, 41, 'int'), ('p', 1, 2, 'int')],
    'decision_tree': [('max_depth', 4, 22, 'int'), ('min_samples_leaf', 2, 60, 'int')],
    'random_forest': [('max_depth', 6, 22, 'int'), ('min_samples_leaf', 2, 40, 'int'), ('max_features', 0.3, 1.0, 'float')],
    'hist_gradient_boosting': [('learning_rate', 0.02, 0.2, 'log'), ('max_leaf_nodes', 7, 47, 'int'), ('min_samples_leaf', 10, 80, 'int'), ('l2_regularization', 0.001, 10, 'log'), ('max_iter', 80, 200, 'int')],
}


def decode(unit, space):
    params = {}
    for u, (name, low, high, kind) in zip(unit, space):
        value = np.exp(np.log(low) + u * (np.log(high) - np.log(low))) if kind == 'log' else low + u * (high - low)
        params[name] = int(round(value)) if kind == 'int' else float(value)
    return params


def propose(history, model_name, config, rng):
    space = SPACES[model_name]
    good = [h for h in history if h['status'] == 'complete']
    if len(good) < config['initial_random_trials']:
        unit = rng.random(len(space))
        return unit, {'strategy': 'initial_random', 'surrogate_warnings': []}
    x = np.array([h['unit_parameters'] for h in good])
    y = np.array([h['objective'] for h in good])
    kernel = ConstantKernel(1.0, (0.01, 100)) * Matern(length_scale=np.ones(len(space)), length_scale_bounds=(0.02, 10), nu=2.5) + WhiteKernel(1e-5, (1e-8, 0.05))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42, n_restarts_optimizer=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gp.fit(x, y)
    pool = rng.random((config['acquisition_candidates'], len(space)))
    # Avoid repeated decoded configurations, especially for integer search spaces.
    used = {tuple(sorted(h['parameters'].items())) for h in history}
    pool = np.array([u for u in pool if tuple(sorted(decode(u, space).items())) not in used])
    if not len(pool):
        raise RuntimeError('No untried candidate configurations remain')
    mean, std = gp.predict(pool, return_std=True)
    improvement = mean - y.max() - 0.001
    z = improvement / np.maximum(std, 1e-12)
    acquisition = improvement * norm.cdf(z) + std * norm.pdf(z)
    acquisition[std < 1e-12] = 0
    best = int(np.argmax(acquisition))
    return pool[best], {'strategy': 'gp_expected_improvement', 'expected_improvement': float(acquisition[best]), 'predicted_objective': float(mean[best]), 'predicted_std': float(std[best]), 'surrogate_kernel': str(gp.kernel_), 'surrogate_warnings': [str(w.message) for w in caught]}
