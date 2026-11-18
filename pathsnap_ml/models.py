"""The five requested estimators share the same feature order and folds."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

MODEL_NAMES = ['logistic_regression', 'distance_weighted_knn', 'decision_tree', 'random_forest', 'hist_gradient_boosting']

DEFAULTS = {
    'logistic_regression': {'C': 1.0, 'max_iter': 500},
    'distance_weighted_knn': {'n_neighbors': 15, 'weights': 'distance', 'algorithm': 'kd_tree', 'leaf_size': 40},
    'decision_tree': {'max_depth': 10, 'min_samples_leaf': 20},
    'random_forest': {'n_estimators': 100, 'max_depth': 14, 'min_samples_leaf': 10, 'max_features': 'sqrt'},
    # Disable row-based internal early stopping: external validation is trip-grouped.
    'hist_gradient_boosting': {'max_iter': 120, 'learning_rate': 0.08, 'max_leaf_nodes': 15, 'min_samples_leaf': 30, 'l2_regularization': 1.0, 'early_stopping': False},
}


def build_model(name, seed=42, threads=4, overrides=None):
    params = {**DEFAULTS[name], **(overrides or {})}
    if name == 'logistic_regression':
        estimator = LogisticRegression(random_state=seed, **params)
    elif name == 'distance_weighted_knn':
        estimator = KNeighborsClassifier(n_jobs=threads, **params)
    elif name == 'decision_tree':
        estimator = DecisionTreeClassifier(random_state=seed, **params)
    elif name == 'random_forest':
        estimator = RandomForestClassifier(random_state=seed, n_jobs=threads, **params)
    elif name == 'hist_gradient_boosting':
        estimator = HistGradientBoostingClassifier(random_state=seed, **params)
    else:
        raise ValueError(name)
    steps = []
    if name in ['logistic_regression', 'distance_weighted_knn']:
        steps.append(('scale', StandardScaler()))
    steps.append(('model', estimator))
    return Pipeline(steps)


def positive_scores(model, x, batch_size=16000):
    if len(x) == 0:
        return np.empty(0)
    column = list(model.classes_).index(1)
    return np.concatenate([model.predict_proba(x[i:i + batch_size])[:, column] for i in range(0, len(x), batch_size)])
