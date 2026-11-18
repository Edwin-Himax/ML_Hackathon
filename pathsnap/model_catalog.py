"""Saved classifier bundles available for deployment and evaluation."""
from .settings import ROOT


DEFAULT_MODEL_ID = 'hist_gradient_boosting'
MODELS = {
    'logistic_regression': ('Logistic Regression', ROOT / 'artifacts/member2/logistic_regression.joblib'),
    'distance_weighted_knn': ('Distance-weighted KNN', ROOT / 'artifacts/member2/distance_weighted_knn.joblib'),
    'decision_tree': ('Decision Tree', ROOT / 'artifacts/member2/decision_tree.joblib'),
    'random_forest': ('Random Forest', ROOT / 'artifacts/member2/random_forest.joblib'),
    DEFAULT_MODEL_ID: ('Histogram Gradient Boosting', ROOT / 'artifacts/member2/final_model.joblib'),
    'lightgbm': ('LightGBM', ROOT / 'artifacts/lightgbm/lightgbm.joblib'),
}
