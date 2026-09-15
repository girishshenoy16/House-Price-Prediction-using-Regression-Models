from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from .preprocessing import build_model_pipeline
import numpy as np


def get_estimators():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.001, random_state=42, max_iter=10000),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                 random_state=42, n_jobs=-1),
    }


def _rmse_dollar(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def _mae_dollar(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def _r2_dollar(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def cross_validate_models(X_train, y_train, cv=5):
    results = {}
    for name, estimator in get_estimators().items():
        model, numeric_cols, categorical_cols = build_model_pipeline(estimator, X_train)
        rmse_scorer = make_scorer(_rmse_dollar, greater_is_better=False)
        mae_scorer = make_scorer(_mae_dollar, greater_is_better=False)
        r2_scorer = make_scorer(_r2_dollar, greater_is_better=True)
        rmse_scores = -cross_val_score(model, X_train, y_train, cv=cv, scoring=rmse_scorer, n_jobs=-1)
        mae_scores = -cross_val_score(model, X_train, y_train, cv=cv, scoring=mae_scorer, n_jobs=-1)
        r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=r2_scorer, n_jobs=-1)
        results[name] = {
            "cv_rmse": rmse_scores.mean(),
            "cv_rmse_std": rmse_scores.std(),
            "cv_mae": mae_scores.mean(),
            "cv_r2": r2_scores.mean(),
        }
    return results, numeric_cols, categorical_cols


def refit_all_models(X_train, y_train):
    refitted = {}
    for name, estimator in get_estimators().items():
        model, numeric_cols, categorical_cols = build_model_pipeline(estimator, X_train)
        model.fit(X_train, y_train)
        refitted[name] = {
            "model": model,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        }
    return refitted
