import numpy as np
import pandas as pd
import json
import os


def evaluate_on_holdout(refitted_models, X_hold, y_hold):
    holdout_results = {}
    all_preds = []
    for name, info in refitted_models.items():
        model = info["model"]
        y_pred_dollar = model.predict(X_hold)
        rmse = np.sqrt(np.mean((y_hold.values - y_pred_dollar) ** 2))
        mae = np.mean(np.abs(y_hold.values - y_pred_dollar))
        ss_res = np.sum((y_hold.values - y_pred_dollar) ** 2)
        ss_tot = np.sum((y_hold.values - y_hold.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        holdout_results[name] = {
            "holdout_rmse": rmse,
            "holdout_mae": mae,
            "holdout_r2": r2,
        }
        preds_df = pd.DataFrame({
            "Actual": y_hold.values,
            "Predicted": y_pred_dollar,
            "Residual": y_hold.values - y_pred_dollar,
            "Model": name,
        })
        all_preds.append(preds_df)
    return holdout_results, pd.concat(all_preds, ignore_index=True)


def select_deployment_model(cv_results, holdout_results, tolerance=0.05):
    best_cv_name = min(cv_results, key=lambda k: cv_results[k]["cv_rmse"])
    best_cv_rmse = cv_results[best_cv_name]["cv_rmse"]
    browser_compatible = ["Linear Regression", "Ridge Regression", "Lasso Regression"]
    candidates = {}
    for name in browser_compatible:
        if name in cv_results:
            diff = (cv_results[name]["cv_rmse"] - best_cv_rmse) / best_cv_rmse
            candidates[name] = diff
    within_tolerance = {k: v for k, v in candidates.items() if v <= tolerance}
    if best_cv_name in browser_compatible:
        selected = best_cv_name
        reason = "Best predictive model is browser-compatible"
    elif within_tolerance:
        selected = min(within_tolerance, key=lambda k: cv_results[k]["cv_rmse"])
        reason = f"Within {tolerance*100}% tolerance of best predictive model"
    else:
        selected = min(candidates, key=lambda k: cv_results[k]["cv_rmse"])
        reason = "Most accurate browser-compatible model (outside tolerance)"
    selected_cv_rmse = cv_results[selected]["cv_rmse"]
    pct_diff = (selected_cv_rmse - best_cv_rmse) / best_cv_rmse * 100
    return {
        "best_predictive_model": best_cv_name,
        "best_predictive_cv_rmse": best_cv_rmse,
        "selected_deployment_model": selected,
        "selected_cv_rmse": selected_cv_rmse,
        "percentage_difference": pct_diff,
        "selection_reason": reason,
        "cv_results": cv_results,
        "holdout_results": holdout_results,
    }


def save_outputs(holdout_df, selection_info, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    holdout_df.to_csv(os.path.join(output_dir, "holdout_predictions.csv"), index=False)
    with open(os.path.join(output_dir, "model_comparison.json"), "w") as f:
        json.dump(selection_info, f, indent=2, default=str)
