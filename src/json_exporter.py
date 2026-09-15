import numpy as np
import pandas as pd
import json
import os
import joblib
from .feature_engineering import ENGINEERED_FEATURES


def export_model_to_json(refitted_models, selection_info, X_train, output_dir="docs"):
    os.makedirs(output_dir, exist_ok=True)
    selected_name = selection_info["selected_deployment_model"]
    info = refitted_models[selected_name]
    model = info["model"]
    numeric_cols = info["numeric_cols"]
    categorical_cols = info["categorical_cols"]
    inner_pipeline = model.regressor_
    preprocessor = inner_pipeline.named_steps["preprocessor"]
    regressor = inner_pipeline.named_steps["regressor"]
    num_pipeline = preprocessor.named_transformers_["num"]
    cat_pipeline = preprocessor.named_transformers_["cat"]
    imputer_num = num_pipeline.named_steps["imputer"]
    scaler = num_pipeline.named_steps["scaler"]
    imputer_cat = cat_pipeline.named_steps["imputer"]
    encoder = cat_pipeline.named_steps["encoder"]
    cat_categories = encoder.categories_
    cat_maps = {}
    for col, cats in zip(categorical_cols, cat_categories):
        cat_maps[col] = cats.tolist()
    feature_names_num = numeric_cols
    feature_names_cat = []
    for col, cats in zip(categorical_cols, cat_categories):
        for cat in cats:
            feature_names_cat.append(f"{col}_{cat}")
    all_feature_names = feature_names_num + feature_names_cat
    coefficients = regressor.coef_.tolist() if hasattr(regressor, "coef_") else None
    intercept = float(regressor.intercept_) if hasattr(regressor, "intercept_") else None

    # Holdout predictions for Actual vs Predicted scatter chart
    holdout_predictions = selection_info.get("holdout_predictions", [])
    if not holdout_predictions and "holdout_predictions_df" in selection_info:
        hdf = selection_info["holdout_predictions_df"]
        # Filter to selected model only
        selected_hdf = hdf[hdf["Model"] == selected_name]
        holdout_predictions = [
            {"Actual": float(row["Actual"]), "Predicted": float(row["Predicted"])}
            for _, row in selected_hdf.iterrows()
        ]

    # Raw dataset sample for Price vs Living Area chart
    raw_data_sample = []
    try:
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "raw", "train.csv")
        raw_df = pd.read_csv(data_path)
        sample = raw_df[["GrLivArea", "SalePrice"]].dropna()
        # Use all points if small enough, otherwise sample
        if len(sample) <= 300:
            raw_data_sample = [
                {"GrLivArea": int(row["GrLivArea"]), "SalePrice": int(row["SalePrice"])}
                for _, row in sample.iterrows()
            ]
        else:
            sampled = sample.sample(n=300, random_state=42)
            raw_data_sample = [
                {"GrLivArea": int(row["GrLivArea"]), "SalePrice": int(row["SalePrice"])}
                for _, row in sampled.iterrows()
            ]
    except Exception:
        pass

    export = {
        "model_name": selected_name,
        "target_log_transformed": True,
        "engineered_features": ENGINEERED_FEATURES,
        "feature_names": all_feature_names,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "numeric_imputer_strategy": "median",
        "numeric_imputer_values": imputer_num.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "categorical_imputer_strategy": "most_frequent",
        "categorical_imputer_values": {
            col: imputer_cat.statistics_[i] if i < len(imputer_cat.statistics_) else ""
            for i, col in enumerate(categorical_cols)
        },
        "categorical_maps": cat_maps,
        "coefficients": coefficients,
        "intercept": intercept,
        "cv_rmse": selection_info["selected_cv_rmse"],
        "holdout_rmse": selection_info["holdout_results"][selected_name]["holdout_rmse"],
        "holdout_mae": selection_info["holdout_results"][selected_name]["holdout_mae"],
        "holdout_r2": selection_info["holdout_results"][selected_name]["holdout_r2"],
        "best_predictive_model": selection_info["best_predictive_model"],
        "percentage_difference": selection_info["percentage_difference"],
        "selection_reason": selection_info["selection_reason"],
        "feature_order": {
            "engineered": ENGINEERED_FEATURES,
            "numeric_raw": numeric_cols,
            "categorical_raw": categorical_cols,
        },
        "cv_results": selection_info.get("cv_results", {}),
        "holdout_results": selection_info.get("holdout_results", {}),
        "holdout_predictions": holdout_predictions,
        "raw_data_sample": raw_data_sample,
    }
    with open(os.path.join(output_dir, "model_export.json"), "w") as f:
        json.dump(export, f, indent=2)
    joblib.dump(model, "models/trained_model.pkl")
    return export
