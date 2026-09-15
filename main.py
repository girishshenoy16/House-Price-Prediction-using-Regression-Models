import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.data_loader import load_train_data, stratified_split, save_processed_data
from src.feature_engineering import engineer_features, ENGINEERED_FEATURES
from src.model_trainer import cross_validate_models, refit_all_models
from src.evaluator import evaluate_on_holdout, select_deployment_model, save_outputs
from src.json_exporter import export_model_to_json
from src.visualizer import (plot_correlation_heatmap, plot_model_comparison,
                             plot_actual_vs_predicted, plot_feature_importance)


def main():
    print("=" * 60)
    print("House Price Prediction Pipeline")
    print("=" * 60)

    print("\n[1/7] Loading data...")
    df = load_train_data()
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    print("\n[2/7] Splitting data 80/20...")
    X_train, X_hold, y_train, y_hold = stratified_split(df)
    print(f"  Train: {len(X_train)} rows | Holdout: {len(X_hold)} rows")

    print("\n[3/7] Engineering features...")
    X_train_eng = engineer_features(X_train)
    X_hold_eng = engineer_features(X_hold)
    print(f"  Engineered features: {ENGINEERED_FEATURES}")

    print("\n  Saving processed data to data/processed/...")
    save_processed_data(X_train, X_hold, y_train, y_hold, X_train_eng, X_hold_eng)
    print("  Saved: X_train.csv, X_hold.csv, y_train.csv, y_hold.csv, X_train_engineered.csv, X_hold_engineered.csv")

    print("\n[4/7] Cross-validating 5 models (5-fold CV)...")
    cv_results, numeric_cols, categorical_cols = cross_validate_models(X_train, y_train, cv=5)
    print("\n  CV Results (Original Dollar Scale):")
    print(f"  {'Model':<25} {'RMSE':>12} {'MAE':>12} {'R²':>8}")
    print("  " + "-" * 60)
    for name, metrics in cv_results.items():
        print(f"  {name:<25} ${metrics['cv_rmse']:>10,.0f} ${metrics['cv_mae']:>10,.0f} {metrics['cv_r2']:>8.4f}")

    print("\n[5/7] Refitting all models on 80% training set...")
    refitted = refit_all_models(X_train, y_train)

    print("\n[6/7] Evaluating on holdout set...")
    holdout_results, holdout_df = evaluate_on_holdout(refitted, X_hold, y_hold)
    print("\n  Holdout Results:")
    print(f"  {'Model':<25} {'RMSE':>12} {'MAE':>12} {'R²':>8}")
    print("  " + "-" * 60)
    for name, metrics in holdout_results.items():
        print(f"  {name:<25} ${metrics['holdout_rmse']:>10,.0f} ${metrics['holdout_mae']:>10,.0f} {metrics['holdout_r2']:>8.4f}")

    print("\n[7/7] Selecting deployment model & exporting...")
    selection_info = select_deployment_model(cv_results, holdout_results)
    selection_info["holdout_predictions_df"] = holdout_df
    print(f"\n  Best Predictive Model: {selection_info['best_predictive_model']}")
    print(f"  Selected Deployment Model: {selection_info['selected_deployment_model']}")
    print(f"  CV RMSE Difference: {selection_info['percentage_difference']:.2f}%")
    print(f"  Reason: {selection_info['selection_reason']}")

    save_outputs(holdout_df, selection_info)
    model_export = export_model_to_json(refitted, selection_info, X_train)

    X_train_eng = engineer_features(X_train)
    plot_correlation_heatmap(X_train_eng)
    plot_model_comparison(cv_results)
    best_model_name = selection_info["selected_deployment_model"]
    y_pred_dollar = refitted[best_model_name]["model"].predict(X_hold)
    plot_actual_vs_predicted(y_hold, y_pred_dollar)
    plot_feature_importance(model_export)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"  Outputs: outputs/")
    print(f"  Model Export: docs/model_export.json")
    print(f"  Images: outputs/plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
