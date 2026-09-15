import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_train_data, stratified_split
from src.model_trainer import refit_all_models


def python_model_predict(refitted_models, X_row, model_name):
    model = refitted_models[model_name]["model"]
    return float(model.predict(X_row)[0])


def js_equivalent_predict(model_export, raw_input):
    num_imputer = np.array(model_export["numeric_imputer_values"])
    scaler_mean = np.array(model_export["scaler_mean"])
    scaler_scale = np.array(model_export["scaler_scale"])
    cat_imputer = model_export["categorical_imputer_values"]
    cat_maps = model_export["categorical_maps"]
    num_cols = model_export["numeric_columns"]
    cat_cols = model_export["categorical_columns"]
    coefficients = np.array(model_export["coefficients"])
    intercept = model_export["intercept"]

    numeric_vals = []
    for i, col in enumerate(num_cols):
        val = raw_input.get(col, np.nan)
        if pd.isna(val):
            val = num_imputer[i]
        numeric_vals.append(float(val))
    numeric_vals = np.array(numeric_vals)
    numeric_scaled = (numeric_vals - scaler_mean) / scaler_scale

    cat_vec = []
    for col in cat_cols:
        val = raw_input.get(col, "")
        if val == "" or pd.isna(val):
            val = cat_imputer[col]
        categories = cat_maps[col]
        for cat in categories:
            cat_vec.append(1.0 if val == cat else 0.0)

    full_vector = np.concatenate([numeric_scaled, cat_vec])
    inner_log = float(np.dot(full_vector, coefficients) + intercept)
    dollar_pred = float(np.expm1(inner_log))
    return dollar_pred


def create_test_cases(df, X_hold, y_hold):
    cases = []
    sorted_idx = y_hold.sort_values().index
    n = len(sorted_idx)
    cases.append({
        "name": "lower_priced",
        "row_idx": sorted_idx[n // 4],
    })
    cases.append({
        "name": "median",
        "row_idx": sorted_idx[n // 2],
    })
    cases.append({
        "name": "higher_priced",
        "row_idx": sorted_idx[3 * n // 4],
    })

    X_train, _, y_train, _ = stratified_split(df)
    test_row = X_train.iloc[0].copy()
    test_row["LotFrontage"] = np.nan
    test_row["MasVnrArea"] = np.nan
    test_row["GarageYrBlt"] = np.nan
    cases.append({
        "name": "missing_imputed",
        "raw_data": test_row,
        "actual_price": float(y_train.iloc[0]),
    })

    cases.append({
        "name": "holdout_sample",
        "row_idx": sorted_idx[0],
    })
    return cases


def run_parity_test():
    with open("docs/model_export.json") as f:
        model_export = json.load(f)

    df = load_train_data()
    X_train, X_hold, y_train, y_hold = stratified_split(df)
    refitted = refit_all_models(X_train, y_train)
    selected_name = model_export["model_name"]

    cases = create_test_cases(df, X_hold, y_hold)
    tolerance = 0.0001
    results = []
    all_pass = True

    for case in cases:
        name = case["name"]
        if "row_idx" in case:
            idx = case["row_idx"]
            raw_data = X_hold.loc[idx].to_dict()
            X_row = X_hold.loc[idx:idx]
            actual_price = float(y_hold.loc[idx])
        else:
            raw_data = case["raw_data"].to_dict()
            X_row = case["raw_data"].to_frame().T
            actual_price = case["actual_price"]

        py_dollar = python_model_predict(refitted, X_row, selected_name)
        js_dollar = js_equivalent_predict(model_export, raw_data)

        diff = abs(py_dollar - js_dollar)
        threshold = tolerance * max(abs(py_dollar), 1.0)
        passed = diff <= threshold
        if not passed:
            all_pass = False

        results.append({
            "case": name,
            "python_pred": py_dollar,
            "js_pred": js_dollar,
            "diff": diff,
            "threshold": threshold,
            "passed": passed,
            "actual": actual_price,
        })
        status = "PASS" if passed else "FAIL"
        print("  {:<20s} Py: ${:>12,.2f}  JS: ${:>12,.2f}  Diff: ${:>8.4f}  Thresh: ${:>8.4f}  [{}]".format(
              name, py_dollar, js_dollar, diff, threshold, status))

    return all_pass, results


if __name__ == "__main__":
    print("=" * 60)
    print("Python <-> JavaScript Parity Test")
    print("=" * 60)
    passed, results = run_parity_test()
    print("\n" + "=" * 60)
    if passed:
        print("All 5 cases PASSED")
    else:
        print("Some cases FAILED")
    print("=" * 60)
    sys.exit(0 if passed else 1)
