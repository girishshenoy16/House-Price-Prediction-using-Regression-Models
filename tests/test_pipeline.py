import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.data_loader import load_train_data, stratified_split
from src.feature_engineering import engineer_features, ENGINEERED_FEATURES
from src.preprocessing import build_model_pipeline
from src.model_trainer import get_estimators


class TestDataLoader:
    def test_load_train_data(self):
        df = load_train_data()
        assert len(df) > 0
        assert "SalePrice" in df.columns

    def test_required_columns_present(self):
        df = load_train_data()
        required = ["1stFlrSF", "2ndFlrSF", "TotalBsmtSF", "FullBath", "HalfBath",
                     "BsmtFullBath", "BsmtHalfBath", "YrSold", "YearBuilt",
                     "YearRemodAdd", "OverallQual", "OpenPorchSF", "EnclosedPorch",
                     "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_stratified_split_sizes(self):
        df = load_train_data()
        X_train, X_hold, y_train, y_hold = stratified_split(df)
        assert len(X_train) + len(X_hold) == len(df)
        assert abs(len(X_hold) / len(df) - 0.2) < 0.01

    def test_stratified_split_reproducible(self):
        df = load_train_data()
        s1 = stratified_split(df)
        s2 = stratified_split(df)
        pd.testing.assert_frame_equal(s1[0], s2[0])
        pd.testing.assert_series_equal(s1[2], s2[2])


class TestFeatureEngineering:
    def test_engineer_features_count(self):
        df = load_train_data()
        X = df.drop(columns=["SalePrice"])
        features = engineer_features(X)
        assert list(features.columns) == ENGINEERED_FEATURES

    def test_engineer_features_no_target_leakage(self):
        df = load_train_data()
        X = df.drop(columns=["SalePrice"])
        features = engineer_features(X)
        assert "SalePrice" not in features.columns
        assert len(features) == len(df)

    def test_total_sf_formula(self):
        df = load_train_data()
        X = df.drop(columns=["SalePrice"])
        features = engineer_features(X)
        expected = X["1stFlrSF"] + X["2ndFlrSF"] + X["TotalBsmtSF"]
        pd.testing.assert_series_equal(features["TotalSF"], expected, check_names=False)

    def test_total_baths_formula(self):
        df = load_train_data()
        X = df.drop(columns=["SalePrice"])
        features = engineer_features(X)
        expected = (X["FullBath"] + 0.5 * X["HalfBath"]
                    + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"])
        pd.testing.assert_series_equal(features["TotalBaths"], expected, check_names=False)


class TestPreprocessing:
    def test_pipeline_builds(self):
        df = load_train_data()
        X = df.drop(columns=["SalePrice"])
        estimator = get_estimators()["Linear Regression"]
        model, num_cols, cat_cols = build_model_pipeline(estimator, X)
        assert model is not None
        assert len(num_cols) > 0

    def test_pipeline_fits(self):
        df = load_train_data()
        X = df.drop(columns=["SalePrice"])
        y = df["SalePrice"]
        estimator = get_estimators()["Linear Regression"]
        model, _, _ = build_model_pipeline(estimator, X)
        y_log = np.log1p(y)
        model.fit(X, y_log)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.any(np.isnan(preds))
