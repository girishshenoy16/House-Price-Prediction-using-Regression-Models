import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def load_train_data(path="data/raw/train.csv"):
    df = pd.read_csv(path)
    required_cols = ["SalePrice", "1stFlrSF", "2ndFlrSF", "TotalBsmtSF",
                     "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
                     "YrSold", "YearBuilt", "YearRemodAdd", "OverallQual",
                     "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
                     "WoodDeckSF"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def stratified_split(df, test_size=0.2, random_state=42):
    log_target = np.log1p(df["SalePrice"])
    bins = pd.qcut(log_target, q=5, labels=False, duplicates="drop")
    X = df.drop(columns=["SalePrice", "Id"])
    y = df["SalePrice"]
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=bins
    )
    return X_train.reset_index(drop=True), X_hold.reset_index(drop=True), \
           y_train.reset_index(drop=True), y_hold.reset_index(drop=True)


def save_processed_data(X_train, X_hold, y_train, y_hold,
                        X_train_eng=None, X_hold_eng=None,
                        output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_hold.to_csv(os.path.join(output_dir, "X_hold.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_hold.to_csv(os.path.join(output_dir, "y_hold.csv"), index=False)
    if X_train_eng is not None:
        X_train_eng.to_csv(os.path.join(output_dir, "X_train_engineered.csv"), index=False)
    if X_hold_eng is not None:
        X_hold_eng.to_csv(os.path.join(output_dir, "X_hold_engineered.csv"), index=False)
