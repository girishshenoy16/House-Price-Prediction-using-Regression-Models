import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor


def build_preprocessor(X_train):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def build_model_pipeline(estimator, X_train):
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    inner_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", estimator),
    ])
    model = TransformedTargetRegressor(
        regressor=inner_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return model, numeric_cols, categorical_cols
