import pandas as pd

ENGINEERED_FEATURES = [
    "TotalSF", "TotalBaths", "AgeAtSale", "RemodAgeAtSale",
    "QualAreaIndex", "TotalPorchSF"
]


def engineer_features(df):
    out = pd.DataFrame(index=df.index)
    out["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]
    out["TotalBaths"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                         + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    out["AgeAtSale"] = df["YrSold"] - df["YearBuilt"]
    out["RemodAgeAtSale"] = df["YrSold"] - df["YearRemodAdd"]
    out["QualAreaIndex"] = df["OverallQual"] * out["TotalSF"]
    out["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"]
                           + df["3SsnPorch"] + df["ScreenPorch"]
                           + df["WoodDeckSF"])
    return out
