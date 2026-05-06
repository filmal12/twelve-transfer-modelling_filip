import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(".."))

from team_qualities import get_team_qualities
from setup import (
    TEAM_QUALS
)

TEAM_FEATURE_QUALS = TEAM_QUALS  # from setup.py

def clean_label(s):
    return str(s).replace("_", " ").title()

def clean_feature_label(s):
    return (
        str(s)
        .replace("from_z_score_", "")
        .replace("from_", "")
        .replace("to_", "Team to: ")
        .replace("_", " ")
        .strip()
        .capitalize()
    )

def add_team_features(df):
    df = df.copy()
    for qual in TEAM_FEATURE_QUALS:
        df_from = df.copy()
        df_to = df.copy()
        df_from = get_team_qualities(qual, df_from, "from_")
        df_to = get_team_qualities(qual, df_to, "to_")
        col = qual.lower()
        df[f"from_{col}"] = df_from[col]
        df[f"to_{col}"] = df_to[col]
    return df


def get_feature_names(model):
    # statsmodels OLS
    if hasattr(model, "params") and hasattr(model.params, "index"):
        names = [f for f in model.params.index if f != "Intercept"]
        if any("[" in n for n in names):
            return None
        return names
    # XGBoost
    if hasattr(model, "get_booster"):
        names = model.get_booster().feature_names
        if names:
            return list(names)
    # sklearn
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "n_features_in_"):
        return [f"f{i}" for i in range(model.n_features_in_)]
    return None


def get_model_importance(model, feature_names):
    # XGBoost gain
    if hasattr(model, "get_booster"):
        imp = model.get_booster().get_score(importance_type="gain")
        return pd.Series(imp)
    # Random Forest
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names)
    # Linear (Lasso / Ridge)
    if hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_), index=feature_names)
    # OLS
    if hasattr(model, "params"):
        s = model.params.drop("Intercept", errors="ignore")
        return s.abs()
    return None


def load_shap_importance(path_key, target):
    csv_path = f"parameters/{path_key}/{target}_top_features.csv"
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if "feature" not in df.columns or "importance" not in df.columns:
        return None
    return pd.Series(df["importance"].abs().values, index=df["feature"].values)


def predict_safe(model, df, feature_names):
    local = df.copy()
    for feat in feature_names:
        if feat not in local.columns:
            local[feat] = 0.0
    X = local[feature_names].fillna(0)
    return model.predict(X)
