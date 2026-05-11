import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(".."))

from team_qualities import get_team_qualities
from setup import (
    TEAM_QUALS,
    IND_VARS,
    POS_ABBREV,
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

def get_standardized_position_values(df, player_predictions, from_pos, to_competition, season = 2025):
    # Build position-specific league distributions for normalization
    # Filter to the target competition and season, then split by target position
    standardized_positions = {}

    for position, quality_scores in player_predictions.items():
        # Get position-specific data: players who transferred TO this position in the league
        df_pos = df[
            (df["to_position"] == position) &
            (df["to_competition"] == to_competition) &
            (df["to_season"] == season)
        ]

        # Also include same-position rows (from_position == to_position) for more data
        df_pos_from = df[
            (df["from_position"] == position) &
            (df["from_competition"] == to_competition) &
            (df["from_season"] == season)
        ]

        # Combine from/to views with stripped prefixes
        cols_from = [c for c in IND_VARS if c in df_pos_from.columns]
        cols_to = [c.replace("from", "to") for c in IND_VARS]
        cols_to = [c for c in cols_to if c in df_pos.columns]

        df_a = df_pos_from[cols_from].copy()
        df_a.columns = [c.replace("from_", "") for c in df_a.columns]

        df_b = df_pos[cols_to].copy()
        df_b.columns = [c.replace("to_", "") for c in df_b.columns]

        df_league_pos = pd.concat([df_a, df_b], ignore_index=True)

        if df_league_pos.empty:
            continue

        # Determine the path key for loading SHAP importance
        if position == from_pos:
            path_key = f"same_position/{from_pos}"
        else:
            path_key = f"{POS_ABBREV[from_pos]}_to_{POS_ABBREV[position]}"

        weighted_z_scores = []
        total_weight = 0.0

        for quality, prediction in quality_scores.items():
            if quality not in df_league_pos.columns:
                continue

            col = df_league_pos[quality].dropna()
            if len(col) < 2:
                continue

            mean = col.mean()
            std = col.std()
            if std == 0:
                continue

            z = (prediction - mean) / std

            weight = 1.0
            importance = load_shap_importance(path_key, quality)
            if importance is not None:
                weight = (importance.abs() > importance.abs().median()).sum()
            weighted_z_scores.append(z * weight)
            total_weight += weight

        if weighted_z_scores and total_weight > 0:
            standardized_positions[position] = sum(weighted_z_scores) / total_weight

    if standardized_positions:
        best_position = max(standardized_positions, key=standardized_positions.get)
        best_score = standardized_positions[best_position]

    return standardized_positions

    