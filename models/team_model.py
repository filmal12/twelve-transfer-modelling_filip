"""
Team Success Prediction Model with XGBoost
Predicts team performance metric changes when a player transitions positions.
Uses pre-transition player attributes to predict changes in team metrics
after the positional change.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from positional_model import remove_correlated_features
import shap

import sys

sys.path.append(os.path.abspath(".."))

from setup import (
    POSITION_CATEGORIES,
    TEAM_QUALS,
    IND_VARS,
    POSITIONAL_CHANGES,
    normal_quals,
    ALL_QUALITIES
)

from data_loader import get_all_data

from team_qualities import get_team_qualities, convertToZscores

matplotlib.use("Agg")

# ========== GLOBAL SETTINGS ==========
TRAIN_MODELS = True
SAVE_MODELS = True
EVALUATE_PERFORMANCE = True
PLOT_RESULTS = True
QUALITIES = True  # If True, use team qualities instead of team metrics as targets
TEST_SIZE = 0.3  # Increased from 0.2 for better validation
RANDOM_STATE = 42
MAX_FEATURES = 35  # Limit features to prevent overfitting

team_scatter = False

def getTeamStatsChanges(only_position_changes, full_df):
    team_stat_cols = [c for c in only_position_changes.columns if c.startswith("to_team_stats_")]

    if not team_stat_cols or "to_team_id" not in full_df.columns or "to_season" not in full_df.columns:
        return only_position_changes

    # Build previous-season lookup: rename stats to prev_to_team_stats_*
    prev_lookup = (
        full_df[["to_team_id", "to_season"] + team_stat_cols]
        .drop_duplicates(subset=["to_team_id", "to_season"])
        .rename(columns={"to_season": "_prev_lookup_season"} | {col: f"prev_{col}" for col in team_stat_cols})
    )

    # Vectorized join: match each row to the same team one season prior
    df = only_position_changes.copy()
    df["_join_season"] = df["to_season"] - 1
    merged = df.merge(
        prev_lookup,
        left_on=["to_team_id", "_join_season"],
        right_on=["to_team_id", "_prev_lookup_season"],
        how="inner",
    ).drop(columns=["_join_season", "_prev_lookup_season"])

    # Compute delta columns (current season minus previous season)
    for col in team_stat_cols:
        delta_col = "delta_" + col.removeprefix("to_team_stats_")
        merged[delta_col] = merged[col] - merged[f"prev_{col}"]

    return merged



def categorize_position(position):
    for category, positions in POSITION_CATEGORIES.items():
        if position in positions:
            return category
    return None


def identify_positional_changes(df):
    # Ensure from_position and to_position columns exist
    if "from_position" not in df.columns or "to_position" not in df.columns:
        return df[df["from_position"] != df["to_position"]].copy() if all(c in df.columns for c in ["from_position", "to_position"]) else df
    
    # Filter for positional changes
    position_changes = df[df["from_position"] != df["to_position"]].copy()
    
    return position_changes


def prepare_category_model_data(df, target_metrics):    
    # Filter for transitions TO this category
    df_filtered = df.copy()
    
    if len(df_filtered) < 10:
        return None
    
    z_score_cols = []

    # Get feature columns: z-scores from BEFORE the position change
    z_score_cols = IND_VARS.copy()

    for quality_name in TEAM_QUALS:
        qual = quality_name.lower()

        z_score_cols.append(f"from_{qual}")

    if not z_score_cols:
        return None

    # One-hot encode from_position and add to features
    if "from_position" in df_filtered.columns and "to_position" in df_filtered.columns:
        pos_changes_dummies = pd.get_dummies(POSITIONAL_CHANGES.copy()).astype(float)

        # Build per-row key e.g. "Striker-Winger"
        row_pos_changes = df_filtered["from_position"] + "-" + df_filtered["to_position"]
        # Encode per row, reindex to the full POSITIONAL_CHANGES columns (fills unknown combos with 0)
        row_dummies = (
            pd.get_dummies(row_pos_changes)
            .reindex(columns=pos_changes_dummies.columns, fill_value=0)
            .astype(float)
        )
        row_dummies.index = df_filtered.index
        df_filtered = pd.concat([df_filtered, row_dummies], axis=1)


        z_score_cols = z_score_cols + pos_changes_dummies.columns.tolist()

    # Get target columns: team metric or quality changes
    if QUALITIES:
        team_metrics = [c for c in df_filtered.columns if c.startswith("from_team_stats")]

        df_to = df_filtered.copy()

        # Calculate qualities on to_ team stats
        for quality_name in target_metrics:
            df_to = get_team_qualities(quality_name, df_to, "from_")

        
        df_filtered = convertToZscores(team_metrics, df_filtered)

        # Compute previous-season qualities if prev_to_team_stats_* columns exist
        prev_team_stat_cols = [c for c in df_filtered.columns if c.startswith("prev_to_team_stats_")]
        to_stat_cols = [c for c in df_filtered.columns if c.startswith("to_team_stats_")]
        has_prev = bool(prev_team_stat_cols)
        if has_prev:
            # Drop current-season to_team_stats_* so there are no duplicate columns when
            # prev_to_team_stats_* is renamed to to_team_stats_* for quality calculation.
            df_prev = (
                df_filtered
                .drop(columns=to_stat_cols)
                .rename(columns={c: c.removeprefix("prev_") for c in prev_team_stat_cols})
            )
            for quality_name in target_metrics:
                df_prev = get_team_qualities(quality_name, df_prev, "to_")

        # Create delta columns for qualities
        delta_cols = []
        for quality_name in target_metrics:
            quality_col_lower = quality_name.lower()
            if has_prev:
                delta_col = f"delta_{quality_col_lower}"
                df_filtered[delta_col] = df_to[quality_col_lower] - df_prev[quality_col_lower]
            else:
                delta_col = f"to_{quality_col_lower}"
                df_filtered[delta_col] = df_to[quality_col_lower]
            delta_cols.append(delta_col)
    else:
        # Use team metrics as targets
        # target_metrics = TEAM_METRICS[target_category]
        
        # Calculate metric deltas (after - before)
        delta_cols = []
        for metric in target_metrics:
            from_col = f"from_{metric}"
            to_col = f"to_{metric}"
            
            if from_col in df_filtered.columns and to_col in df_filtered.columns:
                delta_col = f"delta_{metric}"
                df_filtered[delta_col] = df_filtered[to_col] - df_filtered[from_col]
                delta_cols.append(delta_col)
    
    if not delta_cols:
        return None
    
    # Feature selection: reduce number of features to prevent overfitting
    if len(z_score_cols) > MAX_FEATURES:
        # Calculate variance for each feature in filtered data
        feature_variance = df_filtered[z_score_cols].var().sort_values(ascending=False)
        z_score_cols = feature_variance.head(MAX_FEATURES).index.tolist()
    
    # z_score_cols = remove_correlated_features(df_filtered, z_score_cols)    

    all_data_cols = z_score_cols + delta_cols
    df_clean = df_filtered[all_data_cols]

    if len(df_clean) < 10:
        return None
    
    # Separate features and targets
    X = df_clean[z_score_cols].fillna(0)
    # Create multi-target dataframe
    y = df_clean[delta_cols]

    return {
        "X": X,
        "y": y,
        "feature_cols": z_score_cols,
        "target_cols": delta_cols,
        "target_metrics": target_metrics,
        "df": df_clean,
        "df_filtered": df_filtered,
        "from_positions": df_clean.index.map(df_filtered["from_position"]),
        "to_positions": df_clean.index.map(df_filtered["to_position"])
    }


def train_xgboost_category_model(X, y, category_name, metric_names, test_size=TEST_SIZE):    
    results = {}
    
    # Train one model per metric
    for metric_col in y.columns:
        y_metric = y[metric_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_metric, test_size=test_size, random_state=RANDOM_STATE
        )

        print(f"Category: {category_name} - sample size {len(X)}")
        
        # Train XGBoost model with regularization to prevent overfitting
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        
        xgb_model.fit(
            X_train, y_train
        )

        if SAVE_MODELS:
            os.makedirs(f"../team_models", exist_ok=True)
            joblib.dump(xgb_model, f'../team_models/{metric_col}_xgboost.pkl')
        
        # Make predictions
        y_pred_train = xgb_model.predict(X_train)
        y_pred_test = xgb_model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Create SHAP plot for XGBoost
        path_prefix = "team_models"

        explainer_xgb = shap.TreeExplainer(xgb_model)
        shap_values_xgb = explainer_xgb.shap_values(X_train)
        
        feature_importance = pd.DataFrame({
            'feature': X_train.columns.tolist(),
            'importance': shap_values_xgb.mean(axis=0)
        }).sort_values('importance', ascending=False)

        os.makedirs(f'../parameters/{path_prefix}', exist_ok=True)
            
        feature_importance.to_csv(f'../parameters/{path_prefix}/{metric_col}_top_features.csv', index=False)

        np.save(f"../parameters/{path_prefix}/{metric_col}_xgboost_shap_values.npy", shap_values_xgb)

        
        results[metric_col] = {
            "model": xgb_model,
            "train_metrics": {
                "r2": train_r2,
                "mse": train_mse,
                "mae": train_mae
            },
            "test_metrics": {
                "r2": test_r2,
                "mse": test_mse,
                "mae": test_mae
            },
            "feature_importance": feature_importance,
            "shap_values": shap_values_xgb,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test
        }
    
    return results


def train_category_models(df):    
    results = {}

   
    for category in TEAM_QUALS:
        
        # Prepare data for this category
        data = prepare_category_model_data(df, [category])
        if data is None or len(data["X"]) < 10:
            continue
        
        # Train models for this category
        category_results = train_xgboost_category_model(
            data["X"],
            data["y"],
            category,
            data["target_metrics"]
        )
        
        # Save models
        if SAVE_MODELS:
            category_lower = category.lower()
            
            for metric_col, model_result in category_results.items():
                file_path = "../parameters/team_models/rsquared.csv"

                trained_res = model_result["train_metrics"]
                r2 = trained_res["r2"]

                stats_tot = pd.DataFrame(data=[[category, r2]], columns = ["Target", "R^2"])

                if os.path.isfile(file_path):
                    stats_tot.to_csv(file_path, mode='a', header=False, index=False)
                else:
                    os.makedirs(os.path.dirname("../parameters/team_models"), exist_ok=True)
                    stats_tot.to_csv(file_path, mode='w', header=True, index=False)

        results[category] = {
            "data": data,
            "models": category_results
        }
    
    return results


def plot_model_performance(model_result, title="Team Metric Delta Prediction", metric_name=""):    
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    
    # Plot 1: Train predictions vs actual
    ax = axes[0]
    ax.scatter(model_result["y_train"], model_result["y_pred_train"], alpha=0.6, s=50, color='blue')
    min_val = min(model_result["y_train"].min(), model_result["y_pred_train"].min())
    max_val = max(model_result["y_train"].max(), model_result["y_pred_train"].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel("Actual Metric Delta")
    ax.set_ylabel("Predicted Metric Delta")
    ax.set_title(f"Train Set (R² = {model_result['train_metrics']['r2']:.4f})")
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Plot 2: Training residuals
    ax = axes[1]
    train_residuals = model_result["y_train"] - model_result["y_pred_train"]
    ax.scatter(model_result["y_pred_train"], train_residuals, alpha=0.6, s=50, color='blue')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel("Predicted Metric Delta")
    ax.set_ylabel("Residuals")
    ax.set_title("Training Residuals")
    ax.grid(alpha=0.3)
    
    plt.suptitle(f"{title}\n{metric_name.replace("delta", "").replace("_", " ").strip().capitalize()}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_feature_importance(feature_importance_df, shap_values, X_train, path_prefix, y_metric, title="Top 15 Features", top_n=15):
    shap_importance = np.abs(shap_values).mean(axis=0)

    # Beeswarm plot with top 15 features by mean |SHAP value|
    top_15_idx = np.argsort(shap_importance)[-15:]
    clean_names = []
    for i in top_15_idx:
        col_name = X_train.columns[i]
        if col_name in normal_quals:
            clean_name = col_name.replace('from_', 'Team from: ').replace('to_', 'Team to: ').replace('_', ' ').strip().capitalize()
        else:
            clean_name = col_name.replace('from_', 'Player: ').replace('_', ' ').strip().capitalize()
        clean_names.append(clean_name)

    shap_explanation_top15 = shap.Explanation(
        values=shap_values[:, top_15_idx],
        data=X_train.values[:, top_15_idx],
        feature_names=clean_names
    )
    os.makedirs(f"../Figures/{path_prefix}/features", exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_explanation_top15, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f"../Figures/{path_prefix}/features/{y_metric}_xgboost_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()

    shap_df = pd.DataFrame({'feature': X_train.columns, 'importance': shap_importance})
    shap_df_top = shap_df.nlargest(10, 'importance')
    shap_df_bottom = shap_df.nsmallest(10, 'importance')

    plt.figure(figsize=(10, 6))
    plt.barh(shap_df_top['feature'].str.replace('from_', '').str.replace('to', '').str.replace('_', ' ').str.capitalize(), shap_df_top['importance'], color='forestgreen')
    plt.xlabel('Mean |SHAP value|', fontweight='bold')
    plt.title(f'Top 10 Features - XGBoost: {y_metric}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"../Figures/{path_prefix}/features/{y_metric}_xgboost_shap.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.barh(shap_df_bottom['feature'].str.replace('from_', '').str.replace('to', '').str.replace('_', ' ').str.capitalize(), shap_df_bottom['importance'], color='forestgreen')
    plt.xlabel('Mean |SHAP value|', fontweight='bold')
    plt.title(f'Bottom 10 Features - XGBoost: {y_metric}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"../Figures/{path_prefix}/features/{y_metric}_xgboost_shap_bottom.png", dpi=300, bbox_inches='tight')
    plt.close()

def team_scatter_plots(df, quality_type, metric, top_features):

    positions = POSITION_CATEGORIES["Defensive"]

    df_pos = df[df["from_position"].isin(positions)].copy()

    df_pos = df_pos.iloc[:50]

    top_5 = top_features.head(5)["feature"].tolist()
    
    for feature in top_5:
        df_pos = df_pos.dropna(subset=[metric, feature])
        y = df_pos[metric]
        x = df_pos[feature]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, color="steelblue", alpha=0.7, s=60, zorder=3)

        for _, row in df_pos.iterrows():
            ax.annotate(row["short_name"], (row[feature], row[metric]),
                        fontsize=7, alpha=0.7, xytext=(4, 4), textcoords="offset points")

        # Trend line
        if len(y) > 1:
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), color="tomato", linewidth=1.5, linestyle="--", label="Trend")
            ax.legend(fontsize=9)

        clean_tq = metric.replace("from_", "").replace("_", " ").title()
        feature_label = feature.replace("from_z_score_", "").replace("_", " ").capitalize()
        ax.set_xlabel(feature_label, fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Score for quality {clean_tq.capitalize()}", fontsize=11, fontweight="bold")
        ax.set_title(f"{clean_tq.capitalize()} vs {feature_label}", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        os.makedirs(f"../Figures/team_models/metric_analysis/{metric}", exist_ok=True)
        fig.savefig(f"../Figures/team_models/metric_analysis/{metric}/{feature}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

def main():
    """Main execution function"""
    
    # Load data
    df = get_all_data().copy()
    
    # Standardize column names
    df.columns = df.columns.str.replace(" ", "_")

    # Identify positional changes
    df_pos_changes = identify_positional_changes(df)

    # Keep only teams that received at most one transfer where the player played 400+ minutes
    if "to_Minutes" in df_pos_changes.columns and "to_team_id" in df_pos_changes.columns and "to_season" in df_pos_changes.columns:
        qualified = df_pos_changes[df_pos_changes["to_Minutes"] >= 400]

        transfer_counts = qualified.groupby(["to_team_id", "to_season"]).size()
        multi_transfer_keys = transfer_counts[transfer_counts > 1].index
        
        mask = df_pos_changes.set_index(["to_team_id", "to_season"]).index.isin(multi_transfer_keys)
        df_pos_changes = df_pos_changes[~mask]

    main_df = getTeamStatsChanges(df_pos_changes, df)

    
    df_from = main_df.copy()

    for quality_name in TEAM_QUALS:
        df_from = get_team_qualities(quality_name, df_from, "from_")

        main_df[f"from_{quality_name.lower()}"] = df_from[quality_name.lower()]

    if len(main_df) == 0:
        return

    if TRAIN_MODELS:
        # Train category-specific models
        results = train_category_models(main_df)
        
        # Create summary report
        for category, category_data in results.items():
            for metric_col, model_result in category_data['models'].items():
                if PLOT_RESULTS:
                    category_lower = category.lower()
                    # Subdirectory based on metric or quality
                    subdir = "quality" if QUALITIES else "metric"
                    os.makedirs(f"../Figures/team_models/{category_lower}/{subdir}", exist_ok=True)
                    
                    # Performance plot
                    plot_model_performance(
                        model_result,
                        title=f"{category} Category - Team Success Predictions",
                        metric_name=metric_col
                    )
                    # Feature importance / SHAP plot
                    plot_feature_importance(
                        model_result["feature_importance"],
                        shap_values=model_result["shap_values"],
                        X_train=model_result["X_train"],
                        path_prefix=f"team_models/{category_lower}/{subdir}",
                        y_metric=metric_col,
                        title=f"{category} Category - {metric_col.replace('delta', '').replace('_', ' ').strip().capitalize()} Feature Importance",
                    )

                    if team_scatter:
                        team_scatter_plots(category_data["data"]["df_filtered"], category, metric_col, model_result["feature_importance"])

if __name__ == "__main__":
    main()
