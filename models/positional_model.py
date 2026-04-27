import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
import joblib
import os
import shap
import sys

sys.path.append(os.path.abspath(".."))
from team_qualities import get_team_qualities, convertToZscores
from setup import (
    TEAM_QUALS,
    IND_VARS,
    normal_quals
)
 
from misc.plots import create_r2_residuals_plot

stepwise = True
plot = True
FEATURE_COUNT = 10
PRINT = False
SAVE_PARAMS = True
TRAIN = True
Train_small_model = False
full_model = False
quality_model = False

folder_path = "metrics"

if quality_model:
    folder_path = "qualities"

def plot_stats(coefficients_sorted, quality_cols, quality_name, path, df_clean, linear_model, name, pos):
    # Create figure
    plt.figure(figsize=(12, max(8, len(quality_cols) * 0.5)))
    
    # Create horizontal bar plot
    colors = ['green' if x > 0 else 'red' for x in coefficients_sorted.values]
    if len(quality_cols) > 1:
        plt.barh(range(len(coefficients_sorted)), coefficients_sorted.values, color=colors, alpha=0.7)
        # Clean up labels for readability
        labels = [
            label.replace("from_z_score_", "").replace("_", " ").title()
            for label in coefficients_sorted.index
        ]
        
        plt.yticks(range(len(coefficients_sorted)), labels, fontsize=10)
        plt.xlim(min(coefficients_sorted.values) - 1, max(coefficients_sorted.values) + 1)
        plt.xlabel("Coefficient Value", fontsize=12, fontweight='bold')
        plt.title(f"Winger {quality_name} Attributes - Good {quality_name} {pos}\n(Impact on {quality_name} Improvement)", 
                fontsize=14, fontweight='bold')
        
        # Add grid for better readability
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add a vertical line at x=0
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add coefficient values on bars
        for i, v in enumerate(coefficients_sorted.values):
            plt.text(v + 0.01 if v > 0 else v - 0.01, i, f'{v:.3f}', 
                    va='center', ha='left' if v > 0 else 'right', fontsize=9)
        
        # Calculate MSE from residuals
        mse = np.mean(linear_model.resid ** 2)
        
        # Add legend with R² and MSE
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Positive Impact'),
                        Patch(facecolor='red', alpha=0.7, label='Negative Impact'),
                        Patch(facecolor='white', alpha=0, label=f'R² = {linear_model.rsquared:.4f}'),
                        Patch(facecolor='white', alpha=0, label=f'MSE = {mse:.4f}')]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig(f"../Figures/{path}/{pos}_{quality_name.lower()}.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        predictions = linear_model.predict(df_clean)

        actuals = df_clean["Target"]
        valid_mask = ~np.isnan(actuals)
        valid_actuals = actuals[valid_mask]
        valid_predictions = predictions[valid_mask]

        min_val = min(valid_actuals.min(), valid_predictions.min())
        max_val = max(valid_actuals.max(), valid_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        plt.scatter(valid_actuals, valid_predictions)
        plt.xlabel("Actual Score", fontweight='bold')
        plt.ylabel("Predicted Score", fontweight='bold')
        plt.title(f"Fit")
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"../Figures/{path}/{pos}_{quality_name.lower()}.png", dpi=300, bbox_inches='tight')
        plt.close()
    if PRINT:
        # Print summary
        print("\n" + "="*70)
        print(f"WINGER {quality_name.upper()} ATTRIBUTES - POACHING STRIKER TRANSITION")
        print("="*70)
        print(f"\nModel R-squared: {linear_model.rsquared:.4f}")
        print(f"Number of {quality_name} attributes: {len(quality_cols)}")
        print(f"Sample size: {df_clean.shape[0]}")
        print(f"\nModel Summary:\n")
        print(linear_model.summary())
        print("\n" + "-"*70)
        print("ATTRIBUTE IMPACT ON POACHING IMPROVEMENT:")
        print("-"*70)
        for var, coef in coefficients_sorted.items():
            pval = linear_model.pvalues[var]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "NS"
            direction = "increases" if coef > 0 else "decreases"
            var_clean = var.replace("from_z_score_", "").replace("_", " ")
            print(f"  {var_clean:40s}: {direction:10s} poaching ({coef:+.4f}) [{sig}]")
        print("="*70 + "\n")

def remove_correlated_features(df, target, columns, correlation_threshold=0.8):
    """
    Remove highly correlated features to reduce multicollinearity.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the features
    columns : list
        List of column names to analyze for correlation
    correlation_threshold : float
        Correlation threshold above which to remove one feature (default: 0.8)
    
    Returns:
    --------
    list : Filtered list of column names with highly correlated features removed
    """
    
    # Calculate correlation matrix for the specified columns
    corr_matrix = df[columns].corr().abs()

    # Select upper triangle of correlation matrix to avoid duplicates
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation greater than threshold
    to_drop = set()
    for column in upper.columns:
        high_corr = upper[column][upper[column] > correlation_threshold].index.tolist()
        for corr_col in high_corr:
            # Remove the column with lower variance to keep the one with more information
            if df[column].var() < df[corr_col].var():
                to_drop.add(column)
            else:
                to_drop.add(corr_col)
    
    # Get filtered list of columns
    filtered_columns = [c for c in columns if c not in to_drop]
    
    return filtered_columns

def createShapPlot(model, X, model_type, target, path_prefix):
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    shap_df = pd.DataFrame({
        'feature': X.columns,
        'importance': shap_importance
    }).sort_values('importance', ascending=True).tail(10)
    
    plt.figure(figsize=(10, 6))
    plt.barh(shap_df['feature'].str.replace('from_z_score_', '').str.replace('_', ' '), shap_df['importance'], color='coral')
    plt.xlabel('Mean SHAP', fontweight='bold')
    plt.title(f'Top 10 Features - {model_type}: {target}', fontweight='bold')
    os.makedirs(f"../Figures/{path_prefix}/features", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"../Figures/{path_prefix}/features/{target}_lasso_shap.png", dpi=300, bbox_inches='tight')
    plt.close()

OLS = True
XGBOOST = False
RIDGE = True
LASSO = True
RF = True
TARGET_CHOICE = "QUAL"

CD_IGNORED_VARS = ["from_Box_threat", "from_Run_quality", "from_Poaching", "from_Finishing", "from_Passing_quality", "from_Defensive_heading", "from_Progression"]

INDEPENDENT_VARIABLES_GENERAL = IND_VARS
INDEPENDENT_CD = [c for c in IND_VARS if c not in CD_IGNORED_VARS]

def ols_model(suggest_features, clean_df, path_prefix, target, save_params):
    # Stepwise regression to select final OLS features
    suggest_features_ols = suggest_features.copy()

    competition_encoded = []

    while len(suggest_features_ols) > 0:
        formula = "Target ~ " + " + ".join(suggest_features_ols)
        linear_model = smf.ols(formula=formula, data=clean_df).fit()
        pvals = linear_model.pvalues
        worst_p = pvals.max()
        worst_var = pvals.idxmax()
        
        if worst_var == "Intercept":
            top2 = pvals.nlargest(2)
            worst_var = top2.index[1]
            worst_p = top2.iloc[1]
        
        if worst_p > 0.05 and worst_var != "Intercept":
            if "competition" in worst_var:
                competition_encoded.append(worst_var)
                # Find next worst variable that is not a competition variable
                non_comp_pvals = pvals[
                    ~pvals.index.astype(str).str.contains("competition") & (pvals.index != "Intercept")
                ]
                if non_comp_pvals.empty or non_comp_pvals.max() <= 0.05:
                    break
                worst_var = non_comp_pvals.idxmax()
                worst_p = non_comp_pvals.max()

            suggest_features_ols.remove(worst_var)
        else:
            break
    
    # Train OLS model
    formula_ols = "Target ~ " + " + ".join(suggest_features_ols)
    ols_model = smf.ols(formula=formula_ols, data=clean_df).fit()
    ols_r2 = ols_model.rsquared
    
    # Create residual plot for OLS
    ols_predictions = ols_model.predict(clean_df)
    ols_residuals = clean_df["Target"].values - ols_predictions.values
    os.makedirs(f"../Figures/{path_prefix}", exist_ok=True)
    create_r2_residuals_plot(clean_df["Target"].values, ols_predictions.values, ols_residuals, ols_r2, f"{target}_ols", path_prefix)
    
    # Create SHAP plot for OLS
    try:
        createShapPlot(ols_models, clean_df, "ols", target, path_prefix)
    except:
        pass
    
    if save_params:
        os.makedirs(f"../parameters/{path_prefix}", exist_ok=True)
        joblib.dump(ols_model, f'../parameters/{path_prefix}/{target}_ols.pkl')
        params_df = pd.DataFrame(columns=['Factor', 'mean', 'min', 'max'])
        bse = ols_model.bse
        b = ols_model.params
        bmin = b - bse
        bmax = b + bse
        
        for name, val in b.items():
            param_row = pd.DataFrame({
                'Factor': [name],
                'mean': [val],
                'min': [bmin[name]],
                'max': [bmax[name]],
            })
            params_df = pd.concat([params_df, param_row], ignore_index=True)
        
        params_df.to_csv(f'../parameters/{path_prefix}/{target}.csv', index=False)

    return ols_r2

def xgboost_model(suggest_features, clean_df, path_prefix, target, save_models):
    # Prepare data for tree/regression models
    X_full = clean_df[suggest_features].copy().fillna(0)
    # Cast competition text columns to category dtype so XGBoost can handle them
    for col in X_full.select_dtypes(include="object").columns:
        X_full[col] = X_full[col].astype("category")
    y = clean_df["Target"].copy()
    
    # ========== XGBOOST ==========
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_child_weight=2,
        gamma=0,
        random_state=42,
        enable_categorical=True
    )
    
    xgb_model.fit(X_full, y)
    # Iteratively remove zero-importance features until all remaining features contribute
    while True:
        active_mask = xgb_model.feature_importances_ > 0
        active_features = X_full.columns[active_mask].tolist()
        if len(active_features) == len(X_full.columns) or len(active_features) == 0:
            break
        X_full = X_full[active_features]
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            min_child_weight=2,
            gamma=0,
            random_state=42,
            enable_categorical=True
        )
        xgb_model.fit(X_full, y)

    y_pred_xgb = xgb_model.predict(X_full)
    xgb_r2 = r2_score(y, y_pred_xgb)
    xgb_residuals = y.values - y_pred_xgb

    # df_predictions = pd.concat([df_predictions, X_full.assign(Target=y, Prediction=xgb_model.predict(X_full), Model=f"{target}_xgboost")], ignore_index=True)
    
    # Create residual plot for XGBoost
    os.makedirs(f"../Figures/{path_prefix}", exist_ok=True)
    create_r2_residuals_plot(y.values, y_pred_xgb, xgb_residuals, xgb_r2, f"{target}_xgboost", path_prefix)
    print(f"\nTarget: {target} - R^2: {xgb_r2:.4f}\n")

    # Create SHAP plot for XGBoost
    try:
        explainer_xgb = shap.TreeExplainer(xgb_model)
        shap_values_xgb = explainer_xgb.shap_values(X_full)
        shap_importance_xgb = np.abs(shap_values_xgb).mean(axis=0)

        np.save(f"../parameters/{path_prefix}/{target}_xgboost_shap_values.npy", shap_values_xgb)
        
        shap_df_xgb = pd.DataFrame({
            'feature': X_full.columns,
            'importance': shap_values_xgb.mean(axis=0)
        }).sort_values('importance', ascending=True)

        if save_models:
            os.makedirs(f"../parameters/{path_prefix}", exist_ok=True)
            joblib.dump(xgb_model, f'../parameters/{path_prefix}/{target}_xgboost.pkl')
            
            feature_importance = pd.DataFrame({
                'feature': X_full.columns.tolist(),
                'importance': shap_values_xgb.mean(axis=0)
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv(f'../parameters/{path_prefix}/{target}_top_features.csv', index=False)

            createShapPlot(xgb_model, X_full, "xgboost", target, path_prefix)
    except Exception as e:
        print(f"Error creating SHAP plot for {target} XGBoost: {e}")
        pass
    
    return xgb_r2

def lasso_model(suggest_features, clean_df, path_prefix, target, save_models):
    X_full = clean_df[suggest_features].copy().fillna(0)
    y = clean_df["Target"].copy()
    lasso_model = Lasso(alpha=0.1, random_state=42)
    lasso_model.fit(X_full, y)
    y_pred_lasso = lasso_model.predict(X_full)
    lasso_r2 = r2_score(y, y_pred_lasso)
    lasso_residuals = y.values - y_pred_lasso
    
    if save_models:
        joblib.dump(lasso_model, f'../parameters/{path_prefix}/{target}_lasso.pkl')
    
    # Create residual plot for Lasso
    create_r2_residuals_plot(y.values, y_pred_lasso, lasso_residuals, lasso_r2, f"{target}_lasso", path_prefix)
    
    # Create SHAP plot for Lasso
    try:
        createShapPlot(lasso_model, X_full, "lasso", target, path_prefix)
    except:
        pass

    return lasso_r2

def ridge_model(suggest_features, clean_df, path_prefix, target, save_models):
    # ========== RIDGE ==========
    X_full = clean_df[suggest_features].copy().fillna(0)
    y = clean_df["Target"].copy()
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_full, y)
    y_pred_ridge = ridge_model.predict(X_full)
    ridge_r2 = r2_score(y, y_pred_ridge)
    ridge_residuals = y.values - y_pred_ridge
    
    if save_models:
        joblib.dump(ridge_model, f'../parameters/{path_prefix}/{target}_ridge.pkl')
    
    # Create residual plot for Ridge
    create_r2_residuals_plot(y.values, y_pred_ridge, ridge_residuals, ridge_r2, f"{target}_ridge", path_prefix)
    
    # Create SHAP plot for Ridge
    try:
        createShapPlot(ridge_model, X_full, "ridge", target, path_prefix)
    except:
        pass

    return ridge_r2

def forest_model(suggest_features, clean_df, path_prefix, target, save_models):
    X_full = clean_df[suggest_features].copy().fillna(0)
    y = clean_df["Target"].copy()
    # ========== RANDOM FOREST ==========
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_full, y)
    y_pred_rf = rf_model.predict(X_full)
    rf_r2 = r2_score(y, y_pred_rf)
    rf_residuals = y.values - y_pred_rf
    
    if save_models:
        joblib.dump(rf_model, f'../parameters/{path_prefix}/{target}_randomforest.pkl')
    
    # Create residual plot for Random Forest
    create_r2_residuals_plot(y.values, y_pred_rf, rf_residuals, rf_r2, f"{target}_randomforest", path_prefix)
    
    # Create SHAP plot for Random Forest
    try:
        createShapPlot(rf_model, X_full, "randomforest", target, path_prefix)
    except:
        pass

    return rf_r2
        
# ========== TRAINING FUNCTIONS FOR POSITION TRANSITIONS ==========
def _train_position_models(df, from_pos, to_pos, targets, path_prefix, save_params=True, save_models=True, verbose=True, df_predictions=None):  
    # Clean column names
    df = df.copy()
    df.columns = df.columns.str.replace(" ", "_")
    
    # Get feature columns
    z_score_cols = []

    if (to_pos == "Central Defender") or (to_pos == "Full Back"):
        z_score_cols = INDEPENDENT_CD
    else:
        z_score_cols = INDEPENDENT_VARIABLES_GENERAL

    all_z_features = z_score_cols.copy()

    all_z_features.append("wyscout_weight_scaled")
    all_z_features.append("player_season_age_scaled")
    all_z_features.append("wyscout_height_scaled")

    # One-hot encode competition columns; competitions with fewer than min_comp_freq
    # appearances are grouped into "Other" to avoid sparse, uninformative dummies.
    min_comp_freq = max(5, len(df) // 20)
    for comp_col in ["from_competition_name", "to_competition_name"]:
        if comp_col in df.columns:
            freq = df[comp_col].value_counts()
            valid_comps = freq[freq >= min_comp_freq].index
            col_filtered = df[comp_col].where(df[comp_col].isin(valid_comps), other="Other")
            dummies = pd.get_dummies(col_filtered, prefix=comp_col, drop_first=True, dtype=float)
            dummies.columns = dummies.columns.astype(str).str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
            for dummy_col in dummies.columns:
                df[dummy_col] = dummies[dummy_col]
                all_z_features.append(dummy_col)

    team_quals = TEAM_QUALS

    df_get_team = df.copy()
    df_to_team = df.copy()

    for qual in team_quals:
        df_get_team = get_team_qualities(qual, df_get_team, "from_")
        df_to_team = get_team_qualities(qual, df_to_team, "to_")
        quality_col = qual.lower()
        all_z_features.append(f"from_{quality_col}")
        all_z_features.append(f"to_{quality_col}")

        df[f"from_{quality_col}"] = df_get_team[quality_col]
        df[f"to_{quality_col}"] = df_to_team[quality_col]

    stats_df = pd.DataFrame()

    for target in targets:
        if verbose:
            print(f"{target:20s}", end=" | ")
        
        target_col = f"from_{target}"
        target_to_col = f"to_{target}"
        
        # Remove self-target from features
        suggest_features = [c for c in all_z_features if c != target_col]
        all_features = [c for c in all_z_features if c != target_col]
        
        # Create target: absolute value in new position
        if TARGET_CHOICE == "CHANGE":
            df["Target"] = df[target_to_col] - df[target_col]
        else:
            df["Target"] = df[target_to_col]
        suggest_features = remove_correlated_features(df, target, suggest_features, correlation_threshold=0.8)

        # Get clean subset
        clean_df = df[suggest_features + ["Target"]].fillna(0)

        if len(clean_df) < 5:
            continue
        
        if OLS:            
            r2 = ols_model(suggest_features, clean_df, path_prefix, target, save_params)

            tmp_stats = pd.DataFrame(data=[["ols", target, r2]], columns = ["Model", "Target", "R^2"])

            stats_df = pd.concat([stats_df, tmp_stats], ignore_index=True)
        
        if XGBOOST:
            r2 = xgboost_model(suggest_features, clean_df, path_prefix, target, save_models)

            tmp_stats = pd.DataFrame(data=[["xgboost", target, r2]], columns = ["Model", "Target", "R^2"])

            stats_df = pd.concat([stats_df, tmp_stats], ignore_index=True)
        
        if LASSO: 
            r2 = lasso_model(suggest_features, clean_df, path_prefix, target, save_models)

            tmp_stats = pd.DataFrame(data=[["lasso", target, r2]], columns = ["Model", "Target", "R^2"])

            stats_df = pd.concat([stats_df, tmp_stats], ignore_index=True)
        
        if RIDGE:
            r2 = ridge_model(suggest_features, clean_df, path_prefix, target, save_models)

            tmp_stats = pd.DataFrame(data=[["ridge", target, r2]], columns = ["Model", "Target", "R^2"])

            stats_df = pd.concat([stats_df, tmp_stats], ignore_index=True)
        if RF:
            r2 = forest_model(suggest_features, clean_df, path_prefix, target, save_models)

            tmp_stats = pd.DataFrame(data=[["rf", target, r2]], columns = ["Model", "Target", "R^2"])

            stats_df = pd.concat([stats_df, tmp_stats], ignore_index=True)

    stats_tot = stats_df.groupby("Model")["R^2"].mean().reset_index()

    stats_tot["from_pos"] = from_pos

    stats_tot["to_pos"] = to_pos

    file_path = "../Figures/model_evaluation/model_metrics_no_team_to.csv"

    if os.path.isfile(file_path):
        stats_tot.to_csv(file_path, mode='a', header=False, index=False)
    else:
        os.makedirs(os.path.dirname("../Figures/model_evaluation"), exist_ok=True)
        stats_tot.to_csv(file_path, mode='w', header=True, index=False)


# ========== SPECIFIC POSITION TRANSITION TRAINING FUNCTIONS ==========

def train_winger_to_striker(df, targets, save_params=True, save_models=True, verbose=True, df_predictions=None):
    _train_position_models(df, "Winger", "Striker", targets, "winger_to_st", save_params, save_models, verbose, df_predictions)

def train_winger_to_fb(df, targets, save_params=True, save_models=True, verbose=True, df_predictions=None):
    _train_position_models(df, "Winger", "Full Back", targets, "winger_to_fb", save_params, save_models, verbose, df_predictions)

def train_fb_to_cd(df, targets, save_params=True, save_models=True, verbose=True, df_predictions=None):
    _train_position_models(df, "Full Back", "Central Defender", targets, "fb_to_cd", save_params, save_models, verbose, df_predictions)

def train_fb_to_winger(df, targets, save_params=True, save_models=True, verbose=True, df_predictions=None):
    _train_position_models(df, "Full Back", "Winger", targets, "fb_to_winger", save_params, save_models, verbose, df_predictions)

def train_cd_to_fb(df, targets, save_params=True, save_models=True, verbose=True, df_predictions=None):
    _train_position_models(df, "Central Defender", "Full Back", targets, "cd_to_fb", save_params, save_models, verbose, df_predictions)

def train_cd_to_mf(df, targets, save_params=True, save_models=True, verbose=True, df_predictions=None):
    _train_position_models(df, "Central Defender", "Midfielder", targets, "cd_to_mf", save_params, save_models, verbose, df_predictions)

def train_st_to_winger(df, targets, save_params=True, save_models=True, verbose=True, df_predictions=None):
    _train_position_models(df, "Striker", "Winger", targets, "st_to_winger", save_params, save_models, verbose, df_predictions)

def train_st_to_mf(df, targets, save_params=True, save_models=True, verbose=True, df_predictions=None):
    _train_position_models(df, "Striker", "Midfielder", targets, "st_to_mf", save_params, save_models, verbose, df_predictions)

def train_midfielders(df, targets, save_params=True, save_models=True, verbose=True, from_subPos=None, to_subPos=None, df_predictions=None):
    _train_position_models(df, from_subPos, to_subPos, targets, f"midfielders/{from_subPos}_to_{to_subPos}", save_params, save_models, verbose, df_predictions)

def train_striker_to_am(df, targets, save_params=True, save_models=True, verbose=True, df_predictions=None):
    _train_position_models(df, "Striker", "Attacking Midfielder", targets, "st_to_am", save_params, save_models, verbose, df_predictions)

def train_defender_to_dm(df, targets, save_params=True, save_models=True, verbose=True, df_predictions=None):
    _train_position_models(df, "Defender", "Defensive Midfielder", targets, "cd_to_dm", save_params, save_models, verbose, df_predictions)

def train_same_position(df, targets, save_params=True, save_models=True, verbose=True, position=None, df_predictions=None):
    _train_position_models(df, position, position, targets, f"same_position/{position}", save_params, save_models, verbose, df_predictions)

def train_mf_to_st(df, targets, save_params=True, save_models=True, verbose=True, position=None, df_predictions=None):
    _train_position_models(df, "Midfielder", "Striker", targets, f"mf_to_st", save_params, save_models, verbose, df_predictions)

def train_mf_to_cd(df, targets, save_params=True, save_models=True, verbose=True, position=None, df_predictions=None):
    _train_position_models(df, "Midfielder", "Central Defender", targets, f"mf_to_cd", save_params, save_models, verbose, df_predictions)