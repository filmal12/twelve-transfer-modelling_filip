import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import seaborn as sns

from misc.helpers import (
    clean_label
)

full_model = False
quality_model = False

folder_path = "metrics"

if quality_model:
    folder_path = "qualities"

def create_radar_plot_top_features(shap_values, X, predictions, top_n=5, target_name="", name="", path="", folder_path="", player_idx=0, player_name=""):    
    try:
        # Calculate mean absolute SHAP values
        shap_importance = np.abs(shap_values.values).mean(axis=0)
        
        # Create DataFrame and get top N features
        shap_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': shap_importance
        }).sort_values('importance', ascending=False)
        
        top_features_df = shap_importance_df.head(top_n)
        top_features = top_features_df['feature'].tolist()
        
        # Get the specified player's data
        if player_idx >= len(X):
            print(f"Player index {player_idx} out of bounds, using index 0")
            player_idx = 0
            
        player_data = X.iloc[player_idx]
        predicted_value = predictions[player_idx] if player_idx < len(predictions) else None
        
        # Extract values for top features
        values = [player_data[feature] for feature in top_features]
        
        # Normalize values to 0-1 range for better radar plot visualization
        values_normalized = []
        for i, feature in enumerate(top_features):
            val = player_data[feature]
            # Use the feature values from X (which may be standardized)
            feature_values = X[feature].values
            feature_min = feature_values.min()
            feature_max = feature_values.max()
            
            if feature_max > feature_min:
                normalized_val = (val - feature_min) / (feature_max - feature_min)
            else:
                normalized_val = 0.5
            
            values_normalized.append(max(0, min(1, normalized_val)))  # Clamp to [0, 1]
        
        # Clean up feature names for better readability
        if quality_model:
            labels = [f.replace('from', '').replace('_', ' ') for f in top_features]
        else:
            labels = [f.replace('from_z_score_', '').replace('_', ' ') for f in top_features]
        
        # Create radar plot with improved label positioning
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values_plot = values_normalized + values_normalized[:1]  # Complete the circle
        angles_plot = angles + angles[:1]
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        ax.plot(angles_plot, values_plot, 'o-', linewidth=2.5, color='steelblue', markersize=8, label='Player Stats')
        ax.fill(angles_plot, values_plot, alpha=0.25, color='steelblue')

        # Smooth radial gradient: white center -> dark-green perimeter.
        theta = np.linspace(0, 2 * np.pi, 240)
        max_radius = 1.0
        num_rings = 90
        inner_color = np.array([0.98, 1.00, 0.98])
        outer_color = np.array([0.05, 0.32, 0.10])
        for i in range(num_rings):
            ratio = i / max(1, num_rings - 1)
            ring_color = inner_color * (1 - ratio) + outer_color * ratio
            radius = (i + 1) * max_radius / num_rings
            ax.fill(theta, np.full_like(theta, radius), color=tuple(ring_color), alpha=1.0, zorder=0)
        
        # Place labels around the arc with tangential rotation.
        ax.set_xticks(angles)
        ax.set_xticklabels([])
        for angle, label in zip(angles, labels):
            angle_deg = np.degrees(angle)
            rotation = angle_deg - 90
            if 90 <= angle_deg <= 270:
                rotation += 180
                ha = 'right'
            else:
                ha = 'left'

            ax.text(
                angle,
                1.08,
                label,
                fontsize=12,
                fontweight='bold',
                color='white',
                rotation=rotation,
                rotation_mode='anchor',
                ha=ha,
                va='center'
            )
        
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.45, color='white', linewidth=1.2)
        ax.set_facecolor('#0d4f1f')
        fig.patch.set_facecolor('#0d4f1f')
        ax.tick_params(colors='white')
        ax.spines['polar'].set_color('white')
        
        # Build title with player info and prediction
        title_str = f"Top {top_n} Features for {target_name} Success\n"
        if player_name:
            title_str += f"Player: {player_name}\n"
        title_str += f"({name.title() if name else 'All'} Wingers - Based on SHAP Importance)"
        if predicted_value is not None:
            title_str += f"\nPredicted {target_name}: {predicted_value:.4f}"
        
        ax.set_title(title_str, fontsize=14, fontweight='bold', pad=30, color='white')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=11)
        
        plt.tight_layout()
        os.makedirs(f"../Figures/{path}/xgboost_predictions/{folder_path}", exist_ok=True)
        
        # Use player name in filename if available
        filename = f"{target_name}_radar_top_features.png"
        if player_name:
            filename = f"{target_name}_radar_{player_name.replace(' ', '_')}.png"
            
        plt.savefig(f"../Figures/{path}/xgboost_predictions/{folder_path}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Radar plot created for player (index {player_idx}) with top {top_n} features based on SHAP importance")
        
    except Exception as e:
        print(f"Error creating radar plot: {e}")


def print_stats(valid_actuals, valid_predictions, valid_residuals, target_name, name, model_name):
        mse = mean_squared_error(valid_actuals, valid_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(valid_actuals, valid_predictions)
        r2 = r2_score(valid_actuals, valid_predictions)
        
        print(f"\n" + "="*70)
        print(f"{model_name.upper()} REGRESSION STATISTICS - {name.upper() if name else 'ALL'}")
        print("="*70)
        print(f"\nNumber of players: {len(valid_actuals)}")
        print(f"Model performance metrics:")
        print(f"  R² Score:     {r2:.4f}")
        print(f"  MSE:          {mse:.4f}")
        print(f"  RMSE:         {rmse:.4f}")
        print(f"  MAE:          {mae:.4f}")
        
        print(f"\nActual {target_name} scores - Statistics:")
        print(f"  Mean:   {valid_actuals.mean():.4f}")
        print(f"  Median: {np.median(valid_actuals):.4f}")
        print(f"  Std:    {valid_actuals.std():.4f}")
        
        print(f"\nPredicted {target_name} scores - Statistics:")
        print(f"  Mean:   {valid_predictions.mean():.4f}")
        print(f"  Median: {np.median(valid_predictions):.4f}")
        print(f"  Std:    {valid_predictions.std():.4f}")

        return r2, mse

def evaluate_model_performance(models_dict, X_test, y_test, target_name="", output_folder="Statistics"):    
    # Create output directory
    output_path = f"../Figures/{output_folder}"
    os.makedirs(output_path, exist_ok=True)
    
    performance_data = []
    
    # Evaluate each model
    for model_name, model in models_dict.items():
        try:
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            performance_data.append({
                'Model': model_name,
                'R² Score': r2,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae
            })
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue
    
    # Create DataFrame
    performance_df = pd.DataFrame(performance_data)
    
    # Save as CSV
    csv_path = f"../{output_path}/model_performance_metrics_{target_name}.csv" if target_name else f"../{output_path}/model_performance_metrics.csv"
    performance_df.to_csv(csv_path, index=False)
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Model Performance Comparison{f' - {target_name}' if target_name else ''}", 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: R² Score
    ax = axes[0, 0]
    bars1 = ax.bar(performance_df['Model'], performance_df['R² Score'], color='steelblue', alpha=0.7)
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_title('R² Score (Higher is Better)', fontweight='bold')
    ax.set_ylim(0, max(performance_df['R² Score'].max(), 1.0) * 1.1)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: MSE
    ax = axes[0, 1]
    bars2 = ax.bar(performance_df['Model'], performance_df['MSE'], color='coral', alpha=0.7)
    ax.set_ylabel('MSE', fontweight='bold')
    ax.set_title('Mean Squared Error (Lower is Better)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: RMSE
    ax = axes[1, 0]
    bars3 = ax.bar(performance_df['Model'], performance_df['RMSE'], color='mediumseagreen', alpha=0.7)
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('Root Mean Squared Error (Lower is Better)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: MAE
    ax = axes[1, 1]
    bars4 = ax.bar(performance_df['Model'], performance_df['MAE'], color='mediumpurple', alpha=0.7)
    ax.set_ylabel('MAE', fontweight='bold')
    ax.set_title('Mean Absolute Error (Lower is Better)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars4:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = f"../{output_path}/model_performance_comparison_{target_name}.png" if target_name else f"{output_path}/model_performance_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return performance_df

def plot_model_comparison(models_performance_dict, output_folder="Statistics"):    
    # Create output directory
    output_path = f"../Figures/{output_folder}"
    os.makedirs(output_path, exist_ok=True)
    
    # Combine all performance data
    all_performance = []
    for target_name, perf_df in models_performance_dict.items():
        perf_df_copy = perf_df.copy()
        perf_df_copy['Target'] = target_name
        all_performance.append(perf_df_copy)
    
    combined_df = pd.concat(all_performance, ignore_index=True)
    
    # Create multi-plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Across All Targets', fontsize=18, fontweight='bold')
    
    metrics = ['R² Score', 'MSE', 'RMSE', 'MAE']
    colors_map = plt.cm.Set3(np.linspace(0, 1, len(combined_df['Model'].unique())))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Pivot data for grouped bar plot
        pivot_df = combined_df.pivot_table(values=metric, index='Target', columns='Model')
        
        pivot_df.plot(kind='bar', ax=ax, color=colors_map, alpha=0.8, width=0.8)
        
        ax.set_title(f'{metric} Across Targets', fontweight='bold', fontsize=12)
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_xlabel('Target Variable', fontweight='bold')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = f"../{output_path}/all_models_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save combined data
    csv_path = f"../{output_path}/all_models_performance.csv"
    combined_df.to_csv(csv_path, index=False)

def create_r2_residuals_plot(valid_actuals, valid_predictions, valid_residuals, r2, target_name="", path=""):
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    
    # Plot 1: Actual vs Predicted
    axes[0].scatter(valid_actuals, valid_predictions, alpha=0.6, s=50)
    min_val = min(valid_actuals.min(), valid_predictions.min())
    max_val = max(valid_actuals.max(), valid_predictions.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel(f"Actual {target_name} Score", fontweight='bold')
    axes[0].set_ylabel(f"Predicted {target_name} Score", fontweight='bold')
    axes[0].set_title(f"Actual vs Predicted (R² = {r2:.4f})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Residuals vs Predicted
    axes[1].scatter(valid_predictions, valid_residuals, alpha=0.6, s=50)
    axes[1].axhline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel(f"Predicted {target_name} Score", fontweight='bold')
    axes[1].set_ylabel("Residuals", fontweight='bold')
    axes[1].set_title("Residuals vs Predicted Values")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(f"../Figures/{path}/models", exist_ok=True)
    plt.savefig(f"../Figures/{path}/models/{target_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_transition_diagram(tc_df, OUT_DIR):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(tc_df["transition"], tc_df["n_players"], color="darkorange")
    ax.set_xlabel("Number of Players Predicted", fontweight="bold")
    ax.set_title("Players Predicted per Position Transition", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for i, v in enumerate(tc_df["n_players"]):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "player_counts_per_transition.png"), dpi=220)
    plt.close(fig)

def plot_distribution_of_transition(counts_df, positions, n_pos, OUT_DIR):
    fig, axes = plt.subplots(1, n_pos, figsize=(5 * n_pos, 5), squeeze=False)

    for i, pos in enumerate(positions):
        ax = axes[0][i]
        pos_df = counts_df[counts_df["from_position"] == pos]
        tally = pos_df["n_transitions"].value_counts().sort_index()
        ax.bar(tally.index.astype(str), tally.values, color="steelblue")
        ax.set_title(clean_label(pos), fontweight="bold")
        ax.set_xlabel("# Transitions Predicted")
        ax.set_ylabel("# Players")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for x, y in zip(tally.index, tally.values):
            ax.text(str(x), y + 0.3, str(y), ha="center", fontsize=9)

    fig.suptitle(
        "Distribution: How Many Transitions Each Player Was Predicted For",
        fontweight="bold",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "player_transition_counts_by_position.png"), dpi=220
    )
    plt.close(fig)

    # Also save a heatmap: source position × n_transitions
    pivot = (
        counts_df.groupby(["from_position", "n_transitions"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.index = [clean_label(i) for i in pivot.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title(
        "Player Count by Source Position and # Transitions Predicted",
        fontweight="bold",
    )
    ax.set_xlabel("# Transitions Predicted")
    ax.set_ylabel("Source Position")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "transition_count_heatmap.png"), dpi=220
    )
    plt.close(fig)

def plot_switch_analysis(x, from_positions, width, stay_counts, switch_counts, OUT_DIR):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_stay = ax.bar(
        x - width / 2,
        [stay_counts.get(p, 0) for p in from_positions],
        width, label="Stay", color="steelblue",
    )
    bars_sw = ax.bar(
        x + width / 2,
        [switch_counts.get(p, 0) for p in from_positions],
        width, label="Switch", color="darkorange",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([clean_label(p) for p in from_positions], fontsize=10)
    ax.set_ylabel("Number of Sampled Players", fontweight="bold")
    ax.set_title(
        "Predicted to Stay vs Switch Position",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars_stay:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                    str(int(h)), ha="center", fontsize=8)
    for bar in bars_sw:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                    str(int(h)), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "position_switch_vs_stay.png"), dpi=220)
    plt.close(fig)

def plot_shap_features(combined_shap, labels, colors, OUT_DIR, PNG_NAME):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], combined_shap.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean SHAP Importance", fontweight="bold")
    ax.set_title(
        "Attributes influence",
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{PNG_NAME}.png"), dpi=220)
    plt.close(fig)

def plot_position_breakdown(cd_df, dest_counts, OUT_DIR, PNG_NAME):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart
    ax = axes[0]
    colors_cd = ["darkorange" if d == "Full Back" else "steelblue" for d in dest_counts.index]
    ax.bar([clean_label(d) for d in dest_counts.index], dest_counts.values, color=colors_cd)
    ax.set_ylabel("Number of Sampled Players", fontweight="bold")
    ax.set_title("Central Defender: Predicted Destination", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for x, y in zip([clean_label(d) for d in dest_counts.index], dest_counts.values):
        ax.text(x, y + 2, str(y), ha="center", fontsize=9)

    # Pie chart
    ax2 = axes[1]
    wedge_colors = ["darkorange" if d == "Full Back" else "steelblue" for d in dest_counts.index]
    wedges, texts, autotexts = ax2.pie(
        dest_counts.values,
        labels=[clean_label(d) for d in dest_counts.index],
        colors=wedge_colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax2.set_title("Central Defender: Destination Share", fontweight="bold")

    fig.suptitle(
        f"Where Central Defenders Are Predicted to Move\n(n={len(cd_df)} samples)",
        fontweight="bold", fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{PNG_NAME}.png"), dpi=220)
    plt.close(fig)

def plot_feature_similarity(compare_df, cd_mean, fb_mean, first_pos, second_pos, OUT_DIR, PNG_NAME):
    fig, ax = plt.subplots(figsize=(7, 7))
    for feat, cd_val, fb_val in zip(
        compare_df.index, cd_mean.values, fb_mean.values
    ):
        ax.scatter(cd_val, fb_val, color="steelblue", zorder=3)
        ax.annotate(feat, (cd_val, fb_val), fontsize=7,
                    xytext=(4, 2), textcoords="offset points")
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.4, label="y = x (perfect similarity)")
    ax.set_xlabel(f"{first_pos} Mean Quality", fontweight="bold")
    ax.set_ylabel(f"{second_pos} Mean Quality", fontweight="bold")
    ax.set_title(
        f"Feature Similarity: {first_pos} vs {second_pos}\n",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{PNG_NAME}.png"), dpi=220)
    plt.close(fig)