import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT_DIR = "Figures/mae_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# ── colours per model type ──────────────────────────────────────────
TEAM_COLORS = {
    "Naive": "#bdbdbd",
    "Player only": "#6baed6",
    "Team only": "#fd8d3c",
    "Full model": "#31a354",
}

PLAYER_COLORS = {
    "Naive": "#bdbdbd",
    "Player baseline": "#fd8d3c",
    "Full model": "#31a354",
}


def _pct_diff(val, ref):
    """Percentage change from ref to val (negative = improvement)."""
    """Percentage improvement from ref to val (positive = lower MAE = better)."""
    if ref == 0:
        return 0.0
    return (ref - val) / abs(ref) * 100


# =====================================================================
# 1.  TEAM MODEL CHART
# =====================================================================
def plot_team_mae():
    naive = pd.read_csv("parameters/team_models/rsquared_naive.csv")
    qual = pd.read_csv("parameters/team_models/rsquared_qual.csv")
    team = pd.read_csv("parameters/team_models/rsquared_team.csv")
    full = pd.read_csv("parameters/team_models/rsquared_full.csv")

    # Clean target names
    for df in [naive, qual, team, full]:
        df["Target"] = df["Target"].str.replace("_", " ").str.title()

    qualities = naive["Target"].tolist()

    models = {
        "Naive": naive,
        "Player only": qual,
        "Team only": team,
        "Full model": full,
    }

    n_models = len(models)
    bar_height = 0.18
    y_pos = np.arange(len(qualities))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (label, df) in enumerate(models.items()):
        mae_vals = df.set_index("Target").reindex(qualities)["MAE"].values
        naive_vals = naive.set_index("Target").reindex(qualities)["MAE"].values
        offsets = y_pos + (i - n_models / 2 + 0.5) * bar_height

        bars = ax.barh(offsets, mae_vals, height=bar_height,
                       label=label, color=TEAM_COLORS[label], edgecolor="white")

        for bar, mae, naive_mae in zip(bars, mae_vals, naive_vals):
            if label == "Naive":
                continue
            pct = _pct_diff(mae, naive_mae)
            sign = "+" if pct > 0 else ""
            ax.text(
                bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{sign}{pct:.1f}%", va="center", fontsize=8, fontweight="bold",
                color="green" if pct > 0 else "red",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(qualities, fontsize=10)
    ax.set_xlabel("MAE", fontsize=12, fontweight="bold")
    ax.set_title("Team Model – MAE by Quality and Model Type", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/team_model_mae_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/team_model_mae_comparison.png")


# =====================================================================
# 2.  PLAYER MODEL CHART
# =====================================================================
def plot_player_mae():
    naive = pd.read_csv("Figures/model_evaluation/model_metrics_baseline_naive.csv")
    team = pd.read_csv("Figures/model_evaluation/model_metrics_baseline_team.csv")
    full = pd.read_csv("Figures/model_evaluation/model_metrics.csv")

    ml_models = sorted(naive["Model"].unique().tolist())

    # Average MAE across all transitions per ML model
    naive_avg = naive.groupby("Model")["MAE"].mean()
    team_avg = team.groupby("Model")["MAE"].mean()
    full_avg = full.groupby("Model")["MAE"].mean()

    baselines = {
        "Naive": naive_avg,
        "Player baseline": team_avg,
        "Full model": full_avg,
    }

    n_baselines = len(baselines)
    bar_height = 0.22
    y_pos = np.arange(len(ml_models))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (label, avg_series) in enumerate(baselines.items()):
        mae_vals = avg_series.reindex(ml_models).values
        naive_vals = naive_avg.reindex(ml_models).values
        offsets = y_pos + (i - n_baselines / 2 + 0.5) * bar_height

        bars = ax.barh(offsets, mae_vals, height=bar_height,
                       label=label, color=PLAYER_COLORS[label], edgecolor="white")

        for bar, mae, naive_mae in zip(bars, mae_vals, naive_vals):
            if label == "Naive":
                continue
            pct = _pct_diff(mae, naive_mae)
            sign = "+" if pct > 0 else ""
            ax.text(
                bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{sign}{pct:.1f}%", va="center", fontsize=8, fontweight="bold",
                color="green" if pct > 0 else "red",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ml_models, fontsize=10)
    ax.set_xlabel("MAE (avg. across transitions)", fontsize=12, fontweight="bold")
    ax.set_title("Player Model – Average MAE by Model Type", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/player_model_mae_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/player_model_mae_comparison.png")


def main():
    plot_team_mae()
    plot_player_mae()
