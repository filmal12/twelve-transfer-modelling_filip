import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from data_loader import load_data
from setup import (
    IND_VARS,
    POSITION_QUALITIES,
    POSITION_TRANSITIONS,
    TEAM_QUALS,
)

THRESHOLDS = [0, 200, 400, 600, 800, 1000, 1200, 1500, 2000]
OUT_DIR = "Figures/sensitivity"


def run_sensitivity():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data once (without minute filtering)
    base_df = load_data()

    results = []

    for min_minutes in THRESHOLDS:
        df = base_df.copy()
        if min_minutes > 0:
            df = df[(df["from_Minutes"] > min_minutes) & (df["to_Minutes"] > min_minutes)]

        total_before = len(base_df)
        total_after = len(df)
        print(f"\n=== Threshold: {min_minutes} min  |  {total_after}/{total_before} rows kept ===")

        for from_pos, to_positions in POSITION_TRANSITIONS.items():
            targets = POSITION_QUALITIES.get(from_pos, [])
            if not targets:
                continue

            for to_pos in to_positions:
                subset = df[
                    (df["from_position"] == from_pos) & (df["to_position"] == to_pos)
                ]

                for target in targets:
                    target_col = f"from_{target}"
                    target_to_col = f"to_{target}"
                    feat_cols = [c for c in IND_VARS if c != target_col]

                    cols_needed = feat_cols + [target_to_col]
                    clean = subset[cols_needed].dropna()

                    if len(clean) < 10:
                        continue

                    X = clean[feat_cols].values
                    y = clean[target_to_col].values

                    model = XGBRegressor(
                        n_estimators=100, max_depth=4,
                        learning_rate=0.1, random_state=42,
                    )
                    cv = min(5, len(clean))
                    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
                    mae_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")

                    results.append({
                        "min_minutes": min_minutes,
                        "from_pos": from_pos,
                        "to_pos": to_pos,
                        "target": target,
                        "mean_r2": r2_scores.mean(),
                        "std_r2": r2_scores.std(),
                        "mean_mae": -mae_scores.mean(),
                        "n_samples": len(clean),
                    })

        print(f"  Collected {sum(1 for r in results if r['min_minutes'] == min_minutes)} results")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "sensitivity_results.csv"), index=False)
    print(f"\nResults saved to {OUT_DIR}/sensitivity_results.csv")

    # --- Aggregate plot: threshold vs mean R² + sample count ---
    agg = res_df.groupby("min_minutes").agg(
        mean_r2=("mean_r2", "mean"),
        mean_mae=("mean_mae", "mean"),
        n_samples=("n_samples", "sum"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(agg["min_minutes"], agg["mean_r2"], "o-", color="steelblue", linewidth=2, label="Mean R²")
    ax1.set_xlabel("Minimum Minutes Threshold", fontweight="bold")
    ax1.set_ylabel("Mean R² (5-fold CV)", color="steelblue", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.bar(agg["min_minutes"], agg["n_samples"], alpha=0.25, color="orange", width=80, label="Total samples")
    ax2.set_ylabel("Total samples", color="orange", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="orange")

    fig.suptitle("Sensitivity: Minutes Threshold vs Model Performance", fontweight="bold")
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.88))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "minutes_sensitivity_r2.png"), dpi=220)
    plt.close(fig)

    # --- Same for MAE ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(agg["min_minutes"], agg["mean_mae"], "s-", color="crimson", linewidth=2, label="Mean MAE")
    ax1.set_xlabel("Minimum Minutes Threshold", fontweight="bold")
    ax1.set_ylabel("Mean MAE (5-fold CV)", color="crimson", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="crimson")

    ax2 = ax1.twinx()
    ax2.bar(agg["min_minutes"], agg["n_samples"], alpha=0.25, color="orange", width=80, label="Total samples")
    ax2.set_ylabel("Total samples", color="orange", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="orange")

    fig.suptitle("Sensitivity: Minutes Threshold vs MAE", fontweight="bold")
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.88))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "minutes_sensitivity_mae.png"), dpi=220)
    plt.close(fig)

    # --- Per-transition breakdown ---
    transitions = res_df.groupby(["min_minutes", "from_pos", "to_pos"]).agg(
        mean_r2=("mean_r2", "mean"),
        n_samples=("n_samples", "sum"),
    ).reset_index()
    transitions["transition"] = transitions["from_pos"] + " -> " + transitions["to_pos"]

    unique_transitions = sorted(transitions["transition"].unique())
    n_trans = len(unique_transitions)
    ncols = 3
    nrows = (n_trans + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True)
    axes = axes.flatten()

    for i, trans_label in enumerate(unique_transitions):
        ax = axes[i]
        t_df = transitions[transitions["transition"] == trans_label]
        ax.plot(t_df["min_minutes"], t_df["mean_r2"], "o-", color="steelblue")
        ax.set_title(trans_label, fontweight="bold", fontsize=10)
        ax.set_ylabel("Mean R²")
        ax.grid(alpha=0.3, linestyle="--")

        ax_r = ax.twinx()
        ax_r.bar(t_df["min_minutes"], t_df["n_samples"], alpha=0.2, color="orange", width=80)
        ax_r.set_ylabel("n", fontsize=8, color="orange")
        ax_r.tick_params(axis="y", labelcolor="orange", labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Transition Sensitivity to Minutes Threshold", fontweight="bold", fontsize=13)
    fig.supxlabel("Minimum Minutes", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "minutes_sensitivity_per_transition.png"), dpi=200)
    plt.close(fig)

    # --- Print summary table ---
    print("\n=== Summary: Mean R² by threshold ===")
    print(agg[["min_minutes", "mean_r2", "mean_mae", "n_samples"]].to_string(index=False))

    return res_df


if __name__ == "__main__":
    run_sensitivity()
