import os
import sys
import joblib

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import (
    clean_label,
    clean_feature_label,
    add_team_features,
    get_feature_names,
    load_shap_importance,
    predict_safe
)

sys.path.append(os.path.abspath(".."))

from data_loader import load_data
from setup import (
    WINGER_QUALITIES,
    STRIKER_QUALITIES,
    FB_QUALITIES,
    CENTRAL_DEFENDER_QUALITIES,
    MIDFIELDER_QUALITIES,
    TEAM_QUALS,
)

PATH_CONFIG = {
    "winger_to_st": ("Winger", "Striker", STRIKER_QUALITIES),
    "winger_to_fb": ("Winger", "Full Back", FB_QUALITIES),
    "same_position/Winger": ("Winger", "Winger", WINGER_QUALITIES),
    "fb_to_winger": ("Full Back", "Winger", WINGER_QUALITIES),
    "fb_to_cd": ("Full Back", "Central Defender", CENTRAL_DEFENDER_QUALITIES),
    "same_position/Full Back": ("Full Back", "Full Back", FB_QUALITIES),
    "cd_to_fb": ("Central Defender", "Full Back", FB_QUALITIES),
    "cd_to_mf": ("Central Defender", "Midfielder", MIDFIELDER_QUALITIES),
    "mf_to_cd": ("Midfielder", "Central Defender", CENTRAL_DEFENDER_QUALITIES),
    "mf_to_st": ("Midfielder", "Striker", STRIKER_QUALITIES),
    "same_position/Midfielder": ("Midfielder", "Midfielder", MIDFIELDER_QUALITIES),
    "same_position/Central Defender": ("Central Defender", "Central Defender", CENTRAL_DEFENDER_QUALITIES),
    "st_to_winger": ("Striker", "Winger", WINGER_QUALITIES),
    "st_to_mf": ("Striker", "Midfielder", MIDFIELDER_QUALITIES),
    "same_position/Striker":("Striker", "Striker", STRIKER_QUALITIES)
}

# Model types tried in priority order.
MODEL_PRIORITY = ["xgboost"]

TEAM_FEATURE_QUALS = TEAM_QUALS  # from setup.py

OUT_DIR = "../Figures/prediction_analysis"

def _run_position_switch_analysis(
    full_df: pd.DataFrame,
    n_samples: int = 1000,
) -> tuple[list[dict], list[pd.Series], list[pd.Series]]:
    rng = np.random.default_rng(42)

    # Group path_keys by from_position
    from_pos_paths: dict[str, list[str]] = {}
    for path_key, (from_pos, _to, _tgts) in PATH_CONFIG.items():
        from_pos_paths.setdefault(from_pos, []).append(path_key)

    _team_qual_names = {q.lower() for q in TEAM_FEATURE_QUALS}

    switch_results: list[dict] = []
    switcher_shap: list[pd.Series] = []
    switcher_team_shap: list[pd.Series] = []

    for from_pos, path_keys in from_pos_paths.items():
        pos_df = full_df[full_df["from_position"] == from_pos].copy()
        if pos_df.empty:
            print(f"  [switch] No data for {from_pos}, skipping.")
            continue

        idx = rng.choice(len(pos_df), size=n_samples, replace=True)
        sample_df = pos_df.iloc[idx].reset_index(drop=True)

        path_max_preds: dict[str, np.ndarray] = {}
        path_average_preds: dict[str, np.ndarray] = {}
        path_agg_shap: dict[str, pd.Series] = {}

        for path_key in path_keys:
            _fp, _tp, targets = PATH_CONFIG[path_key]
            per_target_preds: list[np.ndarray] = []
            shap_imps: list[pd.Series] = []

            for target in targets:
                model = None
                for mtype in MODEL_PRIORITY:
                    mp = f"parameters/{path_key}/{target}_{mtype}.pkl"
                    if os.path.exists(mp):
                        try:
                            model = joblib.load(mp)
                            break
                        except Exception:
                            pass

                if model is None:
                    continue

                feature_names = get_feature_names(model)
                if not feature_names:
                    continue

                try:
                    preds = predict_safe(model, sample_df, feature_names)
                    per_target_preds.append(preds)
                except Exception as e:
                    print(f"  [{path_key}/{target}] Prediction error: {e}")
                    continue

                imp = load_shap_importance(path_key, target)
                if imp is not None:
                    shap_imps.append(imp)

            if per_target_preds:
                path_max_preds[path_key] = np.max(
                    np.stack(per_target_preds, axis=1), axis=1
                )

                path_average_preds[path_key] = np.mean(
                    np.stack(per_target_preds, axis=1), axis=1
                )
            if shap_imps:
                path_agg_shap[path_key] = (
                    pd.concat(shap_imps, axis=1).fillna(0).mean(axis=1)
                )

        if not path_max_preds:
            continue

        pred_df = pd.DataFrame(path_max_preds)  # (n_samples, n_paths)

        pred_average_df = pd.DataFrame(path_average_preds)

        # The same-position path key for this from_pos
        same_key = next(
            (k for k in path_keys if k.startswith("same_position/")), None
        )

        for i in range(len(pred_average_df)):
            row = pred_average_df.iloc[i]
            best_key = row.idxmax()
            best_to_pos = PATH_CONFIG[best_key][1]
            switched = same_key is None or best_key != same_key
            if from_pos == "Central Defender":
                print(f"Central defender {from_pos} - {best_to_pos}. switched: {switched}")
            switch_results.append(
                {
                    "from_position": from_pos,
                    "best_to_position": best_to_pos,
                    "switched": switched,
                }
            )

            if switched and best_key in path_agg_shap:
                full_shap = path_agg_shap[best_key]
                # Top-5 overall features
                top5 = full_shap.nlargest(5)
                switcher_shap.append(top5)
                # Team features only
                team_feats = full_shap[
                    full_shap.index.map(
                        lambda f: (
                            (f.startswith("from_") and f[5:] in _team_qual_names)
                            or (f.startswith("to_") and f[3:] in _team_qual_names)
                        )
                    )
                ]
                if not team_feats.empty:
                    switcher_team_shap.append(team_feats)

        n_switch = sum(r["switched"] for r in switch_results if r["from_position"] == from_pos)
        n_switch_avg = sum(r["switched"] for r in switch_results if r["from_position"] == from_pos)

    return switch_results, switcher_shap, switcher_team_shap

def analyze_top_shap_cooccurrence(
    path_key: str,
    n_top: int = 5,
    high_percentile: float = 75,
    full_df: pd.DataFrame | None = None,
):
    if path_key not in PATH_CONFIG:
        raise ValueError(f"Unknown path_key '{path_key}'. Choose from: {list(PATH_CONFIG)}")

    from_pos, to_pos, targets = PATH_CONFIG[path_key]

    shap_imps: list[pd.Series] = []
    for target in targets:
        imp = load_shap_importance(path_key, target)
        if imp is not None:
            shap_imps.append(imp)

    if not shap_imps:
        print(f"[{path_key}] No SHAP importances found.")
        return

    combined = (
        pd.concat(shap_imps, axis=1)
        .fillna(0)
        .mean(axis=1)
        .abs()
        .sort_values(ascending=False)
    )
    top_features = combined.head(n_top).index.tolist()
    print(f"[{path_key}] Top-{n_top} SHAP features: {top_features}")

    if full_df is None:
        full_df = load_data()
        full_df = add_team_features(full_df)

    df = full_df[
        (full_df["from_position"] == from_pos)
        & (full_df["to_position"] == to_pos)
    ].copy()

    # Keep only columns that exist
    available = [f for f in top_features if f in df.columns]
    if len(available) < 2:
        print(f"[{path_key}] Not enough top features present in data ({available}).")
        return

    feat_df = df[available].dropna()

    cooccurrence = pd.DataFrame(index=available, columns=available, dtype=float)

    for feat in available:
        threshold = np.percentile(feat_df[feat], high_percentile)
        high_mask = feat_df[feat] >= threshold
        high_means = feat_df.loc[high_mask, available].mean()
        cooccurrence.loc[feat] = high_means

    overall_means = feat_df[available].mean()

    # Normalise: difference from overall mean (how much higher/lower)
    cooccurrence_diff = cooccurrence.subtract(overall_means, axis=1)

    out_dir = os.path.join(OUT_DIR, "shap_cooccurrence")
    os.makedirs(out_dir, exist_ok=True)

    clean_labels = [clean_feature_label(f) for f in available]

    fig, ax = plt.subplots(figsize=(max(8, len(available) * 1.6), max(6, len(available) * 1.2)))
    sns.heatmap(
        cooccurrence_diff.values.astype(float),
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        linewidths=0.5,
        linecolor="lightgrey",
        ax=ax,
    )
    ax.set_ylabel("Feature with high value (≥ p75)", fontweight="bold")
    ax.set_xlabel("Other feature (mean difference from population)", fontweight="bold")
    ax.set_title(
        f"Feature Co-occurrence: {clean_label(from_pos)} → {clean_label(to_pos)}\n"
        f"When one top-SHAP feature is high, how do others deviate?",
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    pltKey = path_key.replace("/", "_")
    plt.savefig(os.path.join(out_dir, f"{pltKey}_cooccurrence_heatmap.png"), dpi=220)
    plt.close(fig)

    n_feats = len(available)
    fig, axes = plt.subplots(
        1, n_feats, figsize=(5 * n_feats, 5), squeeze=False, sharey=True,
    )

    for i, feat in enumerate(available):
        ax = axes[0][i]
        others = [f for f in available if f != feat]
        if not others:
            continue

        threshold = np.percentile(feat_df[feat], high_percentile)
        high_mask = feat_df[feat] >= threshold

        high_means = feat_df.loc[high_mask, others].mean()
        pop_means = feat_df[others].mean()

        x = np.arange(len(others))
        width = 0.35
        other_labels = [clean_feature_label(f) for f in others]

        ax.bar(x - width / 2, pop_means.values, width, label="All players", color="steelblue", alpha=0.8)
        ax.bar(x + width / 2, high_means.values, width, label=f"High {clean_feature_label(feat)}", color="darkorange", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(other_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"When {clean_feature_label(feat)} is high", fontweight="bold", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if i == 0:
            ax.set_ylabel("Mean feature value", fontweight="bold")

    fig.suptitle(
        f"Top-{n_top} SHAP Feature Co-occurrence: {clean_label(from_pos)} → {clean_label(to_pos)}",
        fontweight="bold",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{pltKey}_cooccurrence_bars.png"), dpi=220)
    plt.close(fig)
    
    return cooccurrence_diff
