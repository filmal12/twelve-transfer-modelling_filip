import os
import sys
import joblib

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from misc.helpers import (
    clean_label,
    clean_feature_label,
    add_team_features,
    get_feature_names,
    get_model_importance,
    load_shap_importance,
    predict_safe
)

from misc.analysis import (
    _run_position_switch_analysis,
    analyze_top_shap_cooccurrence
)

from misc.plots import (
    plot_transition_diagram,
    plot_distribution_of_transition,
    plot_switch_analysis,
    plot_shap_features,
    plot_position_breakdown,
    plot_feature_similarity,
)

sys.path.append(os.path.abspath(".."))

from data_loader import load_data
from team_qualities import get_team_qualities
from setup import (
    WINGER_QUALITIES,
    STRIKER_QUALITIES,
    FB_QUALITIES,
    CENTRAL_DEFENDER_QUALITIES,
    MIDFIELDER_QUALITIES,
    TEAM_QUALS,
    IND_VARS
)

OUT_DIR = "Figures/prediction_analysis"

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

def run_analysis():
    os.makedirs(OUT_DIR, exist_ok=True)

    full_df = load_data()
    full_df = add_team_features(full_df)

    has_player_id = "wy_player_id" in full_df.columns

    # player_id {set of transitions predicted}
    player_transitions: dict[str, set] = {}
    # player_id from_position
    player_from_pos: dict[str, str] = {}
    # path_key n_players successfully predicted
    transition_player_counts: dict[str, int] = {}
    # path_key aggregated importance Series (top 10)
    transition_importances: dict[str, pd.Series] = {}

    for path_key, (from_pos, to_pos, targets) in PATH_CONFIG.items():
        df = full_df[
            (full_df["from_position"] == from_pos)
            & (full_df["to_position"] == to_pos)
        ].copy()

        if df.empty:
            print(f"[{path_key}] No data, skipping.")
            continue

        predicted_ids: set = set()
        all_importances: list[pd.Series] = []

        for target in targets:
            model = None
            model_type_used = None

            for mtype in MODEL_PRIORITY:
                mp = f"parameters/{path_key}/{target}_{mtype}.pkl"
                if os.path.exists(mp):
                    try:
                        model = joblib.load(mp)
                        model_type_used = mtype
                        break
                    except Exception as e:
                        print(f"  Failed loading {mp}: {e}")

            if model is None:
                print(f"  [{path_key}] No model found for target '{target}'")
                continue

            feature_names = get_feature_names(model)
            if not feature_names:
                print(f"  [{path_key}/{target}] Could not extract feature names.")
                continue

            try:
                predict_safe(model, df, feature_names)
                if has_player_id:
                    predicted_ids.update(df["wy_player_id"].dropna().unique())
            except Exception as e:
                print(f"  [{path_key}/{target}] Prediction error: {e}")
                continue

            # Prefer pre-saved SHAP importances (XGBoost); fall back to model.
            imp = load_shap_importance(path_key, target)
            if imp is None:
                imp = get_model_importance(model, feature_names)

            if imp is not None and not imp.empty:
                all_importances.append(imp)

        if all_importances:
            combined = (
                pd.concat(all_importances, axis=1)
                .fillna(0)
                .mean(axis=1)
                .sort_values(ascending=False)
                .head(10)
            )
            transition_importances[path_key] = combined

        n_pred = len(predicted_ids)
        transition_player_counts[path_key] = n_pred
        print(f"[{path_key}] {n_pred} players predicted.")
        
        for pid in predicted_ids:
            if pid not in player_transitions:
                player_transitions[pid] = set()
            player_transitions[pid].add(path_key)
            if pid not in player_from_pos:
                rows = df[df["wy_player_id"] == pid]
                if not rows.empty:
                    player_from_pos[pid] = from_pos

    # Players predicted per transition
    if transition_player_counts:
        tc_df = (
            pd.DataFrame(
                [{"transition": clean_label(k), "n_players": v}
                 for k, v in transition_player_counts.items()]
            )
            .sort_values("n_players", ascending=True)
        )
        plot_transition_diagram(tc_df, OUT_DIR)

    # Per source position to distribution of transition counts
    if player_transitions and has_player_id:
        counts_df = pd.DataFrame(
            [
                {
                    "wy_player_id": pid,
                    "n_transitions": len(trans),
                    "from_position": player_from_pos.get(pid, "Unknown"),
                }
                for pid, trans in player_transitions.items()
            ]
        )

        positions = sorted(counts_df["from_position"].unique())
        n_pos = len(positions)
        plot_distribution_of_transition(counts_df, positions, n_pos, OUT_DIR)

    # Position switch analysis (n samples per position)
    print("\n Running position switch analysis (1000 samples per position)")
    switch_results, switcher_shap, switcher_team_shap = _run_position_switch_analysis(full_df)

    if switch_results:
        sw_df = pd.DataFrame(switch_results)

        # Stay vs Switch per source position
        from_positions = sorted(sw_df["from_position"].unique())
        stay_counts = sw_df.groupby("from_position")["switched"].apply(
            lambda x: int((~x).sum())
        )
        switch_counts = sw_df.groupby("from_position")["switched"].apply(
            lambda x: int(x.sum())
        )

        x = np.arange(len(from_positions))
        width = 0.4
        plot_switch_analysis(x, from_positions, width, stay_counts, switch_counts, OUT_DIR)

        # Top-10 overall SHAP features among switchers
        if switcher_shap:
            combined_shap = (
                pd.concat(switcher_shap, axis=1)
                .fillna(0)
                .mean(axis=1)
                .sort_values(ascending=False)
                .head(10)
            )
            labels = [clean_feature_label(f) for f in combined_shap.index]
            colors = plt.cm.plasma_r(np.linspace(0.2, 0.8, len(combined_shap)))
            plot_shap_features(combined_shap, labels, colors, OUT_DIR, "top_10_switchers")

        # Team attribute influence on position switches
        if switcher_team_shap:
            _team_qual_names = {q.lower() for q in TEAM_FEATURE_QUALS}

            def _clean_team_label(feat):
                if feat.startswith("from_") and feat[5:] in _team_qual_names:
                    return "Team from: " + feat[5:].replace("_", " ").capitalize()
                if feat.startswith("to_") and feat[3:] in _team_qual_names:
                    return "Team to: " + feat[3:].replace("_", " ").capitalize()
                return feat.replace("_", " ").capitalize()

            combined_team_shap = (
                pd.concat(switcher_team_shap, axis=1)
                .fillna(0)
                .mean(axis=1)
                .sort_values(ascending=False)
            )
            labels_t = [_clean_team_label(f) for f in combined_team_shap.index]
            colors_t = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(combined_team_shap)))
            
            plot_shap_features(combined_team_shap, labels_t, colors_t, OUT_DIR, "team_influence")

    # Central Defender destination breakdown 
    if switch_results:
        sw_df = pd.DataFrame(switch_results)
        cd_df = sw_df[sw_df["from_position"] == "Central Defender"]
        if not cd_df.empty:
            dest_counts = cd_df["best_to_position"].value_counts()

            plot_position_breakdown(cd_df, dest_counts, OUT_DIR, "dest_breakdown")

    # Feature correlation CD vs FB players
    # Show how similar CDs and FBs are across IND_VARS, explaining why CD models
    # score well on FB quality targets.
    cd_players = full_df[full_df["from_position"] == "Central Defender"][IND_VARS].dropna()
    fb_players = full_df[full_df["from_position"] == "Full Back"][IND_VARS].dropna()
    winger_players = full_df[full_df["from_position"] == "Winger"][IND_VARS]

    if not cd_players.empty and not fb_players.empty:
        cd_mean = cd_players.mean().rename("Central Defender")
        fb_mean = fb_players.mean().rename("Full Back")
        winger_mean = winger_players.mean().rename("Winger")
        mf_players = full_df[full_df["from_position"] == "Midfielder"][IND_VARS].dropna()
        mf_mean = mf_players.mean().rename("Midfielder") if not mf_players.empty else None
        striker_players = full_df[full_df["from_position"] == "Striker"][IND_VARS].dropna()
        striker_mean = striker_players.mean().rename("Striker") if not striker_players.empty else None

        compare_df = pd.concat(
            [cd_mean, fb_mean] + ([mf_mean] if mf_mean is not None else []),
            axis=1,
        )
        compare_df.index = [clean_feature_label(f) for f in compare_df.index]

    
        compare_w_df = pd.concat(
            [fb_mean, winger_mean] + ([striker_mean] if striker_mean is not None else []),
            axis=1,
        )

        compare_w_df.index = [clean_feature_label(f) for f in compare_w_df.index]


        combined_pos = pd.concat(
            [
                cd_players.assign(_pos="Central Defender"),
                fb_players.assign(_pos="Full Back"),
            ] + ([mf_players.assign(_pos="Midfielder")] if mf_mean is not None else []),
            ignore_index=True,
        )
        corr = combined_pos[IND_VARS].corr()
        corr.index = [clean_feature_label(f) for f in corr.index]
        corr.columns = [clean_feature_label(f) for f in corr.columns]

        plot_feature_similarity(compare_df, cd_mean, fb_mean, "Central Defender", "Full Back", OUT_DIR, "cd_vs_fb_feature_similarity")

        # Scatter for Winger mean vs FB mean per feature (similarity plot)
        plot_feature_similarity(compare_df, fb_mean, winger_mean, "Full Back", "Winger", OUT_DIR, "fb_vs_winger_feature_similarity")

    # Top-5 SHAP co-occurrence per transition
    for path_key in PATH_CONFIG:
        try:
            analyze_top_shap_cooccurrence(path_key, full_df=full_df)
        except Exception as e:
            print(f"  [{path_key}] Co-occurrence analysis failed: {e}")

if __name__ == "__main__":
    run_analysis()
