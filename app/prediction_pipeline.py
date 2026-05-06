import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os
import sys
import matplotlib.pyplot as plt

from app.descriptors import generate_player_transition_description

from app.helper_function import (
    display_position_change,
    create_top_features_radar
)

sys.path.append(os.path.abspath(".."))
from setup import (
    POS_ABBREV,
    POSITION_QUALITIES,
    TEAM_QUALS,
    TEAM_QUALITIES,
    POSITIONAL_CHANGES,
    POSITION_TRANSITIONS
)

from helper_function import prepare_player_df

MODEL_TYPE = "XGBOOST"

################################
# Runpoint for predicting the player
# runs the entire pipeline for predicting 
# a selected player
################################

def predict_player(player_name, season, df_full, competition_data):
    player_row, X_player_df, z_score_cols = prepare_player_df(df_full, player_name, season)

    if (player_row is None) or (X_player_df is None) or (z_score_cols is None):
        st.write(f"Not enoough data for the player exists {player_name}")

    # Set competition dummy columns.
    # to_competition is always Swedish first division (the target league).
    # from_competition is taken from the player's actual competition.
    _TO_COMPETITION = "Swedish first division"

    from_pos = player_row["from_position"]

    if from_pos not in POSITION_TRANSITIONS:
        return {}

    quals = POSITION_QUALITIES[from_pos]
    to_positions = POSITION_TRANSITIONS[from_pos]

    def _sanitize_comp(name):
        return re.sub(r"[^a-zA-Z0-9_]", "_", str(name))

    X_player_df[f"to_competition_name_{_sanitize_comp(_TO_COMPETITION)}"] = 1.0

    from_comp_name = player_row.get("from_competition_name", None)
    if from_comp_name is not None and pd.notna(from_comp_name):
        X_player_df[f"from_competition_name_{_sanitize_comp(from_comp_name)}"] = 1.0

    # Predict each transition
    transition_scores = {}
    transition_targets = {}
    transition_team_impr = {}
    average = {}

    positions = ["Full Back", "Central Defender", "Winger", "Midfielder", "Striker", "Goalkeeper"]

    for to_pos in to_positions:
        path = f"{POS_ABBREV[from_pos]}_to_{POS_ABBREV[to_pos]}"
        target_quals = POSITION_QUALITIES[to_pos]
        team_tgts = TEAM_QUALS

        pos_scores = {}
        for target in target_quals:
            model_path = f"parameters/{path}/{target}_xgboost.pkl"
            if MODEL_TYPE == "OLS":
                model_path = f"parameters/{path}/{target}.csv"
        
            if os.path.exists(model_path):
                try:
                    if MODEL_TYPE != "OLS":
                        xgb_model = joblib.load(model_path)
                        feature_names = xgb_model.get_booster().feature_names
                        
                        if feature_names is None:
                            feature_names = [f"f{i}" for i in range(xgb_model.n_features_in_)]

                        player_df_feat = X_player_df.reindex(columns=feature_names, fill_value=0).fillna(0)
                        print(player_df_feat.dtypes)
                        pred = xgb_model.predict(player_df_feat)[0]
                        print(f"PREDAJSDASD: {pred}")
                        pos_scores[target] = pred
                    else:
                        df = pd.read_csv(model_path)
                        pred = 0

                        for _ ,row in df.iterrows():
                            
                            feature = row["Factor"]                      

                            weight = row["mean"]
                            
                            if feature == "Intercept":
                                pred += weight
                                continue
                            quality_rating = X_player_df[feature].iloc[0] if feature in X_player_df.columns else 0.0
                            pred += weight * quality_rating

                        pos_scores[target] = pred
                except Exception as e:
                    st.warning(f"Error loading model {path}/{target}: {e}")
            print(pos_scores)
        valid_scores = {k: v for k, v in pos_scores.items() if v is not None}
        average[to_pos] = np.mean(list(valid_scores.values()))

        if valid_scores:
            transition_targets[to_pos] = max(valid_scores, key=valid_scores.get)
            transition_scores[to_pos] = valid_scores[transition_targets[to_pos]]
        else:
            transition_scores[to_pos] = np.nan
    # Stay in current position score
    current_quality_values = {}
    qualities_same = POSITION_QUALITIES[from_pos]
    for qual in qualities_same:
        path = f"parameters/same_position/{from_pos}/{qual}_xgboost.pkl"
        if MODEL_TYPE == "OLS":
            path = f"parameters/same_position/{from_pos}/{qual}.csv"

        if os.path.exists(path):
            try:
                if MODEL_TYPE != "OLS":
                    model = joblib.load(path)
                    feature_names = model.get_booster().feature_names
                    if feature_names is None:
                        feature_names = [f"f{i}" for i in range(model.n_features_in_)]
                    player_df_feat = X_player_df.reindex(columns=feature_names, fill_value=0).fillna(0)
                    pred = model.predict(player_df_feat)[0]
                    current_quality_values[qual] = pred
                else:
                    df = pd.read_csv(path)
                    pred = 0

                    for _ ,row in df.iterrows():
                        
                        feature = row["Factor"]                      

                        weight = row["mean"]
                        
                        if feature == "Intercept":
                            pred += weight
                            continue
                        quality_rating = X_player_df[feature].iloc[0] if feature in X_player_df.columns else 0.0
                        pred += weight * quality_rating

                    current_quality_values[qual] = pred
            except Exception as e:
                st.warning(f"Error loading same position model for {qual}: {e}")

    average[from_pos] = np.mean(list(current_quality_values.values()))

    valid_scores = {k: v for k, v in current_quality_values.items() if v is not None}
    if valid_scores:
        transition_targets[from_pos] = max(valid_scores, key=valid_scores.get)
        transition_scores[from_pos] = valid_scores[transition_targets[from_pos]]
    else:
        transition_scores[from_pos] = np.nan

    all_positions = to_positions + [from_pos]
    # Make recommendation
    positions = {to_pos: average[to_pos] for to_pos in all_positions}
    best_position = max(positions, key=lambda k: positions[k] if pd.notna(positions[k]) else -np.inf)
    best_score = positions[best_position]  

    pos_changes_cols = POSITIONAL_CHANGES.copy()
    
    X_player_df[pos_changes_cols] = 0

    row_pos_changes = from_pos + "-" + best_position

    X_player_df[row_pos_changes] = 1

    z_score_cols = z_score_cols + pos_changes_cols

    # Team improvement predictions
    team_impr = {}
    for target in TEAM_QUALS:
        model_path = f"team_models/delta_{target}_xgboost.pkl"
        if os.path.exists(model_path):
            try:
                xgb_model = joblib.load(model_path)
                feature_names = xgb_model.get_booster().feature_names
                if feature_names is None:
                    feature_names = [f"f{i}" for i in range(xgb_model.n_features_in_)]
                X_player_df = X_player_df.convert_dtypes()
                player_df = X_player_df[feature_names].fillna(0)
                pred = xgb_model.predict(player_df)[0]

                team_impr[target] = pred
            except Exception as e:
                print(f"Error loading team model for {target}: {e}")
                team_impr[target] = None
    transition_team_impr = team_impr


    # Display results
    st.subheader(f"Player success analysis")
    st.write(f"**Current position:** {from_pos}")
    # st.write(f"**Recommended position:** {best_position} (score: {best_score:.4f})")
    print(transition_targets)
    scores_df = pd.DataFrame([
        {"Position": f"{pos} - {transition_targets[pos].replace("_", " ")}", "Score": f"{score:.4f}" if pd.notna(score) else "N/A"}
        for pos, score in sorted(positions.items(), key=lambda x: x[1] if pd.notna(x[1]) else -1, reverse=True)
    ])
    # st.table(scores_df)

    # Radar plot for best transition
    position_prefix = None
    target_name = None
    if best_position in transition_targets:
        position_prefix = f"{POS_ABBREV[from_pos]}_to_{POS_ABBREV[best_position]}"
        target_name = transition_targets[best_position]


    col1, col2 = st.columns(2, vertical_alignment="top")
    with col1:
        non_conclusive = all(score > 0.5 for score in positions.values())

        has_second_position = False
        second_position = None
        second_position_target = None

        if not non_conclusive:
            second_position = max((p for p in all_positions if p != best_position), key=lambda k: positions[k] if pd.notna(positions[k]) else -np.inf)
            second_position_target = transition_targets.get(second_position, "N/A").replace("_", " ")

            if transition_scores[second_position] >= 0.5:

                has_second_position = True

        description = generate_player_transition_description(player_name, from_pos, best_position, df_full, target_name, non_conclusive, has_second_position, second_position, second_position_target)
        st.markdown(f"### Transition Analysis")
        st.markdown(description)    

        other_position = None
        if has_second_position:
            other_position = second_position
        fig = display_position_change(from_pos, best_position, best_position, transition_scores[best_position], second_position = other_position)
        if fig:
            st.pyplot(fig)
        else:
            st.warning("No image loaded")
    with col2:
        st.markdown("### Position Suitability Scores")

        fig = create_top_features_radar(
            player_row, player_name, best_position, best_score,
            position_prefix, target_name, df_full, from_pos,
            figsize=(12, 12), font_scale=0.85,
        )

        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No feature radar available for this transition.")

    position_qualities_map = {to_pos: POSITION_QUALITIES[to_pos] for to_pos in to_positions}
    position_qualities_map[from_pos] = quals
    
    st.divider()
    st.title("Players effect on team")

    to_team_id = 6710
    has_team_col = competition_data is not None and "team_stats_team_id" in competition_data.columns

    n_cols = max(len(transition_team_impr), 1)
    cols = st.columns(3)

    quality_dict = transition_team_impr

    if not quality_dict or competition_data is None or competition_data.empty:
        st.info("No team data available.")
        return
    
    valid_quals = [
        (k, v) for k, v in quality_dict.items()
        if k.lower() in competition_data.columns and v is not None
    ]



    for col, quality_names_matches in zip(cols, TEAM_QUALITIES.values()):
        with col:

            group_quals = [
                (k, v) for k, v in valid_quals
                if k.upper() in quality_names_matches
            ]

            if len(group_quals) < 2:
                continue

            x_qual, x_pred = group_quals[0]
            y_qual, y_pred = group_quals[1]

            BG   = "#0e1117"
            GRID = "#2a2d3a"
            TEXT = "#e0e0e0"

            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)

            ax.scatter(
                competition_data[x_qual.lower()], competition_data[y_qual.lower()], 
                color="#42a5f5", alpha=0.55, s=35, zorder=3, label="Teams",
            )

            # Highlight the player's current team and draw predicted-improvement arrow
            if to_team_id is not None and has_team_col:
                from_team_row = competition_data[
                    competition_data["team_stats_team_id"] == to_team_id
                ]
                if not from_team_row.empty:
                    tx = float(from_team_row[x_qual.lower()].iloc[0])
                    ty = float(from_team_row[y_qual.lower()].iloc[0])
                    ax.scatter(tx, ty, color="#ffcc00", s=70, zorder=5, label="Current team")
                    ax.annotate(
                        "",
                        xy=(tx + x_pred, ty + y_pred),
                        xytext=(tx, ty),
                        arrowprops=dict(arrowstyle="->", color="#ef5350", lw=2),
                        zorder=6,
                    )
                    ax.scatter(
                        tx + x_pred, ty + y_pred,
                        color="#ef5350", s=80, marker="*", zorder=7, label="Predicted",
                    )
            else:
                print("NONE VALUES")

            ax.set_xlabel(x_qual.lower().replace("_", " ").title(), fontsize=9, color=TEXT)
            ax.set_ylabel(y_qual.lower().replace("_", " ").title(), fontsize=9, color=TEXT)
            ax.tick_params(colors=TEXT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            ax.grid(alpha=0.2, color=GRID, linestyle="--")
            ax.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)    