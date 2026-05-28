import streamlit as st
import pandas as pd
import matplotlib
import numpy as np
import joblib

matplotlib.use("Agg")
import os

from app.prediction_pipeline import predict_player

from app.helper_function import (
    display_top_features,
)

from app.descriptors import get_predefined_description

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from team_qualities import get_team_qualities

from setup import (
    mock_teams_to,
    TEAM_QUALS,
    POSITION_TRANSITIONS,
    POS_ABBREV,
    POSITION_QUALITIES,
    ALL_QUALITIES
)

from data_loader import (
    load_data,
    get_team_stats
)

st.set_page_config(page_title="Transfer Modelling", layout="wide")

TYPE_ANALYSIS = "XGBOOST"

################################
# Main entry point of the application
# Using models and parameters taken from training phase 
# Provides better runtime and quicker performance
# Otherwise fallsback to training a quick model using the
# type that is defined in type analysis
################################

# ========== DARK MODE STYLING ==========
@st.cache_data
def get_full_data():
    df_full = load_data().copy()

    df_from_full = df_full.copy()

    
    df_to, competition_df = get_team_stats(6710, 2025) # Hammarbys stats for 2025 season
    
    for quality_name in TEAM_QUALS:
        df_from_full = get_team_qualities(quality_name, df_from_full, prefix="from_")
        
        quality_col = quality_name.lower()

        df_full[f"from_{quality_col}"] = df_from_full[quality_col]
        df_full[f"to_{quality_col}"] = df_to[quality_col].values[0]

    return (df_full, competition_df)

def main():
    transfer_data, competition_data = get_full_data()
    print(competition_data)

    allsvenskan_data = transfer_data[(transfer_data["from_season"] == 2025)].copy()

    mock_players = {
        row["short_name"]: {"team": row["from_team_id"], "season": row["from_season"]}
        for _, row in allsvenskan_data.drop_duplicates(subset=["short_name", "from_season"]).iterrows()
    }


    # def getTopFeatures(position):
    #     df_parsed = pd.DataFrame()

    #     pos_prefix = f"{POS_ABBREV[position].lower()}"

    #     for pos_to in POSITION_TRANSITIONS[position]:
    #         to_pos = POS_ABBREV[pos_to].lower()

    #         path = ""

    #         if pos_prefix == to_pos:
    #             path = f"parameters/same_position/{position}/"
    #         else: 
    #             path = f"parameters/{pos_prefix}_to_{to_pos}/"

    #         for quality in POSITION_QUALITIES[pos_to]:
    #             final_path = ""

    #             if TYPE_ANALYSIS == "OLS":
    #                 final_path = f"{path}/{quality}.csv"
    #             else:
    #                 final_path = f"{path}/{quality}_top_features.csv"

    #             df = pd.read_csv(final_path)

    #             df.drop(columns=["max", "min"], inplace=True, errors="ignore")

    #             df.rename(columns={"mean": "importance", "Factor": "feature"}, inplace=True)

    #             df = df[df["feature"] != "Intercept"].copy()

    #             df["To position"] = pos_to

    #             df["From position"] = position

    #             df_parsed = pd.concat([df_parsed, df], ignore_index=True)
        
    #     df_parsed = df_parsed.groupby(['feature', 'To position', 'From position'], as_index=False)['importance'].mean()
    #     return df_parsed.loc[df_parsed.groupby('feature')['importance'].idxmax()]

    def getTopFeatures(position):

        df_parsed = pd.DataFrame()
        pos_prefix = POS_ABBREV[position].lower()

        for pos_to in POSITION_TRANSITIONS[position]:
            to_pos = POS_ABBREV[pos_to].lower()

            if pos_prefix == to_pos:
                path = f"parameters/same_position/{position}"
            else:
                path = f"parameters/{pos_prefix}_to_{to_pos}"

            for quality in POSITION_QUALITIES[pos_to]:
                shap_path = f"{path}/{quality}_xgboost_shap_values.npy"
                model_path = f"{path}/{quality}_xgboost.pkl"

                if not os.path.exists(shap_path) or not os.path.exists(model_path):
                    continue

                shap_values = np.load(shap_path)           # (n_samples, n_features)
                model = joblib.load(model_path)
                feature_names = model.get_booster().feature_names

                # Mean absolute SHAP = robust importance (no cancellation)
                abs_importance = np.abs(shap_values).mean(axis=0)
                # Signed mean SHAP = directional effect
                signed_importance = shap_values.mean(axis=0)
                # Std of SHAP = how variable the feature's effect is across players
                shap_std = shap_values.std(axis=0)

                df = pd.DataFrame({
                    "feature": feature_names,
                    "abs_importance": abs_importance,
                    "importance": signed_importance,
                    "shap_std": shap_std,
                    "quality": quality,
                    "To position": pos_to,
                    "From position": position,
                })

                df_parsed = pd.concat([df_parsed, df], ignore_index=True)

        # Aggregate per feature per target position:
        # - abs_importance for ranking (avoids sign cancellation)
        # - signed importance for interpretation
        # - mean std for variability signal
        # - quality_count = how many qualities the feature impacts
        agg = df_parsed.groupby(["feature", "To position", "From position"]).agg(
            mean_abs_importance=("abs_importance", "mean"),
            mean_importance=("importance", "mean"),
            mean_shap_std=("shap_std", "mean"),
            quality_count=("quality", "nunique"),
        ).reset_index()

        # For each feature, pick the target position where it matters most
        return agg.loc[agg.groupby("feature")["mean_abs_importance"].idxmax()]

    st.title("Transfer Modelling")

    tab_player, tab_positional_overview = st.tabs(["Player Analysis", "Positional overview"])

    with tab_player:
        player = st.selectbox("Player", list(mock_players.keys()))
        team_from = mock_players[player]["team"]
        st.write(f"From Team: {team_from} (Season {mock_players[player]['season']})")
        team_to = st.selectbox("To Team", mock_teams_to)

        if st.button("Refresh"):
            predict_player(player, mock_players[player]["season"], transfer_data, competition_data)

    with tab_positional_overview:
        all_position = list(POSITION_TRANSITIONS.keys())

        position = st.selectbox("Position", all_position, key="pos")    

        if st.button("Refresh", key="overview"):
            col_figure, col_description = st.columns(2)

            with col_figure:

                parsedDf = getTopFeatures(position)

                parsedDf = parsedDf[parsedDf["feature"].isin(ALL_QUALITIES)]

                fig = display_top_features(parsedDf, position)
                if fig:
                    st.pyplot(fig)
                else:
                    st.warning("No image loaded")

            with col_description:
                
                desc = get_predefined_description(position)

                st.text(desc)
if __name__ == "__main__":
    main()