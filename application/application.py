import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import requests
from io import BytesIO
from PIL import Image
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os

import plotly.graph_objects as go
from helper_function import (
    predict_player,
    get_predefined_description,
    display_top_features,
    normal_quals
)

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

transfer_data, competition_data = get_full_data()

allsvenskan_data = transfer_data[(transfer_data["from_season"] == 2025)].copy()

mock_players = {
    row["short_name"]: {"team": row["from_team_id"], "season": row["from_season"]}
    for _, row in allsvenskan_data.drop_duplicates(subset=["short_name", "from_season"]).iterrows()
}


def getTopFeatures(position):
    df_parsed = pd.DataFrame()

    pos_prefix = f"{POS_ABBREV[position].lower()}"

    for pos_to in POSITION_TRANSITIONS[position]:
        to_pos = POS_ABBREV[pos_to].lower()

        path = ""

        if pos_prefix == to_pos:
            path = f"../parameters/same_position/{position}/"
        else: 
            path = f"../parameters/{pos_prefix}_to_{to_pos}/"

        for quality in POSITION_QUALITIES[pos_to]:
            final_path = ""

            if TYPE_ANALYSIS == "OLS":
                final_path = f"{path}/{quality}.csv"
            else:
                final_path = f"{path}/{quality}_top_features.csv"

            df = pd.read_csv(final_path)

            df.drop(columns=["max", "min"], inplace=True, errors="ignore")

            df.rename(columns={"mean": "importance", "Factor": "feature"}, inplace=True)

            df = df[df["feature"] != "Intercept"].copy()

            df["To position"] = pos_to

            df["From position"] = position

            df_parsed = pd.concat([df_parsed, df], ignore_index=True)
    
    df_parsed = df_parsed.groupby(['feature', 'To position', 'From position'], as_index=False)['importance'].mean()
    return df_parsed.loc[df_parsed.groupby('feature')['importance'].idxmax()]


def _display_table_name(feature):
        if feature in normal_quals:
            return feature.replace('from_', 'Team from: ').replace('to_', 'Team to: ').replace('_', ' ').title()
        
        return feature.replace('from', 'Player: ').replace('_', ' ').title()

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
    # col_description, empty = st.columns(2)
    # with col_description:

    desc = get_predefined_description(position)

    st.text(desc)

    if st.button("Refresh", key="overview"):
        col_figure, col_description = st.columns(2)

        with col_figure:

            parsedDf = getTopFeatures(position)
            print(parsedDf)
            parsedDf = parsedDf[parsedDf["feature"].isin(ALL_QUALITIES)]

            fig = display_top_features(parsedDf, position)
            if fig:
                st.pyplot(fig)
            else:
                st.warning("No image loaded")