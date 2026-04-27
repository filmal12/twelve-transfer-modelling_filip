"""
Training orchestration script for position transition models.
This centralizes all model training (OLS, XGBoost, Ridge, Lasso, Random Forest).
"""

import pandas as pd
import numpy as np
from positional_model import (
    train_winger_to_striker,
    train_winger_to_fb,
    train_fb_to_cd,
    train_fb_to_winger,
    train_cd_to_fb,
    train_cd_to_mf,
    train_st_to_winger,
    train_st_to_mf,
    train_midfielders,
    train_striker_to_am,
    train_defender_to_dm,
    train_same_position,
    train_mf_to_st,
    train_mf_to_cd
)

import sys
import os

sys.path.append(os.path.abspath(".."))

from data_loader import (
    get_data,
    get_mf,
    load_data,
)

from setup import (
    WINGER_QUALITIES,
    STRIKER_QUALITIES,
    FB_QUALITIES,
    CENTRAL_DEFENDER_QUALITIES,
    MIDFIELDER_QUALITIES,
)

# ========== TRAINING FLAGS ==========
TRAIN_WINGER_STRIKER = True
TRAIN_WINGER_FB = True
TRAIN_FB_CD = True
TRAIN_FB_WINGER = True
TRAIN_CD_FB = True
TRAIN_CD_MF = True
TRAIN_ST_WINGER = True
TRAIN_ST_MF = True
TRAIN_MIDFIELDER_STRIKER = True
TRAIN_MIDFIELDER_CD = True

TRAIN_ST_ST = True
TRAIN_WINGER_WINGER = True
TRAIN_FB_FB = True
TRAIN_CD_CD = True
TRAIN_MIDFIELD_MIDFIELD = True

# NOTE: Train_midfielders is aimed for subpositions within midfielders, so it will not be trained if TRAIN_OTHER_TO_MF is False
TRAIN_MIDFIELDERS = False
TRAIN_OTHER_TO_MF = True

SAVE_PARAMS = True
SAVE_MODELS = True
VERBOSE = False

predicted_df = pd.DataFrame()  # Initialize an empty DataFrame to store predictions from all models

df = load_data()

def dropVals(df, quals):
    return df
# ========== WINGER POSITION TRANSITIONS ==========
if TRAIN_WINGER_STRIKER:
    print("\nWinger - Striker")
    df_w_st = df[(df["from_position"] == "Winger") & (df["to_position"] == "Striker")].copy()
    df_w_st = dropVals(df_w_st, STRIKER_QUALITIES)
    print(f"Samples: {len(df_w_st)}")
    train_winger_to_striker(df_w_st, targets=STRIKER_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, df_predictions=predicted_df)

if TRAIN_WINGER_FB:
    print("\nWinger - Full Back")
    df_w_fb = df[(df["from_position"] == "Winger") & (df["to_position"] == "Full Back")].copy()
    df_w_fb = dropVals(df_w_fb, FB_QUALITIES)
    print(f"Samples: {len(df_w_fb)}")
    train_winger_to_fb(df_w_fb, targets=FB_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, df_predictions=predicted_df)

# ========== FULL BACK POSITION TRANSITIONS ==========
if TRAIN_FB_CD:
    print("\nFull Back - Central Defender")
    df_fb_cd = df[(df["from_position"] == "Full Back") & (df["to_position"] == "Central Defender")].head(2000).copy()
    df_fb_cd = dropVals(df_fb_cd, CENTRAL_DEFENDER_QUALITIES)
    print(f"Samples: {len(df_fb_cd)}")
    train_fb_to_cd(df_fb_cd, targets=CENTRAL_DEFENDER_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, df_predictions=predicted_df)

if TRAIN_FB_WINGER:
    print("\nFull Back - Winger")
    df_fb_w = df[(df["from_position"] == "Full Back") & (df["to_position"] == "Winger")].copy()
    df_fb_w= dropVals(df_fb_w, WINGER_QUALITIES)
    print(f"Samples: {len(df_fb_w)}")
    train_fb_to_winger(df_fb_w, targets=WINGER_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, df_predictions=predicted_df)

# ========== CENTRAL DEFENDER POSITION TRANSITIONS ==========
if TRAIN_CD_FB:
    print("\nCentral Defender - Full Back")
    df_cd_fb = df[(df["from_position"] == "Central Defender") & (df["to_position"] == "Full Back")].copy()
    df_cd_fb = dropVals(df_cd_fb, FB_QUALITIES)
    print(f"Samples: {len(df_cd_fb)}")
    train_cd_to_fb(df_cd_fb, targets=FB_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, df_predictions=predicted_df)

if TRAIN_CD_MF:
    print("\nCentral Defender - Midfielder")
    df_cd_mf = df[(df["from_position"] == "Central Defender") & (df["to_position"] == "Midfielder")].copy()
    df_cd_mf = dropVals(df_cd_mf, MIDFIELDER_QUALITIES)
    print(f"Samples: {len(df_cd_mf)}")
    train_cd_to_mf(df_cd_mf, targets=MIDFIELDER_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, df_predictions=predicted_df)

# ========== STRIKER POSITION TRANSITIONS ==========
if TRAIN_ST_WINGER:
    print("\nStriker - Winger")
    df_st_w = df[(df["from_position"] == "Striker") & (df["to_position"] == "Winger")].copy()
    df_st_w = dropVals(df_st_w, WINGER_QUALITIES)
    print(f"Samples: {len(df_st_w)}")
    train_st_to_winger(df_st_w, targets=WINGER_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, df_predictions=predicted_df)

if TRAIN_ST_MF:
    print("\nStriker - Midfielder")
    df_st_mf = df[(df["from_position"] == "Striker") & (df["to_position"] == "Midfielder")].copy()
    df_st_mf = dropVals(df_st_mf, MIDFIELDER_QUALITIES)
    print(f"Samples: {len(df_st_mf)}")
    train_st_to_mf(df_st_mf, targets=MIDFIELDER_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, df_predictions=predicted_df)

if TRAIN_ST_ST:
    print("\nStriker - Striker")
    df_st_st = df[(df["from_position"] == "Striker") & (df["to_position"] == "Striker")].head(1500).copy()
    df_st_st = dropVals(df_st_st, STRIKER_QUALITIES)
    print(f"Samples: {len(df_st_st)}")
    train_same_position(df_st_st, targets=STRIKER_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, position="Striker", df_predictions=predicted_df)

if TRAIN_WINGER_WINGER:
    print("\nWinger - Winger")
    df_w_w = df[(df["from_position"] == "Winger") & (df["to_position"] == "Winger")].head(1500).copy()
    df_w_w = dropVals(df_w_w, WINGER_QUALITIES)
    print(f"Samples: {len(df_w_w)}")
    train_same_position(df_w_w, targets=WINGER_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, position="Winger", df_predictions=predicted_df)

if TRAIN_FB_FB:
    print("\nFull Back - Full Back")
    df_fb_fb = df[(df["from_position"] == "Full Back") & (df["to_position"] == "Full Back")].head(1500).copy()
    df_fb_fb = dropVals(df_fb_fb, FB_QUALITIES)
    print(f"Samples: {len(df_fb_fb)}")
    train_same_position(df_fb_fb, targets=FB_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, position="Full Back", df_predictions=predicted_df)

if TRAIN_CD_CD:
    print("\nCentral Defender - Central Defender")
    df_cd_cd = df[(df["from_position"] == "Central Defender") & (df["to_position"] == "Central Defender")].head(1500).copy()
    df_cd_cd = dropVals(df_cd_cd, CENTRAL_DEFENDER_QUALITIES)
    print(f"Samples: {len(df_cd_cd)}")
    train_same_position(df_cd_cd, targets=CENTRAL_DEFENDER_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, position="Central Defender", df_predictions=predicted_df)

if TRAIN_MIDFIELD_MIDFIELD:
    print("\nMidfielder - Midfielder")
    df_mf_mf = df[(df["from_position"] == "Midfielder") & (df["to_position"] == "Midfielder")].head(1500).copy()
    df_mf_mf = dropVals(df_mf_mf, MIDFIELDER_QUALITIES)
    print(f"Samples: {len(df_mf_mf)}")
    train_same_position(df_mf_mf, targets=MIDFIELDER_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, position="Midfielder", df_predictions=predicted_df)

if TRAIN_MIDFIELDER_STRIKER:
    print("\nMidfielder - Striker")
    df_mf_st = df[(df["from_position"] == "Midfielder") & (df["to_position"] == "Striker")].copy()
    df_mf_st = dropVals(df_mf_st, STRIKER_QUALITIES)
    print(f"Samples: {len(df_mf_st)}")
    train_mf_to_st(df_mf_st, targets=STRIKER_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, position="Striker", df_predictions=predicted_df)

if TRAIN_MIDFIELDER_CD:
    print("\nMidfielder - Central Defender")
    df_mf_cd = df[(df["from_position"] == "Midfielder") & (df["to_position"] == "Central Defender")].copy()
    df_mf_cd = dropVals(df_mf_cd, CENTRAL_DEFENDER_QUALITIES)
    print(f"Samples: {len(df_mf_cd)}")
    train_mf_to_cd(df_mf_cd, targets=CENTRAL_DEFENDER_QUALITIES, save_params=SAVE_PARAMS, save_models=SAVE_MODELS, verbose=VERBOSE, position="Central Defender", df_predictions=predicted_df)