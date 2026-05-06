import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(".."))
from setup import (
    POS_ABBREV,
    TEAM_QUALITY_SUFFIXES,
    POSITION_CATEGORY,
    FEATURE_CATEGORIES,
    CATEGORY_LABELS,
    TEAM_DEST_DESCRIPTIONS,
)


################################
# Contains functionality for generating 
# descriptive texts for explaining outcomes
# of the predictions in the application
################################


def _join_bold(names):
    if len(names) == 1:
        return f"**{names[0]}**"
    if len(names) == 2:
        return f"**{names[0]}** and **{names[1]}**"
    return ", ".join(f"**{n}**" for n in names[:-1]) + f", and **{names[-1]}**"


def _display_name(feature):
    return feature.replace("_", " ").lower()


def _parse_ols_features(ols_df):
    player_pos, player_neg = [], []
    team_to_pos, team_to_neg = [], []
    team_from_features = []

    for _, row in ols_df.iterrows():
        factor = row["Factor"]
        coeff = row["mean"]
        if factor == "Intercept":
            continue

        if factor.startswith("to_"):
            suffix = factor[3:]
            if suffix in TEAM_QUALITY_SUFFIXES:
                (team_to_pos if coeff > 0 else team_to_neg).append((factor, coeff))
        elif factor.startswith("from_"):
            suffix = factor[5:]
            if suffix in TEAM_QUALITY_SUFFIXES:
                team_from_features.append((factor, coeff))
            else:
                (player_pos if coeff > 0 else player_neg).append((suffix, coeff))

    player_pos.sort(key=lambda x: x[1], reverse=True)
    player_neg.sort(key=lambda x: x[1])
    team_to_pos.sort(key=lambda x: x[1], reverse=True)
    return player_pos, player_neg, team_to_pos, team_to_neg, team_from_features

def _parse_xgboost_features(xg_df):
    player_pos, player_neg = [], []
    team_to_pos, team_to_neg = [], []
    team_from_features = []

    for _, row in xg_df.iterrows():
        feature = row["feature"]
        coeff = row["importance"]

        if feature.startswith("to_"):
            suffix = feature[3:]
            if suffix in TEAM_QUALITY_SUFFIXES:
                (team_to_pos if coeff > 0 else team_to_neg).append((feature, coeff))
        elif feature.startswith("from_"):
            suffix = feature[5:]
            if suffix in TEAM_QUALITY_SUFFIXES:
                team_from_features.append((feature, coeff))
            else:
                (player_pos if coeff > 0 else player_neg).append((suffix, coeff))

    player_pos.sort(key=lambda x: x[1], reverse=True)
    player_neg.sort(key=lambda x: x[1])
    team_to_pos.sort(key=lambda x: x[1], reverse=True)
    return player_pos, player_neg, team_to_pos, team_to_neg, team_from_features


def __get_attribute_category(feature_names):
    categories = {"offensive": 0, "creative": 0, "defensive": 0, "general": 0}
    for feat in feature_names:
        cat = FEATURE_CATEGORIES.get(feat, "general")
        categories[cat] += 1
    dominant_cat = max(categories, key=categories.get)
    return CATEGORY_LABELS.get(dominant_cat, "general")

def generate_transition_description(from_position, to_position, path_prefix, target_qual):
    params_dir = f"parameters/{path_prefix}"

    all_player_impacts = {}
    all_team_to_impacts = {}
    quality_descriptions = {}

    target = target_qual
    xg_path = os.path.join(params_dir, f"{target}_top_features.csv")
    if not os.path.exists(xg_path):
        return None, None
    xg_df = pd.read_csv(xg_path)
    player_pos, player_neg, team_to_pos, team_to_neg, _ = _parse_xgboost_features(xg_df)

    # Accumulate for overall summary
    for feat, coeff in player_pos + player_neg:
        all_player_impacts.setdefault(feat, []).append(coeff)
    for feat, coeff in team_to_pos + team_to_neg:
        all_team_to_impacts.setdefault(feat, []).append(coeff)

    # Per-quality description
    target_display = target.replace("_", " ").lower()
    parts = []

    if player_pos:
        names = [_display_name(f) for f, _ in player_pos[:3]]
        parts.append(
            f"Players with strong {_join_bold(names)} are more likely to develop "
            f"into a good {target_display} {to_position} when making the transition from {from_position} to {to_position}."
        )

    if player_neg:
        names = [_display_name(f) for f, _ in player_neg[:2]]
        parts.append(
            f"Players who rely heavily on {_join_bold(names)} may find this "
            f"quality harder to develop, as these traits are less relevant in the new role. Which is most likely due to the fact that these "
            f"qualities are more important fo the {__get_attribute_category(names)} aspect of the game and less so for the "
            f"{POSITION_CATEGORY[to_position]} aspect that a {to_position} requires to be successful."
        )

    if team_to_pos:
        descs = [TEAM_DEST_DESCRIPTIONS.get(f, f.replace("_", " ")) for f, _ in team_to_pos[:2]]
        parts.append(
            f"Moving to a team that is {' and '.join(descs)} further supports "
            f"development in this area."
        )

    quality_descriptions[target] = " ".join(parts) if parts else "Insufficient model data for this quality."

    # --- Overall summary ---
    avg_player = {f: np.mean(cs) for f, cs in all_player_impacts.items()}
    sorted_player = sorted(avg_player.items(), key=lambda x: x[1], reverse=True)
    top_positive = [(f, c) for f, c in sorted_player if c > 0][:4]
    top_negative = [(f, c) for f, c in sorted_player if c < 0][-3:]

    summary_parts = []
    if top_positive:
        cats = {}
        for f, _ in top_positive:
            cat = FEATURE_CATEGORIES.get(f, "general")
            cats.setdefault(cat, []).append(f)
        dominant_cat = max(cats, key=lambda k: len(cats[k]))
        cat_label = CATEGORY_LABELS.get(dominant_cat, "")

        names = [_display_name(f) for f, _ in top_positive]
        summary_parts.append(
            f"The transition from **{from_position}** to **{to_position}** favors "
            f"players with strong {cat_label} attributes, particularly {_join_bold(names)}."
        )

    if top_negative:
        names = [_display_name(f) for f, _ in top_negative]
        summary_parts.append(
            f"Players heavily reliant on {_join_bold(names)} may need to adapt their playing style."
        )

    avg_team_to = {f: np.mean(cs) for f, cs in all_team_to_impacts.items()}
    top_team = sorted(avg_team_to.items(), key=lambda x: x[1], reverse=True)
    top_team_pos = [(f, c) for f, c in top_team if c > 0][:2]
    if top_team_pos:
        descs = [TEAM_DEST_DESCRIPTIONS.get(f, f.replace("_", " ")) for f, _ in top_team_pos]
        summary_parts.append(f"The ideal destination team should be {' and '.join(descs)}.")

    overall_summary = " ".join(summary_parts) if summary_parts else ""
    return overall_summary, quality_descriptions

def generate_player_transition_description(player_name, from_position, to_position, df_full, quality, non_conclusive, two_positions=False, other_position=None, other_position_target=None, delta_def =None):
    """Generate a natural language description of the player's suitability for the transition."""
    same_position = from_position == to_position

    path_prefix = f"{POS_ABBREV[from_position]}_to_{POS_ABBREV[to_position]}/" if not same_position else f"same_position/{from_position}/"

    general_desc = ""

    if non_conclusive:
        general_desc = "Since the model outputs good indicators for all positions, the model cannot conclusively determine a best position for this player. However, here are some insights on the potential transition: \n\n"
    else:
        general_desc = "The model identifies a clear best position for this player, but here are some insights on the potential transition to the other position: \n\n"

    _, desc = generate_transition_description(from_position, to_position, path_prefix, quality)
    

    if not desc:
        return "Insufficient model data to generate a description for this transition."

    result = f"According to the prediction, {player_name} is most suited to play in the {to_position} position. " if not same_position else f"{player_name} is likely to have the most success by staying in the {to_position} position. "
    
    parsed_desc = desc.get(quality, "No specific insights for this quality.")

    result += parsed_desc

    general_desc += result

    if two_positions:
        second_desc = f"\n\n However, according to the prediction {player_name} also fits well to play within the {other_position} position aswell."

        second_desc += " This is his current position, which he also fits to play within the desired team because: \n\n" if other_position == from_position else " This is a different position than his current one, but he also fits well within it and could be a versatile asset for the team because: \n\n"
        same_position = from_position == other_position

        path_prefix = f"{POS_ABBREV[from_position]}_to_{POS_ABBREV[other_position]}/" if not same_position else f"same_position/{from_position}/"
        _, desc_player = generate_transition_description(from_position, other_position, path_prefix, other_position_target)

        if not desc_player:
            return "Insufficient model data to generate a description for this transition."
        
        parsed_desc = desc_player.get(other_position_target, "No specific insights for this quality.")

        general_desc += second_desc + parsed_desc

    if delta_def is not None:
        string_def_competition = "better" if delta_def == True else "worse"

        comp_desc = f"\n\nWhen comparing the player to the other players in the same position with the same quality, {player_name} would perform {string_def_competition} than his competition in the position in which the model sway the player towards."

        general_desc += comp_desc

    return general_desc

def get_predefined_description(position):
    if position == "Striker":
        return "In general for the Striker its clear to see that having attributes that are more attributed to the defensive part of the game, as well as having strong values within passing qualities, generally means that the striker should move towards the midfielder position to experienc the most success. \n\n" \
            " While being a striker with a skill-set that is more appropriate to movement, such as run quality and involvement, points toward a fit more likely to succeed as a winger." \
            "Staying as a striker is usually better when the player shows qualities in box-threat, pressing and providing teammates. "


    if position == "Midfielder":
        return "being a midfielder a transition to the midfield is most likely the scenario which ends up bringing the most success to the player.\n\n" \
        " However, being good defensively might prove to make the player a better central defender than midfielder. While being a good finisher and presser are mostly a result of the player being more successful at the winger position. "
        
    if position == "Full Back":
        return "Looking into the full back and looking at where that player would fit in another team, being good at pressing and an intelligent defender, points toward the player continuing their success in the fullback position when transitioning teams. \n\n" \
        " Being good as an active defender and on the ball usually makes towards a successful central defender." \
        " While being better forward, making runs into the box and being good infront of goal is more suited as a winger, which is visible in the figure"
    
    if position == "Central Defender":
        return "Being a Central defender that shows signs of being good at distributing the ball and an intelligent defender usually transitions best as a central defender when switching teams. \n\n" \
        "Being good at pressing and a more active defender transitions best to the full back spot. " \
        "While being a good passer and holder of the ball, while also showing signs of wanting to move forward usually results in a better transition into the midfield. "
    
    return "As a winger, moving to a midfielder position and succeeding is rarely encountered. While being a winger that is good on the ball as well in movement should stay out on the wing. \n\n" \
    "However, being a winger that shows better signs in front of goal and also being more involved and composed forward should move up to the striker position as they will experience the most success there.  "