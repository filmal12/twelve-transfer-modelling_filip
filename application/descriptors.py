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

def __get_attribute_category(feature_names):
    categories = {"offensive": 0, "creative": 0, "defensive": 0, "general": 0}
    for feat in feature_names:
        cat = FEATURE_CATEGORIES.get(feat, "general")
        categories[cat] += 1
    dominant_cat = max(categories, key=categories.get)
    return CATEGORY_LABELS.get(dominant_cat, "general")

def generate_transition_description(from_position, to_position, path_prefix, target_qual):
    params_dir = f"../parameters/{path_prefix}"

    all_player_impacts = {}
    all_team_to_impacts = {}
    quality_descriptions = {}

    target = target_qual
    ols_path = os.path.join(params_dir, f"{target}.csv")
    if not os.path.exists(ols_path):
        return None, None
    ols_df = pd.read_csv(ols_path)
    player_pos, player_neg, team_to_pos, team_to_neg, _ = _parse_ols_features(ols_df)

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

def generate_player_transition_description(player_name, from_position, to_position, df_full, quality, non_conclusive, two_positions=False, other_position=None, other_position_target=None):
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
    return general_desc

def get_predefined_description(position):
    if position == "Striker":
        return "In general a Striker that carries the ball into the box, is calm with the ball and comfortable making decisions, " \
        "is good infront of goal and goes to a team that plays more counter attacking football is more suited as a winger. These traits are usually something that a striker possess, but" \
        " being dominant in them might mean that the player is more suited playing wide As these traits are often linked to players ability to create chances, which is very much needed on the wing. "\
        "Being good infront of goal is also a trait that usually suits the winger \n\n" \
        "While a striker that is better on the ball and plays deeper down and distributes the ball is more suited as a midfielder. Playing as a midfielder often means being apart of the build up of the game, being a player that thrives with having the ball and a player that "\
        "is comfortable in making decision. If a striker is dominant in these traits, moving them down the field and letting them be a bigger part of the team is most likely the best option. \n\n" \
        "To continue their success as a striker the player needs to show good composure and play infront of goal."\
        " The striker role requires players that has abilities"\
        "that will make them counter the defenders inside the box. As well as this the striker should also be fearless infront of goal and always be available for the final pass. Therefore traits such as box-threat, poaching and composure" \
        "are needed."

    if position == "Midfielder":
        return "A midfielder that shows a good sign in attributes that regards play infront of goal and how a player moves between the lines is suited to play striker. \n\n" \
        " The striker role requires players that has abilities"\
        "that will make them counter the defenders inside the box. As well as this the striker should also be fearless infront of goal and always be available for the final pass. Therefore traits such as box-threat, poaching and composure" \
        "are needed." \
        "While a midfielder that shows more security in defending and is better in the air should play as a Central Defender." \
        " Playing as a central defender requires defensive traits, as well as understanding of the game. Being able to win the ball back is crucial, "\
        "but also being able to read the game and understand when to retrieve the ball and minimizing wrongdoings during a game is also important. " \
        "Therefore, midfielders that exceeds in these traits are more likely to succeed as central defenders. \n\n"\
        "A midfielder should stay in midfield if they are good on the ball, and shows signs of being involved in the game more." \
        "Playing as a midfielder often means being apart of the build up of the game, being a player that thrives with having the ball and a player that "\
        "is comfortable in making decision. If a striker is dominant in these traits, moving them down the field and letting them be a bigger part of the team is most likely the best option. \n\n" \
        "To continue their success as a striker the player needs to show good composure and play infront of goal."\
        
    if position == "Full Back":
        return "A Full back should play as a Full back if they are good defensively while also showing signs of being good offensively and are more involved in the buildup of the game. These are traits that favor both the defensive aspect of the game"\
        " as well as the build up play which the full back is a part of. Playing wide in the defensive line usually requires a lot of tracking back and defensive attendance, while it's a position that also "\
        "wants a player that has the ability to go forward an be part in the offensive attacks. \n\n" \
        "While a full back that is better one on one and good at making runs forward is more suited as playing in the winger role. \n\n" \
        "As these traits are often linked to players ability to create chances, which is very much needed on the wing. "\
        "Being good infront of goal is also a trait that usually suits the winger"\
        "Lastly, a full back being good in the air and good defensively is more suited at playing as a Central defender."\
        " Playing as a central defender requires defensive traits, as well as understanding of the game. Being able to win the ball back is crucial, "\
        "but also being able to read the game and understand when to retrieve the ball and minimizing wrongdoings during a game is also important. " \
        "Therefore, midfielders that exceeds in these traits are more likely to succeed as central defenders. \n\n"\
    
    if position == "Central Defender":
        return "A central defender that is good in the air and good defensively while also showing signs of being good at defending with pressure should stick to playing central defender." \
        " Playing as a central defender requires defensive traits, as well as understanding of the game. Being able to win the ball back is crucial, "\
        "but also being able to read the game and understand when to retrieve the ball and minimizing wrongdoings during a game is also important. " \
        "Therefore, midfielders that exceeds in these traits are more likely to succeed as central defenders. \n\n"\
        "While a central defender that is good on the ball and good at distributing the ball should move to a role within midfield." \
        " Playing as a midfielder often means being apart of the build up of the game, being a player that thrives with having the ball and a player that "\
        "is comfortable in making decision. If a striker is dominant in these traits, moving them down the field and letting them be a bigger part of the team is most likely the best option. \n\n" \
        "To continue their success as a striker the player needs to show good composure and play infront of goal."\
        "Lastly, a central defender that is good defensively while also showing signs of being more involved in the game and calm on the ball should move to the full back spot. These are traits that favor both the defensive aspect of the game"\
        " as well as the build up play which the full back is a part of. Playing wide in the defensive line usually requires a lot of tracking back and defensive attendance, while it's a position that also "\
        "wants a player that has the ability to go forward an be part in the offensive attacks. \n\n" \
    
    return "A winger that is good while progressing with the ball, good one on one and carrying the ball forward" \
    "should stay out on the wing. As these traits are often linked to players ability to create chances, which is very much needed on the wing. "\
    "Being good infront of goal is also a trait that usually suits the winger"\
    ", as their presence infront of goal is also needed in the offensive areas. \n\n" \
    "A winger that is more involved in the game while also good defensively at winning the ball back is more suited to play the full back position. These are traits that favor both the defensive aspect of the game"\
    " as well as the build up play which the full back is a part of. Playing wide in the defensive line usually requires a lot of tracking back and defensive attendance, while it's a position that also "\
    "wants a player that has the ability to go forward an be part in the offensive attacks. \n\n" \
    "Lastly, a winger that is more progressive going forward, showing signs of wanting to move into the box and is good infront of goal is more suited to play Striker. The striker role requires players that has abilities"\
    "that will make them counter the defenders inside the box. As well as this the striker should also be fearless infront of goal and always be available for the final pass. Therefore traits such as box-threat, poaching and composure" \
    "are needed. "