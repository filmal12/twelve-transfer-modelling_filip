import re
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
from mplsoccer import Pitch
import mplcursors
import plotly.graph_objects as go
import sys
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

from mplsoccer import Pitch, VerticalPitch

from mplsoccer import Pitch
import mplcursors

sys.path.append(os.path.abspath(".."))
from setup import (
    POSITION_TRANSITIONS,
    POS_ABBREV,
    POSITION_QUALITIES,
    TEAM_QUALITY_SUFFIXES,
    POSITION_CATEGORY,
    CATEGORY_TEAM_TARGETS,
    FEATURE_CATEGORIES,
    CATEGORY_LABELS,
    TEAM_DEST_DESCRIPTIONS,
    mapped_quals,
    normal_quals,
    ALL_QUALITIES,
    IND_VARS,
    IND_TEAM_VARS,
    TEAM_QUALS,
    TEAM_QUALITIES,
    POSITIONAL_CHANGES
)

from data_loader import load_data, get_mf, get_team_stats
from team_qualities import get_team_qualities

MODEL_TYPE = "OLS"

# ========== RADAR PLOT HELPERS ==========
def _set_arc_labels(ax, angles, labels, label_radius=1.44, fontsize=10):
    """Place labels around the perimeter, rotated tangentially."""
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    for angle, label in zip(angles, labels):
        angle_deg = np.degrees(angle)
        rotation = angle_deg - 90
        if rotation > 90:
            rotation -= 180
        elif rotation < -90:
            rotation += 180
        ax.text(
            angle, label_radius, label,
            fontsize=fontsize, fontweight="bold", color="#fafafa",
            rotation=rotation, rotation_mode="anchor",
            ha="center", va="center", clip_on=False,
        )
def _apply_figure_radial_gradient(fig, resolution=900):
    """Paint the full figure with a radial gradient: dark-green edge to white center."""
    outer_color = np.array([0.05, 0.32, 0.10])
    inner_color = np.array([0.99, 1.00, 0.99])

    grid = np.linspace(-1.0, 1.0, resolution)
    xx, yy = np.meshgrid(grid, grid)
    radius = np.sqrt(xx**2 + yy**2)
    radius = np.clip(radius / np.sqrt(2), 0, 1)

    image = np.empty((resolution, resolution, 3))
    for channel in range(3):
        image[:, :, channel] = inner_color[channel] * (1 - radius) + outer_color[channel] * radius

    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-10)
    bg_ax.imshow(image, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    bg_ax.axis('off')

def create_top_features_radar(player_row, player_name, best_position, best_score,
                              position_prefix, target_name, all_df, source_position,
                              figsize=(3.5, 3.5), font_scale=1.0):
    """Create dark-mode radar plot of top 10 model features and return the figure."""
    if not position_prefix or not target_name:
        return None

    if best_position == source_position:
        features_csv_path = f"../parameters/same_position/{best_position}/{target_name}_top_features.csv"
    else:
        features_csv_path = f"../parameters/{position_prefix}/{target_name}_top_features.csv"
    if not os.path.exists(features_csv_path):
        return None

    try:
        top_features_df = pd.read_csv(features_csv_path)
        if "feature" not in top_features_df.columns:
            return None
        top_features = [feature for feature in top_features_df["feature"].dropna().tolist() if feature not in normal_quals][:10]
    except Exception:
        return None

    feature_values = {}
    for feature in top_features:
        val = player_row.get(feature, np.nan)
        if pd.notna(val):
            feature_values[feature] = float(val)

    if not feature_values:
        return None

    players_pool = all_df[all_df["from_position"] == source_position].copy()

    labels = []
    values_normalized = []
    for feature, value in feature_values.items():
        if feature in players_pool.columns:
            dist_vals = players_pool[feature].dropna()
            if len(dist_vals) > 0:
                min_val, max_val = dist_vals.min(), dist_vals.max()
                if max_val > min_val:
                    normalized = np.clip((value - min_val) / (max_val - min_val), 0, 1)
                else:
                    normalized = 0.5
            else:
                normalized = 0.5
        else:
            normalized = 0.5

        if feature in normal_quals:
            labels.append(feature.replace('from_', 'Team from ').replace('to_', 'Team to ').replace('_', ' ').title())
        else:
            labels.append(feature.replace('from', '').replace('_', ' ').title())
        values_normalized.append(normalized)

    values_normalized = np.array(values_normalized)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_plot = list(values_normalized) + [values_normalized[0]]
    angles_plot = angles + [angles[0]]

    fs = font_scale
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    _apply_figure_radial_gradient(fig)
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.10, top=0.88)

    ax.plot(angles_plot, values_plot, 'o-', linewidth=2 * fs, color='#00d4aa', markersize=6 * fs)
    ax.fill(angles_plot, values_plot, alpha=0.25, color='#00d4aa')

    _set_arc_labels(ax, angles, labels, label_radius=1.08, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=10, color='black')
    ax.grid(True, linestyle='--', alpha=0.35, color='black', linewidth=0.6)
    ax.set_facecolor('none')
    ax.tick_params(colors='black', labelsize=7 * fs)
    ax.spines['polar'].set_color('black')

    ax.text(0.5, -0.09, f"Top model features vs other {source_position}s",
            transform=ax.transAxes, ha='center', va='center',
            color='white', fontsize=10 * fs, fontweight='bold')

    fig.patch.set_alpha(0)
    title = f"Positional suitability: {best_position}\nScore: {best_score:.4f} — {player_name}"
    ax.set_title("")
    ax.text(0.5, 1.07, title, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color='white', zorder=10)

    return fig

def _join_bold(names):
    """Join a list of names with commas/and, wrapped in bold markdown."""
    if len(names) == 1:
        return f"**{names[0]}**"
    if len(names) == 2:
        return f"**{names[0]}** and **{names[1]}**"
    return ", ".join(f"**{n}**" for n in names[:-1]) + f", and **{names[-1]}**"


def _display_name(feature):
    return feature.replace("_", " ").lower()


def _parse_ols_features(ols_df):
    """Parse an OLS params DataFrame into player features, dest team, and source team features."""
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
    """Determine the dominant attribute category among a list of feature names."""
    categories = {"offensive": 0, "creative": 0, "defensive": 0, "general": 0}
    for feat in feature_names:
        cat = FEATURE_CATEGORIES.get(feat, "general")
        categories[cat] += 1
    dominant_cat = max(categories, key=categories.get)
    return CATEGORY_LABELS.get(dominant_cat, "general")

def generate_transition_description(from_position, to_position, path_prefix, target_qual):
    """Generate natural language descriptions for a positional transition."""
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

def draw_second_position(ax, position, from_pos, from_x, from_y, POSITION_COORDS, COLOR_MAP):
    transitioned_x, transitioned_y = POSITION_COORDS[position]

    print(transitioned_x, transitioned_y)
    ax.add_patch(mpatches.Ellipse(
        (transitioned_x, transitioned_y), 8, 8,
        facecolor="#FFFFFF", edgecolor="white", linewidth=3,
        alpha=0.95, zorder=1,
    ))

    ax.text(
        transitioned_x, transitioned_y, POS_ABBREV[position],
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="black", zorder=2,
    )

    _dx = transitioned_x - from_x
    _dy = transitioned_y - from_y
    _dist = (_dx ** 2 + _dy ** 2) ** 0.5
    _ux, _uy = _dx / _dist, _dy / _dist
    _radius = 4  

    _start_x = from_x + _ux * _radius
    _start_y = from_y + _uy * _radius
    _end_x = transitioned_x - _ux * _radius
    _end_y = transitioned_y - _uy * _radius

    ax.annotate(
        "",
        xy=(_end_x, _end_y),
        xytext=(_start_x, _start_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#ba4f45",
            lw=2.5,
            mutation_scale=20,
        ),
        zorder=7,
    )

def draw_self_loop_position(ax, from_x, from_y):
    # Draw a curved self-referencing arrow looping above the ellipse
    _loop_radius = 3
    _cx = from_x - 2
    _cy = from_y - 6.5
    ax.add_patch(matplotlib.patches.Arc(
        (_cx, _cy), (_loop_radius * 2), (_loop_radius * 2),
        angle=0, theta1=100, theta2=400,
        color="#ba4f45", linewidth=2.5, zorder=7,
    ))
    # Arrowhead at the end of the arc (theta2=350 ≈ just before 0°, so tip points downward-left)
    _tip_x = (_cx + _loop_radius * np.cos(np.radians(360))) - 0.5
    _tip_y = (_cy + _loop_radius * np.sin(np.radians(360))) + 2
    _tangent_x = -np.sin(np.radians(370))
    _tangent_y =  np.cos(np.radians(360))
    ax.annotate(
        "",
        xy=(_tip_x, _tip_y),
        xytext=(_tip_x - _tangent_x * 0.005, _tip_y - _tangent_y * 0.01),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#ba4f45",
            lw=2.5,
            mutation_scale=30,
        ),
        zorder=8,
    )

def display_position_change(from_pos, to_position, best_position, value, second_position=None):
    print(from_pos, to_position, best_position, second_position)
    to_positions = [p for p in POSITION_TRANSITIONS.get(from_pos, [])
                    if p in to_position]
    if not to_positions:
        to_positions = list(to_position)
    n_to = len(to_positions)

    POSITION_COORDS = {
        "Central Defender": (40, 65),
        "Full Back":        (10, 70),
        "Midfielder":       (40, 90),
        "Winger":           (10, 100),
        "Striker":          (40, 105),
    }

    COLOR_MAP = {
        "Central Defender": "#888888",
        "Full Back":        "#888888",
        "Midfielder":       "#888888",
        "Winger":           "#888888",
        "Striker":          "#888888",
    }
    
    pitch = VerticalPitch(half=True, pitch_color="grass", line_color = "white", stripe=True)
    fig, ax = pitch.draw()

    from_x, from_y = POSITION_COORDS.get(from_pos, (52, 34))
    from_color = COLOR_MAP.get(from_pos, "#FFFFFF")

    # FROM zone
    ax.add_patch(mpatches.Ellipse(
        (from_x, from_y), 8, 8,
        facecolor="#FFFFFF", edgecolor="white", linewidth=3,
        alpha=0.95, zorder=1,
    ))
    ax.text(
        from_x, from_y, POS_ABBREV[from_pos],
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="black", zorder=2,
    )

    print(best_position, value)

    if best_position == from_pos:
        draw_self_loop_position(ax, from_x, from_y)
    else:
        draw_second_position(ax, best_position, from_pos, from_x, from_y, POSITION_COORDS, COLOR_MAP)

    if second_position != None: 
        if second_position == from_pos:
            draw_self_loop_position(ax, from_x, from_y)
        else:
            draw_second_position(ax, second_position, from_pos, from_x, from_y, POSITION_COORDS, COLOR_MAP)

    return fig

def draw_other_position(ax, position, label, from_pos, from_x, from_y, POSITION_COORDS, COLOR_MAP, rad=0.0, color="#ba4f45"):
    transitioned_x, transitioned_y = POSITION_COORDS[position]

    ax.add_patch(mpatches.Ellipse(
        (transitioned_x, transitioned_y), 8, 8,
        facecolor="#FFFFFF", edgecolor="white", linewidth=3,
        alpha=0.95, zorder=1,
    ))

    ax.text(
        transitioned_x, transitioned_y, POS_ABBREV[position],
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="black", zorder=2,
    )

    _dx = transitioned_x - from_x
    _dy = transitioned_y - from_y
    _dist = (_dx ** 2 + _dy ** 2) ** 0.5
    _ux, _uy = _dx / _dist, _dy / _dist
    _px, _py = -_uy, _ux  # left-perpendicular unit vector
    _radius = 4

    _start_x = from_x + _ux * _radius
    _start_y = from_y + _uy * _radius
    _end_x = transitioned_x - _ux * _radius
    _end_y = transitioned_y - _uy * _radius

    ax.annotate(
        "",
        xy=(_end_x, _end_y),
        xytext=(_start_x, _start_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=2.5,
            mutation_scale=20,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=7,
    )

    if label:
        # Place label at the actual curve midpoint (quadratic bezier at t=0.5)
        _straight_mid_x = (_start_x + _end_x) / 2
        _straight_mid_y = (_start_y + _end_y) / 2
        # Curve bulge: control point is at mid + px*rad*dist, bezier midpoint is mid + px*rad*dist*0.5
        _curve_perp = rad * _dist * 0.5
        _label_clearance = 3.5 if rad >= 0 else -3.5
        _mid_x = _straight_mid_x + _px * (_curve_perp + _label_clearance)
        _mid_y = _straight_mid_y + _py * (_curve_perp + _label_clearance)
        _angle = np.degrees(np.arctan2(_dy, _dx))
        if _angle > 90 or _angle < -90:
            _angle += 180
        ax.text(
            _mid_x, _mid_y, label,
            ha="center", va="center", fontsize=5, fontweight="bold",
            color="white", zorder=9,
            rotation=_angle, rotation_mode="anchor",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none", alpha=0.85),
        )

def draw_self_loop(ax, from_x, from_y, label="", side="bottom", color="#ba4f45"):
    # Self-loop using arc3 connectionstyle — exits/enters the node diagonally
    _node_r = 4
    _rad = -1  # negative: arc3 is computed in display coords (y-down), so negative rad curves AWAY from node
    _offset = _node_r * np.cos(np.radians(45))  # ~2.83

    if side == "bottom":
        start = (from_x + _offset, from_y - _offset)
        end   = (from_x - _offset, from_y - _offset)
        _label_x = from_x
        _label_y = from_y - _node_r - 3.5
        _label_va = "top"
        _label_ha = "center"
    elif side == "top":
        start = (from_x - _offset, from_y + _offset)
        end   = (from_x + _offset, from_y + _offset)
        _label_x = from_x
        _label_y = from_y + _node_r + 3.5
        _label_va = "bottom"
        _label_ha = "center"
    elif side == "left":
        start = (from_x - _offset, from_y - _offset)
        end   = (from_x - _offset, from_y + _offset)
        _label_x = from_x - _node_r - 3.5
        _label_y = from_y
        _label_va = "center"
        _label_ha = "right"
    else:  # right
        start = (from_x + _offset, from_y + _offset)
        end   = (from_x + _offset, from_y - _offset)
        _label_x = from_x + _node_r + 3.5
        _label_y = from_y
        _label_va = "center"
        _label_ha = "left"

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=2.5,
            mutation_scale=20,
            connectionstyle=f"arc3,rad={_rad}",
        ),
        zorder=8,
    )
    if label:
        ax.text(
            _label_x, _label_y, label,
            ha=_label_ha, va=_label_va, fontsize=5, fontweight="bold",
            color="white", zorder=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none", alpha=0.85),
        )

def display_top_features(df_features, from_pos):
    POSITION_COORDS = {
        "Central Defender": (30, 65),
        "Full Back":        (75, 70),
        "Midfielder":       (40, 90),
        "Winger":           (10, 100),
        "Striker":          (40, 115),
    }

    COLOR_MAP = {
        "Central Defender": "#AC8E34",
        "Full Back":        "#6db925",
        "Midfielder":       "#3e1d70",
        "Winger":           "#458dba",
        "Striker":          "#ba4f45",
    }

    def clean_feat(feat):
        """Return a human-readable feature label."""
        if feat in normal_quals:
            return (feat
                    .replace("to_", "New team: ")
                    .replace("from_", "Old team: ")
                    .replace("_", " ").title())
        return (feat
                .replace("from_z_score_", "")
                .replace("from_", "")
                .replace("_", " ").title())
    
    pitch = VerticalPitch(half=True, pitch_color="grass", line_color = "white", stripe=True)
    fig, ax = pitch.draw()

    from_x, from_y = POSITION_COORDS.get(from_pos, (52, 34))
    from_color = COLOR_MAP.get(from_pos, "#FFFFFF")

    # FROM zone
    ax.add_patch(mpatches.Ellipse(
        (from_x, from_y), 8, 8,
        facecolor="#FFFFFF", edgecolor="white", linewidth=3,
        alpha=0.95, zorder=1,
    ))
    ax.text(
        from_x, from_y, POS_ABBREV[from_pos],
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="black", zorder=2,
    )
    
    # Group features by target position; chunk into groups of 2 so labels don't overflow
    from collections import defaultdict
    pos_feature_groups = defaultdict(list)
    for _, row in df_features.iterrows():
        pos_feature_groups[row["To position"]].append(row["feature"])

    _CHUNK = 2
    _RAD_STEP = 0.22  # curvature separation between parallel arrows

    for position, features in pos_feature_groups.items():
        chunks = [features[i:i + _CHUNK] for i in range(0, len(features), _CHUNK)]
        n = len(chunks)
        # Spread rads symmetrically around 0 (straight arrow when only one chunk)
        rads = [_RAD_STEP * (i - (n - 1) / 2) for i in range(n)]

        for i, chunk in enumerate(chunks):
            combined_label = " / ".join(clean_feat(f) for f in chunk)
            if position == from_pos:
                _side_order = []

                if position == "Striker":
                    _side_order = ["top", "right", "left", "bottom"]
                elif position == "Midfielder":
                    _side_order = ["left", "right", "top", "bottom"]
                elif position == "Central Defender":
                    _side_order = ["left", "right", "bottom", "top"]
                else:
                    _side_order = ["bottom", "top", "left" if from_x < 40 else "right"]
                side = _side_order[min(i, len(_side_order) - 1)]
                
                draw_self_loop(ax, from_x, from_y, label=combined_label, side=side, color=COLOR_MAP[position])
            else:
                draw_other_position(ax, position, combined_label, from_pos, from_x, from_y, POSITION_COORDS, COLOR_MAP, rad=rads[i], color=COLOR_MAP[position])


    return fig

def predict_player(player_name, season, df_full, competition_data):
    """Predict best position fit for the selected player."""
    player_df = df_full[
        (df_full["short_name"] == player_name) & (df_full["from_season"] == season)
    ]
    if player_df.empty:
        st.error(f"No data found for {player_name} in season {season}")
        return

    player_row = player_df.iloc[0].copy()
    from_pos = player_row["from_position"]

    if from_pos not in POSITION_TRANSITIONS:
        st.warning(f"No transitions defined for position: {from_pos}")
        return

    quals = POSITION_QUALITIES[from_pos]
    to_positions = POSITION_TRANSITIONS[from_pos]

    # Prepare features
    z_score_cols = []

    z_score_cols.extend(IND_VARS)

    team_stats = [c for c in df_full.columns if "from_team_stats" in c]
    z_score_cols.extend(team_stats)
    z_score_cols.extend(IND_TEAM_VARS)
    z_score_cols.append("wyscout_weight_scaled")
    z_score_cols.append("player_season_age_scaled")
    z_score_cols.append("wyscout_height_scaled")

    z_score_cols = [c for c in z_score_cols if c in list(df_full.columns)]

    player_row = player_row.convert_dtypes()
    X_player = player_row[z_score_cols].fillna(0).values.reshape(1, -1)
    X_player_df = pd.DataFrame(X_player, columns=z_score_cols)

    # Set competition dummy columns.
    # to_competition is always Swedish first division (the target league).
    # from_competition is taken from the player's actual competition.
    _TO_COMPETITION = "Swedish first division"

    def _sanitize_comp(name):
        return re.sub(r"[^a-zA-Z0-9_]", "_", str(name))

    X_player_df[f"to_competition_name_{_sanitize_comp(_TO_COMPETITION)}"] = 1.0

    print(list(X_player_df.columns))

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
            model_path = f"../parameters/{path}/{target}_xgboost.pkl"
            if MODEL_TYPE == "OLS":
                model_path = f"../parameters/{path}/{target}.csv"
        
            if os.path.exists(model_path):
                try:
                    if MODEL_TYPE != "OLS":
                        xgb_model = joblib.load(model_path)
                        feature_names = xgb_model.get_booster().feature_names
                        
                        if feature_names is None:
                            feature_names = [f"f{i}" for i in range(xgb_model.n_features_in_)]

                        player_df_feat = X_player_df.reindex(columns=feature_names, fill_value=0).fillna(0)
                        
                        pred = xgb_model.predict(player_df_feat)[0]
                        
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
        path = f"../parameters/same_position/{from_pos}/{qual}_xgboost.pkl"
        if MODEL_TYPE == "OLS":
            path = f"../parameters/same_position/{from_pos}/{qual}.csv"

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
        model_path = f"../team_models/delta_{target}_xgboost.pkl"
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
    