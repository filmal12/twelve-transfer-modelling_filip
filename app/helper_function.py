import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from mplsoccer import VerticalPitch

sys.path.append(os.path.abspath(".."))
from setup import (
    POSITION_TRANSITIONS,
    POS_ABBREV,
    POSITION_QUALITIES,
    IND_VARS,
    IND_TEAM_VARS,
    normal_quals,
)

################################
# Different helper functions for 
# generating themes for plots
# and figures in the application
################################

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
        features_csv_path = f"parameters/same_position/{best_position}/{target_name}_top_features.csv"
    else:
        features_csv_path = f"parameters/{position_prefix}/{target_name}_top_features.csv"
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

def prepare_player_df(df_full, player_name, season):
    player_df = df_full[
        (df_full["short_name"] == player_name) & (df_full["from_season"] == season)
    ]
    if player_df.empty:
        return None, None, None

    player_row = player_df.iloc[0].copy()
    from_pos = player_row["from_position"]

    if from_pos not in POSITION_TRANSITIONS:
        return None, None, None

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

    X_player = player_row[z_score_cols].fillna(0).to_numpy(dtype=float).reshape(1, -1)
    X_player_df = pd.DataFrame(X_player, columns=z_score_cols)

    return player_row, X_player_df, z_score_cols