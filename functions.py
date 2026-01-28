# ─────────────────────────────
# Function to plot field with labels AND legend
# ─────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.patches as patches
import pandas as pd

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap


import numpy as np
import matplotlib.pyplot as plt

def plot_intent_impact(
    batter,
    batter_stats,
    bowl_kind="pace",
    min_count=5
):
    """
    Plot cumulative intent impact curves (raw vs controlled)
    for a single batter vs spin or pace.
    """

    if batter not in batter_stats:
        raise ValueError(f"{batter} not found in batter_stats")

    if bowl_kind not in batter_stats[batter]:
        raise ValueError(f"{batter} has no data vs {bowl_kind}")

    data = batter_stats[batter][bowl_kind]

    cnts = data["batter_ith_ball_count"]

    raw_bat = data["batter_ith_ball_raw_runs"]
    ctl_bat = data["batter_ith_ball_controlled_runs"]

    raw_ns  = data["non_striker_ith_ball_raw_runs"]
    ctl_ns  = data["non_striker_ith_ball_controlled_runs"]

    valid = sorted(i for i in cnts if cnts[i] >= min_count)
    if not valid:
        raise ValueError("No points satisfy min_count filter")

    # ── Per-ball expected values
    raw_brpb = np.array([raw_bat[i] / cnts[i] for i in valid])
    ctl_brpb = np.array([ctl_bat[i] / cnts[i] for i in valid])

    raw_nrpb = np.array([raw_ns.get(i, 0) / cnts[i] for i in valid])
    ctl_nrpb = np.array([ctl_ns.get(i, 0) / cnts[i] for i in valid])

    # ── Cumulative intent impact
    raw_impact = np.cumsum(raw_brpb - raw_nrpb)
    ctl_impact = np.cumsum(ctl_brpb - ctl_nrpb)

    # ─────────────────────────────
    # FIGURE SETUP
    # ─────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    colors = {
        "raw": "#ff9100",        # orange
        "controlled": "#00e5ff"  # cyan
    }

    # ── Glow layers
    for impact, color in [(raw_impact, colors["raw"]),
                          (ctl_impact, colors["controlled"])]:
        for lw, alpha in [(10, 0.06), (7, 0.10), (5, 0.14)]:
            ax.plot(valid, impact,
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                    solid_capstyle="round",
                    zorder=2)

    # ── Main curves
    ax.plot(valid, raw_impact,
            color=colors["raw"],
            linewidth=2.8,
            label="Raw intent impact",
            zorder=4)

    ax.plot(valid, ctl_impact,
            color=colors["controlled"],
            linewidth=2.8,
            label="Controlled intent impact",
            zorder=5)

    # Zero line
    ax.axhline(0, color="white", linestyle="--",
               linewidth=1.6, alpha=0.8, zorder=1)

    # ─────────────────────────────
    # Labels & title
    # ─────────────────────────────
    ax.set_xlabel("Balls Faced",
                  fontsize=13, fontweight="bold", color="white")
    ax.set_ylabel("Cumulative Intent Impact",
                  fontsize=13, fontweight="bold", color="white")

    ax.set_title(
        f"Intent Impact Curve — {batter} vs {bowl_kind.capitalize()}",
        fontsize=15,
        fontweight="bold",
        color="white",
        pad=14
    )

    # Legend
    legend = ax.legend(loc="upper left", frameon=True,
                       fontsize=11, labelcolor="white")
    legend.get_frame().set_facecolor("#1a1a1a")
    legend.get_frame().set_edgecolor("white")
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_alpha(0.9)

    # Grid & axes
    ax.grid(True, linestyle="--", alpha=0.15)
    ax.tick_params(colors="white", labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig


def plot_field_setting(field_data):
    """
    Ultra-modern cricket field visualization with transparent background
    and sleek design elements
    """
    LIMIT = 400
    THIRTY_YARD_RADIUS_M = LIMIT/2 - 15

    # Create figure with TRANSPARENT background
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_alpha(0.0)  # Transparent figure background
    ax.patch.set_alpha(0.0)   # Transparent axes background

    inside_info = field_data['infielder_positions']
    outside_info = field_data['outfielder_positions']
    special_fielders = field_data['special_fielders']

    # Create fielder labels
    infielder_labels = {}
    outfielder_labels = {}
    
    # ═══════════════════════════════
    # FIELD BASE - Gradient green circle
    # ═══════════════════════════════
    
    # Create gradient effect with multiple circles
    gradient_colors = ['#15901e']
    gradient_radii = [LIMIT + 25]
    
    for color, radius in zip(gradient_colors, gradient_radii):
        circle = plt.Circle(
            (0, 0), 
            radius, 
            color=color, 
            alpha=0.8,
            zorder=0
        )
        ax.add_artist(circle)
    
    # ═══════════════════════════════
    # FIELD MARKINGS - Modern minimalist
    # ═══════════════════════════════
    
    # 30-yard circle (inner) - Neon style
    circle_30 = plt.Circle(
        (0, 0), 
        THIRTY_YARD_RADIUS_M + 20, 
        color='#00ff88', 
        fill=False, 
        linestyle='--', 
        linewidth=2.5, 
        alpha=0.6
    )
    ax.add_artist(circle_30)
    
    # Boundary circle (outer) - Bold white
    circle_boundary = plt.Circle(
        (0, 0), 
        LIMIT + 25, 
        color='white', 
        fill=False, 
        linewidth=4, 
        alpha=0.95
    )
    ax.add_artist(circle_boundary)
    
    # Pitch - Modern style with rounded corners
    from matplotlib.patches import FancyBboxPatch
    pitch = FancyBboxPatch(
        (-15, -60), 
        30, 
        120, 
        boxstyle="round,pad=2",
        facecolor='#8b7355',
        edgecolor='white',
        linewidth=2.5,
        zorder=2,
        alpha=0.9
    )
    ax.add_patch(pitch)
    
    # Pitch center line - Glowing effect
    ax.plot([0, 0], [-60, 60], color='white', linewidth=2, alpha=0.8, zorder=3)
    ax.plot([0, 0], [-60, 60], color='#00ff88', linewidth=4, alpha=0.3, zorder=2)
    
    # Crease lines
    ax.plot([-15, 15], [0, 0], color='white', linewidth=2.5, alpha=0.9, zorder=3)
    ax.plot([-15, 15], [-50, -50], color='white', linewidth=2, alpha=0.7, zorder=3)
    ax.plot([-15, 15], [50, 50], color='white', linewidth=2, alpha=0.7, zorder=3)
    
    # ═══════════════════════════════
    # INFIELDERS - Modern design
    # ═══════════════════════════════
    wall_angle = special_fielders.get('30_yard_wall')
    
    for idx, angle in enumerate(inside_info):
        ang_rad = np.deg2rad(angle)
        is_wall = (angle == wall_angle)
        label = f"I{idx+1}"
        infielder_labels[angle] = label
        
        x_pos = THIRTY_YARD_RADIUS_M * np.sin(ang_rad)
        y_pos = THIRTY_YARD_RADIUS_M * np.cos(ang_rad)
        
        if is_wall:
            # 30-Yard Wall - Red hexagon with glow
            color = '#ff1744'
            size = 750
            marker = 'h'  # hexagon
            edge_width = 3.5
            glow_color = '#ff1744'
        else:
            # Regular infielder - Cyan with modern style
            color = '#00e5ff'
            size = 550
            marker = 'o'
            edge_width = 3
            glow_color = '#00e5ff'
        
        # Outer glow (3 layers for smooth effect)
        for glow_size, glow_alpha in [(size * 2.5, 0.1), (size * 2, 0.15), (size * 1.5, 0.2)]:
            ax.scatter(
                x_pos, y_pos,
                c=glow_color,
                s=glow_size,
                marker=marker,
                alpha=glow_alpha,
                zorder=8
            )
        
        # Main marker with gradient effect
        ax.scatter(
            x_pos, y_pos,
            c=color,
            s=size,
            marker=marker,
            edgecolors='white',
            linewidth=edge_width,
            alpha=0.95,
            zorder=10
        )
        
        # Inner highlight
        ax.scatter(
            x_pos, y_pos,
            c='white',
            s=size * 0.3,
            marker='o',
            alpha=0.4,
            zorder=11
        )
        
        # Label with modern font style
        ax.text(
            x_pos, y_pos,
            label,
            ha='center',
            va='center',
            color='white',
            fontsize=15 if is_wall else 13,
            fontweight='bold',
            zorder=12,
            family='monospace'
        )

    # ═══════════════════════════════
    # OUTFIELDERS - Modern design
    # ═══════════════════════════════
    sprinter_angle = special_fielders.get('sprinter')
    catcher_angle = special_fielders.get('catcher')
    superfielder_angle = special_fielders.get('superfielder')

    for idx, angle in enumerate(outside_info):
        ang_rad = np.deg2rad(angle)
        label = f"O{idx+1}"
        outfielder_labels[angle] = label
        
        x_pos = LIMIT * np.sin(ang_rad)
        y_pos = LIMIT * np.cos(ang_rad)
        
        # Determine special fielder types with modern colors
        if angle == superfielder_angle:
            color = '#ffd600'  # Bright gold
            marker = 'D'
            size = 750
            edge_width = 3.5
            glow_color = '#ffd600'
            special = True
        elif angle == sprinter_angle:
            color = '#ff6d00'  # Vibrant orange
            marker = '^'
            size = 750
            edge_width = 3.5
            glow_color = '#ff6d00'
            special = True
        elif angle == catcher_angle:
            color = '#76ff03'  # Neon lime
            marker = '*'
            size = 800
            edge_width = 3.5
            glow_color = '#76ff03'
            special = True
        else:
            color = '#e040fb'  # Bright magenta
            marker = 'o'
            size = 550
            edge_width = 3
            glow_color = '#e040fb'
            special = False

        # Outer glow (3 layers)
        for glow_size, glow_alpha in [(size * 2.5, 0.1), (size * 2, 0.15), (size * 1.5, 0.2)]:
            ax.scatter(
                x_pos, y_pos,
                c=glow_color,
                s=glow_size,
                marker=marker,
                alpha=glow_alpha,
                zorder=8
            )
        
        # Main marker
        ax.scatter(
            x_pos, y_pos,
            s=size,
            c=color,
            marker=marker,
            edgecolors='white',
            linewidth=edge_width,
            alpha=0.95,
            zorder=10
        )
        
        # Inner highlight
        if marker in ['o', 'D', '^']:
            ax.scatter(
                x_pos, y_pos,
                c='white',
                s=size * 0.3,
                marker='o',
                alpha=0.4,
                zorder=11
            )
        
        # Label
        text_color = 'black' if color in ['#ffd600', '#76ff03'] else 'white'
        ax.text(
            x_pos, y_pos,
            label,
            ha='center',
            va='center',
            color=text_color,
            fontsize=15 if special else 13,
            fontweight='bold',
            zorder=12,
            family='monospace'
        )

    # ═══════════════════════════════
    # DIRECTION INDICATOR - Modern sleek arrow
    # ═══════════════════════════════
    
    # Glow layers
    for width, head_w, head_l, alpha in [(30, 55, 40, 0.15), (25, 50, 35, 0.2), (20, 45, 30, 0.25)]:
        ax.arrow(
            0, 50, 0, -75,
            width=width,
            head_width=head_w,
            head_length=head_l,
            fc='#ff1744',
            ec='none',
            linewidth=0,
            zorder=13,
            alpha=alpha
        )
    
    # Main arrow
    ax.arrow(
        0, 50, 0, -75,
        width=15,
        head_width=40,
        head_length=25,
        fc='#ff1744',
        ec='white',
        linewidth=3,
        zorder=15,
        alpha=0.95
    )
    
    # Direction label - Modern badge
    ax.text(
        0, 70, 
        'FACING',
        ha='center',
        va='center',
        color='white',
        fontsize=11,
        fontweight='bold',
        bbox=dict(
            facecolor='#ff1744',
            alpha=0.95,
            boxstyle='round,pad=0.7',
            edgecolor='white',
            linewidth=2.5
        ),
        zorder=16,
        family='sans-serif'
    )

    # ═══════════════════════════════
    # ANGLE MARKERS - Minimalist badges
    # ═══════════════════════════════
    
    
    

    # ═══════════════════════════════
    # LEGEND - Ultra-modern glass-morphic style
    # ═══════════════════════════════
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='h', color='w', markerfacecolor='#ff1744', 
               markersize=13, label='30-Yard Wall', markeredgecolor='white', markeredgewidth=2.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00e5ff', 
               markersize=11, label='Infielder', markeredgecolor='white', markeredgewidth=2),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff6d00', 
               markersize=13, label='Sprinter', markeredgecolor='white', markeredgewidth=2.5),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#76ff03', 
               markersize=15, label='Catcher', markeredgecolor='white', markeredgewidth=2.5),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#ffd600', 
               markersize=11, label='Superfielder', markeredgecolor='white', markeredgewidth=2.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e040fb', 
               markersize=11, label='Outfielder', markeredgecolor='white', markeredgewidth=2),
    ]
    
    legend = ax.legend(
        handles=legend_elements,
        facecolor='#1a1a1a',
        edgecolor='white',
        framealpha=0.9,
        loc='upper left',
        fontsize=9,
        labelcolor='white',
        title='FIELDERS',
        title_fontsize=11,
        frameon=True,
        shadow=False,
        borderpad=1.2,
        labelspacing=1
    )
    legend.get_title().set_color('white')
    legend.get_title().set_weight('bold')
    legend.get_frame().set_linewidth(2.5)

    # ═══════════════════════════════
    # FINAL SETTINGS
    # ═══════════════════════════════
    ax.set_xlim(-(LIMIT + 80), LIMIT + 80)
    ax.set_ylim(-(LIMIT + 80), LIMIT + 80)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    
    return fig, infielder_labels, outfielder_labels


def plot_sector_ev_heatmap(
    ev_dict, 
    batter_name, 
    selected_lengths, 
    bowl_kind,
    length_dict,
    LIMIT=350, 
    THIRTY_YARD_RADIUS_M=171.25 * 350 / 500
):
    """
    Combined polar heatmap with modern design and transparent background:
    - Inner sector (≤30-yard): running EV (ev_run)
    - Outer sector (>30-yard): boundary EV (ev_bd)
    Both use a common color scale for consistent intensity interpretation.
    """
    try:
        # Normalize selected_lengths to a list
        if isinstance(selected_lengths, (str, tuple)):
            sel_lens = [selected_lengths] if isinstance(selected_lengths, str) else list(selected_lengths)
        else:
            sel_lens = list(selected_lengths)
        n_lens = len(sel_lens)
        band_width = 15

        # Collect all theta centers across selected lengths
        all_theta = set()
        per_len_dfs = {}
        for ln in sel_lens:
            try:
                df = ev_dict[batter_name].get(ln, {}).get(bowl_kind)
                if df is None:
                    # missing length/bowl kind -> will be treated as zeros
                    per_len_dfs[ln] = None
                else:
                    per_len_dfs[ln] = df.copy()
                    all_theta.update(df['theta_center_deg'].values % 360)
            except Exception:
                per_len_dfs[ln] = None

        if len(all_theta) == 0:
            st.warning('No sector EV data available for the selected lengths.')
            return None

        all_theta = sorted(all_theta)
        theta_centers = np.array(all_theta)
        # Build aggregated arrays (average across lengths; missing treated as zero)
        agg_ev_run = []
        agg_ev_bd = []

        for theta in theta_centers:
            run_num = 0.0
            bd_num = 0.0
            denom = 0.0

            for ln in sel_lens:
                balls = length_dict.get(batter_name, {}) \
                                    .get(bowl_kind, {}) \
                                    .get(ln, 0)

                if balls == 0:
                    continue

                df = per_len_dfs.get(ln)
                if df is None:
                    continue

                row = df.loc[df['theta_center_deg'] % 360 == theta]
                if row.empty:
                    continue

                run_num += float(row['ev_run'].values[0]) * balls
                bd_num += float(row['ev_bd'].values[0]) * balls
                denom += balls

            agg_ev_run.append(run_num / denom if denom > 0 else 0.0)
            agg_ev_bd.append(bd_num / denom if denom > 0 else 0.0)

        ev_run = np.array(agg_ev_run)
        ev_bd = np.array(agg_ev_bd)

        # Common normalization across both datasets
        all_vals = np.concatenate([ev_bd, ev_run])
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

        # Create figure with TRANSPARENT background
        fig = plt.figure(figsize=(9, 9))
        fig.patch.set_alpha(0.0)  # Transparent figure
        ax = fig.add_subplot(111, polar=True)
        ax.patch.set_alpha(0.0)  # Transparent axes
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # Modern red/orange gradient colormap
        
        
        colors_list = ['#fde047', '#fbbf24', '#f97316', '#dc2626', '#991b1b', '#450a0a', '#1a0000']
        cmap = LinearSegmentedColormap.from_list('modern_red', colors_list, N=256)

        # Inner ring (Running EV) with glow effect
        for theta, ev in zip(theta_centers, ev_run):
            if not np.isnan(ev):
                color = cmap((ev - vmin) / (vmax - vmin + 1e-9))
                
                # Glow layer
                ax.bar(
                    np.deg2rad(theta),
                    THIRTY_YARD_RADIUS_M,
                    width=np.deg2rad(band_width * 1.2),
                    bottom=0,
                    color=color,
                    edgecolor='none',
                    linewidth=0,
                    alpha=0.2,
                    zorder=1
                )
                
                # Main bar
                ax.bar(
                    np.deg2rad(theta),
                    THIRTY_YARD_RADIUS_M,
                    width=np.deg2rad(band_width),
                    bottom=0,
                    color=color,
                    edgecolor='white',
                    linewidth=1,
                    alpha=0.95,
                    zorder=2
                )

        # Outer ring (Boundary EV) with glow effect
        for theta, ev in zip(theta_centers, ev_bd):
            if not np.isnan(ev):
                color = cmap((ev - vmin) / (vmax - vmin + 1e-9))
                
                # Glow layer
                ax.bar(
                    np.deg2rad(theta),
                    LIMIT - THIRTY_YARD_RADIUS_M,
                    width=np.deg2rad(band_width * 1.2),
                    bottom=THIRTY_YARD_RADIUS_M,
                    color=color,
                    edgecolor='none',
                    linewidth=0,
                    alpha=0.2,
                    zorder=1
                )
                
                # Main bar
                ax.bar(
                    np.deg2rad(theta),
                    LIMIT - THIRTY_YARD_RADIUS_M,
                    width=np.deg2rad(band_width),
                    bottom=THIRTY_YARD_RADIUS_M,
                    color=color,
                    edgecolor='white',
                    linewidth=1,
                    alpha=0.95,
                    zorder=2
                )

        # Draw visual guides with modern styling
        inner_circle = plt.Circle(
            (0, 0), THIRTY_YARD_RADIUS_M, 
            color='white', fill=False, 
            linestyle='--', linewidth=3, 
            transform=ax.transData._b,
            alpha=0.7,
            zorder=3
        )
        boundary_circle = plt.Circle(
            (0, 0), LIMIT, 
            color='white', fill=False, 
            linewidth=3.5, 
            transform=ax.transData._b,
            alpha=0.9,
            zorder=3
        )
        ax.add_artist(inner_circle)
        ax.add_artist(boundary_circle)

        # Title styling - Modern
        ax.set_title(
            f"Sector Importance\n{', '.join(map(str, selected_lengths))} • {bowl_kind}",
            fontsize=15, 
            weight='bold', 
            color='white',
            pad=25,
            family='sans-serif'
        )
        
        # Axis styling - Modern badges
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
        ax.set_xticklabels(
            [f'{int(t)}°' for t in np.arange(0, 360, 30)], 
            fontsize=10,
            color='white',
            weight='bold',
            family='monospace'
        )
        ax.grid(True, color='white', alpha=0.15, linewidth=1, linestyle='-')
        ax.set_yticklabels([])
        ax.spines['polar'].set_color('white')
        ax.spines['polar'].set_linewidth(3)

        # Modern labels with glassmorphic effect
        ax.text(
            0, THIRTY_YARD_RADIUS_M / 2, 
            'Running', 
            ha='center', va='center',
            fontsize=10, weight='bold', color='white',
            bbox=dict(
                facecolor='#1a1a1a', 
                alpha=0.9, 
                boxstyle='round,pad=0.6',
                edgecolor='white',
                linewidth=2
            ),
            zorder=10,
            family='sans-serif'
        )
        
        ax.text(
            np.pi, (THIRTY_YARD_RADIUS_M + LIMIT) / 2, 
            'Boundary', 
            ha='center', va='center',
            fontsize=10, weight='bold', color='white',
            bbox=dict(
                facecolor='#1a1a1a', 
                alpha=0.9, 
                boxstyle='round,pad=0.6',
                edgecolor='white',
                linewidth=2
            ),
            zorder=10,
            family='sans-serif'
        )

        # Modern colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(all_vals)
        sm.set_clim(vmin, vmax)
        
        cbar = plt.colorbar(
            sm, 
            ax=ax, 
            pad=0.12, 
            fraction=0.046,
            aspect=25
        )
        cbar.set_label(
            'Importance', 
            fontsize=11, 
            color='white',
            weight='bold',
            family='sans-serif'
        )
        cbar.ax.tick_params(colors='white', labelsize=10, width=2)
        cbar.outline.set_edgecolor('white')
        cbar.outline.set_linewidth(2)
        cbar.ax.set_facecolor('#1a1a1a')
        cbar.ax.patch.set_alpha(0.9)

        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating EV heatmap: {e}")
        return None


def create_zone_strength_table(dict_360, batter_name, selected_lengths, bowl_kind, length_dict, kind):
    """
    Clean stacked bar chart showing zone distributions across run classes
    """
    try:
        # Normalize selected_lengths to list
        if isinstance(selected_lengths, (str, tuple)):
            sel_lens = [selected_lengths] if isinstance(selected_lengths, str) else list(selected_lengths)
        else:
            sel_lens = list(selected_lengths)

        # Aggregate data across lengths
        run_classes = ['overall', 'running', 'boundary']
        aggregated = {rc: {} for rc in run_classes}
        
        for rc in run_classes:
            keys_union = set()
            per_len_data = {}
            
            for ln in sel_lens:
                try:
                    per = dict_360[batter_name].get(ln, {}).get(bowl_kind, {}).get(rc)
                    per_len_data[ln] = per
                    if per:
                        keys_union.update(per.keys())
                except Exception:
                    per_len_data[ln] = None
            
            # Build averaged data
            for key in keys_union:
                num = 0.0
                denom = 0.0

                for ln in sel_lens:
                    balls = length_dict.get(batter_name, {}) \
                                        .get(bowl_kind, {}) \
                                        .get(ln, 0)

                    if balls == 0:
                        continue

                    per = per_len_data.get(ln)
                    val = per.get(key, 0.0) if per else 0.0

                    num += val * balls
                    denom += balls

                aggregated[rc][key] = num / denom if denom > 0 else 0.0


        # Calculate zone percentages for each run class
        all_zones = {}
        for rc in run_classes:
            data = aggregated[rc]
            total = data.get('total_runs', 0)
            
            all_zones[rc] = {
                'Straight': (data.get(f'st_{kind}', 0) / total * 100) if total else 0,
                'Leg': (data.get(f'leg_{kind}', 0) / total * 100) if total else 0,
                'Off': (data.get(f'off_{kind}', 0) / total * 100) if total else 0,
                'Behind': (data.get(f'bk_{kind}', 0) / total * 100) if total else 0
            }

        # CREATE FIGURE - Single horizontal stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        
        # Zone colors (red gradient theme)
        zone_colors = {
            'Straight': '#dc2626',
            'Leg': '#f97316',
            'Off': '#fbbf24',
            'Behind': '#991b1b'
        }
        
        # Labels
        rc_labels = {
            'overall': 'Overall',
            'running': 'Running',
            'boundary': 'Boundary'
        }
        
        zones_order = ['Straight', 'Leg', 'Off', 'Behind']
        y_positions = [2, 1, 0]  # reversed for top-to-bottom
        
        # Draw stacked bars
        for idx, rc in enumerate(run_classes):
            zones = all_zones[rc]
            left = 0
            
            for zone in zones_order:
                pct = zones[zone]
                
                bar = ax.barh(
                    y_positions[idx],
                    pct,
                    left=left,
                    height=0.6,
                    color=zone_colors[zone],
                    edgecolor='white',
                    linewidth=2,
                    alpha=0.9,
                    label=zone if idx == 0 else ""
                )
                
                # Add percentage text if segment is large enough
                if pct > 0:
                    ax.text(
                        left + pct/2,
                        y_positions[idx],
                        f'{pct:.0f}',
                        ha='center',
                        va='center',
                        fontsize=15,  # Increased from 10
                        fontweight='bold',
                        color='white'
                    )
                
                left += pct
        
        # Styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [rc_labels[rc] for rc in run_classes], 
            fontsize=25,  # Increased from 12
            fontweight='bold', 
            color='white'
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel('Percentage (%)', fontsize=20, fontweight='bold', color='white')
        
        # Grid
        ax.grid(axis='x', color='white', alpha=0.2, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Spines
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(1.5)
        
        ax.tick_params(colors='white', labelsize=20)
        
        # Legend - INCREASED FONT SIZE
        legend = ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.25),  # Moved up slightly
            ncol=4,
            frameon=True,
            facecolor='#1a0000',
            edgecolor='white',
            framealpha=0.9,
            fontsize=20,  # Increased from 10
            labelcolor='white',
            handlelength=1.5,
            handleheight=1.5,
            columnspacing=1.5
        )
        
        # Make legend text bold
        for text in legend.get_texts():
            text.set_weight('bold')
        
        plt.tight_layout()
        
        # Return overall zones for compatibility
        return fig, all_zones['overall']
        
    except Exception as e:
        st.error(f"Error creating zone strength visualization: {e}")
        return None, None




def create_shot_profile_chart(
    shot_per,
    batter_name,
    selected_lengths,
    bowl_kind,
    length_dict,
    value_type="runs"   # "runs" or "avg_runs"
):
    """
    Modern horizontal bar chart with transparent background and glow effects
    """
    try:
        # Normalize selected lengths
        if isinstance(selected_lengths, (str, tuple)):
            sel_lens = [selected_lengths] if isinstance(selected_lengths, str) else list(selected_lengths)
        else:
            sel_lens = list(selected_lengths)

        # Aggregate shots across lengths (average, missing treated as 0)
        shots_set = set()
        per_len_shots = {}
        for ln in sel_lens:
            per = shot_per.get(batter_name, {}).get(ln, {}).get(bowl_kind, {})
            per_len_shots[ln] = per
            shots_set.update([s for s, v in (per or {}).items() if isinstance(v, dict)])

        shots = {}

        for shot in shots_set:
            num = 0.0
            denom = 0.0

            for ln in sel_lens:
                balls = length_dict.get(batter_name, {}) \
                                    .get(bowl_kind, {}) \
                                    .get(ln, 0)

                if balls == 0:
                    continue

                per = per_len_shots.get(ln) or {}
                val = per.get(shot, {}).get(value_type, 0)

                num += val * balls
                denom += balls

            shots[shot] = num / denom if denom > 0 else 0.0


        if not shots:
            return None

        # SORT
        sorted_shots = sorted(
            shots.items(),
            key=lambda x: x[1],
            reverse=True
        )

        shot_names = [shot for shot, _ in sorted_shots]
        shot_values = [val for _, val in sorted_shots]

        # TRANSPARENT FIGURE
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')

        # Modern gradient colormap
        
        
        colors_list = ['#fde047', '#fbbf24', '#f97316', '#dc2626', '#991b1b', '#450a0a', '#1a0000']
        cmap = LinearSegmentedColormap.from_list('modern_red', colors_list, N=256)

        vmin, vmax = min(shot_values), max(shot_values)

        # Create bars with glow effect
        y_positions = np.arange(len(shot_names))
        
        for i, (y, value) in enumerate(zip(y_positions, shot_values)):
            color = cmap((value - vmin) / (vmax - vmin + 1e-9))
            
            # Glow effect (3 layers)
            for glow_width, glow_alpha in [(0.8, 0.15), (0.6, 0.2), (0.4, 0.25)]:
                ax.barh(
                    y,
                    value,
                    height=glow_width,
                    color=color,
                    edgecolor='none',
                    alpha=glow_alpha,
                    zorder=1
                )
            
            # Main bar
            ax.barh(
                y,
                value,
                height=0.7,
                color=color,
                edgecolor='white',
                linewidth=2,
                alpha=0.95,
                zorder=2
            )
            
            # Value label with badge
            ax.text(
                value + (vmax * 0.02),
                y,
                f'{value:.1f}%',
                va='center',
                ha='left',
                color='white',
                fontweight='bold',
                fontsize=10,
                bbox=dict(
                    facecolor='#1a1a1a',
                    alpha=0.9,
                    edgecolor=color,
                    linewidth=2,
                    boxstyle='round,pad=0.4'
                ),
                family='monospace',
                zorder=3
            )

        # STYLING
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            shot_names,
            color='white',
            fontsize=11,
            fontweight='bold',
            family='sans-serif'
        )

        xlabel = "Run Share (%)" if value_type == "runs" else "Avg Batter Run Share (%)"
        title_suffix = "Actual Runs" if value_type == "runs" else "Avg Batter Runs"

        ax.set_xlabel(
            xlabel, 
            color='white', 
            fontsize=12, 
            fontweight='bold',
            family='sans-serif'
        )
        ax.set_title(
            f'Shot Strength Profile ({title_suffix})\n'
            f"{', '.join(map(str, selected_lengths))} • {bowl_kind}",
            color='white',
            fontsize=14,
            fontweight='bold',
            pad=20,
            family='sans-serif'
        )

        # Modern grid
        ax.grid(
            axis='x',
            color='white',
            alpha=0.15,
            linestyle='-',
            linewidth=1
        )
        ax.set_axisbelow(True)

        # Spine styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_color('white')
        ax.spines['bottom'].set_linewidth(2)

        ax.tick_params(colors='white', labelsize=10, width=2, length=6)

        ax.set_xlim(0, max(shot_values) * 1.2)
        ax.invert_yaxis()

        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Error creating shot profile: {e}")
        return None

def get_top_similar_batters(
    sim_matrices,
    batter_name,
    selected_lengths,
    bowl_kind,
    top_n=5
):
    """
    Average similarities across selected lengths and return top-N batters.
    """
    # Normalize lengths
    if isinstance(selected_lengths, (str, tuple)):
        sel_lens = [selected_lengths] if isinstance(selected_lengths, str) else list(selected_lengths)
    else:
        sel_lens = list(selected_lengths)

    sims_accum = {}

    valid_count = 0
    for ln in sel_lens:
        key = (ln, bowl_kind)
        if key not in sim_matrices:
            continue

        sim_df = sim_matrices[key]

        if batter_name not in sim_df.index:
            continue

        row = sim_df.loc[batter_name]
        valid_count += 1

        for bat, val in row.items():
            if bat == batter_name:
                continue
            sims_accum[bat] = sims_accum.get(bat, 0) + val

    if valid_count == 0:
        return None

    # Average
    avg_sims = {k: v / valid_count for k, v in sims_accum.items()}

    out = (
        pd.DataFrame(avg_sims.items(), columns=["batter", "similarity"])
        .sort_values("similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return out

def create_similarity_chart(
    sim_df,
   
    batter_name,
    selected_lengths,
    bowl_kind
):
    """
    Horizontal similarity bar chart with player photos on Y-axis.
    """
    if sim_df is None or sim_df.empty:
        return None

    names = sim_df["batter"].tolist()
    values = sim_df["similarity"].tolist()

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

   
    cmap = LinearSegmentedColormap.from_list(
        "sim_red",
        ['#fde047',  '#f97316', '#dc2626', '#991b1b', '#450a0a', '#1a0000'],
        N=256
    )

    vmin, vmax = min(values), max(values)
    y_pos = np.arange(len(names))

    # Bars with glow
    for y, val in zip(y_pos, values):
        color = cmap((val - vmin) / (vmax - vmin + 1e-9))

        for h, a in [(0.8, 0.15), (0.6, 0.2), (0.4, 0.25)]:
            ax.barh(y, val, height=h, color=color, alpha=a)

        ax.barh(y, val, height=0.6, color=color, edgecolor="white", linewidth=2)

        ax.text(
            val + 0.01,
            y,
            f"{val:.2f}",
            va="center",
            ha="left",
            color="white",
            fontweight="bold",
            fontsize=10,
            bbox=dict(
                facecolor="#111",
                edgecolor=color,
                boxstyle="round,pad=0.3",
                linewidth=2
            )
        )

    # Player labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color="white", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    ax.set_xlabel("Similarity Score", color="white", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Most Similar Batters to {batter_name}\n"
        f"{', '.join(map(str, selected_lengths))} • {bowl_kind}",
        color="white",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    ax.grid(axis="x", alpha=0.15)
    ax.tick_params(colors="white")
    ax.spines[['top','right']].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    ax.set_xlim(0, max(values) * 1.2)

    plt.tight_layout()
    return fig



def plot_intrel_pitch(
    metric,
    heading,    
    intrel_results,
    batter,
    lengths,
    bowl_kind,
    min_balls=10
):
    """
    3D-perspective pitch showing intent-relative by length.
    Returns matplotlib figure.
    """
    if bowl_kind=='pace bowler':
        bowl_kind = 'pace'
    else:
        bowl_kind = 'spin'    
    data = intrel_results.get(batter, {}).get(bowl_kind, {})
    if not data:
        raise ValueError(f"No data for {batter} ({bowl_kind})")

    length_data = data[metric]

    # --- figure ---
    fig, ax = plt.subplots(figsize=(4.5, 6))
    fig.patch.set_alpha(0)     # <-- IMPORTANT
    ax.set_facecolor("none")  
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- perspective transform (simple trapezoid pitch) ---
    top_y = 0.90
    bot_y = 0.05

    pitch = np.array([
        [0.20 + top_y * 0.15, top_y],
        [0.80 - top_y * 0.15, top_y],
        [0.80 - bot_y * 0.15, bot_y],
        [0.20 + bot_y * 0.15, bot_y],
    ])

    ax.add_patch(
        patches.Polygon(
            pitch,
            closed=True,
            fill=False,
            edgecolor="white",
            linewidth=3.2,
            alpha=0.95,
            joinstyle="round"
        )
    )


    # --- normalize int-rel for colors ---
    intrels = [
        v[0] for v in length_data.values()
        if not np.isnan(v[0]) and v[1] >= min_balls
    ]

    if not intrels:
        raise ValueError("No lengths with sufficient balls")

    colors_list = [
    '#fde047', '#fbbf24', '#f97316',
    '#dc2626', '#991b1b', '#450a0a'
    ]
    modern_cmap = LinearSegmentedColormap.from_list(
        'modern_red', colors_list, N=256
    )

    norm = Normalize(vmin=0.5, vmax=1.5)
    mapper = ScalarMappable(norm=norm, cmap=modern_cmap)
    LENGTH_ZONES = {
    "FULL": (0.75, 0.90),
    "GOOD_LENGTH": (0.50, 0.75),
    "SHORT_OF_A_GOOD_LENGTH": (0.30, 0.50),
    "SHORT": (0.05, 0.30)
    }
    
    # --- draw length bands ---
    for length, (y0, y1) in LENGTH_ZONES.items():
        if length not in lengths:
         continue 
        intrel, balls = length_data.get(length, (np.nan, 0))
        if balls < min_balls or np.isnan(intrel):
            continue

        color = mapper.to_rgba(intrel)

        # trapezoidal band (perspective scaling)
        band = np.array([
            [0.20 + y0 * 0.15, y0],
            [0.80 - y0 * 0.15, y0],
            [0.80 - y1 * 0.15, y1],
            [0.20 + y1 * 0.15, y1],
        ])

        ax.add_patch(
            patches.Polygon(
                band,
                closed=True,
                facecolor=color,
                edgecolor="white",
                linewidth=2,
                alpha=0.65
            )
        )

        # label
        ax.text(
            0.5,
            (y0 + y1) / 2,
            f"{length.replace('_', ' ')}\n{intrel:.2f}",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            fontweight="bold"
        )

    
    stump_x = [0.48, 0.50, 0.52]
    for x in stump_x:
        ax.plot([x, x], [0.9, 0.975], color="white", linewidth=3)
    
    # --- title ---
    fig.text(
    0.5, 0.92,
    heading,
    ha="center",
    va="top",
    fontsize=12,
    fontweight="bold",
    color="white"
    )

    return fig

def plot_intrel_pitch_avg(
    intrel_results,
    batter,
    lengths,
    bowl_kind,
    min_balls=10
):
    """
    3D-perspective pitch showing Avg Bat (SR, Control%) by length.
    Neutral alternating colors (blue/green), no color grading.
    Returns matplotlib figure.
    """

    if bowl_kind == 'pace bowler':
        bowl_kind = 'pace'
    else:
        bowl_kind = 'spin'

    data = intrel_results.get(batter, {}).get(bowl_kind, {})
    if not data:
        raise ValueError(f"No data for {batter} ({bowl_kind})")

    sr_data = data.get("othsr", {})
    con_data = data.get("othcon", {})

    # --- figure ---
    fig, ax = plt.subplots(figsize=(4.5, 6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- perspective pitch ---
    top_y = 0.90
    bot_y = 0.05

    pitch = np.array([
        [0.20 + top_y * 0.15, top_y],
        [0.80 - top_y * 0.15, top_y],
        [0.80 - bot_y * 0.15, bot_y],
        [0.20 + bot_y * 0.15, bot_y],
    ])

    ax.add_patch(
        patches.Polygon(
            pitch,
            closed=True,
            fill=False,
            edgecolor="white",
            linewidth=3.2,
            alpha=0.95,
            joinstyle="round"
        )
    )

    LENGTH_ZONES = {
        "FULL": (0.75, 0.90),
        "GOOD_LENGTH": (0.50, 0.75),
        "SHORT_OF_A_GOOD_LENGTH": (0.30, 0.50),
        "SHORT": (0.05, 0.30)
    }

    # alternating neutral colors
    colors = ["#2563eb", "#16a34a"]  # blue, green

    # --- draw bands ---
    color_idx = 0
    for length, (y0, y1) in LENGTH_ZONES.items():
        if length not in lengths:
            continue

        sr, balls_sr = sr_data.get(length, (np.nan, 0))
        con, balls_con = con_data.get(length, (np.nan, 0))

        balls = min(balls_sr, balls_con)
        if balls < min_balls or np.isnan(sr) or np.isnan(con):
            continue

        color = colors[color_idx % 2]
        color_idx += 1

        band = np.array([
            [0.20 + y0 * 0.15, y0],
            [0.80 - y0 * 0.15, y0],
            [0.80 - y1 * 0.15, y1],
            [0.20 + y1 * 0.15, y1],
        ])

        ax.add_patch(
            patches.Polygon(
                band,
                closed=True,
                facecolor=color,
                edgecolor="white",
                linewidth=2,
                alpha=0.65
            )
        )

        ax.text(
            0.5,
            (y0 + y1) / 2,
            f"{length.replace('_', ' ')}\n{sr:.0f}, {con:.0f}%",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            fontweight="bold"
        )

    # --- stumps ---
    for x in [0.48, 0.50, 0.52]:
        ax.plot([x, x], [0.9, 0.975], color="white", linewidth=3)

    # --- heading ---
    fig.text(
        0.5, 0.92,
        "Avg Bat (SR, Control%)",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="white"
    )

    return fig