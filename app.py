import streamlit as st
st.set_page_config(layout="wide", page_title="Optimal Field Setting | Cricket Analytics")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.patches as patches

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────
# Custom CSS for RED premium design with RESPONSIVE elements
# ─────────────────────────────
st.markdown("""
<style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Global styling */
    .main {
        background: linear-gradient(135deg, #1a0a0a 0%, #2d1414 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling - RESPONSIVE */
    .main-header {
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%);
        padding: clamp(1.5rem, 4vw, 2.5rem) clamp(1rem, 3vw, 2rem);
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(220,38,38,0.4);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-title {
        font-size: clamp(1.8rem, 5vw, 2.8rem);
        font-weight: 800;
        color: white;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .author-info {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.15);
        flex-wrap: wrap;
    }
    
    .author-name {
        font-size: clamp(0.85rem, 2vw, 1rem);
        font-weight: 600;
        color: rgba(255,255,255,0.95);
        margin: 0;
    }
    
    .author-link {
        font-size: clamp(0.8rem, 1.8vw, 0.9rem);
        color: #fca5a5;
        text-decoration: none;
        padding: 0.3rem 0.8rem;
        background: rgba(252, 165, 165, 0.1);
        border-radius: 6px;
        transition: all 0.3s;
    }
    
    .author-link:hover {
        background: rgba(252, 165, 165, 0.2);
        color: #fecaca;
    }
    
    /* Player name styling - RESPONSIVE */
    .player-name {
        font-size: clamp(1.5rem, 4vw, 2.5rem);
        font-weight: 800;
        color: white;
        margin-bottom: clamp(1rem, 3vw, 2rem);
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #000000 !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: #000000 !important;
    }
    
    section[data-testid="stSidebar"] {
        background: #000000 !important;
    }
    
    [data-testid="stSidebar"] .element-container {
        background: transparent;
        padding: 0;
        margin-bottom: 1rem;
        border: none;
    }
    
    /* Section headers - RESPONSIVE */
    .section-header {
        font-size: clamp(1.1rem, 2.5vw, 1.3rem);
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(220, 38, 38, 0.5);
    }
    
    /* Metric cards - RESPONSIVE */
    [data-testid="stMetricValue"] {
        font-size: clamp(1.3rem, 3vw, 1.8rem);
        font-weight: 700;
        color: #fca5a5;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: clamp(0.7rem, 1.5vw, 0.85rem);
        font-weight: 600;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.3) 0%, rgba(220, 38, 38, 0.3) 100%);
        padding: clamp(0.8rem, 2vw, 1.2rem);
        border-radius: 12px;
        border: 1px solid rgba(220,38,38,0.3);
        box-shadow: 0 4px 20px rgba(220,38,38,0.2);
        margin-bottom: 0.8rem;
    }
    
    /* Player image container */
    .player-image-container {
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px solid rgba(220,38,38,0.4);
        box-shadow: 0 8px 24px rgba(220,38,38,0.3);
        text-align: center;
    }
    
    /* Contribution boxes - RESPONSIVE */
    .contribution-box {
        background: rgba(220,38,38,0.08);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(220,38,38,0.2);
        margin-bottom: 0.5rem;
    }
    
    .contribution-title {
        font-size: clamp(0.75rem, 1.5vw, 0.85rem);
        font-weight: 700;
        color: #fca5a5;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.8rem;
    }
    
    .contribution-item {
        font-size: clamp(0.8rem, 1.6vw, 0.9rem);
        color: rgba(255,255,255,0.85);
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(220,38,38,0.1);
    }
    
    /* Info section - RESPONSIVE */
    .info-card {
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
        padding: clamp(1.5rem, 3vw, 2rem);
        border-radius: 12px;
        border: 1px solid rgba(220,38,38,0.3);
        margin-bottom: 1.5rem;
    }
    
    .info-title {
        font-size: clamp(1.4rem, 3vw, 1.8rem);
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
    }
    
    .info-subtitle {
        font-size: clamp(1rem, 2.2vw, 1.2rem);
        font-weight: 600;
        color: #fca5a5;
        margin: 1.5rem 0 0.8rem 0;
    }
    
    .special-category {
        background: rgba(220,38,38,0.1);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc2626;
        margin: 0.8rem 0;
        font-size: clamp(0.85rem, 1.8vw, 1rem);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: clamp(0.5rem, 2vw, 1rem);
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(220,38,38,0.1);
        border-radius: 8px;
        padding: clamp(0.6rem, 1.5vw, 0.8rem) clamp(1rem, 2.5vw, 1.5rem);
        font-weight: 600;
        color: rgba(255,255,255,0.7);
        border: 1px solid rgba(220,38,38,0.2);
        font-size: clamp(0.85rem, 1.8vw, 1rem);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%);
        color: white;
    }
    
    /* Select box - BLACK background */
    .stSelectbox > div > div {
        background: #000000 !important;
        border: 1px solid rgba(100,100,100,0.3);
        color: white;
    }
    
    /* Select box dropdown */
    .stSelectbox [data-baseweb="select"] > div {
        background: #000000 !important;
    }

              
    
    /* RESPONSIVE: Mobile adjustments */
    @media (max-width: 768px) {
        .main-header {
            padding: 1.5rem 1rem;
            margin-bottom: 1.5rem;
        }
        
       
        
        div[data-testid="metric-container"] {
            margin-bottom: 0.5rem;
        }
        
        .contribution-title {
            font-size: 0.75rem;
        }
        
        .contribution-item {
            font-size: 0.8rem;
            padding: 0.3rem 0;
        }
        
        .info-card {
            padding: 1.5rem 1rem;
        }
        
        /* Stack columns on mobile */
        [data-testid="column"] {
            min-width: 100% !important;
        }
    }
    
    /* RESPONSIVE: Tablet adjustments */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main-title {
            font-size: 2.2rem;
        }
        
        .player-name {
            font-size: 2rem;
        }
    }
    
    /* RESPONSIVE: Large screen optimizations */
    @media (min-width: 1920px) {
        .main {
            max-width: 1920px;
            margin: 0 auto;
        }
    }
    
    /* Context info text - RESPONSIVE */
    .context-info {
        font-size: clamp(0.9rem, 2vw, 1.1rem) !important;
    }
</style>
""", unsafe_allow_html=True)



# ─────────────────────────────
# Function to plot field with labels AND legend
# ─────────────────────────────
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

        # Build aggregated arrays (average across lengths; missing treated as zero)
        agg_ev_run = []
        agg_ev_bd = []
        theta_centers = np.array(all_theta)

        for theta in theta_centers:
            run_vals = []
            bd_vals = []
            for ln in sel_lens:
                df = per_len_dfs.get(ln)
                if df is None:
                    run_vals.append(0.0)
                    bd_vals.append(0.0)
                else:
                    row = df.loc[df['theta_center_deg'] % 360 == theta]
                    if row.empty:
                        run_vals.append(0.0)
                        bd_vals.append(0.0)
                    else:
                        run_vals.append(float(row['ev_run'].values[0]))
                        bd_vals.append(float(row['ev_bd'].values[0]))
            agg_ev_run.append(sum(run_vals) / n_lens)
            agg_ev_bd.append(sum(bd_vals) / n_lens)
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
        
        colors_list = ['#1a0000', '#450a0a', '#991b1b', '#dc2626', '#f97316', '#fbbf24', '#fde047']
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


def create_zone_strength_table(dict_360, batter_name, selected_lengths, bowl_kind, kind):
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
                vals = [per_len_data[ln].get(key, 0.0) if per_len_data[ln] else 0.0 
                       for ln in sel_lens]
                aggregated[rc][key] = sum(vals) / len(sel_lens)

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
            vals = []
            for ln in sel_lens:
                per = per_len_shots.get(ln) or {}
                v = per.get(shot, {}).get(value_type, 0)
                vals.append(v)
            shots[shot] = sum(vals) / len(sel_lens)

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
        
        colors_list = ['#1a0000', '#450a0a', '#991b1b', '#dc2626', '#f97316', '#fbbf24', '#fde047']
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
        ['#1a0000', '#450a0a', '#991b1b', '#dc2626', '#f97316', '#fde047'],
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

    length_data = data["intrel_by_length"]

    # --- figure ---
    fig, ax = plt.subplots(figsize=(3, 4))
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
        '#1a0000', '#450a0a', '#991b1b',
        '#dc2626', '#f97316', '#fbbf24', '#fde047'
    ]
    modern_cmap = LinearSegmentedColormap.from_list(
        'modern_red', colors_list, N=256
    )

    norm = Normalize(vmin=min(intrels), vmax=max(intrels))
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
            fontsize=5,
            ha="center",
            va="center",
            fontweight="bold"
        )

    
    stump_x = [0.48, 0.50, 0.52]
    for x in stump_x:
        ax.plot([x, x], [0.9, 0.95], color="white", linewidth=3)
    
    # --- title ---
   

    return fig

    
# ─────────────────────────────
# Data loading
# ─────────────────────────────
@st.cache_data
def load_field_dict(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}
@st.cache_data
 

@st.cache_data
def load_players_data(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=['fullname', 'image_path'])
    
@st.cache_data
def load_ev_dict(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("EVs.bin file not found. Sector importance plot will not be available.")
        return {}


field_dict_t20 = load_field_dict('field_dict_global.bin')
players_df = load_players_data('players.csv')
ev_dict = load_ev_dict('EVs.bin')
dict_360 = load_ev_dict('bat_360.bin')
shot_per = load_ev_dict('shot_percent.bin')
avg_360 = load_ev_dict('bat_360_avg.bin')
intrel = load_ev_dict('intrel.bin')
sim_matrices = load_ev_dict("sim_mat.bin")
# Create a mapping of player names to image URLs
player_images = dict(zip(players_df['fullname'], players_df['image_path']))

# ─────────────────────────────
# Header
# ─────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 class="main-title">T20 Optimal Field Setting</h1>
    <div class="author-info">
        <span class="author-name">Arnav Jain | IITK</span>
        <a href="https://x.com/arnav1204aj" target="_blank" class="author-link">@arnav1204aj</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────
# Tabs
# ─────────────────────────────
tab1, tab2 = st.tabs(["Field Visualizer", "Information"])

with tab1:
    field_dict = field_dict_t20 

    # Sidebar selection
    # Sidebar selection
    with st.sidebar:
        st.markdown('<p class="section-header">Parameters</p>', unsafe_allow_html=True)
        
        with st.form(key='field_form'):
            submit = st.form_submit_button("Generate Results", use_container_width=True)
            # Batter selection
            batter_list = list(field_dict.keys())
            st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">Select Batter</p>', unsafe_allow_html=True)
            selected_batter = st.selectbox("Select Batter", batter_list, label_visibility="collapsed", key="batter")

            # Bowl kind selection
            bowl_kind_list = list(field_dict[selected_batter].keys())
            st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">Select Bowling Type</p>', unsafe_allow_html=True)
            selected_bowl_kind = st.selectbox("Select Bowling Type", bowl_kind_list, label_visibility="collapsed", key="bowl")

            # Length selection
            length_list = list(field_dict[selected_batter][selected_bowl_kind].keys())
            st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">Select Length(s)</p>', unsafe_allow_html=True)

            LENGTH_OPTIONS = ['FULL', 'SHORT', 'GOOD_LENGTH', 'SHORT_OF_A_GOOD_LENGTH']
            available_lengths = [l for l in LENGTH_OPTIONS if l in length_list]
            if not available_lengths:
                available_lengths = length_list

            selected_lengths = st.multiselect("Select Length(s)", available_lengths, default=[available_lengths[0]], label_visibility="collapsed", key="length")

            # Ensure at least one length is selected
            if not selected_lengths:
                st.warning('Please select at least one length.')
                selected_lengths = [available_lengths[0]]

            # Determine length_key (single or tuple)
            if len(selected_lengths) == 1:
                length_key = selected_lengths[0]
            else:
                selected_lengths_sorted = sorted(selected_lengths, key=lambda x: LENGTH_OPTIONS.index(x))
                length_key = tuple(selected_lengths_sorted)

            # Outfielder selection
            try:
                outfielder_list = list(field_dict[selected_batter][selected_bowl_kind][length_key].keys())
            except Exception:
                # Union of outfielders across selected lengths
                out_set = set()
                for ln in selected_lengths:
                    out_set.update(field_dict[selected_batter][selected_bowl_kind].get(ln, {}).keys())
                outfielder_list = sorted(list(out_set))

            st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">Select Outfielders</p>', unsafe_allow_html=True)
            selected_outfielders = st.selectbox("Select Outfielders", outfielder_list, label_visibility="collapsed", key="out")

            # Generate button
            



    if submit:
            
        
            # Use length_key to fetch field setup from field_dict
            try:
                data = field_dict[selected_batter][selected_bowl_kind][length_key][selected_outfielders]
            except KeyError:
                st.error("No field setting found for this combination.")
                raise

            # PLAYER IMAGE AND STATS ROW
            img_col, stats_col = st.columns([1, 2], vertical_alignment="center", gap="large")

            with img_col:
                player_img_url = player_images.get(
                    selected_batter,
                    "https://via.placeholder.com/300x300.png?text=No+Image"
                )

                st.markdown(
                    f"""
                    <div style="
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100%;
                        width: 100%;
                    ">
                        <div style="
                            width: 280px;
                            text-align: center;
                        ">
                            <img src="{player_img_url}"
                                style="
                                    width: 100%;
                                    border-radius: 12px;
                                    display: block;
                                    margin: 0 auto;
                                " /><p style="
                                margin-top: 12px;
                                margin-bottom: 0;
                                font-size: 1.5rem;
                                font-weight: 600;
                                text-align: center;
                            ">
                                {selected_batter}
                            </p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )




            with stats_col:
                

                st.markdown(
                    f'''
                    <p class="context-info" style="
                        color: rgba(255,255,255,0.7);
                        font-size: 1.1rem;
                        font-weight: 500;      
                        letter-spacing: 0.5px;
                    ">
                        {selected_bowl_kind} • {', '.join(selected_lengths)} • {selected_outfielders} outfielders
                    </p>
                    ''',
                    unsafe_allow_html=True
                )

                stats = data['protection_stats']

                # ─────────────────────────────────────────────
                # 📊 ROW 1 : 360 SCORES (Higher = Better)
                # ─────────────────────────────────────────────
                col1, col2 = st.columns(2)

                # Average 360 score across selected lengths (missing treated as 0)
                sel_lens = selected_lengths if isinstance(selected_lengths, list) else [selected_lengths]
                vals = []
                for ln in sel_lens:
                    try:
                        v = dict_360.get(selected_batter, {}).get(ln, {}).get(selected_bowl_kind, {}).get('overall').get('360_score', 0)
                    except Exception:
                        v = 0
                    vals.append(v)
                batter_360 = sum(vals) / len(sel_lens)

                vals = []
                for ln in sel_lens:
                    try:
                        v = avg_360.get('A', {}).get(ln, {}).get(selected_bowl_kind, {}).get('360_score', 0)
                    except Exception:
                        v = 0
                    vals.append(v)
                global_360 = sum(vals) / len(sel_lens)

                with col1:
                    st.metric(
                        "BATTER 360 SCORE",
                        f"{batter_360:.1f}",
                        delta=f"{batter_360 - global_360:.1f}"
                    )

                with col2:
                    st.metric(
                        "GLOBAL AVG (360)",
                        f"{global_360:.1f}"
                    )

                # ─────────────────────────────────────────────
                # 🏃 ROW 2 : RUNNING PROTECTION (Lower = Better)
                # ─────────────────────────────────────────────
                col3, col4 = st.columns(2)

                batter_run = stats.get('running', 0)
                # For protection stats, try composite key first, else average across lengths
                try:
                    global_run = field_dict['average batter'][selected_bowl_kind][length_key][selected_outfielders]['protection_stats']['running']
                except Exception:
                    vals = []
                    for ln in sel_lens:
                        try:
                            v = field_dict['average batter'][selected_bowl_kind][ln][selected_outfielders]['protection_stats']['running']
                        except Exception:
                            v = 0
                        vals.append(v)
                    global_run = sum(vals) / len(sel_lens)

                with col3:
                    st.metric(
                        "RUNNING PROTECTION",
                        f"{batter_run:.1f}%",
                        delta=f"{global_run - batter_run:.1f}%"
                    )

                with col4:
                    st.metric(
                        "GLOBAL AVG (RUN. PROT.)",
                        f"{global_run:.1f}%"
                    )

                # ─────────────────────────────────────────────
                # 🧱 ROW 3 : BOUNDARY PROTECTION (Lower = Better)
                # ─────────────────────────────────────────────
                col5, col6 = st.columns(2)

                batter_bd = stats.get('boundary', 0)
                try:
                    global_bd = field_dict['average batter'][selected_bowl_kind][length_key][selected_outfielders]['protection_stats']['boundary']
                except Exception:
                    vals = []
                    for ln in sel_lens:
                        try:
                            v = field_dict['average batter'][selected_bowl_kind][ln][selected_outfielders]['protection_stats']['boundary']
                        except Exception:
                            v = 0
                        vals.append(v)
                    global_bd = sum(vals) / len(sel_lens)

                with col5:
                    st.metric(
                        "BOUNDARY PROTECTION",
                        f"{batter_bd:.1f}%",
                        delta=f"{global_bd - batter_bd:.1f}%"
                    )

                with col6:
                    st.metric(
                        "GLOBAL AVG (BD. PROT.)",
                        f"{global_bd:.1f}%"
                    )

            st.markdown("---")

                        # ─────────────────────────────────────────────
            # 🔍 SIMILAR BATTERS
            # ─────────────────────────────────────────────
            
            # FIELD AND CONTRIBUTIONS
            col1, col2 = st.columns([1.6, 1.4])

            with col1:
                st.markdown('<p class="section-header">Field Placement</p>', unsafe_allow_html=True)
                
                try:
                    fig, inf_labels, out_labels = plot_field_setting(data)
                
                    st.pyplot(fig, use_container_width=True)
                except Exception:
                    st.warning('Unavailable')

                

            with col2:
                st.markdown('<p class="section-header">Fielder Contributions</p>', unsafe_allow_html=True)
                
                inf_contrib = data.get('infielder_ev_run_percent', [])
                out_contrib = data.get('outfielder_ev_bd_percent', [])

                inf_col, out_col = st.columns(2)

                with inf_col:
                    st.markdown('<p class="contribution-title">Infielders</p>', unsafe_allow_html=True)
                    if inf_contrib:
                        for f in inf_contrib:
                            angle = f["angle"]
                            label = inf_labels.get(angle, f"Angle {angle}°")
                            st.markdown(f'<div class="contribution-item">{label} → {f.get("ev_run_percent", 0):.1f}% runs saved</div>', unsafe_allow_html=True)
                    else:
                        st.write("No data available")

                with out_col:
                    st.markdown('<p class="contribution-title">Outfielders</p>', unsafe_allow_html=True)
                    if out_contrib:
                        for f in out_contrib:
                            angle = f["angle"]
                            label = out_labels.get(angle, f"Angle {angle}°")
                            st.markdown(f'<div class="contribution-item">{label} → {f.get("ev_bd_percent", 0):.1f}% runs saved</div>', unsafe_allow_html=True)
                    else:
                        st.write("No data available")
            st.markdown("---")
            
            # SECTOR IMPORTANCE PLOT
            if ev_dict and selected_batter in ev_dict:
                st.markdown('<p class="section-header">Sector Importance Analysis</p>', unsafe_allow_html=True)
                
                # Wrapper with flexbox
                st.markdown('<div style="display: flex; align-items: center; gap: 2rem;">', unsafe_allow_html=True)
                
                plot_col, info_col = st.columns([1.6, 1.4])
                
                with plot_col:
                   try: 
                    ev_fig = plot_sector_ev_heatmap(
                        ev_dict,
                        selected_batter,
                        selected_lengths,
                        selected_bowl_kind,
                        LIMIT=350,
                        THIRTY_YARD_RADIUS_M=171.25 * 350 / 500
                    )

                    if ev_fig:
                        st.pyplot(ev_fig, use_container_width=True)
                   except Exception:
                    st.warning('Unavailable')     
                
                with info_col:
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(153, 27, 27, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid rgba(220,38,38,0.3);
                        height: 100%;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    ">
                        <h3 style="color: #fca5a5; font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem;">
                            Understanding the Heatmap
                        </h3>
                        <p style="color: rgba(255,255,255,0.85); line-height: 1.7; font-size: 0.95rem; margin-bottom: 1rem;">
                            This polar heatmap shows the <strong>Importance (SR × Probability in that sector)</strong> of different sectors of the field.
                        </p>
                        <div style="margin: 0.8rem 0;">
                            <strong style="color: #fca5a5;">Inner Ring:</strong>
                            <span style="color: rgba(255,255,255,0.85);"> Running Class Runs</span>
                        </div>
                        <div style="margin: 0.8rem 0;">
                            <strong style="color: #fca5a5;">Outer Ring:</strong>
                            <span style="color: rgba(255,255,255,0.85);"> Boundary Class Runs</span>
                        </div>
                        <p style="color: rgba(255,255,255,0.75); font-size: 0.9rem; margin-top: 1rem; font-style: italic;">
                            Brighter colors indicate higher sector importance and thus a priority region for the fielding teams.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

                    # After the Sector Importance section, add:
            
            st.markdown("---")
            col1, col2 = st.columns([1, 2])
            with col2:
                st.markdown('<p class="section-header">Similar Batters</p>', unsafe_allow_html=True)

                

                sim_df = get_top_similar_batters(
                    sim_matrices=sim_matrices,
                    batter_name=selected_batter,
                    selected_lengths=selected_lengths,
                    bowl_kind=selected_bowl_kind,
                    top_n=5
                )

                if sim_df is None or sim_df.empty:
                    st.info("No similarity data available for this selection.")
                else:
                    try: 
                        fig = create_similarity_chart(
                            sim_df,
                            
                            selected_batter,
                            selected_lengths,
                            selected_bowl_kind
                        )

                        if fig:
                            st.pyplot(fig)
                    except Exception:
                            st.warning('Unavailable')
            with col1:
                st.markdown('<p class="section-header">Int-Con values by length</p>', unsafe_allow_html=True)
                try:
                    fig = plot_intrel_pitch(intrel,selected_batter,selected_lengths,selected_bowl_kind,10)     
                    st.pyplot(fig)
                except Exception:
                            st.warning('Unavailable') 
            st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(153, 27, 27, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid rgba(220,38,38,0.3);
                        height: 100%;
                    ">
                        <h3 style="color: #fca5a5; font-size: 1.2rem; font-weight: 700; margin-top: 0;">
                            Understanding Batter Similarity and Int-Con Values
                        </h3>
                        <p style="color: rgba(255,255,255,0.85); line-height: 1.7; font-size: 0.95rem;">
                            Batter Similarity a vector based similarity score considering shots, 
                            zones, control%, boundary% on different lines, lengths and bowler kinds.
                        </p>
                                <p style="color: rgba(255,255,255,0.85); line-height: 1.7; font-size: 0.95rem;">
                            Int-Con is an intent-control measuring metric. It is a multiplication of SRs and Control%
                            the batter achieves compared to other batters in the same innings. So a value of 1.20 for example means
                            the batter was 20% better, 0.8 means 20% worse, 1 is average performance.
                        </p>
                    
                    </div>
                    """, unsafe_allow_html=True)
                

            st.markdown("---")
            
            # RELATIVE ZONE STRENGTHS
            if dict_360 and selected_batter in dict_360:
                  try:  
                    st.markdown('<p class="section-header">Relative Zone Strengths</p>', unsafe_allow_html=True)

                    reg_col, avg_col = st.columns([1.5, 1.5], gap="small")

                    # -------- LEFT: TABLE --------
                    with reg_col:
                        st.markdown(f'<p class="subsection-header">Batter\'s Run Distribution</p>', unsafe_allow_html=True)
                        zone_fig, zone_data = create_zone_strength_table(
                            dict_360,
                            selected_batter,
                            selected_lengths,
                            selected_bowl_kind,
                            'runs'
                        )
                        if zone_fig:
                            st.pyplot(zone_fig, use_container_width=True)
                    with avg_col:  
                        st.markdown('<p class="subsection-header">Avg Batter\'s Run Distribution</p>', unsafe_allow_html=True)      
                        zone_fig, zone_data = create_zone_strength_table(
                            dict_360,
                            selected_batter,
                            selected_lengths,
                            selected_bowl_kind,
                            'avg_runs'
                        )
                        if zone_fig:
                            st.pyplot(zone_fig, use_container_width=True) 
                  except Exception:
                    st.warning('Unavailable')

                  try:  
                    
                    st.markdown('<p class="section-header">Relative Shot Strengths</p>', unsafe_allow_html=True)

                    reg_col, avg_col = st.columns([1.5, 1.5], gap="small")        
                    
                    with reg_col:
                            st.markdown(f'<p class="subsection-header">Batter\'s Run Distribution</p>', unsafe_allow_html=True)
                            shot_fig = create_shot_profile_chart(
                                shot_per,
                                selected_batter,
                                selected_lengths,
                                selected_bowl_kind,
                                value_type="runs"
                            )
                            if shot_fig:
                                st.pyplot(shot_fig, use_container_width=True) 

                    with avg_col:
                            st.markdown('<p class="subsection-header">Avg Batter\'s Run Distribution</p>', unsafe_allow_html=True)
                            shot_fig = create_shot_profile_chart(
                                shot_per,
                                selected_batter,
                                selected_lengths,
                                selected_bowl_kind,
                                value_type="avg_runs"
                            )
                            if shot_fig:
                                st.pyplot(shot_fig, use_container_width=True)           
                  except Exception:
                    st.warning('Unavailable')
                    # -------- RIGHT: EXPLAINER --------
                  try:  
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(153, 27, 27, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid rgba(220,38,38,0.3);
                        height: 100%;
                    ">
                        <h3 style="color: #fca5a5; font-size: 1.2rem; font-weight: 700; margin-top: 0;">
                            Understanding Zone and Shot Strengths
                        </h3>
                        <p style="color: rgba(255,255,255,0.85); line-height: 1.7; font-size: 0.95rem;">
                            The charts show how the batter distributes their runs across four key regions and different shots.<strong> To understand the batter's true strength 
                            in a particular region or playing a particular shot, we compare his distributions to an 
                            average batter's distributions </strong>. Average batter's calculations are done on the same 
                            line-length distribution the batter has faced in his career. The calculations consider run scoring
                            difficulty of a region or shot for the given line-length-bathand-pace/spin combination. 
                        </p>
                                <p style="color: rgba(255,255,255,0.85); line-height: 1.7; font-size: 0.95rem;">
                            The drives include both lofted and grounded drives.
                        </p>
                    
                    </div>
                    """, unsafe_allow_html=True)
                  except Exception:
                    st.warning('Unavailable')


            
                
                
                
                
                
                    # Add spacer for vertical centering
                
        
            

        
    
    else:
        st.info("Please select parameters and click **Generate Results**")
        
# ─────────────────────────────
# Info Tab
# ─────────────────────────────
with tab2:
    st.markdown("""
    <div class="info-card">
        <h2 class="info-title">Methodology</h2>
        <p style="color: rgba(255,255,255,0.85); line-height: 1.8; font-size: 1rem;">
        A sector-based distribution of the batter's running and boundary runs is calculated. 
        Fielders are then placed <strong>greedily</strong> using <strong>Dynamic Programming</strong>, 
        keeping in mind the field restrictions and rules, to maximize the overall protection percentage. 
        Each fielder protects runs in their coverage area. <strong style="color: #26F7FD;">Protection stats show the percentage of runs saved
        by the optimal field (Running for running class runs, Boundary for boundary class runs, less protection is better for the batter).</strong> Infielders protect running runs,
        while outfielders protect boundary runs.</p><h3 class="info-subtitle">Special Fielder Categories</h3><div class="special-category">
            <strong style="color: #dc2626;">30 Yard Wall</strong> — Your best infielder, placed where most grounded shots are expected.
        </div><div class="special-category">
            <strong style="color: #f97316;">Sprinter</strong> — The best runner, placed where batters tend to hit and run singles/doubles in the outfield.
        </div><div class="special-category">
            <strong style="color: #84cc16;">Catcher</strong> — The best catcher, placed where batters hit the most boundaries.
        </div><div class="special-category">
            <strong style="color: #fbbf24;">Superfielder</strong> — A combination of sprinter and catcher, used if both positions coincide.
        </div><h3 class="info-subtitle">Further Reading</h3>
        <p style="color: rgba(255,255,255,0.85); line-height: 1.8; font-size: 1rem;">
        For a detailed explanation of the methodology, read the full article on Substack: 
        <a href="https://arnavj.substack.com/p/the-sacred-nine-spots" target="_blank" 
           style="color: #fca5a5; text-decoration: none; font-weight: 600;">
           The Sacred Nine Spots
        </a>
        </p>
    </div>
    """, unsafe_allow_html=True)







