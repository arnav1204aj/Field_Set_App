import streamlit as st
st.set_page_config(layout="wide", page_title="Optimal Field Setting | Cricket Analytics")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
    LIMIT = 350
    THIRTY_YARD_RADIUS_M = 171.25 * LIMIT / 500

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#1a0a0a')
    ax.set_facecolor('#166534')

    inside_info = field_data['infielder_positions']
    outside_info = field_data['outfielder_positions']
    special_fielders = field_data['special_fielders']

    # Create fielder labels
    infielder_labels = {}
    outfielder_labels = {}
    
    wall_angle = special_fielders.get('30_yard_wall')
    for idx, angle in enumerate(inside_info):
        ang_rad = np.deg2rad(angle)
        is_wall = (angle == wall_angle)
        label = f"I{idx+1}"
        infielder_labels[angle] = label
        
        ax.scatter(
            THIRTY_YARD_RADIUS_M * np.sin(ang_rad),
            THIRTY_YARD_RADIUS_M * np.cos(ang_rad),
            c='#dc2626' if is_wall else '#06b6d4',
            s=400 if is_wall else 250,
            marker='s' if is_wall else 'o',
            edgecolors='white',
            linewidth=2.5 if is_wall else 2,
            alpha=0.95,
            zorder=5,
            label='30-Yard Wall' if is_wall else None
        )
        
        # Add label text
        ax.text(
            THIRTY_YARD_RADIUS_M * np.sin(ang_rad),
            THIRTY_YARD_RADIUS_M * np.cos(ang_rad),
            label,
            ha='center',
            va='center',
            color='white',
            fontsize=10,
            fontweight='bold',
            zorder=6
        )

    sprinter_angle = special_fielders.get('sprinter')
    catcher_angle = special_fielders.get('catcher')
    superfielder_angle = special_fielders.get('superfielder')

    for idx, angle in enumerate(outside_info):
        ang_rad = np.deg2rad(angle)
        label = f"O{idx+1}"
        outfielder_labels[angle] = label
        
        props = {'c': '#d946ef', 'marker': 'o', 'label': None}
        if angle == superfielder_angle:
            props = {'c': '#fbbf24', 'marker': 'D', 'label': 'Superfielder'}
        elif angle == sprinter_angle:
            props = {'c': '#f97316', 'marker': '^', 'label': 'Sprinter'}
        elif angle == catcher_angle:
            props = {'c': '#84cc16', 'marker': '*', 'label': 'Catcher'}

        ax.scatter(
            LIMIT * np.sin(ang_rad),
            LIMIT * np.cos(ang_rad),
            s=400 if props['label'] else 250,
            edgecolors='white',
            linewidth=2.5 if props['label'] else 2,
            alpha=0.95,
            zorder=5,
            **props
        )
        
        # Add label text
        ax.text(
            LIMIT * np.sin(ang_rad),
            LIMIT * np.cos(ang_rad),
            label,
            ha='center',
            va='center',
            color='white',
            fontsize=10,
            fontweight='bold',
            zorder=6
        )

    ax.add_artist(plt.Circle((0, 0), THIRTY_YARD_RADIUS_M+20, color='white', fill=False, linestyle='--', linewidth=2.5, alpha=0.6))
    ax.add_artist(plt.Circle((0, 0), LIMIT+20, color='white', fill=False, linewidth=3, alpha=0.8))
    ax.add_artist(plt.Rectangle((-10, -50), 20, 100, facecolor='#92400e', zorder=0, alpha=0.9))
    ax.arrow(0, 40, 0, -60, width=15, head_width=35, head_length=25, fc='white', ec='white', zorder=10, alpha=0.9)
    ax.text(0, 60, 'BATTER FACING', ha='center', va='top', color='white', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='#2d1414', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='white', linewidth=1.5))

    for angle in [0, 90, 180, 270]:
        angle_rad = np.deg2rad(angle)
        ax.text((LIMIT + 40) * np.sin(angle_rad), (LIMIT + 40) * np.cos(angle_rad),
                f'{angle}°', color='white', ha='center', va='center', fontweight='bold', fontsize=10,
                bbox=dict(facecolor='#2d1414', alpha=0.9, pad=4, edgecolor='white', linewidth=1.5))
    
    # Add legend
    ax.legend(facecolor='#2d1414', edgecolor='white', framealpha=0.95, loc='upper right', fontsize=9, labelcolor='white')
    
    ax.set_xlim(-(LIMIT + 50), LIMIT + 50)
    ax.set_ylim(-(LIMIT + 50), LIMIT + 50)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, infielder_labels, outfielder_labels

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
def load_players_data(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=['fullname', 'image_path'])

field_dict_t20 = load_field_dict('field_dict_global.bin')
players_df = load_players_data('players.csv')

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
    with st.sidebar:
        st.markdown('<p class="section-header">Parameters</p>', unsafe_allow_html=True)

        batter_list = list(field_dict.keys())
        st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">Select Batter</p>', unsafe_allow_html=True)
        selected_batter = st.selectbox("Select Batter", batter_list, label_visibility="collapsed", key="batter")

        bowl_kind_list = list(field_dict[selected_batter].keys())
        st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">Select Bowling Type</p>', unsafe_allow_html=True)
        selected_bowl_kind = st.selectbox("Select Bowling Type", bowl_kind_list, label_visibility="collapsed", key="bowl")

        length_list = list(field_dict[selected_batter][selected_bowl_kind].keys())
        st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">Select Length</p>', unsafe_allow_html=True)
        selected_length = st.selectbox("Select Length", length_list, label_visibility="collapsed", key="length")

        outfielder_list = list(field_dict[selected_batter][selected_bowl_kind][selected_length].keys())
        st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">Select Outfielders</p>', unsafe_allow_html=True)
        selected_outfielders = st.selectbox("Select Outfielders", outfielder_list, label_visibility="collapsed", key="out")

    # Display section
    try:
        data = field_dict[selected_batter][selected_bowl_kind][selected_length][selected_outfielders]

        # PLAYER IMAGE AND STATS ROW
        img_col, stats_col = st.columns([1, 2], vertical_alignment="center", gap="large")

        with img_col:
            # Get player image URL
            player_img_url = player_images.get(selected_batter, "https://via.placeholder.com/300x300.png?text=No+Image")
            st.image(player_img_url, use_container_width=True)

        with stats_col:
            st.markdown(f'<h1 class="player-name">{selected_batter}</h1>', unsafe_allow_html=True)
            st.markdown(f'''
            <p class="context-info" style="
                color: rgba(255,255,255,0.7);
                font-size: 1.1rem;
                font-weight: 500;      
                letter-spacing: 0.5px;
                margin-bottom: 1rem;
            ">
                {selected_bowl_kind} • {selected_length} • {selected_outfielders} outfielders
            </p>
            ''', unsafe_allow_html=True)
            stats = data['protection_stats']
            st.metric("OVERALL PROTECTION", f"{stats.get('overall', 0):.1f}%")
            st.metric("RUNNING PROTECTION", f"{stats.get('running', 0):.1f}%")
            st.metric("BOUNDARY PROTECTION", f"{stats.get('boundary', 0):.1f}%")

        st.markdown("---")

        # FIELD AND CONTRIBUTIONS
        col1, col2 = st.columns([1.6, 1.4])

        with col1:
            st.markdown('<p class="section-header">Field Placement</p>', unsafe_allow_html=True)
            fig, inf_labels, out_labels = plot_field_setting(data)
            st.pyplot(fig, use_container_width=True)

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

    except KeyError:
        st.error("No data available for this combination.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

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
        by the optimal field (Running for running class runs, Boundary for boundary class runs, Overall for both).</strong> Infielders protect running runs,
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







