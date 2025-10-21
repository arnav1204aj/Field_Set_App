import streamlit as st
st.set_page_config(layout="wide", page_title="🏏 Optimal Field Setting")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


# ─────────────────────────────
# Function to plot field
# ─────────────────────────────
def plot_field_setting(field_data):
    LIMIT = 350
    THIRTY_YARD_RADIUS_M = 171.25 * LIMIT / 500

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('darkgreen')

    inside_info = field_data['infielder_positions']
    outside_info = field_data['outfielder_positions']
    special_fielders = field_data['special_fielders']

    wall_angle = special_fielders.get('30_yard_wall')
    for angle in inside_info:
        ang_rad = np.deg2rad(angle)
        is_wall = (angle == wall_angle)
        ax.scatter(
            THIRTY_YARD_RADIUS_M * np.sin(ang_rad),
            THIRTY_YARD_RADIUS_M * np.cos(ang_rad),
            c='red' if is_wall else 'cyan',
            s=250 if is_wall else 150,
            marker='s' if is_wall else 'o',
            edgecolors='black',
            linewidth=2 if is_wall else 1.5,
            label='30-Yard Wall' if is_wall else None
        )

    sprinter_angle = special_fielders.get('sprinter')
    catcher_angle = special_fielders.get('catcher')
    superfielder_angle = special_fielders.get('superfielder')

    for angle in outside_info:
        ang_rad = np.deg2rad(angle)
        props = {'c': 'magenta', 'marker': 'o', 'label': None}
        if angle == superfielder_angle:
            props = {'c': 'gold', 'marker': 'D', 'label': 'Superfielder'}
        elif angle == sprinter_angle:
            props = {'c': 'orange', 'marker': '^', 'label': 'Sprinter'}
        elif angle == catcher_angle:
            props = {'c': 'lime', 'marker': '*', 'label': 'Catcher'}

        ax.scatter(
            LIMIT * np.sin(ang_rad),
            LIMIT * np.cos(ang_rad),
            s=250 if props['label'] else 150,
            edgecolors='black',
            linewidth=2 if props['label'] else 1.5,
            **props
        )

    ax.add_artist(plt.Circle((0, 0), THIRTY_YARD_RADIUS_M+20, color='white', fill=False, linestyle='--', linewidth=2))
    ax.add_artist(plt.Circle((0, 0), LIMIT+20, color='white', fill=False, linewidth=2.5))
    ax.add_artist(plt.Rectangle((-10, -50), 20, 100, facecolor='burlywood', zorder=0))
    ax.arrow(0, 40, 0, -60, width=15, head_width=35, head_length=25, fc='white', ec='black', zorder=10)
    ax.text(0, 60, 'Batter Facing', ha='center', va='top', color='white', fontsize=12,
            bbox=dict(facecolor='black', alpha=0.3, boxstyle='round,pad=0.3'))

    for angle in [0, 90, 180, 270]:
        angle_rad = np.deg2rad(angle)
        ax.text((LIMIT + 35) * np.sin(angle_rad), (LIMIT + 35) * np.cos(angle_rad),
                f'{angle}°', color='white', ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, pad=2))
    
    ax.set_xlim(-(LIMIT + 50), LIMIT + 50)
    ax.set_ylim(-(LIMIT + 50), LIMIT + 50)
    ax.set_aspect('equal')
    ax.legend(facecolor='lightgray')
    ax.set_xticks([]); ax.set_yticks([])

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

field_dict_t20 = load_field_dict('field_dict_global.bin')
field_dict_odi = load_field_dict('field_dict_odi.bin')


# ─────────────────────────────
# App layout
# ─────────────────────────────
st.title("🏏 Optimal Cricket Field Setting")

# Unified format selector
format_choice = st.radio("Select Format", ["T20", "ODI"], horizontal=True)

# Load correct dataset
field_dict = field_dict_t20 if format_choice == "T20" else field_dict_odi
if not field_dict:
    st.error(f"No data available for {format_choice}.")
    st.stop()

# Sidebar selection
with st.sidebar:
    st.header(f"{format_choice} Parameters")

    batter_list = list(field_dict.keys())
    selected_batter = st.selectbox("Select Batter", batter_list)

    bowl_kind_list = list(field_dict[selected_batter].keys())
    selected_bowl_kind = st.selectbox("Select Bowling Type", bowl_kind_list)

    length_list = list(field_dict[selected_batter][selected_bowl_kind].keys())
    selected_length = st.selectbox("Select Length", length_list)

    outfielder_list = list(field_dict[selected_batter][selected_bowl_kind][selected_length].keys())
    selected_outfielders = st.selectbox("Select Outfielders", outfielder_list)


# ─────────────────────────────
# Display section
# ─────────────────────────────
try:
    data = field_dict[selected_batter][selected_bowl_kind][selected_length][selected_outfielders]

    # ROW 1 → Field Plot (left) + Fielder Contributions (right)
    col1, col2 = st.columns([1.6, 1.4])

    with col1:
        st.subheader("🟢 Field Placement")
        fig = plot_field_setting(data)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("🧤 Fielder Contributions")

        inf_contrib = data.get('infielder_ev_run_percent', [])
        out_contrib = data.get('outfielder_ev_bd_percent', [])

        inf_col, out_col = st.columns(2)

        with inf_col:
            st.markdown("**Infielders (Running Saved %)**")
            if inf_contrib:
                for f in inf_contrib:
                    st.write(f"Angle {f['angle']}° → `{f.get('ev_run_percent', 0):.1f}%`")
            else:
                st.write("_No infielder data_")

        with out_col:
            st.markdown("**Outfielders (Boundary Saved %)**")
            if out_contrib:
                for f in out_contrib:
                    st.write(f"Angle {f['angle']}° → `{f.get('ev_bd_percent', 0):.1f}%`")
            else:
                st.write("_No outfielder data_")

    st.markdown("---")

    # ROW 2 → Protection Stats
    stats = data['protection_stats']
    c1, c2, c3 = st.columns(3)
    c1.metric("🛡️ Overall Protection", f"{stats.get('overall', 0):.1f}%")
    c2.metric("🏃‍♂️ Running Protection", f"{stats.get('running', 0):.1f}%")
    c3.metric("💥 Boundary Protection", f"{stats.get('boundary', 0):.1f}%")

except KeyError:
    st.error("No data for this combination.")
except Exception as e:
    st.error(f"Unexpected error: {e}")






