import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle  # Import the pickle library to load binary files
import os

# --- Plotting Function ---
# This function contains the field visualization logic from your original code.
def plot_field_setting(field_data):
    """Generates a matplotlib plot of the cricket field setting."""
    LIMIT = 350
    THIRTY_YARD_RADIUS_M = 171.25 * LIMIT / 500
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('darkgreen')

    # Unpack data from the dictionary
    inside_info = field_data['infielder_positions']
    outside_info = field_data['outfielder_positions']
    special_fielders = field_data['special_fielders']

    # --- Plot Infielders and 30-Yard Wall ---
    wall_angle = special_fielders.get('30_yard_wall')
    for angle in inside_info:
        ang_rad = np.deg2rad(angle)
        is_wall = (angle == wall_angle)
        marker_props = {
            'c': 'red' if is_wall else 'cyan', 's': 250 if is_wall else 150,
            'marker': 's' if is_wall else 'o', 'edgecolors': 'black',
            'linewidth': 2 if is_wall else 1.5,
            'label': '30-Yard Wall' if is_wall else None
        }
        ax.scatter(THIRTY_YARD_RADIUS_M * np.sin(ang_rad), THIRTY_YARD_RADIUS_M * np.cos(ang_rad), **marker_props)

    # --- Plot Outfielders and Special Roles ---
    sprinter_angle = special_fielders.get('sprinter')
    catcher_angle = special_fielders.get('catcher')
    superfielder_angle = special_fielders.get('superfielder')

    for angle in outside_info:
        ang_rad = np.deg2rad(angle)
        props = {}
        if angle == superfielder_angle:
            props = {'c': 'gold', 'marker': 'D', 'label': 'Superfielder'}
        elif angle == sprinter_angle:
            props = {'c': 'orange', 'marker': '^', 'label': 'Sprinter'}
        elif angle == catcher_angle:
            props = {'c': 'lime', 'marker': '*', 'label': 'Catcher'}
        else:
            props = {'c': 'magenta', 'marker': 'o', 's': 150, 'label': None}
        
        base_props = {'s': 250 if props.get('label') else 150, 'edgecolors': 'black', 'linewidth': 2 if props.get('label') else 1.5}
        ax.scatter(LIMIT * np.sin(ang_rad), LIMIT * np.cos(ang_rad), **{**base_props, **props})

    # --- Draw Field Markings ---
    ax.add_artist(plt.Circle((0, 0), THIRTY_YARD_RADIUS_M+20, color='white', fill=False, linestyle='--', linewidth=2))
    ax.add_artist(plt.Circle((0, 0), LIMIT+20, color='white', fill=False, linewidth=2.5))
    ax.add_artist(plt.Rectangle((-10, -50), 20, 100, facecolor='burlywood', zorder=0)) # Pitch
    ax.arrow(0, 40, 0, -60, width=15, head_width=35, head_length=25, fc='white', ec='black', zorder=10)
    ax.text(0, 60, 'Batter Facing', ha='center', va='top', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.3, boxstyle='round,pad=0.3'))
    for angle in [0, 90, 180, 270]:
        angle_rad = np.deg2rad(angle)
        # Position the text slightly outside the boundary line
        text_radius = LIMIT + 35 
        x = text_radius * np.sin(angle_rad)
        y = text_radius * np.cos(angle_rad)
        ax.text(x, y, f'{angle}°', color='white', ha='center', va='center',
                fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=2))
    
    ax.set_xlim(-(LIMIT + 50), LIMIT + 50)
    ax.set_ylim(-(LIMIT + 50), LIMIT + 50)
    ax.set_aspect('equal')
    ax.legend(facecolor='lightgray')
    ax.set_xticks([]); ax.set_yticks([])

    return fig

# --- STREAMLIT APP ---

st.set_page_config(layout="wide")
st.title("🏏 Optimal Cricket Field Setting")

# --- Load data from the binary file 'field_dict.bin' ---
try:
    with open('field_dict.bin', 'rb') as f:
        field_dict = pickle.load(f)
except FileNotFoundError:
    st.error("Error: The data file 'field_dict.bin' was not found.")
    st.info("Please make sure 'field_dict.bin' is in the same directory as your Streamlit app.")
    st.stop() # Stop the app from running further

# --- Sidebar for User Selections ---
with st.sidebar:
    st.header("Field Parameters")
    
    # 1. Select Batter
    batter_list = list(field_dict.keys())
    selected_batter = st.selectbox("Select Batter", batter_list)
    
    # 2. Select Bowl Kind (dynamically)
    bowl_kind_list = list(field_dict.get(selected_batter, {}).keys())
    if not bowl_kind_list:
        st.warning("No data available for this batter.")
        st.stop()
    selected_bowl_kind = st.selectbox("Select Bowl Kind", bowl_kind_list)

    # 3. Select Length (dynamically)
    length_list = list(field_dict.get(selected_batter, {}).get(selected_bowl_kind, {}).keys())
    if not length_list:
        st.warning("No data available for this combination.")
        st.stop()
    selected_length = st.selectbox("Select Length", length_list)

    # 4. Select Outfielders (dynamically)
    outfielder_list = list(field_dict.get(selected_batter, {}).get(selected_bowl_kind, {}).get(selected_length, {}).keys())
    if not outfielder_list:
        st.warning("No data available for this combination.")
        st.stop()
    selected_outfielders = st.selectbox("Select Number of Outfielders", outfielder_list)

# --- Main Panel for Displaying Results ---

# Retrieve the specific dictionary for the selected options
try:
    data = field_dict[selected_batter][selected_bowl_kind][selected_length][selected_outfielders]
    
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Optimal Field Placement")
        # Display the field plot
        fig = plot_field_setting(data)
        st.pyplot(fig)

    with col2:
        st.subheader("Protection Statistics")
        
        # Display Overall Protection Stats using st.metric
        stats = data['protection_stats']
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Protection", f"{stats.get('overall', 0):.1f}%")
        c2.metric("Running Protection", f"{stats.get('running', 0):.1f}%")
        c3.metric("Boundary Protection", f"{stats.get('boundary', 0):.1f}%")

        st.markdown("---")

        # Display Individual Fielder Contributions
        st.subheader("Individual Fielder Contribution")
        
        with st.expander("Infielders (vs. Total Running EV)"):
            for fielder in data.get('infielder_ev_run_percent', []):
                st.write(f"**Angle {fielder['angle']}°:** Saves {fielder.get('ev_run_percent', 0):.1f}%")

        with st.expander("Outfielders (vs. Total Boundary EV)"):
            for fielder in data.get('outfielder_ev_bd_percent', []):
                st.write(f"**Angle {fielder['angle']}°:** Saves {fielder.get('ev_bd_percent', 0):.1f}%")

except KeyError:
    st.error("No data available for the selected combination. Please make another selection.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

