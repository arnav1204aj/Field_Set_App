import streamlit as st
st.set_page_config(layout="wide", page_title="Optimal Field Setting | Cricket Analytics")

import pandas as pd
from functions import plot_int_wagons, plot_intent_impact, plot_field_setting, plot_intrel_pitch, plot_intrel_pitch_avg, plot_sector_ev_heatmap, create_shot_profile_chart, create_similarity_chart, create_zone_strength_table, get_top_similar_batters

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
        font-size: clamp(2rem, 5vw, 2.8rem);
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
        font-size: clamp(1.8rem, 4vw, 2.5rem);
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
        font-size: clamp(1.2rem, 2.5vw, 1.3rem);
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

    .player-img-wrapper {
    width: clamp(300px, 35vw, 450px);
    margin: 0 auto;
    }

    .player-img-wrapper img {
    width: 100%;
    border-radius: 16px;
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
        color: white;
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
        font-size: clamp(1.5rem, 3vw, 1.8rem);
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
        font-size: clamp(1rem, 2vw, 1.1rem) !important;
    }
</style>
""", unsafe_allow_html=True)





    
# ─────────────────────────────
# Data loading with Women's Mode Support
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
    
@st.cache_data
def load_ev_dict(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


# Initialize session state for efficient data switching
if 'current_mode' not in st.session_state:
    st.session_state['current_mode'] = None  # None until user selects

if 'loaded_data' not in st.session_state:
    st.session_state['loaded_data'] = {}

# Define mode constants
MODES = {
    'MENS_T20': 'Men\'s T20',
    'WOMENS_T20': 'Women\'s T20',
    'MENS_ODI': 'Men\'s ODI'
}

def get_data_paths(mode):
    """
    Returns dict of all data file paths based on mode (MENS_T20, WOMENS_T20, MENS_ODI).
    Efficient: avoids redundant path construction.
    """
    # Determine prefix based on mode
    if mode == 'WOMENS_T20':
        prefix = 'w'
    elif mode == 'MENS_ODI':
        prefix = 'odi_'
    else:  # MENS_T20
        prefix = ''
    
    return {
        'field_dict': f'{prefix}field_dict_global.bin',
        'ev': f'{prefix}EVs.bin',
        'dict_360': f'{prefix}bat_360.bin',
        'shot_per': f'{prefix}shot_percent.bin',
        'length_dict': f'{prefix}length_dict.bin',
        'avg_360': f'{prefix}bat_360_avg.bin',
        'intrel': f'{prefix}intrel.bin',
        'sim_matrices': f'{prefix}sim_mat.bin',
        'intent_impact': f'{prefix}intent_impact.bin',
        'intel_ww': f'{prefix}intel_ww.bin',
        'players': 'players.csv'  # Same for all modes
    }

def load_all_data(mode):
    """
    Load all data files for the selected mode (MENS_T20, WOMENS_T20, MENS_ODI).
    Uses session state caching to avoid redundant file I/O on each toggle.
    """
    # Return cached data if already loaded for this mode
    if mode in st.session_state['loaded_data']:
        return st.session_state['loaded_data'][mode]
    
    paths = get_data_paths(mode)
    
    # Load all data files
    data = {
        'field_dict': load_field_dict(paths['field_dict']),
        'players_df': load_players_data(paths['players']),
        'ev_dict': load_ev_dict(paths['ev']),
        'dict_360': load_ev_dict(paths['dict_360']),
        'shot_per': load_ev_dict(paths['shot_per']),
        'length_dict': load_ev_dict(paths['length_dict']),
        'avg_360': load_ev_dict(paths['avg_360']),
        'intrel': load_ev_dict(paths['intrel']),
        'sim_matrices': load_ev_dict(paths['sim_matrices']),
        'intent_impact': load_ev_dict(paths['intent_impact']),
        'intel_ww': load_ev_dict(paths['intel_ww'])
    }
    
    # Cache the loaded data in session state for fast switching
    st.session_state['loaded_data'][mode] = data
    return data

# Load initial mens T20 data
initial_data = load_all_data(mode='MENS_T20')
field_dict_t20 = initial_data['field_dict']
players_df = initial_data['players_df']
ev_dict = initial_data['ev_dict']
dict_360 = initial_data['dict_360']
shot_per = initial_data['shot_per']
length_dict = initial_data['length_dict']
avg_360 = initial_data['avg_360']
intrel = initial_data['intrel']
sim_matrices = initial_data['sim_matrices']
intent_impact = initial_data['intent_impact']
intel_ww = initial_data['intel_ww']
# Create a mapping of player names to image URLs
player_images = dict(zip(players_df['fullname'], players_df['image_path']))

# ─────────────────────────────
# Mode Selection Interface (Required on Entry)
# ─────────────────────────────

# Display mode selection if no mode is selected
if st.session_state['current_mode'] is None:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(220,38,38,0.4);
        border: 1px solid rgba(255,255,255,0.1);
        margin: 2rem 0;
    ">
        <h2 style="
            font-size: 2rem;
            font-weight: 800;
            color: white;
            margin-bottom: 1rem;
        ">Select Game Mode</h2>
        <p style="
            font-size: 1.1rem;
            color: rgba(255,255,255,0.9);
            margin-bottom: 2rem;
        ">Choose the format you want to analyze</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode selection buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("MEN'S T20", use_container_width=True, key='mode_mens_t20'):
            st.session_state['current_mode'] = 'MENS_T20'
            st.rerun()
    
    with col2:
        if st.button("WOMEN'S T20", use_container_width=True, key='mode_womens_t20'):
            st.session_state['current_mode'] = 'WOMENS_T20'
            st.rerun()
    
    with col3:
        if st.button("MEN'S ODI", use_container_width=True, key='mode_mens_odi'):
            st.session_state['current_mode'] = 'MENS_ODI'
            st.rerun()
    
    st.stop()  # Stop execution until mode is selected

# ─────────────────────────────
# Update Global Data for Selected Mode
# ─────────────────────────────

current_mode = st.session_state['current_mode']

# Check if mode changed and load new data
if 'previous_mode' not in st.session_state or st.session_state['previous_mode'] != current_mode:
    data = load_all_data(mode=current_mode)
    st.session_state['current_field_dict'] = data['field_dict']
    st.session_state['current_players_df'] = data['players_df']
    st.session_state['current_ev_dict'] = data['ev_dict']
    st.session_state['current_dict_360'] = data['dict_360']
    st.session_state['current_shot_per'] = data['shot_per']
    st.session_state['current_length_dict'] = data['length_dict']
    st.session_state['current_avg_360'] = data['avg_360']
    st.session_state['current_intrel'] = data['intrel']
    st.session_state['current_sim_matrices'] = data['sim_matrices']
    st.session_state['current_intent_impact'] = data['intent_impact']
    st.session_state['current_intel_ww'] = data['intel_ww']
    st.session_state['previous_mode'] = current_mode

# Assign from session state
field_dict_t20 = st.session_state.get('current_field_dict', initial_data['field_dict'])
players_df = st.session_state.get('current_players_df', initial_data['players_df'])
ev_dict = st.session_state.get('current_ev_dict', initial_data['ev_dict'])
dict_360 = st.session_state.get('current_dict_360', initial_data['dict_360'])
shot_per = st.session_state.get('current_shot_per', initial_data['shot_per'])
length_dict = st.session_state.get('current_length_dict', initial_data['length_dict'])
avg_360 = st.session_state.get('current_avg_360', initial_data['avg_360'])
intrel = st.session_state.get('current_intrel', initial_data['intrel'])
sim_matrices = st.session_state.get('current_sim_matrices', initial_data['sim_matrices'])
intent_impact = st.session_state.get('current_intent_impact', initial_data['intent_impact'])
intel_ww = st.session_state.get('current_intel_ww', initial_data['intel_ww'])
player_images = dict(zip(players_df['fullname'], players_df['image_path']))

# Full width header
mode_title = MODES.get(current_mode, 'Optimal Field Setting')
st.markdown(f"""
<div class="main-header">
    <h1 class="main-title">{mode_title} - Optimal Field Setting</h1>
    <div class="author-info">
        <span class="author-name">Arnav Jain | IITK</span>
        <a href="https://x.com/arnav1204aj" target="_blank" class="author-link">@arnav1204aj</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Mode switcher - allow users to change mode
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("MEN'S T20", use_container_width=True, key='switch_mens_t20'):
        st.session_state['current_mode'] = 'MENS_T20'
        st.rerun()

with col2:
    if st.button("WOMEN'S T20", use_container_width=True, key='switch_womens_t20'):
        st.session_state['current_mode'] = 'WOMENS_T20'
        st.rerun()

with col3:
    if st.button("MEN'S ODI", use_container_width=True, key='switch_mens_odi'):
        st.session_state['current_mode'] = 'MENS_ODI'
        st.rerun()

st.markdown("---")

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
            st.markdown('Suggestion: Avoid Short length for spinners, results might be weird due to sparsity.')
            # Batter selection
            batter_list = list(ev_dict.keys())
            st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">Select Batter</p>', unsafe_allow_html=True)
            
            # Set default batter based on mode
            if current_mode == 'WOMENS_T20':
                default_batter = "Smriti Mandhana"
            else:  # MENS_T20 and MENS_ODI both default to Virat Kohli
                default_batter = "Virat Kohli"
            
            try:
                default_index = batter_list.index(default_batter) if default_batter in batter_list else 0
            except ValueError:
                default_index = 0
            
            selected_batter = st.selectbox("Select Batter", batter_list, index=default_index, label_visibility="collapsed", key="batter")

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
                st.stop()

            # PLAYER IMAGE AND STATS ROW
            img_col, stats_col = st.columns([1, 2], vertical_alignment="center", gap="large")

            with img_col:
                # Determine if we should show images
                # Show images for Men's T20 and Men's ODI
                # Don't show images for Women's T20
                show_image = current_mode != 'WOMENS_T20'
                
                if not show_image:
                    # Women's T20 mode: display name only without box
                    name_parts = selected_batter.split()
                    if len(name_parts) > 1:
                        # If name has multiple parts, put first part(s) on first line
                        # and remaining on second line (surname typically last)
                        first_line = ' '.join(name_parts[:-1])
                        second_line = name_parts[-1]
                        display_name = f"{first_line}<br/>{second_line}"
                    else:
                        display_name = selected_batter
                    
                    st.markdown(
                        f"""
                        <div style="
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            height: 100%;
                            width: 100%;
                        ">
                            <p style="
                                margin: 0;
                                font-size: 2rem;
                                font-weight: 700;
                                text-align: center;
                                color: white;
                                line-height: 1.3;
                            ">
                                {display_name}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # Men's T20 and Men's ODI modes: display image and name
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
                            <div class="player-img-wrapper">
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
                        v = dict_360.get(selected_batter, {}).get(ln, {}).get(selected_bowl_kind, {}).get('running').get('360_score', 0)
                    except Exception:
                        v = 0
                    vals.append(v)
                batter_360 = sum(vals) / len(sel_lens)

                vals = []
                for ln in sel_lens:
                    try:
                        v = avg_360.get('A', {}).get(ln, {}).get(selected_bowl_kind, {}).get('running').get('360_score', 0)
                    except Exception:
                        v = 0
                    vals.append(v)
                global_360 = sum(vals) / len(sel_lens)

                with col1:
                    st.metric(
                        "BATTER 360 SCORE (RUNNING)",
                        f"{batter_360:.1f}",
                        delta=f"{batter_360 - global_360:.1f}"
                    )

                with col2:
                    st.metric(
                        "GLOBAL AVG (RUNNING 360)",
                        f"{global_360:.1f}"
                    )
                
                col1, col2 = st.columns(2)

                # Average 360 score across selected lengths (missing treated as 0)
                sel_lens = selected_lengths if isinstance(selected_lengths, list) else [selected_lengths]
                vals = []
                for ln in sel_lens:
                    try:
                        v = dict_360.get(selected_batter, {}).get(ln, {}).get(selected_bowl_kind, {}).get('boundary').get('360_score', 0)
                    except Exception:
                        v = 0
                    vals.append(v)
                batter_360 = sum(vals) / len(sel_lens)

                vals = []
                for ln in sel_lens:
                    try:
                        v = avg_360.get('A', {}).get(ln, {}).get(selected_bowl_kind, {}).get('boundary').get('360_score', 0)
                    except Exception:
                        v = 0
                    vals.append(v)
                global_360 = sum(vals) / len(sel_lens)

                with col1:
                    st.metric(
                        "BATTER 360 SCORE (BOUNDARY)",
                        f"{batter_360:.1f}",
                        delta=f"{batter_360 - global_360:.1f}"
                    )

                with col2:
                    st.metric(
                        "GLOBAL AVG (BOUNDARY 360)",
                        f"{global_360:.1f}"
                    )

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
                        v = avg_360.get('A', {}).get(ln, {}).get(selected_bowl_kind, {}).get('overall').get('360_score', 0)
                    except Exception:
                        v = 0
                    vals.append(v)
                global_360 = sum(vals) / len(sel_lens)

                with col1:
                    st.metric(
                        "BATTER 360 SCORE (OVERALL)",
                        f"{batter_360:.1f}",
                        delta=f"{batter_360 - global_360:.1f}"
                    )

                with col2:
                    st.metric(
                        "GLOBAL AVG (OVERALL 360)",
                        f"{global_360:.1f}"
                    )    
                # ─────────────────────────────────────────────
                # 🏃 ROW 2 : RUNNING PROTECTION (Lower = Better)
                # ─────────────────────────────────────────────
                

            st.markdown("---")

                        # ─────────────────────────────────────────────
            # 🔍 SIMILAR BATTERS
            # ─────────────────────────────────────────────
            
            # FIELD AND CONTRIBUTIONS
            col1, col2 = st.columns([1.6, 1.4])

            with col1:
                st.markdown('<p class="section-header">Optimal Field Placement</p>', unsafe_allow_html=True)
                
                try:
                    fig, inf_labels, out_labels = plot_field_setting(data)
                
                    st.pyplot(fig, use_container_width=True)
                except Exception:
                    st.warning('Unavailable')

                

            with col2:
                st.markdown('<p class="section-header">Protection Stats and Fielder Contributions</p>', unsafe_allow_html=True)
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
                        length_dict,
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
                            Darker colors indicate higher sector importance and thus a priority region for the fielding teams.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

                    # After the Sector Importance section, add:
            
            st.markdown("---")
            
            st.markdown('<p class="section-header">Intelligent Wagon Wheel</p>', unsafe_allow_html=True)
            col1, col2 = st.columns([1.6, 1.4])
            with col1:
                try:
                   
                    fig = plot_int_wagons(selected_batter,selected_lengths,selected_bowl_kind,10,intel_ww,theme='green')   
                    st.pyplot(fig)
                except Exception:
                    st.warning('Unavailable')
            with col2:
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(153, 27, 27, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid rgba(220,38,38,0.3);
                        height: 100%;
                    ">
                        <h3 style="color: #fca5a5; font-size: 1.2rem; font-weight: 700; margin-top: 0;">
                            Understanding the Intelligent Wagon Wheel
                        </h3>
                        <p style="color: rgba(255,255,255,0.85); line-height: 1.7; font-size: 0.95rem;">
                            This wagon wheel visualizes batter's true strength in different regions. Each line here is a shot played by the batter
                            and length is a multiplication of runs and shot difficulty given the delivery characteristics. Thus a region concentrated by 
                            longer lines is a region of good ability for the batter.
                        </p>
                    
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("---")            

            st.markdown('<p class="section-header">Similar Batters</p>', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            with col1:
                

                

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
            with col2:
                
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(153, 27, 27, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid rgba(220,38,38,0.3);
                        height: 100%;
                    ">
                        <h3 style="color: #fca5a5; font-size: 1.2rem; font-weight: 700; margin-top: 0;">
                            Understanding Batter Similarity
                        </h3>
                        <p style="color: rgba(255,255,255,0.85); line-height: 1.7; font-size: 0.95rem;">
                            Batter Similarity a vector based similarity score considering shots, 
                            zones, control%, boundary%, dot%, running%, average on different lines, lengths and bowler kinds.
                       
                    
                    </div>
                    """, unsafe_allow_html=True)
                

            st.markdown("---")
            st.markdown('<p class="section-header">Intent, Reliability, Int-Rel by length</p>', unsafe_allow_html=True)
            col1,col2,col3, col4 = st.columns([1, 1, 1, 1])
            with col4:
                try:
                        fig = plot_intrel_pitch_avg(intrel,selected_batter,selected_lengths,selected_bowl_kind,5)     
                        st.pyplot(fig)
                except Exception:
                                st.warning('Unavailable')
            
            with col3:
                
                try:
                        fig = plot_intrel_pitch('intrel_by_length','Int-Rel',intrel,selected_batter,selected_lengths,selected_bowl_kind,5)     
                        st.pyplot(fig)
                except Exception:
                                st.warning('Unavailable')
            with col2:
                
                try:
                        fig = plot_intrel_pitch('reliability_by_length','Reliability',intrel,selected_batter,selected_lengths,selected_bowl_kind,5)     
                        st.pyplot(fig)
                except Exception:
                                st.warning('Unavailable')
            with col1:
                
                try:
                        fig = plot_intrel_pitch('intent_by_length','Intent',intrel,selected_batter,selected_lengths,selected_bowl_kind,5)     
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
                            Understanding Intent, Reliability, Int-Rel
                        </h3>
                        <p style="color: rgba(255,255,255,0.85); line-height: 1.7; font-size: 0.95rem;">
                            Int-Rel is an intent-reliability measuring metric. It is a multiplication of SRs (Intent) and Control% (Reliability)
                            the batter achieves compared to other batters in the same innings. Keeping in mind the nature of T20s,
                            Intent is given a 2x weight during multiplication. For ODIs, both are given equal weight. So for all Intent, Reliability and Int-Rel, a value of 1.20 for example means
                            the batter was 20% better, 0.8 means 20% worse, 1 is average performance. For reference, numbers of an average batter playing 
                            in same conditions as the batter are provided. Values use a time decay factor. Last 2 years data is given 50% weight.
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
                            length_dict,
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
                            length_dict,
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
                                length_dict,
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
                                length_dict,
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
                            in a particular region or playing a particular shot, we compare his/her distributions to an 
                            average batter's distributions </strong>. Average batter's calculations are done on the same 
                            line-length distribution the batter has faced in his/her career. The calculations consider run scoring
                            difficulty of a region or shot for the given line-length-bathand-pace/spin combination. 
                        </p>
                                <p style="color: rgba(255,255,255,0.85); line-height: 1.7; font-size: 0.95rem;">
                            The drives include both lofted and grounded drives.
                        </p>
                    
                    </div>
                    """, unsafe_allow_html=True)
                  except Exception:
                    st.warning('Unavailable')

            st.markdown("---")

            st.markdown('<p class="section-header">Intent Impact Progression</p>', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                try:
                    fig = plot_intent_impact(
                        selected_batter,
                        intent_impact,
                        'all bowlers',
                        min_count=5
                    )
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.warning('Intent Impact Analysis Unavailable')
                
                
            with col2:
                try:
                    fig = plot_intent_impact(
                        selected_batter,
                        intent_impact,
                        selected_bowl_kind,
                        min_count=5
                    )
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.warning('Intent Impact Analysis Unavailable')
                
            st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(153, 27, 27, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid rgba(220,38,38,0.3);
                        height: 100%;
                    ">
                        <h3 style="color: #fca5a5; font-size: 1.2rem; font-weight: 700; margin-top: 0;">
                            Understanding Intent Impact Progression
                        </h3>
                        <p style="color: rgba(255,255,255,0.85); line-height: 1.7; font-size: 0.95rem;">
                            Intent Impact for a ball is the extra runs batter adds to the team's total due to their intent on that ball.
                            Cumulative intent impact thus shows total extra runs as the innings progresses. The slope of the plot is an 
                            indicator of how much aggressive the batter is at that stage of the innings. A steeper positive slope means more aggression.
                            Negative slope means batter is affecting the team's total negatively due to low intent. Controlled runs is a product of control and runs.
                            Values are weighed to give roughly 50% weight to recent 2 years of data.
                        </p>
                               
                    
                    </div>
                    """, unsafe_allow_html=True)       # Add spacer for vertical centering
                
        
            

        
    
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
         <h2 class="info-title">Data Timeline</h2>   
        <p style="color: rgba(255,255,255,0.85); line-height: 1.8; font-size: 1rem;"> 
         Mens ODIs - since 2014, Mens T20s - since 2015, Womens T20s - since 2020                
    </div>
    """, unsafe_allow_html=True)







