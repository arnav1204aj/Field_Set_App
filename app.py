import streamlit as st
st.set_page_config(layout="wide", page_title="Optimal Field Setting | Cricket Analytics")

import pandas as pd
import numpy as np
from functions import plot_int_wagons, plot_intent_impact, plot_field_setting, plot_intrel_pitch, plot_intrel_pitch_avg, plot_sector_ev_heatmap, create_shot_profile_chart, create_similarity_chart, create_zone_strength_table, get_top_similar_batters

import requests
import time
from typing import Dict, Any, List, Optional


# # ─────────────────────────────
API_KEY = st.secrets["API_KEY"]
BACKEND_URL = st.secrets["BACKEND_URL"]


API_HEADERS = {"X-API-Key": API_KEY}
REQUEST_TIMEOUT = 60
MAX_RETRIES = 4
RETRY_STATUS_CODES = {502, 503, 504}

def make_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Optional[Dict]:
    """Helper function to make requests to FastAPI backend with error handling"""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        for attempt in range(MAX_RETRIES):
            try:
                if method == "GET":
                    response = requests.get(url, headers=API_HEADERS, timeout=REQUEST_TIMEOUT)
                else:
                    response = requests.post(url, json=data, headers=API_HEADERS, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else None
                should_retry = status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES - 1
                if should_retry:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend at {BACKEND_URL}")
        st.info("Start the backend with: python backend.py")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("Backend request timed out. Please try again.")
        st.stop()
    except requests.exceptions.HTTPError as e:
        error_detail = "Unknown error"
        try:
            error_detail = e.response.json().get('detail', str(e))
        except:
            error_detail = str(e)
        st.error(f"Backend error: {error_detail}")
        st.stop()
    except Exception as e:
        st.error(f"Error communicating with backend: {str(e)}")
        st.stop()

# ─────────────────────────────
# Backend Data Fetching Functions
# ─────────────────────────────

MODES = {
    'MENS_T20': 'Men\'s T20',
    'WOMENS_T20': 'Women\'s T20',
    'MENS_ODI': 'Men\'s ODI'
}

ANALYSIS_SECTIONS = [
    "Field Overview",
    "Sector Importance Analysis",
    "Intelligent Wagon Wheel",
    "Similar Batters",
    "Intent, Reliability, Int-Rel by length",
    "Relative Zone Strengths",
    "Relative Shot Strengths",
    "Intent Impact Progression",
]

COMPARE_SECTIONS = [
    "360 Ability",
    "Zone Strengths",
    "Shot Strengths",
    "Lengthwise Intent",
    "Lengthwise Reliability",
    "Lengthwise Int-Rel",
]

DEFAULT_COMPARE_BATTERS = {
    "MENS_T20": ("Ab de Villiers", "Suryakumar Yadav"),
    "MENS_ODI": ("Virat Kohli", "Joe Root"),
    "WOMENS_T20": ("Smriti Mandhana", "Meg Lanning"),
}

@st.cache_data(ttl=900, max_entries=50)
def fetch_batters(mode: str) -> List[str]:
    """Fetch list of batters from backend"""
    response = make_request(f"/batters/{mode}")
    return response['batters'] if response else []

@st.cache_data(ttl=900, max_entries=50)
def fetch_players(mode: str) -> List[Dict]:
    """Fetch players data from backend"""
    response = make_request(f"/players/{mode}")
    return response['players'] if response else []

@st.cache_data(ttl=900, max_entries=50)
def fetch_bowl_kinds(mode: str, batter: str) -> List[str]:
    """Fetch available bowling kinds from backend"""
    response = make_request(f"/bowl-kinds/{mode}/{batter}")
    return response['bowl_kinds'] if response else []

@st.cache_data(ttl=900, max_entries=50)
def fetch_lengths(mode: str, batter: str, bowl_kind: str) -> List[str]:
    """Fetch available lengths from backend"""
    response = make_request(f"/lengths/{mode}/{batter}/{bowl_kind}")
    return response['lengths'] if response else []

@st.cache_data(ttl=900, max_entries=50)
def fetch_outfielders(mode: str, batter: str, bowl_kind: str, lengths: List[str]) -> List[str]:
    """Fetch available outfielder configurations from backend"""
    response = make_request(
        f"/outfielders/{mode}/{batter}/{bowl_kind}",
        method="POST",
        data={"lengths": lengths}
    )
    return response['outfielders'] if response else []

@st.cache_data(ttl=600, max_entries=50)
def fetch_field_setup(mode: str, batter: str, bowl_kind: str, lengths: List[str], outfielders: str) -> Optional[Dict]:
    """Fetch field setup from backend"""
    response = make_request(
        "/field-setup",
        method="POST",
        data={
            "mode": mode,
            "batter": batter,
            "bowl_kind": bowl_kind,
            "lengths": lengths,
            "outfielders": outfielders
        }
    )
    return response['field_setup'] if response else None

@st.cache_data(ttl=600, max_entries=50)
def fetch_protection_stats(mode: str, batter: str, bowl_kind: str, lengths: List[str], outfielders: str) -> Optional[Dict]:
    """Fetch protection stats from backend"""
    response = make_request(
        "/protection-stats",
        method="POST",
        data={
            "mode": mode,
            "batter": batter,
            "bowl_kind": bowl_kind,
            "lengths": lengths,
            "outfielders": outfielders
        }
    )
    return response if response else None

@st.cache_data(ttl=600, max_entries=50)
def fetch_ev_heatmap_data(mode: str, batter: str, bowl_kind: str, lengths: List[str]) -> Optional[Dict]:
    """Fetch EV heatmap data from backend"""
    response = make_request(
        "/ev-heatmap-data",
        method="POST",
        data={
            "mode": mode,
            "batter": batter,
            "bowl_kind": bowl_kind,
            "lengths": lengths,
            "outfielders": ""
        }
    )
    return response if response else None

@st.cache_data(ttl=600, max_entries=50)
def fetch_zone_strength(mode: str, batter: str, bowl_kind: str, lengths: List[str]) -> Optional[Dict]:
    """Fetch zone strength data from backend"""
    response = make_request(
        "/zone-strength",
        method="POST",
        data={
            "mode": mode,
            "batter": batter,
            "bowl_kind": bowl_kind,
            "lengths": lengths,
            "outfielders": ""
        }
    )
    return response if response else None

@st.cache_data(ttl=600, max_entries=50)
def fetch_similar_batters(mode: str, batter: str, bowl_kind: str, lengths: List[str]) -> Optional[Dict]:
    """Fetch similar batters data from backend"""
    response = make_request(
        "/similar-batters",
        method="POST",
        data={
            "mode": mode,
            "batter": batter,
            "bowl_kind": bowl_kind,
            "lengths": lengths,
            "top_n": 5
        }
    )
    return response if response else None

@st.cache_data(ttl=600, max_entries=50)
def fetch_intrel_data(mode: str, batter: str, bowl_kind: str, lengths: List[str]) -> Optional[Dict]:
    """Fetch intent-reliability data from backend"""
    response = make_request(
        "/intrel-data",
        method="POST",
        data={
            "mode": mode,
            "batter": batter,
            "bowl_kind": bowl_kind,
            "lengths": lengths,
            "outfielders": ""
        }
    )
    return response if response else None

@st.cache_data(ttl=600, max_entries=50)
def fetch_intent_impact_data(mode: str, batter: str, bowl_kind: str) -> Optional[Dict]:
    """Fetch intent impact data from backend"""
    response = make_request(
        "/intent-impact-data",
        method="POST",
        data={
            "mode": mode,
            "batter": batter,
            "bowl_kind": bowl_kind,
            "lengths": [],
            "outfielders": ""
        }
    )
    return response if response else None

@st.cache_data(ttl=600, max_entries=50)
def fetch_intelligent_wagon_wheel(mode: str, batter: str, bowl_kind: str, lengths: List[str]) -> Optional[Dict]:
    """Fetch intelligent wagon wheel data from backend"""
    response = make_request(
        "/intelligent-wagon-wheel",
        method="POST",
        data={
            "mode": mode,
            "batter": batter,
            "bowl_kind": bowl_kind,
            "lengths": lengths,
            "outfielders": ""
        }
    )
    return response if response else None

@st.cache_data(ttl=600, max_entries=50)
def fetch_shot_profile(mode: str, batter: str, bowl_kind: str, lengths: List[str]) -> Optional[Dict]:
    """Fetch shot profile data from backend"""
    response = make_request(
        "/shot-profile",
        method="POST",
        data={
            "mode": mode,
            "batter": batter,
            "bowl_kind": bowl_kind,
            "lengths": lengths,
            "outfielders": ""
        }
    )
    return response if response else None


def _avg(values: List[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float, np.floating))]
    return sum(vals) / len(vals) if vals else 0.0


def _fmt(v: float, suffix: str = "", precision: int = 1) -> str:
    return f"{float(v):.{precision}f}{suffix}"


def _cmp_class(v1: float, v2: float, higher_is_better: bool) -> tuple[str, str]:
    if abs(v1 - v2) < 1e-9:
        return "#d1d5db", "#d1d5db"
    if higher_is_better:
        return ("#22c55e", "#ef4444") if v1 > v2 else ("#ef4444", "#22c55e")
    return ("#22c55e", "#ef4444") if v1 < v2 else ("#ef4444", "#22c55e")


def _render_compare_rows(
    title: str,
    left_label: str,
    right_label: str,
    rows: List[Dict[str, Any]],
    key_prefix: str,
) -> None:
    st.markdown(
        """
        <style>
            .cmp-wrap { margin-top: 1.1rem; }
            .cmp-title {
                margin-bottom: 0.8rem;
                padding: 0.85rem 1rem;
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.22);
                background: linear-gradient(90deg, rgba(59,130,246,0.30) 0%, rgba(245,158,11,0.30) 100%);
                font-size: 1.05rem;
                font-weight: 800;
                letter-spacing: 0.5px;
                color: #e2e8f0;
                text-transform: uppercase;
            }
            .cmp-head-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
                margin-bottom: 0.75rem;
            }
            .cmp-head {
                color: white;
                font-size: 1.08rem;
                font-weight: 700;
                padding: 0.45rem 0.65rem;
                border-radius: 8px;
            }
            .cmp-head-1 {
                border-left: 3px solid #60a5fa;
                background: rgba(96,165,250,0.14);
            }
            .cmp-head-2 {
                border-left: 3px solid #f59e0b;
                background: rgba(245,158,11,0.14);
            }
            .cmp-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
                margin-bottom: 0.35rem;
            }
            .cmp-cell {
                padding: 0.7rem 0.85rem;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 10px;
                background: rgba(255,255,255,0.03);
            }
            .cmp-cell-1 {
                border-left: 4px solid rgba(96,165,250,0.95);
                background: linear-gradient(135deg, rgba(96,165,250,0.18) 0%, rgba(255,255,255,0.03) 70%);
            }
            .cmp-cell-2 {
                border-left: 4px solid rgba(245,158,11,0.95);
                background: linear-gradient(135deg, rgba(245,158,11,0.18) 0%, rgba(255,255,255,0.03) 70%);
            }
            .cmp-label {
                font-size: 0.82rem;
                color: #fca5a5;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 0.2rem;
            }
            .cmp-val {
                font-size: 1.22rem;
                font-weight: 700;
                margin-top: 0.2rem;
            }
            .cmp-info details { display: inline-block; margin-left: 10px; vertical-align: middle; position: relative; }
            .cmp-info summary {
                list-style: none; cursor: pointer; color: #67e8f9;
                font-size: 0.86rem; font-weight: 800; display: inline-block; user-select: none;
            }
            .cmp-info-pop {
                position: absolute; top: 22px; left: 0; z-index: 9999; width: 260px;
                padding: 6px 8px; border-radius: 8px; border: 1px solid rgba(103,232,249,0.45);
                background: rgba(2,6,23,0.92); color: #e2e8f0; font-size: 0.78rem; line-height: 1.35;
                text-transform: none; letter-spacing: 0; font-weight: 500; box-shadow: 0 10px 24px rgba(0,0,0,0.35);
            }
            @media (max-width: 768px) {
                .cmp-head-grid { grid-template-columns: 1fr; gap: 0.45rem; }
                .cmp-row { grid-template-columns: 1fr; gap: 0.45rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="cmp-wrap"><div class="cmp-title">{title}</div></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="cmp-head-grid">
            <div class="cmp-head cmp-head-1">{left_label}</div>
            <div class="cmp-head cmp-head-2">{right_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for r in rows:
        color1, color2 = _cmp_class(float(r["v1"]), float(r["v2"]), bool(r.get("higher_is_better", True)))
        help_text = (r.get("help") or "").replace('"', "&quot;")
        info_html = (
            f"""
            <span class="cmp-info">
                <details>
                    <summary title="{help_text}">?</summary>
                    <div class="cmp-info-pop">{help_text}</div>
                </details>
            </span>
            """
            if help_text else ""
        )
        st.markdown(
            f"""
            <div class="cmp-row">
                <div class="cmp-cell cmp-cell-1">
                    <div class="cmp-label">{r['label']}{info_html}</div>
                    <div class="cmp-val" style="color:{color1};">{r['s1']}</div>
                </div>
                <div class="cmp-cell cmp-cell-2">
                    <div class="cmp-label">{r['label']}{info_html}</div>
                    <div class="cmp-val" style="color:{color2};">{r['s2']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

def _aggregate_zone_perc(zone_data: Dict[str, Any], lengths: List[str]) -> Dict[str, Dict[str, float]]:
    dict_360 = zone_data.get("dict_360_selected", {}) if zone_data else {}
    weights = zone_data.get("length_weights", {}) if zone_data else {}
    out: Dict[str, Dict[str, float]] = {}
    for run_class in ["overall", "running", "boundary"]:
        total_runs = 0.0
        zone_vals = {"st": 0.0, "leg": 0.0, "off": 0.0, "bk": 0.0}
        for ln in lengths:
            w = float(weights.get(ln, 0) or 0)
            rc = dict_360.get(ln, {}).get(run_class, {})
            t = float(rc.get("total_runs", 0) or 0)
            total_runs += w * t
            for z in zone_vals:
                zone_vals[z] += w * float(rc.get(f"{z}_runs", 0) or 0)
        denom = total_runs if total_runs > 0 else 1.0
        out[run_class] = {
            "Straight": 100.0 * zone_vals["st"] / denom,
            "Leg": 100.0 * zone_vals["leg"] / denom,
            "Off": 100.0 * zone_vals["off"] / denom,
            "Behind": 100.0 * zone_vals["bk"] / denom,
        }
    return out


def _aggregate_shot_perc(shot_data: Dict[str, Any], lengths: List[str]) -> Dict[str, float]:
    shot_selected = shot_data.get("shot_profile_selected", {}) if shot_data else {}
    weights = shot_data.get("length_weights", {}) if shot_data else {}
    agg: Dict[str, float] = {}
    for ln in lengths:
        w = float(weights.get(ln, 0) or 0)
        ln_block = shot_selected.get(ln, {}) if isinstance(shot_selected, dict) else {}
        for shot, vals in (ln_block or {}).items():
            if isinstance(vals, dict):
                agg[shot] = agg.get(shot, 0.0) + w * float(vals.get("runs", 0) or 0)
    total = sum(agg.values())
    if total <= 0:
        return {k: 0.0 for k in agg}
    return {k: (v / total) * 100.0 for k, v in agg.items()}


def _extract_intrel_by_length(intrel_data: Dict[str, Any], metric_key: str, lengths: List[str]) -> Dict[str, float]:
    payload = intrel_data.get("intrel_selected", {}) if intrel_data else {}
    metric_map = payload.get(metric_key, {}) if isinstance(payload, dict) else {}
    out = {}
    for ln in lengths:
        v = metric_map.get(ln, 0) if isinstance(metric_map, dict) else 0
        if isinstance(v, (list, tuple)) and len(v) >= 1 and isinstance(v[0], (int, float, np.floating)):
            out[ln] = float(v[0])
        elif isinstance(v, (int, float, np.floating)):
            out[ln] = float(v)
        else:
            out[ln] = 0.0
    return out


def _calc_p95_radius(ww_data: Dict[str, Any], lengths: List[str]) -> float:
    selected = ww_data.get("intel_ww_selected", {}) if ww_data else {}
    norms: List[float] = []
    for ln in lengths:
        evs = (selected.get(ln, {}) or {}).get("evs", [])
        if evs:
            arr = np.asarray(evs, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                norms.extend(np.linalg.norm(arr[:, :2], axis=1).tolist())
    return float(np.percentile(norms, 95)) if norms else 0.0


# Initialize session state
if 'current_mode' not in st.session_state:
    st.session_state['current_mode'] = None

# ─────────────────────────────
# Custom CSS
# ─────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    .main {
        background: linear-gradient(135deg, #1a0a0a 0%, #2d1414 100%);
        font-family: 'Inter', sans-serif;
    }
    
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
    
    .player-name {
        font-size: clamp(1.8rem, 4vw, 2.5rem);
        font-weight: 800;
        color: white;
        margin-bottom: clamp(1rem, 3vw, 2rem);
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
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
    
    .section-header {
        font-size: clamp(1.2rem, 2.5vw, 1.3rem);
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(220, 38, 38, 0.5);
    }
    
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
    
    .stSelectbox > div > div {
        background: #000000 !important;
        border: 1px solid rgba(100,100,100,0.3);
        color: white;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background: #000000 !important;
    }
    
    .context-info {
        font-size: clamp(1rem, 2vw, 1.1rem) !important;
    }

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
        
        [data-testid="column"] {
            min-width: 100% !important;
        }
    }
    
    @media (min-width: 769px) and (max-width: 1024px) {
        .main-title {
            font-size: 2.2rem;
        }
        
        .player-name {
            font-size: 2rem;
        }
    }
    
    @media (min-width: 1920px) {
        .main {
            max-width: 1920px;
            margin: 0 auto;
        }
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# Mode Selection Interface
# ─────────────────────────────

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
    
    st.stop()

# ─────────────────────────────
# Initialize Mode-Specific Data
# ─────────────────────────────

current_mode = st.session_state['current_mode']

# Header
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

# Mode switcher
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
VIEW_OPTIONS = ["Analysis", "Compare", "Information"]
if "active_view" not in st.session_state or st.session_state["active_view"] not in VIEW_OPTIONS:
    st.session_state["active_view"] = "Analysis"

active_view = st.radio(
    "View",
    VIEW_OPTIONS,
    index=VIEW_OPTIONS.index(st.session_state["active_view"]),
    horizontal=True,
    key="active_view_selector",
    label_visibility="collapsed",
)
st.session_state["active_view"] = active_view

if active_view == "Analysis":
    zone_data_cached = None
    shot_data_cached = None
    intrel_data_cached = None
    impact_data_cached = None

    with st.sidebar:
        st.markdown('<p class="section-header">Parameters</p>', unsafe_allow_html=True)
        with st.form(key='field_form'):
            submit = st.form_submit_button("Generate Results", use_container_width=True)
            st.markdown('Suggestion: Avoid Short length for spinners, results might be weird due to sparsity.')

            batter_list = fetch_batters(current_mode)
            if not batter_list:
                st.error('No batters available for selected mode.')
                st.stop()

            default_batter = "Smriti Mandhana" if current_mode == 'WOMENS_T20' else "Virat Kohli"
            default_index = batter_list.index(default_batter) if default_batter in batter_list else 0

            st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">Select Batter</p>', unsafe_allow_html=True)
            selected_batter = st.selectbox("Select Batter", batter_list, index=default_index, label_visibility="collapsed", key="batter")

            bowl_kind_list = fetch_bowl_kinds(current_mode, selected_batter)
            st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">Select Bowling Type</p>', unsafe_allow_html=True)
            selected_bowl_kind = st.selectbox("Select Bowling Type", bowl_kind_list, label_visibility="collapsed", key="bowl")

            length_list = fetch_lengths(current_mode, selected_batter, selected_bowl_kind)
            st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">Select Length(s)</p>', unsafe_allow_html=True)
            length_options = ['FULL', 'SHORT', 'GOOD_LENGTH', 'SHORT_OF_A_GOOD_LENGTH']
            available_lengths = [l for l in length_options if l in length_list] or length_list
            selected_lengths = st.multiselect("Select Length(s)", available_lengths, default=[available_lengths[0]] if available_lengths else [], label_visibility="collapsed", key="length")
            if not selected_lengths and available_lengths:
                selected_lengths = [available_lengths[0]]

            outfielder_list = fetch_outfielders(current_mode, selected_batter, selected_bowl_kind, selected_lengths)
            st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">Select Outfielders</p>', unsafe_allow_html=True)
            selected_outfielders = st.selectbox("Select Outfielders", outfielder_list, label_visibility="collapsed", key="out") if outfielder_list else ""    

            st.markdown('<p style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1rem;">I want to see</p>', unsafe_allow_html=True)
            selected_sections = st.multiselect(
                "I want to see",
                ANALYSIS_SECTIONS,
                default=["Field Overview"],
                label_visibility="collapsed",
                key="analysis_sections"
            )

            

    if submit and "Field Overview" in selected_sections:
        st.markdown('<p class="section-header">Field Overview</p>', unsafe_allow_html=True)
        if not selected_outfielders:
            st.warning('No outfielder option available for this filter.')
        else:
            field_setup = fetch_field_setup(current_mode, selected_batter, selected_bowl_kind, selected_lengths, selected_outfielders)
            if not field_setup:
                st.warning("No field setting found for this combination.")
            else:
                data = field_setup
                img_col, stats_col = st.columns([1, 2], vertical_alignment="center", gap="large")
                with img_col:
                    show_image = current_mode != 'WOMENS_T20'

                    if not show_image:
                        name_parts = selected_batter.split()
                        if len(name_parts) > 1:
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
                            unsafe_allow_html=True,
                        )
                    else:
                        players_list = fetch_players(current_mode)
                        player_images = {player['fullname']: player.get('image_path', '') for player in players_list} if players_list else {}
                        player_img_url = player_images.get(selected_batter, "https://via.placeholder.com/300x300.png?text=No+Image")

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
                                        " />
                                    <p style="
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
                            unsafe_allow_html=True,
                        )
                with stats_col:
                    st.markdown(f'<p class="context-info" style="color: rgba(255,255,255,0.7); font-size:1.1rem; font-weight:500;">{selected_bowl_kind} | {", ".join(selected_lengths)} | {selected_outfielders} outfielders</p>', unsafe_allow_html=True)

                    if zone_data_cached is None:
                        zone_data_cached = fetch_zone_strength(current_mode, selected_batter, selected_bowl_kind, selected_lengths)
                    zone_metrics = zone_data_cached
                    dict_360 = zone_metrics.get('dict_360_selected', {}) if zone_metrics else {}
                    avg_360 = zone_metrics.get('avg_360_selected', {}) if zone_metrics else {}
                    sel_lens = selected_lengths if isinstance(selected_lengths, list) else [selected_lengths]

                    def avg_score(scope: str, run_class: str) -> float:
                        vals = []
                        for ln in sel_lens:
                            try:
                                if scope == 'batter':
                                    v = dict_360.get(ln, {}).get(run_class, {}).get('360_score', 0)
                                else:
                                    v = avg_360.get(ln, {}).get(run_class, {}).get('360_score', 0)
                            except Exception:
                                v = 0
                            vals.append(v)
                        return sum(vals) / len(sel_lens) if sel_lens else 0

                    for run_class, label in [("running", "RUNNING"), ("boundary", "BOUNDARY"), ("overall", "OVERALL")]:
                        batter_score = avg_score('batter', run_class)
                        global_score = avg_score('global', run_class)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric(f"BATTER 360 SCORE ({label})", f"{batter_score:.1f}", delta=f"{batter_score - global_score:.1f}")
                        with c2:
                            st.metric(f"GLOBAL AVG ({label} 360)", f"{global_score:.1f}")

                st.markdown('---')
                c1, c2 = st.columns([1.6, 1.4])
                with c1:
                    st.markdown('<p class="section-header">Optimal Field Placement</p>', unsafe_allow_html=True)
                    try:
                        fig, inf_labels, out_labels = plot_field_setting(data)
                        st.pyplot(fig, use_container_width=True)
                    except Exception:
                        st.warning('Unavailable')

                with c2:
                    st.markdown('<p class="section-header">Protection Stats and Fielder Contributions</p>', unsafe_allow_html=True)
                    prot_stats = fetch_protection_stats(current_mode, selected_batter, selected_bowl_kind, selected_lengths, selected_outfielders)
                    batter_running = float((data.get('protection_stats', {}) or {}).get('running', prot_stats.get('batter_running', 0)) or 0)
                    batter_boundary = float((data.get('protection_stats', {}) or {}).get('boundary', prot_stats.get('batter_boundary', 0)) or 0)

                    # Match app2 logic: try exact average-batter composite length key first, then fallback.
                    global_running = float(prot_stats.get('global_running', 0) or 0)
                    global_boundary = float(prot_stats.get('global_boundary', 0) or 0)
                    try:
                        avg_field_setup = fetch_field_setup(
                            current_mode,
                            "average batter",
                            selected_bowl_kind,
                            selected_lengths,
                            selected_outfielders
                        )
                        if avg_field_setup:
                            avg_stats = avg_field_setup.get("protection_stats", {}) or {}
                            global_running = float(avg_stats.get("running", global_running) or global_running)
                            global_boundary = float(avg_stats.get("boundary", global_boundary) or global_boundary)
                    except Exception:
                        pass

                    a1, a2 = st.columns(2)
                    with a1:
                        st.metric("RUNNING PROTECTION", f"{batter_running:.1f}%", delta=f"{global_running - batter_running:.1f}%")
                    with a2:
                        st.metric("GLOBAL AVG (RUN. PROT.)", f"{global_running:.1f}%")

                    b1, b2 = st.columns(2)
                    with b1:
                        st.metric("BOUNDARY PROTECTION", f"{batter_boundary:.1f}%", delta=f"{global_boundary - batter_boundary:.1f}%")
                    with b2:
                        st.metric("GLOBAL AVG (BD. PROT.)", f"{global_boundary:.1f}%")

                    inf_contrib = data.get('infielder_ev_run_percent', [])
                    out_contrib = data.get('outfielder_ev_bd_percent', [])
                    inf_col, out_col = st.columns(2)

                    with inf_col:
                        st.markdown('<p class="contribution-title">Infielders</p>', unsafe_allow_html=True)
                        if inf_contrib:
                            for f in inf_contrib:
                                angle = f.get("angle")
                                label = inf_labels.get(angle, f"Angle {angle}°") if 'inf_labels' in locals() else f"Angle {angle}°"
                                st.markdown(
                                    f'<div class="contribution-item">{label} → {f.get("ev_run_percent", 0):.1f}% runs saved</div>',
                                    unsafe_allow_html=True
                                )
                        else:
                            st.write("No data available")

                    with out_col:
                        st.markdown('<p class="contribution-title">Outfielders</p>', unsafe_allow_html=True)
                        if out_contrib:
                            for f in out_contrib:
                                angle = f.get("angle")
                                label = out_labels.get(angle, f"Angle {angle}°") if 'out_labels' in locals() else f"Angle {angle}°"
                                st.markdown(
                                    f'<div class="contribution-item">{label} → {f.get("ev_bd_percent", 0):.1f}% runs saved</div>',
                                    unsafe_allow_html=True
                                )
                        else:
                            st.write("No data available")

    if submit and "Sector Importance Analysis" in selected_sections:
        st.markdown('---')
        st.markdown('<p class="section-header">Sector Importance Analysis</p>', unsafe_allow_html=True)
        plot_col, info_col = st.columns([1.6, 1.4])
        with plot_col:
            try:
                ev_data = fetch_ev_heatmap_data(current_mode, selected_batter, selected_bowl_kind, selected_lengths)
                ev_selected_payload = ev_data.get('ev_selected', {}) if ev_data else {}
                ev_selected = {ln: pd.DataFrame(v.get("data", []), columns=v.get("columns", [])) for ln, v in ev_selected_payload.items()}
                ev_fig = plot_sector_ev_heatmap(
                    ev_selected,
                    selected_batter,
                    selected_lengths,
                    selected_bowl_kind,
                    ev_data.get('length_weights', {}) if ev_data else {},
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
                    Answers where does the batter score more. This polar heatmap shows the 
                    <strong>Importance (SR × Probability in that sector)</strong> of different sectors of the field.
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

    if submit and "Intelligent Wagon Wheel" in selected_sections:
        st.markdown('---')
        st.markdown('<p class="section-header">Intelligent Wagon Wheel</p>', unsafe_allow_html=True)
        col1, col2 = st.columns([1.6, 1.4])
        with col1:
            try:
                ww_data = fetch_intelligent_wagon_wheel(current_mode, selected_batter, selected_bowl_kind, selected_lengths)
                fig = plot_int_wagons(
                    selected_batter,
                    selected_lengths,
                    selected_bowl_kind,
                    95,
                    ww_data.get('intel_ww_selected', {}),
                    theme='green'
                )
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
                        Answers where does the batter play more difficult shots. This wagon wheel visualizes batter's true strength in different regions. Each line here is a shot played by the batter
                        and length is a multiplication of runs and shot difficulty given the delivery characteristics. Thus a region concentrated by 
                        longer lines is a region of good ability (relative to an average batter) for the batter. The p95 radius is 95 percentile of all shot lengths. A higher value means 
                        batter plays more difficult shots. This value is choosen as the boundary of the plot to visualise where more of these shots are played.
                    </p>
                </div>
                """, unsafe_allow_html=True)

    if submit and "Similar Batters" in selected_sections:
        st.markdown('---')
        st.markdown('<p class="section-header">Similar Batters</p>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            try:
                sim_data = fetch_similar_batters(current_mode, selected_batter, selected_bowl_kind, selected_lengths)
                if sim_data and sim_data.get('similarity_data'):
                    sim_df = pd.DataFrame(sim_data['similarity_data'])
                    fig = create_similarity_chart(sim_df, selected_batter, selected_lengths, selected_bowl_kind)
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
                    </p>
                </div>
                """, unsafe_allow_html=True)

    if submit and "Intent, Reliability, Int-Rel by length" in selected_sections:
        st.markdown('---')
        st.markdown('<p class="section-header">Intent, Reliability, Int-Rel by length</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        try:
            if intrel_data_cached is None:
                intrel_data_cached = fetch_intrel_data(current_mode, selected_batter, selected_bowl_kind, selected_lengths)
            intrel_data = intrel_data_cached
            intrel_payload = intrel_data.get('intrel_selected', {}) if intrel_data else {}
            with c4:
                st.pyplot(plot_intrel_pitch_avg(intrel_payload, selected_batter, selected_lengths, selected_bowl_kind, 5))
            with c3:
                st.pyplot(plot_intrel_pitch('intrel_by_length', 'Int-Rel', intrel_payload, selected_batter, selected_lengths, selected_bowl_kind, 5))
            with c2:
                st.pyplot(plot_intrel_pitch('reliability_by_length', 'Reliability', intrel_payload, selected_batter, selected_lengths, selected_bowl_kind, 5))
            with c1:
                st.pyplot(plot_intrel_pitch('intent_by_length', 'Intent', intrel_payload, selected_batter, selected_lengths, selected_bowl_kind, 5))
        except Exception:
            st.warning('Intent-Reliability data unavailable')
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

    if submit and "Relative Zone Strengths" in selected_sections:
        st.markdown('---')
        st.markdown('<p class="section-header">Relative Zone Strengths</p>', unsafe_allow_html=True)
        reg_col, avg_col = st.columns([1.5, 1.5], gap='small')
        try:
            if zone_data_cached is None:
                zone_data_cached = fetch_zone_strength(current_mode, selected_batter, selected_bowl_kind, selected_lengths)
            zone_data = zone_data_cached
            with reg_col:
                st.markdown(f'<p class="subsection-header">Batter\'s Run Distribution</p>', unsafe_allow_html=True)
                zone_fig, _ = create_zone_strength_table(
                    zone_data.get('dict_360_selected', {}),
                    selected_batter,
                    selected_lengths,
                    selected_bowl_kind,
                    zone_data.get('length_weights', {}),
                    'runs'
                )
                if zone_fig:
                    st.pyplot(zone_fig, use_container_width=True)
            with avg_col:
                st.markdown(f'<p class="subsection-header">Avg Batter\'s Run Distribution</p>', unsafe_allow_html=True)
                zone_fig, _ = create_zone_strength_table(
                    zone_data.get('dict_360_selected', {}),
                    selected_batter,
                    selected_lengths,
                    selected_bowl_kind,
                    zone_data.get('length_weights', {}),
                    'avg_runs'
                )
                if zone_fig:
                    st.pyplot(zone_fig, use_container_width=True)
        except Exception:
            st.warning('Unavailable')

    if submit and "Relative Shot Strengths" in selected_sections:
        st.markdown('---')
        st.markdown('<p class="section-header">Relative Shot Strengths</p>', unsafe_allow_html=True)
        reg_col, avg_col = st.columns([1.5, 1.5], gap='small')
        try:
            if shot_data_cached is None:
                shot_data_cached = fetch_shot_profile(current_mode, selected_batter, selected_bowl_kind, selected_lengths)
            shot_data = shot_data_cached
            with reg_col:
                shot_fig = create_shot_profile_chart(
                    shot_data.get('shot_profile_selected', {}),
                    selected_batter,
                    selected_lengths,
                    selected_bowl_kind,
                    shot_data.get('length_weights', {}),
                    value_type='runs'
                )
                if shot_fig:
                    st.pyplot(shot_fig, use_container_width=True)
            with avg_col:
                shot_fig = create_shot_profile_chart(
                    shot_data.get('shot_profile_selected', {}),
                    selected_batter,
                    selected_lengths,
                    selected_bowl_kind,
                    shot_data.get('length_weights', {}),
                    value_type='avg_runs'
                )
                if shot_fig:
                    st.pyplot(shot_fig, use_container_width=True)
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

    if submit and "Intent Impact Progression" in selected_sections:
        st.markdown('---')
        st.markdown('<p class="section-header">Intent Impact Progression</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        try:
            if impact_data_cached is None:
                impact_data_cached = fetch_intent_impact_data(current_mode, selected_batter, selected_bowl_kind)
            impact_data = impact_data_cached
            impact_payload = impact_data.get('intent_impact_selected', {}) if impact_data else {}
            with c1:
                st.pyplot(plot_intent_impact(selected_batter, impact_payload, 'all bowlers', min_count=5), use_container_width=True)
            with c2:
                st.pyplot(plot_intent_impact(selected_batter, impact_payload, selected_bowl_kind, min_count=5), use_container_width=True)
        except Exception:
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
                """, unsafe_allow_html=True)

    if not submit:
        st.info("Please select parameters and click **Generate Results**")
# Compare Tab
if active_view == "Compare":
    with st.sidebar:
        st.markdown('<p class="section-header">Compare Filters</p>', unsafe_allow_html=True)
        with st.form(key="compare_form"):
            compare_submit = st.form_submit_button("Compare", use_container_width=True)

            compare_batters = fetch_batters(current_mode)
            if not compare_batters:
                st.error("No batters available for selected mode.")
                st.stop()
            if len(compare_batters) < 2:
                st.error("Need at least two batters to compare.")
                st.stop()

            d1, d2 = DEFAULT_COMPARE_BATTERS.get(current_mode, (compare_batters[0], compare_batters[1]))
            idx1 = compare_batters.index(d1) if d1 in compare_batters else 0
            batter1 = st.selectbox("Batter 1", compare_batters, index=idx1, key="cmp_b1")
            batter2_options = [b for b in compare_batters if b != batter1]
            idx2 = batter2_options.index(d2) if d2 in batter2_options else 0
            batter2 = st.selectbox("Batter 2", batter2_options, index=idx2, key="cmp_b2")

            bk1 = set(fetch_bowl_kinds(current_mode, batter1))
            bk2 = set(fetch_bowl_kinds(current_mode, batter2))
            common_bk = sorted(list(bk1.intersection(bk2)))
            bowl_kind_compare = st.selectbox("Bowl Kind", common_bk, key="cmp_bk") if common_bk else ""

            l1 = set(fetch_lengths(current_mode, batter1, bowl_kind_compare)) if bowl_kind_compare else set()
            l2 = set(fetch_lengths(current_mode, batter2, bowl_kind_compare)) if bowl_kind_compare else set()
            length_order = ['FULL', 'SHORT', 'GOOD_LENGTH', 'SHORT_OF_A_GOOD_LENGTH']
            common_lengths = [l for l in length_order if l in l1.intersection(l2)]
            selected_compare_lengths = st.multiselect(
                "Length",
                common_lengths,
                default=[common_lengths[0]] if common_lengths else [],
                key="cmp_lengths",
            )
            if not selected_compare_lengths and common_lengths:
                selected_compare_lengths = [common_lengths[0]]

            o1 = set(fetch_outfielders(current_mode, batter1, bowl_kind_compare, selected_compare_lengths)) if selected_compare_lengths and bowl_kind_compare else set()
            o2 = set(fetch_outfielders(current_mode, batter2, bowl_kind_compare, selected_compare_lengths)) if selected_compare_lengths and bowl_kind_compare else set()
            common_outfielders = sorted(list(o1.intersection(o2)), key=lambda x: str(x))
            outfielders_compare = st.selectbox("Number of Outfielders", common_outfielders, key="cmp_out") if common_outfielders else ""

            compare_on = st.multiselect(
                "Compare On",
                COMPARE_SECTIONS,
                default=COMPARE_SECTIONS,
                key="cmp_on",
            )

    if compare_submit:
        if not bowl_kind_compare:
            st.warning("No common bowl kind found between selected batters.")
        elif not selected_compare_lengths:
            st.warning("Select at least one common length.")
        elif not outfielders_compare:
            st.warning("No common outfielder count available for this filter.")
        else:
            st.markdown('Click ⓘ for small metric explainers')

            z1 = fetch_zone_strength(current_mode, batter1, bowl_kind_compare, selected_compare_lengths) or {}
            z2 = fetch_zone_strength(current_mode, batter2, bowl_kind_compare, selected_compare_lengths) or {}
            p1 = fetch_protection_stats(current_mode, batter1, bowl_kind_compare, selected_compare_lengths, outfielders_compare) or {}
            p2 = fetch_protection_stats(current_mode, batter2, bowl_kind_compare, selected_compare_lengths, outfielders_compare) or {}
            w1 = fetch_intelligent_wagon_wheel(current_mode, batter1, bowl_kind_compare, selected_compare_lengths) or {}
            w2 = fetch_intelligent_wagon_wheel(current_mode, batter2, bowl_kind_compare, selected_compare_lengths) or {}
            s1 = fetch_shot_profile(current_mode, batter1, bowl_kind_compare, selected_compare_lengths) or {}
            s2 = fetch_shot_profile(current_mode, batter2, bowl_kind_compare, selected_compare_lengths) or {}
            i1 = fetch_intrel_data(current_mode, batter1, bowl_kind_compare, selected_compare_lengths) or {}
            i2 = fetch_intrel_data(current_mode, batter2, bowl_kind_compare, selected_compare_lengths) or {}

            if "360 Ability" in compare_on:
                d1 = z1.get("dict_360_selected", {})
                d2 = z2.get("dict_360_selected", {})
                rows = []
                for rc, lbl in [("running", "Running 360"), ("boundary", "Boundary 360"), ("overall", "Overall 360")]:
                    v1 = _avg([d1.get(ln, {}).get(rc, {}).get("360_score", 0) for ln in selected_compare_lengths])
                    v2 = _avg([d2.get(ln, {}).get(rc, {}).get("360_score", 0) for ln in selected_compare_lengths])
                    help_text = {
                        "running": "360 degree score of running class runs.",
                        "boundary": "360 degree score of boundary class runs.",
                        "overall": "360 degree score of overall runs.",
                    }[rc]
                    rows.append({"label": lbl, "v1": v1, "v2": v2, "s1": _fmt(v1), "s2": _fmt(v2), "higher_is_better": True, "help": help_text})
                rp1 = float(p1.get("batter_running", 0) or 0)
                rp2 = float(p2.get("batter_running", 0) or 0)
                bp1 = float(p1.get("batter_boundary", 0) or 0)
                bp2 = float(p2.get("batter_boundary", 0) or 0)
                rows.append({"label": "Running Protection", "v1": rp1, "v2": rp2, "s1": _fmt(rp1, "%"), "s2": _fmt(rp2, "%"), "higher_is_better": False, "help": "% of running runs saved by optimal field."})
                rows.append({"label": "Boundary Protection", "v1": bp1, "v2": bp2, "s1": _fmt(bp1, "%"), "s2": _fmt(bp2, "%"), "higher_is_better": False, "help": "% of boundary runs saved by optimal field."})
                p95_1 = _calc_p95_radius(w1, selected_compare_lengths)
                p95_2 = _calc_p95_radius(w2, selected_compare_lengths)
                rows.append({"label": "P95 Radius", "v1": p95_1, "v2": p95_2, "s1": _fmt(p95_1), "s2": _fmt(p95_2), "higher_is_better": True, "help": "A higher p95 means batter plays more difficult shots."})
                _render_compare_rows("360 Ability", batter1, batter2, rows, "cmp_360")

            if "Zone Strengths" in compare_on:
                zp1 = _aggregate_zone_perc(z1, selected_compare_lengths)
                zp2 = _aggregate_zone_perc(z2, selected_compare_lengths)
                rows = []
                for rc in ["overall", "running", "boundary"]:
                    for zn in ["Straight", "Leg", "Off", "Behind"]:
                        v1 = float(zp1.get(rc, {}).get(zn, 0.0))
                        v2 = float(zp2.get(rc, {}).get(zn, 0.0))
                        rows.append({"label": f"{rc.title()} - {zn}", "v1": v1, "v2": v2, "s1": _fmt(v1, "%"), "s2": _fmt(v2, "%"), "higher_is_better": True, "help": f"{rc.title()} runs share % in {zn.lower()} region."})
                _render_compare_rows("Zone Strengths (%)", batter1, batter2, rows, "cmp_zone")

            if "Shot Strengths" in compare_on:
                sp1 = _aggregate_shot_perc(s1, selected_compare_lengths)
                sp2 = _aggregate_shot_perc(s2, selected_compare_lengths)
                rows = []
                for sh in sorted(set(sp1.keys()).union(sp2.keys())):
                    v1 = float(sp1.get(sh, 0.0))
                    v2 = float(sp2.get(sh, 0.0))
                    rows.append({"label": sh, "v1": v1, "v2": v2, "s1": _fmt(v1, "%"), "s2": _fmt(v2, "%"), "higher_is_better": True, "help": f"Runs share % playing {sh}."})
                _render_compare_rows("Shot Strengths (%)", batter1, batter2, rows, "cmp_shots")

            if "Lengthwise Intent" in compare_on:
                li1 = _extract_intrel_by_length(i1, "intent_by_length", selected_compare_lengths)
                li2 = _extract_intrel_by_length(i2, "intent_by_length", selected_compare_lengths)
                rows = [{"label": ln, "v1": li1.get(ln, 0.0), "v2": li2.get(ln, 0.0), "s1": _fmt(li1.get(ln, 0.0), "", 3), "s2": _fmt(li2.get(ln, 0.0), "", 3), "higher_is_better": True, "help": f"A measure of intent vs {ln}."} for ln in selected_compare_lengths]
                _render_compare_rows("Lengthwise Intent", batter1, batter2, rows, "cmp_intent")

            if "Lengthwise Reliability" in compare_on:
                lr1 = _extract_intrel_by_length(i1, "reliability_by_length", selected_compare_lengths)
                lr2 = _extract_intrel_by_length(i2, "reliability_by_length", selected_compare_lengths)
                rows = [{"label": ln, "v1": lr1.get(ln, 0.0), "v2": lr2.get(ln, 0.0), "s1": _fmt(lr1.get(ln, 0.0), "", 3), "s2": _fmt(lr2.get(ln, 0.0), "", 3), "higher_is_better": True, "help": f"A measure of reliability vs {ln}."} for ln in selected_compare_lengths]
                _render_compare_rows("Lengthwise Reliability", batter1, batter2, rows, "cmp_rel")

            if "Lengthwise Int-Rel" in compare_on:
                lir1 = _extract_intrel_by_length(i1, "intrel_by_length", selected_compare_lengths)
                lir2 = _extract_intrel_by_length(i2, "intrel_by_length", selected_compare_lengths)
                rows = [{"label": ln, "v1": lir1.get(ln, 0.0), "v2": lir2.get(ln, 0.0), "s1": _fmt(lir1.get(ln, 0.0), "", 3), "s2": _fmt(lir2.get(ln, 0.0), "", 3), "higher_is_better": True, "help": f"A combined measure of intent and reliability vs {ln}."} for ln in selected_compare_lengths]
                _render_compare_rows("Lengthwise Int-Rel", batter1, batter2, rows, "cmp_intrel")
    else:
        st.info("Set compare filters in sidebar and click **Compare**.")

# Info Tab
# ─────────────────────────────
if active_view == "Information":
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
            <strong style="color: #dc2626;">30 Yard Wall</strong> â€” Your best infielder, placed where most grounded shots are expected.
        </div><div class="special-category">
            <strong style="color: #f97316;">Sprinter</strong> â€” The best runner, placed where batters tend to hit and run singles/doubles in the outfield.
        </div><div class="special-category">
            <strong style="color: #84cc16;">Catcher</strong> â€” The best catcher, placed where batters hit the most boundaries.
        </div><div class="special-category">
            <strong style="color: #fbbf24;">Superfielder</strong> â€” A combination of sprinter and catcher, used if both positions coincide.
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









