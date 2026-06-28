import streamlit as st
st.set_page_config(layout="wide", page_title="Batter Analysis | Cricket Analytics")

NEW_APP_URL = "https://cricbitbat.up.railway.app/"

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: #05060a; }
        .moved-wrap {
            min-height: 80vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            gap: 1.1rem;
            padding: 2rem 1rem;
        }
        .moved-wrap h1 {
            color: white;
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0;
        }
        .moved-wrap p {
            color: rgba(255,255,255,0.65);
            font-size: 1.05rem;
            max-width: 480px;
            margin: 0;
        }
        .moved-wrap a.cta {
            display: inline-block;
            margin-top: 0.6rem;
            padding: 0.8rem 1.8rem;
            border-radius: 12px;
            background: linear-gradient(135deg,#14b8a6,#22d3ee);
            color: #04141a;
            font-weight: 800;
            font-size: 1.05rem;
            text-decoration: none;
        }
        .moved-wrap .url-text {
            color: #22d3ee;
            font-size: 0.9rem;
            word-break: break-all;
        }
    </style>
    <div class="moved-wrap">
        <h1>We've moved! 🚀</h1>
        <p>CricBit has a new home with a faster, better experience. This Streamlit app is no longer active —
        head over to the link below to keep using CricBit.</p>
        <a class="cta" href="__URL__" target="_blank">Go to CricBit →</a>
        <p class="url-text">__URL__</p>
    </div>
    """.replace("__URL__", NEW_APP_URL),
    unsafe_allow_html=True,
)
st.stop()
