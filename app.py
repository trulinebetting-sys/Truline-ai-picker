import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TruLine Betting",
    page_icon="assets/logo.png",  # use your logo as favicon
    layout="wide"
)

# --- HIDE SIDEBAR ---
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        section[data-testid="stSidebar"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HEADER (logo + title) ---
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px;">
        <img src="app/static/logo.png" width="60">
        <h1 style="margin: 0;">TruLine Betting</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- HERO SECTION ---
col1, col2 = st.columns([7, 5], gap="large")

with col1:
    st.markdown(
        """
        <div class="hero">
            <h2>We scan the lines.<br>You place the bets.</h2>
            <p class="lead">Find high-edge opportunities using AI-driven picks and bankroll controls.</p>
            <div style="margin-top: 16px;">
                <a class="btn btn-primary" href="#ai">Try AI Picks</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.image("assets/logo.png", use_container_width=True)

# --- AI PICKER (placeholder for now) ---
st.markdown("---")
st.markdown("## 🔮 AI Genius Picks")
st.info("This is where your AI betting picks will appear soon. 🚀")

# --- HOW IT WORKS ---
st.markdown("---")
st.markdown("## How does AI Betting work?")
st.markdown(
    """
    - **AI Predictions**: Analyze stats, trends, and odds.  
    - **Find value bets**: Identify opportunities with positive edge.  
    - **Smart bankroll management**: Avoid chasing losses, maximize ROI.  
    """
)

# --- FOOTER ---
st.markdown("---")
st.markdown("© 2025 TruLine Betting · Built with Streamlit")
