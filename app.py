import streamlit as st
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TruLine Betting",
    page_icon="assets/logo.png",  # small logo in browser tab
    layout="wide"
)

# --- HERO SECTION ---
col1, col2 = st.columns([7, 5], gap="large")

with col1:
    st.markdown(
        """
        <div class="hero">
            <h2>TruLine Betting</h2>
            <h3>We scan the lines.<br>You place the bets.</h3>
            <p class="lead">
                Find high-edge opportunities using AI-driven picks and bankroll controls.
            </p>
            <div style="margin-top: 16px;">
                <a class="btn btn-primary" href="#ai">Try AI Picks</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ❌ Removed col2 image (big logo) so it doesn’t show up anymore

# --- HOW IT WORKS ---
st.markdown("---")
st.markdown("## How does TruLine Betting work?")
st.markdown(
    """
    - **AI-driven picks**: Smart predictions tailored to sports markets.  
    - **Bankroll discipline**: Helps you size bets responsibly.  
    - **Simplified insights**: Only the best opportunities, not noise.  
    """
)

# --- AI GENIUS PICKS SECTION ---
st.markdown("---")
st.markdown("## 🧠 AI Genius Picks", unsafe_allow_html=True)

st.info("⚡ Coming soon: AI-powered betting recommendations.")
