import os
import streamlit as st

# -------------------------
# PAGE CONFIG (favicon/logo)
# -------------------------
# If the logo exists, use it as the page icon; otherwise, use a default emoji.
logo_path = "assets/logo.png"
page_icon = logo_path if os.path.exists(logo_path) else "✨"

st.set_page_config(
    page_title="TruLine Betting",
    page_icon=page_icon,
    layout="wide"
)

# -------------------------
# HEADER / BRAND
# -------------------------
st.markdown(
    """
    <style>
      /* keep things clean and readable */
      .app-header {display:flex; align-items:center; gap:14px; margin:8px 0 18px 0;}
      .brand-title {font-weight:800; font-size:1.6rem; margin:0; color:#0d1b2a;}
      .lead {color:#243447;}
    </style>
    """,
    unsafe_allow_html=True,
)

col_logo, col_title = st.columns([1, 9])

with col_logo:
    # Show a small logo if present; otherwise, show nothing (no crash)
    if os.path.exists(logo_path):
        st.image(logo_path, width=60)
with col_title:
    st.markdown('<div class="app-header"><h1 class="brand-title">TruLine Betting</h1></div>', unsafe_allow_html=True)

st.markdown(
    """
    ### We scan the lines.  
    You place the bets.
    """,
)
st.write(
    "Find high-edge opportunities using AI-driven picks and bankroll controls."
)

# -------------------------
# ACTIONS / CTA
# -------------------------
st.markdown("---")
c1, c2, c3 = st.columns(3)

with c1:
    st.button("Try AI Picks", key="cta_try_ai")
with c2:
    st.button("Arbitrage (coming soon)", key="cta_arb", disabled=True)
with c3:
    st.button("Parlay Builder (coming soon)", key="cta_parlay", disabled=True)

# -------------------------
# AI PICKS SECTION (placeholder)
# -------------------------
st.markdown("---")
st.subheader("🤖 AI Genius Picks")

st.write(
    "This section will eventually connect to your AI model that generates betting picks. "
    "For now, here are placeholder picks to confirm the app runs end-to-end."
)

example_picks = [
    {"Game": "Team A vs Team B", "Pick": "Team A ML", "Confidence": "78%"},
    {"Game": "Team C vs Team D", "Pick": "Over 45.5", "Confidence": "72%"},
    {"Game": "Team E vs Team F", "Pick": "Team F +3.5", "Confidence": "81%"},
]

for i, pick in enumerate(example_picks, start=1):
    st.markdown(f"**Pick {i}:** {pick['Game']} → {pick['Pick']} ({pick['Confidence']})")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption("© 2025 TruLine Betting | Powered by Streamlit")
