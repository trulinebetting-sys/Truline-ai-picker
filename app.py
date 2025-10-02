import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TruLine Betting",
    page_icon="assets/logo.png",  # favicon in tab bar
    layout="wide"
)

# --- LOGO + HEADER ---
st.image("assets/logo.png", width=120)  # top-left logo
st.title("TruLine Betting")

st.markdown(
    """
    ### We scan the lines.  
    You place the bets.  

    Find high-edge opportunities using AI-driven picks and bankroll controls.
    """
)

# --- CTA BUTTONS ---
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.button("Try AI Picks", key="try_ai_picks")

with col2:
    st.button("Arbitrage (coming soon)", key="arbitrage_button", disabled=True)

with col3:
    st.button("Parlay Builder (coming soon)", key="parlay_button", disabled=True)

# --- AI PICKS SECTION ---
st.markdown("---")
st.subheader("🤖 AI Genius Picks")

st.write(
    """
    This section will eventually connect to your AI model that generates betting picks.  
    For now, you can use it as a placeholder for testing.
    """
)

# Example: Placeholder picks
example_picks = [
    {"Game": "Team A vs Team B", "Pick": "Team A ML", "Confidence": "78%"},
    {"Game": "Team C vs Team D", "Pick": "Over 45.5", "Confidence": "72%"},
    {"Game": "Team E vs Team F", "Pick": "Team F +3.5", "Confidence": "81%"},
]

for i, pick in enumerate(example_picks, start=1):
    st.markdown(f"**Pick {i}:** {pick['Game']} → {pick['Pick']} ({pick['Confidence']})")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2025 TruLine Betting | Powered by Streamlit")
