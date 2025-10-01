import streamlit as st

# --- PAGE CONFIG (sets tab bar logo + title) ---
st.set_page_config(
    page_title="TruLine Betting",
    page_icon="assets/logo.png2",  # favicon / tab bar logo
    layout="wide"
)

# --- GLOBAL STYLE ---
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        section[data-testid="stSidebar"] {display: none !important;}
        h1, h2, h3, h4, h5, h6 {color: #0d1b2a;}  /* Dark navy headings */
        p, li {color: #1b263b;}                  /* Softer dark gray text */
        .btn-primary {
            background-color: #0077b6;
            color: white;
            padding: 10px 20px;
            border-radius: 6px;
            text-decoration: none;
        }
        .btn-primary:hover {
            background-color: #023e8a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HERO SECTION ---
col1, _ = st.columns([7, 5], gap="large")

with col1:
    st.image("assets/logo.png2", width=100)  # new small logo top-left
    st.markdown(
        """
        # TruLine Betting  
        **We scan the lines.**  
        **You place the bets.**  

        Find high-edge opportunities using AI-driven picks and bankroll controls.  

        [Try AI Picks](#ai-genius-picks)
        """,
        unsafe_allow_html=True,
    )

# --- HOW IT WORKS ---
st.markdown("---")
st.markdown("## How does Positive EV Betting work?")
st.markdown(
    """
    - **Compute fair odds** by removing the bookmaker’s vig.  
    - **Reference price**: Use a sharp book (like Pinnacle) when available.  
    - **Find edge**: Bets where offered odds exceed fair odds.  
    - **Stake sizing**: Uses capped Kelly.  
    """
)

# --- AI GENIUS PICKS PLACEHOLDER ---
st.markdown("---")
st.markdown("## 🧠 AI Genius Picks")
st.info("This is where your AI betting picks will appear soon!")
