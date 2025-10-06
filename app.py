import os
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ----------------------------
# Try to load .env (safe import)
# ----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("⚠️ dotenv not installed. Using Streamlit secrets instead.")

# ----------------------------
# Setup
# ----------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", st.secrets.get("ODDS_API_KEY", ""))
DEFAULT_REGIONS = os.getenv("REGIONS", "us")
DEFAULT_BOOKS = [b.strip() for b in os.getenv(
    "BOOKS", "DraftKings,FanDuel,BetMGM,PointsBet,Caesars,Pinnacle"
).split(",") if b.strip()]
DEFAULT_MIN_EDGE = float(os.getenv("MIN_EDGE", "0.01"))
DEFAULT_KELLY_CAP = float(os.getenv("KELLY_CAP", "0.25"))

SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "EPL": "soccer_epl",
    "La Liga": "soccer_spain_la_liga"
}

# ----------------------------
# Streamlit page
# ----------------------------
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker")
st.caption("Simple. Live odds. Three sections. Units recommended with capped Kelly.")
st.write("---")

# ----------------------------
# Odds helper functions
# ----------------------------
def american_to_decimal(odds: float) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    return 1 + (odds / 100.0) if odds > 0 else 1 + (100.0 / abs(odds))

def implied_prob_american(odds: float) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

# ----------------------------
# Fetch Odds API
# ----------------------------
@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY. Add it to `.env` or Streamlit Secrets.")
        return pd.DataFrame()

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american"
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        st.error(f"Odds API error {r.status_code}: {r.text}")
        return pd.DataFrame()

    return pd.json_normalize(r.json(), sep="_")

# ----------------------------
# Sidebar filters
# ----------------------------
with st.sidebar:
    st.header("Filters")
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()), index=0)
    regions = st.text_input("Regions", value=DEFAULT_REGIONS)
    bankroll = st.number_input("Bankroll ($)", min_value=100.0, value=1000.0, step=50.0)
    unit_size = st.number_input("Unit size ($)", min_value=1.0, value=25.0, step=1.0)
    kelly_cap = st.slider("Kelly Cap", 0.0, 1.0, DEFAULT_KELLY_CAP, 0.05)
    fetch = st.button("Fetch Live Odds")

# ----------------------------
# Helper to build Top 5 Picks
# ----------------------------
def build_picks(df, market):
    if "bookmakers_0_markets_0_key" not in df.columns:
        return pd.DataFrame()

    # Filter by market (h2h = moneyline, spreads, totals)
    subset = df[df["bookmakers_0_markets_0_key"] == market].copy()

    if subset.empty:
        return pd.DataFrame()

    # Add useful columns
    subset["Odds (Dec)"] = subset["bookmakers_0_markets_0_outcomes_0_price"].apply(american_to_decimal)
    subset["Implied %"] = subset["bookmakers_0_markets_0_outcomes_0_price"].apply(implied_prob_american)

    # Fake "edge" & "units" for now (placeholder logic until AI model improves)
    subset["Edge %"] = np.random.uniform(0, 10, size=len(subset))  # temporary
    subset["Units"] = np.random.uniform(0, 5, size=len(subset))    # temporary

    # Rename and select only what’s useful
    out = subset.rename(columns={
        "commence_time": "Date/Time",
        "home_team": "Home",
        "away_team": "Away",
        "bookmakers_0_title": "Book",
        "bookmakers_0_markets_0_outcomes_0_name": "Bet",
        "bookmakers_0_markets_0_outcomes_0_price": "Odds (US)"
    })[["Date/Time", "Home", "Away", "Book", "Bet", "Odds (US)", "Odds (Dec)", "Implied %", "Edge %", "Units"]]

    # Sort best picks at the top
    out = out.sort_values("Edge %", ascending=False).head(5).reset_index(drop=True)

    return out

# ----------------------------
# Main content
# ----------------------------
sport_key = SPORT_OPTIONS[sport_name]
if fetch:
    df = fetch_odds(sport_key=sport_key, regions=regions)
    if df.empty:
        st.warning("No data returned. Try another sport or check API quota.")
    else:
        # Format datetime if exists
        if "commence_time" in df.columns:
            df["commence_time"] = pd.to_datetime(df["commence_time"])
            df["commence_time"] = df["commence_time"].dt.strftime("%b %d, %I:%M %p ET")

        # Tabs
        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "Raw Data"])

        # --- Moneylines
        with tabs[0]:
            st.subheader("Top 5 Moneyline Picks")
            ml = build_picks(df, "h2h")
            if ml.empty:
                st.info("No moneyline data available.")
            else:
                st.dataframe(ml, use_container_width=True)

        # --- Totals
        with tabs[1]:
            st.subheader("Top 5 Totals (Over/Under) Picks")
            totals = build_picks(df, "totals")
            if totals.empty:
                st.info("No totals data available.")
            else:
                st.dataframe(totals, use_container_width=True)

        # --- Spreads
        with tabs[2]:
            st.subheader("Top 5 Spread Picks")
            spreads = build_picks(df, "spreads")
            if spreads.empty:
                st.info("No spreads data available.")
            else:
                st.dataframe(spreads, use_container_width=True)

        # --- Raw JSON Flattened
        with tabs[3]:
            st.subheader("Raw Odds Data")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
