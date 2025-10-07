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
    min_edge = st.slider("Min Edge (%)", 0.0, 10.0, DEFAULT_MIN_EDGE*100, 0.25) / 100.0
    kelly_cap = st.slider("Kelly Cap", 0.0, 1.0, DEFAULT_KELLY_CAP, 0.05)
    fetch = st.button("Fetch Live Odds")

# ----------------------------
# Helper for safe filtering
# ----------------------------
def safe_filter(df, key, value):
    if key not in df.columns:
        return pd.DataFrame()
    return df[df[key] == value]

# ----------------------------
# Format table
# ----------------------------
def format_table(df):
    cols_to_keep = [
        "commence_time",
        "home_team",
        "away_team",
        "bookmakers_0_title",
        "bookmakers_0_markets_0_outcomes_0_name",
        "bookmakers_0_markets_0_outcomes_0_price"
    ]
    available = [c for c in cols_to_keep if c in df.columns]
    table = df[available].copy()

    if "commence_time" in table.columns:
        table["commence_time"] = pd.to_datetime(table["commence_time"])
        table["commence_time"] = table["commence_time"].dt.strftime("%b %d, %I:%M %p ET")

    table.rename(columns={
        "commence_time": "Date/Time",
        "home_team": "Home",
        "away_team": "Away",
        "bookmakers_0_title": "Sportsbook",
        "bookmakers_0_markets_0_outcomes_0_name": "Pick",
        "bookmakers_0_markets_0_outcomes_0_price": "Odds"
    }, inplace=True)

    return table

# ----------------------------
# Main content
# ----------------------------
sport_key = SPORT_OPTIONS[sport_name]
if fetch:
    df = fetch_odds(sport_key=sport_key, regions=regions)
    if df.empty:
        st.warning("No data returned. Try another sport or check API quota.")
    else:
        # Tabs
        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "Raw Data"])

        # --- Moneylines
        with tabs[0]:
            st.subheader("Best Moneyline Picks")
            ml = safe_filter(df, "bookmakers_0_markets_0_key", "h2h")
            if not ml.empty:
                st.dataframe(format_table(ml), use_container_width=True)
            else:
                st.info("No moneyline data available.")

        # --- Totals
        with tabs[1]:
            st.subheader("Best Over/Under Picks")
            totals = safe_filter(df, "bookmakers_0_markets_0_key", "totals")
            if not totals.empty:
                st.dataframe(format_table(totals), use_container_width=True)
            else:
                st.info("No totals data available.")

        # --- Spreads
        with tabs[2]:
            st.subheader("Best Spread Picks")
            spreads = safe_filter(df, "bookmakers_0_markets_0_key", "spreads")
            if not spreads.empty:
                st.dataframe(format_table(spreads), use_container_width=True)
            else:
                st.info("No spreads data available.")

        # --- Raw Data
        with tabs[3]:
            st.subheader("Raw Odds Data")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
