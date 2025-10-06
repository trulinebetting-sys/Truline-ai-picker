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
st.caption("Live odds. AI confidence units. Three sections.")
st.write("---")

# ----------------------------
# Helper functions
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

def assign_units(prob: float) -> float:
    if pd.isna(prob):
        return 0
    if prob > 0.70:
        return 3.0
    elif prob > 0.60:
        return 2.0
    elif prob > 0.55:
        return 1.5
    elif prob > 0.50:
        return 1.0
    else:
        return 0.5

# ----------------------------
# Fetch Odds API
# ----------------------------
@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY. Add it to `.env` or Streamlit Secrets.")
        return pd.DataFrame()

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {"apiKey": ODDS_API_KEY, "regions": regions, "markets": markets, "oddsFormat": "american"}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        st.error(f"Odds API error {r.status_code}: {r.text}")
        return pd.DataFrame()

    df = pd.json_normalize(r.json(), sep="_")

    # Add implied prob + units safely if price column exists
    price_cols = [c for c in df.columns if "price" in c]
    if price_cols:
        df["implied_prob"] = df[price_cols[0]].apply(implied_prob_american)
        df["units"] = df["implied_prob"].apply(assign_units)

    return df

# ----------------------------
# Sidebar filters
# ----------------------------
with st.sidebar:
    st.header("Filters")
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()), index=0)
    regions = st.text_input("Regions", value=DEFAULT_REGIONS)
    fetch = st.button("Fetch Live Odds")

# ----------------------------
# Main content
# ----------------------------
sport_key = SPORT_OPTIONS[sport_name]
if fetch:
    df = fetch_odds(sport_key=sport_key, regions=regions)
    if df.empty:
        st.warning("No data returned. Try another sport or check API quota.")
    else:
        # Format datetime
        if "commence_time" in df.columns:
            df["commence_time"] = pd.to_datetime(df["commence_time"])
            df["commence_time"] = df["commence_time"].dt.strftime("%b %d, %I:%M %p ET")

        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "Raw Data"])

        def format_display(df, market_key):
            """Extract and clean display table for a market"""
            if "bookmakers_0_markets_0_key" not in df.columns:
                return pd.DataFrame()
            sub = df[df["bookmakers_0_markets_0_key"] == market_key].copy()
            if sub.empty:
                return sub
            # Select columns safely
            cols = {
                "commence_time": "Date/Time",
                "home_team": "Home Team",
                "away_team": "Away Team",
                "bookmakers_0_title": "Sportsbook",
                "bookmakers_0_markets_0_outcomes_0_name": "Pick",
                "bookmakers_0_markets_0_outcomes_0_price": "Odds (US)",
                "units": "Units"
            }
            available = [c for c in cols if c in sub.columns]
            return sub[available].rename(columns=cols).head(5)

        with tabs[0]:
            st.subheader("Best Moneyline Picks (Top 5)")
            ml = format_display(df, "h2h")
            if ml.empty:
                st.info("No moneyline data available.")
            else:
                st.dataframe(ml, use_container_width=True)

        with tabs[1]:
            st.subheader("Best Totals Picks (Top 5)")
            totals = format_display(df, "totals")
            if totals.empty:
                st.info("No totals data available.")
            else:
                st.dataframe(totals, use_container_width=True)

        with tabs[2]:
            st.subheader("Best Spread Picks (Top 5)")
            spreads = format_display(df, "spreads")
            if spreads.empty:
                st.info("No spreads data available.")
            else:
                st.dataframe(spreads, use_container_width=True)

        with tabs[3]:
            st.subheader("Raw Odds Data")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
