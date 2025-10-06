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
st.caption("Live odds. Best picks only. Confidence-weighted units.")
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

        def process_market(df, market_key, label):
            if "bookmakers_0_markets_0_key" not in df.columns:
                return pd.DataFrame()

            market_df = df[df["bookmakers_0_markets_0_key"] == market_key].copy()
            if market_df.empty:
                return pd.DataFrame()

            # Basic info
            market_df["Home"] = market_df.get("home_team", "Unknown")
            market_df["Away"] = market_df.get("away_team", "Unknown")
            market_df["Book"] = market_df.get("bookmakers_0_title", "Unknown")
            market_df["Pick"] = market_df.get("bookmakers_0_markets_0_outcomes_0_name", "")

            # Odds & probabilities
            market_df["Odds"] = market_df.get("bookmakers_0_markets_0_outcomes_0_price", np.nan)
            market_df["Decimal Odds"] = market_df["Odds"].apply(american_to_decimal)
            market_df["Implied Prob"] = market_df["Odds"].apply(implied_prob_american)
            market_df["True Prob"] = market_df["Implied Prob"]  # simplification

            # Edge
            market_df["Edge"] = market_df["Decimal Odds"] * market_df["True Prob"] - 1

            # Confidence-weighted Units
            market_df["Confidence"] = market_df["True Prob"] * (market_df["Edge"] + 1)
            market_df["Stake ($)"] = (market_df["Confidence"] * bankroll * kelly_cap).round(2)
            market_df["Units"] = (market_df["Stake ($)"] / unit_size).round(2)

            # Keep only useful columns
            out = market_df[[
                "commence_time", "Home", "Away", "Book", "Pick", "Odds",
                "Decimal Odds", "Implied Prob", "True Prob", "Edge", "Stake ($)", "Units"
            ]].copy()

            # Drop duplicates → keep best Edge per game
            out = out.sort_values(by="Edge", ascending=False)
            out = out.drop_duplicates(subset=["Home", "Away"], keep="first")

            return out.head(5).reset_index(drop=True)

        # --- Moneylines
        with tabs[0]:
            st.subheader("Best Moneyline Picks")
            ml = process_market(df, "h2h", "Moneyline")
            if ml.empty:
                st.info("No moneyline data available.")
            else:
                st.dataframe(ml, use_container_width=True)

        # --- Totals
        with tabs[1]:
            st.subheader("Best Totals Picks")
            totals = process_market(df, "totals", "Totals")
            if totals.empty:
                st.info("No totals data available.")
            else:
                st.dataframe(totals, use_container_width=True)

        # --- Spreads
        with tabs[2]:
            st.subheader("Best Spread Picks")
            spreads = process_market(df, "spreads", "Spreads")
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
    if not ODDS_API_KEY:
        st.error("No ODDS_API_KEY detected. Add it to your `.env` or Streamlit **Secrets** (cloud).")
