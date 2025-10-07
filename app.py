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
DEFAULT_MIN_EDGE = float(os.getenv("MIN_EDGE", "0.01"))

SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb"
}

# ----------------------------
# Load historical data
# ----------------------------
def load_historical_data():
    hist = {}
    for sport in SPORT_OPTIONS.keys():
        path = f"data/{sport.lower()}_results.csv"
        if os.path.exists(path):
            hist[sport] = pd.read_csv(path)
        else:
            hist[sport] = pd.DataFrame()
    return hist

HIST_DATA = load_historical_data()

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
# AI Genius Picker Logic
# ----------------------------
def attach_ai_confidence(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Combine historical win rates with live implied probabilities for confidence + units."""
    df = df.copy()
    if df.empty:
        df["Confidence"] = np.nan
        df["Units"] = np.nan
        return df

    hist = HIST_DATA.get(sport, pd.DataFrame())

    if not hist.empty and "team" in hist.columns and "win" in hist.columns:
        team_win_rates = hist.groupby("team")["win"].mean().to_dict()
    else:
        team_win_rates = {}

    def conf(row):
        team = row.get("home_team", "")
        imp_prob = implied_prob_american(row.get("bookmakers_0_markets_0_outcomes_0_price", np.nan))
        hist_rate = team_win_rates.get(team, 0.5)  # fallback to 50% if no data
        return 0.5 * imp_prob + 0.5 * hist_rate if not pd.isna(imp_prob) else hist_rate

    df["Confidence"] = df.apply(conf, axis=1)
    df["Units"] = df["Confidence"].apply(lambda x: round(5 * x, 1) if not pd.isna(x) else np.nan)
    return df

# ----------------------------
# Streamlit page
# ----------------------------
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker")
st.caption("Live odds + historical data → AI Genius Picks with unit size suggestions")
st.write("---")

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
        df = attach_ai_confidence(df, sport_name)

        if "commence_time" in df.columns:
            df["commence_time"] = pd.to_datetime(df["commence_time"])
            df["commence_time"] = df["commence_time"].dt.strftime("%b %d, %I:%M %p ET")

        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "Raw Data"])

        def safe_filter(df, key, value):
            if key not in df.columns:
                return pd.DataFrame()
            return df[df[key] == value]

        def format_table(subdf):
            if subdf.empty:
                return subdf
            return subdf[[
                "commence_time", "home_team", "away_team",
                "bookmakers_0_title", "bookmakers_0_markets_0_outcomes_0_name",
                "bookmakers_0_markets_0_outcomes_0_price", "Confidence", "Units"
            ]].rename(columns={
                "commence_time": "Date/Time",
                "home_team": "Home",
                "away_team": "Away",
                "bookmakers_0_title": "Sportsbook",
                "bookmakers_0_markets_0_outcomes_0_name": "Pick",
                "bookmakers_0_markets_0_outcomes_0_price": "Odds (US)"
            }).head(5)

        # --- Moneylines
        with tabs[0]:
            st.subheader("Best Moneyline Picks")
            ml = safe_filter(df, "bookmakers_0_markets_0_key", "h2h")
            st.dataframe(format_table(ml), use_container_width=True) if not ml.empty else st.info("No moneyline data available.")

        # --- Totals
        with tabs[1]:
            st.subheader("Best Over/Under Picks")
            totals = safe_filter(df, "bookmakers_0_markets_0_key", "totals")
            st.dataframe(format_table(totals), use_container_width=True) if not totals.empty else st.info("No totals data available.")

        # --- Spreads
        with tabs[2]:
            st.subheader("Best Spread Picks")
            spreads = safe_filter(df, "bookmakers_0_markets_0_key", "spreads")
            st.dataframe(format_table(spreads), use_container_width=True) if not spreads.empty else st.info("No spreads data available.")

        # --- Raw Data
        with tabs[3]:
            st.subheader("Raw Odds Data")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
