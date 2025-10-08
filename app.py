import os
import pandas as pd
import numpy as np
import requests
import streamlit as st

# ----------------------------
# Safe dotenv load
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
    "MLB": "baseball_mlb",
    "EPL": "soccer_epl",
    "La Liga": "soccer_spain_la_liga"
}

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker")
st.caption("Live odds with AI-assisted pick confidence.")
st.write("---")

# ----------------------------
# Helpers
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
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)

# ----------------------------
# Fetch Odds
# ----------------------------
@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets="h2h,spreads,totals"):
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY. Add to `.env` or Streamlit secrets.")
        return pd.DataFrame()

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {"apiKey": ODDS_API_KEY, "regions": regions, "markets": markets, "oddsFormat": "american"}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        st.error(f"Odds API error {r.status_code}: {r.text}")
        return pd.DataFrame()

    df = pd.json_normalize(r.json(), sep="_")
    if "commence_time" in df.columns:
        df["commence_time"] = pd.to_datetime(df["commence_time"])
        df["commence_time"] = df["commence_time"].dt.strftime("%b %d, %I:%M %p ET")
    return df

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Filters")
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()), index=0)
    regions = st.text_input("Regions", value=DEFAULT_REGIONS)
    fetch = st.button("Fetch Live Odds")

sport_key = SPORT_OPTIONS[sport_name]

# ----------------------------
# Main Content
# ----------------------------
if fetch:
    df = fetch_odds(sport_key, regions)
    if df.empty:
        st.warning("No data returned.")
    else:
        tabs = st.tabs(["Moneylines", "Spreads", "Totals", "AI Picker", "Raw Data"])

        # --- Moneylines
        with tabs[0]:
            ml = df[df.get("bookmakers_0_markets_0_key") == "h2h"]
            if ml.empty:
                st.info("No moneyline data.")
            else:
                table = ml[["commence_time", "home_team", "away_team",
                            "bookmakers_0_title", "bookmakers_0_markets_0_outcomes_0_name",
                            "bookmakers_0_markets_0_outcomes_0_price"]].copy()
                table.rename(columns={
                    "commence_time": "Date/Time",
                    "home_team": "Home",
                    "away_team": "Away",
                    "bookmakers_0_title": "Sportsbook",
                    "bookmakers_0_markets_0_outcomes_0_name": "Pick",
                    "bookmakers_0_markets_0_outcomes_0_price": "Odds"
                }, inplace=True)
                st.dataframe(table.head(10), use_container_width=True)

        # --- Spreads
        with tabs[1]:
            spreads = df[df.get("bookmakers_0_markets_0_key") == "spreads"]
            if spreads.empty:
                st.info("No spreads data.")
            else:
                st.dataframe(spreads.head(10), use_container_width=True)

        # --- Totals
        with tabs[2]:
            totals = df[df.get("bookmakers_0_markets_0_key") == "totals"]
            if totals.empty:
                st.info("No totals data.")
            else:
                st.dataframe(totals.head(10), use_container_width=True)

        # --- AI Picker (very basic for now)
        with tabs[3]:
            st.subheader("🤖 AI Suggested Picks (Placeholder)")
            if "bookmakers_0_markets_0_outcomes_0_price" in df.columns:
                df["Implied_Prob"] = df["bookmakers_0_markets_0_outcomes_0_price"].apply(implied_prob_american)
                top_picks = df.sort_values("Implied_Prob", ascending=False).head(5)
                st.write("Top 5 safest picks (highest implied probability):")
                st.dataframe(top_picks[["commence_time", "home_team", "away_team",
                                        "bookmakers_0_title",
                                        "bookmakers_0_markets_0_outcomes_0_name",
                                        "bookmakers_0_markets_0_outcomes_0_price",
                                        "Implied_Prob"]], use_container_width=True)

        # --- Raw Data
        with tabs[4]:
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
