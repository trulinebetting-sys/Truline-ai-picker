import os
import requests
import httpx
import pandas as pd
import numpy as np
import streamlit as st

# ----------------------------
# Load keys
# ----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("⚠️ dotenv not installed. Using Streamlit secrets instead.")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", st.secrets.get("ODDS_API_KEY", ""))
API_SPORTS_KEY = os.getenv("API_SPORTS_KEY", st.secrets.get("API_SPORTS_KEY", ""))

# ----------------------------
# Config
# ----------------------------
SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
}

st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker")
st.caption("Live odds + historical stats → smarter picks.")
st.write("---")

# ----------------------------
# Fetch live odds (Odds API)
# ----------------------------
@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY.")
        return pd.DataFrame()

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {"apiKey": ODDS_API_KEY, "regions": regions, "markets": markets, "oddsFormat": "american"}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        st.error(f"Odds API error {r.status_code}: {r.text}")
        return pd.DataFrame()
    return pd.json_normalize(r.json(), sep="_")

# ----------------------------
# Fetch historical stats (API-Sports)
# ----------------------------
@st.cache_data(ttl=3600)
def fetch_team_stats(sport: str, team_id: int, season: str = "2024") -> dict:
    if not API_SPORTS_KEY:
        return {}

    url = f"https://v1.{sport}.api-sports.io/teams/statistics"
    headers = {"x-apisports-key": API_SPORTS_KEY}
    params = {"season": season, "team": team_id, "league": 1}  # league=1 = NFL, NBA, MLB differ
    try:
        r = httpx.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        else:
            return {}
    except Exception:
        return {}

# ----------------------------
# Sidebar filters
# ----------------------------
with st.sidebar:
    st.header("Filters")
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()), index=0)
    regions = st.text_input("Regions", value="us")
    min_edge = st.slider("Min Edge (%)", 0.0, 10.0, 1.0, 0.25) / 100.0
    fetch = st.button("Fetch Live Odds")

sport_key = SPORT_OPTIONS[sport_name]

# ----------------------------
# Main app
# ----------------------------
if fetch:
    df = fetch_odds(sport_key, regions)

    if df.empty:
        st.warning("No data returned.")
    else:
        if "commence_time" in df.columns:
            df["commence_time"] = pd.to_datetime(df["commence_time"])
            df["commence_time"] = df["commence_time"].dt.strftime("%b %d, %I:%M %p ET")

        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "Raw Data"])

        def format_table(df, market_key: str, label: str):
            if "bookmakers_0_markets_0_key" not in df.columns:
                return pd.DataFrame()

            filtered = df[df["bookmakers_0_markets_0_key"] == market_key].copy()
            if filtered.empty:
                return pd.DataFrame()

            # Example placeholder: assign confidence based on odds
            filtered["confidence"] = filtered["bookmakers_0_markets_0_outcomes_0_price"].apply(
                lambda x: 0.5 if pd.isna(x) else 1 - abs(int(x)) / 1000
            )

            # Unit size based on confidence
            filtered["units"] = (filtered["confidence"] * 5).round(2)

            # Select key columns
            return filtered[[
                "commence_time",
                "home_team",
                "away_team",
                "bookmakers_0_title",
                "bookmakers_0_markets_0_outcomes_0_name",
                "bookmakers_0_markets_0_outcomes_0_price",
                "confidence",
                "units"
            ]].rename(columns={
                "bookmakers_0_title": "book",
                "bookmakers_0_markets_0_outcomes_0_name": "pick",
                "bookmakers_0_markets_0_outcomes_0_price": "odds"
            }).sort_values("confidence", ascending=False).head(5)

        # Moneylines
        with tabs[0]:
            ml = format_table(df, "h2h", "Moneylines")
            st.subheader("Top 5 Moneyline Picks")
            if ml.empty:
                st.info("No edges found.")
            else:
                st.dataframe(ml, use_container_width=True)

        # Totals
        with tabs[1]:
            totals = format_table(df, "totals", "Totals")
            st.subheader("Top 5 Totals Picks")
            if totals.empty:
                st.info("No totals found.")
            else:
                st.dataframe(totals, use_container_width=True)

        # Spreads
        with tabs[2]:
            spreads = format_table(df, "spreads", "Spreads")
            st.subheader("Top 5 Spread Picks")
            if spreads.empty:
                st.info("No spreads found.")
            else:
                st.dataframe(spreads, use_container_width=True)

        # Raw
        with tabs[3]:
            st.subheader("Raw Data")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters and click **Fetch Live Odds**.")
