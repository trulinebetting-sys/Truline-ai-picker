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
# Fetch Odds API
# ----------------------------
@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    """Fetch odds and flatten into a clean DataFrame"""
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

    data = r.json()
    rows = []
    for event in data:
        dt = event.get("commence_time", "")
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        for bm in event.get("bookmakers", []):
            book = bm.get("title", "")
            for mk in bm.get("markets", []):
                market_type = mk.get("key", "")
                for outcome in mk.get("outcomes", []):
                    rows.append({
                        "Date/Time": dt,
                        "Home": home,
                        "Away": away,
                        "Sportsbook": book,
                        "Market": market_type,
                        "Pick": outcome.get("name"),
                        "Odds": outcome.get("price")
                    })
    return pd.DataFrame(rows)

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
        df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")
        df["Date/Time"] = df["Date/Time"].dt.strftime("%b %d, %I:%M %p ET")

        # Tabs
        tabs = st.tabs(["Moneylines", "Spreads", "Totals", "Raw Data"])

        # --- Moneylines
        with tabs[0]:
            st.subheader("Moneyline Picks")
            ml = df[df["Market"] == "h2h"]
            if ml.empty:
                st.info("No moneyline data.")
            else:
                st.dataframe(ml.head(20), use_container_width=True)

        # --- Spreads
        with tabs[1]:
            st.subheader("Spread Picks")
            spreads = df[df["Market"] == "spreads"]
            if spreads.empty:
                st.info("No spreads data.")
            else:
                st.dataframe(spreads.head(20), use_container_width=True)

        # --- Totals
        with tabs[2]:
            st.subheader("Over/Under Picks")
            totals = df[df["Market"] == "totals"]
            if totals.empty:
                st.info("No totals data.")
            else:
                st.dataframe(totals.head(20), use_container_width=True)

        # --- Raw
        with tabs[3]:
            st.subheader("Raw Odds Data")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
