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
st.caption("Live odds + AI Genius recommendations. Organized picks with confidence & unit size.")
st.write("---")

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
                    point = outcome.get("point")
                    pick = outcome.get("name")
                    # Build readable line
                    if market_type == "spreads" and point is not None:
                        bet = f"{pick} {point:+}"
                    elif market_type == "totals" and point is not None:
                        bet = f"{pick} {point}"
                    else:
                        bet = pick
                    rows.append({
                        "Date/Time": dt,
                        "Home": home,
                        "Away": away,
                        "Sportsbook": book,
                        "Market": market_type,
                        "Pick": pick,
                        "Line": bet,
                        "Odds": outcome.get("price"),
                        "Point": point
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
# AI Genius Confidence Function
# ----------------------------
def attach_ai_confidence(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    # Simple heuristic: favorites = higher confidence
    df["Confidence"] = df["Odds"].apply(lambda x: 70 if x and x < 0 else 55)
    df["Units"] = df["Confidence"].apply(lambda c: round((c - 50) / 10, 1))  # 0.5 to 2 units
    return df

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
        tabs = st.tabs(["Moneylines", "Spreads", "Totals", "Raw Data", "AI Genius Picks"])

        # --- Moneylines
        with tabs[0]:
            ml = df[df["Market"] == "h2h"]
            st.subheader("Moneyline Picks")
            if ml.empty:
                st.info("No moneyline data available.")
            else:
                st.dataframe(ml[["Date/Time","Home","Away","Sportsbook","Line","Odds"]],
                             use_container_width=True)

        # --- Spreads
        with tabs[1]:
            spreads = df[df["Market"] == "spreads"]
            st.subheader("Spread Picks")
            if spreads.empty:
                st.info("No spread data available.")
            else:
                st.dataframe(spreads[["Date/Time","Home","Away","Sportsbook","Line","Odds"]],
                             use_container_width=True)

        # --- Totals
        with tabs[2]:
            totals = df[df["Market"] == "totals"]
            st.subheader("Totals Picks")
            if totals.empty:
                st.info("No totals data available.")
            else:
                st.dataframe(totals[["Date/Time","Home","Away","Sportsbook","Line","Odds"]],
                             use_container_width=True)

        # --- Raw JSON Flattened
        with tabs[3]:
            st.subheader("Raw Odds Data")
            st.dataframe(df.head(50), use_container_width=True)

        # --- AI Genius Picks
        with tabs[4]:
            st.subheader("🤖 AI Genius Top 5 Picks")
            ai_df = attach_ai_confidence(df)
            if ai_df.empty:
                st.info("No data available for AI Genius Picks.")
            else:
                best = ai_df.sort_values("Confidence", ascending=False).head(5)
                st.dataframe(best[["Date/Time","Home","Away","Sportsbook","Market","Line","Odds","Confidence","Units"]],
                             use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
