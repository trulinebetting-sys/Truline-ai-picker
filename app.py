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
st.caption("Best AI-backed plays, simplified. One pick per game, top 5 shown.")
st.write("---")

# ----------------------------
# Odds helper functions
# ----------------------------
def implied_prob_from_american(odds):
    """Convert American odds to implied probability."""
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

# ----------------------------
# Fetch Odds API
# ----------------------------
@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h,spreads,totals"):
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY. Add it to `.env` or Streamlit Secrets.")
        return []

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
        return []

    return r.json()

# ----------------------------
# Build clean table
# ----------------------------
def build_table(data: list, market_type: str) -> pd.DataFrame:
    rows = []
    for ev in data:
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        for bk in ev.get("bookmakers", []):
            book = bk.get("title")
            for mk in bk.get("markets", []):
                if mk.get("key") != market_type:
                    continue
                for o in mk.get("outcomes", []):
                    odds = o.get("price")
                    rows.append({
                        "Date/Time": commence,
                        "Home": home,
                        "Away": away,
                        "Sportsbook": book,
                        "Market": market_type,
                        "Pick": o.get("name"),
                        "Line": o.get("point"),
                        "Odds": odds,
                        "ImpliedProb": implied_prob_from_american(odds)
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Format datetime
    if "Date/Time" in df.columns:
        df["Date/Time"] = pd.to_datetime(df["Date/Time"])
        df["Date/Time"] = df["Date/Time"].dt.strftime("%b %d, %I:%M %p ET")

    # Deduplicate → keep best pick per game + book
    df = df.sort_values("ImpliedProb", ascending=False)
    df = df.groupby(["Home", "Away", "Sportsbook", "Market"], as_index=False).first()

    # Only show top 5
    df = df.sort_values("ImpliedProb", ascending=False).head(5)

    return df[["Date/Time", "Home", "Away", "Sportsbook", "Pick", "Line", "Odds", "ImpliedProb"]]

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
    data = fetch_odds(sport_key=sport_key, regions=regions)
    if not data:
        st.warning("No data returned. Try another sport or check API quota.")
    else:
        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "Raw Data"])

        with tabs[0]:
            st.subheader("Top 5 Moneyline Picks")
            ml = build_table(data, "h2h")
            st.dataframe(ml, use_container_width=True) if not ml.empty else st.info("No moneyline data.")

        with tabs[1]:
            st.subheader("Top 5 Totals (Over/Under) Picks")
            totals = build_table(data, "totals")
            st.dataframe(totals, use_container_width=True) if not totals.empty else st.info("No totals data.")

        with tabs[2]:
            st.subheader("Top 5 Spread Picks")
            spreads = build_table(data, "spreads")
            st.dataframe(spreads, use_container_width=True) if not spreads.empty else st.info("No spread data.")

        with tabs[3]:
            st.subheader("Raw Odds Data")
            st.json(data[:1])  # show just 1 raw event for inspection
else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
