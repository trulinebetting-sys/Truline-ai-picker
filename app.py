import os
import requests
import pandas as pd
import numpy as np
import streamlit as st

# Try dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------
# Config
# ----------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", st.secrets.get("ODDS_API_KEY", ""))
SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "EPL": "soccer_epl",
    "La Liga": "soccer_spain_la_liga"
}

st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker")

# ----------------------------
# Fetch Odds API (raw JSON)
# ----------------------------
@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h,spreads,totals") -> list:
    if not ODDS_API_KEY:
        return []
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {"apiKey": ODDS_API_KEY, "regions": regions, "markets": markets, "oddsFormat": "american"}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return []
    return r.json()

# ----------------------------
# Clean into table
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
                    rows.append({
                        "Date/Time": commence,
                        "Home": home,
                        "Away": away,
                        "Sportsbook": book,
                        "Market": market_type,
                        "Pick": o.get("name"),
                        "Line": o.get("point"),
                        "Odds": o.get("price")
                    })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if not df.empty and "Date/Time" in df.columns:
        df["Date/Time"] = pd.to_datetime(df["Date/Time"])
        df["Date/Time"] = df["Date/Time"].dt.strftime("%b %d, %I:%M %p ET")
    return df

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()), index=0)
    regions = st.text_input("Regions", value="us")
    fetch = st.button("Fetch Live Odds")

# ----------------------------
# Main
# ----------------------------
if fetch:
    raw = fetch_odds(SPORT_OPTIONS[sport_name], regions)
    if not raw:
        st.error("No data returned. Check API key or quota.")
    else:
        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "Raw JSON"])

        with tabs[0]:
            ml = build_table(raw, "h2h")
            st.subheader("Moneyline Picks")
            st.dataframe(ml.head(5) if not ml.empty else pd.DataFrame([{"Info": "No moneyline data"}]))

        with tabs[1]:
            totals = build_table(raw, "totals")
            st.subheader("Totals (Over/Under)")
            st.dataframe(totals.head(5) if not totals.empty else pd.DataFrame([{"Info": "No totals data"}]))

        with tabs[2]:
            spreads = build_table(raw, "spreads")
            st.subheader("Spreads (+/-)")
            st.dataframe(spreads.head(5) if not spreads.empty else pd.DataFrame([{"Info": "No spreads data"}]))

        with tabs[3]:
            st.subheader("Raw JSON Data")
            st.json(raw[:2])  # show first 2 events for debugging

else:
    st.info("Set filters and click **Fetch Live Odds**.")
