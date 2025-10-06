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
st.caption("Live odds. Clean charts. Top 5 best picks per market.")
st.write("---")

# ----------------------------
# Helpers
# ----------------------------
def implied_prob_american(odds: float) -> float:
    """Convert US odds to implied probability"""
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)

def assign_units(prob: float) -> float:
    """Assign units based on confidence"""
    if pd.isna(prob):
        return 0.5
    if prob > 0.65:
        return 3.0
    elif prob > 0.55:
        return 2.0
    else:
        return 1.0

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

    return r.json()

# ----------------------------
# Format clean picks
# ----------------------------
def extract_picks(data, market_type):
    rows = []
    for ev in data:
        date = ev.get("commence_time")
        home = ev.get("home_team", "Unknown")
        away = ev.get("away_team", "Unknown")
        for bk in ev.get("bookmakers", []):
            book = bk.get("title")
            for mk in bk.get("markets", []):
                if mk.get("key") != market_type:
                    continue
                for o in mk.get("outcomes", []):
                    name = o.get("name")
                    price = o.get("price")
                    prob = implied_prob_american(price)
                    units = assign_units(prob)
                    point = o.get("point")

                    # Label
                    if market_type == "h2h":
                        pick = name
                    elif market_type == "totals":
                        pick = f"{name} {point}"
                    elif market_type == "spreads":
                        pick = f"{name} {point:+}"
                    else:
                        pick = name

                    rows.append({
                        "Date/Time": pd.to_datetime(date).strftime("%b %d, %I:%M %p ET"),
                        "Home": home,
                        "Away": away,
                        "Sportsbook": book,
                        "Pick": pick,
                        "Odds (US)": price,
                        "Units": units
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values("Units", ascending=False).head(5).reset_index(drop=True)

# ----------------------------
# Sidebar
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
        tabs = st.tabs(["Moneylines", "Totals", "Spreads"])

        with tabs[0]:
            st.subheader("Top 5 Moneyline Picks")
            ml = extract_picks(data, "h2h")
            st.dataframe(ml, use_container_width=True, hide_index=True) if not ml.empty else st.info("No moneyline data.")

        with tabs[1]:
            st.subheader("Top 5 Totals Picks")
            totals = extract_picks(data, "totals")
            st.dataframe(totals, use_container_width=True, hide_index=True) if not totals.empty else st.info("No totals data.")

        with tabs[2]:
            st.subheader("Top 5 Spread Picks")
            spreads = extract_picks(data, "spreads")
            st.dataframe(spreads, use_container_width=True, hide_index=True) if not spreads.empty else st.info("No spreads data.")

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
