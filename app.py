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
# Fetch Odds API (flattened)
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

    data = r.json()
    rows = []
    for ev in data:
        event_id = ev.get("id")
        home = ev.get("home_team")
        away = ev.get("away_team")
        start = ev.get("commence_time")

        for book in ev.get("bookmakers", []):
            book_name = book.get("title")
            for market in book.get("markets", []):
                market_key = market.get("key")
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "event_id": event_id,
                        "commence_time": start,
                        "home_team": home,
                        "away_team": away,
                        "book": book_name,
                        "market": market_key,         # h2h, totals, spreads
                        "name": outcome.get("name"),  # Home/Away/Over/Under
                        "price": outcome.get("price"),
                        "point": outcome.get("point") # spread or total line if applicable
                    })

    return pd.DataFrame(rows)

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
        # Format datetime
        if "commence_time" in df.columns:
            df["commence_time"] = pd.to_datetime(df["commence_time"])
            df["commence_time"] = df["commence_time"].dt.strftime("%b %d, %I:%M %p ET")

        # Tabs
        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "Raw Data"])

        # --- Moneylines
        with tabs[0]:
            st.subheader("Moneyline Picks")
            ml = df[df["market"] == "h2h"]
            if ml.empty:
                st.info("No moneyline data available.")
            else:
                st.dataframe(
                    ml[["commence_time","home_team","away_team","book","name","price"]],
                    use_container_width=True
                )

        # --- Totals
        with tabs[1]:
            st.subheader("Over/Under Picks")
            totals = df[df["market"] == "totals"]
            if totals.empty:
                st.info("No totals data available.")
            else:
                st.dataframe(
                    totals[["commence_time","home_team","away_team","book","name","point","price"]],
                    use_container_width=True
                )

        # --- Spreads
        with tabs[2]:
            st.subheader("Spread Picks")
            spreads = df[df["market"] == "spreads"]
            if spreads.empty:
                st.info("No spreads data available.")
            else:
                st.dataframe(
                    spreads[["commence_time","home_team","away_team","book","name","point","price"]],
                    use_container_width=True
                )

        # --- Raw JSON Flattened
        with tabs[3]:
            st.subheader("Raw Odds Data")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
