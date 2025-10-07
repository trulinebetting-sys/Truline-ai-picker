import os
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ----------------------------
# Load environment
# ----------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", st.secrets.get("ODDS_API_KEY", ""))
DEFAULT_REGIONS = "us"

SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
}

# ----------------------------
# Streamlit setup
# ----------------------------
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("🏈 TruLine – AI Genius Picker")
st.caption("AI-driven picks using live odds + historical data")
st.write("---")

# ----------------------------
# Odds helpers
# ----------------------------
def american_to_decimal(odds: float) -> float:
    if odds is None or pd.isna(odds): return np.nan
    odds = float(odds)
    return 1 + (odds/100.0) if odds > 0 else 1 + (100.0/abs(odds))

def implied_prob(odds: float) -> float:
    if odds is None or pd.isna(odds): return np.nan
    odds = float(odds)
    return 100.0/(odds+100.0) if odds > 0 else abs(odds)/(abs(odds)+100.0)

# ----------------------------
# Fetch live odds
# ----------------------------
@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str = "us", markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY")
        return pd.DataFrame()
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {"apiKey": ODDS_API_KEY, "regions": regions, "markets": markets, "oddsFormat": "american"}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        st.error(f"Odds API error {r.status_code}: {r.text}")
        return pd.DataFrame()
    return pd.json_normalize(r.json(), sep="_")

# ----------------------------
# Load historical CSV
# ----------------------------
def load_historical(sport: str) -> pd.DataFrame:
    fname = f"data/{sport.lower()}.csv"
    if not os.path.exists(fname):
        return pd.DataFrame()
    return pd.read_csv(fname)

# ----------------------------
# AI Genius confidence
# ----------------------------
def attach_ai_confidence(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    hist = load_historical(sport)
    if df.empty or hist.empty:
        df["ai_confidence"] = np.nan
        df["units"] = np.nan
        return df

    team_win_rates = hist.groupby("winner").size().div(len(hist)).to_dict()

    def conf(row):
        teams = [row.get("home_team",""), row.get("away_team","")]
        vals = [team_win_rates.get(t, 0.5) for t in teams]
        return np.mean(vals)

    df["ai_confidence"] = df.apply(conf, axis=1)
    df["units"] = (df["ai_confidence"]*5).round(1)  # scale 0–5 units
    return df

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()), index=0)
    fetch = st.button("Fetch AI Picks")

# ----------------------------
# Main
# ----------------------------
sport_key = SPORT_OPTIONS[sport_name]

if fetch:
    df = fetch_odds(sport_key)
    if df.empty:
        st.warning("No data returned.")
    else:
        # Merge AI confidence
        df = attach_ai_confidence(df, sport_name)

        # Tabs
        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "AI Genius Picks", "Raw Data"])

        # Moneylines
        with tabs[0]:
            ml = df[df.get("bookmakers_0_markets_0_key")=="h2h"]
            st.subheader("Moneyline Picks")
            st.dataframe(ml.head(5), use_container_width=True)

        # Totals
        with tabs[1]:
            totals = df[df.get("bookmakers_0_markets_0_key")=="totals"]
            st.subheader("Totals Picks")
            st.dataframe(totals.head(5), use_container_width=True)

        # Spreads
        with tabs[2]:
            spreads = df[df.get("bookmakers_0_markets_0_key")=="spreads"]
            st.subheader("Spread Picks")
            st.dataframe(spreads.head(5), use_container_width=True)

        # AI Genius Picks
        with tabs[3]:
            st.subheader("🤖 Top 5 AI Genius Picks")
            picks = df.sort_values("ai_confidence", ascending=False).head(5)
            show = picks[[
                "commence_time","home_team","away_team","bookmakers_0_title",
                "bookmakers_0_markets_0_outcomes_0_name","bookmakers_0_markets_0_outcomes_0_price",
                "ai_confidence","units"
            ]]
            show.rename(columns={
                "commence_time":"Date/Time",
                "home_team":"Home",
                "away_team":"Away",
                "bookmakers_0_title":"Book",
                "bookmakers_0_markets_0_outcomes_0_name":"Pick",
                "bookmakers_0_markets_0_outcomes_0_price":"Odds",
                "ai_confidence":"AI Confidence",
                "units":"Units"
            }, inplace=True)
            st.dataframe(show, use_container_width=True)

        # Raw
        with tabs[4]:
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Click Fetch to get live odds + AI Genius picks.")
