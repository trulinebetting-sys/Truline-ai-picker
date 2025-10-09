import os
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ----------------------------
# Load keys safely
# ----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("⚠️ dotenv not installed. Using Streamlit secrets instead.")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", st.secrets.get("ODDS_API_KEY", ""))
APISPORTS_KEY = os.getenv("APISPORTS_KEY", st.secrets.get("APISPORTS_KEY", ""))

# ----------------------------
# Sports map
# ----------------------------
SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "College Football": "americanfootball_ncaaf",
    "NBA": "basketball_nba",
    "College Basketball": "basketball_ncaab",
    "MLB": "baseball_mlb",
    "Soccer (Top Leagues)": [
        "soccer_epl", "soccer_spain_la_liga",
        "soccer_italy_serie_a", "soccer_france_ligue_one",
        "soccer_germany_bundesliga", "soccer_uefa_champs_league"
    ]
}

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker")
st.caption("Live + historical stats to make the smartest picks.")
st.write("---")

# ----------------------------
# Helper functions
# ----------------------------
def clean_datetime(x):
    try:
        return pd.to_datetime(x).strftime("%b %d, %I:%M %p")
    except:
        return x

def safe_filter(df, column, value):
    if column not in df.columns:
        return pd.DataFrame()
    return df[df[column] == value]

# ----------------------------
# API Calls
# ----------------------------
@st.cache_data(ttl=60)
def fetch_live_odds(sport_key, regions="us", markets="h2h,spreads,totals"):
    if not ODDS_API_KEY:
        return pd.DataFrame()

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {"apiKey": ODDS_API_KEY, "regions": regions, "markets": markets, "oddsFormat": "american"}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.json_normalize(r.json(), sep="_")

@st.cache_data(ttl=3600)
def fetch_historical_data(sport="nfl"):
    if not APISPORTS_KEY:
        return pd.DataFrame()

    headers = {"x-apisports-key": APISPORTS_KEY}

    if sport == "nfl":
        url = "https://v1.american-football.api-sports.io/games?league=1&season=2023"
    elif sport == "nba":
        url = "https://v1.basketball.api-sports.io/games?league=12&season=2023"
    else:
        return pd.DataFrame()

    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json().get("response", [])
    rows = []
    for g in data[:50]:
        home = g.get("teams", {}).get("home", {}).get("name", "Unknown")
        away = g.get("teams", {}).get("away", {}).get("name", "Unknown")
        winner = g.get("scores", {}).get("winner", {}).get("name", None)
        rows.append({
            "Date": g.get("date"),
            "Home": home,
            "Away": away,
            "Winner": winner
        })
    return pd.DataFrame(rows)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    sport_name = st.selectbox("Choose Sport", list(SPORT_OPTIONS.keys()), index=0)
    fetch = st.button("Fetch Live Odds")

# ----------------------------
# Main logic
# ----------------------------
if fetch:
    # Soccer multi-league
    if sport_name == "Soccer (Top Leagues)":
        dfs = []
        for league in SPORT_OPTIONS[sport_name]:
            part = fetch_live_odds(league)
            if not part.empty:
                part["league"] = league
                dfs.append(part)
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        df = fetch_live_odds(SPORT_OPTIONS[sport_name])

    if df.empty:
        st.warning("No live odds returned. Check API quota or sport.")
    else:
        if "commence_time" in df.columns:
            df["Date/Time"] = df["commence_time"].apply(clean_datetime)

        # Tabs
        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "AI Genius Picks", "Raw Data"])

        # --- Moneyline
        with tabs[0]:
            st.subheader("Moneyline Picks")
            ml = safe_filter(df, "bookmakers_0_markets_0_key", "h2h")
            if ml.empty:
                st.info("No moneyline data available.")
            else:
                st.dataframe(ml.head(10), use_container_width=True)

        # --- Totals
        with tabs[1]:
            st.subheader("Over/Under Picks")
            totals = safe_filter(df, "bookmakers_0_markets_0_key", "totals")
            if totals.empty:
                st.info("No totals available.")
            else:
                st.dataframe(totals.head(10), use_container_width=True)

        # --- Spreads
        with tabs[2]:
            st.subheader("Spread Picks")
            spreads = safe_filter(df, "bookmakers_0_markets_0_key", "spreads")
            if spreads.empty:
                st.info("No spreads available.")
            else:
                st.dataframe(spreads.head(10), use_container_width=True)

        # --- AI Genius Picks
        with tabs[3]:
            st.subheader("🤖 AI Genius Picks (Live + Historical)")
            hist = fetch_historical_data("nfl" if sport_name == "NFL" else "nba")
            if hist.empty:
                st.warning("No historical data pulled yet.")
            else:
                win_rates = hist["Winner"].value_counts(normalize=True).to_dict()
                picks = []
                for team, rate in win_rates.items():
                    picks.append({"Team": team, "Confidence": f"{rate*100:.1f}%", "Units": round(rate*5, 1)})
                ai_df = pd.DataFrame(picks).head(5)
                st.dataframe(ai_df, use_container_width=True)

        # --- Raw
        with tabs[4]:
            st.subheader("Raw API Data")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Pick a sport and click **Fetch Live Odds**")
