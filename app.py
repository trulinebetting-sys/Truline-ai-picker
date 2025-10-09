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

# Soccer leagues combined
SOCCER_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_germany_bundesliga",
    "soccer_uefa_champions_league",
]

SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "College Football (NCAAF)": "americanfootball_ncaaf",
    "College Basketball (NCAAB)": "basketball_ncaab",
    "Soccer (All Major Leagues)": SOCCER_KEYS,
}

# ----------------------------
# Streamlit page
# ----------------------------
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker")
st.caption("Live odds with AI-style filtering. Units recommended based on confidence.")
st.write("---")

# ----------------------------
# Odds helper functions
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
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def assign_units(confidence: float) -> float:
    """Assign unit size 0.5 to 5.0 based on confidence %."""
    if pd.isna(confidence):
        return 0.5
    return round(0.5 + 4.5 * confidence, 1)  # Scale confidence [0–1] → [0.5–5 units]

# ----------------------------
# Fetch Odds API
# ----------------------------
@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    """Fetch odds for a given sport."""
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

    return pd.json_normalize(r.json(), sep="_")

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
    if isinstance(sport_key, list):
        # Soccer case: fetch multiple leagues
        df_list = [fetch_odds(sk, regions) for sk in sport_key]
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = fetch_odds(sport_key=sport_key, regions=regions)

    if df.empty:
        st.warning("No data returned. Try another sport or check API quota.")
    else:
        # Format datetime if exists
        if "commence_time" in df.columns:
            df["commence_time"] = pd.to_datetime(df["commence_time"])
            df["commence_time"] = df["commence_time"].dt.strftime("%b %d, %I:%M %p ET")

        # Tabs
        tabs = st.tabs(["AI Genius Picks", "Moneylines", "Totals", "Spreads", "Raw Data"])

        def safe_filter(df, key, value):
            """Helper to avoid KeyError if column missing"""
            if key not in df.columns:
                return pd.DataFrame()
            return df[df[key] == value]

        # --- AI Genius Picks ---
        with tabs[0]:
            st.subheader("🤖 AI Genius Top Picks")

            # Combine all markets (moneyline, spreads, totals)
            picks = []
            for market in ["h2h", "totals", "spreads"]:
                sub = safe_filter(df, "bookmakers_0_markets_0_key", market)
                if not sub.empty:
                    sub["decimal_odds"] = sub["bookmakers_0_markets_0_outcomes_0_price"].apply(american_to_decimal)
                    sub["confidence"] = sub["bookmakers_0_markets_0_outcomes_0_price"].apply(implied_prob_american)
                    sub["Units"] = sub["confidence"].apply(assign_units)
                    sub["Bet Type"] = market
                    picks.append(sub)

            if picks:
                all_picks = pd.concat(picks, ignore_index=True)

                # Deduplicate by matchup + bet type
                if "home_team" in all_picks.columns and "away_team" in all_picks.columns:
                    all_picks["Matchup"] = all_picks["home_team"] + " vs " + all_picks["away_team"]

                all_picks = all_picks.drop_duplicates(subset=["Matchup", "Bet Type"])

                # Sort by confidence and show top 10
                best = all_picks.sort_values(by="confidence", ascending=False).head(10)

                display_cols = [
                    "commence_time",
                    "Matchup",
                    "bookmakers_0_title",
                    "Bet Type",
                    "bookmakers_0_markets_0_outcomes_0_name",
                    "bookmakers_0_markets_0_outcomes_0_price",
                    "confidence",
                    "Units"
                ]
                best = best[display_cols].rename(columns={
                    "commence_time": "Date/Time",
                    "bookmakers_0_title": "Sportsbook",
                    "bookmakers_0_markets_0_outcomes_0_name": "Pick",
                    "bookmakers_0_markets_0_outcomes_0_price": "Odds (US)",
                    "confidence": "Confidence"
                })
                best["Confidence"] = best["Confidence"].apply(lambda x: f"{100*x:.1f}%" if not pd.isna(x) else "")

                st.dataframe(best, use_container_width=True, hide_index=True)
            else:
                st.info("No AI Genius picks available.")

        # --- Moneylines
        with tabs[1]:
            st.subheader("Moneyline Picks")
            ml = safe_filter(df, "bookmakers_0_markets_0_key", "h2h")
            if ml.empty:
                st.info("No moneyline data available.")
            else:
                st.dataframe(ml.head(10), use_container_width=True)

        # --- Totals
        with tabs[2]:
            st.subheader("Over/Under Picks")
            totals = safe_filter(df, "bookmakers_0_markets_0_key", "totals")
            if totals.empty:
                st.info("No totals data available.")
            else:
                st.dataframe(totals.head(10), use_container_width=True)

        # --- Spreads
        with tabs[3]:
            st.subheader("Spread Picks")
            spreads = safe_filter(df, "bookmakers_0_markets_0_key", "spreads")
            if spreads.empty:
                st.info("No spreads data available.")
            else:
                st.dataframe(spreads.head(10), use_container_width=True)

        # --- Raw JSON Flattened
        with tabs[4]:
            st.subheader("Raw Odds Data")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
