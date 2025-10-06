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
st.caption("Live odds. Top 5 picks for each market. Unit size based on confidence.")
st.write("---")

# ----------------------------
# Helper functions
# ----------------------------
def implied_prob_american(odds: float) -> float:
    """Convert American odds → implied probability."""
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def assign_units(prob: float) -> float:
    """Confidence → unit size mapping."""
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

    return pd.json_normalize(r.json(), sep="_")

# ----------------------------
# Format Display
# ----------------------------
def format_display(df, market_key):
    """Extract and clean table for a specific market."""
    market_cols = [c for c in df.columns if "markets" in c and c.endswith("key")]
    if not market_cols:
        return pd.DataFrame()
    market_col = market_cols[0]

    sub = df[df[market_col] == market_key].copy()
    if sub.empty:
        return sub

    # Identify pick & odds columns
    pick_col = next((c for c in sub.columns if c.endswith("outcomes_0_name")), None)
    price_col = next((c for c in sub.columns if c.endswith("outcomes_0_price")), None)

    if price_col:
        sub["prob"] = sub[price_col].apply(implied_prob_american)
        sub["Units"] = sub["prob"].apply(assign_units)

    # Build clean table
    cols = {
        "commence_time": "Date/Time",
        "home_team": "Home",
        "away_team": "Away",
        "bookmakers_0_title": "Sportsbook",
    }
    if pick_col: cols[pick_col] = "Pick"
    if price_col: cols[price_col] = "Odds (US)"
    if "Units" in sub.columns: cols["Units"] = "Units"

    available = [c for c in cols if c in sub.columns]
    out = sub[available].rename(columns=cols)

    # Clean up datetime
    if "Date/Time" in out.columns:
        out["Date/Time"] = pd.to_datetime(out["Date/Time"], errors="coerce")
        out["Date/Time"] = out["Date/Time"].dt.strftime("%b %d, %I:%M %p ET")

    # Top 5 picks
    if "Units" in out.columns:
        out = out.sort_values("Units", ascending=False).head(5)

    return out.reset_index(drop=True)

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
        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "Raw Data"])

        with tabs[0]:
            st.subheader("Top 5 Moneyline Picks")
            ml = format_display(df, "h2h")
            if ml.empty:
                st.info("No moneyline data available.")
            else:
                st.dataframe(ml, use_container_width=True, hide_index=True)

        with tabs[1]:
            st.subheader("Top 5 Totals Picks (Over/Under)")
            totals = format_display(df, "totals")
            if totals.empty:
                st.info("No totals data available.")
            else:
                st.dataframe(totals, use_container_width=True, hide_index=True)

        with tabs[2]:
            st.subheader("Top 5 Spread Picks")
            spreads = format_display(df, "spreads")
            if spreads.empty:
                st.info("No spreads data available.")
            else:
                st.dataframe(spreads, use_container_width=True, hide_index=True)

        with tabs[3]:
            st.subheader("Raw Odds Data")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
