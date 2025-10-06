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
st.caption("AI-driven best bets with Kelly staking. Only Top 5 picks per market.")
st.write("---")

# ----------------------------
# Helper functions
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

def kelly_fraction(true_p: float, dec_odds: float) -> float:
    if true_p is None or pd.isna(true_p) or dec_odds is None or pd.isna(dec_odds):
        return 0.0
    b = dec_odds - 1.0
    q = 1.0 - true_p
    f = (b * true_p - q) / b if b > 0 else 0.0
    return max(0.0, f)

def edge_from_true_p(dec_odds: float, true_p: float) -> float:
    if dec_odds is None or pd.isna(dec_odds) or true_p is None or pd.isna(true_p):
        return np.nan
    return dec_odds * true_p - 1.0

def clean_and_rank(df: pd.DataFrame, market_type: str, bankroll: float, unit_size: float, kelly_cap: float):
    if df.empty:
        return pd.DataFrame()

    df = df[df["market_key"] == market_type].copy()
    if df.empty:
        return pd.DataFrame()

    # Convert odds to metrics
    df["dec_odds"] = df["price"].apply(american_to_decimal)
    df["implied_prob"] = df["price"].apply(implied_prob_american)
    df["true_prob"] = df.groupby("id")["implied_prob"].transform(lambda x: x / x.sum())  # remove vig
    df["edge"] = df.apply(lambda r: edge_from_true_p(r["dec_odds"], r["true_prob"]), axis=1)
    df["kelly"] = df.apply(lambda r: kelly_fraction(r["true_prob"], r["dec_odds"]), axis=1)
    df["kelly_capped"] = df["kelly"].clip(lower=0.0, upper=kelly_cap)
    df["stake_$"] = (df["kelly_capped"] * bankroll).clip(lower=0.0)
    df["units"] = df["stake_$"] / unit_size

    # Drop duplicates (keep best odds per outcome across books)
    df = df.sort_values(by="edge", ascending=False)
    df = df.drop_duplicates(subset=["id", "name"], keep="first")

    # Format display columns
    df["Edge %"] = (df["edge"] * 100).round(2).astype(str) + "%"
    df["Implied %"] = (df["implied_prob"] * 100).round(1).astype(str) + "%"
    df["True %"] = (df["true_prob"] * 100).round(1).astype(str) + "%"
    df["Units"] = df["units"].round(2)
    df["Stake ($)"] = df["stake_$"].round(2)
    df["Odds"] = df["price"].astype(str) + f" / " + df["dec_odds"].round(2).astype(str)

    # Final columns
    df = df.rename(columns={
        "commence_time": "Date/Time",
        "home_team": "Home",
        "away_team": "Away",
        "name": "Bet",
        "book": "Book"
    })

    return df[["Date/Time", "Home", "Away", "Bet", "Odds", "Book", "Implied %", "True %", "Edge %", "Stake ($)", "Units"]].head(5)

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

    rows = []
    data = r.json()
    for ev in data:
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for o in mk.get("outcomes", []):
                    rows.append({
                        "id": ev["id"],
                        "commence_time": ev["commence_time"],
                        "home_team": ev["home_team"],
                        "away_team": ev["away_team"],
                        "book": bk["title"],
                        "market_key": mk["key"],  # h2h, totals, spreads
                        "name": o["name"],
                        "price": o["price"]
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
        df["commence_time"] = pd.to_datetime(df["commence_time"])
        df["commence_time"] = df["commence_time"].dt.strftime("%b %d, %I:%M %p ET")

        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "Raw Data"])

        # --- Moneylines
        with tabs[0]:
            st.subheader("Top 5 Moneyline Picks")
            ml = clean_and_rank(df, "h2h", bankroll, unit_size, kelly_cap)
            if ml.empty:
                st.info("No moneyline data available.")
            else:
                st.dataframe(ml, use_container_width=True)

        # --- Totals
        with tabs[1]:
            st.subheader("Top 5 Totals Picks")
            totals = clean_and_rank(df, "totals", bankroll, unit_size, kelly_cap)
            if totals.empty:
                st.info("No totals data available.")
            else:
                st.dataframe(totals, use_container_width=True)

        # --- Spreads
        with tabs[2]:
            st.subheader("Top 5 Spread Picks")
            spreads = clean_and_rank(df, "spreads", bankroll, unit_size, kelly_cap)
            if spreads.empty:
                st.info("No spreads data available.")
            else:
                st.dataframe(spreads, use_container_width=True)

        # --- Raw Data
        with tabs[3]:
            st.subheader("Raw Odds Data (debugging)")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
