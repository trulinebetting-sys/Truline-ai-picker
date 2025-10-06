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
st.caption("Simple. Live odds. Best single pick per game per market.")
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

def kelly_fraction(true_p: float, dec_odds: float) -> float:
    if pd.isna(true_p) or pd.isna(dec_odds) or dec_odds <= 1.0:
        return 0.0
    b = dec_odds - 1
    q = 1 - true_p
    f = (b*true_p - q) / b
    return max(0.0, f)

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

    data = r.json()
    rows = []
    for ev in data:
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        for bk in ev.get("bookmakers", []):
            book = bk.get("title")
            for mk in bk.get("markets", []):
                mkey = mk.get("key")
                for out in mk.get("outcomes", []):
                    rows.append({
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": mkey,
                        "name": out.get("name"),
                        "price": out.get("price"),
                        "point": out.get("point")
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
    kelly_cap = st.slider("Kelly Cap", 0.0, 1.0, 0.25, 0.05)
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
            df["Date/Time"] = df["commence_time"].dt.strftime("%b %d, %I:%M %p ET")

        # Build Pick column
        df["Pick"] = df.apply(lambda r: f"{r['name']} {r['point']}" if pd.notna(r['point']) else r['name'], axis=1)

        # Probabilities + Kelly stake
        df["Decimal Odds"] = df["price"].apply(american_to_decimal)
        df["True Prob"] = df["price"].apply(implied_prob_american)
        df["Kelly"] = df.apply(lambda r: min(kelly_fraction(r["True Prob"], r["Decimal Odds"]), kelly_cap), axis=1)
        df["Stake ($)"] = (df["Kelly"] * bankroll).round(2)
        df["Units"] = (df["Stake ($)"] / unit_size).round(2)

        # Group to keep only best pick per game+market
        df["Matchup"] = df["home_team"] + " vs " + df["away_team"]
        df_best = df.loc[df.groupby(["Matchup", "market"])["Units"].idxmax()]

        # Tabs
        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "Raw Data"])

        def clean_table(sub):
            return sub[[
                "Date/Time", "home_team", "away_team", "book", "Pick", "price", "Stake ($)", "Units"
            ]].rename(columns={
                "home_team": "Home",
                "away_team": "Away",
                "book": "Sportsbook",
                "price": "Odds (US)"
            }).sort_values(by="Units", ascending=False)

        # --- Moneylines
        with tabs[0]:
            st.subheader("Best Moneyline Pick per Game")
            ml = df_best[df_best["market"] == "h2h"]
            if ml.empty:
                st.info("No moneyline data available.")
            else:
                st.dataframe(clean_table(ml), use_container_width=True)

        # --- Totals
        with tabs[1]:
            st.subheader("Best Total (Over/Under) Pick per Game")
            totals = df_best[df_best["market"] == "totals"]
            if totals.empty:
                st.info("No totals data available.")
            else:
                st.dataframe(clean_table(totals), use_container_width=True)

        # --- Spreads
        with tabs[2]:
            st.subheader("Best Spread Pick per Game")
            spreads = df_best[df_best["market"] == "spreads"]
            if spreads.empty:
                st.info("No spreads data available.")
            else:
                st.dataframe(clean_table(spreads), use_container_width=True)

        # --- Raw JSON Flattened
        with tabs[3]:
            st.subheader("Raw Odds Data")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
