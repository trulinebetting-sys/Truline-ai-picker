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

def kelly_fraction(true_p: float, dec_odds: float, cap: float) -> float:
    if pd.isna(true_p) or pd.isna(dec_odds):
        return 0.0
    b = dec_odds - 1
    q = 1 - true_p
    f = (b * true_p - q) / b if b > 0 else 0
    return max(0, min(f, cap))

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
    for ev in r.json():
        home, away = ev.get("home_team"), ev.get("away_team")
        for bk in ev.get("bookmakers", []):
            book = bk.get("title")
            for mk in bk.get("markets", []):
                market = mk.get("key")  # h2h, spreads, totals
                for o in mk.get("outcomes", []):
                    odds = o.get("price")
                    line = o.get("point")
                    dec = american_to_decimal(odds)
                    true_prob = implied_prob_american(odds)
                    edge = dec * true_prob - 1
                    kelly = kelly_fraction(true_prob, dec, DEFAULT_KELLY_CAP)
                    rows.append({
                        "Matchup": f"{away} @ {home}",
                        "Market": market,
                        "Book": book,
                        "Bet": f"{o['name']} {line if line is not None else ''}".strip(),
                        "Odds (US)": odds,
                        "Odds (Dec)": f"{dec:.2f}",
                        "Implied %": f"{true_prob*100:.1f}%",
                        "Edge %": edge,
                        "Kelly %": f"{kelly*100:.1f}%",
                        "Units": f"{kelly*40:.2f}"
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
        # Sort by edge
        df = df.sort_values("Edge %", ascending=False)

        # Banner for best pick
        best = df.iloc[0]
        st.success(
            f"🔥 **Best Pick Today:** {best['Bet']} in {best['Matchup']} "
            f"@ {best['Book']} | Odds: {best['Odds (US)']} "
            f"| Edge: {best['Edge %']*100:.2f}% | Units: {best['Units']}"
        )

        # Tabs
        tabs = st.tabs(["Moneylines", "Totals", "Spreads", "Raw Data"])

        def top_table(df, market):
            subset = df[df["Market"] == market].sort_values("Edge %", ascending=False)
            subset = subset[subset["Edge %"] > min_edge]
            return subset.head(5)[[
                "Matchup", "Book", "Bet", "Odds (US)", "Odds (Dec)",
                "Implied %", "Edge %", "Kelly %", "Units"
            ]]

        # --- Moneylines
        with tabs[0]:
            st.subheader("Top 5 Moneyline Picks")
            ml = top_table(df, "h2h")
            if ml.empty:
                st.info("No edges found.")
            else:
                st.dataframe(ml, use_container_width=True)

        # --- Totals
        with tabs[1]:
            st.subheader("Top 5 Totals Picks")
            tot = top_table(df, "totals")
            if tot.empty:
                st.info("No totals found.")
            else:
                st.dataframe(tot, use_container_width=True)

        # --- Spreads
        with tabs[2]:
            st.subheader("Top 5 Spread Picks")
            sp = top_table(df, "spreads")
            if sp.empty:
                st.info("No spreads found.")
            else:
                st.dataframe(sp, use_container_width=True)

        # --- Raw Data
        with tabs[3]:
            st.subheader("Raw Odds Data")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
