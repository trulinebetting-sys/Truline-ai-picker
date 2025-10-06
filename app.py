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
st.caption("Live odds. Best picks only. Confidence-weighted units.")
st.write("---")

# ----------------------------
# Odds helper functions
# ----------------------------
def american_to_decimal(odds: float) -> float:
    try:
        odds = float(odds)
    except:
        return np.nan
    return 1 + (odds / 100.0) if odds > 0 else 1 + (100.0 / abs(odds))

def implied_prob_american(odds: float) -> float:
    try:
        odds = float(odds)
    except:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

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
        start = pd.to_datetime(ev.get("commence_time"))
        home = ev.get("home_team", "Unknown")
        away = ev.get("away_team", "Unknown")

        for bk in ev.get("bookmakers", []):
            book = bk.get("title", "")
            for mk in bk.get("markets", []):
                market = mk.get("key")  # h2h / spreads / totals
                for out in mk.get("outcomes", []):
                    name = out.get("name", "")
                    price = out.get("price")
                    point = out.get("point")

                    dec = american_to_decimal(price)
                    imp = implied_prob_american(price)

                    # Simplified "true prob" = implied prob (no sharp ref book yet)
                    true_prob = imp
                    edge = dec * true_prob - 1

                    # Confidence-weighted stake
                    confidence = true_prob * (edge + 1)
                    stake = max(0, confidence * 1000 * DEFAULT_KELLY_CAP)  # assume bankroll=1000
                    units = stake / 25  # assume unit=25

                    pick_label = name
                    if point is not None:
                        pick_label += f" {point}"

                    rows.append({
                        "Date/Time": start.strftime("%b %d, %I:%M %p ET"),
                        "Home": home,
                        "Away": away,
                        "Book": book,
                        "Market": market,
                        "Pick": pick_label,
                        "Odds (US)": price,
                        "Odds (Dec)": round(dec, 2) if dec else "",
                        "Implied %": f"{imp*100:.1f}%" if imp else "",
                        "True %": f"{true_prob*100:.1f}%" if true_prob else "",
                        "Edge %": f"{edge*100:.2f}%" if not pd.isna(edge) else "",
                        "Stake ($)": f"${stake:.2f}",
                        "Units": f"{units:.2f}"
                    })

    return pd.DataFrame(rows)

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

        def best_picks(df, market):
            if df.empty:
                return pd.DataFrame()
            mdf = df[df["Market"] == market].copy()
            if mdf.empty:
                return pd.DataFrame()
            # Keep best pick per game by Edge %
            mdf = mdf.sort_values("Edge %", ascending=False)
            mdf = mdf.drop_duplicates(subset=["Home", "Away"], keep="first")
            return mdf.head(5).reset_index(drop=True)

        with tabs[0]:
            st.subheader("Best Moneyline Picks")
            ml = best_picks(df, "h2h")
            st.dataframe(ml if not ml.empty else pd.DataFrame([{"Info": "No moneyline picks"}]), use_container_width=True)

        with tabs[1]:
            st.subheader("Best Totals Picks")
            totals = best_picks(df, "totals")
            st.dataframe(totals if not totals.empty else pd.DataFrame([{"Info": "No totals picks"}]), use_container_width=True)

        with tabs[2]:
            st.subheader("Best Spread Picks")
            spreads = best_picks(df, "spreads")
            st.dataframe(spreads if not spreads.empty else pd.DataFrame([{"Info": "No spread picks"}]), use_container_width=True)

        with tabs[3]:
            st.subheader("Raw Odds Data")
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
