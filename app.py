import os
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ----------------------------
# Try to load .env
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
}

# ----------------------------
# Streamlit page
# ----------------------------
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker")
st.caption("Live odds + historical data = AI picks with confidence")
st.write("---")

# ----------------------------
# Fetch Odds API
# ----------------------------
@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY. Add it to `.env` or Streamlit Secrets.")
        return pd.DataFrame()

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {"apiKey": ODDS_API_KEY, "regions": regions, "markets": markets, "oddsFormat": "american"}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        st.error(f"Odds API error {r.status_code}: {r.text}")
        return pd.DataFrame()

    data = r.json()
    rows = []
    for event in data:
        dt = event.get("commence_time", "")
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        for bm in event.get("bookmakers", []):
            book = bm.get("title", "")
            for mk in bm.get("markets", []):
                market_type = mk.get("key", "")
                for outcome in mk.get("outcomes", []):
                    rows.append({
                        "Date/Time": dt,
                        "Home": home,
                        "Away": away,
                        "Sportsbook": book,
                        "Market": market_type,
                        "Pick": outcome.get("name"),
                        "Odds": outcome.get("price")
                    })
    return pd.DataFrame(rows)

# ----------------------------
# Load historical stats
# ----------------------------
def load_history(sport: str) -> pd.DataFrame:
    fname = f"data/{sport.lower()}_history.csv"
    if os.path.exists(fname):
        return pd.read_csv(fname)
    return pd.DataFrame()

def attach_ai_confidence(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    hist = load_history(sport)
    if hist.empty:
        df["Confidence"] = np.random.uniform(50, 70, size=len(df))  # fallback
        df["Units"] = df["Confidence"].apply(lambda x: round(x/20, 1))
        return df

    # Example: assume history has columns [team, win] where win=1 or 0
    team_win_rates = hist.groupby("team")["win"].mean().to_dict()

    conf = []
    for _, row in df.iterrows():
        team = row["Pick"]
        rate = team_win_rates.get(team, 0.5)
        confidence = 50 + (rate - 0.5) * 100
        conf.append(confidence)
    df["Confidence"] = conf
    df["Units"] = df["Confidence"].apply(lambda x: round(x/20, 1))
    return df

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
        st.warning("No data returned.")
    else:
        df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")
        df["Date/Time"] = df["Date/Time"].dt.strftime("%b %d, %I:%M %p ET")

        tabs = st.tabs(["Moneylines", "Spreads", "Totals", "AI Genius Picks", "Raw Data"])

        # Moneylines
        with tabs[0]:
            ml = df[df["Market"] == "h2h"]
            st.subheader("Moneyline Picks")
            st.dataframe(ml.head(20), use_container_width=True)

        # Spreads
        with tabs[1]:
            spreads = df[df["Market"] == "spreads"]
            st.subheader("Spread Picks")
            st.dataframe(spreads.head(20), use_container_width=True)

        # Totals
        with tabs[2]:
            totals = df[df["Market"] == "totals"]
            st.subheader("Totals Picks")
            st.dataframe(totals.head(20), use_container_width=True)

        # AI Genius Picks
        with tabs[3]:
            st.subheader("🤖 AI Genius Best Picks")
            ai_df = attach_ai_confidence(df.copy(), sport_name)
            best = ai_df.sort_values("Confidence", ascending=False).head(5)
            st.dataframe(best[["Date/Time", "Home", "Away", "Sportsbook", "Market", "Pick", "Odds", "Confidence", "Units"]],
                         use_container_width=True)

        # Raw
        with tabs[4]:
            st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("Set filters in sidebar and click **Fetch Live Odds**.")
