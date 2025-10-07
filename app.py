import os
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ----------------------------
# Setup
# ----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("⚠️ dotenv not installed. Using Streamlit secrets instead.")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", st.secrets.get("ODDS_API_KEY", ""))
DEFAULT_REGIONS = os.getenv("REGIONS", "us")
DEFAULT_BOOKS = [b.strip() for b in os.getenv(
    "BOOKS", "DraftKings,FanDuel,BetMGM,PointsBet,Caesars,Pinnacle"
).split(",") if b.strip()]

SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb"
}

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("🤖 TruLine – AI Genius Picker")
st.caption("Combining live odds with historical stats for smarter picks.")
st.write("---")

# ----------------------------
# Helper Functions
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

@st.cache_data
def load_historical(sport: str) -> pd.DataFrame:
    path_map = {
        "NFL": "data/nfl_results.csv",
        "NBA": "data/nba_results.csv",
        "MLB": "data/mlb_results.csv"
    }
    path = path_map.get(sport)
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])

def compute_ai_confidence(historical_df, home, away):
    """Very simple AI model: win % from past games"""
    if historical_df.empty:
        return 0.5
    home_games = historical_df[(historical_df["home_team"] == home) | (historical_df["away_team"] == home)]
    home_wins = ((home_games["home_team"] == home) & (home_games["home_score"] > home_games["away_score"])) | \
                ((home_games["away_team"] == home) & (home_games["away_score"] > home_games["home_score"]))
    home_win_pct = home_wins.mean() if not home_games.empty else 0.5

    away_games = historical_df[(historical_df["home_team"] == away) | (historical_df["away_team"] == away)]
    away_wins = ((away_games["home_team"] == away) & (away_games["home_score"] > away_games["away_score"])) | \
                ((away_games["away_team"] == away) & (away_games["away_score"] > away_games["home_score"]))
    away_win_pct = away_wins.mean() if not away_games.empty else 0.5

    confidence = home_win_pct / (home_win_pct + away_win_pct)
    return confidence

@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h") -> pd.DataFrame:
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
# Sidebar Filters
# ----------------------------
with st.sidebar:
    st.header("Filters")
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()), index=0)
    regions = st.text_input("Regions", value=DEFAULT_REGIONS)
    fetch = st.button("Fetch AI Picks")

# ----------------------------
# Main Content
# ----------------------------
sport_key = SPORT_OPTIONS[sport_name]

if fetch:
    df = fetch_odds(sport_key, regions, "h2h")
    if df.empty:
        st.warning("No data returned. Try another sport or check API quota.")
    else:
        hist_df = load_historical(sport_name)
        rows = []
        for _, row in df.iterrows():
            home = row.get("home_team", "Unknown")
            away = row.get("away_team", "Unknown")
            commence = row.get("commence_time", "")

            book = row.get("bookmakers_0_title", "")
            try:
                price = row["bookmakers_0_markets_0_outcomes_0_price"]
                team = row["bookmakers_0_markets_0_outcomes_0_name"]
            except KeyError:
                continue

            imp_prob = implied_prob_american(price)
            ai_conf = compute_ai_confidence(hist_df, home, away)
            final_conf = round((0.6 * imp_prob + 0.4 * ai_conf) * 100, 1)

            # Assign units (1–5 based on confidence)
            units = 1 if final_conf < 55 else 2 if final_conf < 65 else 3 if final_conf < 75 else 4 if final_conf < 85 else 5

            rows.append({
                "Date/Time": commence,
                "Home": home,
                "Away": away,
                "Sportsbook": book,
                "Pick": f"{team} ML",
                "Odds": price,
                "Implied %": round(imp_prob*100,1),
                "AI Confidence %": final_conf,
                "Units": units
            })

        out = pd.DataFrame(rows)
        out = out.sort_values(by="AI Confidence %", ascending=False).head(5)
        st.subheader("Top 5 AI Picks (Moneyline)")
        st.dataframe(out, use_container_width=True)

else:
    st.info("Choose filters and click **Fetch AI Picks**.")
