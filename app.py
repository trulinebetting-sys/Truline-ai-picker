import os
import requests
import streamlit as st
from dotenv import load_dotenv

# --- Load ENV ---
load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")
REGIONS = os.getenv("REGIONS", "us")
MARKETS = os.getenv("MARKETS", "h2h,spreads,totals")
BOOKS = os.getenv("BOOKS", "DraftKings,FanDuel,BetMGM")

# --- PAGE CONFIG ---
st.set_page_config(page_title="TruLine Betting", page_icon="assets/logo.png", layout="wide")

# --- HEADER ---
st.image("assets/logo.png", width=100)
st.title("TruLine Betting – AI Genius Picks")

# --- AI GENIUS PICKS ---
st.markdown("---")
st.subheader("🤖 AI Genius Live Picks")

def get_odds(sport="basketball_nba"):
    """Fetch odds from TheOddsAPI"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": "american",
        "bookmakers": BOOKS,
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return None, f"Error {response.status_code}: {response.text}"
    return response.json(), None

# --- Choose Sport ---
sport = st.selectbox(
    "Select a sport:",
    ["basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl"]
)

data, error = get_odds(sport)

if error:
    st.error(error)
else:
    if not data:
        st.warning("No games found right now. Try again later.")
    else:
        for game in data[:5]:  # Show top 5 games
            teams = game["home_team"] + " vs " + game["away_team"]
            st.markdown(f"### {teams}")

            for bookmaker in game["bookmakers"]:
                book = bookmaker["title"]
                st.write(f"**{book} Odds:**")
                for market in bookmaker["markets"]:
                    if market["key"] == "h2h":  # Moneyline
                        for outcome in market["outcomes"]:
                            st.write(f"Moneyline {outcome['name']}: {outcome['price']}")
                    elif market["key"] == "spreads":
                        for outcome in market["outcomes"]:
                            st.write(f"Spread {outcome['name']} {outcome['point']} → {outcome['price']}")
                    elif market["key"] == "totals":
                        for outcome in market["outcomes"]:
                            st.write(f"Total {outcome['name']} {outcome['point']} → {outcome['price']}")

            st.markdown("---")
