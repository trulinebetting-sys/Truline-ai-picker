import os
import requests
import streamlit as st
import math
from dotenv import load_dotenv

# --- Load ENV ---
load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")
REGIONS = os.getenv("REGIONS", "us")
MARKETS = os.getenv("MARKETS", "h2h,spreads,totals,player_props")
BOOKS = os.getenv("BOOKS", "DraftKings,FanDuel,BetMGM,Bet365")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TruLine Betting – AI Genius",
    page_icon="assets/logo.png",
    layout="wide"
)

# --- HEADER ---
st.image("assets/logo.png", width=120)
st.title("🤖 TruLine Betting – AI Genius Picks")

st.markdown("AI-powered betting assistant. Real odds, real picks, single legs & parlays.")

# --- Helper functions ---
def american_to_prob(odds):
    """Convert American odds to implied probability"""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def prob_to_american(prob):
    """Convert probability (0–1) to American odds"""
    if prob == 0:
        return None
    decimal = 1 / prob
    if decimal >= 2:
        return int((decimal - 1) * 100)
    else:
        return int(-100 / (decimal - 1))

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

def parlay_odds(odds_list):
    """Calculate parlay combined odds"""
    probs = [american_to_prob(o) for o in odds_list if o is not None]
    if not probs:
        return None
    combined_prob = math.prod(probs)
    return prob_to_american(combined_prob)

# --- Choose Sport ---
sport = st.selectbox(
    "Select a sport:",
    ["basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl", "soccer_epl", "soccer_uefa_champs_league"]
)

data, error = get_odds(sport)

if error:
    st.error(error)
else:
    if not data:
        st.warning("No games available right now. Try again later.")
    else:
        st.markdown("### Top AI Picks")

        best_picks = []
        for game in data:
            teams = game["home_team"] + " vs " + game["away_team"]

            for bookmaker in game["bookmakers"]:
                book = bookmaker["title"]
                for market in bookmaker["markets"]:
                    for outcome in market["outcomes"]:
                        pick = {
                            "Game": teams,
                            "Book": book,
                            "Market": market["key"],
                            "Pick": f"{outcome['name']} ({market['key']})",
                            "Odds": outcome["price"]
                        }
                        best_picks.append(pick)

        # For demo: sort by best odds (high positive EV look)
        best_picks = sorted(best_picks, key=lambda x: x["Odds"], reverse=True)[:10]

        for i, pick in enumerate(best_picks, start=1):
            st.markdown(f"**Pick {i}:** {pick['Game']} → {pick['Pick']} @ {pick['Odds']} ({pick['Book']})")

        # Build a suggested parlay
        if len(best_picks) >= 2:
            st.markdown("### Suggested Parlay")
            parlay = best_picks[:3]  # take top 3 for demo
            odds_list = [p["Odds"] for p in parlay]
            combined = parlay_odds(odds_list)
            if combined:
                st.success(f"Parlay: {', '.join([p['Pick'] for p in parlay])} → {combined} odds")
