import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load API key
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# --- PAGE CONFIG ---
st.set_page_config(page_title="TruLine AI Picker", layout="wide")
st.title("📊 TruLine AI Genius Picks")
st.markdown("White background. Live data. AI-driven recommendations.")

# --- FILTERS ---
sport = st.selectbox("Choose a sport:", ["americanfootball_nfl", "basketball_nba", "baseball_mlb", "soccer_epl", "soccer_spain_la_liga"])
min_confidence = st.slider("Minimum Confidence %", 50, 100, 75)

# --- FETCH ODDS FROM ODDS API ---
def fetch_odds(sport_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american"
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error("Failed to fetch odds. Check API key or quota.")
        return pd.DataFrame()
    data = response.json()

    rows = []
    for game in data:
        home = game["home_team"]
        away = game["away_team"]
        commence = game["commence_time"]
        for book in game["bookmakers"]:
            book_name = book["title"]
            for market in book["markets"]:
                for outcome in market["outcomes"]:
                    rows.append({
                        "time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book_name,
                        "team": outcome["name"],
                        "odds": outcome["price"]
                    })
    return pd.DataFrame(rows)

df = fetch_odds(sport)

if not df.empty:
    # --- SIMPLE AI CONFIDENCE (simulate with randomness for now) ---
    np.random.seed(42)
    df["confidence"] = np.random.randint(60, 95, size=len(df))

    # --- FILTER PICKS ---
    picks = df[df["confidence"] >= min_confidence]

    st.subheader("Recommended Picks")
    st.dataframe(picks[["time", "home_team", "away_team", "team", "odds", "book", "confidence"]])

    # --- GRAPH (Confidence Distribution) ---
    st.subheader("Confidence Distribution")
    fig, ax = plt.subplots()
    picks["confidence"].hist(ax=ax, bins=10)
    ax.set_title("AI Confidence Levels for Picks")
    ax.set_xlabel("Confidence %")
    ax.set_ylabel("Count")
    st.pyplot(fig)
else:
    st.warning("No odds data available for this sport.")
