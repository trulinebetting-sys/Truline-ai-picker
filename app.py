import os
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ─────────────────────────────────────────────
# Safe dotenv
# ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ODDS_API_KEY = os.getenv("ODDS_API_KEY", st.secrets.get("ODDS_API_KEY", ""))
APISPORTS_KEY = os.getenv("APISPORTS_KEY", st.secrets.get("APISPORTS_KEY", ""))
DEFAULT_REGIONS = os.getenv("REGIONS", "us")

SOCCER_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_germany_bundesliga",
    "soccer_uefa_champions_league",
]

SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "College Football (NCAAF)": "americanfootball_ncaaf",
    "College Basketball (NCAAB)": "basketball_ncaab",
    "Soccer (All Major Leagues)": SOCCER_KEYS,
}

st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker")
st.caption("Live odds + historical context + AI-style ranking. No duplicates per game.")
st.divider()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def american_to_decimal(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    o = float(odds)
    return 1 + (o / 100.0) if o > 0 else 1 + (100.0 / abs(o))

def implied_prob_american(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)

def assign_units(conf: float, hist_boost: float = 0.0) -> float:
    """Units scale with both live confidence and historical win % boost."""
    if pd.isna(conf):
        return 0.5
    combined = conf + hist_boost
    return round(0.5 + 4.5 * max(0.0, min(1.0, combined)), 1)

def fmt_pct(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{100.0 * x:.1f}%"

# ─────────────────────────────────────────────
# Odds API fetch (live data)
# ─────────────────────────────────────────────
def _odds_get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY. Add it to `.env` or Streamlit Secrets.")
        return None
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            st.warning(f"Odds API error {r.status_code}: {r.text[:250]}")
            return None
        return r.json()
    except Exception as e:
        st.warning(f"Network error: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    data = _odds_get(url, {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american"
    })
    if not data:
        return pd.DataFrame()

    rows = []
    for ev in data:
        event_id = ev.get("id")
        commence = ev.get("commence_time")
        home = ev.get("home_team", "Unknown")
        away = ev.get("away_team", "Unknown")

        for bk in ev.get("bookmakers", []):
            book = bk.get("title")
            for mk in bk.get("markets", []):
                mkey = mk.get("key")
                for oc in mk.get("outcomes", []):
                    rows.append({
                        "event_id": event_id,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": mkey,
                        "outcome": oc.get("name"),
                        "line": oc.get("point"),
                        "odds_american": oc.get("price"),
                        "odds_decimal": american_to_decimal(oc.get("price")),
                        "conf_market": implied_prob_american(oc.get("price")),
                    })
    df = pd.DataFrame(rows)

    if not df.empty and "commence_time" in df.columns:
        df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
        if pd.api.types.is_datetime64tz_dtype(df["commence_time"]):
            df["Date/Time"] = df["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
        else:
            df["Date/Time"] = df["commence_time"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    return df

# ─────────────────────────────────────────────
# API-Sports fetch (historical context, optional)
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_historical(sport: str = "nfl") -> pd.DataFrame:
    if not APISPORTS_KEY:
        return pd.DataFrame()

    headers = {"x-apisports-key": APISPORTS_KEY}

    if sport.lower() == "nfl":
        url = "https://v1.american-football.api-sports.io/games?league=1&season=2023"
    elif sport.lower() == "nba":
        url = "https://v1.basketball.api-sports.io/games?league=12&season=2023"
    else:
        return pd.DataFrame()

    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json().get("response", [])
    rows = []
    for g in data:
        home = g.get("teams", {}).get("home", {}).get("name", "Unknown")
        away = g.get("teams", {}).get("away", {}).get("name", "Unknown")
        winner = g.get("scores", {}).get("winner", {}).get("name", None)
        rows.append({
            "Date": g.get("date"),
            "Home": home,
            "Away": away,
            "Winner": winner
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# Deduplicate best picks
# ─────────────────────────────────────────────
def best_per_event(df: pd.DataFrame, market_key: str, top_n: int = 10, hist: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    sub = df[df["market"] == market_key].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["rank"] = sub.groupby("event_id")["conf_market"].rank(method="first", ascending=False)
    sub = sub[sub["rank"] == 1].copy()
    sub = sub.sort_values("commence_time", ascending=True)  # most recent first
    sub = sub.head(top_n)

    sub["Matchup"] = sub["home_team"] + " vs " + sub["away_team"]

    # Historical boost
    win_rates = hist["Winner"].value_counts(normalize=True).to_dict() if not hist.empty else {}
    def hist_boost(row):
        for t in [row["home_team"], row["away_team"]]:
            if t in win_rates:
                return win_rates[t]
        return 0.0

    out = sub[["Date/Time", "Matchup", "book", "outcome", "line", "odds_american", "odds_decimal", "conf_market"]]
    out = out.rename(columns={
        "book": "Sportsbook",
        "outcome": "Pick",
        "line": "Line",
        "odds_american": "Odds (US)",
        "odds_decimal": "Odds (Dec)",
        "conf_market": "Confidence"
    })
    out["Confidence"] = out["Confidence"].apply(fmt_pct)
    out["Units"] = sub.apply(lambda r: assign_units(r["conf_market"], hist_boost(r)), axis=1)
    return out.reset_index(drop=True)

def ai_genius_top(df: pd.DataFrame, hist: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    frames = []
    for m in ["h2h", "totals", "spreads"]:
        t = best_per_event(df, m, top_n, hist)
        if not t.empty:
            t["Market"] = m
            frames.append(t)
    if not frames:
        return pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True)
    return allp.sort_values("Date/Time", ascending=True).head(top_n).reset_index(drop=True)

# ─────────────────────────────────────────────
# Sidebar + Main
# ─────────────────────────────────────────────
with st.sidebar:
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()), index=0)
    regions = st.text_input("Regions", value=DEFAULT_REGIONS)
    top_n = st.slider("Top picks per tab", 3, 20, 10)
    fetch = st.button("Fetch Live Odds")

if fetch:
    sport_key = SPORT_OPTIONS[sport_name]
    if isinstance(sport_key, list):
        parts = [fetch_odds(k, regions) for k in sport_key]
        raw = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame()
    else:
        raw = fetch_odds(sport_key, regions)

    if raw.empty:
        st.warning("No data returned. Try a different sport or check API quota.")
    else:
        hist = pd.DataFrame()
        if "NFL" in sport_name.upper():
            hist = fetch_historical("nfl")
        elif "NBA" in sport_name.upper():
            hist = fetch_historical("nba")

        tabs = st.tabs(["🤖 AI Genius Picks", "Moneylines", "Totals", "Spreads", "Raw Data"])

        with tabs[0]:
            st.subheader("AI Genius — Top Picks (Live + Historical)")
            board = ai_genius_top(raw, hist, top_n)
            st.dataframe(board, use_container_width=True, hide_index=True)

        with tabs[1]:
            t = best_per_event(raw, "h2h", top_n, hist)
            st.subheader("Best Moneyline per Game")
            st.dataframe(t, use_container_width=True, hide_index=True)

        with tabs[2]:
            t = best_per_event(raw, "totals", top_n, hist)
            st.subheader("Best Totals per Game")
            st.dataframe(t, use_container_width=True, hide_index=True)

        with tabs[3]:
            t = best_per_event(raw, "spreads", top_n, hist)
            st.subheader("Best Spreads per Game")
            st.dataframe(t, use_container_width=True, hide_index=True)

        with tabs[4]:
            st.subheader("Raw Data")
            st.dataframe(raw.head(200), use_container_width=True, hide_index=True)
else:
    st.info("Pick a sport and click **Fetch Live Odds**")
