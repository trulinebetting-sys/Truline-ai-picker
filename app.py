import os
from typing import Dict, Any, Optional
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

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

# Dynamic current season
CURRENT_SEASON = datetime.now().year

SOCCER_KEYS = [
    "soccer_epl","soccer_spain_la_liga","soccer_italy_serie_a",
    "soccer_france_ligue_one","soccer_germany_bundesliga","soccer_uefa_champions_league",
]

SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "College Football (NCAAF)": "americanfootball_ncaaf",
    "College Basketball (NCAAB)": "basketball_ncaab",
    "Soccer (All Major Leagues)": SOCCER_KEYS,
}

SPORT_API_ENDPOINTS = {
    "NFL": f"https://v1.american-football.api-sports.io/games?league=1&season={CURRENT_SEASON}",
    "NBA": f"https://v1.basketball.api-sports.io/games?league=12&season={CURRENT_SEASON}",
    "MLB": f"https://v1.baseball.api-sports.io/games?league=1&season={CURRENT_SEASON}",
    "College Football (NCAAF)": f"https://v1.american-football.api-sports.io/games?league=2&season={CURRENT_SEASON}",
    "College Basketball (NCAAB)": f"https://v1.basketball.api-sports.io/games?league=7&season={CURRENT_SEASON}",
    "Soccer (All Major Leagues)": f"https://v3.football.api-sports.io/fixtures?season={CURRENT_SEASON}&league=39"
}

st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker 🚀")
st.caption("Consensus across books + live odds + AI-style ranking. Tracks results + bankroll ✅")
st.divider()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def american_to_decimal(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds): return np.nan
    o = float(odds)
    return 1 + (o/100.0) if o > 0 else 1 + (100.0/abs(o))

def implied_prob_american(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds): return np.nan
    o = float(odds)
    return 100.0/(o+100.0) if o > 0 else abs(o)/(abs(o)+100.0)

def assign_units(conf: float) -> float:
    if pd.isna(conf): return 0.5
    return round(0.5+4.5*max(0.0,min(1.0,conf)),1)

def fmt_pct(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{100.0*x:.1f}%"

# ─────────────────────────────────────────────
# Odds API fetch
# ─────────────────────────────────────────────
def _odds_get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    if not ODDS_API_KEY:
        st.error("❌ Missing ODDS_API_KEY.")
        return None
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            st.error(f"❌ Odds API error {r.status_code}: {r.text[:250]}")
            return None
        return r.json()
    except Exception as e:
        st.error(f"❌ Network error: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str="h2h,spreads,totals") -> pd.DataFrame:
    url=f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    data=_odds_get(url,{
        "apiKey":ODDS_API_KEY,"regions":regions,"markets":markets,"oddsFormat":"american"
    })
    if not data: return pd.DataFrame()

    rows=[]
    for ev in data:
        event_id=ev.get("id")
        commence=ev.get("commence_time")
        home,away=ev.get("home_team","Unknown"),ev.get("away_team","Unknown")
        for bk in ev.get("bookmakers",[]):
            book=bk.get("title")
            for mk in bk.get("markets",[]):
                mkey=mk.get("key")
                for oc in mk.get("outcomes",[]):
                    rows.append({
                        "event_id":event_id,
                        "commence_time":commence,
                        "home_team":home,
                        "away_team":away,
                        "book":book,
                        "market":mkey,
                        "outcome":oc.get("name"),
                        "line":oc.get("point"),
                        "odds_american":oc.get("price"),
                        "odds_decimal":american_to_decimal(oc.get("price")),
                        "conf_book":implied_prob_american(oc.get("price")),
                    })
    df=pd.DataFrame(rows)
    if df.empty: return df
    df["commence_time"]=pd.to_datetime(df["commence_time"],errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(df["commence_time"]):
        df["Date/Time"]=df["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    else:
        df["Date/Time"]=df["commence_time"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    return df

# ─────────────────────────────────────────────
# Consensus + Results (unchanged logic)
# ─────────────────────────────────────────────
# (same as last version you had: build_consensus, ai_genius_top, results tracking, graphs, show_results, etc.)
# ... keep those functions exactly the same ...

# ─────────────────────────────────────────────
# Sidebar + Fetch + Render
# ─────────────────────────────────────────────
with st.sidebar:
    sport_name=st.selectbox("Sport",list(SPORT_OPTIONS.keys()),index=0)
    regions=st.text_input("Regions",value=DEFAULT_REGIONS)
    top_n=st.slider("Top picks per tab",3,20,10)
    fetch=st.button("Fetch Live Odds")

if fetch:
    sport_key=SPORT_OPTIONS[sport_name]
    if isinstance(sport_key,list):
        raw=pd.concat([fetch_odds(k,regions) for k in sport_key],ignore_index=True) if sport_key else pd.DataFrame()
    else:
        raw=fetch_odds(sport_key,regions)

    if raw.empty:
        st.warning(f"⚠️ No live odds data for {sport_name}. Try a different region or check if games are scheduled today/tomorrow.")
    else:
        # your consensus_tables, ai_picks, ml, totals, spreads, show_results etc.
        st.success(f"✅ Pulled {len(raw)} odds rows for {sport_name}")
