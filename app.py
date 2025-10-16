import os
from typing import Dict, Any, Optional
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ─────────────────────────────────────────────
# Safe dotenv (still loads .env but Odds API key is now hard-coded)
# ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ✅ Hard-coded Odds API key
ODDS_API_KEY = "1d677dc98d978ccc24d9914d835442f1"
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

SPORT_API_ENDPOINTS = {
    "NFL": "https://v1.american-football.api-sports.io/games?league=1&season=2023",
    "NBA": "https://v1.basketball.api-sports.io/games?league=12&season=2023",
    "MLB": "https://v1.baseball.api-sports.io/games?league=1&season=2023",
    "College Football (NCAAF)": "https://v1.american-football.api-sports.io/games?league=2&season=2023",
    "College Basketball (NCAAB)": "https://v1.basketball.api-sports.io/games?league=7&season=2023",
    "Soccer (All Major Leagues)": "https://v3.football.api-sports.io/fixtures?season=2023&league=39"
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
    return 1 + (o / 100.0) if o > 0 else 1 + (100.0 / abs(o))

def implied_prob_american(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds): return np.nan
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)

def assign_units(conf: float) -> float:
    """Units purely from consensus confidence 0..1 (no bankroll dependency)."""
    if pd.isna(conf): return 0.5
    return round(0.5 + 4.5 * max(0.0, min(1.0, conf)), 1)

def fmt_pct(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{100.0 * x:.1f}%"

# ─────────────────────────────────────────────
# Odds API fetch (raw per-book rows)
# ─────────────────────────────────────────────
def _odds_get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY.")
        return None
    try:
        r = requests.get(url, params=params, timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception:
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
        home, away = ev.get("home_team", "Unknown"), ev.get("away_team", "Unknown")

        for bk in ev.get("bookmakers", []):
            book = bk.get("title")
            for mk in bk.get("markets", []):
                mkey = mk.get("key")  # h2h, spreads, totals
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
                        "conf_book": implied_prob_american(oc.get("price")),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize time
    df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(df["commence_time"]):
        df["Date/Time"] = df["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    else:
        df["Date/Time"] = df["commence_time"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    return df

# ─────────────────────────────────────────────
# CONSENSUS across books
# ─────────────────────────────────────────────
def build_consensus(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    idx_best = raw.groupby(["event_id", "market", "outcome", "line"])["odds_decimal"].idxmax()
    best = raw.loc[idx_best, ["event_id","market","outcome","line","odds_american","odds_decimal","book"]]
    best = best.rename(columns={"odds_american":"best_odds_us","odds_decimal":"best_odds_dec","book":"best_book"})

    agg = raw.groupby(["event_id","market","outcome","line"], dropna=False).agg(
        consensus_conf=("conf_book","mean"),
        books=("book","nunique"),
        home_team=("home_team","first"),
        away_team=("away_team","first"),
        commence_time=("commence_time","first"),
        date_time=("Date/Time","first"),
    ).reset_index()

    out = agg.merge(best, on=["event_id","market","outcome","line"], how="left")
    out["Matchup"] = out["home_team"] + " vs " + out["away_team"]
    out["Confidence"] = out["consensus_conf"]
    out["Odds (US)"] = out["best_odds_us"]
    out["Odds (Dec)"] = out["best_odds_dec"]
    out["Sportsbook"] = out["best_book"]
    out["Date/Time"] = out["date_time"]

    return out[[
        "event_id","commence_time","Date/Time","Matchup","market","outcome","line",
        "Sportsbook","Odds (US)","Odds (Dec)","Confidence","books"
    ]].rename(columns={"books":"Books"})

def pick_best_per_event(cons_df: pd.DataFrame, market_key: str, top_n: int) -> pd.DataFrame:
    sub = cons_df[cons_df["market"] == market_key].copy()
    if sub.empty:
        return pd.DataFrame()
    best_idx = sub.groupby("event_id")["Confidence"].idxmax()
    sub = sub.loc[best_idx].copy()
    sub = sub.sort_values("commence_time", ascending=True).head(top_n)
    out = sub[["Date/Time","Matchup","Sportsbook","outcome","line","Odds (US)","Odds (Dec)","Confidence","Books"]].copy()
    out = out.rename(columns={"outcome":"Pick","line":"Line"})
    out["Confidence"] = out["Confidence"].apply(fmt_pct)
    out["Units"] = sub["Confidence"].apply(assign_units)
    return out.reset_index(drop=True)

def ai_genius_top(cons_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if cons_df.empty:
        return pd.DataFrame()
    frames = []
    for m in ["h2h","totals","spreads"]:
        t = pick_best_per_event(cons_df, m, top_n*3)
        if not t.empty:
            t["Market"] = m
            t["_C"] = t["Confidence"].str.replace("%","",regex=False).astype(float)
            frames.append(t)
    if not frames:
        return pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True)
    allp = allp.sort_values("_C", ascending=False).drop(columns=["_C"]).head(top_n)
    return allp.reset_index(drop=True)

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
        raw = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    else:
        raw = fetch_odds(sport_key, regions)

    if raw.empty:
        st.warning("No data returned. Try a different sport or check API quota.")
    else:
        cons = build_consensus(raw)
        ml = pick_best_per_event(cons, "h2h", top_n)
        totals = pick_best_per_event(cons, "totals", top_n)
        spreads = pick_best_per_event(cons, "spreads", top_n)
        ai_picks = ai_genius_top(cons, min(top_n, 5))

        tabs = st.tabs(["🤖 AI Genius Picks","Moneylines","Totals","Spreads","Raw Data"])

        with tabs[0]:
            st.subheader("AI Genius — Highest Consensus Confidence (Top)")
            st.dataframe(ai_picks, use_container_width=True, hide_index=True)

        with tabs[1]:
            st.subheader("Best Moneyline per Game (Consensus)")
            st.dataframe(ml, use_container_width=True, hide_index=True)

        with tabs[2]:
            st.subheader("Best Totals per Game (Consensus)")
            st.dataframe(totals, use_container_width=True, hide_index=True)

        with tabs[3]:
            st.subheader("Best Spreads per Game (Consensus)")
            st.dataframe(spreads, use_container_width=True, hide_index=True)

        with tabs[4]:
            st.subheader("Raw Per-Book Odds (first 200 rows)")
            st.dataframe(raw.head(200), use_container_width=True, hide_index=True)
else:
    st.info("Pick a sport and click **Fetch Live Odds**")
