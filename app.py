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
st.caption("Live odds + historical context + AI-style ranking. Tracks results automatically ✅")
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

def assign_units(conf: float) -> float:
    if pd.isna(conf):
        return 0.5
    return round(0.5 + 4.5 * max(0.0, min(1.0, conf)), 1)

def fmt_pct(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{100.0 * x:.1f}%"

# ─────────────────────────────────────────────
# Odds API fetch
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
                        "sport": sport_name,  # tag picks by sport
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
# Results tracking
# ─────────────────────────────────────────────
RESULTS_FILE = "bets.csv"

def load_results() -> pd.DataFrame:
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        # ✅ Ensure Sport column exists
        if "Sport" not in df.columns:
            df["Sport"] = "Unknown"
        return df
    return pd.DataFrame(columns=["Sport", "Date/Time", "Matchup", "Pick", "Line", "Odds (US)", "Units", "Result"])

def save_results(df: pd.DataFrame):
    df.to_csv(RESULTS_FILE, index=False)

def auto_log_picks(dfs: Dict[str, pd.DataFrame], sport_name: str):
    results = load_results()
    for name, picks in dfs.items():
        if not picks.empty:
            for _, row in picks.iterrows():
                entry = {
                    "Sport": sport_name,
                    "Date/Time": row["Date/Time"],
                    "Matchup": row["Matchup"],
                    "Pick": row["Pick"],
                    "Line": row["Line"],
                    "Odds (US)": row["Odds (US)"],
                    "Units": row["Units"],
                    "Result": "Pending"
                }
                if not ((results["Sport"] == entry["Sport"]) &
                        (results["Date/Time"] == entry["Date/Time"]) &
                        (results["Matchup"] == entry["Matchup"]) &
                        (results["Pick"] == entry["Pick"])).any():
                    results = pd.concat([results, pd.DataFrame([entry])], ignore_index=True)
    save_results(results)

def show_results():
    results = load_results()
    if results.empty:
        st.info("No bets logged yet.")
        return
    st.subheader("📊 Results Tracker by Sport")
    st.dataframe(results, use_container_width=True, hide_index=True)

    grouped = results.groupby("Sport")
    for sport, df in grouped:
        total = len(df)
        wins = (df["Result"] == "Win").sum()
        losses = (df["Result"] == "Loss").sum()
        if total > 0:
            win_pct = (wins / total) * 100
            st.metric(f"{sport} Win %", f"{win_pct:.1f}% ({wins}-{losses})")

# ─────────────────────────────────────────────
# Sidebar + Main
# ─────────────────────────────────────────────
with st.sidebar:
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()), index=0)
    regions = st.text_input("Regions", value=DEFAULT_REGIONS)
    top_n = st.slider("Top picks per tab", 3, 20, 10)
    fetch = st.button("Fetch Live Odds")

def best_per_event(df: pd.DataFrame, market_key: str, top_n: int = 10) -> pd.DataFrame:
    sub = df[df["market"] == market_key].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.loc[sub.groupby("event_id")["conf_market"].idxmax()].copy()
    sub = sub.sort_values("commence_time", ascending=True).head(top_n)
    sub["Matchup"] = sub["home_team"] + " vs " + sub["away_team"]
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
    out["Units"] = sub["conf_market"].apply(assign_units)
    return out.reset_index(drop=True)

def ai_genius_top(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    best = df.loc[df.groupby("event_id")["conf_market"].idxmax()].copy()
    best = best.sort_values("conf_market", ascending=False).head(top_n)
    best["Matchup"] = best["home_team"] + " vs " + best["away_team"]
    out = best[["Date/Time", "Matchup", "book", "outcome", "line", "odds_american", "odds_decimal", "conf_market"]]
    out = out.rename(columns={
        "book": "Sportsbook",
        "outcome": "Pick",
        "line": "Line",
        "odds_american": "Odds (US)",
        "odds_decimal": "Odds (Dec)",
        "conf_market": "Confidence"
    })
    out["Confidence"] = out["Confidence"].apply(fmt_pct)
    out["Units"] = best["conf_market"].apply(assign_units)
    return out.reset_index(drop=True)

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
        ml = best_per_event(raw, "h2h", top_n)
        totals = best_per_event(raw, "totals", top_n)
        spreads = best_per_event(raw, "spreads", top_n)
        ai_picks = ai_genius_top(raw, top_n)

        auto_log_picks({"Moneyline": ml, "Totals": totals, "Spreads": spreads, "AI Genius": ai_picks}, sport_name)

        tabs = st.tabs(["🤖 AI Genius Picks", "Moneylines", "Totals", "Spreads", "Raw Data", "📊 Results"])

        with tabs[0]:
            st.subheader("AI Genius — Highest Confidence Picks")
            st.dataframe(ai_picks, use_container_width=True, hide_index=True)

        with tabs[1]:
            st.subheader("Best Moneyline per Game")
            st.dataframe(ml, use_container_width=True, hide_index=True)

        with tabs[2]:
            st.subheader("Best Totals per Game")
            st.dataframe(totals, use_container_width=True, hide_index=True)

        with tabs[3]:
            st.subheader("Best Spreads per Game")
            st.dataframe(spreads, use_container_width=True, hide_index=True)

        with tabs[4]:
            st.subheader("Raw Data")
            st.dataframe(raw.head(200), use_container_width=True, hide_index=True)

        with tabs[5]:
            show_results()
else:
    st.info("Pick a sport and click **Fetch Live Odds**")
