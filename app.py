import os
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ───────────────────────────────────────────────────────────────────────────────
# Safe .env load (won’t crash if dotenv isn’t installed)
# ───────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # we’ll use Streamlit Secrets if dotenv isn’t present

# ───────────────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", st.secrets.get("ODDS_API_KEY", ""))
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

# ───────────────────────────────────────────────────────────────────────────────
# Page
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker")
st.caption("Live odds + (optional) recent form + AI-style ranking. No duplicates per game.")
st.divider()

# ───────────────────────────────────────────────────────────────────────────────
# Helpers (odds, math, formatting)
# ───────────────────────────────────────────────────────────────────────────────
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
    """Map confidence in [0..1] → units [0.5..5.0]."""
    if pd.isna(conf):
        return 0.5
    return round(0.5 + 4.5 * max(0.0, min(1.0, conf)), 1)

def fmt_pct(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{100.0 * x:.1f}%"

# ───────────────────────────────────────────────────────────────────────────────
# Odds API fetches
# ───────────────────────────────────────────────────────────────────────────────
def _odds_get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY. Add it to `.env` or Streamlit Secrets.")
        return None
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            st.warning(f"Odds API {url.split('/')[-1]} returned {r.status_code}: {r.text[:250]}")
            return None
        return r.json()
    except Exception as e:
        st.warning(f"Network error: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    """Return one row per (book, market, outcome) with point/line & price."""
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
        sport = ev.get("sport_key")
        commence = ev.get("commence_time")
        home = ev.get("home_team", "Unknown")
        away = ev.get("away_team", "Unknown")

        for bk in ev.get("bookmakers", []):
            book = bk.get("title")
            for mk in bk.get("markets", []):
                mkey = mk.get("key")  # "h2h" | "spreads" | "totals"
                for oc in mk.get("outcomes", []):
                    name = oc.get("name")              # Home/Away/Draw or Over/Under
                    price = oc.get("price")            # American odds
                    point = oc.get("point")            # spread/total line
                    rows.append({
                        "event_id": event_id,
                        "sport_key": sport,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": mkey,
                        "outcome": name,
                        "line": point,
                        "odds_american": price,
                        "odds_decimal": american_to_decimal(price),
                        "conf_market": implied_prob_american(price),  # market-based confidence
                    })
    df = pd.DataFrame(rows)
    # clean time for display
    if not df.empty and "commence_time" in df.columns:
        df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
        df["Date/Time"] = df["commence_time"].dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    return df

@st.cache_data(ttl=600)
def fetch_recent_scores(sport_key: str, days: int = 60) -> pd.DataFrame:
    """
    Pull recent final scores (when supported by The Odds API) to compute basic
    team win rates (recent form). Falls back silently if not available.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores/"
    data = _odds_get(url, {"apiKey": ODDS_API_KEY, "daysFrom": days})
    if not data:
        return pd.DataFrame()

    rows = []
    for g in data:
        # only completed games tend to have final scores
        ht = g.get("home_team")
        at = g.get("away_team")
        hs = g.get("home_score")
        as_ = g.get("away_score")
        if ht and at and isinstance(hs, (int, float)) and isinstance(as_, (int, float)):
            rows.append({"home_team": ht, "away_team": at, "home_score": hs, "away_score": as_})
    return pd.DataFrame(rows)

def team_win_rates_from_scores(scores: pd.DataFrame) -> Dict[str, float]:
    if scores.empty:
        return {}
    # compute win for home/away
    scores = scores.copy()
    scores["home_win"] = (scores["home_score"] > scores["away_score"]).astype(int)
    scores["away_win"] = (scores["away_score"] > scores["home_score"]).astype(int)

    home_rates = scores.groupby("home_team")["home_win"].mean()
    away_rates = scores.groupby("away_team")["away_win"].mean()

    # merge into a single dict (average home+away when both exist)
    teams = set(home_rates.index).union(set(away_rates.index))
    win_rate = {}
    for t in teams:
        vals = []
        if t in home_rates.index:
            vals.append(home_rates.loc[t])
        if t in away_rates.index:
            vals.append(away_rates.loc[t])
        win_rate[t] = float(np.mean(vals)) if vals else 0.5
    return win_rate

def attach_recent_form(df: pd.DataFrame, sport_name: str) -> pd.DataFrame:
    """Blend recent team win rates into confidence for moneylines. Non-destructive."""
    if df.empty or "market" not in df.columns:
        return df
    # Soccer option maps to many keys; use first for scores (others may not be supported for scores)
    sport_key = SPORT_OPTIONS[sport_name]
    key_for_scores = sport_key[0] if isinstance(sport_key, list) else sport_key

    scores = fetch_recent_scores(key_for_scores)
    if scores.empty:
        df["conf_blend"] = df["conf_market"]
        return df

    win_rates = team_win_rates_from_scores(scores)
    df = df.copy()
    df["home_wr"] = df["home_team"].map(win_rates)
    df["away_wr"] = df["away_team"].map(win_rates)

    # For H2H: if outcome == home team → use home_wr; if outcome == away team → use away_wr
    def blended(row):
        if row["market"] == "h2h":
            if row["outcome"] == row["home_team"]:
                wr = row.get("home_wr", np.nan)
            elif row["outcome"] == row["away_team"]:
                wr = row.get("away_wr", np.nan)
            else:
                wr = np.nan
            # simple blend: 60% market, 40% recent form (if available)
            if not pd.isna(wr):
                return 0.6 * row["conf_market"] + 0.4 * wr
        # for spreads/totals, keep market confidence
        return row["conf_market"]

    df["conf_blend"] = df.apply(blended, axis=1)
    return df

# ───────────────────────────────────────────────────────────────────────────────
# Build tables (dedupe per game; show top picks)
# ───────────────────────────────────────────────────────────────────────────────
def best_per_event(df: pd.DataFrame, market_key: str, top_n: int = 10) -> pd.DataFrame:
    """
    Pick the single best outcome per (event, market) by blended confidence.
    This removes duplicates for the same game.
    """
    sub = df[df["market"] == market_key].copy()
    if sub.empty:
        return pd.DataFrame()

    # Rank within each event by conf_blend (desc), then take top 1 per event
    sub["rank"] = sub.groupby("event_id")["conf_blend"].rank(method="first", ascending=False)
    sub = sub[sub["rank"] == 1].copy()
    sub = sub.sort_values("conf_blend", ascending=False).head(top_n)

    # user-friendly columns
    sub["Matchup"] = sub["home_team"] + " vs " + sub["away_team"]
    display = ["Date/Time", "Matchup", "book", "outcome", "line", "odds_american", "odds_decimal", "conf_blend"]
    # some markets don’t have a line (moneyline) – keep the column but it can be NaN
    out = sub[display].rename(columns={
        "book": "Sportsbook",
        "outcome": "Pick",
        "line": "Line",
        "odds_american": "Odds (US)",
        "odds_decimal": "Odds (Dec)",
        "conf_blend": "Confidence"
    })
    out["Confidence"] = out["Confidence"].apply(fmt_pct)
    # units from blended confidence
    out["Units"] = sub["conf_blend"].apply(assign_units)
    # line pretty for spreads/totals
    if market_key == "spreads":
        out["Line"] = out["Line"].apply(lambda x: "" if pd.isna(x) else f"{x:+}")
    elif market_key == "totals":
        out["Line"] = out["Line"].apply(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    else:
        # moneyline: line is not applicable
        out["Line"] = ""
    return out.reset_index(drop=True)

def ai_genius_top(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Cross-market leaderboard: take the best per event for each market,
    then overall top N by confidence.
    """
    frames = []
    for m in ["h2h", "totals", "spreads"]:
        t = best_per_event(df, m, top_n=top_n)
        if not t.empty:
            t["Market"] = m
            frames.append(t)
    if not frames:
        return pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True)
    # sort by Confidence (string like "64.3%") → convert to float
    tmp = allp.copy()
    tmp["_conf"] = tmp["Confidence"].str.replace("%", "", regex=False).astype(float)
    tmp = tmp.sort_values("_conf", ascending=False).drop(columns=["_conf"])
    return tmp.head(top_n).reset_index(drop=True)

# ───────────────────────────────────────────────────────────────────────────────
# Sidebar
# ───────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()), index=0)
    regions = st.text_input("Regions", value=DEFAULT_REGIONS)
    top_n = st.slider("Top picks per tab", 3, 20, 10)
    fetch = st.button("Fetch Live Odds")

# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
if fetch:
    sport_key = SPORT_OPTIONS[sport_name]
    # Aggregate soccer leagues
    if isinstance(sport_key, list):
        parts = [fetch_odds(k, regions) for k in sport_key]
        raw = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame()
    else:
        raw = fetch_odds(sport_key, regions)

    if raw.empty:
        st.warning("No data returned. Try a different sport or check API quota.")
    else:
        # blend in recent form (safe, optional)
        work = attach_recent_form(raw, sport_name)
        # guard defaults
        if "conf_blend" not in work.columns:
            work["conf_blend"] = work.get("conf_market", np.nan)

        tabs = st.tabs(["🤖 AI Genius Picks", "Moneylines", "Totals", "Spreads", "Raw Data"])

        # AI Genius
        with tabs[0]:
            board = ai_genius_top(work, top_n=top_n)
            st.subheader("AI Genius — Top Picks (deduplicated by game)")
            if board.empty:
                st.info("No picks available.")
            else:
                st.dataframe(board, use_container_width=True, hide_index=True)

        # Moneylines
        with tabs[1]:
            t = best_per_event(work, "h2h", top_n=top_n)
            st.subheader("Best Moneyline per Game")
            if t.empty:
                st.info("No moneyline data available.")
            else:
                st.dataframe(t, use_container_width=True, hide_index=True)

        # Totals
        with tabs[2]:
            t = best_per_event(work, "totals", top_n=top_n)
            st.subheader("Best Total (O/U) per Game")
            if t.empty:
                st.info("No totals data available.")
            else:
                st.dataframe(t, use_container_width=True, hide_index=True)

        # Spreads
        with tabs[3]:
            t = best_per_event(work, "spreads", top_n=top_n)
            st.subheader("Best Spread per Game")
            if t.empty:
                st.info("No spreads data available.")
            else:
                st.dataframe(t, use_container_width=True, hide_index=True)

        # Raw
        with tabs[4]:
            show = work.copy()
            # tidy display
            base_cols = ["Date/Time", "home_team", "away_team", "book", "market", "outcome", "line", "odds_american", "odds_decimal", "conf_market", "conf_blend"]
            cols = [c for c in base_cols if c in show.columns]
            if cols:
                show = show[cols]
                show = show.rename(columns={
                    "home_team": "Home",
                    "away_team": "Away",
                    "book": "Sportsbook",
                    "market": "Market",
                    "outcome": "Pick",
                    "line": "Line",
                    "odds_american": "Odds (US)",
                    "odds_decimal": "Odds (Dec)",
                    "conf_market": "Conf (Market)",
                    "conf_blend": "Conf (Blended)"
                })
                show["Conf (Market)"] = show["Conf (Market)"].apply(fmt_pct)
                show["Conf (Blended)"] = show["Conf (Blended)"].apply(fmt_pct)
            st.dataframe(show.head(1000), use_container_width=True, hide_index=True)
else:
    st.info("Set your filters in the sidebar, then click **Fetch Live Odds**.")
