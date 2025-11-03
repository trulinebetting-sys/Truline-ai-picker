# app.py
import os
from typing import Dict, Any, Optional, List
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────────
# Page / Header
# ─────────────────────────────────────────────
st.set_page_config(page_title="TruLine – AI Moneylines & Parlays", layout="wide")
st.title("TruLine – AI Moneylines & Parlays 🚀")
st.caption("Each sport: ALL moneyline picks (next 7 days) + a 5-leg parlay from the top ML edges.")
st.divider()

# ─────────────────────────────────────────────
# Config & Keys
# ─────────────────────────────────────────────
# Odds API key (env or fallback literal)
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip() or "1d677dc98d978ccc24d9914d835442f1"
DEFAULT_REGIONS = os.getenv("REGIONS", "us").strip() or "us"

# We only use moneylines for this build
MARKETS = "h2h"

# Sports enabled (Soccer is a bundle of leagues)
SOCCER_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_germany_bundesliga",
    "soccer_uefa_champions_league",
]

SPORT_OPTIONS: Dict[str, Any] = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "NCAAF": "americanfootball_ncaaf",
    "NCAAB": "basketball_ncaab",
    "Soccer": SOCCER_KEYS,  # aggregated
}

# Odds sanity filter: ignore ridiculous prices (keeps books' mistakes from breaking output)
ODDS_MIN_US = -2000
ODDS_MAX_US = 2000

# Ensemble → Units mapping (0.5 .. 5.0)
def assign_units_from_score(score: float) -> float:
    score = max(0.0, min(1.0, float(score)))
    return round(0.5 + 4.5 * score, 1)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def american_to_decimal(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    o = float(odds)
    return 1.0 + (o / 100.0) if o > 0 else 1.0 + (100.0 / abs(o))

def decimal_to_american(dec: float) -> Optional[int]:
    if dec is None or pd.isna(dec) or dec <= 1.0:
        return None
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100))
    else:
        return int(round(-100.0 / (dec - 1.0)))

def implied_prob_american(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)

def fmt_pct(x: float) -> str:
    try:
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return ""

def _odds_get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    """Basic GET with Odds API key."""
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY.")
        return None
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

# ─────────────────────────────────────────────
# Fetch odds (moneylines only)
# ─────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner=False)
def fetch_odds(sport_key: str, regions: str, markets: str = MARKETS) -> pd.DataFrame:
    """
    Fetch moneyline odds for a single sport key.
    """
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
                if mk.get("key") != "h2h":
                    continue
                for oc in mk.get("outcomes", []):
                    # outcome is "Home" / "Away" team string
                    rows.append({
                        "sport_key": sport_key,
                        "event_id": event_id,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": "h2h",
                        "outcome": oc.get("name"),
                        "line": None,  # not used for ML
                        "odds_american": oc.get("price"),
                        "odds_decimal": american_to_decimal(oc.get("price")),
                        "conf_book": implied_prob_american(oc.get("price")),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Parse / format time
    df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
    # Human-readable Eastern
    df["Date/Time"] = df["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    return df

@st.cache_data(ttl=120, show_spinner=False)
def fetch_sport(label: str, sport_value: Any, regions: str) -> pd.DataFrame:
    """
    Fetch for a sport label that is either a single key or a list of keys (e.g., Soccer).
    Adds 'sport_label' column for downstream filtering.
    """
    frames = []
    if isinstance(sport_value, list):
        for sub in sport_value:
            df = fetch_odds(sub, regions)
            if not df.empty:
                df["sport_label"] = label
                frames.append(df)
    else:
        df = fetch_odds(sport_value, regions)
        if not df.empty:
            df["sport_label"] = label
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ─────────────────────────────────────────────
# Consensus (moneylines)
# ─────────────────────────────────────────────
def build_consensus_ml(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Consensus per (event, outcome) for moneylines — keep best price per side,
    and compute aggregate stats.
    """
    if raw.empty:
        return raw

    # Best price row index per group (event + outcome)
    idx_best = raw.groupby(["event_id", "outcome"])["odds_decimal"].idxmax()
    best = raw.loc[idx_best, ["event_id", "outcome", "odds_american", "odds_decimal", "book"]]
    best = best.rename(columns={
        "odds_american": "best_odds_us",
        "odds_decimal": "best_odds_dec",
        "book": "best_book"
    })

    agg = raw.groupby(["event_id", "outcome"], dropna=False).agg(
        consensus_conf=("conf_book", "mean"),
        books=("book", "nunique"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        commence_time=("commence_time", "first"),
        date_time=("Date/Time", "first"),
        sport=("sport_label", "first"),
        sport_key=("sport_key", "first"),
        avg_odds_dec=("odds_decimal", "mean"),
    ).reset_index()

    out = agg.merge(best, on=["event_id", "outcome"], how="left")

    # Human fields
    out["Matchup"] = out["home_team"] + " vs " + out["away_team"]
    out["Date/Time"] = out["date_time"]
    out["market"] = "h2h"

    # Odds sanity filter
    out = out[(out["best_odds_us"] >= ODDS_MIN_US) & (out["best_odds_us"] <= ODDS_MAX_US)]

    return out[[
        "sport", "sport_key", "event_id", "commence_time", "Date/Time",
        "Matchup", "market", "outcome",
        "best_book", "best_odds_us", "best_odds_dec", "consensus_conf", "books", "avg_odds_dec"
    ]]

# ─────────────────────────────────────────────
# Ensemble “argumentation” score (4 voters)
# ─────────────────────────────────────────────
def ensemble_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    V1: Probability voter — higher implied prob (consensus_conf) → better
    V2: Edge vs avg price — (best_odds_dec - avg_odds_dec), sigmoided
    V3: Market depth — more books → more confidence (cap at 10)
    V4: Balance — best_odds_dec closer to 2.0 (evens) gets a small bump
    """
    if df.empty:
        return df.copy()

    d = df.copy()

    # V1
    v1 = d["consensus_conf"].astype(float).fillna(0.0).clip(0.0, 1.0)

    # V2
    edge = (d["best_odds_dec"] - d["avg_odds_dec"]).astype(float).fillna(0.0)
    v2 = 1.0 / (1.0 + np.exp(-6.0 * edge))  # edge>0 gives >0.5

    # V3
    v3 = (d["books"].astype(float).clip(lower=0.0, upper=10.0)) / 10.0

    # V4
    dec = d["best_odds_dec"].astype(float).fillna(2.0)
    v4_raw = np.exp(-((dec - 2.0) ** 2) / (2 * (0.6 ** 2)))  # N(2.0, 0.6)
    v4 = (v4_raw - v4_raw.min()) / (v4_raw.max() - v4_raw.min() + 1e-9)

    d["EnsembleScore"] = (v1 + v2 + v3 + v4) / 4.0
    d["Units"] = d["EnsembleScore"].apply(assign_units_from_score)
    return d

# ─────────────────────────────────────────────
# Build per-sport Moneyline picks for next 7 days
# ─────────────────────────────────────────────
def moneyline_picks_next7_for_sport(sport_label: str, regions: str) -> pd.DataFrame:
    """
    Fetch raw for this sport label, build consensus, ensemble score,
    pick one moneyline per event (higher score between the two sides),
    and return ALL games in next 7 days.
    """
    raw = fetch_sport(sport_label, SPORT_OPTIONS[sport_label], regions)
    if raw.empty:
        return pd.DataFrame()

    # Filter to next 7 days only (UTC)
    now_utc = datetime.now(timezone.utc)
    end_utc = now_utc + timedelta(days=7)
    raw = raw[(raw["commence_time"] >= now_utc) & (raw["commence_time"] <= end_utc)].copy()
    if raw.empty:
        return pd.DataFrame()

    cons = build_consensus_ml(raw)
    if cons.empty:
        return pd.DataFrame()

    scored = ensemble_score(cons)

    # For each event, keep the higher-scored side (Home vs Away)
    idx = scored.groupby("event_id")["EnsembleScore"].idxmax()
    best_side = scored.loc[idx].copy()

    # Display formatting
    best_side = best_side.sort_values("commence_time", ascending=True)
    best_side["Sport"] = best_side["sport"]
    best_side["Market"] = "Moneyline"
    best_side["Pick"] = best_side["outcome"]
    best_side["Line"] = ""  # not applicable for ML
    best_side["Odds (US)"] = best_side["best_odds_us"].astype(int)
    best_side["Odds (Dec)"] = best_side["best_odds_dec"].round(3)
    best_side["Confidence"] = best_side["consensus_conf"].apply(fmt_pct)
    best_side = best_side[[
        "Date/Time", "Sport", "Matchup", "Market", "Pick", "Line",
        "best_book", "Odds (US)", "Odds (Dec)", "Confidence", "Units",
        "EnsembleScore", "event_id"
    ]]
    best_side = best_side.rename(columns={"best_book": "Sportsbook"})
    return best_side.reset_index(drop=True)

# ─────────────────────────────────────────────
# Build a single 5-leg parlay for a sport from its ranks
# ─────────────────────────────────────────────
def build_5_leg_parlay_from_picks(picks_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the sport's ML table, take the top 5 by EnsembleScore with unique event_ids.
    Compute parlay odds and present a 1-row summary + legs breakdown table (2 dataframes).
    Returns a single dataframe that lists legs and adds a summary footer row.
    """
    if picks_df is None or picks_df.empty:
        return pd.DataFrame()

    # Sort by score (desc) and ensure unique events
    ranked = picks_df.sort_values("EnsembleScore", ascending=False).copy()
    ranked = ranked.drop_duplicates(subset=["event_id"], keep="first")

    legs = ranked.head(5).copy()
    if legs.empty:
        return pd.DataFrame()

    # Compute parlay decimal odds
    decs = legs["Odds (Dec)"].astype(float).replace([np.nan, 0.0], 1.01)
    parlay_dec = float(np.prod(decs.values))
    parlay_us = decimal_to_american(parlay_dec)

    # Suggested units: modest risk, scale by avg units of legs
    suggested_units = round(max(0.5, min(5.0, legs["Units"].astype(float).mean() * 1.0)), 1)

    # Legs table for display
    legs_display = legs[[
        "Date/Time", "Matchup", "Pick", "Sportsbook", "Odds (US)", "Odds (Dec)", "Units"
    ]].copy()
    legs_display.insert(0, "Leg #", list(range(1, len(legs_display) + 1)))

    # Add a final "summary" row as a separator
    summary_row = {
        "Leg #": "TOTAL",
        "Date/Time": "",
        "Matchup": f"Parlay of {len(legs_display)} legs",
        "Pick": "",
        "Sportsbook": "",
        "Odds (US)": parlay_us if parlay_us is not None else "",
        "Odds (Dec)": round(parlay_dec, 4),
        "Units": suggested_units
    }
    legs_display = pd.concat([legs_display, pd.DataFrame([summary_row])], ignore_index=True)
    return legs_display

# ─────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.subheader("Controls")
    regions = st.text_input("Odds Regions", value=DEFAULT_REGIONS, help="e.g., us, eu, uk, au")
    st.write("**Window**: Next 7 days (auto)")
    st.write("**Parlay size**: 5 legs (fixed)")
    run = st.button("Fetch / Refresh")

# ─────────────────────────────────────────────
# Generate (on click) and stash in session
# ─────────────────────────────────────────────
if run:
    st.session_state.results = {}
    for label in ["NFL", "NBA", "MLB", "NCAAF", "NCAAB", "Soccer"]:
        picks = moneyline_picks_next7_for_sport(label, regions)
        parlay = build_5_leg_parlay_from_picks(picks)
        st.session_state.results[label] = {
            "picks": picks,
            "parlay": parlay
        }
    st.session_state.has_data = True

# ─────────────────────────────────────────────
# Render Tabs
# ─────────────────────────────────────────────
if st.session_state.get("has_data", False):
    tabs = st.tabs(list(SPORT_OPTIONS.keys()))

    for tab, label in zip(tabs, SPORT_OPTIONS.keys()):
        with tab:
            st.markdown(f"## {label}")

            # Moneyline picks table
            st.markdown("### 🧠 AI Moneyline Picks (next 7 days)")
            picks_df = st.session_state.results.get(label, {}).get("picks", pd.DataFrame())
            if picks_df is None or picks_df.empty:
                st.info("No games found for the next 7 days.")
            else:
                show = picks_df.copy()
                # Make the score pretty
                show["EnsembleScore"] = show["EnsembleScore"].map(lambda x: f"{float(x):.3f}")
                st.dataframe(show.drop(columns=["event_id"]), use_container_width=True, hide_index=True)

            st.markdown("---")

            # Parlay table
            st.markdown("### 🎯 Suggested 5-Leg Parlay")
            parlay_df = st.session_state.results.get(label, {}).get("parlay", pd.DataFrame())
            if parlay_df is None or parlay_df.empty:
                st.info("Could not build a 5-leg parlay from current odds.")
            else:
                st.dataframe(parlay_df, use_container_width=True, hide_index=True)
else:
    st.info("Click **Fetch / Refresh** in the sidebar to populate each sport’s tab.")
