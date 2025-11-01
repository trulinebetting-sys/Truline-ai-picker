import os
from typing import Dict, Any, Optional, List, Tuple
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ─────────────────────────────────────────────
# Safe dotenv (optional)
# ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ─────────────────────────────────────────────
# Config / Constants
# ─────────────────────────────────────────────
st.set_page_config(page_title="TruLine – AI Picker + Parlay Lab", layout="wide")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "") or "1d677dc98d978ccc24d9914d835442f1"
DEFAULT_REGIONS = os.getenv("REGIONS", "us")

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
    "College Football (NCAAF)": "americanfootball_ncaaf",
    "College Basketball (NCAAB)": "basketball_ncaab",
    "Soccer (All Major Leagues)": SOCCER_KEYS,
}

MARKETS_ALL = "h2h,spreads,totals"   # (moneyline = h2h)

# Parlay Lab knobs (tweak if you want)
PARLAY_SIZES = [2, 2, 3, 3, 4]      # how many legs in each of the five suggested parlays
MIN_SINGLE_LEG_PROB = 0.55          # ignore low-probability legs for parlays
MAX_LEGS_PER_EVENT = 1              # avoid using more than 1 leg from the same game

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def american_to_decimal(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    o = float(odds)
    return 1 + (o / 100.0) if o > 0 else 1 + (100.0 / abs(o))

def decimal_to_american(dec: float) -> Optional[int]:
    try:
        if dec <= 1:
            return None
        if dec >= 2:
            return int(round((dec - 1) * 100))
        else:
            return int(round(-100 / (dec - 1)))
    except Exception:
        return None

def implied_prob_american(odds: Optional[float]) -> float:
    # Returns a 0..1 probability
    if odds is None or pd.isna(odds):
        return np.nan
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)

def fmt_pct_100(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{100.0 * x:.1f}%"

def odds_get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = MARKETS_ALL) -> pd.DataFrame:
    """
    Returns a long table of all bookmaker outcomes for a sport (and soccer leagues)
    Columns:
      event_id, commence_time, home_team, away_team, book, market, outcome, line, odds_american, odds_decimal, conf_book, Date/Time
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    data = odds_get(url, {
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
                mkey = mk.get("key")  # 'h2h' | 'spreads' | 'totals'
                for oc in mk.get("outcomes", []):
                    rows.append({
                        "event_id": event_id,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": mkey,
                        "outcome": oc.get("name"),       # Home/Away/Over/Under
                        "line": oc.get("point"),         # may be None for ML
                        "odds_american": oc.get("price"),
                        "odds_decimal": american_to_decimal(oc.get("price")),
                        "conf_book": implied_prob_american(oc.get("price")),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize datetimes
    df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
    try:
        if pd.api.types.is_datetime64tz_dtype(df["commence_time"]):
            df["Date/Time"] = df["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
        else:
            df["Date/Time"] = df["commence_time"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    except Exception:
        df["Date/Time"] = df["commence_time"].astype(str)

    return df

def build_consensus(raw: pd.DataFrame) -> pd.DataFrame:
    """
    For each (event, market, outcome, line), pick best odds across books and compute consensus confidence
    Returns compact table:
      event_id, commence_time, Date/Time, Matchup, sport_key, market, outcome, line, Sportsbook, Odds (US), Odds (Dec), Confidence, Books
    """
    if raw.empty:
        return raw

    # Keep a reference to which sport each row came from (already present upstream)
    # The fetcher will add 'sport_key' upstream before concat. If missing, fill Unknown.
    if "sport_key" not in raw.columns:
        raw["sport_key"] = "Unknown"

    idx_best = raw.groupby(["event_id", "market", "outcome", "line"])["odds_decimal"].idxmax()
    best = raw.loc[idx_best, ["event_id","market","outcome","line","odds_american","odds_decimal","book"]]
    best = best.rename(columns={
        "odds_american":"best_odds_us",
        "odds_decimal":"best_odds_dec",
        "book":"best_book"
    })

    agg = raw.groupby(["event_id","market","outcome","line"], dropna=False).agg(
        consensus_conf=("conf_book","mean"),
        books=("book","nunique"),
        home_team=("home_team","first"),
        away_team=("away_team","first"),
        commence_time=("commence_time","first"),
        date_time=("Date/Time","first"),
        sport_key=("sport_key","first"),
    ).reset_index()

    out = agg.merge(best, on=["event_id","market","outcome","line"], how="left")
    out["Matchup"] = out["home_team"] + " vs " + out["away_team"]
    out["Confidence"] = out["consensus_conf"]
    out["Odds (US)"] = out["best_odds_us"]
    out["Odds (Dec)"] = out["best_odds_dec"]
    out["Sportsbook"] = out["best_book"]
    out["Date/Time"] = out["date_time"]

    cols = [
        "event_id","commence_time","Date/Time","Matchup","sport_key","market","outcome","line",
        "Sportsbook","Odds (US)","Odds (Dec)","Confidence","books"
    ]
    return out[cols].rename(columns={"books":"Books"})

def top_legs_across_all(cons: pd.DataFrame, top_n_per_market: int = 20) -> pd.DataFrame:
    """
    Build a pool of strong legs to choose from. For each market, keep the best (highest Confidence)
    pick per event; then take top_n_per_market across all events for that market. Concatenate
    ML + Spreads + Totals pools together.
    """
    if cons.empty:
        return pd.DataFrame()

    def best_per_event(cons_df: pd.DataFrame, mkey: str) -> pd.DataFrame:
        sub = cons_df[cons_df["market"] == mkey].copy()
        if sub.empty:
            return pd.DataFrame()
        best_idx = sub.groupby("event_id")["Confidence"].idxmax()
        sub = sub.loc[best_idx].copy()
        sub = sub.sort_values("Confidence", ascending=False).head(top_n_per_market)
        sub["Market"] = mkey
        return sub

    ml = best_per_event(cons, "h2h")
    sp = best_per_event(cons, "spreads")
    tot = best_per_event(cons, "totals")

    frames = [x for x in [ml, sp, tot] if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame()

    pool = pd.concat(frames, ignore_index=True)
    # Add Pick label and pretty line
    def pretty_line(row) -> str:
        if row["market"] == "h2h":
            return ""
        try:
            ln = float(row["line"])
            if row["market"] == "spreads":
                return f"{'+' if ln > 0 else ''}{ln:.1f}"
            else:
                # totals keep the number only; 'outcome' is Over/Under
                return f"{ln:.1f}"
        except Exception:
            return str(row["line"]) if pd.notna(row["line"]) else ""

    def pick_label(row) -> str:
        if row["market"] == "h2h":
            # outcome is team
            return f"{row['outcome']}"
        elif row["market"] == "spreads":
            return f"{row['outcome']} ({pretty_line(row)})"
        else:
            # totals
            return f"{row['outcome']} ({pretty_line(row)})"

    pool["Pick"] = pool.apply(pick_label, axis=1)
    pool["ConfidencePct"] = pool["Confidence"].apply(lambda x: 100.0*x if pd.notna(x) else np.nan)
    return pool

def ai_top5_across_all(cons: pd.DataFrame) -> pd.DataFrame:
    """
    Rank the best 5 legs across all sports/markets by consensus Confidence (probability).
    """
    pool = top_legs_across_all(cons, top_n_per_market=40)
    if pool.empty:
        return pd.DataFrame()

    pool = pool.sort_values("Confidence", ascending=False)
    top5 = pool.head(5).copy()

    # Final display table
    out = top5[[
        "Date/Time","sport_key","Matchup","Market","Pick","Odds (US)","Odds (Dec)","Confidence"
    ]].copy()
    out = out.rename(columns={
        "sport_key":"Sport",
        "Market":"Market"
    })
    out["Confidence"] = out["Confidence"].apply(fmt_pct_100)
    out["Odds (Dec)"] = out["Odds (Dec)"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    return out.reset_index(drop=True)

def parlay_lab_5(cons: pd.DataFrame) -> Tuple[pd.DataFrame, List[List[dict]]]:
    """
    Build 5 parlays from high-probability legs (greedy). Avoid legs from the same event in a single parlay.
    Return:
      display_df, raw_parlays (list of list of leg dicts)
    """
    pool = top_legs_across_all(cons, top_n_per_market=60)
    if pool.empty:
        return pd.DataFrame(), []

    # Only keep strong legs
    pool = pool[pool["Confidence"] >= MIN_SINGLE_LEG_PROB].copy()
    if pool.empty:
        return pd.DataFrame(), []

    # Leg dictionary helper
    def row_to_leg(r: pd.Series) -> dict:
        return {
            "event_id": r["event_id"],
            "Sport": r["sport_key"],
            "Matchup": r["Matchup"],
            "Market": "Moneyline" if r["market"] == "h2h" else ("Spreads" if r["market"] == "spreads" else "Totals"),
            "Pick": r["Pick"],
            "OddsDec": float(r["Odds (Dec)"]) if pd.notna(r["Odds (Dec)"]) else np.nan,
            "OddsUS": int(r["Odds (US)"]) if pd.notna(r["Odds (US)"]) else None,
            "Prob": float(r["Confidence"]) if pd.notna(r["Confidence"]) else np.nan,
            "Date/Time": r["Date/Time"]
        }

    # Sort by prob desc
    pool = pool.sort_values("Confidence", ascending=False)

    # Build parlays with sizes PARLAY_SIZES
    parlays: List[List[dict]] = []
    used_pairs = set()  # optional: avoid building identical parlay sets

    for size in PARLAY_SIZES:
        legs: List[dict] = []
        used_events = set()

        for _, r in pool.iterrows():
            leg = row_to_leg(r)
            ev = leg["event_id"]
            if ev in used_events:
                continue
            # keep at most 1 leg per event for correlation hygiene
            legs.append(leg)
            used_events.add(ev)
            if len(legs) >= size:
                break

        if len(legs) < 2:
            # if we can't get at least a 2-leg parlay, skip
            continue

        # check duplicate parlay content
        key = tuple(sorted([(x["event_id"], x["Pick"]) for x in legs]))
        if key in used_pairs:
            continue
        used_pairs.add(key)
        parlays.append(legs)

    if not parlays:
        return pd.DataFrame(), []

    # Summarize parlays to a display table
    records = []
    for idx, legs in enumerate(parlays, start=1):
        n = len(legs)
        # combined decimal odds (product)
        dec = 1.0
        prob = 1.0
        sample_date = ""
        games = []
        picks_desc = []

        for lg in legs:
            if not pd.isna(lg["OddsDec"]):
                dec *= lg["OddsDec"]
            if not pd.isna(lg["Prob"]):
                prob *= lg["Prob"]  # independence assumption
            games.append(f"{lg['Matchup']}")
            picks_desc.append(f"{lg['Market']}: {lg['Pick']}")
            if not sample_date:
                sample_date = lg["Date/Time"]

        us = decimal_to_american(dec) or ""
        records.append({
            "Parlay #": idx,
            "Legs": n,
            "Sample Date/Time": sample_date,
            "Games": " | ".join(games),
            "Picks": " | ".join(picks_desc),
            "Parlay Odds (Dec)": f"{dec:.2f}",
            "Parlay Odds (US)": f"{us}" if us != "" else "",
            "Est. Hit %": f"{prob*100.0:.1f}%",
        })

    disp = pd.DataFrame(records)
    return disp, parlays

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("TruLine – AI Picker (Top 5) + Parlay Lab")
st.caption("Pulls all supported sports, ranks the 5 best plays overall (ML/Spreads/Totals), and builds 5 parlays from high-probability legs.")

with st.sidebar:
    st.header("Settings")
    regions = st.text_input("Odds Regions", value=DEFAULT_REGIONS)
    # Let user toggle which sports to include today
    active_sports = st.multiselect(
        "Include Sports",
        list(SPORT_OPTIONS.keys()),
        default=list(SPORT_OPTIONS.keys())
    )
    # Button
    generate = st.button("Generate Today's Charts")

# Keep selections in session
if "active_sports" not in st.session_state:
    st.session_state.active_sports = list(SPORT_OPTIONS.keys())

if "last_results" not in st.session_state:
    st.session_state.last_results = {}  # stash anything if needed later

# ─────────────────────────────────────────────
# Fetch + Build once user clicks
# ─────────────────────────────────────────────
def fetch_all_selected(sport_map: Dict[str, Any], selected: List[str], regions: str) -> pd.DataFrame:
    frames = []
    for name in selected:
        key = sport_map[name]
        if isinstance(key, list):
            # soccer bundle
            for sk in key:
                df = fetch_odds(sk, regions)
                if not df.empty:
                    df["sport_key"] = sk
                    frames.append(df)
        else:
            df = fetch_odds(key, regions)
            if not df.empty:
                df["sport_key"] = key
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

if generate:
    st.session_state.active_sports = active_sports
    raw_all = fetch_all_selected(SPORT_OPTIONS, active_sports, regions)

    if raw_all.empty:
        st.warning("No odds returned. Try different regions or come back later (API may not have fresh markets yet).")
        st.session_state.has_data = False
    else:
        cons = build_consensus(raw_all)
        st.session_state.cons = cons
        st.session_state.has_data = True

# ─────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────
if st.session_state.get("has_data", False):
    cons = st.session_state.get("cons", pd.DataFrame())

    # AI Picker — Top 5
    st.subheader("🤖 AI Picker — Top 5 Plays (All Sports / ML • Spreads • Totals)")
    top5 = ai_top5_across_all(cons)
    if top5.empty:
        st.info("No qualifying plays found.")
    else:
        st.dataframe(top5, use_container_width=True, hide_index=True)

    st.divider()

    # Parlay Lab — 5 Parlays
    st.subheader("🎯 Parlay Lab — 5 Generated Parlays")
    parlays_df, _ = parlay_lab_5(cons)
    if parlays_df.empty:
        st.info("Not enough strong legs to build parlays right now.")
    else:
        st.dataframe(parlays_df, use_container_width=True, hide_index=True)

else:
    st.info("Pick your sports on the left and click **Generate Today’s Charts**.")
