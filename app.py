import os
from typing import Dict, Any, Optional, List, Tuple
import itertools
import math
import random

import requests
import pandas as pd
import numpy as np
import streamlit as st

# ─────────────────────────────────────────────
# HOW TO CONFIGURE
# ─────────────────────────────────────────────
# 1) Odds API (already in your project)
#    - Env var: ODDS_API_KEY, or edit the default below.
#
# 2) Props Provider (REAL props, since you chose Option A)
#    - Choose one in PROPS_PROVIDER: "playerprops", "rundown", or "apisports"
#    - Add PROPS_API_KEY to st.secrets or environment.
#    - If the provider limits fail or the key is missing, the app will skip props gracefully.
#
# Example (in Streamlit secrets):
#   [general]
#   ODDS_API_KEY = "YOUR_ODDS_KEY"
#   PROPS_PROVIDER = "playerprops"
#   PROPS_API_KEY = "YOUR_PLAYERPROPS_KEY"
#
# ─────────────────────────────────────────────

# Safe dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ─────────────────────────────────────────────
# Keys & Options
# ─────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", st.secrets.get("ODDS_API_KEY", "")) or "1d677dc98d978ccc24d9914d835442f1"
DEFAULT_REGIONS = os.getenv("REGIONS", st.secrets.get("REGIONS", "us"))

# Real props provider (since you chose A)
PROPS_PROVIDER = os.getenv("PROPS_PROVIDER", st.secrets.get("PROPS_PROVIDER", "playerprops")).lower()
PROPS_API_KEY = os.getenv("PROPS_API_KEY", st.secrets.get("PROPS_API_KEY", ""))

# If you ever want to try APISports again for anything else:
APISPORTS_KEY = os.getenv("APISPORTS_KEY", st.secrets.get("APISPORTS_KEY", ""))

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

# Parlay leg pattern you requested
PARLAY_LEG_PATTERN = [2, 2, 3, 5, 6]

# ─────────────────────────────────────────────
# Streamlit Setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker 🚀")
st.caption("Top 5 overall plays across all sports + 5 high-confidence parlays (with real player props, if configured).")
st.divider()

# ─────────────────────────────────────────────
# Helpers (odds, confidence, units, display)
# ─────────────────────────────────────────────
def american_to_decimal(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds): 
        return np.nan
    o = float(odds)
    return 1 + (o / 100.0) if o > 0 else 1 + (100.0 / abs(o))

def implied_prob_american(odds: Optional[float]) -> float:
    """Returns probability in decimal form [0..1]."""
    if odds is None or pd.isna(odds):
        return np.nan
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)

def confidence_pct_from_decimal(decimal_prob: float) -> float:
    """Convert decimal prob [0..1] to percentage [0..100]."""
    if decimal_prob is None or pd.isna(decimal_prob):
        return np.nan
    return float(decimal_prob) * 100.0

def units_U1(conf_pct: float) -> float:
    """
    You chose U1:
    units = max(0, (confidence_pct - 50) / 10), rounded to 1 decimal, capped at 5.
    """
    if conf_pct is None or pd.isna(conf_pct):
        return 0.0
    u = max(0.0, (conf_pct - 50.0) / 10.0)
    return round(min(u, 5.0), 1)

def fmt_pct(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{x:.1f}%"

def to_american(decimal_odds: float) -> Optional[int]:
    """Convert decimal odds to American odds."""
    if decimal_odds is None or pd.isna(decimal_odds) or decimal_odds <= 1:
        return None
    if decimal_odds >= 2.0:
        # positive odds
        return int(round((decimal_odds - 1.0) * 100))
    else:
        # negative odds
        return int(round(-100 / (decimal_odds - 1.0)))

# ─────────────────────────────────────────────
# Odds API fetch (ML/Spreads/Totals)
# ─────────────────────────────────────────────
def _odds_get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    if not ODDS_API_KEY:
        return None
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

@st.cache_data(ttl=90)
def fetch_odds_for_sport(sport_key: str, regions: str, markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    """
    Returns per-book offers (moneyline/spreads/totals) for a given sport key.
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
                mkey = mk.get("key")  # "h2h","spreads","totals"
                for oc in mk.get("outcomes", []):
                    rows.append({
                        "event_id": event_id,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": mkey,
                        "outcome": oc.get("name"),   # "Home"/"Away"/"Over"/"Under" etc
                        "line": oc.get("point"),     # may be None for ML
                        "odds_american": oc.get("price"),
                        "odds_decimal": american_to_decimal(oc.get("price")),
                        "conf_book": implied_prob_american(oc.get("price")),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
    try:
        if pd.api.types.is_datetime64tz_dtype(df["commence_time"]):
            df["Date/Time"] = df["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
        else:
            df["Date/Time"] = df["commence_time"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    except Exception:
        # Fallback if tz conversion fails for any reason
        df["Date/Time"] = df["commence_time"].dt.strftime("%b %d, %I:%M %p ET")
    return df

@st.cache_data(ttl=90)
def fetch_all_markets_across_sports(regions: str) -> pd.DataFrame:
    """
    Pulls ML/Spreads/Totals across ALL configured sports into one combined DataFrame.
    """
    frames = []
    for name, key in SPORT_OPTIONS.items():
        if isinstance(key, list):
            parts = [fetch_odds_for_sport(k, regions) for k in key]
            sport_df = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
        else:
            sport_df = fetch_odds_for_sport(key, regions)

        if not sport_df.empty:
            sport_df["sport_name"] = name
            frames.append(sport_df)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

# ─────────────────────────────────────────────
# Real Player Props — provider adapters
# ─────────────────────────────────────────────
def fetch_props_playerprops(regions: str) -> pd.DataFrame:
    """
    Placeholder for PlayerProps.io (example structure).
    Since third-party docs vary, this function is built to be fault-tolerant.
    Must return a DataFrame with columns:
      ['event_id','Date/Time','Matchup','market','prop_name','player','line',
       'odds_american','odds_decimal','conf_book','sport_name']
    If provider fails, returns empty DataFrame.
    """
    if not PROPS_API_KEY:
        return pd.DataFrame()

    try:
        # NOTE: Replace this placeholder endpoint & params with the actual PlayerProps.io endpoints you have access to.
        # This block is written to be easily swapped with your real call.
        url = "https://api.playerprops.io/v1/props"  # <— example, likely different
        headers = {"Authorization": f"Bearer {PROPS_API_KEY}"}
        params = {
            "region": regions,   # might differ
            "limit": 500
        }
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    # Transform into expected schema
    rows = []
    for item in data if isinstance(data, list) else []:
        try:
            event_id = item.get("event_id") or item.get("eventId")
            dt_iso = item.get("commence_time") or item.get("startTime")
            home = item.get("home_team") or item.get("homeTeam", "Unknown")
            away = item.get("away_team") or item.get("awayTeam", "Unknown")
            matchup = f"{home} vs {away}"
            market = item.get("market") or item.get("type", "player_prop")
            player = item.get("player") or item.get("athlete", "Unknown")
            line = item.get("line") or item.get("point")
            price = item.get("price") or item.get("odds_american")  # prefer american
            if price is None and "odds" in item and isinstance(item["odds"], dict):
                price = item["odds"].get("american")
            dec = american_to_decimal(price)
            conf = implied_prob_american(price)
            sport_n = item.get("sport") or item.get("sport_name") or "Unknown"

            # Date formatting
            dt = pd.to_datetime(dt_iso, errors="coerce")
            if pd.isna(dt):
                dt_str = ""
            else:
                try:
                    dt_str = dt.tz_convert("US/Eastern").strftime("%b %d, %I:%M %p ET") if pd.api.types.is_datetime64tz_dtype(pd.Series([dt])) else dt.tz_localize("UTC").tz_convert("US/Eastern").strftime("%b %d, %I:%M %p ET")
                except Exception:
                    dt_str = dt.strftime("%b %d, %I:%M %p ET")

            rows.append({
                "event_id": event_id,
                "Date/Time": dt_str,
                "Matchup": matchup,
                "market": "player_prop",
                "prop_name": market,
                "player": player,
                "line": line,
                "odds_american": price,
                "odds_decimal": dec,
                "conf_book": conf,
                "sport_name": sport_n
            })
        except Exception:
            continue

    return pd.DataFrame(rows)

def fetch_props_rundown(regions: str) -> pd.DataFrame:
    """
    Placeholder for TheRundown props endpoint.
    Must return same schema as fetch_props_playerprops().
    If not configured, returns empty DataFrame.
    """
    if not PROPS_API_KEY:
        return pd.DataFrame()

    try:
        # Placeholder endpoint – replace with real TheRundown props endpoint & params.
        url = "https://therundown-io.p.rapidapi.com/props"
        headers = {
            "X-RapidAPI-Key": PROPS_API_KEY,
            "X-RapidAPI-Host": "therundown-io.p.rapidapi.com"
        }
        params = {"region": regions, "limit": 500}
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    rows = []
    # Transform to schema
    for item in data.get("props", []) if isinstance(data, dict) else []:
        try:
            event_id = item.get("event_id")
            dt_iso = item.get("start_time")
            home = item.get("home_team", "Unknown")
            away = item.get("away_team", "Unknown")
            matchup = f"{home} vs {away}"
            player = item.get("player_name", "Unknown")
            market = item.get("market", "player_prop")
            line = item.get("line")
            price = item.get("american_odds")
            dec = american_to_decimal(price)
            conf = implied_prob_american(price)
            sport_n = item.get("sport", "Unknown")

            dt = pd.to_datetime(dt_iso, errors="coerce")
            if pd.isna(dt):
                dt_str = ""
            else:
                try:
                    dt_str = dt.tz_convert("US/Eastern").strftime("%b %d, %I:%M %p ET") if pd.api.types.is_datetime64tz_dtype(pd.Series([dt])) else dt.tz_localize("UTC").tz_convert("US/Eastern").strftime("%b %d, %I:%M %p ET")
                except Exception:
                    dt_str = dt.strftime("%b %d, %I:%M %p ET")

            rows.append({
                "event_id": event_id,
                "Date/Time": dt_str,
                "Matchup": matchup,
                "market": "player_prop",
                "prop_name": market,
                "player": player,
                "line": line,
                "odds_american": price,
                "odds_decimal": dec,
                "conf_book": conf,
                "sport_name": sport_n
            })
        except Exception:
            continue

    return pd.DataFrame(rows)

def fetch_props_apisports(regions: str) -> pd.DataFrame:
    """
    Placeholder for API-Sports props (only if your tier supports it).
    Returns same schema; otherwise empty.
    """
    if not PROPS_API_KEY and not APISPORTS_KEY:
        return pd.DataFrame()

    try:
        # Placeholder – update with your actual API-Sports props endpoints & headers.
        url = "https://v1.apisports.io/odds/props"  # likely different
        headers = {"x-apisports-key": APISPORTS_KEY or PROPS_API_KEY}
        params = {"region": regions}
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    rows = []
    # Transform if you have the real schema
    for item in data.get("response", []) if isinstance(data, dict) else []:
        try:
            event_id = item.get("event_id")
            dt_iso = item.get("date")
            home = item.get("home", "Unknown")
            away = item.get("away", "Unknown")
            matchup = f"{home} vs {away}"
            player = item.get("player", "Unknown")
            market = item.get("market", "player_prop")
            line = item.get("line")
            price = item.get("odds_american")
            dec = american_to_decimal(price)
            conf = implied_prob_american(price)
            sport_n = item.get("sport", "Unknown")

            dt = pd.to_datetime(dt_iso, errors="coerce")
            if pd.isna(dt):
                dt_str = ""
            else:
                try:
                    dt_str = dt.tz_convert("US/Eastern").strftime("%b %d, %I:%M %p ET") if pd.api.types.is_datetime64tz_dtype(pd.Series([dt])) else dt.tz_localize("UTC").tz_convert("US/Eastern").strftime("%b %d, %I:%M %p ET")
                except Exception:
                    dt_str = dt.strftime("%b %d, %I:%M %p ET")

            rows.append({
                "event_id": event_id,
                "Date/Time": dt_str,
                "Matchup": matchup,
                "market": "player_prop",
                "prop_name": market,
                "player": player,
                "line": line,
                "odds_american": price,
                "odds_decimal": dec,
                "conf_book": conf,
                "sport_name": sport_n
            })
        except Exception:
            continue

    return pd.DataFrame()

@st.cache_data(ttl=90)
def fetch_player_props(regions: str) -> pd.DataFrame:
    """
    Calls the chosen props provider adapter and returns normalized props DataFrame.
    """
    provider = PROPS_PROVIDER
    if provider == "playerprops":
        return fetch_props_playerprops(regions)
    elif provider == "rundown":
        return fetch_props_rundown(regions)
    elif provider == "apisports":
        return fetch_props_apisports(regions)
    else:
        return pd.DataFrame()

# ─────────────────────────────────────────────
# Consensus logic for ML/Spreads/Totals
# ─────────────────────────────────────────────
def build_consensus(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    idx_best = raw.groupby(["event_id", "market", "outcome", "line"])["odds_decimal"].idxmax()
    best = raw.loc[idx_best, ["event_id", "market", "outcome", "line", "odds_american", "odds_decimal", "book"]]
    best = best.rename(columns={"odds_american":"best_odds_us","odds_decimal":"best_odds_dec","book":"best_book"})

    agg = raw.groupby(["event_id","market","outcome","line"], dropna=False).agg(
        consensus_conf=("conf_book","mean"),
        books=("book","nunique"),
        home_team=("home_team","first"),
        away_team=("away_team","first"),
        commence_time=("commence_time","first"),
        date_time=("Date/Time","first"),
        sport=("sport_name","first")
    ).reset_index()

    out = agg.merge(best, on=["event_id","market","outcome","line"], how="left")
    out["Matchup"] = out["home_team"] + " vs " + out["away_team"]
    out["ConfidencePct"] = out["consensus_conf"].apply(lambda x: x * 100.0 if pd.notna(x) else np.nan)
    out["Odds (US)"] = out["best_odds_us"]
    out["Odds (Dec)"] = out["best_odds_dec"]
    out["Sportsbook"] = out["best_book"]
    out["Date/Time"] = out["date_time"]
    out["Sport"] = out["sport"]
    return out[[
        "event_id","commence_time","Date/Time","Matchup","Sport","market","outcome","line",
        "Sportsbook","Odds (US)","Odds (Dec)","ConfidencePct","books"
    ]].rename(columns={"books":"Books"})

def labeled_pick_row(df_row: pd.Series) -> Dict[str, Any]:
    """
    Convert consensus row -> unified pick object used in AI picker & parlays.
    """
    market = df_row.get("market")
    outcome = str(df_row.get("outcome"))
    line = df_row.get("line")
    odds_us = df_row.get("Odds (US)")
    odds_dec = df_row.get("Odds (Dec)")
    conf_pct = df_row.get("ConfidencePct")
    dt = df_row.get("Date/Time")
    matchup = df_row.get("Matchup")
    sport = df_row.get("Sport")
    if market == "h2h":
        pick_label = f"{outcome} ML"
    elif market == "totals":
        pick_label = f"{outcome} {line}"
    elif market == "spreads":
        try:
            ln = float(line)
            sign = "+" if ln > 0 else ""
            pick_label = f"{outcome} {sign}{ln}"
        except Exception:
            pick_label = f"{outcome} {line}"
    else:
        pick_label = f"{outcome}"

    return {
        "Date/Time": dt,
        "Sport": sport,
        "Matchup": matchup,
        "Market": {"h2h":"Moneyline","spreads":"Spreads","totals":"Totals"}.get(market, market),
        "Pick": pick_label,
        "Line": line,
        "Odds (US)": odds_us,
        "Odds (Dec)": odds_dec,
        "Confidence %": conf_pct,
        "event_id": df_row.get("event_id")
    }

def pick_best_per_event(cons: pd.DataFrame, market_key: str, top_n: int) -> List[Dict[str, Any]]:
    sub = cons[cons["market"] == market_key].copy()
    if sub.empty:
        return []
    # best per event
    best_idx = sub.groupby("event_id")["ConfidencePct"].idxmax()
    sub = sub.loc[best_idx].copy()
    sub = sub.sort_values("commence_time", ascending=True).head(top_n)
    # Convert to unified dict rows
    return [labeled_pick_row(r) for _, r in sub.iterrows()]

# ─────────────────────────────────────────────
# Combining ML/Spreads/Totals + REAL PROPS for AI TOP 5
# ─────────────────────────────────────────────
def normalize_props_rows(props_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert props df into same unified pick format as labeled_pick_row().
    """
    if props_df is None or props_df.empty:
        return []
    rows = []
    for _, r in props_df.iterrows():
        odds_us = r.get("odds_american")
        odds_dec = r.get("odds_decimal")
        conf = r.get("conf_book")
        conf_pct = confidence_pct_from_decimal(conf)
        player = r.get("player", "Unknown")
        prop_name = r.get("prop_name", "Prop")
        line = r.get("line")
        matchup = r.get("Matchup", "")
        dt = r.get("Date/Time", "")
        sport = r.get("sport_name", "Unknown")

        label = f"{player} {prop_name} {line}" if pd.notna(line) else f"{player} {prop_name}"
        rows.append({
            "Date/Time": dt,
            "Sport": sport,
            "Matchup": matchup,
            "Market": "Player Prop",
            "Pick": label,
            "Line": line,
            "Odds (US)": odds_us,
            "Odds (Dec)": odds_dec,
            "Confidence %": conf_pct,
            "event_id": r.get("event_id")
        })
    return rows

def make_ai_top5(all_markets_df: pd.DataFrame, regions: str) -> pd.DataFrame:
    """
    Build consensus on ML/Spreads/Totals across all sports; fold in real props (if fetched);
    rank by confidence% and return the overall top 5 with U1 units.
    """
    if all_markets_df is None or all_markets_df.empty:
        return pd.DataFrame(columns=["Date/Time","Sport","Matchup","Market","Pick","Line","Odds (US)","Odds (Dec)","Confidence %","Units"])

    cons = build_consensus(all_markets_df)
    # get best moneyline/spreads/totals
    ml_rows = pick_best_per_event(cons, "h2h", 50)
    tot_rows = pick_best_per_event(cons, "totals", 50)
    spr_rows = pick_best_per_event(cons, "spreads", 50)

    # props (real, from provider)
    props_df = fetch_player_props(regions)
    props_rows = normalize_props_rows(props_df) if props_df is not None else []

    # Combine, score, select top 5 (dedupe by event & pick label)
    combined = ml_rows + tot_rows + spr_rows + props_rows
    if not combined:
        return pd.DataFrame(columns=["Date/Time","Sport","Matchup","Market","Pick","Line","Odds (US)","Odds (Dec)","Confidence %","Units"])

    # DataFrame for sorting
    df = pd.DataFrame(combined)
    df["Confidence %"] = pd.to_numeric(df["Confidence %"], errors="coerce")
    df = df.dropna(subset=["Confidence %"])
    # Deduplicate on event_id + Pick label
    if "event_id" in df.columns:
        df["_k"] = df["event_id"].astype(str) + " | " + df["Pick"].astype(str)
        df = df.drop_duplicates("_k")
    df = df.sort_values("Confidence %", ascending=False).head(5)

    # Apply U1 units
    df["Units"] = df["Confidence %"].apply(units_U1)
    # nice formatting
    # ensure Odds (Dec) present if missing
    if "Odds (Dec)" not in df.columns or df["Odds (Dec)"].isna().all():
        df["Odds (Dec)"] = df["Odds (US)"].apply(american_to_decimal)

    # Reorder columns
    cols = ["Date/Time","Sport","Matchup","Market","Pick","Line","Odds (US)","Odds (Dec)","Confidence %","Units"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].reset_index(drop=True)

# ─────────────────────────────────────────────
# Parlay builder
# ─────────────────────────────────────────────
def parlay_combined_odds(legs: List[Dict[str, Any]]) -> Tuple[float, Optional[int], float]:
    """
    Multiply decimal odds, return (decimal, american, implied_prob%).
    """
    decs = [float(american_to_decimal(l.get("Odds (US)"))) if pd.notna(l.get("Odds (US)")) else float(l.get("Odds (Dec)", 0)) for l in legs]
    decs = [d for d in decs if d and d > 1.0]
    if not decs:
        return (np.nan, None, np.nan)

    dec_product = 1.0
    for d in decs:
        dec_product *= d

    am = to_american(dec_product)
    implied = 100.0 / dec_product  # naive independence assumption (not strictly true, but standard)
    return (dec_product, am, implied)

def candidate_pool_for_parlays(all_markets_df: pd.DataFrame, regions: str, max_pool: int = 60) -> List[Dict[str, Any]]:
    """
    Build a candidate pool of legs from:
      - best ML/Spreads/Totals per event across sports (top ~50 each)
      - player props from provider (top ~30 by confidence)
    Returns list of unified pick dicts.
    """
    if all_markets_df is None or all_markets_df.empty:
        return []

    cons = build_consensus(all_markets_df)
    ml = pick_best_per_event(cons, "h2h", 80)
    tot = pick_best_per_event(cons, "totals", 80)
    spr = pick_best_per_event(cons, "spreads", 80)

    props_df = fetch_player_props(regions)
    props_rows = normalize_props_rows(props_df)
    # Rank props by confidence
    pr_df = pd.DataFrame(props_rows)
    if not pr_df.empty:
        pr_df["Confidence %"] = pd.to_numeric(pr_df["Confidence %"], errors="coerce")
        pr_df = pr_df.dropna(subset=["Confidence %"]).sort_values("Confidence %", ascending=False).head(40)
        props_rows = pr_df.to_dict("records")

    combined = ml + tot + spr + props_rows
    # Keep best by confidence, dedupe by event_id+Pick
    cdf = pd.DataFrame(combined)
    if cdf.empty:
        return []

    # Build decimal odds column if needed
    if "Odds (Dec)" not in cdf.columns or cdf["Odds (Dec)"].isna().all():
        cdf["Odds (Dec)"] = cdf["Odds (US)"].apply(american_to_decimal)

    if "event_id" in cdf.columns:
        cdf["_k"] = cdf["event_id"].astype(str) + " | " + cdf["Pick"].astype(str)
        cdf = cdf.drop_duplicates("_k")

    cdf["Confidence %"] = pd.to_numeric(cdf["Confidence %"], errors="coerce")
    cdf = cdf.dropna(subset=["Confidence %"])
    cdf = cdf.sort_values("Confidence %", ascending=False).head(max_pool)
    return cdf.to_dict("records")

def assemble_parlays(pool: List[Dict[str, Any]], pattern: List[int]) -> List[Dict[str, Any]]:
    """
    Greedy assembly:
      - Avoid duplicate event_id in the same parlay
      - Prefer higher-confidence legs
    Returns a list of parlay dicts with fields:
      {'Name','Legs','Parlay Dec','Parlay US','Implied %'}
    """
    parlays = []
    # Index by event to avoid conflicts
    by_event = {}
    for leg in pool:
        ev = str(leg.get("event_id"))
        by_event.setdefault(ev, []).append(leg)

    # Flatten a "ranked list" by confidence:
    ranked = sorted(pool, key=lambda x: (x.get("Confidence %") or 0), reverse=True)

    used_leg_keys = set()
    def leg_key(l): 
        return f"{l.get('event_id')}|{l.get('Pick')}|{l.get('Market')}"

    for idx, size in enumerate(pattern, start=1):
        legs = []
        seen_events = set()
        for cand in ranked:
            if len(legs) >= size:
                break
            ev = str(cand.get("event_id"))
            k = leg_key(cand)
            if ev in seen_events:
                continue
            if k in used_leg_keys:
                continue
            # must have odds
            dec = cand.get("Odds (Dec)")
            if dec is None or pd.isna(dec) or dec <= 1.0:
                continue
            legs.append(cand)
            seen_events.add(ev)

        # fallback if we couldn't fill
        if len(legs) < size and ranked:
            for cand in ranked:
                if len(legs) >= size:
                    break
                k = leg_key(cand)
                if k in used_leg_keys:
                    continue
                dec = cand.get("Odds (Dec)")
                if dec is None or pd.isna(dec) or dec <= 1.0:
                    continue
                legs.append(cand)

        # compute odds
        dec_odds, am_odds, implied = parlay_combined_odds(legs)
        # register legs as used
        for l in legs:
            used_leg_keys.add(leg_key(l))

        parlays.append({
            "Name": f"Parlay #{idx} – {size} legs",
            "Legs": legs,
            "Parlay Dec": dec_odds,
            "Parlay US": am_odds,
            "Implied %": implied
        })

    return parlays

# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────
with st.sidebar:
    regions = st.text_input("Odds Regions", value=DEFAULT_REGIONS)
    st.caption("Examples: us, eu, uk, au. Comma-separate for multiple.")
    run_btn = st.button("Generate Today's Top 5 + Parlays")

# ─────────────────────────────────────────────
# Main Run
# ─────────────────────────────────────────────
if run_btn:
    with st.spinner("Fetching odds & props..."):
        all_markets = fetch_all_markets_across_sports(regions)

        if all_markets.empty:
            st.error("No odds returned. Try a different region or check your Odds API quota.")
        else:
            # AI TOP 5 (any sport, ML/Spread/Total/Props)
            top5 = make_ai_top5(all_markets, regions)

            st.subheader("🤖 AI Picker — Top 5 Overall Plays")
            if top5.empty:
                st.info("No picks available.")
            else:
                # Units already computed via U1
                show = top5.copy()
                show["Confidence %"] = show["Confidence %"].apply(lambda x: f"{x:.1f}%")
                st.dataframe(show, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Parlays
            st.subheader("🎯 5 Suggested Parlays (2, 2, 3, 5, 6 legs)")
            pool = candidate_pool_for_parlays(all_markets, regions, max_pool=80)
            if not pool:
                st.info("No candidate legs available for parlays.")
            else:
                parlays = assemble_parlays(pool, PARLAY_LEG_PATTERN)
                for p in parlays:
                    with st.expander(f"{p['Name']} — Implied {p['Implied %']:.1f}% — Parlay Odds (US) {p['Parlay US']}", expanded=False):
                        legs_df = pd.DataFrame(p["Legs"])
                        # Tidy columns for display
                        keep = ["Date/Time","Sport","Matchup","Market","Pick","Odds (US)","Odds (Dec)","Confidence %"]
                        for c in keep:
                            if c not in legs_df.columns:
                                legs_df[c] = ""
                        legs_df = legs_df[keep].copy()
                        legs_df["Confidence %"] = legs_df["Confidence %"].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(legs_df, use_container_width=True, hide_index=True)
                        st.caption(f"Parlay Decimal Odds: {p['Parlay Dec']:.3f} | American: {p['Parlay US']} | Implied: {p['Implied %']:.1f}%")

            # Helpful notice about props provider
            st.markdown("---")
            if not PROPS_API_KEY:
                st.warning(
                    "Real player props are enabled, but no PROPS_API_KEY was found. "
                    "Add PROPS_API_KEY in your environment or Streamlit secrets to include real props in Top 5 and Parlays."
                )
            else:
                st.success(f"Props provider: **{PROPS_PROVIDER}** (API key detected).")

else:
    st.info("Click **Generate Today's Top 5 + Parlays** to build the two charts.")
