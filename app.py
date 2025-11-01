import os
from typing import Dict, Any, Optional, List, Tuple
import math
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

# ─────────────────────────────────────────────
# Config & Keys
# ─────────────────────────────────────────────
st.set_page_config(page_title="TruLine – AI Picks & Parlays", layout="wide")
st.title("TruLine – AI Picks & Parlays 🚀")
st.caption("Two charts only: AI Top 5 across all sports, plus 5 Parlays (ensemble-ranked).")
st.divider()

# ✅ Odds API Key (hardcoded here per your prior examples)
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "") or "1d677dc98d978ccc24d9914d835442f1"
DEFAULT_REGIONS = os.getenv("REGIONS", "us")

# Markets we’ll use
MARKETS = "h2h,spreads,totals"

# Sports enabled (you can change this list)
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

# Odds sanity filter (you asked to avoid crazy prices like -30000)
ODDS_MIN_US = -2000
ODDS_MAX_US = 2000

# File outputs
CSV_FILE = "bets.csv"
EXCEL_FILE = "bets.xlsx"

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
    return "" if (x is None or pd.isna(x)) else f"{100.0 * x:.1f}%"

def assign_units_from_score(score: float) -> float:
    """
    Map ensemble score (0..1-ish) to units ~0.5..5.0.
    """
    score = max(0.0, min(1.0, score))
    return round(0.5 + 4.5 * score, 1)

# ─────────────────────────────────────────────
# Odds API
# ─────────────────────────────────────────────
def _odds_get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY.")
        return None
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except Exception:
        return None

@st.cache_data(ttl=120)
def fetch_odds(sport_key: str, regions: str, markets: str = MARKETS) -> pd.DataFrame:
    """
    Fetch odds for a single sport key.
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
                mkey = mk.get("key")  # "h2h" | "spreads" | "totals"
                for oc in mk.get("outcomes", []):
                    rows.append({
                        "sport_key": sport_key,
                        "event_id": event_id,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": mkey,
                        "outcome": oc.get("name"),   # "Home"/"Away" or "Over"/"Under"
                        "line": oc.get("point"),     # spread or total line; None for ML
                        "odds_american": oc.get("price"),
                        "odds_decimal": american_to_decimal(oc.get("price")),
                        "conf_book": implied_prob_american(oc.get("price")),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Timestamp
    df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(df["commence_time"]):
        df["Date/Time"] = df["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    else:
        df["Date/Time"] = df["commence_time"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    return df

def fetch_all_enabled_sports(regions: str) -> pd.DataFrame:
    """
    Fetch odds across all configured sports (flattening Soccer leagues).
    """
    frames = []
    for label, s in SPORT_OPTIONS.items():
        if isinstance(s, list):
            for sub in s:
                df = fetch_odds(sub, regions)
                if not df.empty:
                    df["sport_label"] = label
                    frames.append(df)
        else:
            df = fetch_odds(s, regions)
            if not df.empty:
                df["sport_label"] = label
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out

# ─────────────────────────────────────────────
# Consensus + Ensemble
# ─────────────────────────────────────────────
def build_consensus(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Consensus per (event, market, outcome, line). Keep best price and aggregates.
    """
    if raw.empty:
        return raw

    # best price row index per group
    idx_best = raw.groupby(["event_id", "market", "outcome", "line"])["odds_decimal"].idxmax()

    best = raw.loc[idx_best, ["event_id", "market", "outcome", "line",
                              "odds_american", "odds_decimal", "book"]]
    best = best.rename(columns={
        "odds_american": "best_odds_us",
        "odds_decimal": "best_odds_dec",
        "book": "best_book"
    })

    agg = raw.groupby(["event_id", "market", "outcome", "line"], dropna=False).agg(
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

    out = agg.merge(best, on=["event_id", "market", "outcome", "line"], how="left")
    out["Matchup"] = out["home_team"] + " vs " + out["away_team"]
    out["Date/Time"] = out["date_time"]

    # filter insane odds by US range
    out = out[(out["best_odds_us"] >= ODDS_MIN_US) & (out["best_odds_us"] <= ODDS_MAX_US)]

    return out[[
        "sport", "sport_key", "event_id", "commence_time", "Date/Time", "Matchup",
        "market", "outcome", "line",
        "best_book", "best_odds_us", "best_odds_dec", "consensus_conf", "books", "avg_odds_dec"
    ]]

def ensemble_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Four simple voters over consensus features:
      V1: Probability voter — higher implied prob (consensus_conf)
      V2: Best-price voter — better best price vs group avg (edge)
      V3: Market depth voter — more books = more confidence
      V4: Balance voter — best_odds_dec nearer to 2.0 (balanced) rather than extreme (optional)
    We normalize each 0..1, average them.
    """
    d = df.copy()

    # V1: consensus_conf (already in 0..1-ish)
    v1 = d["consensus_conf"].astype(float).fillna(0.0)
    v1 = v1.clip(0.0, 1.0)

    # V2: edge vs avg price => (best_odds_dec - avg_odds_dec) / avg_odds_dec, then normalize
    edge = (d["best_odds_dec"] - d["avg_odds_dec"]).astype(float)
    v2 = 1.0 / (1.0 + np.exp(-6.0 * edge.fillna(0.0)))  # edge>0 better → v2>0.5

    # V3: books count normalized by 10 (cap)
    v3 = (d["books"].astype(float).clip(lower=0.0, upper=10.0)) / 10.0

    # V4: balance voter – reward dec odds around ~2.0 (evens). Use Gaussian centered at 2.0.
    dec = d["best_odds_dec"].astype(float).fillna(2.0)
    v4 = np.exp(-((dec - 2.0) ** 2) / (2 * (0.6 ** 2)))  # σ ≈ 0.6
    v4 = (v4 - v4.min()) / (v4.max() - v4.min() + 1e-9)

    d["V1_prob"] = v1
    d["V2_edge"] = v2
    d["V3_depth"] = v3
    d["V4_balance"] = v4

    d["EnsembleScore"] = (d["V1_prob"] + d["V2_edge"] + d["V3_depth"] + d["V4_balance"]) / 4.0
    d["Units"] = d["EnsembleScore"].apply(assign_units_from_score)
    return d

def pool_candidates(cons: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only ML/Totals/Spreads and dedupe by (event, market) to best outcome by score later.
    """
    valid = cons[cons["market"].isin(["h2h", "totals", "spreads"])].copy()
    if valid.empty:
        return valid
    scored = ensemble_score(valid)

    # For each (event, market), keep the higher-scored outcome (e.g., Home vs Away; Over vs Under)
    idx = scored.groupby(["event_id", "market"])["EnsembleScore"].idxmax()
    best = scored.loc[idx].copy()

    # Nice display columns
    best["Pick"] = best["outcome"]

    def fmt_line(row):
        if row["market"] == "spreads":
            try:
                ln = float(row["line"])
                return f"{'+' if ln>0 else ''}{ln:.1f}"
            except Exception:
                return str(row["line"])
        elif row["market"] == "totals":
            return f"{row['outcome']} ({row['line']})"
        else:
            return ""
    best["Line"] = best.apply(fmt_line, axis=1)

    # Human market names
    market_map = {"h2h": "Moneyline", "spreads": "Spreads", "totals": "Totals"}
    best["Market"] = best["market"].map(market_map)

    # Ensure both 'sport' and 'Sport' exist so later code (parlays) can use either
    if "sport" in best.columns:
        best["Sport"] = best["sport"]

    # Order by ensemble score descending
    best = best.sort_values("EnsembleScore", ascending=False)
    return best

def select_top_ai(best_pool: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Select top-k across all sports/markets after odds filter.
    """
    if best_pool.empty:
        return pd.DataFrame()
    # Enforce odds filter again just in case:
    filtered = best_pool[(best_pool["best_odds_us"] >= ODDS_MIN_US) & (best_pool["best_odds_us"] <= ODDS_MAX_US)].copy()
    top = filtered.head(k).copy()

    # Format final display columns
    top["Confidence"] = top["consensus_conf"].apply(fmt_pct)
    # Odds may be floats/NaN – make them clean
    def safe_int(x):
        try:
            return int(round(float(x)))
        except Exception:
            return ""
    top["Odds (US)"] = top["best_odds_us"].apply(safe_int)
    top["Odds (Dec)"] = top["best_odds_dec"].round(3)
    top = top.rename(columns={"sport": "Sport", "best_book": "Sportsbook"})
    return top[[
        "Date/Time", "Sport", "Matchup", "Market", "Pick", "Line",
        "Sportsbook", "Odds (US)", "Odds (Dec)", "Confidence", "Units",
        "EnsembleScore", "event_id", "market"
    ]].reset_index(drop=True)

# ─────────────────────────────────────────────
# Parlays
# ─────────────────────────────────────────────
def build_parlays(candidate_pool: pd.DataFrame, leg_plan: List[int]) -> List[Dict[str, Any]]:
    """
    Build 5 parlays with leg lengths in leg_plan (e.g., [2,2,3,5,6]).
    Ensures no duplicate event_ids within the same parlay.
    Tries to pick highest-scoring remaining legs greedily.
    """
    parlays = []
    if candidate_pool.empty:
        return parlays

    # Work from a larger pool to avoid conflicts
    pool = candidate_pool.copy()

    for legs in leg_plan:
        used_events = set()
        legs_rows = []
        for _, r in pool.iterrows():
            if len(legs_rows) >= legs:
                break
            if r["event_id"] in used_events:
                continue
            legs_rows.append(r)
            used_events.add(r["event_id"])

        # If we couldn't get enough legs, just skip this parlay
        if len(legs_rows) < legs:
            continue

        # Compute parlay decimal odds & projected units (sum of units * mild bonus)
        dec_odds = 1.0
        total_units = 0.0
        for rr in legs_rows:
            d = rr["best_odds_dec"]
            if pd.isna(d) or d <= 1.0:
                d = max(1.01, float(d) if not pd.isna(d) else 1.01)
            dec_odds *= d
            total_units += float(rr["Units"])

        am = decimal_to_american(dec_odds)
        parlay = {
            "legs": legs_rows,
            "legs_count": legs,
            "parlay_decimal": round(dec_odds, 4),
            "parlay_american": am,
            "suggested_units": round(max(0.5, min(5.0, total_units * 0.4)), 1)  # scale down risk
        }
        parlays.append(parlay)

        # Remove used event_ids from pool so next parlay mixes differently
        pool = pool[~pool["event_id"].isin(used_events)].copy()
        if pool.empty:
            break

    return parlays

def parlays_to_dataframe(parlays: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten parlays into a nice table for display and export.
    Tolerant to column-name casing (Sport vs sport, Market vs market).
    """
    rows = []
    for idx, p in enumerate(parlays, start=1):
        legs_rows = p["legs"]
        leg_summaries = []
        for rr in legs_rows:
            # Accept both 'Sport' and 'sport', 'Market' and 'market'
            sport_val = rr["Sport"] if "Sport" in rr and pd.notna(rr["Sport"]) else (rr["sport"] if "sport" in rr else "")
            market_val = rr["Market"] if "Market" in rr and pd.notna(rr["Market"]) else (rr["market"] if "market" in rr else "")
            matchup_val = rr["Matchup"] if "Matchup" in rr else ""
            pick_val = rr["Pick"] if "Pick" in rr else rr.get("outcome", "")
            line_val = rr["Line"] if "Line" in rr else rr.get("line", "")

            # Odds for leg: prefer 'Odds (US)' if present, else fallback to best_odds_us
            if "Odds (US)" in rr and str(rr["Odds (US)"]).strip() != "":
                try:
                    odds_us = int(round(float(rr["Odds (US)"])))
                except Exception:
                    odds_us = ""
            else:
                try:
                    odds_us = int(round(float(rr.get("best_odds_us", ""))))
                except Exception:
                    odds_us = ""

            part = f"{sport_val} • {matchup_val} • {market_val}: {pick_val}"
            if str(line_val).strip() not in ["", "None", "nan"] and market_val in ["Spreads", "Totals"]:
                part += f" ({line_val})"
            if odds_us != "":
                part += f" @ {odds_us}"
            leg_summaries.append(part)

        rows.append({
            "Parlay #": idx,
            "Legs": p["legs_count"],
            "Leg Details": "  |  ".join(leg_summaries),
            "Parlay Odds (Dec)": p["parlay_decimal"],
            "Parlay Odds (US)": p["parlay_american"],
            "Suggested Units": p["suggested_units"]
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# Persistence (CSV + Excel)
# ─────────────────────────────────────────────
def load_results() -> pd.DataFrame:
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        # backfill columns
        for col in ["Sport", "Market", "Date/Time", "Matchup", "Pick", "Line",
                    "Odds (US)", "Units", "Result"]:
            if col not in df.columns:
                df[col] = "" if col not in ["Units"] else 1.0
        df["Result"] = df["Result"].fillna("Pending")
        return df
    return pd.DataFrame(columns=[
        "Sport", "Market", "Date/Time", "Matchup", "Pick", "Line",
        "Odds (US)", "Units", "Result"
    ])

def save_results_csv(df: pd.DataFrame):
    df.to_csv(CSV_FILE, index=False)

def auto_log_ai_picks(ai5: pd.DataFrame):
    """
    Append AI picks to CSV (dedup-safe).
    """
    if ai5 is None or ai5.empty:
        return
    results = load_results()
    for _, r in ai5.iterrows():
        entry = {
            "Sport": r.get("Sport", ""),
            "Market": r.get("Market", ""),
            "Date/Time": r.get("Date/Time", ""),
            "Matchup": r.get("Matchup", ""),
            "Pick": r.get("Pick", ""),
            "Line": r.get("Line", ""),
            "Odds (US)": r.get("Odds (US)", ""),
            "Units": float(r.get("Units", 1.0)) if str(r.get("Units", "")).strip() != "" else 1.0,
            "Result": "Pending"
        }
        dup = (
            (results["Sport"] == entry["Sport"]) &
            (results["Market"] == entry["Market"]) &
            (results["Date/Time"] == entry["Date/Time"]) &
            (results["Matchup"] == entry["Matchup"]) &
            (results["Pick"] == entry["Pick"]) &
            (results["Line"].fillna("").astype(str) == str(entry["Line"]))
        )
        if not dup.any():
            results = pd.concat([results, pd.DataFrame([entry])], ignore_index=True)
    save_results_csv(results)

def export_excel(ai5: pd.DataFrame, parlays_df: pd.DataFrame, results_df: pd.DataFrame) -> bytes:
    """
    Write an Excel workbook with 3 sheets:
      - AI_Picks
      - Parlays
      - Results
    Returns the binary content.
    """
    try:
        with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
            # Sheet 1: AI picks
            ai_out = ai5.copy()
            if not ai_out.empty:
                ai_out = ai_out.rename(columns={
                    "Odds (US)": "Odds_US",
                    "Odds (Dec)": "Odds_Dec",
                })
                if "Result" not in ai_out.columns:
                    ai_out["Result"] = ""
                ai_out.to_excel(writer, index=False, sheet_name="AI_Picks")
            else:
                pd.DataFrame(columns=["Message"]).to_excel(writer, index=False, sheet_name="AI_Picks")

            # Sheet 2: Parlays
            if parlays_df is not None and not parlays_df.empty:
                parlays_df.to_excel(writer, index=False, sheet_name="Parlays")
            else:
                pd.DataFrame(columns=["Message"]).to_excel(writer, index=False, sheet_name="Parlays")

            # Sheet 3: Results (running log)
            results_df.to_excel(writer, index=False, sheet_name="Results")

        with open(EXCEL_FILE, "rb") as f:
            return f.read()
    except Exception:
        from io import BytesIO
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            a = ai5.copy()
            if not a.empty:
                if "Result" not in a.columns:
                    a["Result"] = ""
                a.to_excel(writer, index=False, sheet_name="AI_Picks")
            else:
                pd.DataFrame(columns=["Message"]).to_excel(writer, index=False, sheet_name="AI_Picks")

            if parlays_df is not None and not parlays_df.empty:
                parlays_df.to_excel(writer, index=False, sheet_name="Parlays")
            else:
                pd.DataFrame(columns=["Message"]).to_excel(writer, index=False, sheet_name="Parlays")

            results_df.to_excel(writer, index=False, sheet_name="Results")
        bio.seek(0)
        return bio.read()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.subheader("Controls")
    regions = st.text_input("Regions", value=DEFAULT_REGIONS)
    top_k = st.slider("AI Picks (Top K)", 5, 10, 5)
    parlay_plan = st.text_input("Parlay leg plan (comma separated)", value="2,2,3,5,6")
    leg_plan = [int(x.strip()) for x in parlay_plan.split(",") if x.strip().isdigit()]
    fetch = st.button("Generate Today's Picks")

# ─────────────────────────────────────────────
# Generate & Store in Session
# ─────────────────────────────────────────────
if fetch:
    raw_all = fetch_all_enabled_sports(regions)
    if raw_all.empty:
        st.warning("No odds returned. Try again later or adjust regions.")
        st.session_state.has_data = False
    else:
        cons = build_consensus(raw_all)
        pool = pool_candidates(cons)
        ai5 = select_top_ai(pool, k=top_k)

        # log AI picks to CSV (dedup-safe)
        auto_log_ai_picks(ai5)

        # build parlays from a slightly larger pool (top 40) to have variety
        pool_for_parlays = pool.head(40).copy()
        parlays = build_parlays(pool_for_parlays, leg_plan=leg_plan if leg_plan else [2,2,3,5,6])
        parlays_df = parlays_to_dataframe(parlays)

        st.session_state.cons = cons
        st.session_state.pool = pool
        st.session_state.ai5 = ai5
        st.session_state.parlays = parlays
        st.session_state.parlays_df = parlays_df
        st.session_state.has_data = True

# ─────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────
if st.session_state.get("has_data", False):
    tabs = st.tabs([
        "🤖 AI Top 5",
        "🎯 Parlays (5)",
        "📥 Export"
    ])

    # Tab 0 — AI Top 5
    with tabs[0]:
        st.subheader("AI Top 5 — Ensemble Ranked (across all sports & markets)")
        if st.session_state.ai5 is not None and not st.session_state.ai5.empty:
            show = st.session_state.ai5.copy()
            # prettier display
            show = show.rename(columns={
                "EnsembleScore": "Score",
            })
            show["Score"] = show["Score"].map(lambda x: f"{x:.3f}")
            st.dataframe(show.drop(columns=["event_id", "market"]), use_container_width=True, hide_index=True)
        else:
            st.info("No AI picks available.")

    # Tab 1 — Parlays
    with tabs[1]:
        st.subheader("Parlays (ensemble-built)")
        if st.session_state.parlays_df is not None and not st.session_state.parlays_df.empty:
            st.dataframe(st.session_state.parlays_df, use_container_width=True, hide_index=True)
        else:
            st.info("No parlays could be built from the current pool.")

    # Tab 2 — Export
    with tabs[2]:
        st.subheader("Export to Excel")
        results_df = load_results()
        data = export_excel(
            st.session_state.ai5 if st.session_state.ai5 is not None else pd.DataFrame(),
            st.session_state.parlays_df if st.session_state.parlays_df is not None else pd.DataFrame(),
            results_df
        )
        st.download_button(
            "Download bets.xlsx",
            data=data,
            file_name="bets.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Click **Generate Today's Picks** in the sidebar.")
