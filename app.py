import os
from typing import Dict, Any, Optional, List, Tuple
import requests
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

# =============== PAGE CONFIG ===============
st.set_page_config(page_title="TruLine – AI Picks & Parlays (All Sports)", layout="wide")
st.title("TruLine – AI Picks & Parlays 🚀")
st.caption("Top 5 AI picks across ALL sports + 5 Parlays (2,2,3,5,6 legs). No player props in this build.")
st.divider()

# =============== LOAD .ENV SAFELY ==========
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# =============== CONFIG ====================
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

ALL_SPORT_KEYS: List[str] = [
    "americanfootball_nfl",
    "basketball_nba",
    "baseball_mlb",
    "americanfootball_ncaaf",
    "basketball_ncaab",
] + SOCCER_KEYS

PARLAY_LEG_PATTERN = [2, 2, 3, 5, 6]

# =============== HELPERS ===================
def american_to_decimal(o: Optional[float]) -> float:
    if o is None or pd.isna(o): 
        return np.nan
    o = float(o)
    return 1 + (o / 100.0) if o > 0 else 1 + (100.0 / abs(o))

def implied_prob_from_american(o: Optional[float]) -> float:
    """Return probability in [0,1]"""
    if o is None or pd.isna(o):
        return np.nan
    o = float(o)
    return (100.0 / (o + 100.0)) if o > 0 else (abs(o) / (abs(o) + 100.0))

def assign_units(conf_0_to_1: float) -> float:
    # 0.5u to 5.0u linearly with confidence
    if pd.isna(conf_0_to_1):
        return 0.5
    return round(0.5 + 4.5 * max(0.0, min(1.0, conf_0_to_1)), 1)

def fmt_pct(p: float) -> str:
    return f"{p*100:.1f}%" if p == p else ""

def _odds_api(url: str, params: Dict[str, Any]):
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    data = _odds_api(url, {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
    })
    if not data:
        return pd.DataFrame()
    rows = []
    for ev in data:
        home = ev.get("home_team")
        away = ev.get("away_team")
        eid = ev.get("id")
        comm = ev.get("commence_time")
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                mkey = mk.get("key")  # "h2h","spreads","totals"
                for oc in mk.get("outcomes", []):
                    rows.append({
                        "sport_key": sport_key,
                        "event_id": eid,
                        "home": home,
                        "away": away,
                        "comm": comm,
                        "market": mkey,
                        "outcome": oc.get("name"),   # Home/Away or Over/Under
                        "line": oc.get("point"),
                        "odds_us": oc.get("price"),
                        "odds_dec": american_to_decimal(oc.get("price")),
                        "p_book": implied_prob_from_american(oc.get("price")),  # 0..1
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["comm"] = pd.to_datetime(df["comm"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(df["comm"]):
        df["Date/Time"] = df["comm"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    else:
        df["Date/Time"] = df["comm"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    df["Matchup"] = df["away"].astype(str) + " @ " + df["home"].astype(str)
    return df

@st.cache_data(ttl=60)
def fetch_all_sports(regions: str) -> pd.DataFrame:
    parts = []
    for k in ALL_SPORT_KEYS:
        df = fetch_odds(k, regions)
        if not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def consensus_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Choose best price per (event,market,outcome,line). Confidence = avg prob across books."""
    if raw.empty:
        return raw
    # Best decimal odds row index per identity:
    idx_best = raw.groupby(["event_id", "market", "outcome", "line"])["odds_dec"].idxmax()
    best = raw.loc[idx_best, ["event_id","market","outcome","line","odds_us","odds_dec","sport_key"]].copy()

    agg = raw.groupby(["event_id","market","outcome","line"], dropna=False).agg(
        Confidence=("p_book","mean"),
        home=("home","first"),
        away=("away","first"),
        DateTime=("Date/Time","first"),
        sport=("sport_key","first"),
    ).reset_index()

    out = agg.merge(best, on=["event_id","market","outcome","line"], how="left")
    out["Matchup"] = out["away"].astype(str) + " @ " + out["home"].astype(str)
    return out

def best_per_event(cons: pd.DataFrame, market_key: str) -> pd.DataFrame:
    sub = cons[cons["market"] == market_key].copy()
    if sub.empty:
        return sub
    idx = sub.groupby("event_id")["Confidence"].idxmax()
    return sub.loc[idx].copy()

def format_pick(market: str, outcome: str, line) -> str:
    if market == "h2h":
        return outcome
    if market == "spreads":
        if line is None or (isinstance(line, float) and np.isnan(line)):
            return f"{outcome}"
        try:
            ln = float(line)
            sign = "+" if ln > 0 else ""
            return f"{outcome} ({sign}{ln})"
        except Exception:
            return f"{outcome} ({line})"
    if market == "totals":
        return f"{outcome} ({line})"
    return outcome

def ai_top5(cons: pd.DataFrame) -> pd.DataFrame:
    """Take top per-event picks across markets, then highest-confidence 5 unique events."""
    frames = []
    for m in ["h2h", "spreads", "totals"]:
        bp = best_per_event(cons, m)
        if not bp.empty:
            bp["Market"] = m
            frames.append(bp)
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)
    # pick top by Confidence ensuring unique event_ids
    pool = pool.sort_values("Confidence", ascending=False)
    used = set()
    rows = []
    for _, r in pool.iterrows():
        if r["event_id"] in used:
            continue
        used.add(r["event_id"])
        rows.append(r)
        if len(rows) == 5:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    df["Pick"] = df.apply(lambda rr: format_pick(rr["Market"], rr["outcome"], rr["line"]), axis=1)
    df["Units"] = df["Confidence"].apply(assign_units)
    df["Confidence %"] = df["Confidence"].apply(fmt_pct)
    df = df.rename(columns={"odds_us":"Odds (US)","odds_dec":"Odds (Dec)","DateTime":"Date/Time","sport":"Sport"})
    return df[["Date/Time","Sport","Matchup","Market","Pick","Odds (US)","Odds (Dec)","Confidence","Confidence %","Units"]].reset_index(drop=True)

def build_parlay_pool(cons: pd.DataFrame) -> pd.DataFrame:
    """Build a pool of strong picks to assemble parlays."""
    frames = []
    for m in ["h2h", "spreads", "totals"]:
        bp = best_per_event(cons, m)
        if not bp.empty:
            bp["Market"] = m
            frames.append(bp)
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)
    # filter out NaN odds
    pool = pool[pool["odds_dec"].notna() & pool["Confidence"].notna()]
    # sort by confidence and odds quality
    pool = pool.sort_values(["Confidence","odds_dec"], ascending=[False, False]).reset_index(drop=True)
    # add pretty pick label
    pool["Pick"] = pool.apply(lambda r: format_pick(r["Market"], r["outcome"], r["line"]), axis=1)
    pool = pool.rename(columns={"odds_us":"Odds (US)","odds_dec":"Odds (Dec)","DateTime":"Date/Time","sport":"Sport"})
    return pool

def assemble_parlay_from_pool(pool: pd.DataFrame, legs: int, used_event_ids: set) -> Optional[pd.DataFrame]:
    """
    Greedy pick top confident legs with unique event_ids and avoid reusing the same event in another parlay if possible.
    """
    if pool.empty:
        return None
    picks = []
    taken = set()
    for _, r in pool.iterrows():
        eid = r["event_id"]
        if eid in taken:
            continue
        # prefer not to reuse an event already used globally for other parlays
        if eid in used_event_ids:
            continue
        picks.append(r)
        taken.add(eid)
        used_event_ids.add(eid)
        if len(picks) == legs:
            break
    # if we couldn't fill because of used_event_ids, relax constraint:
    if len(picks) < legs:
        for _, r in pool.iterrows():
            eid = r["event_id"]
            if eid in taken:
                continue
            picks.append(r)
            taken.add(eid)
            if len(picks) == legs:
                break
    if len(picks) < legs:
        return None
    parlay = pd.DataFrame(picks).copy()
    return parlay

def compute_parlay_summary(parlay_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Combine leg odds and confidence:
    - Parlay Decimal Odds: product of legs' odds_dec
    - Approx Hit %: product of legs' Confidence (assuming independence)
    """
    dec_odds = float(np.prod(parlay_df["Odds (Dec)"].astype(float)))
    approx_hit = float(np.prod(parlay_df["Confidence"].astype(float)))
    units = assign_units(approx_hit)  # suggest stake size based on hit prob
    return {
        "Legs": len(parlay_df),
        "Parlay Odds (Dec)": round(dec_odds, 3),
        "Approx Hit %": fmt_pct(approx_hit),
        "Stake Units": units,
    }

def generate_5_parlays(pool: pd.DataFrame, pattern: List[int]) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Returns a high-level table and a list of detail DataFrames (one per parlay).
    """
    overviews = []
    details = []
    used_events_globally = set()
    for idx, legs in enumerate(pattern, start=1):
        p = assemble_parlay_from_pool(pool, legs, used_events_globally)
        if p is None or p.empty:
            continue
        summary = compute_parlay_summary(p)
        # overview row
        overviews.append({
            "Parlay #": idx,
            "Legs": summary["Legs"],
            "Parlay Odds (Dec)": summary["Parlay Odds (Dec)"],
            "Approx Hit %": summary["Approx Hit %"],
            "Stake Units": summary["Stake Units"],
        })
        # detail (with leg numbering)
        p = p.reset_index(drop=True)
        p.insert(0, "Leg", p.index + 1)
        p["Confidence %"] = p["Confidence"].apply(fmt_pct)
        details.append(p[["Leg","Date/Time","Sport","Matchup","Market","Pick","Odds (US)","Odds (Dec)","Confidence","Confidence %"]])
    overview_df = pd.DataFrame(overviews) if overviews else pd.DataFrame()
    return overview_df, details

def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Create an Excel file in-memory with multiple sheets."""
    bio = BytesIO()
    try:
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
    except Exception:
        # Fallback to xlsxwriter if openpyxl not installed
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
    bio.seek(0)
    return bio.read()

# =============== SIDEBAR ====================
with st.sidebar:
    st.subheader("Options")
    regions = st.text_input("Odds Regions", value=DEFAULT_REGIONS)
    st.caption("Tip: use 'us' for US books. (The Odds API plan must include your sports/markets.)")
    colA, colB = st.columns(2)
    gen_ai = colA.button("Generate Top-5 AI Picks")
    gen_parlays = colB.button("Generate 5 Parlays")

# =============== DATA FETCH (on demand) =====
if gen_ai or gen_parlays:
    raw = fetch_all_sports(regions)
    if raw.empty:
        st.warning("No odds returned. Try later or adjust regions / subscription plan.")
        st.stop()
    cons = consensus_table(raw)
    st.session_state.raw = raw
    st.session_state.cons = cons

# Ensure keys exist for first render
if "raw" not in st.session_state:
    st.session_state.raw = pd.DataFrame()
if "cons" not in st.session_state:
    st.session_state.cons = pd.DataFrame()
if "ai5" not in st.session_state:
    st.session_state.ai5 = pd.DataFrame()
if "parlays_overview" not in st.session_state:
    st.session_state.parlays_overview = pd.DataFrame()
if "parlays_details" not in st.session_state:
    st.session_state.parlays_details = []

# Build AI Top-5 when requested
if gen_ai and not st.session_state.cons.empty:
    st.session_state.ai5 = ai_top5(st.session_state.cons)

# Build parlays when requested
if gen_parlays and not st.session_state.cons.empty:
    pool = build_parlay_pool(st.session_state.cons)
    ovw, dets = generate_5_parlays(pool, PARLAY_LEG_PATTERN)
    st.session_state.parlays_overview = ovw
    st.session_state.parlays_details = dets

# =============== UI TABS =====================
tabs = st.tabs(["🤖 AI Top-5 Across All Sports", "🧩 5 Parlays", "Raw Data (debug)"])

# ---- Tab 0: AI Top-5
with tabs[0]:
    st.subheader("AI Top-5 (Moneyline / Spreads / Totals mixed)")
    if st.session_state.ai5.empty:
        st.info("Click **Generate Top-5 AI Picks** in the sidebar.")
    else:
        df = st.session_state.ai5.copy()
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Simple chart of confidence
        try:
            chart_df = df.copy()
            chart_df["Conf"] = chart_df["Confidence"]
            chart_df = chart_df.set_index("Matchup")[["Conf"]]
            st.caption("Confidence (model blend, higher = better)")
            st.bar_chart(chart_df)
        except Exception:
            pass

        # Download buttons
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        xls_bytes = to_excel_bytes({"AI Top 5": df})
        c1, c2 = st.columns(2)
        c1.download_button(
            label="⬇️ Download AI Top-5 (CSV)",
            data=csv_bytes,
            file_name="ai_top5.csv",
            mime="text/csv",
        )
        c2.download_button(
            label="⬇️ Download AI Top-5 (Excel)",
            data=xls_bytes,
            file_name="ai_top5.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ---- Tab 1: Parlays
with tabs[1]:
    st.subheader("5 Parlays (legs: 2, 2, 3, 5, 6)")
    if st.session_state.parlays_overview.empty or not st.session_state.parlays_details:
        st.info("Click **Generate 5 Parlays** in the sidebar.")
    else:
        st.markdown("#### Overview")
        st.dataframe(st.session_state.parlays_overview, use_container_width=True, hide_index=True)

        st.markdown("#### Details")
        for i, dfp in enumerate(st.session_state.parlays_details, start=1):
            with st.expander(f"Parlay #{i} — {len(dfp)} legs", expanded=False):
                st.dataframe(dfp, use_container_width=True, hide_index=True)

        # Downloads: one Excel with overview + each parlay on its own sheet
        sheets = {"Overview": st.session_state.parlays_overview}
        for i, dfp in enumerate(st.session_state.parlays_details, start=1):
            sheets[f"Parlay {i}"] = dfp
        xls_parlays = to_excel_bytes(sheets)
        st.download_button(
            label="⬇️ Download Parlays (Excel)",
            data=xls_parlays,
            file_name="parlays.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ---- Tab 2: Raw (debug)
with tabs[2]:
    st.subheader("Raw Per-Book Odds (sample)")
    if st.session_state.raw.empty:
        st.info("No data yet. Use the buttons in the sidebar.")
    else:
        st.dataframe(st.session_state.raw.head(300), use_container_width=True, hide_index=True)
        st.caption("Tip: this raw feed is aggregated to the consensus table used by the AI picks/parlays.")
