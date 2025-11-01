import os
from typing import Dict, Any, Optional, List, Tuple
import requests
import pandas as pd
import numpy as np
import streamlit as st

# =============== Page & Title =================
st.set_page_config(page_title="TruLine – AI & Parlays (All Sports)", layout="wide")
st.title("TruLine – AI Picks & Parlays 🚀")
st.caption("Top 5 AI picks across ALL sports (ML/Spreads/Totals) + 5 Parlays (2,2,3,5,6 legs). Tracks results. No props.")
st.divider()

# =============== Safe dotenv ==================
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# =============== Config =======================
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "") or "1d677dc98d978ccc24d9914d835442f1"
DEFAULT_REGIONS = os.getenv("REGIONS", "us")

# Sports we’ll aggregate (ALL six, including major soccer)
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

PARLAY_LEG_PATTERN = [2, 2, 3, 5, 6]  # Q2 YES

RESULTS_FILE = "bets.csv"  # single CSV for results

# =============== Helpers ======================
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

def assign_units_from_conf(conf_0to1: float) -> float:
    """Map [0..1] → 0.5..5.0 units."""
    if pd.isna(conf_0to1):
        return 0.5
    return round(0.5 + 4.5 * max(0.0, min(1.0, float(conf_0to1))), 1)

def fmt_pct(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{100.0 * x:.1f}%"

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
    """Fetch v4 odds for a single sport+regions; return normalized long DF."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    data = _odds_get(url, {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
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
                mkey = mk.get("key")  # 'h2h' | 'spreads' | 'totals'
                for oc in mk.get("outcomes", []):
                    rows.append({
                        "sport_key": sport_key,
                        "event_id": event_id,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": mkey,
                        "outcome": oc.get("name"),       # Home/Away/Over/Under
                        "line": oc.get("point"),         # None for h2h
                        "odds_american": oc.get("price"),
                        "odds_decimal": american_to_decimal(oc.get("price")),
                        "conf_book": implied_prob_american(oc.get("price")) / 100.0,  # 0..1
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
    # localize to ET for display
    if pd.api.types.is_datetime64tz_dtype(df["commence_time"]):
        df["Date/Time"] = df["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    else:
        df["Date/Time"] = df["commence_time"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    return df

def build_consensus(raw: pd.DataFrame) -> pd.DataFrame:
    """Consensus across books; keep best odds per event/market/outcome/line; average confidence."""
    if raw.empty:
        return raw
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
    out["Matchup"] = out["away_team"] + " @ " + out["home_team"]
    out["Confidence"] = out["consensus_conf"]  # 0..1
    out["Odds (US)"] = out["best_odds_us"]
    out["Odds (Dec)"] = out["best_odds_dec"]
    out["Sportsbook"] = out["best_book"]
    out["Date/Time"] = out["date_time"]
    return out[[
        "sport_key","event_id","commence_time","Date/Time","Matchup","market","outcome","line",
        "Sportsbook","Odds (US)","Odds (Dec)","Confidence","books"
    ]].rename(columns={"books":"Books"})

def best_per_event(cons_df: pd.DataFrame, market_key: str) -> pd.DataFrame:
    """Within a market, keep highest-confidence row per event."""
    if cons_df.empty:
        return pd.DataFrame()
    sub = cons_df[cons_df["market"] == market_key].copy()
    if sub.empty:
        return pd.DataFrame()
    # idx of best confidence per event
    idx = sub.groupby("event_id")["Confidence"].idxmax()
    return sub.loc[idx].copy()

def ai_top5_all_sports(cons: pd.DataFrame) -> pd.DataFrame:
    """
    Build a pool from ML/Spreads/Totals across ALL SPORTS, pick 5 by:
    - Best confidence
    - Diversify: no duplicate event_id in final 5
    """
    if cons.empty:
        return pd.DataFrame()
    pool_parts = []
    for m in ["h2h", "spreads", "totals"]:
        p = best_per_event(cons, m)
        if not p.empty:
            p = p.copy()
            p["Market"] = m
            pool_parts.append(p)
    if not pool_parts:
        return pd.DataFrame()

    pool = pd.concat(pool_parts, ignore_index=True)
    pool = pool.sort_values("Confidence", ascending=False)

    chosen_rows = []
    used_events = set()
    for _, r in pool.iterrows():
        if r["event_id"] in used_events:
            continue
        chosen_rows.append(r)
        used_events.add(r["event_id"])
        if len(chosen_rows) == 5:
            break

    if not chosen_rows:
        return pd.DataFrame()

    ai = pd.DataFrame(chosen_rows).copy()
    # Display columns
    ai["Pick"] = ai.apply(lambda rr: _format_pick(rr["Market"], rr["outcome"], rr["line"]), axis=1)
    ai["Confidence %"] = ai["Confidence"].apply(fmt_pct)
    ai["Units"] = ai["Confidence"].apply(assign_units_from_conf)
    show_cols = ["Date/Time", "Matchup", "Market", "Pick", "Sportsbook", "Odds (US)", "Odds (Dec)", "Confidence %", "Units"]
    return ai[show_cols].reset_index(drop=True)

def _format_pick(market: str, outcome: Any, line: Any) -> str:
    if market == "h2h":
        return f"{outcome}"
    if market == "spreads":
        try:
            ln = float(line)
            sign = "+" if ln > 0 else ""
            return f"{outcome} ({sign}{ln})"
        except Exception:
            return f"{outcome} ({line})"
    if market == "totals":
        return f"{outcome} ({line})"
    return f"{outcome}"

def _legs_from_pool(pool: pd.DataFrame, max_legs: int, used_events_in_parlay: set) -> List[Dict[str, Any]]:
    """Greedily pick legs (distinct event_id) from a high-confidence pool."""
    legs = []
    for _, r in pool.iterrows():
        if r["event_id"] in used_events_in_parlay:
            continue
        legs.append({
            "sport_key": r["sport_key"],
            "event_id": r["event_id"],
            "Date/Time": r["Date/Time"],
            "Matchup": r["Matchup"],
            "Market": r["market"],
            "Pick": _format_pick(r["market"], r["outcome"], r["line"]),
            "Odds (US)": r["Odds (US)"],
            "Odds (Dec)": r["Odds (Dec)"],
            "Confidence": r["Confidence"],
        })
        used_events_in_parlay.add(r["event_id"])
        if len(legs) == max_legs:
            break
    return legs

def build_parlays(cons: pd.DataFrame, leg_pattern: List[int]) -> pd.DataFrame:
    """
    Create 5 parlays with legs= [2,2,3,5,6].
    Use a wide high-confidence pool across ML/Spreads/Totals (no props).
    Ensure within each parlay: distinct events.
    """
    if cons.empty:
        return pd.DataFrame()

    # Build a big pool (top-N per market)
    parts = []
    for m in ["h2h", "spreads", "totals"]:
        sub = cons[cons["market"] == m].copy()
        if sub.empty:
            continue
        # Get a generous slice per market
        sub = sub.sort_values("Confidence", ascending=False).head(150)
        parts.append(sub)
    if not parts:
        return pd.DataFrame()

    pool = pd.concat(parts, ignore_index=True)
    pool = pool.sort_values("Confidence", ascending=False)

    parlays = []
    for legs_needed in leg_pattern:
        used_events = set()
        legs = _legs_from_pool(pool, legs_needed, used_events)
        if len(legs) < legs_needed:
            # If not enough distinct events, just skip this parlay
            continue

        # Price & implied prob (assume independence)
        dec_odds = np.prod([l["Odds (Dec)"] if not pd.isna(l["Odds (Dec)"]) else 1.0 for l in legs])
        prob = np.prod([max(0.001, min(0.999, float(l["Confidence"]))) for l in legs])
        us_odds = _decimal_to_american(dec_odds)

        # Units: be conservative for longer parlays -> scale with min leg conf
        min_conf = float(min(l["Confidence"] for l in legs))
        units = max(0.5, round(0.5 + 2.0 * min_conf, 1))  # 0.5 to ~2.5

        legs_str = " | ".join([f"{l['Pick']} @ {l['Odds (US)']} ({l['Matchup']})" for l in legs])
        parlays.append({
            "Parlay": f"{len(legs)}-Leg",
            "Legs": legs_str,
            "Decimal Odds": round(dec_odds, 4),
            "US Odds": us_odds,
            "Implied Prob %": f"{prob*100:.1f}%",
            "Units": units,
        })

    return pd.DataFrame(parlays)

def _decimal_to_american(dec_odds: float) -> Optional[int]:
    if pd.isna(dec_odds) or dec_odds <= 1.0:
        return None
    if dec_odds >= 2.0:
        return int(round((dec_odds - 1.0) * 100))
    else:
        return int(round(-100 / (dec_odds - 1.0)))

# =============== Results storage =============
def load_results() -> pd.DataFrame:
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        # normalize columns
        for col in ["Type", "Sport", "Market", "Date/Time", "Matchup", "Pick", "Odds (US)", "Units", "Result", "Legs", "US Odds"]:
            if col not in df.columns:
                df[col] = ""
        if "Units" in df.columns:
            df["Units"] = pd.to_numeric(df["Units"], errors="coerce").fillna(1.0)
        df["Result"] = df["Result"].replace("", "Pending").fillna("Pending")
        return df
    return pd.DataFrame(columns=[
        "Type","Sport","Market","Date/Time","Matchup","Pick","Odds (US)","Units","Result","Legs","US Odds"
    ])

def save_results(df: pd.DataFrame):
    df.to_csv(RESULTS_FILE, index=False)

def log_ai_picks(ai_df: pd.DataFrame):
    """Append AI picks to results as individual rows of Type='AI' (dedup by Date/Time+Matchup+Pick)."""
    if ai_df is None or ai_df.empty:
        return
    results = load_results()
    for _, r in ai_df.iterrows():
        entry = {
            "Type": "AI",
            "Sport": "Mixed-All",
            "Market": r.get("Market",""),
            "Date/Time": r.get("Date/Time",""),
            "Matchup": r.get("Matchup",""),
            "Pick": r.get("Pick",""),
            "Odds (US)": r.get("Odds (US)",""),
            "Units": float(r.get("Units", 1.0)),
            "Result": "Pending",
            "Legs": "",
            "US Odds": r.get("Odds (US)",""),
        }
        dup_mask = (
            (results["Type"] == "AI") &
            (results["Date/Time"] == entry["Date/Time"]) &
            (results["Matchup"] == entry["Matchup"]) &
            (results["Pick"] == entry["Pick"])
        )
        if not dup_mask.any():
            results = pd.concat([results, pd.DataFrame([entry])], ignore_index=True)
    save_results(results)

def log_parlays(parlays_df: pd.DataFrame):
    """Append Parlays to results as Type='Parlay' (one row per parlay; 'Legs' holds the string)."""
    if parlays_df is None or parlays_df.empty:
        return
    results = load_results()
    for _, r in parlays_df.iterrows():
        entry = {
            "Type": "Parlay",
            "Sport": "Mixed-All",
            "Market": "Parlay",
            "Date/Time": "",        # Not applicable (multiple games)
            "Matchup": "",
            "Pick": r.get("Parlay",""),
            "Odds (US)": r.get("US Odds",""),
            "Units": float(r.get("Units", 1.0)),
            "Result": "Pending",
            "Legs": r.get("Legs",""),
            "US Odds": r.get("US Odds",""),
        }
        # Dedup by exact Legs string + Parlay tag
        dup_mask = (
            (results["Type"] == "Parlay") &
            (results["Pick"] == entry["Pick"]) &
            (results["Legs"] == entry["Legs"])
        )
        if not dup_mask.any():
            results = pd.concat([results, pd.DataFrame([entry])], ignore_index=True)
    save_results(results)

def calc_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Win%, Units Won, ROI based on Units only (no odds settlement model)."""
    if df is None or df.empty:
        return {"win_pct":0.0,"units_won":0.0,"roi":0.0,"wins":0,"losses":0,"total":0}
    total = len(df)
    wins = (df["Result"] == "Win").sum()
    losses = (df["Result"] == "Loss").sum()
    df = df.copy()
    df["Risked"] = df["Units"].abs()
    df["PnL"] = df.apply(lambda r: r["Units"] if r["Result"] == "Win" else (-r["Units"] if r["Result"] == "Loss" else 0.0), axis=1)
    units_won = float(df["PnL"].sum())
    units_risked = float(df.loc[df["Result"].isin(["Win","Loss"]), "Risked"].sum())
    roi = (units_won/units_risked*100.0) if units_risked>0 else 0.0
    win_pct = (wins/total*100.0) if total>0 else 0.0
    return {"win_pct":win_pct,"units_won":units_won,"roi":roi,"wins":wins,"losses":losses,"total":total}

# =============== Manual Editors ==============
def manual_editor_ai():
    st.markdown("### ✍️ Manual Result Editor — AI Top 5")
    res = load_results()
    ai_df = res[res["Type"] == "AI"].copy()
    if ai_df.empty:
        st.info("No AI picks logged yet.")
        return

    # Metrics
    summ = calc_summary(ai_df[ai_df["Result"].isin(["Win","Loss"])])
    c1,c2,c3 = st.columns(3)
    c1.metric("AI Win %", f"{summ['win_pct']:.1f}% ({summ['wins']}-{summ['losses']})")
    c2.metric("AI Units Won", f"{summ['units_won']:.1f}")
    c3.metric("AI ROI", f"{summ['roi']:.1f}%")

    with st.expander("Pending — AI Picks", expanded=False):
        pend = ai_df[ai_df["Result"] == "Pending"].copy()
        if pend.empty:
            st.info("No pending AI picks.")
        else:
            for i, r in pend.iterrows():
                left, right = st.columns([6,2])
                with left:
                    st.write(f"{r['Date/Time']} — {r['Matchup']} — {r['Market']} — {r['Pick']} @ {r['Odds (US)']} — Units: {r['Units']}")
                with right:
                    sel = st.selectbox("Set Result", ["Pending","Win","Loss"], index=0, key=f"ai_p_sel_{i}")
                    if st.button("Save", key=f"ai_p_save_{i}"):
                        res.loc[res.index == i, "Result"] = sel
                        save_results(res)
                        st.success("Saved.")

    with st.expander("Completed — AI Picks", expanded=False):
        comp = ai_df[ai_df["Result"].isin(["Win","Loss"])].copy()
        if comp.empty:
            st.info("No completed AI picks yet.")
        else:
            for i, r in comp.iterrows():
                left, right = st.columns([6,2])
                with left:
                    st.write(f"{r['Date/Time']} — {r['Matchup']} — {r['Market']} — {r['Pick']} @ {r['Odds (US)']} — Units: {r['Units']} — **{r['Result']}**")
                with right:
                    sel = st.selectbox("Adjust", ["Win","Loss","Pending"], index=["Win","Loss","Pending"].index(r["Result"]), key=f"ai_c_sel_{i}")
                    if st.button("Save", key=f"ai_c_save_{i}"):
                        res.loc[res.index == i, "Result"] = sel
                        save_results(res)
                        st.success("Updated.")

def manual_editor_parlays():
    st.markdown("### ✍️ Manual Result Editor — Parlays")
    res = load_results()
    pl = res[res["Type"] == "Parlay"].copy()
    if pl.empty:
        st.info("No parlays logged yet.")
        return

    # Metrics (parlays treated same as singles wrt Units)
    summ = calc_summary(pl[pl["Result"].isin(["Win","Loss"])])
    c1,c2,c3 = st.columns(3)
    c1.metric("Parlay Win %", f"{summ['win_pct']:.1f}% ({summ['wins']}-{summ['losses']})")
    c2.metric("Parlay Units Won", f"{summ['units_won']:.1f}")
    c3.metric("Parlay ROI", f"{summ['roi']:.1f}%")

    with st.expander("Pending — Parlays", expanded=False):
        pend = pl[pl["Result"] == "Pending"].copy()
        if pend.empty:
            st.info("No pending parlays.")
        else:
            for i, r in pend.iterrows():
                left, right = st.columns([6,2])
                with left:
                    st.write(f"{r['Pick']} — {r['US Odds']} — Units: {r['Units']}")
                    st.caption(r["Legs"])
                with right:
                    sel = st.selectbox("Set Result", ["Pending","Win","Loss"], index=0, key=f"pl_p_sel_{i}")
                    if st.button("Save", key=f"pl_p_save_{i}"):
                        res.loc[res.index == i, "Result"] = sel
                        save_results(res)
                        st.success("Saved.")

    with st.expander("Completed — Parlays", expanded=False):
        comp = pl[pl["Result"].isin(["Win","Loss"])].copy()
        if comp.empty:
            st.info("No completed parlays yet.")
        else:
            for i, r in comp.iterrows():
                left, right = st.columns([6,2])
                with left:
                    st.write(f"{r['Pick']} — {r['US Odds']} — Units: {r['Units']} — **{r['Result']}**")
                    st.caption(r["Legs"])
                with right:
                    sel = st.selectbox("Adjust", ["Win","Loss","Pending"], index=["Win","Loss","Pending"].index(r["Result"]), key=f"pl_c_sel_{i}")
                    if st.button("Save", key=f"pl_c_save_{i}"):
                        res.loc[res.index == i, "Result"] = sel
                        save_results(res)
                        st.success("Updated.")

# =============== Sidebar =====================
with st.sidebar:
    st.markdown("**Data Settings**")
    regions = st.text_input("Regions", value=DEFAULT_REGIONS)
    st.caption("Tip: 'us' is typical. You can try 'us,eu,uk' if your plan allows multiple regions.")

    st.markdown("**Actions**")
    go = st.button("Generate Today’s 5 AI Picks + 5 Parlays")

# =============== Main Orchestration ==========
def fetch_all_markets(regions: str) -> pd.DataFrame:
    dfs = []
    for key in ALL_SPORT_KEYS:
        df = fetch_odds(key, regions, markets="h2h,spreads,totals")
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

if go:
    raw = fetch_all_markets(regions)
    if raw.empty:
        st.warning("No odds returned. Check your API key / quota / regions.")
        st.session_state.has_data = False
    else:
        cons = build_consensus(raw)
        ai5 = ai_top5_all_sports(cons)
        parlays = build_parlays(cons, PARLAY_LEG_PATTERN)

        # Log to results
        log_ai_picks(ai5)
        log_parlays(parlays)

        # Stash in session
        st.session_state.cons = cons
        st.session_state.ai5 = ai5
        st.session_state.parlays = parlays
        st.session_state.has_data = True

# =============== Render ======================
if st.session_state.get("has_data", False):
    tabs = st.tabs(["🤖 AI Top 5 (All Sports)", "🏈+🏀+⚾ Parlays (2,2,3,5,6)"])

    # Tab 0: AI Top 5
    with tabs[0]:
        st.subheader("AI Top 5 — Across All Sports (ML / Spreads / Totals)")
        st.dataframe(st.session_state.ai5, use_container_width=True, hide_index=True)

        # Simple bar visualization by confidence
        df = st.session_state.ai5.copy()
        if not df.empty and "Confidence %" in df.columns:
            try:
                vals = df["Confidence %"].str.replace("%","", regex=False).astype(float)
                chart_df = pd.DataFrame({"Confidence": vals.values}, index=df["Matchup"].values)
                st.caption("Confidence heat — AI Top 5")
                st.bar_chart(chart_df)
            except Exception:
                pass

        st.markdown("---")
        manual_editor_ai()

    # Tab 1: Parlays
    with tabs[1]:
        st.subheader("5 Parlays — Legs = 2, 2, 3, 5, 6")
        st.dataframe(st.session_state.parlays, use_container_width=True, hide_index=True)
        st.caption("Each parlay avoids duplicate events among its legs and shows estimated implied probability (independence assumption).")
        st.markdown("---")
        manual_editor_parlays()

else:
    st.info("Click the button in the sidebar to generate today’s AI top 5 and parlays.")
