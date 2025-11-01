# app.py — TruLine (Argumentative Ensemble Edition)
# -------------------------------------------------
# - Two tabs: "🤖 AI Ensemble Picks" and "📊 Results"
# - Uses The Odds API (v4) for markets: h2h, spreads, totals
# - Argumentative Ensemble: four simple ML-inspired scorers “argue” and vote
# - Manual Result Editor per-market (Pending & Completed), no redirect issues
# - Results stored in bets.csv, dedup-safe

import os
from typing import Dict, Any, Optional, Tuple
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
# Configuration
# ─────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "1d677dc98d978ccc24d9914d835442f1")
DEFAULT_REGIONS = os.getenv("REGIONS", "us")

SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "College Football (NCAAF)": "americanfootball_ncaaf",
    "College Basketball (NCAAB)": "basketball_ncaab",
    "Soccer (All Major Leagues)": [
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_italy_serie_a",
        "soccer_france_ligue_one",
        "soccer_germany_bundesliga",
        "soccer_uefa_champions_league",
    ],
}

RESULTS_FILE = "bets.csv"

# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="TruLine – AI Ensemble Picker", layout="wide")
st.title("TruLine – AI Ensemble Picker 🧠⚖️")
st.caption("Four simple models argue & vote across Moneyline / Spreads / Totals. Manual tracking & ROI included.")
st.divider()

# ─────────────────────────────────────────────
# Helper math
# ─────────────────────────────────────────────
def american_to_decimal(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds): return np.nan
    o = float(odds)
    return 1 + (o / 100.0) if o > 0 else 1 + (100.0 / abs(o))

def implied_prob_american(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds): return np.nan
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)

def kelly_fraction(p: float, dec: float, bcap: float = 0.1) -> float:
    """
    Kelly on decimal odds: f* = (bp - (1-p)) / (b) with b = dec - 1
    Clamp to [0, bcap]
    """
    if pd.isna(p) or pd.isna(dec) or dec <= 1.0:
        return 0.0
    b = dec - 1.0
    f = (b * p - (1 - p)) / b
    return float(max(0.0, min(bcap, f)))

def fmt_pct(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{x*100:.1f}%"

# ─────────────────────────────────────────────
# Odds API fetch
# ─────────────────────────────────────────────
def _odds_get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY (set env var ODDS_API_KEY).")
        return None
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 401:
            st.error("Odds API says: Invalid API key.")
            return None
        if r.status_code != 200:
            st.warning(f"Odds API non-200: {r.status_code} — {r.text[:200]}")
            return None
        return r.json()
    except Exception as e:
        st.error(f"Network error: {e}")
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
                outs = mk.get("outcomes", [])
                for oc in outs:
                    rows.append({
                        "event_id": event_id,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": mkey,
                        "outcome": oc.get("name"),       # Home/Away/Over/Under or team name
                        "line": oc.get("point"),         # None for h2h
                        "odds_american": oc.get("price"),
                        "odds_decimal": american_to_decimal(oc.get("price")),
                        "conf_book": implied_prob_american(oc.get("price")),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize times
    df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(df["commence_time"]):
        df["Date/Time"] = df["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    else:
        df["Date/Time"] = df["commence_time"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")

    return df

# ─────────────────────────────────────────────
# Consensus + Enrichment (for ensemble)
# ─────────────────────────────────────────────
def build_consensus(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      cons: one row per (event, market, outcome, line) with consensus & best price
      spread_stats: price dispersion stats per candidate (std/mean of odds across books)
    """
    if raw.empty:
        return raw, pd.DataFrame()

    # Best odds per candidate
    idx_best = raw.groupby(["event_id", "market", "outcome", "line"])["odds_decimal"].idxmax()
    best = raw.loc[idx_best, ["event_id","market","outcome","line","odds_american","odds_decimal","book"]]
    best = best.rename(columns={"odds_american":"best_odds_us","odds_decimal":"best_odds_dec","book":"best_book"})

    # Consensus & metadata
    agg = raw.groupby(["event_id","market","outcome","line"], dropna=False).agg(
        consensus_conf=("conf_book","mean"),
        books=("book","nunique"),
        home_team=("home_team","first"),
        away_team=("away_team","first"),
        commence_time=("commence_time","first"),
        date_time=("Date/Time","first"),
    ).reset_index()

    # Price dispersion across books (argument for “market disagreement”)
    spread_stats = raw.groupby(["event_id","market","outcome","line"], dropna=False).agg(
        odds_us_list=("odds_american", lambda x: list(pd.Series(x).dropna().astype(float))),
        odds_dec_std=("odds_decimal","std"),
        odds_dec_mean=("odds_decimal","mean")
    ).reset_index()
    spread_stats["odds_dec_std"] = spread_stats["odds_dec_std"].fillna(0.0)

    cons = agg.merge(best, on=["event_id","market","outcome","line"], how="left")
    cons["Matchup"] = cons["home_team"] + " vs " + cons["away_team"]
    cons["Confidence"] = cons["consensus_conf"]
    cons["Odds (US)"] = cons["best_odds_us"]
    cons["Odds (Dec)"] = cons["best_odds_dec"]
    cons["Sportsbook"] = cons["best_book"]
    cons["Date/Time"] = cons["date_time"]

    cons = cons[[
        "event_id","commence_time","Date/Time","Matchup","home_team","away_team",
        "market","outcome","line","Sportsbook","Odds (US)","Odds (Dec)","Confidence","books"
    ]]

    cons = cons.merge(spread_stats, on=["event_id","market","outcome","line"], how="left")

    return cons, spread_stats

def candidates_from_cons(cons: pd.DataFrame) -> pd.DataFrame:
    """
    Build candidate rows (one per event per market choose the higher-confidence outcome),
    but still retain the alternative for voting. We’ll do per-event selection later.
    """
    return cons.copy()

# ─────────────────────────────────────────────
# Argumentative Ensemble (4 scorers)
# ─────────────────────────────────────────────
def scorer_value_edge(row: pd.Series) -> float:
    """
    Value Edge Model:
    - Use consensus_conf as proxy for “true” p (average implied across books)
    - Use best_odds_dec to get fair edge: EV margin ~ p*dec - 1
    """
    p = float(row.get("Confidence", np.nan))
    dec = float(row.get("Odds (Dec)", np.nan))
    if pd.isna(p) or pd.isna(dec) or dec <= 1.0:
        return 0.0
    # Expected value margin (simplified)
    ev_margin = p * dec - 1.0
    return float(ev_margin)

def scorer_agreement(row: pd.Series) -> float:
    """
    Market Agreement Model:
    - More books -> stronger signal; combine with confidence level
    """
    p = float(row.get("Confidence", np.nan))
    b = float(row.get("books", 0.0))
    if pd.isna(p):
        return 0.0
    # smooth scaling
    return float(p * (1.0 + np.log1p(b)))

def scorer_disagreement_alpha(row: pd.Series) -> float:
    """
    Market Disagreement Alpha:
    - If odds dispersion (std) > 0, and best price is on the long side,
      we may be finding positive mispricing.
    """
    std = float(row.get("odds_dec_std", 0.0))
    dec = float(row.get("Odds (Dec)", np.nan))
    if pd.isna(dec):
        return 0.0
    # reward dispersion * price
    return float(std * (dec - 1.0))

def scorer_underdog_boost(row: pd.Series) -> float:
    """
    Underdog Booster:
    - Mild boost when odds_us positive but EV still decent
    """
    us = row.get("Odds (US)", None)
    p = float(row.get("Confidence", np.nan))
    dec = float(row.get("Odds (Dec)", np.nan))
    if us is None or pd.isna(p) or pd.isna(dec):
        return 0.0
    try:
        us = float(us)
    except:
        return 0.0
    base = 0.0
    if us > 0:  # underdog
        base = 0.02  # mild
        # extra if EV positive
        if p * dec - 1.0 > 0.0:
            base += 0.02
    return float(base)

def argumentative_vote(row: pd.Series) -> Tuple[float, int]:
    """
    Run all scorers, convert to (score, votes).
    Vote counted when scorer > 0.
    """
    s1 = scorer_value_edge(row)
    s2 = scorer_agreement(row)
    s3 = scorer_disagreement_alpha(row)
    s4 = scorer_underdog_boost(row)
    scores = np.array([s1, s2, s3, s4], dtype=float)
    votes = int(np.sum(scores > 0.0))
    total = float(np.mean(scores)) + 0.01 * votes  # blend mean & votes
    return total, votes

def assemble_ai_picks(cons: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Build one scoring table across all markets (h2h/totals/spreads), then:
      - score via argumentative_vote
      - per-event, keep only the top-scoring outcome per market
      - pick top N overall (diversified by event+market)
    """
    if cons.empty:
        return pd.DataFrame()

    df = cons.copy()

    # score every candidate
    out_scores = []
    for i, r in df.iterrows():
        s, v = argumentative_vote(r)
        out_scores.append((s, v))
    df["EnsembleScore"], df["Votes"] = zip(*out_scores)

    # keep best candidate per event+market (avoid duplicates per game)
    df["_evmk"] = df["event_id"].astype(str) + "|" + df["market"].astype(str)
    idx = df.groupby("_evmk")["EnsembleScore"].idxmax()
    best = df.loc[idx].copy()

    # rank and take top N across all markets
    best = best.sort_values(["EnsembleScore", "Confidence", "books"], ascending=[False, False, False]).head(top_n)

    # present friendly columns
    best["Pick"] = best["outcome"]
    best["Line"] = best["line"]
    best["Confidence %"] = best["Confidence"].apply(lambda p: f"{p*100:.1f}%")
    # Kelly units (cap @ 4.0)
    best["Units"] = best.apply(
        lambda r: round(min(4.0, 0.5 + 10.0 * kelly_fraction(float(r["Confidence"]), float(r["Odds (Dec)"]))), 1),
        axis=1
    )
    keep = ["Date/Time","Matchup","market","Pick","Line","Sportsbook","Odds (US)","Odds (Dec)","Confidence %","Units","Votes","EnsembleScore"]
    friendly = best[keep].rename(columns={"market":"Market"})
    return friendly.reset_index(drop=True)

# ─────────────────────────────────────────────
# Results tracking (CSV)
# ─────────────────────────────────────────────
def load_results() -> pd.DataFrame:
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        # ensure columns exist
        need = ["Sport","Market","Date/Time","Matchup","Pick","Line","Odds (US)","Units","Result"]
        for c in need:
            if c not in df.columns:
                df[c] = "" if c not in ("Units",) else 1.0
        df["Result"] = df["Result"].fillna("Pending")
        return df
    return pd.DataFrame(columns=["Sport","Market","Date/Time","Matchup","Pick","Line","Odds (US)","Units","Result"])

def save_results(df: pd.DataFrame):
    df.to_csv(RESULTS_FILE, index=False)

def auto_log_picks(dfs: Dict[str, pd.DataFrame], sport_name: str):
    """
    Dedup-safe logging. Keys: Sport, Market, Date/Time, Matchup, Pick, Line
    """
    results = load_results()
    for market_label, picks in dfs.items():
        if picks is None or picks.empty:
            continue
        for _, row in picks.iterrows():
            entry = {
                "Sport": sport_name,
                "Market": market_label,
                "Date/Time": row.get("Date/Time",""),
                "Matchup": row.get("Matchup",""),
                "Pick": row.get("Pick",""),
                "Line": row.get("Line",""),
                "Odds (US)": row.get("Odds (US)",""),
                "Units": float(row.get("Units", 1.0)) if str(row.get("Units","")).strip() != "" else 1.0,
                "Result": "Pending"
            }
            dup_mask = (
                (results["Sport"] == entry["Sport"]) &
                (results["Market"] == entry["Market"]) &
                (results["Date/Time"] == entry["Date/Time"]) &
                (results["Matchup"] == entry["Matchup"]) &
                (results["Pick"] == entry["Pick"]) &
                (results["Line"].fillna("").astype(str) == str(entry["Line"]))
            )
            if not dup_mask.any():
                results = pd.concat([results, pd.DataFrame([entry])], ignore_index=True)
    save_results(results)

def calc_summary(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"win_pct":0.0,"units_won":0.0,"roi":0.0,"wins":0,"losses":0,"total":0}
    total = len(df)
    wins = (df["Result"] == "Win").sum()
    losses = (df["Result"] == "Loss").sum()
    tmp = df.copy()
    tmp["Risked"] = tmp["Units"].astype(float).abs()
    tmp["PnL"] = tmp.apply(lambda r: r["Units"] if r["Result"] == "Win" else (-r["Units"] if r["Result"] == "Loss" else 0.0), axis=1)
    units_won = float(tmp["PnL"].sum())
    units_risked = float(tmp.loc[tmp["Result"].isin(["Win","Loss"]), "Risked"].sum())
    roi = (units_won/units_risked*100.0) if units_risked>0 else 0.0
    win_pct = (wins/total*100.0) if total>0 else 0.0
    return {"win_pct":win_pct,"units_won":units_won,"roi":roi,"wins":wins,"losses":losses,"total":total}

def show_market_editor(sport_name: str, market_label: str, key_prefix: str):
    """
    Collapsible editor for one market:
      - Pending dropdown list (each pick => set Win/Loss)
      - Completed dropdown list (you can adjust a result)
    No reroute/redirect: we avoid experimental_rerun; we toast on save.
    """
    results_df = load_results()
    market_df = results_df[(results_df["Sport"] == sport_name) & (results_df["Market"] == market_label)].copy()

    # metrics
    msum = calc_summary(market_df[market_df["Result"].isin(["Win","Loss"])])
    c1,c2,c3 = st.columns(3)
    c1.metric(f"{market_label} Win %", f"{msum['win_pct']:.1f}% ({msum['wins']}-{msum['losses']})")
    c2.metric(f"{market_label} Units Won", f"{msum['units_won']:.1f}")
    c3.metric(f"{market_label} ROI", f"{msum['roi']:.1f}%")

    # Pending editor
    with st.expander(f"✍️ Edit Pending — {market_label}", expanded=False):
        pending = market_df[market_df["Result"] == "Pending"].copy()
        if not pending.empty:
            pending["_key"] = (
                pending["Date/Time"].astype(str) + " | " +
                pending["Matchup"].astype(str) + " | " +
                pending["Pick"].astype(str) + " | " +
                pending["Line"].fillna("").astype(str)
            )
            pending = pending.drop_duplicates("_key")
            for i, r in pending.iterrows():
                left, right = st.columns([5,2])
                with left:
                    label = f"{r['Date/Time']} — {r['Matchup']} ({market_label}) — Pick: "
                    if market_label == "Totals":
                        label += f"{r['Pick']} ({r['Line']})"
                    elif market_label == "Spreads":
                        try:
                            ln = float(r["Line"])
                            sign = "+" if ln > 0 else ""
                            label += f"{r['Pick']} ({sign}{ln})"
                        except:
                            label += f"{r['Pick']} ({r['Line']})"
                    else:
                        label += f"{r['Pick']}"
                    st.write(label)
                with right:
                    sel = st.selectbox(
                        "Set Result",
                        ["Pending","Win","Loss"],
                        index=0,
                        key=f"{key_prefix}_pend_sel_{i}"
                    )
                    if st.button("Save", key=f"{key_prefix}_pend_save_{i}"):
                        mask = (
                            (results_df["Sport"] == sport_name) &
                            (results_df["Market"] == market_label) &
                            (results_df["Date/Time"] == r["Date/Time"]) &
                            (results_df["Matchup"] == r["Matchup"]) &
                            (results_df["Pick"] == r["Pick"]) &
                            (results_df["Line"].fillna("").astype(str) == str(r["Line"]))
                        )
                        results_df.loc[mask, "Result"] = sel
                        save_results(results_df)
                        st.toast("Saved ✅", icon="✅")
        else:
            st.info("No pending picks here.")

    # Completed editor
    with st.expander(f"🗂 Completed — {market_label}", expanded=False):
        done = market_df[market_df["Result"].isin(["Win","Loss"])].copy()
        if not done.empty:
            done["_key"] = (
                done["Date/Time"].astype(str) + " | " +
                done["Matchup"].astype(str) + " | " +
                done["Pick"].astype(str) + " | " +
                done["Line"].fillna("").astype(str)
            )
            done = done.drop_duplicates("_key").sort_values("Date/Time")
            for i, r in done.iterrows():
                left, right = st.columns([5,2])
                with left:
                    label = f"{r['Date/Time']} — {r['Matchup']} ({market_label}) — Pick: "
                    if market_label == "Totals":
                        label += f"{r['Pick']} ({r['Line']})"
                    elif market_label == "Spreads":
                        try:
                            ln = float(r["Line"])
                            sign = "+" if ln > 0 else ""
                            label += f"{r['Pick']} ({sign}{ln})"
                        except:
                            label += f"{r['Pick']} ({r['Line']})"
                    else:
                        label += f"{r['Pick']}"
                    st.write(label)
                with right:
                    sel = st.selectbox(
                        "Adjust Result",
                        ["Win","Loss","Pending"],
                        index=["Win","Loss","Pending"].index(r["Result"]),
                        key=f"{key_prefix}_done_sel_{i}"
                    )
                    if st.button("Save", key=f"{key_prefix}_done_save_{i}"):
                        mask = (
                            (results_df["Sport"] == sport_name) &
                            (results_df["Market"] == market_label) &
                            (results_df["Date/Time"] == r["Date/Time"]) &
                            (results_df["Matchup"] == r["Matchup"]) &
                            (results_df["Pick"] == r["Pick"]) &
                            (results_df["Line"].fillna("").astype(str) == str(r["Line"]))
                        )
                        results_df.loc[mask, "Result"] = sel
                        save_results(results_df)
                        st.toast("Saved ✅", icon="✅")
        else:
            st.info("No completed picks yet.")

def show_results_summary(sport_name: str):
    results = load_results()
    filt = results[(results["Sport"] == sport_name) & (results["Market"].isin(["Moneyline","Spreads","Totals","AI Ensemble"]))].copy()
    if filt.empty:
        st.info(f"No bets logged yet for {sport_name}.")
        return
    st.subheader(f"📊 Results — {sport_name}")
    st.dataframe(filt, use_container_width=True, hide_index=True)
    summ = calc_summary(filt[filt["Result"].isin(["Win","Loss"])])
    c1,c2,c3 = st.columns(3)
    c1.metric("Win %", f"{summ['win_pct']:.1f}% ({summ['wins']}-{summ['losses']})")
    c2.metric("Units Won", f"{summ['units_won']:.1f}")
    c3.metric("ROI", f"{summ['roi']:.1f}%")

# ─────────────────────────────────────────────
# Sidebar (choose sport set, regions, #picks, fetch)
# ─────────────────────────────────────────────
with st.sidebar:
    if "sport_name" not in st.session_state:
        st.session_state.sport_name = list(SPORT_OPTIONS.keys())[0]

    # Special “All Sports (Where Available)” option by combining keys
    SPORT_WITH_ALL = ["All (combine NFL/NBA/MLB/NCAAF/NCAAB/Soccer*)"] + list(SPORT_OPTIONS.keys())
    choice = st.selectbox(
        "Sports Source",
        SPORT_WITH_ALL,
        index=SPORT_WITH_ALL.index(st.session_state.get("sport_name", SPORT_WITH_ALL[0]))
        if st.session_state.get("sport_name", None) in SPORT_WITH_ALL else 0
    )
    if choice == "All (combine NFL/NBA/MLB/NCAAF/NCAAB/Soccer*)":
        st.session_state.sport_name = choice
    else:
        st.session_state.sport_name = choice

    regions = st.text_input("Regions", value=DEFAULT_REGIONS, key="regions")
    top_n = st.slider("Top AI picks to show", 3, 20, 5, key="top_n")
    fetch = st.button("Fetch Live Odds & Build AI Picks")

# ─────────────────────────────────────────────
# Fetch odds and build ensemble picks
# ─────────────────────────────────────────────
def fetch_all_selected(sport_choice: str, regions: str) -> pd.DataFrame:
    if sport_choice == "All (combine NFL/NBA/MLB/NCAAF/NCAAB/Soccer*)":
        parts = []
        for nm, key in SPORT_OPTIONS.items():
            if isinstance(key, list):
                for sk in key:
                    parts.append(fetch_odds(sk, regions))
            else:
                parts.append(fetch_odds(key, regions))
        if parts:
            return pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True)
        return pd.DataFrame()
    else:
        key = SPORT_OPTIONS[sport_choice]
        if isinstance(key, list):
            parts = [fetch_odds(k, regions) for k in key]
            return pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
        return fetch_odds(key, regions)

if fetch:
    raw = fetch_all_selected(st.session_state.sport_name, regions)
    if raw.empty:
        st.warning("No odds returned. Switch sport or check API key/quota.")
        st.session_state.has_data = False
    else:
        cons, _ = build_consensus(raw)
        ai5 = assemble_ai_picks(cons, top_n=top_n)
        # Log AI picks under a dedicated "AI Ensemble" market for results tracking
        auto_log_picks({"AI Ensemble": ai5.rename(columns={"Market":"market"})}, sport_name=st.session_state.sport_name)
        # Stash
        st.session_state.has_data = True
        st.session_state.raw = raw
        st.session_state.cons = cons
        st.session_state.ai5 = ai5

# ─────────────────────────────────────────────
# Render tabs
# ─────────────────────────────────────────────
if st.session_state.get("has_data", False):
    tabs = st.tabs([
        "🤖 AI Ensemble Picks",
        "📊 Results",
    ])

    # Tab 0 — AI Ensemble
    with tabs[0]:
        st.subheader("Top AI Ensemble Picks")
        if "ai5" in st.session_state and not st.session_state.ai5.empty:
            df = st.session_state.ai5.copy()
            # nicer market labels
            df["Market"] = df["Market"].replace({"h2h":"Moneyline","spreads":"Spreads","totals":"Totals"})
            show = df[["Date/Time","Matchup","Market","Pick","Line","Sportsbook","Odds (US)","Odds (Dec)","Confidence %","Units","Votes","EnsembleScore"]]
            st.dataframe(show, use_container_width=True, hide_index=True)
            # Units / Confidence visualization
            st.caption("Ensemble confidence vs. price (higher EnsembleScore ranks first).")
        else:
            st.info("No AI picks available yet. Click the sidebar button to fetch & build.")

    # Tab 1 — Results (editor + summary)
    with tabs[1]:
        st.subheader("Results — Manual Editor & Summary")
        # Summary + editors for: AI Ensemble, Moneyline, Totals, Spreads
        show_market_editor(st.session_state.sport_name, "AI Ensemble", "ai")
        st.markdown("---")
        show_market_editor(st.session_state.sport_name, "Moneyline", "ml")
        st.markdown("---")
        show_market_editor(st.session_state.sport_name, "Totals", "tot")
        st.markdown("---")
        show_market_editor(st.session_state.sport_name, "Spreads", "spr")
        st.markdown("---")
        show_results_summary(st.session_state.sport_name)

else:
    st.info("Pick a sport (or All) and click **Fetch Live Odds & Build AI Picks**")
