import os
from typing import Dict, Any, Optional
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

# ✅ API keys / config
ODDS_API_KEY = "1d677dc98d978ccc24d9914d835442f1"
APISPORTS_KEY = os.getenv("APISPORTS_KEY", st.secrets.get("APISPORTS_KEY", ""))  # not used in manual flow, kept for future
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

# Files
RESULTS_FILE = "bets.csv"
EXCEL_FILE = "bets.xlsx"

# Streamlit UI
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
    if pd.isna(conf): return 0.5
    return round(0.5 + 4.5 * max(0.0, min(1.0, conf)), 1)

def fmt_pct(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{100.0 * x:.1f}%"

def nice_line(market: str, pick: str, line) -> str:
    """Pretty print pick + line for spreads/totals."""
    if pd.isna(line) or line == "" or line is None:
        return str(pick)
    try:
        ln = float(line)
    except Exception:
        return f"{pick} ({line})"
    if market == "Totals":
        return f"{pick} ({ln})"  # "Over (44.5)" or "Under (44.5)"
    if market == "Spreads":
        sign = "+" if ln > 0 else ""  # negative sign will show automatically
        return f"{pick} ({sign}{ln})"
    return str(pick)

# ─────────────────────────────────────────────
# Odds API fetch
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
                        "outcome": oc.get("name"),       # Home/Away/Over/Under
                        "line": oc.get("point"),         # may be None for ML
                        "odds_american": oc.get("price"),
                        "odds_decimal": american_to_decimal(oc.get("price")),
                        "conf_book": implied_prob_american(oc.get("price")),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(df["commence_time"]):
        df["Date/Time"] = df["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    else:
        df["Date/Time"] = df["commence_time"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    return df

# ─────────────────────────────────────────────
# Consensus logic
# ─────────────────────────────────────────────
def build_consensus(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty: return raw
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
    if sub.empty: return pd.DataFrame()
    best_idx = sub.groupby("event_id")["Confidence"].idxmax()
    sub = sub.loc[best_idx].copy()
    sub = sub.sort_values("commence_time", ascending=True).head(top_n)
    out = sub[["Date/Time","Matchup","Sportsbook","outcome","line","Odds (US)","Odds (Dec)","Confidence","Books"]].copy()
    out = out.rename(columns={"outcome":"Pick","line":"Line"})
    out["Confidence"] = out["Confidence"].apply(fmt_pct)
    out["Units"] = sub["Confidence"].apply(assign_units)
    return out.reset_index(drop=True)

def ai_genius_top(cons_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if cons_df.empty: return pd.DataFrame()
    frames = []
    for m in ["h2h","totals","spreads"]:
        t = pick_best_per_event(cons_df, m, top_n*3)
        if not t.empty:
            t["Market"] = m
            t["_C"] = t["Confidence"].str.replace("%","",regex=False).astype(float)
            frames.append(t)
    if not frames: return pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True)
    allp = allp.sort_values("_C", ascending=False).drop(columns=["_C"]).head(top_n)
    return allp.reset_index(drop=True)

# ─────────────────────────────────────────────
# Results tracking + Excel export
# ─────────────────────────────────────────────
def load_results() -> pd.DataFrame:
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        for col in ["Sport","Market","Date/Time","Matchup","Pick","Line","Odds (US)","Units","Result"]:
            if col not in df.columns:
                df[col] = "" if col != "Units" else 1.0
        df["Result"] = df["Result"].fillna("Pending")
        # Filter out junk/Unknown rows that slipped in historically
        mask_clean = (~df["Matchup"].astype(str).str.contains("Unknown", na=False)) & (df["Matchup"].astype(str).str.strip() != "")
        df = df[mask_clean].copy()
        return df.reset_index(drop=True)
    return pd.DataFrame(columns=["Sport","Market","Date/Time","Matchup","Pick","Line","Odds (US)","Units","Result"])

def write_excel_book(results_df: pd.DataFrame):
    """Writes bets.xlsx with multiple sheets + a Summary sheet with formulas."""
    if results_df is None or results_df.empty:
        # create an empty workbook with headers so download still works
        cols = ["Date/Time","Sport","Matchup","Market","Pick","Line","Odds (US)","Units","Result","Units Gained"]
        empty = pd.DataFrame(columns=cols)
        with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as xw:
            empty.to_excel(xw, index=False, sheet_name="Results_All")
            for s in ["AI Genius","Moneyline","Totals","Spreads","Summary"]:
                empty.to_excel(xw, index=False, sheet_name=s)
        return

    # Compute Units Gained column
    def add_units_gained(df):
        df = df.copy()
        df["Units"] = pd.to_numeric(df["Units"], errors="coerce").fillna(0.0)
        df["Units Gained"] = df.apply(
            lambda r: r["Units"] if r["Result"] == "Win" else (-r["Units"] if r["Result"] == "Loss" else 0.0),
            axis=1
        )
        # Reorder columns exactly
        df = df[["Date/Time","Sport","Matchup","Market","Pick","Line","Odds (US)","Units","Result","Units Gained"]]
        return df

    # Filter out Unknown rows (just in case)
    clean = results_df[~results_df["Matchup"].astype(str).str.contains("Unknown", na=False)].copy()

    # Split per sheet
    all_df = add_units_gained(clean)
    ai_df  = add_units_gained(clean[clean["Market"] == "AI Genius"])
    ml_df  = add_units_gained(clean[clean["Market"] == "Moneyline"])
    tot_df = add_units_gained(clean[clean["Market"] == "Totals"])
    spr_df = add_units_gained(clean[clean["Market"] == "Spreads"])

    # Write book
    with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as xw:
        all_df.to_excel(xw, index=False, sheet_name="Results_All")
        ai_df.to_excel(xw, index=False, sheet_name="AI Genius")
        ml_df.to_excel(xw, index=False, sheet_name="Moneyline")
        tot_df.to_excel(xw, index=False, sheet_name="Totals")
        spr_df.to_excel(xw, index=False, sheet_name="Spreads")

        # Summary with formulas (wins/losses/total/win%/units) per market sheet
        summary = pd.DataFrame({
            "Market": ["AI Genius","Moneyline","Totals","Spreads"],
            "Wins":   [0,0,0,0],
            "Losses": [0,0,0,0],
            "Total":  [0,0,0,0],
            "Win %":  [0.0,0.0,0.0,0.0],
            "Units Gained": [0.0,0.0,0.0,0.0],
        })
        summary.to_excel(xw, index=False, sheet_name="Summary")

        # Insert formulas referencing columns:
        # Our column order puts Result in column I and Units Gained in column J
        from openpyxl.utils import get_column_letter
        wb = xw.book
        sh = wb["Summary"]

        market_sheets = ["AI Genius","Moneyline","Totals","Spreads"]
        for idx, m in enumerate(market_sheets, start=2):  # data starts row 2 (row 1 headers)
            row = idx
            # Wins:   =COUNTIF('<sheet>'!I:I,"Win")
            sh[f"B{row}"] = f'=COUNTIF(\'{m}\'!I:I,"Win")'
            # Losses: =COUNTIF('<sheet>'!I:I,"Loss")
            sh[f"C{row}"] = f'=COUNTIF(\'{m}\'!I:I,"Loss")'
            # Total:  =B+ C
            sh[f"D{row}"] = f"=B{row}+C{row}"
            # Win %:  =IFERROR(B/(B+C),0)
            sh[f"E{row}"] = f"=IFERROR(B{row}/D{row},0)"
            # Units Gained: =SUM('<sheet>'!J:J)
            sh[f"F{row}"] = f"=SUM(\'{m}\'!J:J)"

    # done

def save_results(df: pd.DataFrame):
    # Save CSV for app persistence
    df.to_csv(RESULTS_FILE, index=False)
    # Always refresh Excel export
    write_excel_book(df)

def calc_summary(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"win_pct":0.0,"units_won":0.0,"roi":0.0,"wins":0,"losses":0,"total":0}
    total = len(df)
    wins = (df["Result"] == "Win").sum()
    losses = (df["Result"] == "Loss").sum()
    df = df.copy()
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce").fillna(0.0)
    df["Risked"] = df["Units"].abs()
    df["PnL"] = df.apply(lambda r: r["Units"] if r["Result"] == "Win" else (-r["Units"] if r["Result"] == "Loss" else 0.0), axis=1)
    units_won = float(df["PnL"].sum())
    units_risked = float(df.loc[df["Result"].isin(["Win","Loss"]), "Risked"].sum())
    roi = (units_won/units_risked*100.0) if units_risked>0 else 0.0
    win_pct = (wins/total*100.0) if total>0 else 0.0
    return {"win_pct":win_pct,"units_won":units_won,"roi":roi,"wins":wins,"losses":losses,"total":total}

def auto_log_picks(dfs: Dict[str, pd.DataFrame], sport_name: str):
    """
    Dedup + cleanliness: only log rows that have real teams (no 'Unknown'),
    and skip exact duplicates (Sport, Market, Date/Time, Matchup, Pick, Line).
    """
    results = load_results()
    for market_label, picks in dfs.items():
        if picks is None or picks.empty:
            continue
        for _, row in picks.iterrows():
            matchup = str(row.get("Matchup",""))
            if (matchup.strip() == "") or ("Unknown" in matchup):
                continue  # skip junk
            entry = {
                "Sport": sport_name,
                "Market": market_label,                         # "AI Genius" | "Moneyline" | "Totals" | "Spreads"
                "Date/Time": row.get("Date/Time",""),
                "Matchup": matchup,
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

# ─────────────────────────────────────────────
# Market editor (per-tab)
# ─────────────────────────────────────────────
def show_market_editor(sport_name: str, market_label: str, key_prefix: str):
    """
    Collapsible editor for a single market. Saves without redirect;
    updates metrics immediately and refreshes Excel file.
    """
    results_df = load_results()
    market_df = results_df[(results_df["Sport"] == sport_name) & (results_df["Market"] == market_label)].copy()

    # Summary metrics for this market (only Win/Loss)
    msum = calc_summary(market_df[market_df["Result"].isin(["Win","Loss"])])
    c1,c2,c3 = st.columns(3)
    c1.metric(f"{market_label} Win %", f"{msum['win_pct']:.1f}% ({msum['wins']}-{msum['losses']})")
    c2.metric(f"{market_label} Units Won", f"{msum['units_won']:.1f}")
    c3.metric(f"{market_label} ROI", f"{msum['roi']:.1f}%")

    # Pending editor
    with st.expander(f"✍️ Edit Pending — {market_label}", expanded=False):
        pending = market_df[market_df["Result"] == "Pending"].copy()
        if not pending.empty:
            # Dedup display identity
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
                    label = f"{r['Date/Time']} — {r['Matchup']} — Pick: {nice_line(r['Market'], r['Pick'], r['Line'])}"
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
                            (results_df["Market"] == r["Market"]) &
                            (results_df["Date/Time"] == r["Date/Time"]) &
                            (results_df["Matchup"] == r["Matchup"]) &
                            (results_df["Pick"] == r["Pick"]) &
                            (results_df["Line"].fillna("").astype(str) == str(r["Line"]))
                        )
                        results_df.loc[mask, "Result"] = sel
                        save_results(results_df)
                        st.session_state[f"last_update_{key_prefix}"] = True
                        st.success("Saved ✅")
        else:
            st.info("No pending picks here.")

    # Completed editor (can adjust)
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
                    st.write(f"{r['Date/Time']} — {r['Matchup']} — Pick: {nice_line(r['Market'], r['Pick'], r['Line'])} ({r['Result']})")
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
                            (results_df["Market"] == r["Market"]) &
                            (results_df["Date/Time"] == r["Date/Time"]) &
                            (results_df["Matchup"] == r["Matchup"]) &
                            (results_df["Pick"] == r["Pick"]) &
                            (results_df["Line"].fillna("").astype(str) == str(r["Line"]))
                        )
                        results_df.loc[mask, "Result"] = sel
                        save_results(results_df)
                        st.session_state[f"last_update_{key_prefix}"] = True
                        st.success("Saved ✅")
        else:
            st.info("No completed picks yet.")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    if "sport_name" not in st.session_state:
        st.session_state.sport_name = list(SPORT_OPTIONS.keys())[0]

    sport_name = st.selectbox(
        "Sport",
        list(SPORT_OPTIONS.keys()),
        index=list(SPORT_OPTIONS.keys()).index(st.session_state.sport_name),
        key="sport_name"
    )
    regions = st.text_input("Regions", value=DEFAULT_REGIONS, key="regions")
    top_n = st.slider("Top picks per tab", 3, 20, 10, key="top_n")
    fetch = st.button("Fetch Live Odds")

# ─────────────────────────────────────────────
# Consensus tables + charts
# ─────────────────────────────────────────────
def consensus_tables(raw: pd.DataFrame, top_n: int):
    if raw is None or raw.empty: return (pd.DataFrame(),)*5
    cons = build_consensus(raw)
    ml = pick_best_per_event(cons,"h2h",top_n)
    totals = pick_best_per_event(cons,"totals",top_n)
    spreads = pick_best_per_event(cons,"spreads",top_n)
    ai_picks = ai_genius_top(cons,min(top_n,5))
    return ai_picks,ml,totals,spreads,cons

def confidence_bars(df: pd.DataFrame,title: str):
    if df is None or df.empty or "Confidence" not in df.columns: return
    conf_vals = df["Confidence"].str.replace("%","",regex=False).astype(float)
    lbls = df["Matchup"].astype(str)
    chart_df = pd.DataFrame({"Confidence": conf_vals.values}, index=lbls.values)
    st.caption(title)
    st.bar_chart(chart_df)

# ─────────────────────────────────────────────
# Fetch + stash in session_state
# ─────────────────────────────────────────────
if fetch:
    sport_key = SPORT_OPTIONS[sport_name]
    if isinstance(sport_key, list):
        parts = [fetch_odds(k, regions) for k in sport_key]
        raw = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    else:
        raw = fetch_odds(sport_key, regions)

    if raw.empty:
        st.warning("No data returned. Try a different sport or check API quota.")
        st.session_state.has_data = False
    else:
        ai_picks, ml, totals, spreads, cons = consensus_tables(raw, top_n)
        # Log to results (dedup-safe, skip Unknowns)
        auto_log_picks(
            {"AI Genius": ai_picks, "Moneyline": ml, "Totals": totals, "Spreads": spreads},
            sport_name
        )
        # Stash everything
        st.session_state.raw = raw
        st.session_state.ai_picks = ai_picks
        st.session_state.ml = ml
        st.session_state.totals = totals
        st.session_state.spreads = spreads
        st.session_state.cons = cons
        st.session_state.has_data = True

# ─────────────────────────────────────────────
# Render from session
# ─────────────────────────────────────────────
if st.session_state.get("has_data", False):
    tabs = st.tabs([
        "🤖 AI Genius Picks",
        "Moneylines",
        "Totals",
        "Spreads",
        "Raw Data",
        "📊 Results"
    ])

    # Tab 0 — AI Genius
    with tabs[0]:
        st.subheader("AI Genius — Highest Consensus Confidence (Top)")
        st.dataframe(st.session_state.ai_picks, use_container_width=True, hide_index=True)
        confidence_bars(st.session_state.ai_picks, "Confidence heat — AI Genius")
        st.markdown("---")
        st.markdown("### ✍️ Manual Result Editor — AI Genius")
        show_market_editor(sport_name, "AI Genius", key_prefix="ai")

    # Tab 1 — Moneylines
    with tabs[1]:
        st.subheader("Best Moneyline per Game (Consensus)")
        st.dataframe(st.session_state.ml, use_container_width=True, hide_index=True)
        confidence_bars(st.session_state.ml, "Confidence heat — Moneylines")
        st.markdown("---")
        st.markdown("### ✍️ Manual Result Editor — Moneyline")
        show_market_editor(sport_name, "Moneyline", key_prefix="ml")

    # Tab 2 — Totals
    with tabs[2]:
        st.subheader("Best Totals per Game (Consensus)")
        st.dataframe(st.session_state.totals, use_container_width=True, hide_index=True)
        confidence_bars(st.session_state.totals, "Confidence heat — Totals")
        st.markdown("---")
        st.markdown("### ✍️ Manual Result Editor — Totals")
        show_market_editor(sport_name, "Totals", key_prefix="tot")

    # Tab 3 — Spreads
    with tabs[3]:
        st.subheader("Best Spreads per Game (Consensus)")
        st.dataframe(st.session_state.spreads, use_container_width=True, hide_index=True)
        confidence_bars(st.session_state.spreads, "Confidence heat — Spreads")
        st.markdown("---")
        st.markdown("### ✍️ Manual Result Editor — Spreads")
        show_market_editor(sport_name, "Spreads", key_prefix="spr")

    # Tab 4 — Raw
    with tabs[4]:
        st.subheader("Raw Per-Book Odds (first 200 rows)")
        st.dataframe(st.session_state.raw.head(200), use_container_width=True, hide_index=True)
        st.caption("Tip: this is the source that feeds the consensus tables.")

    # Tab 5 — Results + Excel download
    with tabs[5]:
        st.subheader(f"📊 Results — {sport_name}")
        results = load_results()
        if results.empty:
            st.info("No bets logged yet.")
        else:
            st.dataframe(results, use_container_width=True, hide_index=True)
            summ = calc_summary(results[results["Result"].isin(["Win","Loss"])])
            c1,c2,c3 = st.columns(3)
            c1.metric("Win %", f"{summ['win_pct']:.1f}% ({summ['wins']}-{summ['losses']})")
            c2.metric("Units Won", f"{summ['units_won']:.1f}")
            c3.metric("ROI", f"{summ['roi']:.1f}%")

            # Ensure Excel is current
            write_excel_book(results)

            # Download button
            try:
                with open(EXCEL_FILE, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Results (Excel)",
                        data=f.read(),
                        file_name="bets.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except FileNotFoundError:
                st.warning("Excel file not found yet — add some picks first.")

else:
    st.info("Pick a sport and click **Fetch Live Odds**")
