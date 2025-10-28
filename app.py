import os
from typing import Dict, Any, Optional
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from io import BytesIO

# Excel tools
from openpyxl import load_workbook, Workbook
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

# ─────────────────────────────────────────────
# Safe dotenv
# ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ─────────────────────────────────────────────
# Constants / Config
# ─────────────────────────────────────────────
ODDS_API_KEY = "1d677dc98d978ccc24d9914d835442f1"
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

RESULTS_FILE = "bets.csv"
EXCEL_FILE = "bets.xlsx"

# Excel sheet names to keep consistent everywhere
SHEETS = ["AI Genius", "Moneyline", "Totals", "Spreads"]

# Column layout we’ll write to Excel
EXCEL_COLS = [
    "Date/Time",   # A
    "Sport",       # B
    "Market",      # C
    "Matchup",     # D
    "Pick",        # E
    "Line",        # F
    "Odds (US)",   # G
    "Units",       # H
    "Result",      # I  (dropdown)
    "Units Gained" # J  (formula)
]

# ─────────────────────────────────────────────
# Streamlit app chrome
# ─────────────────────────────────────────────
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker 🚀")
st.caption("Consensus across books + live odds + AI-style ranking. Tracks results + bankroll ✅")
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

def identity_key(row: pd.Series) -> str:
    """Stable identity for a pick to dedup/merge across CSV and Excel."""
    return f"{row.get('Sport','')}|{row.get('Market','')}|{row.get('Date/Time','')}|{row.get('Matchup','')}|{row.get('Pick','')}|{'' if pd.isna(row.get('Line')) else row.get('Line')}"

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
                        "outcome": oc.get("name"),
                        "line": oc.get("point"),
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
    if raw.empty:
        return raw
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
    if sub.empty:
        return pd.DataFrame()
    best_idx = sub.groupby("event_id")["Confidence"].idxmax()
    sub = sub.loc[best_idx].copy()
    sub = sub.sort_values("commence_time", ascending=True).head(top_n)
    out = sub[["Date/Time","Matchup","Sportsbook","outcome","line","Odds (US)","Odds (Dec)","Confidence","Books"]].copy()
    out = out.rename(columns={"outcome":"Pick","line":"Line"})
    out["Confidence"] = out["Confidence"].apply(fmt_pct)
    out["Units"] = sub["Confidence"].apply(assign_units)
    return out.reset_index(drop=True)

def ai_genius_top(cons_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if cons_df.empty:
        return pd.DataFrame()
    frames = []
    for m in ["h2h","totals","spreads"]:
        t = pick_best_per_event(cons_df, m, top_n*3)
        if not t.empty:
            t["Market"] = m
            t["_C"] = t["Confidence"].str.replace("%","",regex=False).astype(float)
            frames.append(t)
    if not frames:
        return pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True)
    allp = allp.sort_values("_C", ascending=False).drop(columns=["_C"]).head(top_n)
    return allp.reset_index(drop=True)

# ─────────────────────────────────────────────
# Results persistence (CSV) + summaries
# ─────────────────────────────────────────────
def ensure_results_schema(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["Sport","Market","Date/Time","Matchup","Pick","Line","Odds (US)","Units","Result"]
    for c in needed:
        if c not in df.columns:
            df[c] = "" if c not in ["Units"] else 1.0
    # Normalize types
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce").fillna(1.0).astype(float)
    df["Result"] = df["Result"].replace({np.nan: "Pending"}).astype(str)
    return df[needed]

def load_results() -> pd.DataFrame:
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        df = ensure_results_schema(df)
        return df
    return ensure_results_schema(pd.DataFrame(columns=[]))

def save_results(df: pd.DataFrame):
    df = ensure_results_schema(df.copy())
    df.to_csv(RESULTS_FILE, index=False)
    # Also export fresh Excel
    export_results_to_excel(df)

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

def auto_log_picks(dfs: Dict[str, pd.DataFrame], sport_name: str):
    """Dedup logging by identity; ignore Unknown matchups."""
    results = load_results()
    before = len(results)
    for market_label, picks in dfs.items():
        if picks is None or picks.empty:
            continue
        # Normalize columns for CSV
        picks = picks.copy()
        picks["Sport"] = sport_name
        picks["Market"] = market_label
        picks["Result"] = "Pending"
        # Remove unknowns (they clutter Excel and results)
        picks = picks[~picks["Matchup"].str.contains("Unknown", na=False)]
        # Ensure same columns
        for c in ["Date/Time","Matchup","Pick","Line","Odds (US)","Units","Sport","Market","Result"]:
            if c not in picks.columns:
                picks[c] = ""
        # Dedup by identity
        results["_id"] = results.apply(identity_key, axis=1)
        picks["_id"] = picks.apply(identity_key, axis=1)
        new_rows = picks[~picks["_id"].isin(results["_id"])]
        if not new_rows.empty:
            results = pd.concat([results.drop(columns=["_id"], errors="ignore"), new_rows.drop(columns=["_id"])], ignore_index=True)
    # Save only if changed
    if len(results) != before:
        save_results(results)

# ─────────────────────────────────────────────
# Excel export / import
# ─────────────────────────────────────────────
def _ensure_workbook(path: str) -> Workbook:
    if os.path.exists(path):
        try:
            return load_workbook(path)
        except Exception:
            pass
    wb = Workbook()
    # Default sheet will be present; let’s clear it
    ws = wb.active
    ws.title = "Summary"
    return wb

def _write_sheet_df(wb: Workbook, sheet_name: str, df: pd.DataFrame):
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)
    # Header
    ws.append(EXCEL_COLS)
    # Rows
    for _, r in df.iterrows():
        # Units Gained formula will be added later after data validation
        ws.append([
            r.get("Date/Time",""),
            r.get("Sport",""),
            r.get("Market",""),
            r.get("Matchup",""),
            r.get("Pick",""),
            r.get("Line",""),
            r.get("Odds (US)",""),
            float(r.get("Units",1.0)),
            r.get("Result","Pending"),
            0  # placeholder for formula, will set after
        ])
    return ws

def _apply_result_dropdown_and_formulas(ws):
    # Result column is "I"
    result_col = 9  # I
    units_col = 8   # H
    units_gained_col = 10  # J
    # Validation dropdown for I2:I{max}
    dv = DataValidation(type="list", formula1='"Pending,Win,Loss"', allow_blank=True)
    ws.add_data_validation(dv)
    max_row = ws.max_row if ws.max_row > 1 else 2
    for row in range(2, max_row+1):
        cell_res = ws.cell(row=row, column=result_col)
        dv.add(cell_res)
        # Units Gained formula in J: =IF($I2="Win",$H2,IF($I2="Loss",- $H2,0))
        cell_units_gained = ws.cell(row=row, column=units_gained_col)
        cell_units_gained.value = f'=IF($I{row}="Win",$H{row},IF($I{row}="Loss",- $H{row},0))'
    # Add Win% label+formula to K1:K2 (K = 11)
    ws.cell(row=1, column=11).value = "Win%"
    ws.cell(row=2, column=11).value = '=IFERROR(COUNTIF($I:$I,"Win")/(COUNTIF($I:$I,"Win")+COUNTIF($I:$I,"Loss")),0)'
    # Autosize a bit
    for col in range(1, ws.max_column+1):
        ws.column_dimensions[get_column_letter(col)].width = 18

def _write_summary_sheet(wb: Workbook):
    # Delete existing Summary and recreate last so it's first tab
    if "Summary" in wb.sheetnames:
        del wb["Summary"]
    ws = wb.create_sheet("Summary", 0)
    ws.append(["Market","Wins","Losses","Total","Win%","Units (Sum of Units Gained)"])
    # For each sheet, insert formulas referencing that sheet
    for m in SHEETS:
        if m in wb.sheetnames:
            # Wins: COUNTIF(m!I:I,"Win")
            # Losses: COUNTIF(m!I:I,"Loss")
            # Total = Wins + Losses
            # Win% = IFERROR(Wins/Total,0)
            # Units = SUM(m!J:J)
            row = [
                m,
                f'=COUNTIF(\'{m}\'!$I:$I,"Win")',
                f'=COUNTIF(\'{m}\'!$I:$I,"Loss")',
                f'=B{ws.max_row+1}+C{ws.max_row+1}',  # We’ll fix after append
                f'=IFERROR(B{ws.max_row+1}/D{ws.max_row+1},0)',
                f'=SUM(\'{m}\'!$J:$J)'
            ]
            ws.append(row)
            # Fix the formulas referencing the just-added row
            r = ws.max_row
            ws.cell(row=r, column=4).value = f"=B{r}+C{r}"
            ws.cell(row=r, column=5).value = f"=IFERROR(B{r}/D{r},0)"
    # Autosize columns
    for col in range(1, ws.max_column+1):
        ws.column_dimensions[get_column_letter(col)].width = 28

def export_results_to_excel(results: pd.DataFrame):
    """Export current CSV results into a structured Excel with sheets and validation."""
    results = ensure_results_schema(results.copy())
    # Ensure all required columns exist
    for c in EXCEL_COLS:
        if c not in results.columns:
            results[c] = ""
    # Create workbook (or reuse)
    wb = _ensure_workbook(EXCEL_FILE)
    # Write each market sheet
    for m in SHEETS:
        df_m = results[results["Market"] == m].copy()
        if df_m.empty:
            # Still write headers for consistency
            ws = _write_sheet_df(wb, m, df_m)
            _apply_result_dropdown_and_formulas(ws)
            continue
        # Order/trim columns for Excel
        df_m = df_m.reindex(columns=EXCEL_COLS, fill_value="")
        ws = _write_sheet_df(wb, m, df_m)
        _apply_result_dropdown_and_formulas(ws)
    # Summary sheet
    _write_summary_sheet(wb)
    wb.save(EXCEL_FILE)

def sync_results_from_excel() -> bool:
    """
    Read bets.xlsx (four sheets), update CSV 'Result' and 'Units'
    based on matches by identity key. Returns True if something changed.
    """
    if not os.path.exists(EXCEL_FILE):
        return False
    try:
        frames = []
        for m in SHEETS:
            try:
                df = pd.read_excel(EXCEL_FILE, sheet_name=m, engine="openpyxl")
            except Exception:
                df = pd.DataFrame(columns=EXCEL_COLS)
            if df.empty:
                continue
            # Normalize expected columns
            for c in EXCEL_COLS:
                if c not in df.columns:
                    df[c] = "" if c not in ["Units"] else 1.0
            # Only allow valid results
            df["Result"] = df["Result"].fillna("Pending").astype(str)
            df.loc[~df["Result"].isin(["Pending","Win","Loss"]), "Result"] = "Pending"
            # Keep only columns we need to merge
            frames.append(df[EXCEL_COLS].copy())
        if not frames:
            return False
        excel_all = pd.concat(frames, ignore_index=True)
        # Build identity keys
        excel_all["_id"] = excel_all.apply(identity_key, axis=1)
        results = load_results()
        results["_id"] = results.apply(identity_key, axis=1)

        changed = False
        # Merge updates: Result & Units if identity matches
        excel_map = excel_all.set_index("_id")
        for idx, row in results.iterrows():
            k = row["_id"]
            if k in excel_map.index:
                new_res = excel_map.at[k, "Result"]
                new_units = excel_map.at[k, "Units"]
                if new_res != row["Result"] or float(new_units) != float(row["Units"]):
                    results.at[idx, "Result"] = new_res
                    results.at[idx, "Units"] = float(new_units)
                    changed = True
        if changed:
            save_results(results.drop(columns=["_id"]))
        return changed
    except Exception:
        return False

# ─────────────────────────────────────────────
# Sidebar controls
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
    st.markdown("---")
    st.caption("Excel sync")
    sync_btn = st.button("🔄 Sync Results from Excel → App")

# ─────────────────────────────────────────────
# Compute tables
# ─────────────────────────────────────────────
def consensus_tables(raw: pd.DataFrame, top_n: int):
    if raw is None or raw.empty:
        return (pd.DataFrame(),)*5
    cons = build_consensus(raw)
    ml = pick_best_per_event(cons,"h2h",top_n)
    totals = pick_best_per_event(cons,"totals",top_n)
    spreads = pick_best_per_event(cons,"spreads",top_n)
    ai_picks = ai_genius_top(cons, min(top_n,5))
    return ai_picks, ml, totals, spreads, cons

# ─────────────────────────────────────────────
# Fetch data, log picks, export Excel
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
        # Log to results (dedup-safe)
        to_log = {
            "AI Genius": ai_picks,
            "Moneyline": ml,
            "Totals": totals,
            "Spreads": spreads
        }
        # Ensure we only log the columns we need
        for label, dfp in to_log.items():
            if not dfp.empty:
                # Align to EXCEL_COLS/CSV cols
                dfp = dfp.rename(columns={"Pick":"Pick","Line":"Line"})
                for c in ["Date/Time","Matchup","Pick","Line","Odds (US)","Units"]:
                    if c not in dfp.columns:
                        dfp[c] = ""
                to_log[label] = dfp[["Date/Time","Matchup","Pick","Line","Odds (US)","Units"]]

        auto_log_picks(to_log, sport_name)

        # Stash for UI
        st.session_state.raw = raw
        st.session_state.ai_picks = ai_picks
        st.session_state.ml = ml
        st.session_state.totals = totals
        st.session_state.spreads = spreads
        st.session_state.has_data = True

# Sync from Excel to CSV if requested
if sync_btn:
    changed = sync_results_from_excel()
    if changed:
        st.success("Synced Excel changes into the app ✅")
    else:
        st.info("No changes detected from Excel.")

# ─────────────────────────────────────────────
# UI Tabs
# ─────────────────────────────────────────────
def confidence_bars(df: pd.DataFrame, title: str):
    if df is None or df.empty or "Confidence" not in df.columns:
        return
    conf_vals = df["Confidence"].str.replace("%","",regex=False).astype(float)
    lbls = df["Matchup"].astype(str)
    chart_df = pd.DataFrame({"Confidence": conf_vals.values}, index=lbls.values)
    st.caption(title)
    st.bar_chart(chart_df)

def show_market_summary_block(sport_name: str, market_label: str):
    res = load_results()
    mdf = res[(res["Sport"] == sport_name) & (res["Market"] == market_label)].copy()
    msum = calc_summary(mdf[mdf["Result"].isin(["Win","Loss"])])
    c1,c2,c3 = st.columns(3)
    c1.metric(f"{market_label} Win %", f"{msum['win_pct']:.1f}% ({msum['wins']}-{msum['losses']})")
    c2.metric(f"{market_label} Units Won", f"{msum['units_won']:.1f}")
    c3.metric(f"{market_label} ROI", f"{msum['roi']:.1f}%")

if st.session_state.get("has_data", False):
    tabs = st.tabs([
        "🤖 AI Genius Picks",
        "Moneylines",
        "Totals",
        "Spreads",
        "Raw Data",
        "📊 Results / Excel"
    ])

    # Tab 0 — AI Genius
    with tabs[0]:
        st.subheader("AI Genius — Highest Consensus Confidence")
        st.dataframe(st.session_state.ai_picks, use_container_width=True, hide_index=True)
        confidence_bars(st.session_state.ai_picks, "Confidence heat — AI Genius")
        st.markdown("---")
        show_market_summary_block(sport_name, "AI Genius")

    # Tab 1 — Moneylines
    with tabs[1]:
        st.subheader("Best Moneyline per Game (Consensus)")
        st.dataframe(st.session_state.ml, use_container_width=True, hide_index=True)
        confidence_bars(st.session_state.ml, "Confidence heat — Moneylines")
        st.markdown("---")
        show_market_summary_block(sport_name, "Moneyline")

    # Tab 2 — Totals
    with tabs[2]:
        st.subheader("Best Totals per Game (Consensus)")
        st.dataframe(st.session_state.totals, use_container_width=True, hide_index=True)
        confidence_bars(st.session_state.totals, "Confidence heat — Totals")
        st.markdown("---")
        show_market_summary_block(sport_name, "Totals")

    # Tab 3 — Spreads
    with tabs[3]:
        st.subheader("Best Spreads per Game (Consensus)")
        st.dataframe(st.session_state.spreads, use_container_width=True, hide_index=True)
        confidence_bars(st.session_state.spreads, "Confidence heat — Spreads")
        st.markdown("---")
        show_market_summary_block(sport_name, "Spreads")

    # Tab 4 — Raw
    with tabs[4]:
        st.subheader("Raw Per-Book Odds (first 200 rows)")
        st.dataframe(st.session_state.raw.head(200), use_container_width=True, hide_index=True)
        st.caption("Tip: this is the source that feeds the consensus tables.")

    # Tab 5 — Results / Excel
    with tabs[5]:
        st.subheader("Results & Excel")
        results_now = load_results()
        if results_now.empty:
            st.info("No results yet — click **Fetch Live Odds** first.")
        else:
            st.write("Preview of logged results:")
            st.dataframe(results_now, use_container_width=True, hide_index=True)

        st.markdown("#### Download latest Excel")
        # Make sure file exists (export again, then serve)
        export_results_to_excel(results_now)
        if os.path.exists(EXCEL_FILE):
            with open(EXCEL_FILE, "rb") as f:
                st.download_button(
                    label="⬇️ Download `bets.xlsx`",
                    data=f.read(),
                    file_name="bets.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("No Excel file yet. Fetch odds first.")

        st.markdown("---")
        st.markdown("#### How to update results")
        st.caption("""
1) **Download** the Excel file above.  
2) On each sheet (AI Genius, Moneyline, Totals, Spreads), change the **Result** dropdown to `Win` or `Loss`.  
3) (Optional) Adjust **Units** for specific picks.  
4) **Save** the Excel file.  
5) Click **Sync Results from Excel → App** (left sidebar).  
6) The metrics on tabs will update immediately.
        """)

else:
    st.info("Pick a sport and click **Fetch Live Odds**")
