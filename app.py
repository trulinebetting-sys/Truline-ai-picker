import os
from typing import Dict, Any, Optional, List, Tuple
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
st.caption("Two charts only: AI Top 5 across all sports + 5 Parlays (ensemble-ranked).")
st.divider()

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "") or "1d677dc98d978ccc24d9914d835442f1"
DEFAULT_REGIONS = os.getenv("REGIONS", "us")
MARKETS = "h2h,spreads,totals"

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

# Filter out absurd odds (you requested to avoid things like -30000)
ODDS_MIN_US = -2000
ODDS_MAX_US = 2000

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
                mkey = mk.get("key")
                for oc in mk.get("outcomes", []):
                    rows.append({
                        "sport_key": sport_key,
                        "event_id": event_id,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": mkey,                     # "h2h" | "spreads" | "totals"
                        "outcome": oc.get("name"),          # "Home"/"Away" or "Over"/"Under"
                        "line": oc.get("point"),            # spread/total line; None for ML
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

def fetch_all_enabled_sports(regions: str) -> pd.DataFrame:
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
    return pd.concat(frames, ignore_index=True)

# ─────────────────────────────────────────────
# Consensus + Ensemble
# ─────────────────────────────────────────────
def build_consensus(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

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

    # Odds sanity filter
    out = out[(out["best_odds_us"] >= ODDS_MIN_US) & (out["best_odds_us"] <= ODDS_MAX_US)]

    return out[[
        "sport", "sport_key", "event_id", "commence_time", "Date/Time", "Matchup",
        "market", "outcome", "line",
        "best_book", "best_odds_us", "best_odds_dec", "consensus_conf", "books", "avg_odds_dec"
    ]]

def ensemble_score(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    v1 = d["consensus_conf"].astype(float).fillna(0.0).clip(0.0, 1.0)
    edge = (d["best_odds_dec"] - d["avg_odds_dec"]).astype(float)
    v2 = 1.0 / (1.0 + np.exp(-6.0 * edge.fillna(0.0)))
    v3 = (d["books"].astype(float).clip(0.0, 10.0)) / 10.0
    dec = d["best_odds_dec"].astype(float).fillna(2.0)
    v4 = np.exp(-((dec - 2.0) ** 2) / (2 * (0.6 ** 2)))
    v4 = (v4 - v4.min()) / (v4.max() - v4.min() + 1e-9)

    d["EnsembleScore"] = (v1 + v2 + v3 + v4) / 4.0
    d["Units"] = d["EnsembleScore"].apply(assign_units_from_score)
    return d

def pool_candidates(cons: pd.DataFrame) -> pd.DataFrame:
    valid = cons[cons["market"].isin(["h2h", "totals", "spreads"])].copy()
    if valid.empty:
        return valid
    scored = ensemble_score(valid)
    idx = scored.groupby(["event_id", "market"])["EnsembleScore"].idxmax()
    best = scored.loc[idx].copy()

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
    market_map = {"h2h": "Moneyline", "spreads": "Spreads", "totals": "Totals"}
    best["Market"] = best["market"].map(market_map)
    best = best.sort_values("EnsembleScore", ascending=False)
    return best

def select_top_ai(best_pool: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    if best_pool.empty:
        return pd.DataFrame()
    filtered = best_pool[(best_pool["best_odds_us"] >= ODDS_MIN_US) & (best_pool["best_odds_us"] <= ODDS_MAX_US)].copy()
    top = filtered.head(k).copy()
    top["Confidence"] = top["consensus_conf"].apply(fmt_pct)
    top["Odds (US)"] = top["best_odds_us"].astype(int)
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
    parlays = []
    if candidate_pool.empty:
        return parlays
    pool = candidate_pool.copy()
    for legs in leg_plan:
        used = set()
        legs_rows = []
        for _, r in pool.iterrows():
            if len(legs_rows) >= legs:
                break
            if r["event_id"] in used:
                continue
            legs_rows.append(r)
            used.add(r["event_id"])
        if len(legs_rows) < legs:
            continue
        dec_odds = 1.0
        total_units = 0.0
        for rr in legs_rows:
            d = rr["best_odds_dec"]
            if pd.isna(d) or d <= 1.0:
                d = max(1.01, float(d) if not pd.isna(d) else 1.01)
            dec_odds *= d
            total_units += float(rr["Units"])
        am = decimal_to_american(dec_odds)
        parlays.append({
            "legs": legs_rows,
            "legs_count": legs,
            "parlay_decimal": round(dec_odds, 4),
            "parlay_american": am,
            "suggested_units": round(max(0.5, min(5.0, total_units * 0.4)), 1)
        })
        pool = pool[~pool["event_id"].isin(used)].copy()
        if pool.empty:
            break
    return parlays

def parlays_to_dataframe(parlays: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for idx, p in enumerate(parlays, start=1):
        leg_summaries = []
        for rr in p["legs"]:
            part = f"{rr['Sport'] if 'Sport' in rr else rr.get('sport','')} • {rr['Matchup']} • {('Moneyline' if rr['market']=='h2h' else ('Spreads' if rr['market']=='spreads' else 'Totals'))}: {rr['Pick']}"
            if rr["market"] in ["spreads", "totals"] and str(rr['Line']).strip() not in ["", "None", "nan"]:
                part += f" ({rr['Line']})"
            part += f" @ {int(rr['best_odds_us'])}"
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
        for col in ["Sport","Market","Date/Time","Matchup","Pick","Line","Odds (US)","Units","Result"]:
            if col not in df.columns:
                df[col] = "" if col != "Units" else 1.0
        df["Result"] = df["Result"].fillna("Pending")
        return df
    return pd.DataFrame(columns=["Sport","Market","Date/Time","Matchup","Pick","Line","Odds (US)","Units","Result"])

def save_results_csv(df: pd.DataFrame):
    df.to_csv(CSV_FILE, index=False)

def auto_log_ai_picks(ai5: pd.DataFrame):
    if ai5 is None or ai5.empty:
        return
    results = load_results()
    # Ensure all expected columns exist
    for col in ["Sport","Market","Date/Time","Matchup","Pick","Line","Odds (US)","Units","Result"]:
        if col not in results.columns:
            results[col] = "" if col != "Units" else 1.0

    for _, r in ai5.iterrows():
        entry = {
            "Sport": r.get("Sport",""),
            "Market": r.get("Market",""),
            "Date/Time": r.get("Date/Time",""),
            "Matchup": r.get("Matchup",""),
            "Pick": r.get("Pick",""),
            "Line": r.get("Line",""),
            "Odds (US)": r.get("Odds (US)",""),
            "Units": float(r.get("Units", 1.0)) if str(r.get("Units","")).strip() != "" else 1.0,
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
    Builds bets.xlsx with:
      - AI_Picks (pretty)
      - Parlays
      - Results (Summary in row 2, headers row 3, data from row 4)
        + Auto formulas in B2..G2
        + Conditional formatting: Result column (I) green Win, red Loss
    """
    from io import BytesIO
    from openpyxl.styles import PatternFill, Font
    from openpyxl.formatting.rule import Rule, DifferentialStyle

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        # Sheet 1: AI picks
        a = ai5.copy()
        if not a.empty:
            a = a.rename(columns={"Odds (US)":"Odds_US","Odds (Dec)":"Odds_Dec"})
            if "Result" not in a.columns:
                a["Result"] = ""
            a.to_excel(writer, index=False, sheet_name="AI_Picks")
        else:
            pd.DataFrame(columns=["Message"]).to_excel(writer, index=False, sheet_name="AI_Picks")

        # Sheet 2: Parlays
        (parlays_df if parlays_df is not None else pd.DataFrame(columns=["Message"])) \
            .to_excel(writer, index=False, sheet_name="Parlays")

        # Sheet 3: Results — write header at row 3 (startrow=2), so data begins row 4
        results_df = results_df.copy()
        # Ensure column order
        desired_cols = ["Date/Time","Sport","Matchup","Market","Pick","Odds (US)","Odds (Dec)","Units","Result","Line"]
        # Add missing if any
        for c in desired_cols:
            if c not in results_df.columns:
                results_df[c] = ""
        results_df = results_df[desired_cols]
        results_df.to_excel(writer, index=False, sheet_name="Results", startrow=2)
        ws = writer.sheets["Results"]

        # Row 2 summary labels
        ws["A2"] = "SUMMARY"
        ws["A2"].font = Font(bold=True)

        # Define ranges (start at row 4 downwards; Excel uses 1-based rows)
        # Result column is I, Units column is H
        max_row = 100000  # generous range
        rng_result = f"I4:I{max_row}"
        rng_units  = f"H4:H{max_row}"

        # B2 = Total
        ws["B2"] = f"=COUNTA(A4:A{max_row})"
        # C2 = Wins
        ws["C2"] = f'=COUNTIF({rng_result},"Win")'
        # D2 = Losses
        ws["D2"] = f'=COUNTIF({rng_result},"Loss")'
        # E2 = Win %
        ws["E2"] = "=IF((C2+D2)=0,0,C2/(C2+D2))"
        ws["E2"].number_format = "0.0%"

        # F2 = Units Won  -> Wins*Units - Losses*Units
        ws["F2"] = f'=SUMPRODUCT(--({rng_result}="Win"),{rng_units}) - SUMPRODUCT(--({rng_result}="Loss"),{rng_units})'
        # G2 = ROI % -> Units Won / Risked (risked = sum of Units for Win/Loss only)
        ws["G2"] = f'=IF(SUMPRODUCT(--(({rng_result}="Win")+({rng_result}="Loss")),{rng_units})=0,0, F2 / SUMPRODUCT(--(({rng_result}="Win")+({rng_result}="Loss")),{rng_units}))'
        ws["G2"].number_format = "0.0%"

        # Conditional formatting for Result column: green Win, red Loss
        green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
        red_fill   = PatternFill(start_color="F8CBAD", end_color="F8CBAD", fill_type="solid")
        dxf_green = DifferentialStyle(fill=green_fill)
        dxf_red   = DifferentialStyle(fill=red_fill)

        rule_win = Rule(type="containsText", operator="containsText", text="Win", dxf=dxf_green, stopIfTrue=False)
        rule_win.formula = ['NOT(ISERROR(SEARCH("Win",$I4)))']
        rule_loss = Rule(type="containsText", operator="containsText", text="Loss", dxf=dxf_red, stopIfTrue=False)
        rule_loss.formula = ['NOT(ISERROR(SEARCH("Loss",$I4)))']

        ws.conditional_formatting.add(rng_result, rule_win)
        ws.conditional_formatting.add(rng_result, rule_loss)

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

        # Log AI picks
        auto_log_ai_picks(ai5)

        # Parlays from the first 40 items of the pool for variety
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
            show = show.rename(columns={"EnsembleScore": "Score"})
            show["Score"] = show["Score"].map(lambda x: f"{x:.3f}")
            st.dataframe(show.drop(columns=["event_id","market"]), use_container_width=True, hide_index=True)
        else:
            st.info("No AI picks available.")

    # Tab 1 — Parlays
    with tabs[1]:
        st.subheader("Parlays (ensemble-built)")
        if st.session_state.parlays_df is not None and not st.session_state.parlays_df.empty:
            st.dataframe(st.session_state.parlays_df, use_container_width=True, hide_index=True)
        else:
            st.info("No parlays could be built from the current pool.")

    # Tab 2 — Export (writes formulas & conditional formats)
    with tabs[2]:
        st.subheader("Export to Excel (auto-formulas + green/red formatting)")
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
