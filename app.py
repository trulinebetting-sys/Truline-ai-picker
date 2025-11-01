import os
import json
import math
from datetime import datetime, timezone, date
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ─────────────────────────────────────────────
# Page & Global Config
# ─────────────────────────────────────────────
st.set_page_config(page_title="TruLine – Daily 5 Picks Engine", layout="wide")
st.title("TruLine – Daily 5 Picks Engine 🚀")
st.caption("Auto-generate 5 daily picks per chart using an internal scoring model. Includes AI Top 5 + SGP Parlays + Excel export.")
st.divider()

# ─────────────────────────────────────────────
# Env & Secrets
# ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "") or "1d677dc98d978ccc24d9914d835442f1"
DEFAULT_REGIONS = os.getenv("REGIONS", "us")

# Supported sports (you can extend later)
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

# Excel output file (written to working dir and also downloadable)
EXCEL_FILE = "daily_picks.xlsx"

# ─────────────────────────────────────────────
# Helpers: Odds conversions & probabilities
# ─────────────────────────────────────────────
def american_to_decimal(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    o = float(odds)
    return 1.0 + (o / 100.0) if o > 0 else 1.0 + (100.0 / abs(o))

def implied_prob_from_american(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)

def pct(x: float) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{100.0 * x:.1f}%"

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

# Asymmetric Kelly-ish unit sizing from a 0..1 confidence score
def units_from_score(score: float) -> float:
    # score in [0,1] → units in [0.5, 5.0]
    return round(0.5 + 4.5 * clamp01(score), 1)

# ─────────────────────────────────────────────
# Fetch: Odds API
# ─────────────────────────────────────────────
def _odds_get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY.")
        return None
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=120)
def fetch_odds(sport_key: str, regions: str, markets: str) -> pd.DataFrame:
    """
    Returns a normalized dataframe with columns:
    event_id, commence_time, Date/Time, home_team, away_team, book, market, outcome, line, odds_american, odds_decimal, implied_prob
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    data = _odds_get(url, {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,  # e.g. "h2h,spreads,totals,player_props"
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
            book = bk.get("title", "Unknown")
            for mk in bk.get("markets", []):
                mkey = mk.get("key")  # "h2h", "spreads", "totals", maybe "player_props" for some sports
                for oc in mk.get("outcomes", []):
                    odds_us = oc.get("price")
                    line = oc.get("point")
                    name = oc.get("name")
                    rows.append({
                        "event_id": event_id,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": mkey,
                        "outcome": name,
                        "line": line,
                        "odds_american": odds_us,
                        "odds_decimal": american_to_decimal(odds_us),
                        "implied_prob": implied_prob_from_american(odds_us),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize dates → ET string
    df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(df["commence_time"]):
        df["Date/Time"] = df["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    else:
        # assume UTC then convert
        df["Date/Time"] = df["commence_time"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")

    return df

# ─────────────────────────────────────────────
# AI Scoring Model (lightweight, transparent)
# ─────────────────────────────────────────────
def score_row(row: pd.Series, market: str) -> float:
    """
    A simple, transparent score in [0,1] that prefers:
    - higher dec odds (not extreme) * lower implied prob  (underdogs with some value)
    - line proximity for spreads/totals: smaller |line| is slightly preferred
    - boosts if multiple books agree (handled later in consensus)
    """
    dec = row.get("best_odds_dec", np.nan)
    imp = row.get("consensus_prob", np.nan)
    line = row.get("line", None)

    base = 0.0
    if not pd.isna(dec) and not pd.isna(imp):
        # Favor dec odds ~1.9–2.5 range; convert dec to a soft utility
        # u_dec = logistic centered at 2.05
        u_dec = 1.0 / (1.0 + math.exp(-(dec - 2.05) * 2.5))
        # Value edge: market-implied vs. consensus probability (if consensus < market implied, we see edge)
        # Here imp is the consensus (avg of books' implied). Prefer lower consensus prob *with* usable odds.
        u_val = 1.0 - imp  # lower implied → more edge
        base = 0.55 * u_dec + 0.45 * u_val
    else:
        base = 0.4  # fallback

    # Line proximity preference (smaller absolute line is slightly preferred for spreads/totals)
    if market in ("spreads", "totals"):
        try:
            ln = abs(float(line)) if line is not None else 0.0
        except Exception:
            ln = 0.0
        proximity = 1.0 / (1.0 + 0.15 * ln)  # smaller line → closer to 1.0
        base = 0.8 * base + 0.2 * proximity

    # Clamp
    return clamp01(base)

def consensus_view(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Builds per-outcome consensus & best-book snapshots.
    """
    if raw.empty:
        return raw

    # Best-book odds per outcome
    idx_best = raw.groupby(["event_id", "market", "outcome", "line"])["odds_decimal"].idxmax()
    best = raw.loc[idx_best, ["event_id","market","outcome","line","book","odds_american","odds_decimal"]].copy()
    best = best.rename(columns={
        "book":"best_book",
        "odds_american":"best_odds_us",
        "odds_decimal":"best_odds_dec"
    })

    # Consensus probability (avg of implied probs across books)
    agg = raw.groupby(["event_id","market","outcome","line"], dropna=False).agg(
        consensus_prob=("implied_prob","mean"),
        nbooks=("book","nunique"),
        home=("home_team","first"),
        away=("away_team","first"),
        commence=("commence_time","first"),
        dt=("Date/Time","first"),
    ).reset_index()

    out = agg.merge(best, on=["event_id","market","outcome","line"], how="left")
    out["Matchup"] = out["home"] + " vs " + out["away"]
    out["Date/Time"] = out["dt"]

    # "market_label" to match your tabs naming later
    out["market_label"] = out["market"].map({
        "h2h":"Moneyline",
        "spreads":"Spreads",
        "totals":"Totals",
        "player_props":"Player Props"
    }).fillna(out["market"])

    return out

# ─────────────────────────────────────────────
# Pick Builders (Top 5 per chart)
# ─────────────────────────────────────────────
def pick_top5(cons: pd.DataFrame, market_key: str) -> pd.DataFrame:
    sub = cons[cons["market"] == market_key].copy()
    if sub.empty:
        return pd.DataFrame(columns=[
            "Date/Time","Matchup","Category","Pick","Line","Odds (US)","Odds (Dec)","Score","Units"
        ])

    # Prefer one outcome per event (best score)
    # Build score first
    sub["score"] = sub.apply(lambda r: score_row(r, market_key), axis=1)

    # Best outcome per event_id
    idx = sub.groupby("event_id")["score"].idxmax()
    one_per_event = sub.loc[idx].copy()

    # Sort by score desc, take 5
    one_per_event = one_per_event.sort_values("score", ascending=False).head(5)

    # Nice output columns
    out = pd.DataFrame({
        "Date/Time": one_per_event["Date/Time"],
        "Matchup": one_per_event["Matchup"],
        "Category": one_per_event["market_label"],
        "Pick": one_per_event["outcome"],
        "Line": one_per_event["line"].fillna(""),
        "Odds (US)": one_per_event["best_odds_us"],
        "Odds (Dec)": one_per_event["best_odds_dec"],
        "Score": one_per_event["score"].round(3),
    })
    out["Units"] = out["Score"].apply(units_from_score)
    return out.reset_index(drop=True)

def build_player_props(cons: pd.DataFrame) -> pd.DataFrame:
    # Some sports won't deliver player_props. Return empty if missing.
    if "player_props" not in cons["market"].unique():
        return pd.DataFrame(columns=[
            "Date/Time","Matchup","Category","Pick","Line","Odds (US)","Odds (Dec)","Score","Units"
        ])
    return pick_top5(cons, "player_props")

def build_moneyline(cons: pd.DataFrame) -> pd.DataFrame:
    return pick_top5(cons, "h2h")

def build_spreads(cons: pd.DataFrame) -> pd.DataFrame:
    return pick_top5(cons, "spreads")

def build_totals(cons: pd.DataFrame) -> pd.DataFrame:
    return pick_top5(cons, "totals")

def build_ai_top5(all_tables: List[pd.DataFrame]) -> pd.DataFrame:
    """
    AI Top 5 = top five by score across Moneyline/Spreads/Totals/Player Props
    (NOT including Parlays to avoid "doubling" exposure)
    """
    frames = [t.assign(Category=t["Category"]) for t in all_tables if not t.empty]
    if not frames:
        return pd.DataFrame(columns=["Date/Time","Matchup","Category","Pick","Line","Odds (US)","Odds (Dec)","Score","Units"])
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("Score", ascending=False).head(5).reset_index(drop=True)
    return df

# ─────────────────────────────────────────────
# Parlays (5 SGP)
# ─────────────────────────────────────────────
def parlay_price_decimal(legs: List[float]) -> float:
    """ Multiply decimal odds. """
    d = 1.0
    for x in legs:
        try:
            d *= float(x)
        except Exception:
            return np.nan
    return d

def parlay_score(legs_scores: List[float]) -> float:
    """ Combine leg scores (conservative: product in [0..1]) """
    s = 1.0
    for sc in legs_scores:
        s *= clamp01(sc)
    return s

def build_sgp_parlays(cons: pd.DataFrame) -> pd.DataFrame:
    """
    Make 5 SGPs by picking the top-scored legs per event and combining 2–3 legs.
    Use moneyline + spread or total for the *same game*, plus optionally a player prop if available.
    """
    if cons.empty:
        return pd.DataFrame(columns=[
            "Date/Time","Matchup","Category","Legs","Parlay (Dec)","Est. Score","Units"
        ])

    df = cons.copy()
    df["score"] = df.apply(lambda r: score_row(r, r["market"]), axis=1)

    # Group by event, pick best legs per market
    out_rows = []
    for event_id, group in df.groupby("event_id"):
        group = group.sort_values("score", ascending=False)
        if group.empty:
            continue

        # Try to get up to 3 diverse legs (ML + (Spreads or Totals) + maybe a prop)
        leg_pool = []
        # best ML
        ml = group[group["market"] == "h2h"].head(1)
        if not ml.empty:
            leg_pool.append(ml.iloc[0])
        # best Spreads or Totals (prefer best score among them)
        st_two = group[group["market"].isin(["spreads", "totals"])].head(1)
        if not st_two.empty:
            leg_pool.append(st_two.iloc[0])
        # optional prop
        prop = group[group["market"] == "player_props"].head(1)
        if not prop.empty:
            leg_pool.append(prop.iloc[0])

        # If we don't have at least 2 legs, skip
        if len(leg_pool) < 2:
            continue

        # Keep only top 3 legs max
        legs_sel = leg_pool[:3]
        legs_desc = []
        legs_dec = []
        legs_scores = []
        # all should share same game
        dts = set()
        mups = set()
        for r in legs_sel:
            desc = r["market_label"] + ": " + str(r["outcome"])
            if r["market"] in ("spreads","totals") and not pd.isna(r["line"]):
                try:
                    ln = float(r["line"])
                    sign = "+" if ln > 0 else ""
                    desc += f" ({sign}{ln})"
                except Exception:
                    desc += f" ({r['line']})"
            legs_desc.append(desc)
            legs_dec.append(r["best_odds_dec"])
            legs_scores.append(r["score"])
            dts.add(r["Date/Time"])
            mups.add(r["Matchup"])

        parlay_dec = parlay_price_decimal(legs_dec)
        est_score = parlay_score(legs_scores)

        out_rows.append({
            "Date/Time": list(dts)[0] if dts else "",
            "Matchup": list(mups)[0] if mups else "",
            "Category": "Parlay (SGP)",
            "Legs": " + ".join(legs_desc),
            "Parlay (Dec)": round(parlay_dec, 3) if not pd.isna(parlay_dec) else "",
            "Est. Score": round(est_score, 4),
            "Units": units_from_score(est_score),
        })

    if not out_rows:
        return pd.DataFrame(columns=[
            "Date/Time","Matchup","Category","Legs","Parlay (Dec)","Est. Score","Units"
        ])

    dfp = pd.DataFrame(out_rows)
    dfp = dfp.sort_values(["Est. Score","Parlay (Dec)"], ascending=[False, False]).head(5).reset_index(drop=True)
    return dfp

# ─────────────────────────────────────────────
# Excel Export (6 sheets + summary)
# ─────────────────────────────────────────────
def to_excel(daily_sport: str,
             ai_df: pd.DataFrame,
             ml_df: pd.DataFrame,
             sp_df: pd.DataFrame,
             tot_df: pd.DataFrame,
             prop_df: pd.DataFrame,
             parlay_df: pd.DataFrame) -> bytes:
    """
    Build an Excel with:
      - Summary
      - AI_Top_5
      - Moneyline
      - Spreads
      - Totals
      - Player_Props
      - Parlays
    Each sheet has columns prepared for: Result (drop-down later by you), Win%, Units Won (formulas in summary).
    """
    # Create a summary table for formulas
    def _mk(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["Date/Time","Matchup","Category","Pick","Line","Odds (US)","Odds (Dec)","Score","Units","Result"])
        d = df.copy()
        if "Result" not in d.columns:
            d["Result"] = ""  # you'll type Win/Loss later
        return d

    ai = _mk(ai_df)
    ml = _mk(ml_df)
    sp = _mk(sp_df)
    tot = _mk(tot_df)
    prop = _mk(prop_df)
    par = _mk(parlay_df)

    # In-memory bytes
    import io
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        # summary
        summary = pd.DataFrame({
            "Sheet": ["AI_Top_5","Moneyline","Spreads","Totals","Player_Props","Parlays"],
            "Notes": ["Edit Result column in each sheet to compute win% and units in your own workbook if desired. This file does not force formulas (so it’s portable).",
                      "","","","",""]
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)

        ai.to_excel(writer, sheet_name="AI_Top_5", index=False)
        ml.to_excel(writer, sheet_name="Moneyline", index=False)
        sp.to_excel(writer, sheet_name="Spreads", index=False)
        tot.to_excel(writer, sheet_name="Totals", index=False)
        prop.to_excel(writer, sheet_name="Player_Props", index=False)
        par.to_excel(writer, sheet_name="Parlays", index=False)

        # Add metadata sheet
        meta = pd.DataFrame({
            "Sport":[daily_sport],
            "Generated_At":[datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %I:%M %p %Z")]
        })
        meta.to_excel(writer, sheet_name="Meta", index=False)

    bio.seek(0)
    return bio.read()

# ─────────────────────────────────────────────
# UI Helpers
# ─────────────────────────────────────────────
def show_table(df: pd.DataFrame, title: str, caption: Optional[str] = None):
    st.subheader(title)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if caption:
        st.caption(caption)

def today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def ensure_ss_defaults():
    if "sport_name" not in st.session_state:
        st.session_state.sport_name = list(SPORT_OPTIONS.keys())[0]
    if "regions" not in st.session_state:
        st.session_state.regions = DEFAULT_REGIONS
    if "daily_store" not in st.session_state:
        # daily_store holds picks by date+sport to allow lock/publish
        st.session_state.daily_store = {}  # {(date,sport): {tables... , locked: bool}}
    if "has_data" not in st.session_state:
        st.session_state.has_data = False

ensure_ss_defaults()

# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()),
                              index=list(SPORT_OPTIONS.keys()).index(st.session_state.sport_name),
                              key="sport_name")
    regions = st.text_input("Regions (Odds API)", value=st.session_state.regions, key="regions")
    st.info("Daily flow: 1) Generate → 2) Review → 3) Publish/Lock → 4) Export Excel")
    c1, c2 = st.columns(2)
    with c1:
        gen = st.button("Generate Today's 5 Picks")
    with c2:
        publish = st.button("Publish / Lock Today")

    # Excel download appears after generation
    st.markdown("---")
    st.caption("Export current picks:")
    # download button created later after generation

# ─────────────────────────────────────────────
# Generation Pipeline
# ─────────────────────────────────────────────
def generate_for_sport(selected_sport: str, regions: str) -> Dict[str, pd.DataFrame]:
    """Fetch odds (h2h,spreads,totals,player_props), build consensus, score, and pick top-5 tables."""
    sport_key = SPORT_OPTIONS[selected_sport]
    markets = "h2h,spreads,totals,player_props"
    if isinstance(sport_key, list):
        parts = [fetch_odds(k, regions, markets) for k in sport_key]
        raw = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    else:
        raw = fetch_odds(sport_key, regions, markets)

    if raw.empty:
        return {
            "cons": pd.DataFrame(),
            "AI": pd.DataFrame(),
            "Moneyline": pd.DataFrame(),
            "Spreads": pd.DataFrame(),
            "Totals": pd.DataFrame(),
            "Player_Props": pd.DataFrame(),
            "Parlays": pd.DataFrame(),
        }

    cons = consensus_view(raw)
    # Build charts
    ml = build_moneyline(cons)
    sp = build_spreads(cons)
    tot = build_totals(cons)
    props = build_player_props(cons)
    ai = build_ai_top5([ml, sp, tot, props])
    parlays = build_sgp_parlays(cons)

    return {
        "cons": cons,
        "AI": ai,
        "Moneyline": ml,
        "Spreads": sp,
        "Totals": tot,
        "Player_Props": props,
        "Parlays": parlays
    }

# Handle Generate
if gen:
    key = (today_key(), sport_name)
    data = generate_for_sport(sport_name, regions)
    if data["cons"].empty:
        st.warning("No odds returned. Try a different sport or check API limits.")
        st.session_state.has_data = False
    else:
        st.session_state.daily_store[key] = {
            "locked": False,
            "AI": data["AI"],
            "Moneyline": data["Moneyline"],
            "Spreads": data["Spreads"],
            "Totals": data["Totals"],
            "Player_Props": data["Player_Props"],
            "Parlays": data["Parlays"],
            "cons": data["cons"]  # keep for debugging if needed
        }
        st.success(f"Generated 5 picks for each chart — {sport_name} — {today_key()}")
        st.session_state.has_data = True

# Handle Publish/Lock
if publish:
    key = (today_key(), sport_name)
    if key in st.session_state.daily_store:
        st.session_state.daily_store[key]["locked"] = True
        st.success(f"Published/Locked picks for {sport_name} — {today_key()}.")
    else:
        st.warning("Generate today's picks first.")

# ─────────────────────────────────────────────
# Render Tabs (from session)
# ─────────────────────────────────────────────
key = (today_key(), sport_name)
store = st.session_state.daily_store.get(key)

if store and st.session_state.get("has_data", False):
    tabs = st.tabs(["🤖 AI Top 5", "Moneylines", "Spreads", "Totals", "Player Props", "Parlays", "Export"])
    locked = store.get("locked", False)
    lock_badge = "🔒 Locked" if locked else "🟢 Draft"
    st.caption(f"Status: **{lock_badge}** · {sport_name} · {today_key()}")

    with tabs[0]:
        show_table(store["AI"], "AI Top 5")
        st.caption("Top five by internal score across Moneyline, Spreads, Totals, and Player Props.")

    with tabs[1]:
        show_table(store["Moneyline"], "Moneyline — Daily 5")

    with tabs[2]:
        show_table(store["Spreads"], "Spreads — Daily 5")

    with tabs[3]:
        show_table(store["Totals"], "Totals — Daily 5")

    with tabs[4]:
        show_table(store["Player_Props"], "Player Props — Daily 5", caption="If empty: player props may not be available for this sport/region right now.")

    with tabs[5]:
        show_table(store["Parlays"], "Parlays — 5 Same-Game Parlays", caption="Each parlay combines 2–3 legs from the same game with combined score.")

    with tabs[6]:
        st.subheader("Export")
        st.write("Download your **six** sheets (AI, ML, Spreads, Totals, Player Props, Parlays) + Summary + Meta:")
        excel_bytes = to_excel(
            daily_sport=sport_name,
            ai_df=store["AI"],
            ml_df=store["Moneyline"],
            sp_df=store["Spreads"],
            tot_df=store["Totals"],
            prop_df=store["Player_Props"],
            parlay_df=store["Parlays"]
        )
        st.download_button(
            label="⬇️ Download Excel (daily_picks.xlsx)",
            data=excel_bytes,
            file_name=EXCEL_FILE,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
        st.info("Open the Excel file and mark **Result** (Win/Loss) in each sheet later if you want to track historical performance in Excel.")
else:
    st.info("Use **Generate Today's 5 Picks** in the sidebar to build the charts.")
