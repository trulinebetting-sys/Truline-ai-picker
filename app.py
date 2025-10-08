import os
from typing import List, Dict, Any

import requests
import pandas as pd
import numpy as np
import streamlit as st

# ----------------------------
# Optional dotenv (won’t break if missing)
# ----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------------------
# Config / ENV
# ----------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", st.secrets.get("ODDS_API_KEY", ""))
DEFAULT_REGIONS = os.getenv("REGIONS", "us")

SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "EPL": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
}

# ----------------------------
# Streamlit page
# ----------------------------
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker")
st.caption("Live odds → de-vig → confidence → top picks. (No bankroll required.)")
st.write("---")

# ----------------------------
# Odds helpers
# ----------------------------
def american_to_decimal(odds: float) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    return 1 + (odds / 100.0) if odds > 0 else 1 + (100.0 / abs(odds))

def implied_prob_american(odds: float) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def no_vig_probs(implied_probs: List[float]) -> List[float]:
    vals = [p for p in implied_probs if p is not None and not pd.isna(p)]
    s = sum(vals)
    if s <= 0:
        return implied_probs
    return [p / s if (p is not None and not pd.isna(p)) else np.nan for p in implied_probs]

def edge_from_true_p(dec_odds: float, true_p: float) -> float:
    if pd.isna(dec_odds) or pd.isna(true_p) or dec_odds is None or true_p is None:
        return np.nan
    return dec_odds * true_p - 1.0  # expected return per $1

def fmt_pct(x: float, d: int = 1) -> str:
    return "" if (x is None or pd.isna(x)) else f"{100.0 * x:.{d}f}%"

# ----------------------------
# Fetch Odds API and build tidy rows
# ----------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_odds_structured(sport_key: str, regions: str) -> pd.DataFrame:
    """
    Returns a clean table with one row per outcome across H2H / Totals / Spreads.
    Columns: event_id, commence_time, sport_key, home_team, away_team, book,
             market(h2h/totals/spreads), name(Team/Over/Under), point, price_american,
             dec_odds, imp_prob, true_prob
    """
    if not ODDS_API_KEY:
        return pd.DataFrame()

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        # Return a small df with error info so UI can surface it gently
        return pd.DataFrame([{"__error__": f"Odds API error {r.status_code}: {r.text}"}])

    data = r.json()
    rows: List[Dict[str, Any]] = []

    for ev in data or []:
        event_id = ev.get("id") or f"{ev.get('home_team')}-{ev.get('away_team')}-{ev.get('commence_time')}"
        sport = ev.get("sport_key", sport_key)
        commence = ev.get("commence_time")
        home = ev.get("home_team", "Unknown")
        away = ev.get("away_team", "Unknown")

        for bk in ev.get("bookmakers", []) or []:
            book = bk.get("title", "Unknown")
            for mk in bk.get("markets", []) or []:
                mkey = mk.get("key")  # h2h / spreads / totals
                outs = mk.get("outcomes", []) or []

                # Implied probs & de-vig per market
                imp = [implied_prob_american(o.get("price")) for o in outs]
                nv = no_vig_probs(imp)

                for i, o in enumerate(outs):
                    name = o.get("name")  # Home/Away/Draw or Over/Under or team
                    price = o.get("price")
                    point = o.get("point")  # spreads/totals line (may be None)

                    dec = american_to_decimal(price)
                    imp_i = imp[i] if i < len(imp) else np.nan
                    true_i = nv[i] if i < len(nv) else np.nan

                    rows.append({
                        "event_id": event_id,
                        "sport_key": sport,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": mkey,
                        "name": name,
                        "point": point,
                        "price_american": price,
                        "dec_odds": dec,
                        "imp_prob": imp_i,
                        "true_prob": true_i,
                    })

    df = pd.DataFrame(rows)
    # Parse/format time (safe)
    if not df.empty and "commence_time" in df.columns:
        try:
            dt = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
            df["commence_time_fmt"] = dt.dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
        except Exception:
            df["commence_time_fmt"] = df["commence_time"]
    else:
        df["commence_time_fmt"] = ""
    return df

# ----------------------------
# Tiny “historical” placeholder signal
# ----------------------------
def historical_signal(row: pd.Series) -> float:
    """
    Placeholder: returns a small, stable signal so the app works without extra APIs.
    - Slight nudge for home team on H2H
    - Neutral for totals/spreads
    Range ~ [0.48, 0.52]
    """
    if row.get("market") == "h2h":
        if row.get("name") and row.get("home_team") and str(row["name"]).lower().strip() == str(row["home_team"]).lower().strip():
            return 0.52  # home ML slight bonus
        return 0.48
    return 0.50

def confidence_score(true_prob: float, hist_sig: float) -> float:
    """
    Blend live (de-vig) probability with historical placeholder.
    You can later replace hist_sig with real stats (Elo/SRS/H2H).
    """
    if pd.isna(true_prob) or true_prob is None:
        true_prob = 0.50
    if pd.isna(hist_sig) or hist_sig is None:
        hist_sig = 0.50
    return 0.6 * true_prob + 0.4 * hist_sig

def units_from_confidence(c: float) -> float:
    if c >= 0.75: return 2.00
    if c >= 0.65: return 1.50
    if c >= 0.55: return 1.00
    return 0.50

# ----------------------------
# Build tables per market
# ----------------------------
def tidy_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns in a clean, consistent order and format percentages."""
    if df.empty:
        return df

    out = df.copy()
    out["Implied %"] = out["imp_prob"].apply(lambda x: fmt_pct(x, 1))
    out["True %"]    = out["true_prob"].apply(lambda x: fmt_pct(x, 1))
    out["Conf %"]    = out["confidence"].apply(lambda x: fmt_pct(x, 1))
    out["Units"]     = out["units"].apply(lambda x: f"{x:.2f}")

    # Build Pick/Line labels
    def pick_label(r):
        m = (r.get("market") or "").lower()
        if m == "h2h":
            # r["name"] is the team (home/away)
            return f"{r.get('name', '')} ML"
        elif m == "totals":
            return f"{r.get('name','')} {r.get('point','')}"  # Over 210.5 / Under 210.5
        elif m == "spreads":
            p = r.get("point", None)
            if p is None or pd.isna(p):
                return f"{r.get('name','')} (spread)"
            sign = "+" if float(p) >= 0 else ""
            return f"{r.get('name','')} {sign}{p}"
        return str(r.get("name", ""))

    out["Pick"] = out.apply(pick_label, axis=1)

    cols = [
        "commence_time_fmt", "home_team", "away_team", "book",
        "market", "Pick", "price_american", "Implied %", "True %", "Conf %", "Units"
    ]
    out = out[cols].rename(columns={
        "commence_time_fmt": "Date/Time",
        "home_team": "Home",
        "away_team": "Away",
        "book": "Book",
        "market": "Market",
        "price_american": "Odds (US)",
    })
    return out

def best_per_event(df: pd.DataFrame, market_key: str, top_n: int = 10) -> pd.DataFrame:
    """
    1) Filter to a market (h2h/totals/spreads)
    2) Compute confidence and units
    3) Deduplicate: keep highest-confidence pick per event_id
    4) Return Top N by confidence (then edge)
    """
    if df.empty:
        return pd.DataFrame()

    work = df[df["market"] == market_key].copy()
    if work.empty:
        return pd.DataFrame()

    work["hist_sig"]   = work.apply(historical_signal, axis=1)
    work["confidence"] = work.apply(lambda r: confidence_score(r.get("true_prob"), r.get("hist_sig")), axis=1)
    work["edge"]       = work.apply(lambda r: edge_from_true_p(r.get("dec_odds"), r.get("true_prob")), axis=1)
    work["units"]      = work["confidence"].apply(units_from_confidence)

    # Deduplicate by event: keep row with max confidence
    idx = work.groupby("event_id")["confidence"].idxmax()
    keep = work.loc[idx].copy()

    # Sort & top N
    keep = keep.sort_values(["confidence", "edge"], ascending=[False, False]).head(top_n)
    return tidy_for_display(keep)

def ai_genius_top10(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the single best bet per game across ALL markets, return top 10 overall."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["hist_sig"]   = work.apply(historical_signal, axis=1)
    work["confidence"] = work.apply(lambda r: confidence_score(r.get("true_prob"), r.get("hist_sig")), axis=1)
    work["edge"]       = work.apply(lambda r: edge_from_true_p(r.get("dec_odds"), r.get("true_prob")), axis=1)
    work["units"]      = work["confidence"].apply(units_from_confidence)

    idx = work.groupby("event_id")["confidence"].idxmax()
    keep = work.loc[idx].copy()
    keep = keep.sort_values(["confidence", "edge"], ascending=[False, False]).head(10)
    return tidy_for_display(keep)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Filters")
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()), index=0)
    regions = st.text_input("Regions", value=DEFAULT_REGIONS)
    fetch_btn = st.button("Fetch Live Odds")

# ----------------------------
# Fetch & Display
# ----------------------------
sport_key = SPORT_OPTIONS[sport_name]

if not ODDS_API_KEY:
    st.warning("Add your ODDS_API_KEY to `.env` or Streamlit Secrets to fetch live odds.")
    st.stop()

if not fetch_btn:
    st.info("Choose a sport and click **Fetch Live Odds**.")
    st.stop()

df = fetch_odds_structured(sport_key, regions)

# Surface API errors nicely
if not df.empty and "__error__" in df.columns:
    st.error(df["__error__"].iloc[0])
    st.stop()

if df.empty:
    st.warning("No data returned. Try another sport or region, or check API quota.")
    st.stop()

tabs = st.tabs(["AI Genius Top 10", "Moneylines", "Totals", "Spreads", "Raw Data"])

with tabs[0]:
    st.subheader("🔥 AI Genius Top 10 (deduped by game)")
    ai_top = ai_genius_top10(df)
    if ai_top.empty:
        st.info("No picks available.")
    else:
        st.dataframe(ai_top, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Moneylines – Best per game")
    ml = best_per_event(df, "h2h", top_n=10)
    if ml.empty:
        st.info("No moneyline data available.")
    else:
        st.dataframe(ml, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Totals – Best per game")
    tot = best_per_event(df, "totals", top_n=10)
    if tot.empty:
        st.info("No totals data available.")
    else:
        st.dataframe(tot, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Spreads – Best per game")
    spr = best_per_event(df, "spreads", top_n=10)
    if spr.empty:
        st.info("No spreads data available.")
    else:
        st.dataframe(spr, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Raw (for debugging)")
    st.dataframe(df.head(200), use_container_width=True)
