import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional

# ─────────────────────────────────────────────
# Safe dotenv
# ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ODDS_API_KEY = os.getenv("ODDS_API_KEY", st.secrets.get("ODDS_API_KEY", ""))
APISPORTS_KEY = os.getenv("APISPORTS_KEY", st.secrets.get("APISPORTS_KEY", ""))
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

st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker")
st.caption("Live odds + historical context + AI-style ranking + Real Results Tracking.")
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

def assign_units(conf: float, hist_boost: float = 0.0) -> float:
    if pd.isna(conf):
        return 0.5
    combined = conf + hist_boost
    return round(0.5 + 4.5 * max(0.0, min(1.0, combined)), 1)

def fmt_pct(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{100.0 * x:.1f}%"

# ─────────────────────────────────────────────
# Bets Logging (CSV)
# ─────────────────────────────────────────────
BETS_FILE = "bets.csv"

def load_bets() -> pd.DataFrame:
    if os.path.exists(BETS_FILE):
        return pd.read_csv(BETS_FILE)
    return pd.DataFrame(columns=["Date", "Matchup", "Pick", "Odds", "Units", "Result"])

def save_bet(date, matchup, pick, odds, units, result="Pending"):
    bets = load_bets()
    bets = pd.concat([bets, pd.DataFrame([{
        "Date": date,
        "Matchup": matchup,
        "Pick": pick,
        "Odds": odds,
        "Units": units,
        "Result": result
    }])], ignore_index=True)
    bets.to_csv(BETS_FILE, index=False)

def update_bet_result(index, result):
    bets = load_bets()
    if 0 <= index < len(bets):
        bets.at[index, "Result"] = result
        bets.to_csv(BETS_FILE, index=False)

def calc_real_win_pct() -> str:
    bets = load_bets()
    if bets.empty:
        return "N/A"
    valid = bets[bets["Result"].isin(["Win", "Loss"])]
    if valid.empty:
        return "N/A"
    win_pct = (valid["Result"] == "Win").mean() * 100
    return f"{win_pct:.1f}%"

# ─────────────────────────────────────────────
# Odds API fetch
# ─────────────────────────────────────────────
def _odds_get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    if not ODDS_API_KEY:
        return None
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return None
        return r.json()
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
        home = ev.get("home_team", "Unknown")
        away = ev.get("away_team", "Unknown")

        for bk in ev.get("bookmakers", []):
            book = bk.get("title")
            for mk in bk.get("markets", []):
                mkey = mk.get("key")
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
                        "conf_market": implied_prob_american(oc.get("price")),
                    })
    df = pd.DataFrame(rows)

    if not df.empty and "commence_time" in df.columns:
        df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
        if pd.api.types.is_datetime64tz_dtype(df["commence_time"]):
            df["Date/Time"] = df["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
        else:
            df["Date/Time"] = df["commence_time"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    return df

# ─────────────────────────────────────────────
# Deduplicate best picks
# ─────────────────────────────────────────────
def best_per_event(df: pd.DataFrame, market_key: str, top_n: int = 10) -> pd.DataFrame:
    sub = df[df["market"] == market_key].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.loc[sub.groupby("event_id")["conf_market"].idxmax()].copy()
    sub = sub.sort_values("commence_time", ascending=True).head(top_n)
    sub["Matchup"] = sub["home_team"] + " vs " + sub["away_team"]

    out = sub[["Date/Time", "Matchup", "book", "outcome", "line", "odds_american", "odds_decimal", "conf_market"]]
    out = out.rename(columns={
        "book": "Sportsbook", "outcome": "Pick", "line": "Line",
        "odds_american": "Odds (US)", "odds_decimal": "Odds (Dec)",
        "conf_market": "Confidence"
    })
    out["Confidence"] = out["Confidence"].apply(fmt_pct)
    out["Units"] = sub["conf_market"].apply(lambda c: assign_units(c))
    return out.reset_index(drop=True)

# ─────────────────────────────────────────────
# Sidebar + Main
# ─────────────────────────────────────────────
with st.sidebar:
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()), index=0)
    regions = st.text_input("Regions", value=DEFAULT_REGIONS)
    top_n = st.slider("Top picks per tab", 3, 20, 10)
    fetch = st.button("Fetch Live Odds")

if fetch:
    sport_key = SPORT_OPTIONS[sport_name]
    if isinstance(sport_key, list):
        parts = [fetch_odds(k, regions) for k in sport_key]
        raw = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    else:
        raw = fetch_odds(sport_key, regions)

    if raw.empty:
        st.warning("No data returned.")
    else:
        tabs = st.tabs(["🤖 AI Genius Picks", "Moneylines", "Totals", "Spreads", "Results", "Raw Data"])

        with tabs[0]:
            st.subheader("AI Genius — Top Picks")
            ml = best_per_event(raw, "h2h", top_n)
            st.dataframe(ml, use_container_width=True, hide_index=True)
            if not ml.empty:
                row = ml.iloc[0]
                if st.button(f"Log Bet: {row['Matchup']} - {row['Pick']}"):
                    save_bet(row["Date/Time"], row["Matchup"], row["Pick"], row["Odds (US)"], row["Units"])
                    st.success("Bet logged!")

        with tabs[1]:
            st.subheader("Best Moneylines")
            t = best_per_event(raw, "h2h", top_n)
            st.dataframe(t, use_container_width=True, hide_index=True)

        with tabs[2]:
            st.subheader("Best Totals")
            t = best_per_event(raw, "totals", top_n)
            st.dataframe(t, use_container_width=True, hide_index=True)

        with tabs[3]:
            st.subheader("Best Spreads")
            t = best_per_event(raw, "spreads", top_n)
            st.dataframe(t, use_container_width=True, hide_index=True)

        with tabs[4]:
            st.subheader("Your Results")
            bets = load_bets()
            st.dataframe(bets, use_container_width=True, hide_index=True)
            st.markdown(f"**Overall Real Win %:** {calc_real_win_pct()}")

            if not bets.empty:
                index = st.number_input("Enter bet index to update result", min_value=0, max_value=len(bets)-1, step=1)
                result = st.selectbox("Result", ["Pending", "Win", "Loss"])
                if st.button("Update Bet Result"):
                    update_bet_result(index, result)
                    st.success("Result updated!")

        with tabs[5]:
            st.subheader("Raw Odds Data")
            st.dataframe(raw.head(200), use_container_width=True, hide_index=True)

else:
    st.info("Pick a sport and click **Fetch Live Odds**")
