import os
import math
from itertools import combinations
from typing import List, Dict, Any

import requests
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# ----------------------------
# Setup
# ----------------------------
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
DEFAULT_REGIONS = os.getenv("REGIONS", "us")
DEFAULT_BOOKS = [b.strip() for b in os.getenv(
    "BOOKS", "DraftKings,FanDuel,BetMGM,PointsBet,Caesars,Pinnacle"
).split(",") if b.strip()]
DEFAULT_MIN_EDGE = float(os.getenv("MIN_EDGE", "0.01"))
DEFAULT_KELLY_CAP = float(os.getenv("KELLY_CAP", "0.25"))

SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "EPL": "soccer_epl",
    "La Liga": "soccer_spain_la_liga"
}

# Streamlit page
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker")
st.caption("Simple. Live odds. Three sections. Units recommended with capped Kelly.")
st.write("---")

# ----------------------------
# Helpers: odds & math
# ----------------------------
def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    return 1 + (odds / 100.0) if odds > 0 else 1 + (100.0 / abs(odds))

def implied_prob_american(odds: float) -> float:
    """Implied probability from American odds."""
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def no_vig_probs(implied_probs: List[float]) -> List[float]:
    """Normalize implied probs to remove vig (simple proportional de-vig)."""
    vals = [p for p in implied_probs if p is not None and not pd.isna(p)]
    s = sum(vals)
    if s <= 0:
        return implied_probs
    return [p / s if (p is not None and not pd.isna(p)) else np.nan for p in implied_probs]

def kelly_fraction(true_p: float, dec_odds: float) -> float:
    """Kelly for decimal odds. Returns fraction of bankroll."""
    if true_p is None or pd.isna(true_p) or dec_odds is None or pd.isna(dec_odds):
        return 0.0
    b = dec_odds - 1.0
    q = 1.0 - true_p
    f = (b * true_p - q) / b if b > 0 else 0.0
    return max(0.0, f)

def edge_from_true_p(dec_odds: float, true_p: float) -> float:
    """Expected return per $1 = d * p - 1."""
    if dec_odds is None or pd.isna(dec_odds) or true_p is None or pd.isna(true_p):
        return np.nan
    return dec_odds * true_p - 1.0

def fmt_pct(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{100.0 * x:.1f}%"

def fmt_dec(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{x:.2f}"

def usd(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"${x:,.2f}"

# ----------------------------
# Fetch Odds API
# ----------------------------
@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    """Returns one row per outcome across requested markets."""
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY. Add it to .env (locally) or to Streamlit Secrets (cloud).")
        return pd.DataFrame()

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american"
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        st.error(f"Odds API error {r.status_code}: {r.text}")
        return pd.DataFrame()

    data = r.json()
    rows = []
    for ev in data:
        event_id = ev.get("id") or f"{ev.get('home_team')}-{ev.get('away_team')}-{ev.get('commence_time')}"
        sport = ev.get("sport_key")
        commence = ev.get("commence_time")
        home = ev.get("home_team", "Unknown")
        away = ev.get("away_team", "Unknown")

        for bk in ev.get("bookmakers", []):
            book = bk.get("title")
            for mk in bk.get("markets", []):
                mkey = mk.get("key")  # h2h, spreads, totals
                # Outcomes list: name (Home/Away/Draw or Over/Under), price (american), point (for spread/totals)
                outs = mk.get("outcomes", [])
                # Build implied probs & de-vig
                imp = [implied_prob_american(o.get("price")) for o in outs]
                nv = no_vig_probs(imp)

                for i, o in enumerate(outs):
                    name = o.get("name")
                    price = o.get("price")
                    point = o.get("point")  # spread points or total line, may be None
                    rows.append({
                        "event_id": event_id,
                        "sport_key": sport,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": mkey,
                        "name": name,          # Home/Away/Draw/Over/Under
                        "point": point,        # spread/totals line if present
                        "price_american": price,
                        "dec_odds": american_to_decimal(price),
                        "imp_prob": imp[i] if i < len(imp) else np.nan,
                        "true_prob": nv[i] if i < len(nv) else np.nan,
                    })
    return pd.DataFrame(rows)

# ----------------------------
# Build single-bet table
# ----------------------------
def build_single_table(df: pd.DataFrame,
                       market_filter: List[str],
                       selected_books: List[str],
                       min_edge: float,
                       bankroll: float,
                       unit_size: float,
                       kelly_cap: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if market_filter:
        work = work[work["market"].isin(market_filter)]
    if selected_books:
        work = work[work["book"].isin(selected_books)]

    # Compute metrics
    work["edge"] = work.apply(lambda r: edge_from_true_p(r["dec_odds"], r["true_prob"]), axis=1)
    work["kelly"] = work.apply(lambda r: kelly_fraction(r["true_prob"], r["dec_odds"]), axis=1)
    work["kelly_capped"] = work["kelly"].clip(lower=0.0, upper=kelly_cap)
    work["stake_$"] = (work["kelly_capped"] * bankroll).clip(lower=0.0)
    work["units"] = work["stake_$"] / max(1e-9, unit_size)

    # Filter by edge
    work = work[work["edge"] >= min_edge].copy()

    # Presentable columns
    def bet_label(row):
        if row["market"] == "h2h":
            # Moneyline
            return row["name"]
        elif row["market"] == "totals":
            return f"{row['name']} {row['point']}"
        elif row["market"] == "spreads":
            # name is team; point is spread (+/-)
            return f"{row['name']} {row['point']:+}"
        return row["name"]

    work["Matchup"] = work["home_team"] + " vs " + work["away_team"]
    work["Bet"] = work.apply(bet_label, axis=1)

    out = work[[
        "commence_time", "sport_key", "Matchup", "market", "Bet",
        "book", "price_american", "dec_odds", "imp_prob", "true_prob",
        "edge", "kelly_capped", "stake_$", "units"
    ]].copy()

    # Format for readability
    out = out.sort_values(by="edge", ascending=False)
    out.rename(columns={
        "commence_time": "Date/Time",
        "sport_key": "Sport",
        "market": "Market",
        "book": "Book",
        "price_american": "Odds (US)",
        "dec_odds": "Odds (Dec)",
        "imp_prob": "Implied %",
        "true_prob": "True %",
        "edge": "Edge %",
        "kelly_capped": "Kelly %"
    }, inplace=True)

    # Pretty print percentages/decimals
    out["Implied %"] = out["Implied %"].apply(fmt_pct)
    out["True %"] = out["True %"].apply(fmt_pct)
    out["Edge %"] = out["Edge %"].apply(lambda x: f"{100*x:.2f}%" if not pd.isna(x) else "")
    out["Kelly %"] = out["Kelly %"].apply(lambda x: f"{100*x:.1f}%" if not pd.isna(x) else "")
    out["Odds (Dec)"] = out["Odds (Dec)"].apply(fmt_dec)
    out["stake_$"] = out["stake_$"].apply(usd)
    out["units"] = out["units"].apply(lambda x: f"{x:.2f}")

    return out.reset_index(drop=True)

# ----------------------------
# Build parlay table
# ----------------------------
def build_parlays(df: pd.DataFrame,
                  selected_books: List[str],
                  min_edge: float,
                  bankroll: float,
                  unit_size: float,
                  kelly_cap: float,
                  legs: int = 2,
                  top_candidates: int = 20,
                  max_results: int = 25) -> pd.DataFrame:
    """Form parlays from top single picks within the same book and different events."""
    if df.empty:
        return pd.DataFrame()

    # First, build a clean singles table across all markets, then filter
    singles = build_single_table(
        df=df,
        market_filter=["h2h", "totals", "spreads"],
        selected_books=selected_books,
        min_edge=min_edge,
        bankroll=bankroll,
        unit_size=unit_size,
        kelly_cap=kelly_cap
    )
    if singles.empty:
        return pd.DataFrame()

    # Take top per book as candidates
    base = singles.copy()
    # Add event_id back: join on matchup + time if needed
    # We cached event_id in df; reconstruct mapping
    id_map = df.copy()
    id_map["Matchup"] = id_map["home_team"] + " vs " + id_map["away_team"]
    id_map = id_map[["event_id", "Matchup", "commence_time"]].drop_duplicates()
    base = base.merge(id_map, on=["Matchup", "commence_time"], how="left")

    # Work per book
    parlays_rows = []
    for book, bdf in base.groupby("Book"):
        cand = bdf.sort_values("Edge %", ascending=False).head(top_candidates).copy()
        # ensure different events
        cand = cand.dropna(subset=["event_id"])
        # Build combinations
        rows = cand.to_dict("records")
        for combo in combinations(rows, legs):
            event_ids = {r["event_id"] for r in combo}
            if len(event_ids) < legs:
                continue  # skip same-game parlays (simple rule)
            # Combined odds/prob
            decs = []
            trues = []
            labels = []
            for r in combo:
                # reverse-format values
                d = float(r["Odds (Dec)"])
                # True % stored as string like '62.3%' — convert back
                tp = float(r["True %"].replace("%","")) / 100.0
                decs.append(d)
                trues.append(tp)
                labels.append(f"{r['Market']} | {r['Bet']}")

            combo_dec = float(np.prod(decs))
            combo_true = float(np.prod(trues))  # assumes independence
            ev = edge_from_true_p(combo_dec, combo_true)  # per $1

            k = kelly_fraction(combo_true, combo_dec)
            k = min(k, kelly_cap)
            stake = max(0.0, k * bankroll)
            units = stake / max(1e-9, unit_size)

            parlays_rows.append({
                "Book": book,
                "Legs": legs,
                "Parlay": " + ".join(labels),
                "Odds (Dec)": f"{combo_dec:.2f}",
                "True %": f"{100.0*combo_true:.2f}%",
                "Edge %": f"{100.0*ev:.2f}%",
                "Kelly %": f"{100.0*k:.1f}%",
                "stake_$": usd(stake),
                "units": f"{units:.2f}"
            })

    out = pd.DataFrame(parlays_rows)
    if out.empty:
        return out
    # Sort by Edge %
    out["edge_sort"] = out["Edge %"].str.replace("%","").astype(float)
    out = out.sort_values("edge_sort", ascending=False).drop(columns=["edge_sort"])
    return out.head(max_results).reset_index(drop=True)

# ----------------------------
# Sidebar: global filters
# ----------------------------
with st.sidebar:
    st.header("Filters")
    sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()), index=0)
    regions = st.text_input("Regions", value=DEFAULT_REGIONS)
    books_default = DEFAULT_BOOKS
    bankroll = st.number_input("Bankroll ($)", min_value=100.0, value=1000.0, step=50.0)
    unit_size = st.number_input("Unit size ($)", min_value=1.0, value=25.0, step=1.0)
    min_edge = st.slider("Min Edge (%)", 0.0, 10.0, DEFAULT_MIN_EDGE*100, 0.25) / 100.0
    kelly_cap = st.slider("Kelly Cap", 0.0, 1.0, DEFAULT_KELLY_CAP, 0.05)
    legs = st.selectbox("Parlay legs", [2, 3], index=0)
    fetch = st.button("Fetch Live Odds")

# ----------------------------
# Fetch + Show
# ----------------------------
sport_key = SPORT_OPTIONS[sport_name]

if fetch:
    raw = fetch_odds(sport_key=sport_key, regions=regions, markets="h2h,spreads,totals")
    if raw.empty:
        st.warning("No data returned. Try another sport or check your API usage/quota.")
    else:
        # Allow user to filter books dynamically from what came back
        all_books = sorted(raw["book"].dropna().unique().tolist())
        st.write(f"Books found: {', '.join(all_books) or 'None'}")
        selected_books = st.multiselect("Books to include", all_books, default=[b for b in books_default if b in all_books])

        tabs = st.tabs(["Moneylines", "Parlays", "Totals / Spreads"])

        # --- Moneyline
        with tabs[0]:
            ml = build_single_table(
                df=raw, market_filter=["h2h"], selected_books=selected_books,
                min_edge=min_edge, bankroll=bankroll, unit_size=unit_size, kelly_cap=kelly_cap
            )
            st.subheader("Moneyline Picks")
            if ml.empty:
                st.info("No moneyline edges ≥ threshold.")
            else:
                st.dataframe(ml, use_container_width=True, hide_index=True)

        # --- Parlays (built from best singles)
        with tabs[1]:
            pl = build_parlays(
                df=raw, selected_books=selected_books, min_edge=min_edge,
                bankroll=bankroll, unit_size=unit_size, kelly_cap=kelly_cap,
                legs=int(legs), top_candidates=20, max_results=25
            )
            st.subheader(f"Top {legs}-Leg Parlays")
            st.caption("Built from best singles in the same book, different games. Independence assumed.")
            if pl.empty:
                st.info("No parlay combos passed the threshold.")
            else:
                st.dataframe(pl, use_container_width=True, hide_index=True)

        # --- Totals / Spreads
        with tabs[2]:
            choice = st.radio("Market", ["Totals (Over/Under)", "Spreads (±)"], horizontal=True)
            mf = ["totals"] if "Totals" in choice else ["spreads"]
            ts = build_single_table(
                df=raw, market_filter=mf, selected_books=selected_books,
                min_edge=min_edge, bankroll=bankroll, unit_size=unit_size, kelly_cap=kelly_cap
            )
            st.subheader(choice)
            if ts.empty:
                st.info("No edges ≥ threshold for this market.")
            else:
                st.dataframe(ts, use_container_width=True, hide_index=True)
else:
    st.info("Set your filters in the sidebar, then click **Fetch Live Odds**.")
    if not ODDS_API_KEY:
        st.error("No ODDS_API_KEY detected. Add it to your `.env` (local) or Streamlit **Secrets** (cloud).")
