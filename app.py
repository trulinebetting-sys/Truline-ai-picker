import os
from typing import Dict, Any, Optional, List, Tuple
import random
import math
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
# Config / Keys
# ─────────────────────────────────────────────
st.set_page_config(page_title="TruLine – AI Genius Picker", layout="wide")
st.title("TruLine – AI Genius Picker 🚀")
st.caption("Consensus across books + live odds + AI-style ranking. Tracks results + bankroll ✅")
st.divider()

# ✅ Odds API Key (you already use this)
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "") or "1d677dc98d978ccc24d9914d835442f1"
DEFAULT_REGIONS = os.getenv("REGIONS", "us")

# Sports (unchanged)
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
    # more aggressive top-end to make units feel meaningful on strong signals
    return round(0.25 + 5.75 * max(0.0, min(1.0, conf)), 2)

def fmt_pct(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{100.0 * x:.1f}%"

def safe_get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

# ─────────────────────────────────────────────
# Odds API fetch (sides/totals)
# ─────────────────────────────────────────────
def _odds_get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY.")
        return None
    return safe_get(url, params)

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
# Consensus logic (sides/totals)
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
# PROPS via Odds API (NBA + NFL for your 6 categories)
# ─────────────────────────────────────────────
# Map your chosen categories → Odds API market keys
PROP_MARKET_KEYS = {
    # NBA
    "player_points": "player_points",
    "player_assists": "player_assists",
    "player_rebounds": "player_rebounds",
    # NFL (these keys are the common Odds API names)
    "player_pass_yds": "player_passing_yards",
    "player_rush_yds": "player_rushing_yards",
    "player_rec_yds": "player_receiving_yards",
}

# Which sport keys support which props
PROP_SPORTS = {
    "basketball_nba": ["player_points", "player_assists", "player_rebounds"],
    "americanfootball_nfl": ["player_pass_yds", "player_rush_yds", "player_rec_yds"],
}

def build_prop_market_str() -> str:
    keys = [PROP_MARKET_KEYS[k] for k in [
        "player_points","player_assists","player_rebounds",
        "player_pass_yds","player_rush_yds","player_rec_yds"
    ] if k in PROP_MARKET_KEYS]
    return ",".join(keys)

@st.cache_data(ttl=60)
def fetch_props_all_sports(regions: str) -> pd.DataFrame:
    """Fetch props for NBA + NFL across your 6 categories."""
    frames = []
    for sport_key, cat_list in PROP_SPORTS.items():
        markets = ",".join(PROP_MARKET_KEYS[c] for c in cat_list)
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        data = _odds_get(url, {
            "apiKey": ODDS_API_KEY,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american"
        })
        if not data:
            continue
        rows = []
        for ev in data:
            event_id = ev.get("id")
            commence = ev.get("commence_time")
            home, away = ev.get("home_team", "Unknown"), ev.get("away_team", "Unknown")
            for bk in ev.get("bookmakers", []):
                book = bk.get("title")
                for mk in bk.get("markets", []):
                    mkey = mk.get("key")  # e.g., player_points, player_passing_yards
                    for oc in mk.get("outcomes", []):
                        # outcome name should be the player name. "point" is the prop line.
                        rows.append({
                            "sport_key": sport_key,
                            "event_id": event_id,
                            "commence_time": commence,
                            "home_team": home,
                            "away_team": away,
                            "book": book,
                            "market": mkey,
                            "player": oc.get("name"),
                            "line": oc.get("point"),
                            "odds_american": oc.get("price"),
                            "odds_decimal": american_to_decimal(oc.get("price")),
                            "conf_book": implied_prob_american(oc.get("price")),
                        })
        df = pd.DataFrame(rows)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    props = pd.concat(frames, ignore_index=True)
    # time formatting
    props["commence_time"] = pd.to_datetime(props["commence_time"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(props["commence_time"]):
        props["Date/Time"] = props["commence_time"].dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    else:
        props["Date/Time"] = props["commence_time"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%b %d, %I:%M %p ET")
    # Label nice market names
    nice = {
        "player_points": "PTS",
        "player_assists": "AST",
        "player_rebounds": "REB",
        "player_passing_yards": "PassYds",
        "player_rushing_yards": "RushYds",
        "player_receiving_yards": "RecYds",
    }
    props["MarketNice"] = props["market"].map(nice).fillna(props["market"])
    props["Matchup"] = props["home_team"] + " vs " + props["away_team"]
    return props

def score_prop_rows(props: pd.DataFrame, top_per_market: int = 50) -> pd.DataFrame:
    """Rank props by confidence and BMR book presence; pick top items per market."""
    if props.empty:
        return props
    # Best book odds per (player, market, line, event)
    idx_best = props.groupby(["sport_key","event_id","market","player","line"])["odds_decimal"].idxmax()
    best = props.loc[idx_best].copy()

    # Confidence is implied prob from best price; subtle bonus for lines near medians is skipped for now.
    best["Confidence"] = best["conf_book"]
    # small boost if odds are offered at many books:
    books_ct = props.groupby(["sport_key","event_id","market","player","line"])["book"].nunique().reset_index(name="books")
    best = best.merge(books_ct, on=["sport_key","event_id","market","player","line"], how="left")
    best["AdjScore"] = best["Confidence"] + 0.01 * best["books"].clip(upper=10)

    # keep reasonable top N per market to feed parlay builder
    out = (
        best.sort_values("AdjScore", ascending=False)
            .groupby("market", group_keys=False)
            .head(top_per_market)
            .reset_index(drop=True)
    )
    return out

# ─────────────────────────────────────────────
# Parlay builder from top props
# ─────────────────────────────────────────────
def combine_parlay_odds(decimal_odds: List[float]) -> float:
    prod = 1.0
    for d in decimal_odds:
        try:
            if not pd.isna(d) and float(d) > 1.0:
                prod *= float(d)
        except Exception:
            continue
    return round(prod, 4)

def parlay_hit_prob(implied_probs: List[float], corr: float = 0.03) -> float:
    """
    Multiply probabilities with a very small correlation penalty to be conservative.
    corr ~ 0.03 per extra leg.
    """
    p = 1.0
    for ip in implied_probs:
        if pd.isna(ip) or ip <= 0 or ip >= 1:
            continue
        p *= (ip - corr) if (ip - corr) > 0 else 0.0001
    return max(min(p, 0.999), 0.0001)

def build_parlays(top_props: pd.DataFrame, leg_sizes: List[int]) -> List[Dict[str, Any]]:
    """
    Greedy-ish builder:
      - avoid duplicate players in same parlay
      - avoid stacking too many legs from same game in small parlays
      - spread across markets if possible
    """
    parlays = []
    if top_props.empty:
        return parlays

    # shuffle top few per market to mix games/players
    pools = {}
    for m, g in top_props.groupby("market"):
        sh = g.copy().sample(frac=1.0, random_state=42).reset_index(drop=True)
        pools[m] = sh.to_dict(orient="records")

    # flatten in interleaved order (market round-robin)
    flat: List[Dict[str, Any]] = []
    while True:
        added = 0
        for m in sorted(pools.keys()):
            if pools[m]:
                flat.append(pools[m].pop(0))
                added += 1
        if added == 0:
            break

    def valid_add(current: List[Dict[str, Any]], cand: Dict[str, Any]) -> bool:
        # no same player twice
        if any(c["player"] == cand["player"] for c in current):
            return False
        # keep variety of games for small parlays
        if len(current) < 3:
            evt_ids = {c["event_id"] for c in current}
            if cand["event_id"] in evt_ids:
                return False
        return True

    ptr = 0
    for k in leg_sizes:
        legs: List[Dict[str, Any]] = []
        guard = 0
        while len(legs) < k and ptr < len(flat):
            cand = flat[ptr]
            ptr += 1
            guard += 1
            if valid_add(legs, cand):
                legs.append(cand)
            if guard > 2000:
                break

        if not legs:
            continue

        decs = [x.get("odds_decimal", np.nan) for x in legs]
        ips = [x.get("conf_book", np.nan) for x in legs]
        parlay_dec = combine_parlay_odds(decs)
        hit_p = parlay_hit_prob(ips, corr=0.03 if k >= 3 else 0.02)
        parlays.append({
            "legs": legs,
            "legs_count": len(legs),
            "combined_decimal": parlay_dec,
            "implied_parlay_prob": hit_p,
        })

    return parlays

# ─────────────────────────────────────────────
# Results tracking + ROI (unchanged logic)
# ─────────────────────────────────────────────
RESULTS_FILE = "bets.csv"

def load_results() -> pd.DataFrame:
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        # Backward-compat columns
        for col in ["Sport","Market","Date/Time","Matchup","Pick","Line","Odds (US)","Units","Result"]:
            if col not in df.columns:
                df[col] = "" if col != "Units" else 1.0
        df["Result"] = df["Result"].fillna("Pending")
        return df
    return pd.DataFrame(columns=["Sport","Market","Date/Time","Matchup","Pick","Line","Odds (US)","Units","Result"])

def save_results(df: pd.DataFrame):
    df.to_csv(RESULTS_FILE, index=False)

def auto_log_picks(dfs: Dict[str, pd.DataFrame], sport_name: str):
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
    df = df.copy()
    df["Risked"] = df["Units"].astype(float).abs()
    df["PnL"] = df.apply(
        lambda r: r["Units"] if r["Result"] == "Win" else (-r["Units"] if r["Result"] == "Loss" else 0.0), axis=1
    )
    units_won = float(df["PnL"].sum())
    units_risked = float(df.loc[df["Result"].isin(["Win","Loss"]), "Risked"].sum())
    roi = (units_won/units_risked*100.0) if units_risked>0 else 0.0
    win_pct = (wins/total*100.0) if total>0 else 0.0
    return {"win_pct":win_pct,"units_won":units_won,"roi":roi,"wins":wins,"losses":losses,"total":total}

# ─────────────────────────────────────────────
# Manual editors (kept as in your working version)
# ─────────────────────────────────────────────
def show_market_editor(sport_name: str, market_label: str, key_prefix: str):
    results_df = load_results()
    market_df = results_df[(results_df["Sport"] == sport_name) & (results_df["Market"] == market_label)].copy()

    # Summary
    msum = calc_summary(market_df[market_df["Result"].isin(["Win","Loss"])])
    c1,c2,c3 = st.columns(3)
    c1.metric(f"{market_label} Win %", f"{msum['win_pct']:.1f}% ({msum['wins']}-{msum['losses']})")
    c2.metric(f"{market_label} Units Won", f"{msum['units_won']:.1f}")
    c3.metric(f"{market_label} ROI", f"{msum['roi']:.1f}%")

    # Pending
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
                    label = f"{r['Date/Time']} — {r['Matchup']} ({r['Market']}) — Pick: "
                    if r["Market"] == "Totals":
                        label += f"{r['Pick']} ({r['Line']})"
                    elif r["Market"] == "Spreads":
                        try:
                            ln = float(r["Line"])
                            sign = "+" if ln > 0 else ""
                            label += f"{r['Pick']} ({sign}{ln})"
                        except Exception:
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
                            (results_df["Market"] == r["Market"]) &
                            (results_df["Date/Time"] == r["Date/Time"]) &
                            (results_df["Matchup"] == r["Matchup"]) &
                            (results_df["Pick"] == r["Pick"]) &
                            (results_df["Line"].fillna("").astype(str) == str(r["Line"]))
                        )
                        results_df.loc[mask, "Result"] = sel
                        save_results(results_df)
                        st.success("Saved ✅")
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
                    label = f"{r['Date/Time']} — {r['Matchup']} ({r['Market']}) — Pick: "
                    if r["Market"] == "Totals":
                        label += f"{r['Pick']} ({r['Line']})"
                    elif r["Market"] == "Spreads":
                        try:
                            ln = float(r["Line"])
                            sign = "+" if ln > 0 else ""
                            label += f"{r['Pick']} ({sign}{ln})"
                        except Exception:
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
                            (results_df["Market"] == r["Market"]) &
                            (results_df["Date/Time"] == r["Date/Time"]) &
                            (results_df["Matchup"] == r["Matchup"]) &
                            (results_df["Pick"] == r["Pick"]) &
                            (results_df["Line"].fillna("").astype(str) == str(r["Line"]))
                        )
                        results_df.loc[mask, "Result"] = sel
                        save_results(results_df)
                        st.success("Saved ✅")
        else:
            st.info("No completed picks yet.")

def show_results_summary(sport_name: str):
    results = load_results()
    filt = results[(results["Sport"] == sport_name) & (results["Market"].isin(["Moneyline","Spreads","Totals"]))].copy()
    if filt.empty:
        st.info(f"No bets logged yet for {sport_name}.")
        return
    st.subheader(f"📊 Results — {sport_name} (Moneyline / Spreads / Totals)")
    st.dataframe(filt, use_container_width=True, hide_index=True)
    summ = calc_summary(filt[filt["Result"].isin(["Win","Loss"])])
    c1,c2,c3 = st.columns(3)
    c1.metric("Win %", f"{summ['win_pct']:.1f}% ({summ['wins']}-{summ['losses']})")
    c2.metric("Units Won", f"{summ['units_won']:.1f}")
    c3.metric("ROI", f"{summ['roi']:.1f}%")

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
    fetch = st.button("Fetch Live Odds & Props")

# ─────────────────────────────────────────────
# Fetch + stash in session_state
# ─────────────────────────────────────────────
if fetch:
    # sides/totals for selected sport or soccer group
    sport_key = SPORT_OPTIONS[sport_name]
    if isinstance(sport_key, list):
        parts = [fetch_odds(k, regions) for k in sport_key]
        raw = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    else:
        raw = fetch_odds(sport_key, regions)

    if raw.empty:
        st.warning("No sides/totals data returned. Try a different sport or check API quota.")
        st.session_state.has_data = False
    else:
        ai_picks, ml, totals, spreads, cons = None, None, None, None, None
        cons = build_consensus(raw)
        ml = pick_best_per_event(cons,"h2h",top_n)
        totals = pick_best_per_event(cons,"totals",top_n)
        spreads = pick_best_per_event(cons,"spreads",top_n)
        ai_picks = ai_genius_top(cons, min(top_n,5))

        # Log to results (dedup-safe)
        auto_log_picks({"AI Genius": ai_picks, "Moneyline": ml, "Totals": totals, "Spreads": spreads}, sport_name)

        # PROPS (NBA + NFL) for parlays
        props_all = fetch_props_all_sports(regions)
        top_props = score_prop_rows(props_all, top_per_market=60) if not props_all.empty else pd.DataFrame()
        generated_parlays = build_parlays(top_props, leg_sizes=[2,2,3,5,6]) if not top_props.empty else []

        # stash
        st.session_state.raw = raw
        st.session_state.ai_picks = ai_picks
        st.session_state.ml = ml
        st.session_state.totals = totals
        st.session_state.spreads = spreads
        st.session_state.cons = cons
        st.session_state.props_all = props_all
        st.session_state.top_props = top_props
        st.session_state.parlays = generated_parlays
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
        "🧩 Parlays",
        "Raw Data",
        "📊 Results"
    ])

    # Tab 0 — AI Genius (Units shown)
    with tabs[0]:
        st.subheader("AI Genius — Highest Consensus Confidence (Top)")
        st.dataframe(st.session_state.ai_picks, use_container_width=True, hide_index=True)
        if not st.session_state.ai_picks.empty:
            st.caption("Units sized by consensus confidence.")
            st.bar_chart(
                pd.DataFrame(
                    {"Units": st.session_state.ai_picks["Units"].astype(float).values},
                    index=st.session_state.ai_picks["Matchup"].astype(str).values
                )
            )
        st.markdown("---")
        st.markdown("### ✍️ Manual Result Editor — AI Genius")
        show_market_editor(sport_name, "AI Genius", key_prefix="ai")

    # Tab 1 — Moneylines
    with tabs[1]:
        st.subheader("Best Moneyline per Game (Consensus)")
        st.dataframe(st.session_state.ml, use_container_width=True, hide_index=True)
        st.caption("Confidence heat — Moneylines")
        if not st.session_state.ml.empty:
            conf_vals = st.session_state.ml["Confidence"].str.replace("%","",regex=False).astype(float)
            st.bar_chart(pd.DataFrame({"Confidence": conf_vals.values},
                                      index=st.session_state.ml["Matchup"].astype(str).values))
        st.markdown("---")
        st.markdown("### ✍️ Manual Result Editor — Moneyline")
        show_market_editor(sport_name, "Moneyline", key_prefix="ml")

    # Tab 2 — Totals
    with tabs[2]:
        st.subheader("Best Totals per Game (Consensus)")
        st.dataframe(st.session_state.totals, use_container_width=True, hide_index=True)
        st.caption("Confidence heat — Totals")
        if not st.session_state.totals.empty:
            conf_vals = st.session_state.totals["Confidence"].str.replace("%","",regex=False).astype(float)
            st.bar_chart(pd.DataFrame({"Confidence": conf_vals.values},
                                      index=st.session_state.totals["Matchup"].astype(str).values))
        st.markdown("---")
        st.markdown("### ✍️ Manual Result Editor — Totals")
        show_market_editor(sport_name, "Totals", key_prefix="tot")

    # Tab 3 — Spreads
    with tabs[3]:
        st.subheader("Best Spreads per Game (Consensus)")
        st.dataframe(st.session_state.spreads, use_container_width=True, hide_index=True)
        st.caption("Confidence heat — Spreads")
        if not st.session_state.spreads.empty:
            conf_vals = st.session_state.spreads["Confidence"].str.replace("%","",regex=False).astype(float)
            st.bar_chart(pd.DataFrame({"Confidence": conf_vals.values},
                                      index=st.session_state.spreads["Matchup"].astype(str).values))
        st.markdown("---")
        st.markdown("### ✍️ Manual Result Editor — Spreads")
        show_market_editor(sport_name, "Spreads", key_prefix="spr")

    # Tab 4 — Parlays (NEW)
    with tabs[4]:
        st.subheader("5 Daily Parlays (built from NBA/NFL props)")
        parlays = st.session_state.get("parlays", [])
        top_props = st.session_state.get("top_props", pd.DataFrame())
        if not parlays:
            st.info("No props returned from the Odds API (plan/market coverage). Try again later or enable props in your Odds API plan.")
        else:
            for idx, p in enumerate(parlays, start=1):
                with st.container(border=True):
                    st.markdown(f"**Parlay {idx} — {p['legs_count']} legs**")
                    legs_rows = []
                    for leg in p["legs"]:
                        legs_rows.append({
                            "Player": leg.get("player",""),
                            "Market": leg.get("MarketNice", leg.get("market","")),
                            "Line": leg.get("line",""),
                            "Matchup": (leg.get("home_team","") + " vs " + leg.get("away_team","")).strip(),
                            "Book": leg.get("book",""),
                            "Odds(US)": leg.get("odds_american",""),
                            "Odds(Dec)": round(float(leg.get("odds_decimal", np.nan)), 3) if not pd.isna(leg.get("odds_decimal", np.nan)) else ""
                        })
                    st.dataframe(pd.DataFrame(legs_rows), use_container_width=True, hide_index=True)
                    st.write(f"**Combined Decimal:** {p['combined_decimal']}  |  **Conservative Hit Chance:** {fmt_pct(p['implied_parlay_prob'])}")
                    if p["implied_parlay_prob"] >= 0.40:
                        st.caption("Risk Guide: Safer (2–3 legs)")
                    elif p["implied_parlay_prob"] >= 0.20:
                        st.caption("Risk Guide: Medium (3–5 legs)")
                    else:
                        st.caption("Risk Guide: Lottery (5+ legs)")

    # Tab 5 — Raw
    with tabs[5]:
        st.subheader("Raw Per-Book Odds (first 200 rows)")
        st.dataframe(st.session_state.raw.head(200), use_container_width=True, hide_index=True)
        st.caption("Tip: this is the source that feeds the consensus tables.")

    # Tab 6 — Results summary
    with tabs[6]:
        show_results_summary(sport_name)

else:
    st.info("Pick a sport and click **Fetch Live Odds & Props**")
