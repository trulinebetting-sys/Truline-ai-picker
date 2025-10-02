import os
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timezone
from dotenv import load_dotenv

# -----------------------------
# Config / Env
# -----------------------------
load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY", "")
REGIONS = os.getenv("REGIONS", "us")
BOOKS_ENV = os.getenv("BOOKS", "DraftKings,FanDuel,BetMGM,Bet365,PointsBet,Caesars,Barstool,BetRivers,Unibet")
DEFAULT_BOOKS = [b.strip() for b in BOOKS_ENV.split(",") if b.strip()]
MARKETS_ENV = os.getenv("MARKETS", "h2h,spreads,totals")

st.set_page_config(page_title="TruLine Betting • AI Genius", page_icon="assets/logo.png", layout="wide")

# Header
if os.path.exists("assets/logo.png"):
    st.image("assets/logo.png", width=110)
st.title("🤖 TruLine Betting — AI Genius Picks")
st.caption("Consensus-driven fair probabilities • EV ranking • Kelly sizing • Parlay builder")

# -----------------------------
# Odds helpers
# -----------------------------
def american_to_decimal(odds: float) -> float:
    """American -> Decimal"""
    if odds is None:
        return None
    if odds > 0:
        return 1 + (odds / 100.0)
    else:
        return 1 + (100.0 / abs(odds))

def implied_prob_from_american(odds: float) -> float:
    """American -> implied probability (with vig)"""
    if odds is None:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)

def devig_2way(p1: float, p2: float):
    """Remove vig from a 2-outcome market (normalize to 1). Returns (p1, p2)."""
    if p1 is None or p2 is None:
        return None, None
    s = p1 + p2
    if s <= 0:
        return None, None
    return p1 / s, p2 / s

def kelly_fraction(p: float, dec_odds: float):
    """Kelly fraction for decimal odds."""
    if p is None or dec_odds is None:
        return 0.0
    b = dec_odds - 1.0
    q = 1.0 - p
    k = (b * p - q) / b
    return max(0.0, k)

def ev_per_dollar(p: float, dec_odds: float):
    """Expected value per $1 stake."""
    if p is None or dec_odds is None:
        return None
    b = dec_odds - 1.0
    return p * b - (1 - p)

def prob_to_american(prob: float):
    """Probability -> American odds (approx)."""
    if prob <= 0 or prob >= 1:
        return None
    dec = 1.0 / prob
    if dec >= 2:
        return int(round((dec - 1) * 100))
    else:
        return int(round(-100 / (dec - 1)))

def parlay_american(american_list):
    """Combine American odds into parlay American odds via probability domain."""
    probs = [implied_prob_from_american(o) for o in american_list if o is not None]
    if not probs:
        return None
    combined_p = np.prod(probs)
    return prob_to_american(combined_p)

# -----------------------------
# API
# -----------------------------
def fetch_odds(sport_key: str, regions: str, markets: str, books_csv: str):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "bookmakers": books_csv,
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return None, f"API error {r.status_code}: {r.text}"
    return r.json(), None

# -----------------------------
# UI Controls
# -----------------------------
if not API_KEY:
    st.error("Missing ODDS_API_KEY in your environment (.env). Add it and rerun.")
    st.stop()

sport = st.selectbox(
    "Sport",
    [
        "americanfootball_nfl",
        "basketball_nba",
        "baseball_mlb",
        "icehockey_nhl",
        "soccer_epl",
        "soccer_uefa_champs_league",
    ],
    index=1
)

market_choices = ["h2h", "spreads", "totals"]
selected_markets = st.multiselect("Markets", market_choices, default=market_choices)

books_selected = st.multiselect("Books to include", DEFAULT_BOOKS, default=DEFAULT_BOOKS)

c1, c2, c3, c4 = st.columns(4)
with c1:
    min_edge = st.slider("Min Edge (%)", 0.0, 10.0, 1.0, 0.1)
with c2:
    bankroll = st.number_input("Bankroll ($)", min_value=10.0, value=1000.0, step=50.0)
with c3:
    kelly_cap = st.slider("Kelly Cap", 0.0, 1.0, 0.25, 0.05)
with c4:
    top_n = st.number_input("Show top N picks", min_value=1, value=10, step=1)

st.markdown("---")

# -----------------------------
# Fetch + Build fair probs & EV
# -----------------------------
data, err = fetch_odds(
    sport_key=sport,
    regions=REGIONS,
    markets=",".join(selected_markets),
    books_csv=",".join(books_selected)
)

if err:
    st.error(err)
    st.stop()

if not data:
    st.warning("No events returned. Try another sport/region or check API quota.")
    st.stop()

rows = []

for ev in data:
    commence_iso = ev.get("commence_time")
    try:
        dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
        kickoff = dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        kickoff = commence_iso

    home = ev.get("home_team", "Home")
    away = ev.get("away_team", "Away")
    game_name = f"{away} @ {home}"

    # Build book-level pricing for each market/outcome
    # We'll compute fair probs per market via de-vig per book then average.
    market_bucket = {}  # e.g., {"h2h": {"home": [p_i], "away": [p_i]}, ...}
    best_offers = {}    # track best price + book per outcome for EV

    for bk in ev.get("bookmakers", []):
        book = bk.get("title")
        if book not in books_selected:
            continue
        for mk in bk.get("markets", []):
            mkey = mk.get("key")
            if mkey not in selected_markets:
                continue

            # Map outcomes
            outs = mk.get("outcomes", [])
            if mkey == "h2h":
                # two-way: home vs away
                price_home = None
                price_away = None
                for oc in outs:
                    name = oc.get("name")
                    price = oc.get("price")
                    if name == home:
                        price_home = price
                    elif name == away:
                        price_away = price

                p_home_raw = implied_prob_from_american(price_home) if price_home is not None else None
                p_away_raw = implied_prob_from_american(price_away) if price_away is not None else None
                p_home, p_away = devig_2way(p_home_raw, p_away_raw)

                if p_home is not None and p_away is not None:
                    market_bucket.setdefault(("h2h", "home"), []).append(p_home)
                    market_bucket.setdefault(("h2h", "away"), []).append(p_away)

                # Track best offer (highest decimal) for each side
                for side, price in [("home", price_home), ("away", price_away)]:
                    if price is None:
                        continue
                    dec = american_to_decimal(price)
                    key = ("h2h", side)
                    cur = best_offers.get(key)
                    if cur is None or dec > cur["decimal"]:
                        best_offers[key] = {"decimal": dec, "american": price, "book": book}

            elif mkey == "spreads":
                # We treat spread as two-way at given point; choose the side with its point
                # We'll aggregate by (team,point)
                # For EV we compare the same (team,point) across books; here we just take book's pair de-vig.
                if len(outs) >= 2:
                    # usually two outcomes
                    o1, o2 = outs[0], outs[1]
                    p1_raw = implied_prob_from_american(o1.get("price"))
                    p2_raw = implied_prob_from_american(o2.get("price"))
                    p1, p2 = devig_2way(p1_raw, p2_raw)
                    if p1 is not None and p2 is not None:
                        key1 = ("spreads", f"{o1.get('name')} {o1.get('point')}")
                        key2 = ("spreads", f"{o2.get('name')} {o2.get('point')}")
                        market_bucket.setdefault(key1, []).append(p1)
                        market_bucket.setdefault(key2, []).append(p2)

                        # best offer tracking
                        for oc, p_fair in [(o1, p1), (o2, p2)]:
                            price = oc.get("price")
                            if price is None:
                                continue
                            dec = american_to_decimal(price)
                            k = ("spreads", f"{oc.get('name')} {oc.get('point')}")
                            cur = best_offers.get(k)
                            if cur is None or dec > cur["decimal"]:
                                best_offers[k] = {"decimal": dec, "american": price, "book": book}

            elif mkey == "totals":
                # Over/Under with a point
                if len(outs) >= 2:
                    o1, o2 = outs[0], outs[1]
                    p1_raw = implied_prob_from_american(o1.get("price"))
                    p2_raw = implied_prob_from_american(o2.get("price"))
                    p1, p2 = devig_2way(p1_raw, p2_raw)
                    if p1 is not None and p2 is not None:
                        key1 = ("totals", f"{o1.get('name')} {o1.get('point')}")
                        key2 = ("totals", f"{o2.get('name')} {o2.get('point')}")
                        market_bucket.setdefault(key1, []).append(p1)
                        market_bucket.setdefault(key2, []).append(p2)

                        for oc, p_fair in [(o1, p1), (o2, p2)]:
                            price = oc.get("price")
                            if price is None:
                                continue
                            dec = american_to_decimal(price)
                            k = ("totals", f"{oc.get('name')} {oc.get('point')}")
                            cur = best_offers.get(k)
                            if cur is None or dec > cur["decimal"]:
                                best_offers[k] = {"decimal": dec, "american": price, "book": book}

    # Now compute consensus fair probs (mean of devigged)
    fair = {}
    for key, plist in market_bucket.items():
        if plist:
            fair[key] = float(np.mean(plist))

    # Build EV rows using best offers
    for key, offer in best_offers.items():
        mkey, outcome = key
        fair_p = fair.get(key)
        if fair_p is None:
            continue
        dec = offer["decimal"]
        ev = ev_per_dollar(fair_p, dec)
        k = kelly_fraction(fair_p, dec)

        rows.append({
            "kickoff": kickoff,
            "event": game_name,
            "market": mkey,
            "outcome": outcome,
            "fair_prob": fair_p,
            "best_book": offer["book"],
            "best_american": offer["american"],
            "best_decimal": dec,
            "edge_pct": ev * 100.0 if ev is not None else None,
            "kelly": k,
        })

# -----------------------------
# Table + Filters + Parlay
# -----------------------------
df = pd.DataFrame(rows)
if df.empty:
    st.info("No priced markets found across your selected books.")
    st.stop()

# Filter by min edge
df = df[df["edge_pct"] >= min_edge]
# Sort by edge then by kickoff
df = df.sort_values(by=["edge_pct", "kickoff"], ascending=[False, True]).reset_index(drop=True)

# Pretty formatting
def fmt_pct(x, digits=1):
    return f"{x:.{digits}f}%" if pd.notnull(x) else ""

def fmt_prob(x):
    return f"{x*100:.1f}%" if pd.notnull(x) else ""

show = df.copy()
show["Fair Prob"] = show["fair_prob"].apply(fmt_prob)
show["Edge"] = show["edge_pct"].apply(lambda v: fmt_pct(v, 2))
show["Kelly (cap-ready)"] = (show["kelly"] * 100).apply(lambda v: fmt_pct(v, 2))
show["Odds"] = show["best_american"].astype(int)
show = show[[
    "kickoff", "event", "market", "outcome", "best_book", "Odds", "Fair Prob", "Edge", "Kelly (cap-ready)"
]]
show = show.rename(columns={
    "kickoff": "Date/Time (UTC)",
    "event": "Matchup",
    "market": "Market",
    "outcome": "Selection",
    "best_book": "Book",
})

st.subheader("Top Value Picks")
st.dataframe(show.head(int(top_n)), use_container_width=True, hide_index=True)

# Recommended stake (capped Kelly)
st.markdown("##### Suggested stake (capped Kelly)")
if not df.empty:
    top_row = df.iloc[0]
    full_k = top_row["kelly"]
    stake = max(0.0, min(full_k * bankroll, bankroll)) * kelly_cap
    st.write(f"**Top pick suggested stake:** ${stake:.2f} (on {top_row['outcome']} @ {int(top_row['best_american'])} at {top_row['best_book']})")

# Parlay: take the top N unique events
st.subheader("Suggested Parlay")
parlay_legs = min(3, len(df))
legs = []
seen_events = set()
for _, r in df.iterrows():
    if len(legs) >= parlay_legs:
        break
    if r["event"] in seen_events:
        continue
    legs.append(r)
    seen_events.add(r["event"])

if len(legs) >= 2:
    american_list = [int(l["best_american"]) for l in legs]
    parlay_price = parlay_american(american_list)
    leg_text = " • ".join([f"{l['outcome']} ({int(l['best_american'])})" for l in legs])
    if parlay_price is not None:
        st.success(f"{parlay_legs}-leg parlay: {leg_text} → **{parlay_price}**")
else:
    st.info("Need at least 2 eligible legs to build a parlay. Try lowering filters.")
