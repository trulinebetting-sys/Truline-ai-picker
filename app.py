# app.py
from __future__ import annotations
import re
from typing import List, Dict, Tuple
import math

import pandas as pd
import numpy as np
import streamlit as st

# -------------------------------
# Page config & minimalist theme
# -------------------------------
st.set_page_config(page_title="TruLine Betting • AI Picker v0", page_icon="🧠", layout="wide")

# Hide Streamlit's default sidebar toggle to keep things clean
st.markdown(
    """
    <style>
      [data-testid="collapsedControl"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Small helpers
# -------------------------------
def american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability (no vig removal)."""
    try:
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return (-odds) / ((-odds) + 100.0)
    except Exception:
        return np.nan

def devig_two_sides(p_a: float, p_b: float) -> Tuple[float, float]:
    """Remove vig by normalizing two implied probabilities."""
    total = p_a + p_b
    if total <= 0 or not np.isfinite(total):
        return np.nan, np.nan
    return p_a / total, p_b / total

def prob_to_decimal(p: float) -> float:
    return 1.0 / p if p > 0 else np.nan

def kelly_fraction(p_true: float, dec_odds: float) -> float:
    """
    Kelly fraction for decimal odds.
    f* = (b*p - q) / b where b = dec_odds - 1, q = 1 - p.
    Returns 0 if not profitable.
    """
    if not np.isfinite(p_true) or not np.isfinite(dec_odds) or dec_odds <= 1.0:
        return 0.0
    b = dec_odds - 1.0
    q = 1.0 - p_true
    f = (b * p_true - q) / b
    return max(0.0, f)

def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%" if np.isfinite(x) else ""

def fmt_money(x: float) -> str:
    return f"${x:,.2f}"

def parse_freeform_lines(lines: str) -> List[Dict]:
    """
    Parse lines like:
      Dallas Cowboys -110 @ Philadelphia Eagles +100 | book=DraftKings
      Arsenal -145 vs Newcastle +130 | book=FanDuel
      Heat +120 at Celtics -140
    """
    out = []
    for raw in lines.splitlines():
        line = raw.strip()
        if not line:
            continue

        # pick a separator token around team names
        m = re.search(r"\s(@|vs|at)\s", line, flags=re.I)
        if not m:
            # not parseable enough, skip
            continue
        sep = m.group(1).lower()
        left = line[:m.start()].strip()
        right_and_tail = line[m.end():].strip()

        # optional book at end: "| book=FanDuel"
        book = None
        if "| book=" in right_and_tail:
            right, tail = right_and_tail.split("| book=", 1)
            book = tail.strip()
        else:
            right = right_and_tail

        # extract team + american odds pairs: e.g. "Dallas Cowboys -110"
        def split_team_odds(segment: str) -> Tuple[str, float]:
            mm = re.search(r"([+-]?\d+)\s*$", segment)
            if mm:
                odds = float(mm.group(1))
                team = segment[:mm.start()].strip()
                return team, odds
            return segment.strip(), float("nan")

        team_a, odds_a = split_team_odds(left)
        team_b, odds_b = split_team_odds(right)

        out.append({
            "team_a": team_a,
            "team_b": team_b,
            "odds_a": odds_a,
            "odds_b": odds_b,
            "book": book or "",
            "sep": sep
        })
    return out

# -------------------------------
# App UI
# -------------------------------
st.title("🧠 TruLine Betting — AI Picker v0")
st.caption("Minimal, no-API prototype: paste games + odds, get picks with de-vig, heuristic adjustment and Kelly sizing.")

with st.expander("How to use (quick)"):
    st.markdown(
        """
        Paste one game per line. Examples:
        - `Dallas Cowboys -110 @ Philadelphia Eagles +100 | book=DraftKings`  
        - `Arsenal -145 vs Newcastle +130 | book=FanDuel`  
        - `Heat +120 at Celtics -140`  

        **Then** pick a league, bankroll, and sliders below.  
        We de-vig the market, apply a tiny model bias, compute EV and capped Kelly stake.
        """
    )

c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
with c1:
    league = st.selectbox("League (for small defaults only)", ["NFL", "NBA", "MLB", "Soccer"], index=0)
with c2:
    bankroll = st.number_input("Bankroll ($)", min_value=10.0, value=1000.0, step=50.0)
with c3:
    kelly_cap = st.slider("Kelly cap (fraction)", 0.0, 1.0, 0.25, 0.05)
with c4:
    min_edge = st.slider("Min Edge to show", 0.00, 0.20, 0.02, 0.01)

st.markdown("### Paste games + odds")
raw = st.text_area(
    "One per line — examples above",
    height=160,
    placeholder="Lions -120 @ Bears +110 | book=FanDuel\nArsenal -145 vs Newcastle +130 | book=DraftKings"
)

bias = st.slider("Model bias (advanced): favor favorite (+) or underdog (−)", -0.08, 0.08, 0.00, 0.01)
hfa = {"NFL": 0.03, "NBA": 0.06, "MLB": 0.04, "Soccer": 0.04}.get(league, 0.04)

if st.button("Run AI Picker"):
    rows = parse_freeform_lines(raw)
    if not rows:
        st.warning("I couldn't parse any games. Use the examples above as a template.")
        st.stop()

    records = []
    for r in rows:
        p_a_raw = american_to_prob(r["odds_a"])
        p_b_raw = american_to_prob(r["odds_b"])
        p_a, p_b = devig_two_sides(p_a_raw, p_b_raw)

        # tiny heuristic: favorite tends to be the side with larger (more negative) absolute odds
        # apply bias (and a small home-field nudge)
        is_a_fav = (abs(r["odds_a"]) < abs(r["odds_b"])) if (np.isfinite(r["odds_a"]) and np.isfinite(r["odds_b"])) else False
        sep = r["sep"]  # '@', 'vs', 'at'
        a_is_away = (sep == '@' or sep == 'at')

        adj_a = bias * (1 if is_a_fav else -1)
        adj_b = -adj_a

        # home-field: if A is away, give B slight bump, else A
        if a_is_away:
            adj_b += hfa
        else:
            adj_a += hfa

        # renormalize after adjustments to keep sum ~1
        if np.isfinite(p_a) and np.isfinite(p_b):
            p_a_true = max(1e-6, min(0.999, p_a + adj_a))
            p_b_true = max(1e-6, min(0.999, p_b + adj_b))
            s = p_a_true + p_b_true
            p_a_true, p_b_true = p_a_true / s, p_b_true / s
        else:
            p_a_true = p_b_true = np.nan

        dec_a = prob_to_decimal(p_a_raw)  # use raw for EV comparison vs offered price
        dec_b = prob_to_decimal(p_b_raw)

        ev_a = (dec_a * p_a_true - 1.0) if np.isfinite(dec_a) else np.nan
        ev_b = (dec_b * p_b_true - 1.0) if np.isfinite(dec_b) else np.nan

        k_a = kelly_fraction(p_a_true, dec_a) * kelly_cap if np.isfinite(dec_a) else 0.0
        k_b = kelly_fraction(p_b_true, dec_b) * kelly_cap if np.isfinite(dec_b) else 0.0

        stake_a = bankroll * k_a
        stake_b = bankroll * k_b

        pick_side = "A" if (ev_a > ev_b) else "B"
        pick_team = r["team_a"] if pick_side == "A" else r["team_b"]
        pick_ev = max(ev_a, ev_b)
        pick_stake = stake_a if pick_side == "A" else stake_b
        pick_odds = r["odds_a"] if pick_side == "A" else r["odds_b"]

        records.append({
            "Matchup": f'{r["team_a"]} {sep} {r["team_b"]}',
            "Book": r["book"],
            "Pick": pick_team,
            "Odds (Am)": pick_odds,
            "Implied Prob": p_a_raw if pick_side=="A" else p_b_raw,
            "Expected Prob": p_a_true if pick_side=="A" else p_b_true,
            "Edge %": pick_ev,
            "Stake $": pick_stake,
        })

    df = pd.DataFrame.from_records(records)
    if df.empty:
        st.info("No valid rows.")
        st.stop()

    # tidy formats
    df["Implied Prob"] = df["Implied Prob"].apply(lambda p: fmt_pct(p) if np.isfinite(p) else "")
    df["Expected Prob"] = df["Expected Prob"].apply(lambda p: fmt_pct(p) if np.isfinite(p) else "")
    df["Edge %"] = df["Edge %"].apply(lambda x: fmt_pct(x) if np.isfinite(x) else "")
    df["Stake $"] = df["Stake $"].apply(lambda x: fmt_money(x) if np.isfinite(x) else "")

    # filter by min edge (string now, so filter from numeric again)
    def pct_to_float(s: str) -> float:
        try:
            return float(s.replace("%",""))/100.0
        except Exception:
            return -1

    df_numeric_edge = df.copy()
    df_numeric_edge["EdgeFloat"] = df["Edge %"].apply(pct_to_float)
    df_show = df_numeric_edge[df_numeric_edge["EdgeFloat"] >= min_edge].drop(columns=["EdgeFloat"])

    st.success(f"AI picks found: {len(df_show)} (of {len(df)})")
    st.dataframe(df_show, hide_index=True, use_container_width=True)

    csv = df_show.to_csv(index=False).encode("utf-8")
    st.download_button("Download picks (CSV)", data=csv, file_name="ai_picks.csv", mime="text/csv")
else:
    st.info("Paste a few games (see examples) and press **Run AI Picker**.")
