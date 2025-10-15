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

ODDS_API_KEY = os.getenv("ODDS_API_KEY", st.secrets.get("ODDS_API_KEY", ""))
APISPORTS_KEY = os.getenv("APISPORTS_KEY", st.secrets.get("APISPORTS_KEY", ""))
DEFAULT_REGIONS = os.getenv("REGIONS", "us")

SOCCER_KEYS = [
    "soccer_epl","soccer_spain_la_liga","soccer_italy_serie_a",
    "soccer_france_ligue_one","soccer_germany_bundesliga","soccer_uefa_champions_league",
]

SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "College Football (NCAAF)": "americanfootball_ncaaf",
    "College Basketball (NCAAB)": "basketball_ncaab",
    "Soccer (All Major Leagues)": SOCCER_KEYS,
}

SPORT_API_ENDPOINTS = {
    "NFL": "https://v1.american-football.api-sports.io/games?league=1&season=2023",
    "NBA": "https://v1.basketball.api-sports.io/games?league=12&season=2023",
    "MLB": "https://v1.baseball.api-sports.io/games?league=1&season=2023",
    "College Football (NCAAF)": "https://v1.american-football.api-sports.io/games?league=2&season=2023",
    "College Basketball (NCAAB)": "https://v1.basketball.api-sports.io/games?league=7&season=2023",
    "Soccer (All Major Leagues)": "https://v3.football.api-sports.io/fixtures?season=2023&league=39"
}

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
    return 1 + (o/100.0) if o > 0 else 1 + (100.0/abs(o))

def implied_prob_american(odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds): return np.nan
    o = float(odds)
    return 100.0/(o+100.0) if o > 0 else abs(o)/(abs(o)+100.0)

def assign_units(conf: float) -> float:
    if pd.isna(conf): return 0.5
    return round(0.5+4.5*max(0.0,min(1.0,conf)),1)

def fmt_pct(x: float) -> str:
    return "" if (x is None or pd.isna(x)) else f"{100.0*x:.1f}%"

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
# Consensus + AI Genius Picks
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
    frames=[]
    for m in ["h2h","totals","spreads"]:
        t = pick_best_per_event(cons_df,m,top_n*3)
        if not t.empty:
            t["_C"] = t["Confidence"].str.replace("%","",regex=False).astype(float)
            frames.append(t)
    if not frames: return pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True)
    allp = allp.sort_values("_C",ascending=False).drop(columns=["_C"]).head(top_n)
    return allp.reset_index(drop=True)

# ─────────────────────────────────────────────
# Results tracking + auto update
# ─────────────────────────────────────────────
RESULTS_FILE="bets.csv"

def load_results()->pd.DataFrame:
    if os.path.exists(RESULTS_FILE):
        df=pd.read_csv(RESULTS_FILE)
        if "Sport" not in df.columns: df["Sport"]="Unknown"
        if "Market" not in df.columns: df["Market"]="Unknown"
        return df
    return pd.DataFrame(columns=["Sport","Market","Date/Time","Matchup","Pick","Line","Odds (US)","Units","Result"])

def save_results(df:pd.DataFrame):
    df.to_csv(RESULTS_FILE,index=False)

def update_results_auto(sport_name:str)->pd.DataFrame:
    results=load_results()
    if results.empty: return results
    headers={"x-apisports-key":APISPORTS_KEY}
    url=SPORT_API_ENDPOINTS.get(sport_name)
    if not url: return results
    try:
        r=requests.get(url,headers=headers,timeout=30)
        if r.status_code==200:
            games=r.json().get("response",[])
            for i,row in results[(results["Sport"]==sport_name)&(results["Result"]=="Pending")].iterrows():
                for g in games:
                    home=g.get("teams",{}).get("home",{}).get("name")
                    away=g.get("teams",{}).get("away",{}).get("name")
                    winner=g.get("teams",{}).get("winner",{}).get("name",None)
                    if home and away and row["Matchup"]==f"{home} vs {away}":
                        if winner==row["Pick"]: results.at[i,"Result"]="Win"
                        elif winner and winner!=row["Pick"]: results.at[i,"Result"]="Loss"
    except Exception: pass
    save_results(results)
    return results

def show_results(sport_name:str):
    results=update_results_auto(sport_name)
    sport_results=results[results["Sport"]==sport_name].copy()
    if sport_results.empty:
        st.info(f"No bets logged yet for {sport_name}."); return
    st.subheader(f"📊 Results — {sport_name}")
    st.dataframe(sport_results,use_container_width=True,hide_index=True)
    total=len(sport_results)
    wins=(sport_results["Result"]=="Win").sum()
    losses=(sport_results["Result"]=="Loss").sum()
    sport_results["PnL"]=sport_results.apply(lambda r:r["Units"] if r["Result"]=="Win" else (-r["Units"] if r["Result"]=="Loss" else 0.0),axis=1)
    bankroll=sport_results["PnL"].sum()
    if total>0:
        win_pct=(wins/total)*100
        c1,c2=st.columns(2)
        c1.metric("Win %",f"{win_pct:.1f}% ({wins}-{losses})")
        c2.metric("Bankroll (Units)",f"{bankroll:.1f}")

# ─────────────────────────────────────────────
# Sidebar + Main
# ─────────────────────────────────────────────
with st.sidebar:
    sport_name=st.selectbox("Sport",list(SPORT_OPTIONS.keys()),index=0)
    regions=st.text_input("Regions",value=DEFAULT_REGIONS)
    top_n=st.slider("Top picks per tab",3,20,10)
    fetch=st.button("Fetch Live Odds")

# Always update results when page loads
show_results(sport_name)

if fetch:
    sport_key=SPORT_OPTIONS[sport_name]
    raw=pd.concat([fetch_odds(k,regions) for k in sport_key],ignore_index=True) if isinstance(sport_key,list) else fetch_odds(sport_key,regions)
    if raw.empty:
        st.warning("No data returned.")
    else:
        cons=build_consensus(raw)
        ml=pick_best_per_event(cons,"h2h",top_n)
        totals=pick_best_per_event(cons,"totals",top_n)
        spreads=pick_best_per_event(cons,"spreads",top_n)
        ai_picks=ai_genius_top(cons,min(top_n,5))

        # Auto-log picks
        for label,dfp in {"AI Genius":ai_picks,"Moneyline":ml,"Totals":totals,"Spreads":spreads}.items():
            if not dfp.empty:
                results=load_results()
                for _,row in dfp.iterrows():
                    entry={"Sport":sport_name,"Market":label,"Date/Time":row["Date/Time"],"Matchup":row["Matchup"],"Pick":row["Pick"],"Line":row.get("Line",""),"Odds (US)":row.get("Odds (US)",""),"Units":row.get("Units",1.0),"Result":"Pending"}
                    dup=((results["Sport"]==entry["Sport"])&(results["Market"]==entry["Market"])&(results["Date/Time"]==entry["Date/Time"])&(results["Matchup"]==entry["Matchup"])&(results["Pick"]==entry["Pick"]))
                    if not dup.any(): results=pd.concat([results,pd.DataFrame([entry])],ignore_index=True)
                save_results(results)

        tabs=st.tabs(["🤖 AI Genius Picks","Moneylines","Totals","Spreads","Raw Data","📊 Results"])
        with tabs[0]: st.dataframe(ai_picks,use_container_width=True,hide_index=True)
        with tabs[1]: st.dataframe(ml,use_container_width=True,hide_index=True)
        with tabs[2]: st.dataframe(totals,use_container_width=True,hide_index=True)
        with tabs[3]: st.dataframe(spreads,use_container_width=True,hide_index=True)
        with tabs[4]: st.dataframe(raw.head(200),use_container_width=True,hide_index=True)
        with tabs[5]: show_results(sport_name)
