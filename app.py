from __future__ import annotations
import pandas as pd
import streamlit as st
import math
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TruLine Betting • AI Genius Picks",
    page_icon="assets/logo.png",   # favicon in browser tab
    layout="wide"
)

# --- HIDE SIDEBAR ---
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        section[data-testid="stSidebar"] {display: none !important;}
        .block-container {padding-top: 1rem;}
        h1, h2, h3, h4, h5 {font-family: 'Comfortaa', sans-serif;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- HEADER ---
col1, col2 = st.columns([1,5])
with col1:
    st.image("assets/logo.png", width=90)
with col2:
    st.title("AI Genius Picks")

st.caption("⚡ Simple AI-based suggestions (for personal testing only).")

# --- MODEL HELPERS ---
def american_to_decimal(a: float) -> float:
    return 1 + (a/100) if a > 0 else 1 + (100/abs(a))

def implied_prob(a: float) -> float:
    return 100/(a+100) if a > 0 else abs(a)/(abs(a)+100)

def sigmoid(x: float, k=0.035) -> float:
    return 1/(1+math.exp(-k*x))

def model_prob(home_rating: int, away_rating: int) -> tuple[float,float]:
    diff = (home_rating - away_rating) + 10
    hp = sigmoid(diff)
    return hp, 1-hp

def kelly_fraction(p: float, dec: float) -> float:
    b = dec - 1
    return max(0,(p*(b+1)-1)/b) if b>0 else 0

# --- TEAM RATINGS (toy model) ---
RATINGS = {
    "Chiefs":88,"Ravens":86,"Bills":84,"Dolphins":83,
    "49ers":90,"Cowboys":85,"Eagles":86,"Packers":82,
    "Lakers":84,"Warriors":85,"Celtics":90,"Nuggets":89,
    "Dodgers":90,"Braves":89,"Yankees":86,"Astros":87
}

# --- SAMPLE DATA ---
rows = [
    ("NFL","2025-10-01T23:20:00Z","Chiefs","Ravens","DraftKings","Moneyline","Chiefs",-135),
    ("NFL","2025-10-01T23:20:00Z","Chiefs","Ravens","FanDuel","Moneyline","Ravens",120),
    ("NBA","2025-10-02T00:10:00Z","Celtics","Lakers","DraftKings","Moneyline","Celtics",-150),
    ("NBA","2025-10-02T00:10:00Z","Celtics","Lakers","Caesars","Moneyline","Lakers",135),
]
df = pd.DataFrame(rows, columns=["sport","time","home","away","book","market","pick","odds"])
df["time"] = pd.to_datetime(df["time"], utc=True)

# --- CONTROLS ---
min_edge = st.slider("Minimum Edge %",0.0,10.0,2.0,0.5)/100
bankroll = st.number_input("Bankroll ($)",min_value=50.0,value=1000.0,step=50.0)
kelly_cap = st.slider("Kelly Cap",0.0,1.0,0.25,0.05)

# --- CALCULATIONS ---
out=[]
for _,r in df.iterrows():
    dec = american_to_decimal(r["odds"])
    ip  = implied_prob(r["odds"])
    hp,ap = model_prob(RATINGS.get(r["home"],80),RATINGS.get(r["away"],80))
    mp = hp if r["pick"]==r["home"] else ap
    edge = dec*mp-1
    kelly = kelly_fraction(mp,dec)
    stake = round(min(bankroll,kelly*kelly_cap*bankroll),2)

    out.append({
        "Sport":r["sport"],
        "Game Time":r["time"].strftime("%Y-%m-%d %H:%M"),
        "Matchup":f"{r['away']} @ {r['home']}",
        "Pick":r["pick"],
        "Sportsbook":r["book"],
        "Odds":r["odds"],
        "Implied %":f"{round(ip*100,1)}%",
        "Model %":f"{round(mp*100,1)}%",
        "Edge %":f"{round(edge*100,2)}%",
        "Kelly Stake $":stake
    })
picks=pd.DataFrame(out)
picks=picks[picks["Edge %"].str.rstrip("%").astype(float)>=min_edge*100]

# --- DISPLAY ---
st.subheader("Picks")
if picks.empty:
    st.info("No picks met your filter. Adjust settings.")
else:
    st.dataframe(picks,use_container_width=True,hide_index=True)
    st.download_button("Download CSV",picks.to_csv(index=False).encode("utf-8"),"ai_picks.csv","text/csv")

st.markdown("---")
st.caption("© 2025 TruLine Betting – Prototype AI picker")
