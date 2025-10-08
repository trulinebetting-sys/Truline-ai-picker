@st.cache_data(ttl=60)
def fetch_odds(sport_key: str, regions: str, markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY. Add it to `.env` or Streamlit Secrets.")
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
    for event in data:
        dt = event.get("commence_time", "")
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        for bm in event.get("bookmakers", []):
            book = bm.get("title", "")
            for mk in bm.get("markets", []):
                market_type = mk.get("key", "")
                for outcome in mk.get("outcomes", []):
                    rows.append({
                        "Date/Time": dt,
                        "Home": home,
                        "Away": away,
                        "Sportsbook": book,
                        "Market": market_type,
                        "Pick": outcome.get("name"),
                        "Odds": outcome.get("price")
                    })
    return pd.DataFrame(rows)

# ----------------------------
# Main content
# ----------------------------
sport_key = SPORT_OPTIONS[sport_name]
if fetch:
    df = fetch_odds(sport_key=sport_key, regions=regions)
    if df.empty:
        st.warning("No data returned. Try another sport or check API quota.")
    else:
        # Format datetime
        df["Date/Time"] = pd.to_datetime(df["Date/Time"])
        df["Date/Time"] = df["Date/Time"].dt.strftime("%b %d, %I:%M %p ET")

        # Tabs
        tabs = st.tabs(["Moneylines", "Spreads", "Totals", "Raw Data"])

        # --- Moneylines
        with tabs[0]:
            st.subheader("Moneyline Picks")
            ml = df[df["Market"] == "h2h"]
            if ml.empty:
                st.info("No moneyline data.")
            else:
                st.dataframe(ml.head(20), use_container_width=True)

        # --- Spreads
        with tabs[1]:
            st.subheader("Spread Picks")
            spreads = df[df["Market"] == "spreads"]
            if spreads.empty:
                st.info("No spreads data.")
            else:
                st.dataframe(spreads.head(20), use_container_width=True)

        # --- Totals
        with tabs[2]:
            st.subheader("Over/Under Picks")
            totals = df[df["Market"] == "totals"]
            if totals.empty:
                st.info("No totals data.")
            else:
                st.dataframe(totals.head(20), use_container_width=True)

        # --- Raw
        with tabs[3]:
            st.subheader("Raw Odds Data")
            st.dataframe(df.head(50), use_container_width=True)
