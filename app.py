import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Ulaanbaatar Air Quality Advisor", page_icon="🌫️", layout="wide")

CATS = ["Good", "Moderate", "Unhealthy for sensitive", "Unhealthy", "Very unhealthy", "Hazardous"]
CAT_COLORS = ["#2e9e4f", "#e6c200", "#f28c28", "#d62728", "#8e44ad", "#7b1e1e"]
PM_BINS = [-1, 9.0, 35.4, 55.4, 125.4, 225.4, 1e6]
WHO_DAILY = 15
HEATING_MONTHS = [10, 11, 12, 1, 2, 3, 4]

ADVICE = {
    "Good": "Great air. Go outside normally.",
    "Moderate": "Fine for most people. People with asthma or heart problems should avoid long, hard exercise outside.",
    "Unhealthy for sensitive": "Sensitive people should stay inside. Everyone else should shorten hard exercise outside. "
                               "Wear an N95/KN95 mask if you will be outside for a long time.",
    "Unhealthy": "Avoid exercise outside. Wear an N95/KN95 mask when you go out. "
                 "Use an air purifier indoors if you have one.",
    "Very unhealthy": "Stay inside as much as possible with the windows closed. Wear an N95/KN95 mask if you must go out.",
    "Hazardous": "Stay inside. Wear an N95/KN95 mask if you must go out. Take extra care of children and older people.",
}


def category_of(pm):
    return pd.cut([pm], PM_BINS, labels=CATS)[0]


def advise(pm, activity):
    cat = str(category_of(pm))
    used = cat
    # exercise and children breathe more air for their size, so we use one level stricter advice
    if activity in ("Running / exercise", "Kids playing outside"):
        used = CATS[min(CATS.index(cat) + 1, len(CATS) - 1)]
    return cat, used, ADVICE[used]


# ---------------------------------------------------------------
# 1. LOAD, CLEAN AND MERGE THE DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    air = pd.read_csv("ub_air.csv")
    weather = pd.read_csv("ub_weather.csv")
    air["time"] = pd.to_datetime(air["time"])
    weather["time"] = pd.to_datetime(weather["time"])

    air = air.drop_duplicates(subset="time").sort_values("time")
    weather = weather.drop_duplicates(subset="time").sort_values("time")

    for c in ["pm10", "pm2_5", "nitrogen_dioxide"]:
        air.loc[air[c] < 0, c] = np.nan
    weather.loc[weather["relative_humidity_2m"] > 100, "relative_humidity_2m"] = np.nan
    weather.loc[weather["wind_speed_10m"] < 0, "wind_speed_10m"] = np.nan
    wcols = [c for c in weather.columns if c != "time"]
    weather[wcols] = weather[wcols].interpolate(limit=3)

    data = air.merge(weather, on="time", how="inner")
    data = data.rename(columns={"pm2_5": "pm25", "temperature_2m": "temp_c", "relative_humidity_2m": "humidity",
                                "wind_speed_10m": "wind_kmh", "nitrogen_dioxide": "no2"})
    data = data.dropna(subset=["pm25"]).reset_index(drop=True)

    data["hour"] = data["time"].dt.hour
    data["month"] = data["time"].dt.month
    data["year"] = data["time"].dt.year
    data["heating"] = data["month"].isin(HEATING_MONTHS)
    data["wind_band"] = pd.cut(data["wind_kmh"], [-1, 5, 10, 20, 500],
                               labels=["below 5", "5-10", "10-20", "above 20"])
    return data


def make_daily(data):
    daily = data.set_index("time").resample("D").agg(
        pm25=("pm25", "mean"), temp_c=("temp_c", "mean"), wind_kmh=("wind_kmh", "mean"),
        humidity=("humidity", "mean"), hours=("pm25", "count"))
    daily = daily[daily["hours"] >= 18].copy()
    daily["month"] = daily.index.month
    daily["year"] = daily.index.year
    daily["heating"] = daily["month"].isin(HEATING_MONTHS)
    daily["category"] = pd.cut(daily["pm25"], PM_BINS, labels=CATS)
    return daily


FEATURES = ["temp_c", "wind_kmh", "humidity", "precipitation", "is_heating", "hour_sin", "hour_cos"]


def add_features(m):
    m = m.copy()
    m["precipitation"] = m["precipitation"].fillna(0)
    m["is_heating"] = m["heating"].astype(int)
    m["hour_sin"] = np.sin(2 * np.pi * m["hour"] / 24)
    m["hour_cos"] = np.cos(2 * np.pi * m["hour"] / 24)
    return m


@st.cache_resource
def train_models(data):
    m = add_features(data.dropna(subset=["temp_c", "wind_kmh", "humidity"]).sort_values("time"))
    split = int(len(m) * 0.75)
    train, test = m.iloc[:split], m.iloc[split:]
    y_train = np.log1p(train["pm25"])
    lin = LinearRegression().fit(train[FEATURES], y_train)
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5,
                               random_state=42, n_jobs=-1).fit(train[FEATURES], y_train)
    preds = {
        "Baseline (average)": np.full(len(test), train["pm25"].mean()),
        "Linear regression": np.expm1(lin.predict(test[FEATURES])),
        "Random forest": np.expm1(rf.predict(test[FEATURES])),
    }
    table = pd.DataFrame([{"Model": k,
                           "Average error (µg/m³)": round(mean_absolute_error(test["pm25"], p), 1),
                           "R²": round(r2_score(test["pm25"], p), 2)} for k, p in preds.items()])
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    period = (str(train["time"].min().date()), str(train["time"].max().date()),
              str(test["time"].min().date()), str(test["time"].max().date()))
    return rf, table, imp, period


if not (os.path.exists("ub_air.csv") and os.path.exists("ub_weather.csv")):
    st.error("The data files `ub_air.csv` and `ub_weather.csv` are missing. "
             "Upload both files to the same GitHub folder as app.py.")
    st.stop()

data = load_data()
daily_all = make_daily(data)

# ---------------------------------------------------------------
# 2. SIDEBAR
# ---------------------------------------------------------------
st.sidebar.title("🌫️ Settings")
dmin, dmax = data["time"].min().date(), data["time"].max().date()
date_range = st.sidebar.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
season = st.sidebar.radio("Season", ["All year", "Heating season (Oct-Apr)", "Rest of the year (May-Sep)"])
activity = st.sidebar.selectbox("Your activity (for advice)",
                                ["Walking", "Running / exercise", "Kids playing outside"])
st.sidebar.caption("Air quality values are model estimates (CAMS), not readings from a local sensor.")

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Pick both a start date and an end date in the sidebar.")
    st.stop()

start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
f = data[(data["time"] >= start) & (data["time"] < end)]
if season.startswith("Heating"):
    f = f[f["heating"]]
elif season.startswith("Rest"):
    f = f[~f["heating"]]
fd = make_daily(f) if len(f) else pd.DataFrame()

st.title("🌫️ Ulaanbaatar Air Quality Advisor")
st.markdown("**Question:** When is Ulaanbaatar's air worst, how much do cold weather, wind and the heating season "
            "matter, and should I go outside or wear a mask today?")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Today's advice", "Patterns", "Weather effect",
                                        "Model and simulator", "About the data"])

# ---------------------------------------------------------------
# TAB 1: TODAY
# ---------------------------------------------------------------
with tab1:
    last_time = data["time"].max()
    latest = data[data["time"] > last_time - pd.Timedelta(hours=24)]
    pm_now = latest["pm25"].mean()
    cat, used, text = advise(pm_now, activity)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PM2.5, last 24 hours", f"{pm_now:.0f} µg/m³")
    c2.metric("Times the WHO guideline", f"{pm_now / WHO_DAILY:.1f}x", help="WHO 24-hour guideline: 15 µg/m³")
    c3.metric("Category", cat)
    c4.metric("Latest data", last_time.strftime("%d %b %Y"))

    if CATS.index(used) <= 1:
        st.success(f"**{activity}:** {text}")
    elif CATS.index(used) == 2:
        st.warning(f"**{activity}:** {text}")
    else:
        st.error(f"**{activity}:** {text}")
    st.caption("General guidance, not medical advice. Exercise and children use one level stricter advice.")

    fig = px.line(latest, x="time", y="pm25", title="PM2.5 in the last 24 hours of the data",
                  labels={"time": "Hour", "pm25": "PM2.5 (µg/m³)"})
    fig.add_hline(y=WHO_DAILY, line_dash="dash", line_color="green", annotation_text="WHO guideline (15)")
    st.plotly_chart(fig, width="stretch")
    st.caption("The dashed line is the WHO guideline for a 24-hour average. "
               "The latest hours can be a few days old because weather data arrives with a delay.")

# ---------------------------------------------------------------
# TAB 2: PATTERNS
# ---------------------------------------------------------------
with tab2:
    if f.empty or fd.empty:
        st.warning("No data for these settings. Widen the date range or change the season.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Days in selection", f"{len(fd):,}")
        k2.metric("Average PM2.5", f"{fd['pm25'].mean():.0f} µg/m³")
        k3.metric("Days above WHO guideline", f"{(fd['pm25'] > WHO_DAILY).mean() * 100:.0f}%")
        k4.metric("Days Unhealthy or worse", f"{(fd['pm25'] > 55.4).mean() * 100:.0f}%")

        pivot = f.pivot_table(index="month", columns="hour", values="pm25", aggfunc="mean")
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                        labels={"x": "Hour of the day", "y": "Month", "color": "PM2.5"},
                        title="Average PM2.5 (µg/m³) by month and hour")
        st.plotly_chart(fig, width="stretch")
        st.caption("The darkest cells show the worst times: look for the winter months and the evening and night hours.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fd.index, y=fd["pm25"], mode="lines", name="Daily average",
                                 line=dict(color="lightgray", width=1)))
        fig.add_trace(go.Scatter(x=fd.index, y=fd["pm25"].rolling(7, min_periods=3).mean(), mode="lines",
                                 name="7-day average", line=dict(color="firebrick", width=2)))
        if not season.startswith("Rest"):
            for y in range(fd.index.min().year - 1, fd.index.max().year + 1):
                fig.add_vrect(x0=pd.Timestamp(y, 10, 1), x1=pd.Timestamp(y + 1, 4, 30),
                              fillcolor="steelblue", opacity=0.10, line_width=0)
        fig.add_hline(y=WHO_DAILY, line_dash="dash", line_color="green")
        fig.update_xaxes(range=[fd.index.min(), fd.index.max()])
        fig.update_layout(title="Daily PM2.5 (blue = heating season)", yaxis_title="PM2.5 (µg/m³)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Pollution rises every winter and drops in summer. The green line is the WHO guideline.")

        left, right = st.columns(2)
        share = (pd.crosstab(fd["month"], fd["category"], normalize="index")
                 .reindex(columns=CATS, fill_value=0) * 100).reset_index().melt(
            id_vars="month", var_name="Category", value_name="% of days")
        fig = px.bar(share, x="month", y="% of days", color="Category", color_discrete_sequence=CAT_COLORS,
                     category_orders={"Category": CATS}, title="Share of days in each category, by month")
        left.plotly_chart(fig, width="stretch")
        left.caption("Shows how many days in each month are Good, Unhealthy or worse.")

        hs = fd[fd["heating"]].copy()
        hs["season_start"] = np.where(hs["month"] >= 10, hs["year"], hs["year"] - 1)
        by_season = hs.groupby("season_start").agg(pm25=("pm25", "mean"), days=("pm25", "count"))
        by_season = by_season[by_season["days"] >= 90].reset_index()
        if by_season.empty:
            right.info("No full heating season in the selected dates.")
        else:
            by_season["Season"] = by_season["season_start"].astype(int).astype(str) + "-" + \
                (by_season["season_start"].astype(int) + 1).astype(str).str[2:]
            fig = px.bar(by_season, x="Season", y="pm25", title="Average PM2.5 in each full heating season",
                         labels={"pm25": "PM2.5 (µg/m³)"})
            fig.add_hline(y=WHO_DAILY, line_dash="dash", line_color="green")
            right.plotly_chart(fig, width="stretch")
            right.caption("Compares full heating seasons (October to April) to see if the air is improving.")

# ---------------------------------------------------------------
# TAB 3: WEATHER EFFECT
# ---------------------------------------------------------------
with tab3:
    if f.empty or fd.empty:
        st.warning("No data for these settings. Widen the date range or change the season.")
    else:
        d = fd.dropna(subset=["temp_c", "pm25"])
        slope, intercept = np.polyfit(d["temp_c"], d["pm25"], 1)
        xs = np.linspace(d["temp_c"].min(), d["temp_c"].max(), 50)
        fig = px.scatter(d.reset_index(), x="temp_c", y="pm25", color="heating", opacity=0.55,
                         hover_data={"time": True} if "time" in d.reset_index().columns else None,
                         labels={"temp_c": "Average temperature (°C)", "pm25": "Average PM2.5 (µg/m³)",
                                 "heating": "Heating season"},
                         title="Temperature vs PM2.5 (each dot is one day)")
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Trend line",
                                 line=dict(color="black")))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Correlation: {d['temp_c'].corr(d['pm25']):.2f}. "
                   f"On average, each 10 °C colder adds about {-slope * 10:.0f} µg/m³ of PM2.5.")

        l, r = st.columns(2)
        fig = px.box(f.dropna(subset=["wind_band"]), x="wind_band", y="pm25", points=False,
                     category_orders={"wind_band": ["below 5", "5-10", "10-20", "above 20"]},
                     labels={"wind_band": "Wind speed (km/h)", "pm25": "PM2.5 (µg/m³)"},
                     title="Hourly PM2.5 by wind speed")
        l.plotly_chart(fig, width="stretch")
        l.caption("Calm air traps pollution near the ground. Stronger wind usually lowers PM2.5.")

        by_hour = f.groupby(["hour", "heating"])["pm25"].mean().reset_index()
        by_hour["Period"] = by_hour["heating"].map({True: "Heating season", False: "Rest of the year"})
        fig = px.line(by_hour, x="hour", y="pm25", color="Period", markers=True,
                      labels={"hour": "Hour of the day", "pm25": "PM2.5 (µg/m³)"},
                      title="Average PM2.5 by hour of the day")
        r.plotly_chart(fig, width="stretch")
        r.caption("Shows the daily rhythm: when in the day the air is cleanest and dirtiest.")

# ---------------------------------------------------------------
# TAB 4: MODEL
# ---------------------------------------------------------------
with tab4:
    rf, table, imp, period = train_models(data)
    st.markdown("A model predicts PM2.5 from temperature, wind, humidity, rain, the heating season and the hour. "
                f"It was trained on {period[0]} to {period[1]} and tested on {period[2]} to {period[3]}.")
    st.dataframe(table, width="stretch", hide_index=True)
    fig = px.bar(imp.reset_index().rename(columns={"index": "Feature", 0: "Importance"}),
                 x="Importance", y="Feature", orientation="h", title="What matters most for PM2.5?")
    st.plotly_chart(fig, width="stretch")
    st.caption("Higher importance means the random forest relies more on this feature.")

    st.subheader("What if the weather is like this?")
    a, b, c, d_ = st.columns(4)
    t = a.slider("Temperature (°C)", -40, 30, -25)
    w = b.slider("Wind speed (km/h)", 0, 40, 2)
    hmd = c.slider("Humidity (%)", 0, 100, 70)
    hr = d_.slider("Hour of the day", 0, 23, 8)
    heating = st.checkbox("Heating season (Oct-Apr)", value=True)
    row = pd.DataFrame([{"temp_c": t, "wind_kmh": w, "humidity": hmd, "precipitation": 0,
                         "is_heating": int(heating),
                         "hour_sin": np.sin(2 * np.pi * hr / 24), "hour_cos": np.cos(2 * np.pi * hr / 24)}])
    pm_pred = float(np.expm1(rf.predict(row[FEATURES]))[0])
    cat2, used2, text2 = advise(pm_pred, activity)
    s1, s2, s3 = st.columns(3)
    s1.metric("Predicted PM2.5", f"{pm_pred:.0f} µg/m³")
    s2.metric("Times the WHO guideline", f"{pm_pred / WHO_DAILY:.1f}x")
    s3.metric("Category", cat2)
    st.info(f"**{activity}:** {text2}")

# ---------------------------------------------------------------
# TAB 5: ABOUT
# ---------------------------------------------------------------
with tab5:
    st.markdown(f"""
### About the data
- **Air quality:** Open-Meteo Air Quality API (hourly PM2.5, PM10 and NO₂), based on CAMS (Copernicus Atmosphere Monitoring Service) model data. https://open-meteo.com/en/docs/air-quality-api
- **Weather:** Open-Meteo Historical Weather API (hourly temperature, humidity, wind, rain and snow), based on ERA5 reanalysis. https://open-meteo.com/en/docs/historical-weather-api
- **Location:** Ulaanbaatar (47.92 N, 106.92 E), time zone Asia/Ulaanbaatar.
- **Period in this app:** {dmin} to {dmax}, {len(data):,} hourly rows.
- **License:** free for non-commercial use under CC BY 4.0. Credit: Open-Meteo.

### Cleaning steps
1. Parsed the dates and removed duplicate hours.
2. Turned impossible values (negative concentrations, humidity above 100) into missing values.
3. Filled short weather gaps (up to 3 hours).
4. Merged the air quality and weather tables on the hour.
5. Added hour, month, year, heating season (October to April) and wind speed bands.
6. Built a daily table, keeping days with at least 18 hours of data, and assigned a category using the US EPA PM2.5 scale.

### Limits
- Air quality values are model estimates for a large grid cell, not readings from a sensor in one district, so they can be lower than real readings in the ger districts.
- The heating season is assumed to be October to April.
- The model uses weather only. It does not know about traffic or fuel use, so it cannot predict every spike.
- The health advice is general and is not medical advice.
""")
    st.dataframe(data.head(50), width="stretch")
