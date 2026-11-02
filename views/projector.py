import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dcairbnb.data import load
from dcairbnb.model import HOUSEHOLD_SIZES, PROJECTOR_INPUTS, load_artifacts, project_all, project_revenue
from dcairbnb.style import (AXIS, BAND, DC_CENTER, FADED, GRID, INK, INK_2, MUTED, SEQ_BLUE, SERIES, SURFACE, chapter,
                            headline, join_and, md_safe, section, show_chart, short_name, title_case)

b = load()
art = load_artifacts()
YEARS = list(art["demand_index"].index)
BASE_YEAR = art["base_year"]
size_label = lambda h: f"{h}+" if h >= 8 else str(h)
a_size = lambda h: ("an " if h == 8 else "a ") + size_label(h)   # "an 8+-guest", "a 4-guest"
money_k = lambda v: f"${v / 1000:,.1f}k" if v < 100_000 else f"${v / 1000:,.0f}k"


@st.cache_data(show_spinner=False)
def year_table(year):
    return project_all(load_artifacts(), year)



chapter("The Application · Revenue Projector", "What Could a Short-Term Rental Earn?",
        "Choose how many guests a place sleeps, where it is and which year to estimate a year's revenue for a "
        "short-term rental (stays under 30 nights), with an honest range from a model trained on "
        f"{art['n_train']:,} active DC listings.")

areas = sorted(b.neighbourhoods, key=short_name)
with st.container(border=True):
    i1, i2, i3 = st.columns([2, 5, 2], gap="medium")
    household = i1.selectbox("Household size", HOUSEHOLD_SIZES, index=HOUSEHOLD_SIZES.index(4), key="p_household",
                             format_func=lambda h: f"{size_label(h)} guest" + ("" if h == 1 else "s"))
    area = i2.selectbox("Neighborhood", areas, index=areas.index("Capitol Hill, Lincoln Park"), key="p_area",
                        format_func=lambda n: n if len(n) < 60 else n[:57] + "…")
    year = i3.selectbox("Year", YEARS, index=YEARS.index(BASE_YEAR + 1), key="p_year",
                        format_func=lambda y: f"{y} (now)" if y == BASE_YEAR else str(y))

res = project_revenue(art, household, area, year)
year_grid = year_table(year)
same_size = year_grid[year_grid["household"] == household].copy()
same_size["area"] = same_size["neighbourhood"].map(short_name)
same_size["rank"] = same_size["typical"].rank(ascending=False).astype(int)
rank = int(same_size.loc[same_size["neighbourhood"] == area, "rank"].iloc[0])
city_typical = np.average(same_size["typical"], weights=same_size["listings_like_this"].clip(lower=0.01))
vs_city = res["typical"] / city_typical - 1
n_local = res["listings_like_this"]
confidence = ("High", "green", ":material/check_circle:") if n_local >= 30 else \
             ("Medium", "orange", ":material/info:") if n_local >= 10 else ("Low", "red", ":material/warning:")


hero, strip = st.columns([1, 1], gap="large", vertical_alignment="center")
with hero:
    st.html(f'<div class="kicker kicker-plain">Projected Revenue · {year} · {size_label(household)}-Guest Listing in {short_name(area)}</div>'
            f'<div class="hero-number">${res["typical"]:,.0f}<span style="font-size:1.4rem;color:{INK_2}"> a year</span></div>'
            f'<div class="hero-range">80% of similar listings would earn between <b>${res["low"]:,.0f}</b> and '
            f'<b>${res["high"]:,.0f}</b>. That\'s about ${res["typical"] / 12:,.0f} a month, in 2026 dollars.</div>')
with strip:
    # Range strip: the 80% range, the typical value, and the DC-wide typical for this size
    fig = go.Figure()
    fig.add_shape(type="line", x0=res["low"], x1=res["high"], y0=0, y1=0, line=dict(color=BAND.replace("0.18", "0.45"), width=14))
    fig.add_scatter(x=[city_typical], y=[0], mode="markers", name="DC typical for this size",
                    marker=dict(symbol="line-ns", size=26, line=dict(width=3, color=INK_2)),
                    hovertemplate="DC-wide typical for this size: $%{x:,.0f}<extra></extra>")
    fig.add_scatter(x=[res["typical"]], y=[0], mode="markers", name="This projection",
                    marker=dict(size=18, color=SERIES[0], line=dict(width=3, color=SURFACE)),
                    hovertemplate="This projection: $%{x:,.0f}<extra></extra>")
    fig.update_layout(height=120, margin=dict(t=10, b=30, l=10, r=10), showlegend=True,
                      legend=dict(y=1.25, x=0, font=dict(size=12)),
                      yaxis=dict(visible=False, range=[-1, 1]),
                      xaxis=dict(tickprefix="$", tickformat="~s", showgrid=False, linecolor=AXIS,
                                 range=[0, max(res["high"], city_typical) * 1.08]))
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False}, key="range_strip")
    st.caption(md_safe(f"**Chart description:** A bar spans the 80% range from ${res['low']:,.0f} to ${res['high']:,.0f}, "
                       f"with a blue dot marking the typical projection (${res['typical']:,.0f}) and a gray tick marking the "
                       f"DC-wide typical for {a_size(household)}-guest listing (${city_typical:,.0f})."))
s1, s2, s3, s4 = st.columns(4)
s1.metric("Compared with DC Typical", f"{vs_city:+.0%}", f"DC typical: ${city_typical:,.0f}", border=True,
          delta_color="off", height="stretch")
s2.metric("Neighborhood Rank", f"#{rank} of {len(same_size)}", "For this household size", border=True, delta_color="off",
          height="stretch")
s3.metric("Demand vs. Today", f"{res['demand_index']:.2f}×",
          "Base year" if year == BASE_YEAR else f"{year} forecast", border=True, delta_color="off", height="stretch")
s4.metric("Listings Like This Here", f"{n_local}", border=True, height="stretch",
          help="Real listings of this size in the area. Below 10, the estimate leans on patterns from elsewhere in DC.")
with s4:
    st.badge(f"{confidence[0]} Confidence", color=confidence[1], icon=confidence[2])

section("Explore the projection")
left, right = st.columns(2, gap="large")
with left:
    proj = pd.DataFrame([project_revenue(art, household, area, y) for y in YEARS])
    last = proj.iloc[-1]
    headline("The Typical Value Holds Steady, but the Range Widens",
             f"Projected revenue by year for this listing, with the range spanning {money_k(last['low'])}–{money_k(last['high'])} by {YEARS[-1]}.")
    fig = go.Figure()
    fig.add_scatter(x=proj["year"], y=proj["high"], mode="lines", line_width=0, showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=proj["year"], y=proj["low"], mode="lines", line_width=0, fill="tonexty", fillcolor=BAND,
                    name="80% range", hoverinfo="skip")
    fig.add_scatter(x=proj["year"], y=proj["typical"], mode="lines+markers", name="Typical",
                    line=dict(color=SERIES[0], width=2),
                    marker=dict(size=[16 if y == year else 9 for y in proj["year"]], color=SERIES[0], line=dict(width=2, color=SURFACE)),
                    customdata=proj[["low", "high", "demand_index"]],
                    hovertemplate="<b>%{x}</b>: $%{y:,.0f} typical<br>80% range: $%{customdata[0]:,.0f}–$%{customdata[1]:,.0f}"
                                  "<br>Demand index: %{customdata[2]:.2f}<extra></extra>")
    fig.update_layout(xaxis=dict(dtick=1, title=None), hovermode="x",
                      yaxis=dict(title="Annual revenue (2026 $)", tickprefix="$", tickformat="~s", rangemode="tozero"))
    show_chart(fig, height=380, key="proj_years", data=proj[["year", "low", "typical", "high", "demand_index"]].round(2),
               alt=f"Line chart of projected revenue from {YEARS[0]} to {YEARS[-1]}, where the typical value stays near "
                   f"${proj['typical'].mean():,.0f}, while the shaded 80% range widens from "
                   f"${proj['low'].iloc[0]:,.0f}–${proj['high'].iloc[0]:,.0f} to ${last['low']:,.0f}–${last['high']:,.0f}.")
with right:
    sizes = year_grid[year_grid["neighbourhood"] == area].sort_values("household")
    ratio = sizes["typical"].iloc[-1] / sizes["typical"].iloc[0]
    headline(f"In {short_name(area)}, a Listing for 8+ Earns {ratio:.1f}× One for a Single Guest",
             f"Typical projected revenue in {year} by household size, with lines showing the 80% range.")
    fig = go.Figure(go.Bar(
        x=sizes["household"].map(size_label), y=sizes["typical"],
        marker=dict(color=[SERIES[0] if h == household else FADED for h in sizes["household"]], cornerradius=4),
        error_y=dict(type="data", symmetric=False, array=sizes["high"] - sizes["typical"],
                     arrayminus=sizes["typical"] - sizes["low"], color=MUTED, thickness=1.2, width=4),
        customdata=sizes[["low", "high", "listings_like_this"]],
        hovertemplate="Sleeps %{x}: $%{y:,.0f} typical<br>80% range: $%{customdata[0]:,.0f}–$%{customdata[1]:,.0f}"
                      "<br>Listings like this here: %{customdata[2]}<extra></extra>"))
    fig.update_layout(xaxis=dict(title="Guests it sleeps", type="category"),
                      yaxis=dict(title="Annual revenue (2026 $)", tickprefix="$", tickformat="~s", rangemode="tozero"))
    show_chart(fig, height=380, key="proj_sizes", data=sizes[["household", "low", "typical", "high", "listings_like_this"]].round(0),
               alt=f"Bar chart of typical revenue in {short_name(area)} by household size, from ${sizes['typical'].iloc[0]:,.0f} "
                   f"for 1 guest to ${sizes['typical'].iloc[-1]:,.0f} for 8+, with the selected size ({size_label(household)}) highlighted.")

best, worst = same_size.nlargest(1, "typical").iloc[0], same_size.nsmallest(1, "typical").iloc[0]
headline(f"The Same {size_label(household)}-Guest Listing Would Earn the Most in {best['area']} ({money_k(best['typical'])}) "
         f"and the Least in {worst['area']} ({money_k(worst['typical'])})",
         f"Typical projected revenue in {year} for every neighborhood, with {short_name(area)} outlined.")
map_tab, rank_tab = st.tabs(["Map", "Ranking"])
with map_tab:
    fig = go.Figure(go.Choroplethmap(
        geojson=b.geo, featureidkey="properties.neighbourhood", locations=same_size["neighbourhood"], z=same_size["typical"],
        colorscale=SEQ_BLUE, marker=dict(opacity=0.85, line=dict(width=1, color="white")),
        colorbar=dict(title=dict(text="Typical revenue", font=dict(color=INK_2)), tickprefix="$", tickformat="~s",
                      thickness=12, outlinewidth=0),
        customdata=np.stack([same_size["neighbourhood"], same_size["low"], same_size["high"],
                             same_size["listings_like_this"], same_size["rank"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Typical: $%{z:,.0f} a year<br>80% range: $%{customdata[1]:,.0f}–$%{customdata[2]:,.0f}"
                      "<br>Rank: #%{customdata[4]}<br>Listings like this here: %{customdata[3]}<extra></extra>"))
    sel = same_size[same_size["neighbourhood"] == area]
    fig.add_trace(go.Choroplethmap(geojson=b.geo, featureidkey="properties.neighbourhood", locations=sel["neighbourhood"],
                                   z=[1], colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False,
                                   marker=dict(line=dict(width=4, color=INK)), hoverinfo="skip"))
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), map=dict(style="carto-positron", center=DC_CENTER, zoom=10.4))
    show_chart(fig, height=560, key="proj_map",
               alt=f"Map of typical projected revenue in {year} for {a_size(household)}-guest listing in each neighborhood, "
                   f"highest in {join_and(same_size.nlargest(3, 'typical')['area'])} and lowest in "
                   f"{join_and(same_size.nsmallest(3, 'typical')['area'])}, with {short_name(area)} outlined at rank #{rank}.")
with rank_tab:
    ranked = same_size.sort_values("typical")
    is_sel = ranked["neighbourhood"] == area
    fig = go.Figure()
    for lo, hi, a, s in zip(ranked["low"], ranked["high"], ranked["area"], is_sel):
        fig.add_shape(type="line", x0=lo, x1=hi, y0=a, y1=a, line=dict(color=BAND.replace("0.18", "0.5") if s else GRID, width=6))
    fig.add_scatter(x=ranked["typical"], y=ranked["area"], mode="markers",
                    marker=dict(size=[13 if s else 9 for s in is_sel], color=[SERIES[0] if s else MUTED for s in is_sel],
                                line=dict(width=2, color=SURFACE)),
                    customdata=ranked[["low", "high", "listings_like_this", "neighbourhood"]],
                    hovertemplate="<b>%{customdata[3]}</b><br>Typical: $%{x:,.0f}<br>80% range: $%{customdata[0]:,.0f}–$%{customdata[1]:,.0f}"
                                  "<br>Listings like this here: %{customdata[2]}<extra></extra>")
    fig.update_layout(xaxis=dict(title="Annual revenue (2026 $): dot = typical, bar = 80% range", tickprefix="$", tickformat="~s"),
                      yaxis=dict(title=None, tickfont=dict(size=11)), showlegend=False, margin=dict(t=10))
    show_chart(fig, height=900, key="proj_rank",
               data=ranked[["neighbourhood", "low", "typical", "high", "listings_like_this"]].iloc[::-1].round(0),
               alt=f"Dot plot of all {len(ranked)} neighborhoods ranked by typical projected revenue, with bars for the 80% range "
                   f"and {short_name(area)} highlighted at rank #{rank}.")

section("Can you trust it?")
st.markdown(
    "The projector combines **two models**. A *listing model* (gradient-boosted trees) estimates what a listing like this "
    "earns today. A *demand model* (a Holt-Winters time-series forecast) estimates how bookings per listing will change "
    "each year. To fill in everything else about the listing, the projector takes **real DC listings of the chosen size**, "
    "moves them into the chosen neighborhood, and reports their median prediction.")
acc_tab, drv_tab, dem_tab = st.tabs(["Accuracy", "What Drives Revenue", "Demand Forecast"])

with acc_tab:
    scores = art["scores"]
    best_m = scores.iloc[-1]
    a1, a2, a3 = st.columns(3)
    a1.metric("Typical Error per Listing", f"${best_m['median_error']:,.0f}",
              f"{best_m['median_error'] / scores.iloc[1]['median_error'] - 1:+.0%} vs. linear regression", border=True,
              delta_color="inverse")
    a2.metric("Variation Explained (R², Log Revenue)", f"{best_m['r2_log']:.2f}", f"Linear regression: {scores.iloc[1]['r2_log']:.2f}",
              border=True, delta_color="off")
    a3.metric("80% Range Really Covers", f"{art['coverage']['calibrated']:.0%}",
              f"Before calibration: {art['coverage']['raw']:.0%}", border=True, delta_color="off")
    headline("Using the Full Listing Profile Cuts the Typical Error from "
             f"{money_k(scores.iloc[1]['median_error'])} to {money_k(best_m['median_error'])}",
             "Median absolute error per listing from 5-fold cross-validation, where each listing is predicted by a model that never saw it.")
    fig = px.bar(scores, x="median_error", y="model", orientation="h", text=scores["median_error"].map("${:,.0f}".format),
                 custom_data=["r2_log", "within_25pct"])
    fig.update_traces(marker_color=[MUTED] * (len(scores) - 1) + [SERIES[0]], textposition="outside", cliponaxis=False,
                      textfont_color=INK_2,
                      hovertemplate="<b>%{y}</b><br>Median error: $%{x:,.0f}<br>R²: %{customdata[0]:.2f}"
                                    "<br>Within ±25%: %{customdata[1]:.0%}<extra></extra>")
    fig.update_layout(xaxis=dict(title="Median absolute error", tickprefix="$"), yaxis=dict(title=None, autorange="reversed"))
    show_chart(fig, height=300, key="trust_scores", data=scores.round(3),
               alt="Bar chart comparing four models by median error: " +
                   "; ".join(f"{r.model} ${r.median_error:,.0f}" for r in scores.itertuples()) + ".")

with drv_tab:
    imp = art["importance"].copy()
    imp["input"] = np.where(imp["feature"].isin(PROJECTOR_INPUTS), "Projector input", "Filled in from comparables")
    top = imp.nlargest(3, "importance")["feature"].tolist()
    headline(title_case(f"{top[0]}, {top[1]} and {top[2]} drive revenue the most"),
             "How much accuracy drops when each feature is shuffled, though size still matters through bedrooms and capacity together.")
    fig = px.bar(imp, x="importance", y="feature", color="input", orientation="h", error_x="sd",
                 color_discrete_map={"Projector input": SERIES[0], "Filled in from comparables": MUTED})
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Drop in R² when shuffled: %{x:.3f}<extra></extra>", error_x_color=AXIS)
    fig.update_layout(xaxis_title="Drop in R² when the feature is shuffled", yaxis_title=None)
    show_chart(fig, height=480, key="trust_imp", data=imp.sort_values("importance", ascending=False).round(3),
               alt=f"Bar chart of permutation importance, led by {join_and(f[0].lower() + f[1:] for f in top)}, with the projector "
                   "inputs (guests and neighborhood) highlighted in blue.")
    st.caption("Because host track record matters so much, the projector shows a range rather than just one number. "
               "A new listing with no reviews should expect the lower part of it.")

with dem_tab:
    demand, band, bt = art["demand"]["2019-01":], art["band"], art["backtest"]
    headline("Bookings per Listing Are Forecast to Stay Near Today's Level, with Growing Uncertainty",
             f"Monthly reviews per active listing, actual and forecast (median and 80% range from 2,000 simulations, "
             f"with a fitted trend damping of {art['damping']:.2f}).")
    fig = go.Figure()
    fig.add_scatter(x=band.index, y=band["high"], mode="lines", line_width=0, showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=band.index, y=band["low"], mode="lines", line_width=0, fill="tonexty", fillcolor=BAND,
                    name="80% forecast range", hoverinfo="skip")
    fig.add_scatter(x=demand.index, y=demand, mode="lines", line=dict(color=SERIES[0], width=2), name="Actual",
                    hovertemplate="%{x|%b %Y}: %{y:.2f}<extra>Actual</extra>")
    fig.add_scatter(x=band.index, y=band["typical"], mode="lines", line=dict(color=SERIES[0], width=2, dash="dash"),
                    name="Forecast (median)", hovertemplate="%{x|%b %Y}: %{y:.2f}<extra>Forecast</extra>")
    fig.add_vrect(x0="2020-03-01", x1="2021-06-01", fillcolor=GRID, opacity=0.5, line_width=0,
                  annotation_text="COVID-19", annotation_position="top left", annotation_font_color=INK_2)
    fig.update_layout(yaxis=dict(title="Reviews per active listing, per month", rangemode="tozero"), hovermode="x")
    show_chart(fig, height=400, key="trust_demand",
               data=art["demand_index"].round(2).rename_axis("year").reset_index(),
               alt="Line chart of monthly reviews per active listing from 2019, followed by a dashed forecast to 2030 whose "
                   "median stays close to recent levels while its shaded 80% range widens over time.")
    hw, naive = bt["holt_winters"].mean(), bt["same_month_last_year"].mean()
    st.markdown(f"**Backtest.** When the model is refit at three past dates and asked to forecast 18 months ahead, its average "
                f"error is **{hw:.1%}**, vs. **{naive:.1%}** for simply repeating last year. It's no better than that simple baseline, "
                "so the year mainly widens the range rather than moving the estimate.")

