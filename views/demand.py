import calendar as cal
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dcairbnb.data import load
from dcairbnb.model import load_artifacts
from dcairbnb.style import (AXIS, GRID, INK_2, MUTED, ROOM_COLORS, ROOM_ORDER, SERIES, SURFACE, chapter, headline,
                            join_and, next_page, section, show_chart)

b = load()
df, reviews = b.df, b.reviews

STOPWORDS = set("""a an the and or of in on at to for from with by near is are be this that it its as your you our we
i my me us all one two three four five 1 2 3 4 5 6 br amp has have will can just also very so if into
s t dc washington""".split())


@st.cache_data(show_spinner=False)
def top_words(field, n=20):
    counts = Counter()
    for text in load().df[field].dropna():
        text = re.sub(r"<[^>]+>", " ", text.lower())   # strip HTML like <br />
        counts.update(w for w in re.findall(r"[a-z][a-z']+", text) if w not in STOPWORDS)
    return pd.DataFrame(counts.most_common(n), columns=["word", "count"])


@st.cache_data(show_spinner="Comparing amenities…")
def amenity_effects():
    """Median price and average rating with vs. without each amenity, entire homes only."""
    homes = load().df[lambda d: d["room_type"] == "Entire home/apt"]
    has = pd.DataFrame({a: homes["amenity_list"].map(lambda lst, a=a: a in lst)
                        for a in pd.Series(homes["amenity_list"].explode()).value_counts().index})
    has = has.loc[:, has.mean().between(0.05, 0.95)]   # a near-universal amenity can't separate listings
    rated = homes["number_of_reviews"] >= 3
    rows = []
    for a in has.columns:
        h = has[a]
        rows.append(dict(amenity=a, share=h.mean(),
                         price_with=homes.loc[h, "price"].median(), price_without=homes.loc[~h, "price"].median(),
                         rating_with=homes.loc[h & rated, "review_scores_rating"].mean(),
                         rating_without=homes.loc[~h & rated, "review_scores_rating"].mean()))
    eff = pd.DataFrame(rows)
    eff["price_diff"] = eff["price_with"] / eff["price_without"] - 1
    eff["rating_diff"] = eff["rating_with"] - eff["rating_without"]
    return eff


monthly = reviews.set_index("date").resample("MS").size().rename("reviews")
monthly = monthly[monthly.index < b.scrape_date.to_period("M").to_timestamp()]  # drop the partial scrape month
yearly = monthly.groupby(monthly.index.year).sum()
growth = yearly.loc[2025] / yearly.loc[2019]

chapter("Chapter 2 · Demand", "Demand Has Doubled, and It Follows the Seasons",
        f"Reviews stand in for bookings, and 2025 had {growth:.1f}× the reviews of 2019, with guests coming for the "
        "cherry blossoms and in the fall, booking weekends first and choosing listings that sell location.")

# ---- Over time --------------------------------------------------------------------
section("Bookings over time")
view = st.segmented_control("Show", ["Total reviews", "Reviews per active listing"], default="Total reviews", key="ts_view")
if view == "Reviews per active listing":
    per = load_artifacts()["demand"]
    series, ylab, fmt = per["2016-01":], "Reviews per active listing, per month", ".2f"
    headline("Per Listing, Demand Has Slipped Since 2023: New Listings Absorb the Growth",
             "Total reviews divided by the number of listings active that month.")
else:
    series, ylab, fmt = monthly, "Reviews per month", ",.0f"
    headline(f"2025 Had {growth:.1f}× the Reviews of Pre-COVID 2019",
             "Drag the slider under the chart to zoom into a period.")
fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(color=SERIES[0], width=2),
                           hovertemplate=f"%{{x|%b %Y}}: %{{y:{fmt}}}<extra></extra>"))
fig.add_vrect(x0="2020-03-01", x1="2021-06-01", fillcolor=GRID, opacity=0.5, line_width=0,
              annotation_text="COVID-19", annotation_position="top left", annotation_font_color=INK_2)
fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07, bgcolor=GRID))
fig.update_layout(yaxis=dict(title=ylab, rangemode="tozero"), hovermode="x")
show_chart(fig, height=420, key="reviews_ts",
           data=series.rename(ylab).rename_axis("month").reset_index(),
           alt=("Line chart of monthly reviews since 2009, which grew steadily to 2019, fell sharply in 2020 during COVID-19, "
                f"then recovered and kept climbing to {yearly.loc[2025]:,} in 2025 vs. {yearly.loc[2019]:,} in 2019.")
               if view != "Reviews per active listing" else
               "Line chart of reviews per active listing per month since 2016, which dips in 2020, recovers by 2022 "
               "and drifts slightly lower after 2023.")

left, right = st.columns(2, gap="large")
with left:
    recent = reviews[reviews["date"].dt.year.between(2023, 2025)]
    season = (recent.groupby([recent["date"].dt.year.rename("year"), recent["date"].dt.month.rename("month")]).size()
                    .groupby("month").mean().rename("avg_reviews").reset_index())
    season["month_name"] = season["month"].map(lambda m: cal.month_abbr[m])
    peak = season.nlargest(2, "avg_reviews")["month_name"].tolist()
    headline("Demand Peaks in Spring and October", "Average reviews per month, 2023–2025.")
    fig = px.bar(season, x="month_name", y="avg_reviews")
    fig.update_traces(marker_color=[SERIES[0] if m in peak else MUTED for m in season["month_name"]],
                      hovertemplate="%{x}: %{y:,.0f} reviews on average<extra></extra>")
    fig.update_layout(xaxis_title=None, yaxis_title="Average reviews per month")
    show_chart(fig, height=330, key="season", data=season[["month_name", "avg_reviews"]].round(0),
               alt=f"Bar chart of average reviews by calendar month, with the two busiest months, "
                   f"{join_and(cal.month_name[list(cal.month_abbr).index(m)] for m in peak)}, highlighted and winter the quietest.")
with right:
    daily = b.daily.assign(weekday=b.daily["date"].dt.day_name())
    near = daily[daily["date"] < b.scrape_date + pd.Timedelta(days=90)]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day = near.groupby("weekday")["share_available"].mean().reindex(days).reset_index()
    headline("Fridays and Saturdays Book Up First", "Average share of listings still open over the next 90 days.")
    fig = px.bar(by_day, x="weekday", y="share_available")
    fig.update_traces(marker_color=[SERIES[1] if d in ("Friday", "Saturday") else MUTED for d in by_day["weekday"]],
                      hovertemplate="%{x}: %{y:.1%} available<extra></extra>")
    fig.update_layout(xaxis_title=None, yaxis=dict(title="Share of listings available", tickformat=".0%"))
    show_chart(fig, height=330, key="weekday", data=by_day.round(3),
               alt="Bar chart of availability by weekday over the next 90 days, with Friday and Saturday highlighted "
                   "because they have the lowest share of open listings and book first.")

events = {"2026-11-26": "Thanksgiving", "2026-12-31": "New Year's Eve", "2027-03-27": "Cherry blossoms",
          "2027-05-31": "Memorial Day", "2027-07-04": "July 4th"}
headline("Looking Ahead: Open Dates Fill from the Near Term Outward",
         "Share of DC listings open to book on each date over the next 12 months.")
fig = go.Figure(go.Scatter(x=daily["date"], y=daily["share_available"], mode="lines", line=dict(color=SERIES[0], width=2),
                           hovertemplate="%{x|%a %b %d, %Y}: %{y:.1%} available<extra></extra>"))
for d, label in events.items():
    fig.add_vline(x=pd.Timestamp(d).timestamp() * 1000, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=d, y=1, yref="paper", text=label, showarrow=False, textangle=-90, xanchor="right",
                       yanchor="top", font=dict(size=11, color=INK_2))
fig.update_layout(yaxis=dict(title="Share of listings available", tickformat=".0%"), hovermode="x")
show_chart(fig, height=380, key="calendar",
           alt=f"Line chart of daily availability over the next year, rising from about "
               f"{daily['share_available'].iloc[:21].mean():.0%} of listings open in the next three weeks to about "
               f"{daily['share_available'].max():.0%} for far-off dates that simply haven't been booked yet.")

# ---- Ratings ----------------------------------------------------------------------
section("What guests say")
rated = df[df["number_of_reviews"] >= 3]
share_48 = (rated["review_scores_rating"] >= 4.8).mean()
left, right = st.columns([3, 2], gap="large")
with left:
    headline(f"Ratings Are Crowded Near 5 Stars: {share_48:.0%} of Listings Score 4.8+",
             "Overall rating for listings with 3+ reviews.")
    room_pick = st.selectbox("Room type", ["All room types"] + ROOM_ORDER, key="rating_room")
    sub = rated if room_pick == "All room types" else rated[rated["room_type"] == room_pick]
    fig = go.Figure(go.Histogram(x=sub["review_scores_rating"], xbins=dict(start=3.0, end=5.05, size=0.05),
                                 marker=dict(color=ROOM_COLORS.get(room_pick, SERIES[0]), line=dict(width=1, color=SURFACE)),
                                 hovertemplate="Rating %{x}: %{y:,} listings<extra></extra>"))
    fig.update_layout(xaxis=dict(title="Overall rating (ratings under 3 not shown)", range=[3, 5.05]), yaxis_title="Listings")
    show_chart(fig, height=340, key="ratings",
               alt=f"Histogram of overall ratings for {room_pick.lower()}, which cluster just below 5 stars, with "
                   f"{(sub['review_scores_rating'] >= 4.8).mean():.0%} of listings rated 4.8 or higher.")
with right:
    sub_cols = ["accuracy", "cleanliness", "checkin", "communication", "location", "value"]
    subs = (rated[[f"review_scores_{c}" for c in sub_cols]].mean()
              .rename(lambda c: c.replace("review_scores_", "").replace("checkin", "check-in").title())
              .sort_values().rename_axis("category").reset_index(name="score"))
    headline("Guests Rate Value and Location Lowest")
    fig = px.scatter(subs, x="score", y="category")
    fig.update_traces(marker=dict(color=[SERIES[1]] * 2 + [SERIES[0]] * 4, size=13, line=dict(width=2, color=SURFACE)),
                      hovertemplate="%{y}: %{x:.2f}<extra></extra>")
    fig.update_layout(xaxis=dict(title="Average score", range=[4.5, 5]), yaxis_title=None)
    show_chart(fig, height=340, key="subscores", data=subs.round(2),
               alt="Dot plot of average review sub-scores: " + "; ".join(f"{r.category} {r.score:.2f}" for r in subs.itertuples()) + ".")

# ---- Text -------------------------------------------------------------------------
section("What listings promise")
left, right = st.columns([2, 3], gap="large")
with left:
    field = st.segmented_control("Words in", ["Titles", "Descriptions"], default="Titles", key="text_field")
    words = top_words("description" if field == "Descriptions" else "name")
    headline("Titles Sell Location and Parking: “Capitol”, “Metro” and “Parking” Lead" if field != "Descriptions"
             else "Descriptions Sell Getting Around: “Metro”, “Walk” and “Located” Rank Near the Top",
             f"The 20 most common words in listing {(field or 'Titles').lower()}, after removing filler words.")
    fig = px.bar(words, x="count", y="word", orientation="h")
    fig.update_traces(marker_color=SERIES[0], hovertemplate="%{y}: used %{x:,} times<extra></extra>")
    fig.update_layout(yaxis=dict(title=None, autorange="reversed"), xaxis_title="Times used")
    show_chart(fig, height=560, key="words", data=words,
               alt=f"Bar chart of the most common words in listing {(field or 'Titles').lower()}.")
with right:
    eff = amenity_effects()
    metric = st.segmented_control("Compare amenities by", ["Price", "Rating"], default="Price", key="amen_metric")
    col = "rating_diff" if metric == "Rating" else "price_diff"
    ranked = pd.concat([eff.nlargest(10, col), eff.nsmallest(8, col)]).sort_values(col)
    if metric == "Rating":
        headline("Gyms and Elevators Go with Lower Ratings, a Sign of Corporate Apartment Buildings",
                 "Average rating with vs. without each amenity, for entire homes with 3+ reviews.")
        hover = "<b>%{y}</b><br>Avg rating with: %{customdata[0]:.2f}<br>Without: %{customdata[1]:.2f}<extra></extra>"
        custom, xaxis = ranked[["rating_with", "rating_without", "share"]], dict(title="Difference in average rating", tickformat="+.2f")
    else:
        headline("Family and Luxury Amenities Go with 40–65% Higher Prices",
                 "Median nightly price with vs. without each amenity, for entire homes.")
        hover = ("<b>%{y}</b><br>Median with: $%{customdata[0]:,.0f}<br>Without: $%{customdata[1]:,.0f}"
                 "<br>In %{customdata[2]:.0%} of entire homes<extra></extra>")
        custom, xaxis = ranked[["price_with", "price_without", "share"]], dict(title="Price difference", tickformat="+.0%")
    fig = go.Figure(go.Bar(x=ranked[col], y=ranked["amenity"], orientation="h", customdata=custom,
                           marker_color=[SERIES[0] if v > 0 else SERIES[1] for v in ranked[col]], hovertemplate=hover))
    fig.update_layout(xaxis=xaxis, yaxis_title=None)
    show_chart(fig, height=560, key="amenities", data=ranked.round(3),
               alt=f"Diverging bar chart of the {metric.lower() if metric else 'price'} difference for listings with vs. "
                   f"without each amenity, where blue bars are positive (highest: {join_and(ranked['amenity'].iloc[::-1].head(3))}) "
                   f"and orange bars are negative (lowest: {join_and(ranked['amenity'].head(3))}).")


next_page("views/place.py", "3 · Place & Power")
