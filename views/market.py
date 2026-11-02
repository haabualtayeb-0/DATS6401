import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dcairbnb.data import load
from dcairbnb.style import (INK_2, MUTED, ROOM_COLORS, ROOM_ORDER, SERIES, STAY_ORDER, SURFACE, chapter, headline,
                            next_page, section, show_chart, takeaway)

b = load()
df, hosts = b.df, b.hosts

one_host = (hosts["listings"] == 1).mean()
multi_share = hosts.loc[hosts["listings"] > 1, "listings"].sum() / len(df)
long_share = (df["minimum_nights"] >= 30).mean()

chapter("Chapter 1 · The Market", "Many Hosts, but a Few Big Operators",
        f"{one_host:.0%} of DC hosts list a single home, yet hosts with more than one listing run {multi_share:.0%} of all "
        f"listings, and many of the biggest operators skip short stays to rent by the month.")

section("What gets listed")
left, right = st.columns(2, gap="large")

with left:
    room = (df.groupby("room_type").agg(listings=("id", "size"), median_price=("price", "median"),
                                        median_guests=("accommodates", "median"))
              .reindex(ROOM_ORDER).reset_index())
    room["share"] = room["listings"] / room["listings"].sum()
    measures = {"Listings": ("listings", "{:,.0f}", "Number of listings"),
                "Median price": ("median_price", "${:,.0f}", "Median nightly price"),
                "Median guests": ("median_guests", "{:.0f}", "Median guests accommodated")}
    headline(f"{room['share'].iloc[0]:.0%} of Listings Are Entire Homes")
    pick = st.segmented_control("Measure", list(measures), default="Listings", key="room_measure")
    col, fmt, axis_title = measures[pick or "Listings"]
    fig = px.bar(room, x=col, y="room_type", orientation="h", text=room[col].map(fmt.format), color="room_type",
                 color_discrete_map=ROOM_COLORS)
    fig.update_traces(textposition="outside", cliponaxis=False, textfont_color=INK_2,
                      hovertemplate="%{y}: " + fmt.replace("{:", "%{x:") + "<extra></extra>")
    fig.update_layout(showlegend=False, xaxis=dict(title=axis_title, range=[0, room[col].max() * 1.18]),
                      yaxis=dict(title=None, categoryorder="array", categoryarray=ROOM_ORDER[::-1]))
    show_chart(fig, height=300, key="room_bar", data=room,
               alt=f"Bar chart of {axis_title.lower()} by room type: entire homes, private rooms, hotel rooms and shared rooms.")

with right:
    headline("Most Listings Sleep 2 or 4 Guests")
    rooms = st.pills("Room types", ROOM_ORDER, default=ROOM_ORDER, selection_mode="multi", key="cap_rooms")
    sub = df[df["room_type"].isin(rooms or ROOM_ORDER)]
    counts = (sub.groupby(["accommodates", "room_type"]).size().rename("listings").reset_index())
    fig = px.bar(counts, x="accommodates", y="listings", color="room_type", color_discrete_map=ROOM_COLORS,
                 category_orders={"room_type": ROOM_ORDER})
    fig.update_traces(marker_line=dict(width=1, color=SURFACE),
                      hovertemplate="Sleeps %{x}: %{y:,} listings<extra>%{fullData.name}</extra>")
    fig.update_layout(barmode="stack", xaxis=dict(title="Guests accommodated", dtick=1), yaxis_title="Listings")
    show_chart(fig, height=300, key="cap_bar", data=counts,
               alt="Stacked bar chart of listings by guest capacity.")

price_by_cap = (df[(df["accommodates"] <= 10) & df["room_type"].isin(ROOM_ORDER[:2])]
                  .groupby(["room_type", "accommodates"])["price"].agg(median_price="median", listings="size").reset_index())
price_by_cap = price_by_cap[price_by_cap["listings"] >= 10]
homes = price_by_cap[price_by_cap["room_type"] == ROOM_ORDER[0]].set_index("accommodates")["median_price"]
headline(f"Each Extra Guest Adds to the Price: An Entire Home Goes from ${homes.loc[2]:,.0f} for 2 Guests to "
         f"${homes.loc[6]:,.0f} for 6",
         "Median nightly price by guest capacity, hiding groups with fewer than 10 listings.")
fig = px.line(price_by_cap, x="accommodates", y="median_price", color="room_type", color_discrete_map=ROOM_COLORS,
              markers=True, custom_data=["listings", "room_type"])
fig.update_traces(line_width=2, marker=dict(size=9, line=dict(width=2, color=SURFACE)),
                  hovertemplate="%{customdata[1]}: $%{y:,.0f} (%{customdata[0]:,} listings)<extra></extra>")
fig.update_layout(xaxis=dict(title="Guests accommodated", dtick=1), hovermode="x unified",
                  yaxis=dict(title="Median nightly price", tickprefix="$", rangemode="tozero"))
show_chart(fig, height=360, key="price_cap", data=price_by_cap,
           alt="Line chart of median nightly price by guest capacity, where entire homes rise steadily in price as "
               "capacity grows while private rooms stay cheaper at every size.")

section("Who runs it")
port = hosts.groupby("portfolio", observed=False).agg(Hosts=("listings", "size"), Listings=("listings", "sum"))
port_share = (port / port.sum()).reset_index().melt(id_vars="portfolio", var_name="measure", value_name="share")
headline(f"{one_host:.0%} of Hosts Have One Listing, but Multi-Listing Hosts Run {multi_share:.0%} of Listings",
         "Share of all hosts vs. share of all listings, by how many listings a host runs.")
fig = px.bar(port_share, x="portfolio", y="share", color="measure", barmode="group",
             color_discrete_map={"Hosts": MUTED, "Listings": SERIES[0]})
fig.update_traces(marker_line=dict(width=2, color=SURFACE), hovertemplate="%{x} listings: %{y:.1%}<extra>%{fullData.name}</extra>")
fig.update_layout(xaxis_title="Listings per host", yaxis=dict(title="Share of DC total", tickformat=".0%"))
show_chart(fig, height=380, key="portfolio", data=(port / port.sum()).round(3).reset_index(),
           alt=f"Grouped bar chart showing that single-listing hosts are {one_host:.0%} of hosts but hold only "
               f"{port_share.query('portfolio == \"1\" and measure == \"Listings\"')['share'].iloc[0]:.0%} of listings, "
               f"while hosts with 51+ listings are a tiny share of hosts but hold "
               f"{port_share.query('portfolio == \"51+\" and measure == \"Listings\"')['share'].iloc[0]:.0%}.")

top_by = st.segmented_control("Rank hosts by", ["Listings", "Estimated revenue"], default="Listings", key="top_by",
                              format_func=lambda o: {"Listings": ":material/apartment: Listings",
                                                     "Estimated revenue": ":material/payments: Estimated revenue"}[o])
col = "est_revenue" if top_by == "Estimated revenue" else "listings"
top = hosts.sort_values(col, ascending=False).head(15).reset_index()
top["label"] = top["host_name"].fillna("Unknown")
lead = top.iloc[0]
headline(f"{lead['label']} Leads with " + (f"{lead['listings']:,} Listings" if col == "listings"
                                            else f"About ${lead['est_revenue'] / 1e6:,.1f}M in Estimated Revenue"),
         "The 15 largest hosts, some of which appear under more than one host account.")
fig = px.bar(top, x=col, y="label", orientation="h", custom_data=["listings", "est_revenue", "entire_homes"])
fig.update_traces(marker_color=SERIES[0],
                  hovertemplate="<b>%{y}</b><br>%{customdata[0]} listings (%{customdata[2]} entire homes)"
                                "<br>Est. revenue: $%{customdata[1]:,.0f}<extra></extra>")
fig.update_layout(yaxis=dict(title=None, autorange="reversed"),
                  xaxis=dict(title="Listings" if col == "listings" else "Estimated revenue, last 12 months",
                             tickprefix="" if col == "listings" else "$"))
show_chart(fig, height=500, key="top_hosts", data=top[["label", "listings", "entire_homes", "est_revenue"]],
           alt=f"Horizontal bar chart of the 15 largest hosts by {top_by.lower() if top_by else 'listings'}.")

section("Short stays vs. monthly rentals")
left, right = st.columns(2, gap="large")
with left:
    stay = df["stay_type"].value_counts(normalize=True).reindex(STAY_ORDER).rename_axis("stay_type").reset_index(name="share")
    headline(f"{long_share:.0%} of Listings Require a Stay of 30+ Nights")
    fig = px.bar(stay, x="stay_type", y="share", text=stay["share"].map("{:.0%}".format))
    fig.update_traces(marker_color=[SERIES[0]] * 3 + [SERIES[1]], textposition="outside", cliponaxis=False,
                      textangle=0, constraintext="none", textfont_color=INK_2,
                      hovertemplate="Minimum %{x}: %{y:.1%} of listings<extra></extra>")
    fig.update_layout(xaxis_title="Minimum stay",   # headroom so the tallest bar's label fits
                      yaxis=dict(title="Share of listings", tickformat=".0%", range=[0, stay["share"].max() * 1.2]))
    show_chart(fig, height=340, key="stay", data=stay,
               alt="Bar chart of listings by minimum stay: " + "; ".join(f"{r.stay_type} {r.share:.0%}" for r in stay.itertuples()) + ".")
with right:
    long_by_port = (df.groupby("portfolio", observed=False)["minimum_nights"].apply(lambda s: (s >= 30).mean())
                      .rename("share_30plus").reset_index())
    biggest = long_by_port["share_30plus"].iloc[-1]
    headline(f"Bigger Hosts Lean on Monthly Rentals: {biggest:.0%} of Units from Hosts with 51+ Listings")
    fig = px.bar(long_by_port, x="portfolio", y="share_30plus", text=long_by_port["share_30plus"].map("{:.0%}".format))
    fig.update_traces(marker_color=SERIES[1], textposition="outside", cliponaxis=False, textangle=0, constraintext="none",
                      textfont_color=INK_2, hovertemplate="%{x} listings per host: %{y:.0%} require 30+ nights<extra></extra>")
    fig.update_layout(xaxis=dict(title="Listings per host", type="category"),
                      yaxis=dict(title="Share with a 30+ night minimum", tickformat=".0%",
                                 range=[0, long_by_port["share_30plus"].max() * 1.2]))
    show_chart(fig, height=340, key="long_port", data=long_by_port,
               alt=f"Bar chart of the share of 30+ night listings by host size, from "
                   f"{long_by_port['share_30plus'].iloc[0]:.0%} for single-listing hosts to {biggest:.0%} for hosts with 51+ listings.")


# ---- Host quality -----------------------------------------------------------------
section("Does being a Superhost pay?")
rated = df["number_of_reviews"] >= 3
sh = pd.DataFrame({
    "rating": df[rated].groupby("superhost")["review_scores_rating"].mean(),
    "reviews": df.groupby("superhost")["number_of_reviews_ltm"].median(),
    "revenue": df.groupby("superhost")["estimated_revenue_l365d"].median(),
})
s, n = sh.loc["Superhost"], sh.loc["Not superhost"]
headline("Superhosts Earn Better Ratings, Many More Bookings and More Revenue")
c1, c2, c3 = st.columns(3)
c1.metric("Average Rating", f"{s['rating']:.2f} ★", f"{s['rating'] - n['rating']:+.2f} vs. {n['rating']:.2f}", border=True)
c2.metric("Median Reviews, Last 12 Months", f"{s['reviews']:.0f}", f"vs. {n['reviews']:.0f} for other hosts", border=True,
          delta_color="off")
c3.metric("Median Estimated Revenue", f"${s['revenue']:,.0f}", f"vs. ${n['revenue']:,.0f} for other hosts", border=True,
          delta_color="off")
takeaway("Host quality pays: guest rating is the strongest predictor in the revenue model, and Superhost status ranks "
         "ahead of how many guests a listing sleeps.")

next_page("views/demand.py", "2 · Demand")
