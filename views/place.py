import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dcairbnb.data import load
from dcairbnb.style import (AXIS, DC_CENTER, INK_2, MAP_LEGEND, ROOM_COLORS, ROOM_ORDER, SEQ_BLUE, SERIES, SURFACE, chapter,
                            headline, join_and, next_page, section, show_chart, short_name, title_case)

b = load()
df, hosts = b.df, b.hosts

priced = df[df["price"].notna()]
trim = priced[priced["price"].between(*priced["price"].quantile([0.01, 0.99]))]
neigh = (df.groupby("neighbourhood_cleansed")
           .agg(listings=("id", "size"), median_price=("price", "median"),
                pct_entire_home=("room_type", lambda s: (s == "Entire home/apt").mean()),
                total_est_revenue=("estimated_revenue_l365d", "sum"))
           .join((trim["price"] / trim["accommodates"]).groupby(trim["neighbourhood_cleansed"]).median().rename("price_per_guest"))
           .reindex(b.neighbourhoods).rename_axis("neighbourhood").reset_index())
neigh["area"] = neigh["neighbourhood"].map(short_name)

chapter("Chapter 3 · Place & Power", "An East–West Divide, and Operators Who Span the City",
        "Listings and prices concentrate in central and western DC, while east of the Anacostia River a guest pays less "
        "than half as much per night.")

section("Where the listings are")
METRICS = {
    "Listings": ("listings", ",.0f", "", "Listings Cluster in Central DC"),
    "Median price": ("median_price", ",.0f", "$", "The West End, Downtown and Georgetown Are the Priciest Neighborhoods"),
    "Price per guest": ("price_per_guest", ",.0f", "$", "Price per Guest Is Highest Downtown and Lowest East of the River"),
    "Share entire homes": ("pct_entire_home", ".0%", "", "Entire Homes Dominate Everywhere Except the West End and Kalorama"),
    "Est. revenue": ("total_est_revenue", ",.0f", "$", "Estimated Revenue Concentrates in Union Station and Capitol Hill"),
}
pick = st.segmented_control("Color neighborhoods by", list(METRICS), default="Listings", key="map_metric") or "Listings"
col, fmt, prefix, title = METRICS[pick]
measure = {"Listings": "number of listings", "Share entire homes": "share of entire homes",
           "Est. revenue": "estimated revenue"}.get(pick, pick.lower())   # for chart descriptions
headline(title, "Hover over a neighborhood for details.")
map_col, rank_col = st.columns([3, 2], gap="large")
with map_col:
    fig = go.Figure(go.Choroplethmap(
        geojson=b.geo, featureidkey="properties.neighbourhood", locations=neigh["neighbourhood"], z=neigh[col],
        colorscale=SEQ_BLUE, marker=dict(opacity=0.85, line=dict(width=1, color="white")),
        colorbar=dict(title=dict(text=pick, font=dict(color=INK_2)), tickprefix=prefix, tickformat=fmt, thickness=12, outlinewidth=0),
        customdata=np.stack([neigh["neighbourhood"], neigh["listings"]], axis=-1),
        hovertemplate=f"<b>%{{customdata[0]}}</b><br>{pick}: {prefix}%{{z:{fmt}}}<br>Listings: %{{customdata[1]:,}}<extra></extra>"))
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), map=dict(style="carto-positron", center=DC_CENTER, zoom=10.4))
    show_chart(fig, height=560, key="choropleth",
               alt=f"Map of DC's 39 neighborhoods shaded by {measure}, highest in {join_and(neigh.nlargest(3, col)['area'])} "
                   f"and lowest in {join_and(neigh.nsmallest(3, col)['area'])}.")
with rank_col:
    ranked = neigh.dropna(subset=[col]).sort_values(col, ascending=False).head(12)
    fig = px.bar(ranked, x=col, y="area", orientation="h", custom_data=["neighbourhood", "listings"])
    fig.update_traces(marker_color=SERIES[0], hovertemplate=f"<b>%{{customdata[0]}}</b><br>{prefix}%{{x:{fmt}}}<extra></extra>")
    fig.update_layout(yaxis=dict(title=None, autorange="reversed"), xaxis=dict(title=pick, tickprefix=prefix, tickformat=fmt),
                      margin=dict(t=0))
    st.markdown(f"**{title_case('Top 12 by ' + pick.replace('Share', 'Share of'))}**")
    show_chart(fig, height=520, key="choropleth_rank", data=neigh[list(dict.fromkeys(["neighbourhood", col, "listings"]))].round(3),
               alt=f"Bar chart of the 12 neighborhoods with the highest {measure}, led by {ranked['area'].iloc[0]}.")

section("Every listing, filtered")
f1, f2 = st.columns([3, 2])
rooms = f1.pills("Room types", ROOM_ORDER, default=ROOM_ORDER, selection_mode="multi", key="pt_rooms") or ROOM_ORDER
lo, hi = f2.slider("Nightly price", 0, 1000, (0, 1000), step=25, format="$%d", key="pt_price",
                   help="Listings priced over $1,000 are included when the upper handle is at $1,000.")
pts = df[df["room_type"].isin(rooms) & df["price"].notna() & (df["price"] >= lo) & ((df["price"] <= hi) | (hi == 1000))]
headline(f"{len(pts):,} Listings Match", "Each dot is one listing, with its title, neighborhood and price on hover.")
fig = px.scatter_map(pts, lat="latitude", lon="longitude", color="room_type", category_orders={"room_type": ROOM_ORDER},
                     color_discrete_map=ROOM_COLORS, custom_data=["name", "price", "short_name"],
                     map_style="carto-positron", center=DC_CENTER, zoom=10.9, opacity=0.75)
fig.update_traces(marker_size=6, hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[2]}<br>$%{customdata[1]:,.0f} a night<extra></extra>")
fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), legend=MAP_LEGEND)
top_area = pts["short_name"].value_counts()
show_chart(fig, height=520, key="points",
           alt=f"Map of {len(pts):,} listings matching the filters, most of them in {join_and(top_area.index[:3])}."
               if len(pts) else "No listings match the filters.")

section("Who operates where")
min_listings = st.select_slider("Show hosts with at least this many listings", [10, 20, 30, 50, 100], value=10, key="net_min")
big = hosts[hosts["listings"] >= min_listings]
edges = (df[df["host_id"].isin(big.index)].groupby(["host_id", "neighbourhood_cleansed"]).size()
           .rename("listings").reset_index())
spread = (edges.groupby("host_id").agg(neighbourhoods=("neighbourhood_cleansed", "nunique"), listings=("listings", "sum"))
               .join(hosts["host_name"]).sort_values(["neighbourhoods", "listings"], ascending=False))
reach = edges["neighbourhood_cleansed"].nunique()
headline(f"{len(big)} Hosts with {min_listings}+ Listings Reach {reach} of DC's 39 Neighborhoods")
highlight = st.selectbox("Highlight a host", ["None"] + spread["host_name"].fillna("Unknown").tolist(), key="net_host")


@st.cache_data(show_spinner=False)
def layout(min_listings):
    """Spring layout for the host ↔ neighborhood graph (fixed seed, so it doesn't move between reruns)."""
    d, h = load().df, load().hosts
    links = d.loc[d["host_id"].isin(h.index[h["listings"] >= min_listings]), ["host_id", "neighbourhood_cleansed"]]
    g = nx.Graph()
    g.add_edges_from((f"h{hid}", n) for hid, n in links.drop_duplicates().itertuples(index=False))
    return nx.spring_layout(g, k=0.35, iterations=200, seed=42)


pos = layout(min_listings)
hl_id = None if highlight == "None" else spread.index[spread["host_name"].fillna("Unknown") == highlight][0]
fig = go.Figure()
for strong in (False, True):   # draw faded edges first, highlighted edges on top
    xs, ys, widths = [], [], []
    for r in edges.itertuples():
        if (r.host_id == hl_id) != strong:
            continue
        a, c = pos[f"h{r.host_id}"], pos[r.neighbourhood_cleansed]
        fig.add_trace(go.Scatter(x=[a[0], c[0]], y=[a[1], c[1]], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(width=0.5 + np.log1p(r.listings) * (1.4 if strong else 1),
                                           color=SERIES[0] if strong else AXIS)))
nb_listings = edges.groupby("neighbourhood_cleansed")["listings"].sum()
host_nodes = [(hid, f"h{hid}") for hid in spread.index]
fig.add_trace(go.Scatter(
    x=[pos[n][0] for _, n in host_nodes], y=[pos[n][1] for _, n in host_nodes], mode="markers", name="Host",
    marker=dict(size=[8 + 2.2 * np.sqrt(spread.loc[h, "listings"]) for h, _ in host_nodes], symbol="circle",
                color=[SERIES[0] if (hl_id is None or h == hl_id) else "#b8d2f2" for h, _ in host_nodes],
                line=dict(width=2, color=SURFACE)),
    customdata=[[spread.loc[h, "host_name"], spread.loc[h, "listings"], spread.loc[h, "neighbourhoods"]] for h, _ in host_nodes],
    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} listings in %{customdata[2]} neighborhoods<extra>Host</extra>"))
hl_areas = set(edges.loc[edges["host_id"] == hl_id, "neighbourhood_cleansed"])
fig.add_trace(go.Scatter(
    x=[pos[n][0] for n in nb_listings.index], y=[pos[n][1] for n in nb_listings.index], mode="markers+text", name="Neighborhood",
    marker=dict(size=[8 + 2.2 * np.sqrt(v) for v in nb_listings], symbol="square", line=dict(width=2, color=SURFACE),
                color=[SERIES[1] if (hl_id is None or n in hl_areas) else "#f6c7b1" for n in nb_listings.index]),
    text=[short_name(n) if (hl_id is None or n in hl_areas) else "" for n in nb_listings.index],
    textposition="top center", textfont=dict(size=10, color=INK_2),
    customdata=[[n, v] for n, v in nb_listings.items()],
    hovertemplate=f"<b>%{{customdata[0]}}</b><br>%{{customdata[1]}} listings from hosts with {min_listings}+<extra>Neighborhood</extra>"))
fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), hovermode="closest", margin=dict(t=30, l=0, r=0, b=0))
show_chart(fig, height=720, key="network",
           data=spread.reset_index()[["host_name", "listings", "neighbourhoods"]],
           alt=f"Network diagram linking {len(big)} large hosts (blue circles) to the {reach} neighborhoods (orange squares) where "
               f"they have listings, with {join_and(spread['host_name'].head(3).fillna('Unknown'))} the most widespread hosts "
               "and central neighborhoods the most connected.")

next_page("views/rules.py", "4 · Licensing")
