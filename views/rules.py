import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dcairbnb.data import load
from dcairbnb.style import (DC_CENTER, INK_2, LIC_COLORS, LIC_ORDER, MUTED, SEQ_BLUE, SERIES, SURFACE, chapter, headline,
                            join_and, next_page, section, show_chart, short_name)

b = load()
df = b.df

short = df["minimum_nights"] < 30
lic_share = df["license_type"].value_counts(normalize=True).reindex(LIC_ORDER)
exempt_short = short[df["license_type"] == "Exempt"].mean()
missing_long = (~short)[df["license_type"] == "Missing"].mean()
port_exempt = df.groupby("portfolio", observed=False)["license_type"].apply(lambda s: (s == "Exempt").mean())

chapter("Chapter 4 · Licensing", "A Likely Licensing Gap",
        f"DC requires a hosted or unhosted license for any stay under 30 nights, yet {lic_share['Exempt']:.0%} of listings "
        f"say they're exempt and {exempt_short:.0%} of those still accept short stays.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hosted License", f"{lic_share['Hosted license']:.0%}", border=True, help="Share of all listings.")
c2.metric("Claim “Exempt”", f"{lic_share['Exempt']:.0%}", border=True, help="Share of all listings.")
c3.metric("“Exempt” Listings Taking Short Stays", f"{exempt_short:.0%}", border=True,
          help="Minimum stay under 30 nights, so the exemption for long stays doesn't apply.")
c4.metric("Blank-License Listings That Are Monthly", f"{missing_long:.0%}", border=True,
          help="30+ night minimum, so they likely don't need a license.")

section("What listings claim, and what they do")
left, right = st.columns(2, gap="large")
with left:
    lic = lic_share.rename_axis("license_type").reset_index(name="share")
    headline("Only About Half of Listings Show a Hosted License")
    fig = px.bar(lic, x="share", y="license_type", orientation="h", color="license_type", color_discrete_map=LIC_COLORS,
                 text=lic["share"].map("{:.0%}".format))
    fig.update_traces(textposition="outside", cliponaxis=False, textfont_color=INK_2, hovertemplate="%{y}: %{x:.1%}<extra></extra>")
    fig.update_layout(showlegend=False, xaxis=dict(title="Share of listings", tickformat=".0%"),
                      yaxis=dict(title=None, categoryorder="array", categoryarray=LIC_ORDER[::-1]))
    show_chart(fig, height=340, key="lic_share", data=lic.round(3),
               alt="Bar chart of license types: " + "; ".join(f"{r.license_type} {r.share:.0%}" for r in lic.itertuples()) + ".")
with right:
    stays = (pd.crosstab(df["license_type"], np.where(short, "Under 30 nights", "30+ nights"), normalize="index")
               .reindex(LIC_ORDER)[["Under 30 nights", "30+ nights"]].reset_index()
               .melt(id_vars="license_type", var_name="minimum_stay", value_name="share"))
    headline(f"{exempt_short:.0%} of “Exempt” Listings Still Accept Short Stays",
             "Minimum stay within each license group.")
    fig = px.bar(stays, x="share", y="license_type", color="minimum_stay", orientation="h",
                 color_discrete_map={"Under 30 nights": SERIES[1], "30+ nights": MUTED},
                 category_orders={"license_type": LIC_ORDER})
    fig.update_traces(marker_line=dict(width=2, color=SURFACE),
                      hovertemplate="%{y}<br>%{fullData.name}: %{x:.0%}<extra></extra>")
    fig.update_layout(barmode="stack", xaxis=dict(title="Share of listings in the group", tickformat=".0%"), yaxis_title=None)
    show_chart(fig, height=340, key="lic_stay", data=stays.round(3),
               alt=f"Stacked bar chart of minimum stay by license group, showing that {exempt_short:.0%} of “Exempt” listings "
                   f"accept stays under 30 nights while {missing_long:.0%} of listings with a blank license require 30+ nights.")

lic_port = (df.groupby(["portfolio", "license_type"], observed=False).size()
              .groupby(level=0, observed=False).transform(lambda s: s / s.sum()).rename("share").reset_index())
headline(f"The Bigger the Host, the More Listings Claim “Exempt”: From {port_exempt.iloc[0]:.0%} to {port_exempt.iloc[-1]:.0%}",
         "License mix by how many listings a host runs.")
fig = px.bar(lic_port, x="portfolio", y="share", color="license_type", category_orders={"license_type": LIC_ORDER},
             color_discrete_map=LIC_COLORS)
fig.update_traces(marker_line=dict(width=2, color=SURFACE), hovertemplate="%{x} listings per host<br>%{fullData.name}: %{y:.0%}<extra></extra>")
fig.update_layout(barmode="stack", xaxis_title="Listings per host", yaxis=dict(title="Share of listings", tickformat=".0%"))
show_chart(fig, height=400, key="lic_port", data=lic_port.round(3),
           alt=f"Stacked bar chart of license types by host size, where the “Exempt” share (yellow) rises from {port_exempt.iloc[0]:.0%} "
               f"for single-listing hosts to {port_exempt.iloc[-1]:.0%} for hosts with 51+ listings.")

section("Where the gaps are")
lic_neigh = (df.groupby("neighbourhood_cleansed")
               .agg(listings=("id", "size"),
                    exempt_short=("id", lambda s: ((df.loc[s.index, "license_type"] == "Exempt") & short.loc[s.index]).mean()),
                    missing=("license_type", lambda s: (s == "Missing").mean()),
                    licensed=("license_type", lambda s: s.isin(["Hosted license", "Unhosted license"]).mean()))
               .reindex(b.neighbourhoods).rename_axis("neighbourhood").reset_index())
lic_neigh["area"] = lic_neigh["neighbourhood"].map(short_name)
VIEWS = {"“Exempt” but short-stay": ("exempt_short", "Where “Exempt” Listings Take Short Stays Anyway",
                                     "claim “Exempt” but accept short stays"),
         "No license listed": ("missing", "Where Listings Show No License", "show no license"),
         "Licensed": ("licensed", "Where Listings Hold a Hosted or Unhosted License", "hold a hosted or unhosted license")}
view = st.segmented_control("Map", list(VIEWS), default="“Exempt” but short-stay", key="lic_view") or "“Exempt” but short-stay"
col, title, phrase = VIEWS[view]
top3 = lic_neigh.nlargest(3, col)
headline(title, "Share of each neighborhood's listings.")
fig = go.Figure(go.Choroplethmap(
    geojson=b.geo, featureidkey="properties.neighbourhood", locations=lic_neigh["neighbourhood"], z=lic_neigh[col],
    colorscale=SEQ_BLUE, marker=dict(opacity=0.85, line=dict(width=1, color="white")),
    colorbar=dict(title=dict(text=view, font=dict(color=INK_2)), tickformat=".0%", thickness=12, outlinewidth=0),
    customdata=np.stack([lic_neigh["neighbourhood"], lic_neigh["listings"]], axis=-1),
    hovertemplate=f"<b>%{{customdata[0]}}</b><br>{view}: %{{z:.0%}}<br>Listings: %{{customdata[1]:,}}<extra></extra>"))
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), map=dict(style="carto-positron", center=DC_CENTER, zoom=10.4))
show_chart(fig, height=560, key="lic_map", data=lic_neigh[["neighbourhood", "listings", "exempt_short", "missing", "licensed"]].round(3),
           alt=f"Map of DC neighborhoods shaded by the share of listings that {phrase}, highest in {join_and(top3['area'])}.")

next_page("views/projector.py", "Revenue Projector")
