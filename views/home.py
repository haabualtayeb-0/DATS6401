import plotly.express as px
import streamlit as st

from dcairbnb.data import load
from dcairbnb.style import DC_CENTER, MAP_LEGEND, ROOM_COLORS, ROOM_ORDER, chapter, headline, next_page, show_chart

b = load()
df, hosts = b.df, b.hosts

multi_share = hosts.loc[hosts["listings"] > 1, "listings"].sum() / len(df)
exempt = df["license_type"] == "Exempt"

chapter("Inside Airbnb · Washington, DC · September 2026",
        "Who Really Runs Airbnb in DC?",
        f"Most DC hosts rent out a single home but most listings belong to operators with several, so this story "
        f"follows {len(df):,} listings, nearly half a million reviews and a year of calendar data to see "
        f"who controls the market, where the money is and what a new listing could earn.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings", f"{len(df):,}", border=True, help="Active listings as of September 2026.")
c2.metric("Hosts with One Listing", f"{(hosts['listings'] == 1).mean():.0%}", border=True)
c3.metric("Listings Run by Multi-Listing Hosts", f"{multi_share:.0%}", border=True)
c4.metric("Median Nightly Price", f"${df['price'].median():,.0f}", border=True)

headline("Every DC Listing, Colored by Room Type")
fig = px.scatter_map(df, lat="latitude", lon="longitude", color="room_type",
                     category_orders={"room_type": ROOM_ORDER}, color_discrete_map=ROOM_COLORS,
                     custom_data=["name", "price", "short_name"], map_style="carto-positron",
                     center=DC_CENTER, zoom=10.9, opacity=0.75)
fig.update_traces(marker_size=6, hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[2]}"
                                               "<br>$%{customdata[1]:,.0f} a night<extra></extra>")
fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), legend=MAP_LEGEND)
show_chart(fig, height=560, key="home_map",
           alt=f"Map of {len(df):,} listings across Washington, DC, densest in central neighborhoods such as "
               f"Capitol Hill, Shaw, Dupont Circle and Navy Yard.")

st.subheader("The Story in Five Parts", anchor=False)
cards = [
    ("views/market.py", ":material/apartment:", "1 · The Market",
     f"{(hosts['listings'] == 1).mean():.0%} of hosts have one listing, but a few operators run hundreds, and many rent only by the month."),
    ("views/demand.py", ":material/show_chart:", "2 · Demand",
     "Reviews have more than doubled since 2019. What guests book, when, and what listings promise."),
    ("views/place.py", ":material/map:", "3 · Place & Power",
     "An east–west price divide, and a network of large operators spanning most of the city."),
    ("views/rules.py", ":material/gavel:", "4 · Licensing",
     f"{exempt.mean():.0%} of listings claim they're exempt from licensing, yet most still take short stays."),
    ("views/projector.py", ":material/calculate:", "Revenue Projector",
     "Choose a household size, neighborhood and year, and see what a short-term rental could earn."),
]
cols = st.columns(len(cards))
for col, (page, icon, title, text) in zip(cols, cards):
    with col.container(border=True, height="stretch"):
        st.page_link(page, label=f"**{title}**", icon=icon)
        st.write(text)

next_page("views/market.py", "1 · The Market")
