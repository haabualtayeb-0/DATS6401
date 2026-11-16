import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = "https://gaimsds.github.io/dataviz/dats6401/weekly/week10/data/"
POINTS_URL = BASE + "superchargers_us.csv"
STATES_URL = BASE + "states.geojson"
STATS_URL = BASE + "state_stats.csv"

N_CLASSES = 5
RATE_PER = 1_000_000
EQUAL_AREA = "EPSG:5070"

st.set_page_config(page_title="Superchargers by State", layout="wide")


@st.cache_data(show_spinner="Loading data…")
def load_data():
    pts = pd.read_csv(POINTS_URL, parse_dates=["opened"])
    pts["year"] = pts["opened"].dt.year.astype("Int64")

    states = gpd.read_file(STATES_URL)[["fips", "state", "geometry"]]
    states["fips"] = states["fips"].astype(str).str.zfill(2)

    stats = pd.read_csv(STATS_URL, dtype={"fips": str})

    pts_gdf = gpd.GeoDataFrame(
        pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]), crs="EPSG:4326"
    )
    states = states.to_crs(pts_gdf.crs)

    joined = gpd.sjoin(
        pts_gdf,
        states.rename(columns={"state": "state_name"}),
        how="left",
        predicate="within",
    ).drop(columns="index_right")

    miss = joined[joined["fips"].isna()].drop(columns=["fips", "state_name"])
    if len(miss):
        nearest = gpd.sjoin_nearest(
            miss.to_crs(EQUAL_AREA),
            states.to_crs(EQUAL_AREA).rename(columns={"state": "nearest_state"}),
            how="left",
            distance_col="dist_m",
        )
        miss = pd.DataFrame(nearest.drop(columns=["geometry", "index_right"]))
    else:
        miss = pd.DataFrame(columns=["name", "state", "lat", "lon", "nearest_state", "dist_m"])

    key_check = {
        "in_boundaries_not_stats": sorted(set(states["fips"]) - set(stats["fips"])),
        "in_stats_not_boundaries": sorted(set(stats["fips"]) - set(states["fips"])),
    }

    return pd.DataFrame(joined.drop(columns="geometry")), states, stats, miss, key_check


def quantile_classes(rate: pd.Series, k: int):
    """Quantile classification. Ties (e.g. many zeros) can merge classes, so
    the actual number of classes is returned along with the labels."""
    _, edges = pd.qcut(rate, q=k, retbins=True, duplicates="drop")
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([rate.min(), rate.max() + 1e-9])
    labels = [f"{lo:,.1f} – {hi:,.1f}" for lo, hi in zip(edges[:-1], edges[1:])]
    cls = pd.cut(rate, bins=edges, labels=labels, include_lowest=True)
    return cls, labels


points, states, stats, unmatched, key_check = load_data()

st.sidebar.header("Filter (Applies to All Tabs)")
yr_min, yr_max = int(points["year"].min()), int(points["year"].max())
year_range = st.sidebar.slider("Opened between", yr_min, yr_max, (yr_min, yr_max))
mask = points["year"].between(*year_range).fillna(False).astype(bool) | points["year"].isna()
filt = points[mask]

st.sidebar.metric("Stations shown", f"{len(filt):,}", f"of {len(points):,}", delta_color="off")

matched = filt[filt["fips"].notna()]
counts = matched.groupby("fips").agg(stations=("name", "size"), stalls=("stalls", "sum"))

by_state = (
    states.drop(columns="geometry")
    .merge(stats.drop(columns="state"), on="fips", how="left", validate="1:1")
    .merge(counts, on="fips", how="left")
    .fillna({"stations": 0, "stalls": 0})
)
by_state["rate"] = by_state["stations"] / by_state["pop_total"] * RATE_PER
by_state["class"], class_labels = quantile_classes(by_state["rate"], N_CLASSES)
n_classes = len(class_labels)

filt_unmatched = unmatched[unmatched.index.isin(filt.index)]

st.title("Tesla Superchargers by State")
st.caption(
    f"Showing stations opened {year_range[0]}–{year_range[1]}"
    + " plus undated stations."
)

tab_map, tab_joins, tab_trend = st.tabs(["🗺️ Map", "🔗 Joins", "📈 Trends"])

with tab_map:
    palette = px.colors.sequential.Blues[-n_classes:] if n_classes > 1 else ["#4292c6"]
    geojson = states.__geo_interface__

    fig = px.choropleth(
        by_state,
        geojson=geojson,
        locations="fips",
        featureidkey="properties.fips",
        color="class",
        category_orders={"class": class_labels},
        color_discrete_sequence=palette,
        hover_name="state",
        hover_data={"fips": False, "class": False, "stations": True,
                    "pop_total": ":,", "rate": ":.1f"},
        labels={"class": "Stations per 1M People", "rate": "per 1M",
                "pop_total": "population"},
    )
    fig.update_traces(marker_line_color="white", marker_line_width=0.5)

    fig.add_trace(go.Scattergeo(
        lon=matched["lon"], lat=matched["lat"],
        mode="markers",
        marker=dict(size=np.sqrt(matched["stalls"]) * 1.6, color="#d94801",
                    opacity=0.6, line=dict(width=0.3, color="white")),
        text=matched["name"] + "<br>" + matched["stalls"].astype(str) + " stalls",
        hoverinfo="text",
        name="Supercharger (size = stalls)",
    ))
    fig.update_geos(scope="usa", projection_type="albers usa", visible=False)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(title="Stations per 1M People", y=0.5))
    st.plotly_chart(fig)

    st.subheader("States Ranked by Rate")
    st.dataframe(
        by_state.sort_values("rate", ascending=False)[
            ["state", "stations", "stalls", "pop_total", "rate", "class"]
        ].rename(columns={"rate": "per 1M", "pop_total": "population"}),
        hide_index=True, width="stretch",
        column_config={"per 1M": st.column_config.NumberColumn(format="%.1f")},
    )

with tab_joins:
    st.subheader("1. Spatial Join: Stations → State Polygons")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stations in filter", f"{len(filt):,}")
    c2.metric("Matched a state", f"{len(matched):,}")
    c3.metric("Matched nothing", f"{len(filt_unmatched):,}")

    st.markdown(
        "Method: `geopandas.sjoin(points, states, predicate='within')`, both layers in "
        "EPSG:4326. A station that falls inside no polygon gets no state."
    )
    if len(filt_unmatched):
        show = filt_unmatched[["name", "state", "lat", "lon", "nearest_state", "dist_m"]].copy()
        show["dist_m"] = show["dist_m"].round(0)
        show = show.rename(columns={"state": "state in CSV", "dist_m": "metres to nearest polygon"})
        st.dataframe(show, hide_index=True, width="stretch")
        st.markdown(
            f"**Why they didn't match:** each sits on a coastline or state line, within "
            f"{filt_unmatched['dist_m'].max():,.0f} m of a state. The boundary file is "
            "simplified, so its edges fall slightly inside the real ones, and `within` "
            "doesn't count points exactly on a border (0 m). They are left out of the "
            "counts, not snapped to a state."
        )
    else:
        st.success("Every station in the current filter fell inside a state polygon.")

    st.subheader("2. Attribute Join: States → `state_stats.csv`")
    st.markdown(
        "Key: two-digit `fips`, read as **text** in both files so leading zeros "
        "(`01` Alabama … `09` Connecticut) survive. Joined with "
        "`merge(on='fips', validate='1:1')`."
    )
    k1, k2, k3 = st.columns(3)
    k1.metric("Boundary polygons", len(states))
    k2.metric("Rows in state_stats", len(stats))
    k3.metric("Unmatched keys", len(key_check["in_boundaries_not_stats"])
              + len(key_check["in_stats_not_boundaries"]))
    if key_check["in_boundaries_not_stats"] or key_check["in_stats_not_boundaries"]:
        st.warning(f"Unmatched FIPS: {key_check}")
    else:
        st.markdown("All 52 regions (50 states, DC, Puerto Rico) matched on `fips`, "
                    "so every polygon has a population for the rate.")

    st.markdown(
        f"The map shades each state by its **rate**, the number of stations per million "
        f"residents (stations ÷ `pop_total` × 1,000,000). Because raw counts mostly track "
        f"population, the rate shows which states are well served for their size. The "
        f"colours come from a **{n_classes}-class quantile classification**, which ranks "
        f"states by rate and splits them into {n_classes} groups of roughly equal size, "
        f"so the darkest blue marks the top fifth of states."
    )

with tab_trend:
    dated = filt[filt["year"].notna()]
    per_year = dated.groupby("year").size().rename("stations").reset_index()
    fig_y = px.bar(per_year, x="year", y="stations",
                   labels={"year": "Year opened", "stations": "Stations opened"})
    fig_y.update_traces(marker_color="#2171b5")
    fig_y.update_layout(height=350, margin=dict(t=30, b=10))
    st.subheader("Stations Opened per Year")
    st.plotly_chart(fig_y)
    st.caption(f"{filt['year'].isna().sum()} undated stations are in the map but can't appear on this time axis.")

    st.subheader("Top 15 States by Station Count")
    top = by_state.nlargest(15, "stations").sort_values("stations")
    fig_s = px.bar(top, x="stations", y="state", orientation="h",
                   hover_data={"rate": ":.1f"},
                   labels={"stations": "Stations", "state": "", "rate": "per 1M"})
    fig_s.update_traces(marker_color="#d94801")
    fig_s.update_layout(height=450, margin=dict(t=10, b=10))
    st.plotly_chart(fig_s)
    st.caption("These are raw station counts, not adjusted for population, so large "
               "states like California and Texas rank high largely because they have "
               "more people.")
