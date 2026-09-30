import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="One Question, Three Encodings",
    page_icon="📊",
    layout="wide",
)

GROUP_LABELS = ["Lowest GDP", "Lower-Middle GDP", "Upper-Middle GDP", "Highest GDP"]
CONTINENT_COLORS = {
    "Africa": "#e4572e",
    "Americas": "#29b09d",
    "Asia": "#f2b134",
    "Europe": "#4c78a8",
    "Oceania": "#9467bd",
}


@st.cache_data(show_spinner="Loading Gapminder data...")
def load_data() -> pd.DataFrame:
    """Load once; GDP quartiles are fixed on the full dataset so group
    labels keep the same meaning no matter which filters are applied."""
    df = px.data.gapminder()
    df["GDP Group"] = pd.qcut(df["gdpPercap"], q=4, labels=GROUP_LABELS)
    return df


@st.cache_data
def filter_data(year_range: tuple, continents: tuple) -> pd.DataFrame:
    df = load_data()
    mask = df["year"].between(*year_range) & df["continent"].isin(continents)
    return df.loc[mask]


@st.cache_data
def summarize_groups(year_range: tuple, continents: tuple, agg: str) -> pd.DataFrame:
    df = filter_data(year_range, continents)
    return (
        df.groupby("GDP Group", observed=True)["lifeExp"]
        .agg(agg)
        .reset_index()
    )


@st.cache_data
def summarize_groups_by_continent(
    year_range: tuple, continents: tuple, agg: str
) -> pd.DataFrame:
    df = filter_data(year_range, continents)
    return (
        df.groupby(["GDP Group", "continent"], observed=True)["lifeExp"]
        .agg(agg)
        .reset_index()
    )


@st.cache_data
def fit_trend(x: tuple, y: tuple, log_x: bool) -> tuple:
    """Least-squares trend line (in log10 space if the x axis is logarithmic)."""
    x_arr, y_arr = np.asarray(x), np.asarray(y)
    x_fit = np.log10(x_arr) if log_x else x_arr
    slope, intercept = np.polyfit(x_fit, y_arr, 1)
    grid = np.linspace(x_fit.min(), x_fit.max(), 100)
    y_line = slope * grid + intercept
    x_line = 10**grid if log_x else grid
    return x_line.tolist(), y_line.tolist(), float(np.corrcoef(x_fit, y_arr)[0, 1])



full = load_data()
years = sorted(full["year"].unique())

with st.sidebar:
    st.header("Controls")

    year_range = st.select_slider(
        "Year range",
        options=years,
        value=(years[0], years[-1]),
        help="Gapminder records one observation every five years.",
    )
    continents = st.multiselect(
        "Continents",
        options=sorted(full["continent"].unique()),
        default=sorted(full["continent"].unique()),
    )
    agg = st.selectbox(
        "Aggregate for Encodings B and C",
        options=["mean", "median"],
        format_func=str.capitalize,
    )

    st.divider()
    st.subheader("Scatter options")
    log_x = st.checkbox("Log scale for GDP per capita", value=True)
    show_trend = st.checkbox("Show trend line", value=True)
    opacity = st.slider("Point opacity", 0.1, 1.0, 0.6, 0.05)

    st.divider()
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.rerun()

if not continents:
    st.warning("Select at least one continent in the sidebar.")
    st.stop()

year_key, cont_key = tuple(year_range), tuple(sorted(continents))
df = filter_data(year_key, cont_key)


st.title("One Question, Three Encodings")
st.caption("How is GDP per capita related to life expectancy across countries?")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Observations", f"{len(df):,}")
k2.metric("Countries", df["country"].nunique())
k3.metric("Avg life expectancy", f"{df['lifeExp'].mean():.1f} yrs")
k4.metric("Median GDP per capita", f"${df['gdpPercap'].median():,.0f}")

tab_enc, tab_cmp, tab_gestalt, tab_data = st.tabs(
    ["Encodings", "Comparison", "Gestalt principles", "Data"]
)

with tab_enc:
    left, middle, right = st.columns(3)

    with left:
        st.subheader("Encoding A: Position")

        fig_a = px.scatter(
            df,
            x="gdpPercap",
            y="lifeExp",
            opacity=opacity,
            hover_name="country",
            hover_data={"year": True, "continent": True, "gdpPercap": ":,.0f"},
            labels={"gdpPercap": "GDP per Capita", "lifeExp": "Life Expectancy"},
            color_discrete_sequence=["#4c78a8"],
            log_x=log_x,
            title="GDP vs Life Expectancy",
        )
        if show_trend and len(df) > 2:
            x_line, y_line, r = fit_trend(
                tuple(df["gdpPercap"]), tuple(df["lifeExp"]), log_x
            )
            fig_a.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"Trend (r = {r:.2f})",
                    line=dict(color="#e4572e", width=3),
                )
            )
            fig_a.update_layout(showlegend=True, legend=dict(y=0.02, x=0.98, xanchor="right"))
        fig_a.update_layout(height=450, margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig_a, use_container_width=True)

        st.caption(
            "Channel: Position, Ranking: 1. "
            "GDP per capita and life expectancy are quantitative variables, "
            "so position on common scales provides the most perceptually accurate "
            "way to compare values and see their relationship."
        )

    with middle:
        st.subheader("Encoding B: Length")

        summary_b = summarize_groups(year_key, cont_key, agg)
        fig_b = px.bar(
            summary_b,
            x="GDP Group",
            y="lifeExp",
            text="lifeExp",
            category_orders={"GDP Group": GROUP_LABELS},
            labels={
                "GDP Group": "GDP per Capita Group",
                "lifeExp": f"{agg.capitalize()} Life Expectancy",
            },
            color_discrete_sequence=["steelblue"],
            title=f"{agg.capitalize()} Life Expectancy by GDP Group",
        )
        fig_b.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_b.update_yaxes(range=[0, max(85, summary_b["lifeExp"].max() * 1.1)])
        fig_b.update_layout(height=450, margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig_b, use_container_width=True)

        st.caption(
            "Channel: Length, Ranking: 3. "
            "Bar length represents life expectancy within each GDP group. "
            "Length is less perceptually accurate than position on a common scale, "
            "but it makes differences between the group values easy to compare."
        )

    with right:
        st.subheader("Encoding C: Color")

        summary_c = summarize_groups_by_continent(year_key, cont_key, agg)
        fig_c = px.line(
            summary_c,
            x="GDP Group",
            y="lifeExp",
            color="continent",
            markers=True,
            category_orders={"GDP Group": GROUP_LABELS},
            color_discrete_map=CONTINENT_COLORS,
            labels={
                "GDP Group": "GDP per Capita Group",
                "lifeExp": f"{agg.capitalize()} Life Expectancy",
                "continent": "Continent",
            },
            title="Life Expectancy Across GDP Groups",
        )
        fig_c.update_traces(line=dict(width=3), marker=dict(size=9))
        fig_c.update_layout(
            height=450,
            hovermode="x unified",
            margin=dict(t=50, b=20, l=10, r=10),
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig_c, use_container_width=True)

        st.caption(
            "Channel: Color, Ranking: 6. "
            "Color distinguishes continents and makes group differences easy to "
            "identify. Color is less perceptually accurate for quantitative comparison "
            "than position, but it effectively separates categorical groups."
        )


with tab_cmp:
    st.subheader("Comparison of the Three Encodings")
    st.write(
        "Encoding A uses position, the most perceptually accurate channel, to show "
        "the GDP–life expectancy relationship. Encoding B uses length to compare "
        "average life expectancy across GDP groups. Encoding C uses color to distinguish "
        "continents and reveal differences in the relationship across groups. Together, "
        "the three encodings show how different visual channels highlight different patterns."
    )

    ranking = pd.DataFrame(
        {
            "Encoding": ["A: Position", "B: Length", "C: Color"],
            "Channel ranking (1 = most accurate)": [1, 3, 6],
        }
    )
    fig_rank = px.bar(
        ranking,
        x="Channel ranking (1 = most accurate)",
        y="Encoding",
        orientation="h",
        text="Channel ranking (1 = most accurate)",
        color="Encoding",
        color_discrete_sequence=["#4c78a8", "steelblue", "#f2b134"],
    )
    fig_rank.update_yaxes(autorange="reversed")
    fig_rank.update_layout(showlegend=False, height=260, margin=dict(t=10, b=10))
    st.plotly_chart(fig_rank, use_container_width=True)
    st.caption("Lower rank number means the eye decodes the value more accurately.")

with tab_gestalt:
    st.subheader("Gestalt Principles Used")
    g1, g2, g3 = st.columns(3)
    g1.markdown(
        "**Proximity (A)**  \nNearby points are perceived as related and "
        "help reveal patterns."
    )
    g2.markdown(
        "**Similarity (B)**  \nThe bars share the same visual form and can be "
        "compared as a group."
    )
    g3.markdown(
        "**Continuity (C)**  \nConnected lines guide the eye across GDP groups and "
        "show changes in life expectancy."
    )

----------
with tab_data:
    st.subheader("Filtered data")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="gapminder_filtered.csv",
        mime="text/csv",
    )