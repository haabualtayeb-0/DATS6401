import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = px.data.gapminder()


print(df.head())
print(df.tail())
print(df.shape)


st.title("One Question, Three Encodings")
st.caption("How is GDP per capita related to life expectancy across countries?")

left, middle, right = st.columns(3)


with left:
    st.subheader("Encoding A: Position")

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.scatterplot(
        data=df,
        x="gdpPercap",
        y="lifeExp",
        alpha=0.6,
        ax=ax
    )

    ax.set_title("GDP vs Life Expectancy")
    ax.set_xlabel("GDP per Capita")
    ax.set_ylabel("Life Expectancy")

    st.pyplot(fig)

    st.caption(
        "Channel: Position, Ranking: 1. "
        "GDP per capita and life expectancy are quantitative variables, "
        "so position on common scales provides the most perceptually accurate "
        "way to compare values and see their relationship."
    )


with middle:
    st.subheader("Encoding B: Length")

    plot_df = df.copy()

    plot_df["GDP Group"] = pd.qcut(
        plot_df["gdpPercap"],
        q=4,
        labels=[
            "Lowest GDP",
            "Lower-Middle GDP",
            "Upper-Middle GDP",
            "Highest GDP"
        ]
    )

    summary = (
        plot_df
        .groupby("GDP Group", observed=True)["lifeExp"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.barplot(
        data=summary,
        x="GDP Group",
        y="lifeExp",
        color="steelblue",
        ax=ax
    )

    ax.set_title("Average Life Expectancy by GDP Group")
    ax.set_xlabel("GDP per Capita Group")
    ax.set_ylabel("Average Life Expectancy")

    plt.xticks(rotation=30, ha="right")

    st.pyplot(fig)

    st.caption(
        "Channel: Length, Ranking: 3. "
        "Bar length represents average life expectancy within each GDP group. "
        "Length is less perceptually accurate than position on a common scale, "
        "but it makes differences between the group averages easy to compare."
    )


with right:
    st.subheader("Encoding C: Color")

    plot_df = df.copy()

    plot_df["GDP Bin"] = pd.qcut(
        plot_df["gdpPercap"],
        q=8,
        duplicates="drop"
    )

    plot_df["Life Expectancy Bin"] = pd.qcut(
        plot_df["lifeExp"],
        q=8,
        duplicates="drop"
    )

    heatmap_data = pd.crosstab(
        plot_df["Life Expectancy Bin"],
        plot_df["GDP Bin"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        heatmap_data,
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Number of Countries"}
    )

    ax.set_title("GDP vs Life Expectancy")
    ax.set_xlabel("GDP per Capita")
    ax.set_ylabel("Life Expectancy")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    st.pyplot(fig)

    st.caption(
        "Channel: Color, Ranking: 6. "
        "Color intensity represents the number of countries in each GDP and "
        "life-expectancy range. Color is less perceptually accurate for precise "
        "quantitative comparison, but it makes concentrations and overall patterns "
        "in the distribution easier to see."
    )



st.subheader("Comparison of the Three Encodings")

st.write(
    "Encoding A uses position, the most perceptually accurate channel, to show "
    "the GDP–life expectancy relationship. Encoding B uses length to compare "
    "average life expectancy across GDP groups. Encoding C uses color to show "
    "where countries are concentrated. Each encoding highlights a different "
    "pattern while demonstrating how perceptual accuracy affects interpretation."
)
