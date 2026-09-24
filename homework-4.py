"""
Electricity Usage: a time-series walkthrough
Resample -> ACF -> matched window -> decompose -> band

Run with:
    pip install -r requirements.txt
    streamlit run app.py

Uses the local Electric_Production.csv dataset.
"""
 
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

st.set_page_config(page_title="Electricity Usage: Time Series Walkthrough", layout="wide")

C_RAW = "#EB4BE0"
C_TREND = "#2E6E8E"
C_RED = "#d9534f"
C_GREEN = "#6aa84f"
C_GREY = "#8a929a"
C_PURPLE = "#b07aa1"

PERIOD_BY_RES = {"Monthly": 12, "Quarterly": 4, "Yearly": 1}
RULE_BY_RES = {"Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}


@st.cache_data
def load_data():
    data_path = Path(__file__).with_name("Electric_Production.csv")
    out = pd.read_csv(data_path).rename(columns={"DATE": "Date", "IPG2211A2N": "Usage"})
    out["Date"] = pd.to_datetime(out["Date"])
    out["Usage"] = pd.to_numeric(out["Usage"], errors="coerce")
    out = out.dropna(subset=["Date", "Usage"]).set_index("Date").sort_index()
    out = out.asfreq("MS")
    return out["Usage"]


monthly = load_data()

st.title("Electricity Usage: A Time-Series Walkthrough")

start, end = monthly.index.min(), monthly.index.max()
st.subheader("Filters")
resolution_col, date_col = st.columns([1, 3], gap="large")
with resolution_col:
    resolution = st.radio(
        "Resolution", list(PERIOD_BY_RES.keys()), index=0,
        horizontal=True,
        help="This drives the seasonal period, matched-window size, and decomposition period."
    )
with date_col:
    date_range = st.slider(
        "Date range", min_value=start.to_pydatetime(), max_value=end.to_pydatetime(),
        value=(start.to_pydatetime(), end.to_pydatetime()), format="YYYY-MM",
    )
period = PERIOD_BY_RES[resolution]
monthly_view = monthly.loc[date_range[0]:date_range[1]]


# ============================== Section 1: Resample ===========================

st.header("1 · Resample")

if resolution == "Monthly":
    resampled = monthly_view
else:
    rule = RULE_BY_RES[resolution]
    expected = {"Quarterly": 3, "Yearly": 12}[resolution]
    agg = monthly_view.resample(rule).agg(["mean", "count"])
    resampled = agg[agg["count"] == expected]["mean"]

fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Native (Monthly)", f"Resampled ({resolution})"),
                      shared_yaxes=True)
fig1.add_trace(go.Scatter(x=monthly_view.index, y=monthly_view.values, mode="lines",
                           line=dict(color=C_RAW, width=1.3), name="monthly"), row=1, col=1)
fig1.add_trace(go.Scatter(x=resampled.index, y=resampled.values, mode="lines+markers",
                           line=dict(color=C_TREND, width=2), name=resolution), row=1, col=2)
fig1.update_layout(height=340, showlegend=False, margin=dict(t=40, b=30))
st.plotly_chart(fig1, use_container_width=True)

if resolution != "Monthly":
    st.markdown(
        f"I used **{resolution.lower()}** averages here to smooth out some of the seasonal "
        f"variation. This makes the overall trend easier to see. I kept only complete "
        f"{'quarters' if resolution == 'Quarterly' else 'years'} "
        f"(`count == {expected}`), so an incomplete period at the end does not affect the trend."
    )


# ============================== Section 2: ACF =================================

st.header("2 · ACF")

diffed = monthly_view.diff().dropna()
max_lags = min(3 * max(period, 4), len(diffed) // 2 - 1)
if resolution == "Yearly":
    st.info("Seasonality doesn't apply at yearly resolution (one point per cycle). "
             "I am showing the ACF using the monthly changes instead, so the seasonal pattern "
             "can still be seen.")
    diffed = monthly.diff().dropna()
    max_lags = 36

acf_vals, confint = acf(diffed, nlags=max_lags, alpha=0.05, fft=True)
lags = np.arange(len(acf_vals))
ci_half = confint[:, 1] - acf_vals

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=np.concatenate([lags, lags[::-1]]),
                           y=np.concatenate([ci_half, -ci_half[::-1]]),
                           fill="toself", fillcolor="rgba(46,110,142,0.15)",
                           line=dict(width=0), name="95% band", hoverinfo="skip"))
for x, y in zip(lags, acf_vals):
    fig2.add_trace(go.Scatter(x=[x, x], y=[0, y], mode="lines",
                               line=dict(color=C_TREND, width=1.5), showlegend=False, hoverinfo="skip"))
fig2.add_trace(go.Scatter(x=lags, y=acf_vals, mode="markers", marker=dict(color=C_TREND, size=6),
                           name="ACF", showlegend=False))
fig2.add_hline(y=0, line_color="#999", line_width=1)
fig2.update_layout(height=320, margin=dict(t=20, b=30), xaxis_title="Lag", yaxis_title="Autocorrelation")
st.plotly_chart(fig2, use_container_width=True)

candidate_lags = lags[1:][np.abs(acf_vals[1:]) > ci_half[1:]]
detected_period = None
for p in (12, 4, 6, 3):
    if p in candidate_lags:
        detected_period = p
        break
if detected_period:
    st.markdown(
        f"The ACF has noticeable spikes about every **{detected_period} lags**, with the signs "
        f"alternating between them. This suggests a repeating yearly pattern, which is why I "
        f"use that cycle length for the rolling window below."
    )
else:
    st.markdown("No lag clearly stands out at this resolution. The seasonal pattern may be weak "
                "or already smoothed out.")


# ============================== Section 3: Matched window ======================

st.header("3 · Matched Window")

window = period if period > 1 else 12
rolled = monthly_view.rolling(window).mean()

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=monthly_view.index, y=monthly_view.values, mode="lines",
                           line=dict(color=C_GREY, width=1), opacity=0.5, name="monthly"))
fig3.add_trace(go.Scatter(x=rolled.index, y=rolled.values, mode="lines",
                           line=dict(color=C_TREND, width=2.2), name=f"w={window} (one cycle)"))
fig3.update_layout(height=340, margin=dict(t=20, b=30), legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig3, use_container_width=True)

st.markdown(
    f"The **{window}-period rolling average** covers one full seasonal cycle. It smooths out "
    f"the repeating highs and lows while keeping the general trend visible. Unlike resampling, "
    f"this keeps a value for each original time period."
)

with st.expander("Extra: compare a few window sizes"):
    small_w = max(2, window // 4)
    big_w = window * 5
    fig3b = make_subplots(rows=1, cols=2, subplot_titles=(
        f"{small_w} Keeps the Season · {window} Removes It · {big_w} Flattens Real Turns",
        "Rolling Std: Size of the Seasonal Swing"))
    fig3b.add_trace(go.Scatter(x=monthly_view.index, y=monthly_view.values, mode="lines",
                                line=dict(color="#bbbbbb", width=1), name="monthly"), row=1, col=1)
    for w, c in [(small_w, C_RED), (window, C_TREND), (big_w, C_GREEN)]:
        r = monthly_view.rolling(w).mean()
        fig3b.add_trace(go.Scatter(x=r.index, y=r.values, mode="lines",
                                    line=dict(color=c, width=1.8), name=f"w={w}"), row=1, col=1)
    rstd = monthly_view.rolling(window).std()
    fig3b.add_trace(go.Scatter(x=rstd.index, y=rstd.values, mode="lines",
                                line=dict(color=C_PURPLE, width=1.8), showlegend=False), row=1, col=2)
    fig3b.update_layout(height=320, margin=dict(t=40, b=30))
    st.plotly_chart(fig3b, use_container_width=True)
    st.markdown(
        f"The short window (**{small_w}**) still shows the seasonal pattern. The matched window "
        f"(**{window}**) smooths it out, while the long window (**{big_w}**) can smooth away "
        f"real changes in the trend too. The chart on the right shows how much the values vary "
        f"over each rolling window."
    )


# ============================== Section 4: Decompose ============================

st.header("4 · Decompose")

if period == 1:
    st.info("Seasonal decomposition needs sub-annual data, so it is not meaningful at yearly resolution. "
             "Choose Monthly or Quarterly in the filters above to see this section.")
else:
    model_choice = st.radio("Model", ["Compare both", "Additive", "Multiplicative"], horizontal=True)

    add = seasonal_decompose(monthly_view.dropna(), period=period, model="additive")
    mul = seasonal_decompose(monthly_view.dropna(), period=period, model="multiplicative")

    def decompose_fig(res, label, color):
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                             subplot_titles=("Observed", "Trend", "Seasonal", "Resid"))
        fig.add_trace(go.Scatter(x=res.observed.index, y=res.observed.values, line=dict(color=color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=res.trend.index, y=res.trend.values, line=dict(color=color)), row=2, col=1)
        fig.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal.values, line=dict(color=color)), row=3, col=1)
        fig.add_trace(go.Scatter(x=res.resid.index, y=res.resid.values, mode="markers",
                                  marker=dict(color=color, size=4)), row=4, col=1)
        fig.update_layout(height=560, showlegend=False, margin=dict(t=40, b=20), title=label)
        return fig

    if model_choice in ("Compare both", "Additive"):
        st.plotly_chart(decompose_fig(add, "Additive decomposition", C_TREND), use_container_width=True)
    if model_choice in ("Compare both", "Multiplicative"):
        st.plotly_chart(decompose_fig(mul, "Multiplicative decomposition", C_GREEN), use_container_width=True)

    add_resid = add.resid.dropna()
    mul_resid = mul.resid.dropna()
    level = monthly_view.dropna().mean()
    add_mean, add_std = add_resid.mean(), add_resid.std()
    mul_mean, mul_std = mul_resid.mean(), mul_resid.std()
    add_rel = add_std / level * 100
    mul_rel = mul_std * 100

    fig4 = make_subplots(rows=1, cols=2, subplot_titles=("Additive Residual", "Multiplicative Residual"))
    fig4.add_trace(go.Scatter(x=add_resid.index, y=add_resid.values, line=dict(color=C_RED, width=1)), row=1, col=1)
    fig4.add_hline(y=0, line_color="#999", line_width=1, row=1, col=1)
    fig4.add_trace(go.Scatter(x=mul_resid.index, y=mul_resid.values, line=dict(color=C_GREEN, width=1)), row=1, col=2)
    fig4.add_hline(y=1, line_color="#999", line_width=1, row=1, col=2)
    fig4.update_layout(height=280, showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig4, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Additive residual mean", f"{add_mean:.4f}")
    c1.metric("Additive residual std", f"{add_std:.4f}")
    c2.metric("Multiplicative residual mean", f"{mul_mean:.4f}")
    c2.metric("Multiplicative residual std", f"{mul_std:.4f}")

    better = "additive" if add_rel < mul_rel else "multiplicative"
    st.markdown(
        f"The residual means are close to their expected values: 0 for the additive model and 1 "
        f"for the multiplicative model. The residual spread is more useful for comparing them. "
        f"Here, the additive residual is about **{add_rel:.2f}%** of the average usage, while "
        f"the multiplicative residual spread is **{mul_rel:.2f}%**. Based on that comparison, "
        f"the **{better}** model fits a little better."
    )


# ============================== Section 5: Band =================================

st.header("5 · Band")

band_type = st.radio("Uncertainty band", ["Rolling ±2σ", "Bootstrap 95% CI (extra credit)"], horizontal=True)

if band_type == "Rolling ±2σ":
    roll = monthly_view.rolling(window)
    mean, std = roll.mean(), roll.std()
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=mean.index, y=mean.values, mode="lines",
                               line=dict(color=C_TREND, width=2), name=f"{window}-period mean"))
    fig5.add_trace(go.Scatter(x=mean.index, y=(mean + 2 * std).values, mode="lines",
                               line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig5.add_trace(go.Scatter(x=mean.index, y=(mean - 2 * std).values, mode="lines",
                               line=dict(width=0), fill="tonexty", fillcolor="rgba(46,110,142,0.2)",
                               name="±2σ"))
    fig5.update_layout(height=340, margin=dict(t=20, b=30), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown(
        "This band shows how far the individual values spread around the rolling average. It is "
        "showing variation in the data, not uncertainty about the average. A wider band means "
        "the values are more spread out during that part of the series."
    )
else:
    changes = monthly_view.diff().dropna().values
    if len(changes) < 10:
        st.warning("Not enough points in the selected date range for a stable bootstrap.")
    else:
        rng = np.random.default_rng(0)
        boots = np.array([rng.choice(changes, len(changes), replace=True).mean() for _ in range(2000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        fig5 = go.Figure()
        fig5.add_trace(go.Histogram(x=boots, nbinsx=40, marker_color=C_TREND, opacity=0.85))
        fig5.add_vline(x=lo, line_color=C_RED, line_width=2)
        fig5.add_vline(x=hi, line_color=C_RED, line_width=2)
        fig5.update_layout(height=320, margin=dict(t=20, b=30),
                            title=f"Bootstrap Mean Period-over-Period Change, 95% CI [{lo:.3f}, {hi:.3f}]")
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown(
            "This uses a different type of uncertainty. I resampled the observed changes 2,000 "
            "times to estimate the confidence interval for the **average** change. This describes "
            "the average trend, not the expected range of every individual value."
        )


# ============================== Temporal-honesty note ===========================

st.header("Temporal-Honesty Note")
st.markdown(
    f"**Resolution:** I am viewing the data at **{resolution.lower()}** resolution. Using a "
    f"broader resolution makes the trend easier to see, but it also hides some of the "
    f"month-to-month variation.\n\n"
    f"**Partial periods:** I only keep complete periods when resampling "
    f"(`count == {12 if resolution == 'Yearly' else 3 if resolution == 'Quarterly' else 1}`). "
    f"This prevents an incomplete period at the end from changing the last point.\n\n"
    f"**Axis:** the rolling-window and band charts do not start at zero. I chose that because a "
    f"zero-based axis would make the trend and seasonal changes harder to see. The tradeoff is "
    f"that the size of a change depends partly on the zoom level."
)
