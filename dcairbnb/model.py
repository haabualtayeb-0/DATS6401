"""Revenue projector (Part 5 of midterm.ipynb): a listing model and a demand model, combined.

Training takes about a minute, so the results are saved to artifacts/projector.joblib and reused until a
data file changes. Run `python build_model.py` to build them ahead of time.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from dcairbnb.data import DATA_DIR, Bundle, data_fingerprint, load
from dcairbnb.style import ROOM_ORDER

ARTIFACT_PATH = DATA_DIR / "artifacts" / "projector.joblib"
ARTIFACT_VERSION = 1   # bump when the modeling code changes

HOUSEHOLD_SIZES = list(range(1, 9))  # guests the listing sleeps; 8 means 8+
FEATURE_LABELS = {
    "household": "Guests it sleeps", "bedrooms": "Bedrooms", "baths": "Bathrooms", "shared_bath": "Shared bathroom",
    "neighbourhood_cleansed": "Neighborhood", "room_type": "Room type", "amenity_count": "Number of amenities",
    "review_scores_rating": "Guest rating", "is_superhost": "Superhost", "years_listed": "Years since first review",
    "calculated_host_listings_count": "Host's DC listings", "minimum_nights": "Minimum stay", "availability_365": "Open days, next year",
}
FEATURES = list(FEATURE_LABELS)
BASIC = ["household", "neighbourhood_cleansed", "room_type"]
PROJECTOR_INPUTS = ["Guests it sleeps", "Neighborhood"]


def gbm(features, **kw):
    """Gradient-boosted trees; more guests or bedrooms can never lower the prediction."""
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype", max_iter=400, learning_rate=0.05, min_samples_leaf=20,
        l2_regularization=1.0, random_state=0,
        monotonic_cst={f: 1 for f in ["household", "bedrooms"] if f in features}, **kw)


def training_data(b: Bundle):
    """Short-term, priced listings reviewed in the past year, with the model features."""
    df = b.df
    active = (b.scrape_date - df["last_review"]).dt.days <= 365
    mdl = df[(df["minimum_nights"] < 30) & df["price"].notna() & active].copy()
    mdl = mdl[mdl["price"].between(*mdl["price"].quantile([0.01, 0.99]))]

    # `bathrooms` is empty in this scrape, so parse the text version ("1.5 shared baths", "Half-bath")
    baths = pd.to_numeric(mdl["bathrooms_text"].str.extract(r"([\d.]+)")[0], errors="coerce")
    mdl["baths"] = baths.where(~mdl["bathrooms_text"].str.contains("half", case=False, na=False), 0.5)
    mdl["shared_bath"] = mdl["bathrooms_text"].str.contains("shared", case=False, na=False).astype(int)
    mdl["is_superhost"] = mdl["host_is_superhost"].eq("t").astype(int)
    mdl["amenity_count"] = mdl["amenity_list"].str.len()
    mdl["years_listed"] = (b.scrape_date - mdl["first_review"]).dt.days / 365
    mdl["household"] = mdl["accommodates"].clip(upper=8)
    mdl["neighbourhood_cleansed"] = pd.Categorical(mdl["neighbourhood_cleansed"], categories=b.neighbourhoods)
    mdl["room_type"] = pd.Categorical(mdl["room_type"], categories=ROOM_ORDER)
    mdl["log_revenue"] = np.log(mdl["estimated_revenue_l365d"])
    return mdl


def _conformal_pad(miss, coverage=0.8):
    return np.quantile(miss, min(1, coverage * (1 + 1 / len(miss))))


def _listing_model(mdl, neighbourhoods):
    """Cross-validated comparison, calibrated 80% range, feature importance and the household × area grid."""
    kf = KFold(5, shuffle=True, random_state=42)
    y = mdl["log_revenue"]
    candidates = {
        "Median of all listings": None,
        "Linear: size, area, room type": (make_pipeline(OneHotEncoder(handle_unknown="ignore"), Ridge(alpha=1.0)), mdl[BASIC].astype(str)),
        "Boosted trees: size, area, room type": (gbm(BASIC), mdl[BASIC]),
        "Boosted trees: full listing profile": (gbm(FEATURES), mdl[FEATURES]),
    }
    scores = []
    for name, spec in candidates.items():
        pred = (pd.Series(y.median(), index=y.index) if spec is None
                else pd.Series(cross_val_predict(spec[0], spec[1], y, cv=kf), index=y.index))
        err = (np.exp(pred) - mdl["estimated_revenue_l365d"]).abs()
        scores.append(dict(model=name, r2_log=1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                           median_error=err.median(), within_25pct=(err / mdl["estimated_revenue_l365d"] <= 0.25).mean()))

    # 80% range: quantile models, widened by conformalized quantile regression
    oof = {q: pd.Series(cross_val_predict(gbm(FEATURES, loss="quantile", quantile=q), mdl[FEATURES], y, cv=kf), index=y.index)
           for q in (0.1, 0.9)}
    miss = np.maximum(oof[0.1] - y, y - oof[0.9])
    calib = np.random.default_rng(0).random(len(y)) < 0.5
    pad_half = _conformal_pad(miss[calib])
    inside = lambda lo, hi: float(((y >= lo) & (y <= hi))[~calib].mean())
    coverage = dict(raw=inside(oof[0.1], oof[0.9]), calibrated=inside(oof[0.1] - pad_half, oof[0.9] + pad_half))
    pad = _conformal_pad(miss)

    models = {"low": gbm(FEATURES, loss="quantile", quantile=0.1), "typical": gbm(FEATURES),
              "high": gbm(FEATURES, loss="quantile", quantile=0.9)}
    for m in models.values():
        m.fit(mdl[FEATURES], y)

    imp = permutation_importance(models["typical"], mdl[FEATURES], y, n_repeats=10, random_state=0)
    importance = (pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in FEATURES],
                                "importance": imp.importances_mean, "sd": imp.importances_std})
                    .sort_values("importance"))

    # Every household × neighborhood: move real comparables of that size into the area and predict
    parts = []
    areas = pd.DataFrame({"neighbourhood_cleansed": pd.Categorical(neighbourhoods, categories=neighbourhoods)})
    for size in HOUSEHOLD_SIZES:
        comps = mdl.loc[mdl["household"] == size, FEATURES].drop(columns="neighbourhood_cleansed")
        moved = comps.merge(areas, how="cross")[FEATURES]
        preds = pd.DataFrame({k: m.predict(moved) for k, m in models.items()})
        preds["low"] -= pad
        preds["high"] += pad
        parts.append(np.exp(preds).assign(household=size, comparables=len(comps),
                                          neighbourhood=moved["neighbourhood_cleansed"].astype(str).values))
    local = mdl.groupby(["household", "neighbourhood_cleansed"], observed=False).size().rename("listings_like_this")
    grid = (pd.concat(parts).groupby(["household", "neighbourhood"])
              .agg(low=("low", "median"), typical=("typical", "median"), high=("high", "median"),
                   comparables=("comparables", "first"))
              .join(local.rename_axis(["household", "neighbourhood"])))
    grid["low"] = grid[["low", "typical"]].min(axis=1)
    grid["high"] = grid[["high", "typical"]].max(axis=1)
    return dict(scores=pd.DataFrame(scores), coverage=coverage, importance=importance, grid=grid, n_train=len(mdl))


def _fit_demand(series):
    return ExponentialSmoothing(np.log(series), trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit()


def _demand_model(b: Bundle):
    """Monthly reviews per active listing, a damped Holt-Winters forecast, and a yearly demand index."""
    df = b.df
    first_m = df["first_review"].dt.to_period("M")
    last_full_month = (b.scrape_date - pd.DateOffset(months=1)).to_period("M")
    last_m = df["last_review"].dt.to_period("M").mask((b.scrape_date - df["last_review"]).dt.days <= 365, last_full_month)

    months = pd.period_range("2015-01", last_full_month, freq="M")
    live = pd.Series([((first_m <= p) & (last_m >= p)).sum() for p in months], index=months)
    monthly_reviews = b.reviews["date"].dt.to_period("M").value_counts().reindex(months, fill_value=0)
    demand = (monthly_reviews / live).rename("reviews_per_listing")
    demand.index = demand.index.to_timestamp()

    backtest = []
    for cutoff in ["2024-02", "2024-08", "2025-02"]:
        hist = demand["2022-01":cutoff]
        actual = demand[demand.index > hist.index[-1]].iloc[:18]
        hw = np.exp(_fit_demand(hist).forecast(len(actual)))
        naive = np.tile(hist.iloc[-12:].values, 2)[:len(actual)]
        backtest.append(dict(cutoff=cutoff, holt_winters=float((abs(hw.values - actual.values) / actual).mean()),
                             same_month_last_year=float((abs(naive - actual.values) / actual).mean())))

    fit = _fit_demand(demand["2022-01":])
    horizon = (pd.Period("2030-12", "M") - last_full_month).n
    paths = np.exp(fit.simulate(horizon, repetitions=2000, error="add", anchor="end", random_state=0))
    band = paths.quantile([0.1, 0.5, 0.9], axis=1).T.set_axis(["low", "typical", "high"], axis=1)

    base_year = b.scrape_date.year
    yearly = paths.groupby(paths.index.year).mean().loc[base_year + 1:] / demand.iloc[-12:].mean()
    index = yearly.quantile([0.1, 0.5, 0.9], axis=1).T.set_axis(["low", "typical", "high"], axis=1)
    index.loc[base_year] = 1.0
    return dict(demand=demand, band=band, backtest=pd.DataFrame(backtest), demand_index=index.sort_index(),
                damping=float(fit.params["damping_trend"]), base_year=base_year)


def build_artifacts(b: Bundle):
    mdl = training_data(b)
    return dict(version=ARTIFACT_VERSION, fingerprint=data_fingerprint(),
                **_listing_model(mdl, b.neighbourhoods), **_demand_model(b))


@st.cache_resource(show_spinner="Training the revenue model (first run only, about a minute)…")
def load_artifacts():
    if ARTIFACT_PATH.exists():
        art = joblib.load(ARTIFACT_PATH)
        if art.get("version") == ARTIFACT_VERSION and art.get("fingerprint") == data_fingerprint():
            return art
    art = build_artifacts(load())
    try:
        ARTIFACT_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(art, ARTIFACT_PATH)
    except OSError:
        pass   # read-only deploys still work; the model is just retrained per server process
    return art


def project_revenue(art, household, neighbourhood, year):
    """Projected annual revenue (2026 dollars) for a short-term listing: typical value plus an 80% range."""
    base = art["grid"].loc[(min(household, 8), neighbourhood)]
    d = art["demand_index"].loc[year]
    typical = base["typical"] * d["typical"]
    # Listing range and demand range combine on a log scale, assuming they're independent
    down = np.hypot(np.log(base["typical"] / base["low"]), np.log(d["typical"] / d["low"]))
    up = np.hypot(np.log(base["high"] / base["typical"]), np.log(d["high"] / d["typical"]))
    return dict(household=household, neighbourhood=neighbourhood, year=year,
                low=typical * np.exp(-down), typical=typical, high=typical * np.exp(up),
                demand_index=d["typical"], listings_like_this=int(base["listings_like_this"]))


def project_all(art, year):
    """Every household × neighborhood for one year, as a flat table."""
    return pd.DataFrame([project_revenue(art, h, n, year) for h, n in art["grid"].index])
