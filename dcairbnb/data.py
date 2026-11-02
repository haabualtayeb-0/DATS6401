"""Load and clean the Inside Airbnb files, using the same steps as Part 0 of midterm.ipynb."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from dcairbnb.style import ACT_ORDER, LIC_ORDER, PORTFOLIO_BINS, PORTFOLIO_LABELS, STAY_ORDER, TENURE_ORDER

DATA_DIR = Path(__file__).resolve().parent.parent
DATA_FILES = ["listings.csv", "reviews.csv", "neighbourhoods.geojson"]


@dataclass(frozen=True)
class Bundle:
    df: pd.DataFrame          # one row per listing, cleaned, with derived columns
    hosts: pd.DataFrame       # one row per host
    reviews: pd.DataFrame     # listing_id, date
    daily: pd.DataFrame       # share of listings available on each calendar date
    geo: dict                 # neighborhood polygons
    neighbourhoods: list      # neighborhood names, in geojson order
    scrape_date: pd.Timestamp


def _activity(last_review, scrape_date):
    if pd.isna(last_review):
        return "Never reviewed"
    days = (scrape_date - last_review).days
    if days <= 90:
        return "Reviewed in last 3 months"
    if days <= 365:
        return "Reviewed 3–12 months ago"
    return "Last review over a year ago"


def _license_type(val):
    """Group DC license strings: hosted (primary residence), unhosted (vacation rental), exempt, missing."""
    if pd.isna(val):
        return "Missing"
    v = val.lower()
    for prefix, label in [("hosted", "Hosted license"), ("unhosted", "Unhosted license"), ("exempt", "Exempt")]:
        if v.startswith(prefix):
            return label
    return "Other"


@st.cache_resource(show_spinner="Loading the Inside Airbnb data…")
def load() -> Bundle:
    """Read and clean everything once per server process. Treat the returned frames as read-only."""
    with open(DATA_DIR / "neighbourhoods.geojson") as f:
        geo = json.load(f)
    neighbourhoods = [feat["properties"]["neighbourhood"] for feat in geo["features"]]

    df = pd.read_csv(DATA_DIR / "listings.csv")
    df["price"] = pd.to_numeric(df["price"].str.replace(r"[$,]", "", regex=True), errors="coerce")
    for col in ["host_since", "first_review", "last_review", "last_scraped"]:
        df[col] = pd.to_datetime(df[col])
    scrape_date = df["last_scraped"].max()

    df["amenity_list"] = df["amenities"].apply(json.loads)
    df["superhost"] = df["host_is_superhost"].map({"t": "Superhost", "f": "Not superhost"})
    df["host_tenure"] = pd.cut(df["hosts_time_as_host_years"], bins=[-1, 1, 4, 9, np.inf], labels=TENURE_ORDER)
    df["stay_type"] = pd.cut(df["minimum_nights"], bins=[0, 2, 6, 29, np.inf], labels=STAY_ORDER)
    df["review_activity"] = pd.Categorical(df["last_review"].map(lambda d: _activity(d, scrape_date)),
                                           categories=ACT_ORDER, ordered=True)
    df["license_type"] = pd.Categorical(df["license"].map(_license_type), categories=LIC_ORDER)
    df["short_name"] = df["neighbourhood_cleansed"].str.split(",").str[0]

    hosts = (df.groupby("host_id")
               .agg(host_name=("host_name", "first"),
                    listings=("id", "size"),
                    entire_homes=("room_type", lambda s: (s == "Entire home/apt").sum()),
                    listings_with_estimate=("estimated_revenue_l365d", "count"),
                    est_revenue=("estimated_revenue_l365d", "sum"),
                    median_price=("price", "median"))
               .sort_values("listings", ascending=False))
    hosts["portfolio"] = pd.cut(hosts["listings"], bins=PORTFOLIO_BINS, labels=PORTFOLIO_LABELS)
    df = df.merge(hosts["portfolio"], left_on="host_id", right_index=True, how="left")

    open_days, daily = calendar_summary()
    df = df.merge(open_days, left_on="id", right_index=True, how="left")

    reviews = pd.read_csv(DATA_DIR / "reviews.csv", usecols=["listing_id", "date"])
    reviews["date"] = pd.to_datetime(reviews["date"])

    return Bundle(df=df, hosts=hosts, reviews=reviews, daily=daily, geo=geo,
                  neighbourhoods=neighbourhoods, scrape_date=scrape_date)


def calendar_summary():
    """Open days per listing and share of listings available per date.

    calendar.csv is 2.5M rows (85 MB), too big to deploy, so its two summaries are saved as small CSVs the first time
    they're computed and read from those afterwards (e.g. on Streamlit Cloud, where only the summaries are uploaded).
    """
    open_path, daily_path = DATA_DIR / "calendar_open_days.csv", DATA_DIR / "calendar_daily.csv"
    if not (open_path.exists() and daily_path.exists()):
        calendar = pd.read_csv(DATA_DIR / "calendar.csv", usecols=["listing_id", "date", "available"])
        calendar["available"] = calendar["available"].eq("t")
        calendar.groupby("listing_id")["available"].sum().rename("open_days").to_csv(open_path)
        calendar.groupby("date")["available"].mean().rename("share_available").to_csv(daily_path)
    open_days = pd.read_csv(open_path, index_col="listing_id")["open_days"]
    daily = pd.read_csv(daily_path, parse_dates=["date"])
    return open_days, daily


def data_fingerprint():
    """Changes whenever a data file changes, so cached model artifacts can be rebuilt.

    Uses file sizes, not modification times, so the fingerprint survives copying the files to another machine.
    """
    return tuple((name, (DATA_DIR / name).stat().st_size) for name in DATA_FILES)
