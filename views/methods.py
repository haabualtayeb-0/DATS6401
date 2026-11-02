import streamlit as st

from dcairbnb.data import load
from dcairbnb.model import load_artifacts
from dcairbnb.style import chapter, section

b = load()
art = load_artifacts()

chapter("Methods", "How This Was Made",
        "Where the data comes from, how it was cleaned and how the projector works.")

section("Data")
st.markdown(f"""
All data comes from an [Inside Airbnb](https://insideairbnb.com/get-the-data/) snapshot of Washington, DC, collected on
**{b.scrape_date:%B %d, %Y}**.

| File | Rows | Used for |
|---|---|---|
| `listings.csv` | {len(b.df):,} | Listings, hosts, prices, licenses, ratings and amenities |
| `reviews.csv` | {len(b.reviews):,} | Demand over time, with each review standing in for a stay |
| `calendar.csv` | {len(b.daily):,} dates | Availability over the next year |
| `neighbourhoods.geojson` | {len(b.neighbourhoods)} | Neighborhood maps |

- **Cleaning:** prices and dates were converted from text, amenities were read from their JSON lists, hosts were grouped by
  how many DC listings they run, and license entries were sorted into hosted, unhosted, exempt and missing.
- **Outliers:** price comparisons leave out the top and bottom 1% of prices, since the highest listing asks over \\$75,000 a night.""")

section("Revenue projector")
best = art["scores"].iloc[-1]
st.markdown(f"""
1. **Training data:** the model learns from {art['n_train']:,} short-term listings (minimum stay under 30 nights) that have a
   price and a review in the past year, with the top and bottom 1% of prices removed, and it predicts the log of Inside
   Airbnb's estimated annual revenue.
2. **Listing model:** gradient-boosted trees (scikit-learn's `HistGradientBoostingRegressor`) predict revenue from each
   listing's features, with constraints so that more guests or bedrooms never lower the estimate and with review counts
   left out because Inside Airbnb's revenue estimate is calculated from them.
3. **Accuracy:** in 5-fold cross-validation the model explains {best['r2_log']:.0%} of the variation in log revenue, with a
   typical error of \\${best['median_error']:,.0f} per listing.
4. **80% range:** two more models predict the 10th and 90th percentiles, and conformal calibration widens that range until
   it covers {art['coverage']['calibrated']:.0%} of held-out listings (up from {art['coverage']['raw']:.0%}).
5. **Demand model:** a damped Holt-Winters forecast (statsmodels) of monthly reviews per active listing since January 2022,
   simulated 2,000 times, gives each year's demand index and its range.
6. **Projection:** real listings of the chosen size are placed in the chosen neighborhood, and their median prediction is
   multiplied by that year's demand index, with the two ranges combined.
""")
