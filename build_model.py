"""Train the revenue projector once and save it to artifacts/projector.joblib, so the app starts instantly.

    python build_model.py
"""
import joblib

from dcairbnb.data import load
from dcairbnb.model import ARTIFACT_PATH, build_artifacts

if __name__ == "__main__":
    art = build_artifacts(load())
    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(art, ARTIFACT_PATH)
    best = art["scores"].iloc[-1]
    print(f"Saved {ARTIFACT_PATH} | {art['n_train']:,} training listings | "
          f"R² {best['r2_log']:.2f}, median error ${best['median_error']:,.0f} | "
          f"80% range coverage {art['coverage']['calibrated']:.0%}")
