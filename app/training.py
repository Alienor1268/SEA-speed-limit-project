"""Training Pipeline (batch job) + Model Store.

Trains the neural network on the hours where it is used (dark or black ice, SPEC1/SPEC2)
and stores a new model version in MODEL_DIR (REQ20). `current.json` points to the model
that the Speed Limit Service uses.

TRAIN_YEARS selects the training data: "N" (first model) or "N,N+1" (retrained, slide 20).
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURES, MODEL_DIR, EnsembleModel, engine, wait_for_db

TRAIN_YEARS = [y.strip() for y in os.getenv("TRAIN_YEARS", "N,N+1").split(",")]
TEST_DAYS = range(25, 32)          # validation by whole days, not random hours
N_MODELS = 5                       # small ensemble: averages out random initialisation


def make_mlp(seed):
    return make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(16, 8), alpha=1.0, max_iter=3000, random_state=seed))


def fit(df):
    return EnsembleModel([make_mlp(s).fit(df[FEATURES], df.near_accidents) for s in range(N_MODELS)])


def main():
    wait_for_db()
    data = pd.read_sql("SELECT * FROM hourly", engine)
    data = data[data.nn_applies.astype(bool) & data.year.isin(TRAIN_YEARS)].reset_index(drop=True)
    # more weight on the most recent year: N+1 rows appear twice (slide 20)
    if "N+1" in TRAIN_YEARS and len(TRAIN_YEARS) > 1:
        data = pd.concat([data, data[data.year == "N+1"]], ignore_index=True)

    test = data.day.isin(TEST_DAYS)
    check = fit(data[~test])
    pred = check.predict(data[test])
    mae = float(np.mean(np.abs(pred - data.near_accidents[test])))
    baseline = float(np.mean(np.abs(data.near_accidents[~test].mean() - data.near_accidents[test])))

    model = fit(data)                                     # final model on all training data
    os.makedirs(MODEL_DIR, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    path = os.path.join(MODEL_DIR, f"model_{version}.joblib")
    joblib.dump(model, path)
    meta = {
        "version": version, "file": os.path.basename(path), "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_years": TRAIN_YEARS, "training_rows": int(len(data)), "features": FEATURES,
        "test_days": f"{min(TEST_DAYS)}-{max(TEST_DAYS)}", "mae_test": round(mae, 3),
        "mae_baseline_mean": round(baseline, 3),
    }
    with open(os.path.join(MODEL_DIR, f"model_{version}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    previous = os.path.join(MODEL_DIR, "current.json")
    if os.path.exists(previous):                         # keep the previous version for rollback
        os.replace(previous, os.path.join(MODEL_DIR, "previous.json"))
    with open(previous, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"model {version} trained on {TRAIN_YEARS}: MAE test {mae:.3f} (baseline {baseline:.3f})", flush=True)


if __name__ == "__main__":
    main()
