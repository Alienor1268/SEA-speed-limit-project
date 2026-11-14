"""Training Pipeline – test cases TP1–TP5 (SPEC3, NFR3, slide 20)."""
import json

import joblib
import pandas as pd

import training
from common import CANDIDATE_LIMITS, FEATURES, engine
from conftest import MODEL_DIR, use_fast_training


def run(monkeypatch, years):
    use_fast_training(training)
    monkeypatch.setattr(training, "TRAIN_YEARS", years)
    training.main()
    return json.loads((MODEL_DIR / "current.json").read_text())


def test_TP1_only_dark_or_icy_hours_are_used(db_ready, monkeypatch):
    """Train on year N -> only the 389 dark or icy hours are used (SPEC1, SPEC2, SPEC3)."""
    assert run(monkeypatch, ["N"])["training_rows"] == 389


def test_TP2_year_n_plus_1_is_weighted_twice(db_ready, monkeypatch):
    """TRAIN_YEARS = N,N+1 -> the N+1 rows appear twice: 389 + 2 x 389 = 1,167 rows (slide 20)."""
    assert run(monkeypatch, ["N", "N+1"])["training_rows"] == 389 * 3


def test_TP3_test_days_are_kept_out_of_training(db_ready, monkeypatch):
    """Validation split -> no hour of days 25-31 is used to fit the checked model (NFR3)."""
    seen = []
    original_fit = training.fit
    monkeypatch.setattr(training, "fit", lambda df: (seen.append(set(df.day)), original_fit(df))[1])
    run(monkeypatch, ["N"])
    check_fit, final_fit = seen
    assert check_fit.isdisjoint(range(25, 32))
    assert set(range(25, 32)) <= final_fit          # the final model then uses all days


def test_TP4_model_and_metadata_are_saved(trained):
    """Training finishes -> model file + metadata with test error and baseline error (REQ15, NFR8)."""
    assert (MODEL_DIR / trained["file"]).exists()
    assert (MODEL_DIR / f"model_{trained['version']}.json").exists()
    assert {"mae_test", "mae_baseline_mean", "train_years", "features"} <= set(trained)


def test_TP5_predictions_are_never_negative(trained):
    """Any input row -> predicted near-accidents are never negative (SPEC5)."""
    model = joblib.load(MODEL_DIR / trained["file"])
    hourly = pd.read_sql("SELECT * FROM hourly", engine)
    for limit in CANDIDATE_LIMITS:
        X = hourly[FEATURES].assign(speed_limit=limit)
        assert (model.predict(X) >= 0).all()
