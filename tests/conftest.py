"""Shared setup for the 40 test cases (8 components x 5 tests).

The tests run without Docker and without PostgreSQL: they use a temporary SQLite database,
a temporary model store and the CSV files in data/. Nothing outside the temporary folder is changed.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="speedlimit_tests_"))

# must be set before the application modules are imported (they read them at import time)
os.environ["DATABASE_URL"] = f"sqlite:///{(TMP / 'test.db').as_posix()}"
os.environ["MODEL_DIR"] = str(TMP / "models")
os.environ["DATA_DIR"] = str(ROOT / "data")
os.environ["ANTHROPIC_API_KEY"] = ""                    # no real LLM calls in the tests
os.environ["AIRQUALITY_URL"] = "http://127.0.0.1:9"     # unreachable on purpose
sys.path.insert(0, str(ROOT / "app"))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import pytest               # noqa: E402

DATA = ROOT / "data"
MODEL_DIR = TMP / "models"


def use_fast_training(training):
    """Smaller network and one model instead of five, so the tests run in seconds.
    The logic tested (data selection, split, versioning) is the same as in production."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    training.N_MODELS = 1
    training.make_mlp = lambda seed: make_pipeline(
        StandardScaler(), MLPRegressor(hidden_layer_sizes=(8,), alpha=1.0, max_iter=300, random_state=seed))


@pytest.fixture(scope="session")
def db_ready():
    """Runs the Data Ingestion Service once: CSV files -> table 'hourly' + empty table 'decisions'."""
    import ingestion
    ingestion.main()
    return True


@pytest.fixture(scope="session")
def trained(db_ready):
    """Runs the Training Pipeline once (years N and N+1) and returns the model metadata."""
    import training
    use_fast_training(training)
    training.TRAIN_YEARS = ["N", "N+1"]
    training.main()
    return json.loads((MODEL_DIR / "current.json").read_text())


# ---------- helpers for the router tests ----------
READ_COLS = ["Month", "Day", "Hour", "Minute", "Second", "Sensor", "Value"]
NORMAL = {2: 200, 9: 200, 12: 200,        # light (lux)
          15: 5, 18: 5, 19: 5,            # temperature (degC)
          1: 0, 8: 0, 11: 0,              # water on the road (micrometres)
          4: 70, 6: 10, 14: 13,           # humidity, wind strength
          5: 90, 13: 90, 16: 2.0, -1: 80}  # wind direction, traffic, speed limit


def feature_row(light=200.0, temp=5.0, water=0.0, hour=20):
    """Builds one hourly row exactly as the ingestion does, from synthetic sensor readings."""
    import prepare
    values = dict(NORMAL)
    for s in (2, 9, 12):
        values[s] = light
    for s in (15, 18, 19):
        values[s] = temp
    for s in (1, 8, 11):
        values[s] = water
    readings = pd.DataFrame([(1, 1, h, 0, 0, s, v) for h in range(24) for s, v in values.items()], columns=READ_COLS)
    feats, _, _ = prepare.hourly_table(readings, prepare.load_master(DATA))
    row = {k.lower(): (v.item() if hasattr(v, "item") else v) for k, v in feats.reset_index().iloc[hour].to_dict().items()}
    row.update(year="N", day=1, hour=hour, near_accidents=0, accidents=0, injured=0, damage_eur=0)
    return row


class FakeModel:
    """Stands in for the neural network: returns a fixed prediction for each candidate limit."""

    def __init__(self, by_limit):
        self.by_limit = by_limit

    def predict(self, X):
        return np.array([self.by_limit[int(v)] for v in X["speed_limit"]], dtype=float)


@pytest.fixture
def router(db_ready, monkeypatch):
    """The Speed Limit Service with a fake model, no air-quality service and no alerts."""
    import speedlimit_service as sl
    monkeypatch.setitem(sl.STATE, "meta", {"version": "test-model"})
    monkeypatch.setattr(sl, "build_alerts", lambda *a, **k: [])
    monkeypatch.setattr(sl, "ask_air_quality", lambda *a, **k: {"aq_index": 40, "reduction": 0, "poor": False,
                                                               "source": "test", "llm_answer": None, "tool_calls": []})
    return sl
