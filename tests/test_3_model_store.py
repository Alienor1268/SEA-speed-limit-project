"""Model Store – test cases MS1–MS5 (versioning, rollback, NFR8)."""
import json
import shutil
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

import speedlimit_service as sl
import training
from common import FEATURES, engine
from conftest import MODEL_DIR, use_fast_training


@pytest.fixture(scope="module")
def two_versions(db_ready):
    """Trains twice, so the store holds a current and a previous version."""
    use_fast_training(training)
    training.TRAIN_YEARS = ["N", "N+1"]
    training.main()
    time.sleep(1.1)                                   # versions are time stamps (seconds)
    training.main()
    current = json.loads((MODEL_DIR / "current.json").read_text())
    previous = json.loads((MODEL_DIR / "previous.json").read_text())
    return previous, current


def test_MS1_new_training_becomes_current(two_versions):
    """New training run -> current.json points to the new version, the old one is kept in previous.json."""
    previous, current = two_versions
    assert current["version"] != previous["version"]
    assert (MODEL_DIR / previous["file"]).exists() and (MODEL_DIR / current["file"]).exists()


def test_MS2_rollback_restores_the_old_version(two_versions):
    """Rollback: previous.json -> current.json, then /model/reload -> the service reports the old version."""
    previous, current = two_versions
    backup = MODEL_DIR / "current.backup.json"
    shutil.copy(MODEL_DIR / "current.json", backup)
    try:
        shutil.copy(MODEL_DIR / "previous.json", MODEL_DIR / "current.json")
        with TestClient(sl.app) as client:
            assert client.post("/model/reload").json()["version"] == previous["version"]
    finally:
        shutil.copy(backup, MODEL_DIR / "current.json")
        sl.load_model()


def test_MS3_missing_model_file_keeps_the_old_model(two_versions):
    """Reload with a missing model file -> error returned, the old model stays active (NFR2)."""
    good = (MODEL_DIR / "current.json").read_text()
    with TestClient(sl.app, raise_server_exceptions=False) as client:
        active = sl.STATE["meta"]["version"]
        broken = dict(json.loads(good), file="does_not_exist.joblib", version="broken")
        (MODEL_DIR / "current.json").write_text(json.dumps(broken))
        try:
            assert client.post("/model/reload").status_code == 500
            assert sl.STATE["meta"]["version"] == active
        finally:
            (MODEL_DIR / "current.json").write_text(good)


def test_MS4_decision_is_logged_with_the_model_version(two_versions, monkeypatch):
    """Decision after reload -> the decision is logged with the new model version (REQ15)."""
    previous, current = two_versions
    monkeypatch.setattr(sl, "ask_air_quality", lambda *a: {"aq_index": 40, "reduction": 0, "poor": False, "source": "test"})
    sl.load_model()
    sl.decide("N", 9, 20)
    with engine.connect() as c:
        logged = c.execute(text("SELECT model_version FROM decisions ORDER BY id DESC LIMIT 1")).scalar()
    assert logged == current["version"]


def test_MS5_model_uses_the_same_features_as_the_service(two_versions):
    """Model metadata -> feature list identical to the 11 features used by the service (NFR8)."""
    assert two_versions[1]["features"] == FEATURES and len(FEATURES) == 11
