"""Shared settings, database access and constants for all services."""
import os
import time

import numpy as np

from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///speedlimit.db")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# Inputs of the neural network (see "Speed_Limit_Data_Preparation_Q2_Q3", table 3.2)
FEATURES = ["log_light", "temperature_c", "water_um", "humidity_pct", "wind_kmh",
            "wind_dir_sin", "wind_dir_cos", "traffic_cps", "hour_sin", "hour_cos", "speed_limit"]
CANDIDATE_LIMITS = [40, 50, 60, 70, 80]        # SPEC5
MAX_LIMIT = 80                                 # REQ3, REQ4
MAX_NEAR_ACCIDENTS_PER_HOUR = 1.0              # REQ1
YEARS = ["N", "N+1"]

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


def wait_for_db(retries: int = 30, delay: float = 2.0) -> None:
    """The database container may need a few seconds before it accepts connections."""
    for i in range(retries):
        try:
            with engine.connect() as c:
                c.execute(text("SELECT 1"))
            return
        except Exception as exc:  # noqa: BLE001
            print(f"database not ready ({exc.__class__.__name__}), retry {i + 1}/{retries}", flush=True)
            time.sleep(delay)
    raise RuntimeError("database not reachable")


def fetch_one(sql: str, **params):
    with engine.connect() as c:
        row = c.execute(text(sql), params).mappings().first()
    return dict(row) if row else None


def fetch_all(sql: str, **params):
    with engine.connect() as c:
        return [dict(r) for r in c.execute(text(sql), params).mappings().all()]


class EnsembleModel:  # defined here so that every service can load the saved model
    """Average of several small MLPs. predict() returns near-accidents per hour (>= 0)."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.clip(np.mean([m.predict(X[FEATURES]) for m in self.models], axis=0), 0, None)
