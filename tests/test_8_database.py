"""Database – test cases DB1–DB5 (REQ15, REQ22, NFR2, NFR5)."""
import pandas as pd
import pytest
from sqlalchemy import inspect, text

import common
from common import engine


def test_DB1_one_decision_is_one_row(router, monkeypatch):
    """One call of decide -> exactly one new row in 'decisions', with the model version (REQ15)."""
    from conftest import FakeModel
    monkeypatch.setitem(router.STATE, "model", FakeModel({40: .5, 50: .6, 60: .7, 70: .8, 80: .9}))
    with engine.connect() as c:
        before = c.execute(text("SELECT COUNT(*) FROM decisions")).scalar()
    router.decide("N", 9, 20)
    with engine.connect() as c:
        after = c.execute(text("SELECT COUNT(*) FROM decisions")).scalar()
        version = c.execute(text("SELECT model_version FROM decisions ORDER BY id DESC LIMIT 1")).scalar()
    assert after == before + 1 and version == "test-model"


def test_DB2_one_row_per_hour(db_ready):
    """After ingestion -> 744 rows per year in the 'hourly' table (31 days x 24 hours)."""
    counts = pd.read_sql("SELECT year, COUNT(*) AS n FROM hourly GROUP BY year", engine).set_index("year").n
    assert counts["N"] == 744 and counts["N+1"] == 744


def test_DB3_no_licence_plates_are_stored(db_ready):
    """Schema check -> no column of any table contains licence plates (REQ22, GDPR)."""
    insp = inspect(engine)
    columns = [c["name"].lower() for t in insp.get_table_names() for c in insp.get_columns(t)]
    assert not any("licence" in c or "plate" in c for c in columns)


def test_DB4_services_wait_for_a_late_database(monkeypatch):
    """Database starts late -> the services retry and then start without error (NFR2)."""
    calls = {"n": 0}

    class LateEngine:
        def connect(self):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("database not ready")
            return engine.connect()

    monkeypatch.setattr(common, "engine", LateEngine())
    common.wait_for_db(retries=5, delay=0)
    assert calls["n"] == 3


def test_DB5_old_decisions_are_kept(router, monkeypatch):
    """A decision older than 4 years is still stored after new decisions (retention 5 years, NFR5)."""
    from conftest import FakeModel
    monkeypatch.setitem(router.STATE, "model", FakeModel({40: .5, 50: .6, 60: .7, 70: .8, 80: .9}))
    with engine.begin() as c:
        c.execute(text("INSERT INTO decisions (created_at, year, day, hour, final_limit, reason) "
                       "VALUES ('2022-11-01 10:00:00', 'N', 1, 0, 60, 'old decision')"))
    router.decide("N", 9, 21)
    with engine.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM decisions WHERE reason = 'old decision'")).scalar() == 1
