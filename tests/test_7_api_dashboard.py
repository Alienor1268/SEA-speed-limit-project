"""Backend API & Dashboard – test cases UI1–UI5 (REQ6, REQ15, REQ16)."""
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

import speedlimit_service as sl
import web
from common import engine


@pytest.fixture
def client(trained, monkeypatch):
    monkeypatch.setattr(sl, "ask_air_quality", lambda *a: {"aq_index": 40, "reduction": 0, "poor": False, "source": "test"})
    with TestClient(sl.app) as c:
        yield c


def count_decisions():
    with engine.connect() as c:
        return c.execute(text("SELECT COUNT(*) FROM decisions")).scalar()


def test_UI1_valid_hour_returns_limit_and_reason(client):
    """POST /decide for day 9, hour 20 -> 200, limit in {40,...,80}, reason not empty (REQ16)."""
    r = client.post("/decide", json={"year": "N", "day": 9, "hour": 20})
    assert r.status_code == 200
    body = r.json()
    assert body["final_limit"] in (40, 50, 60, 70, 80) and body["reason"]


def test_UI2_day_32_is_rejected(client):
    """Day 32 -> error 400, no decision stored."""
    before = count_decisions()
    r = client.post("/decide", json={"year": "N", "day": 32, "hour": 20})
    assert r.status_code == 400 and "day 1-31" in r.json()["detail"]
    assert count_decisions() == before


def test_UI3_hour_24_is_rejected(client):
    """Hour 24 -> error 400, no decision stored."""
    before = count_decisions()
    r = client.post("/decide", json={"year": "N", "day": 9, "hour": 24})
    assert r.status_code == 400 and "hour 0-23" in r.json()["detail"]
    assert count_decisions() == before


def test_UI4_only_the_latest_decision_per_hour_is_listed(client):
    """Same hour decided twice, then GET /decisions -> only the latest decision for that hour (REQ15)."""
    client.post("/decide", json={"year": "N", "day": 11, "hour": 7})
    client.post("/decide", json={"year": "N", "day": 11, "hour": 7})
    rows = client.get("/decisions", params={"year": "N", "day": 11}).json()
    assert [r["hour"] for r in rows].count(7) == 1


def test_UI5_decision_is_ready_within_one_minute(client):
    """New decision -> available for the dashboard within 1 minute (REQ6); the dashboard page is served."""
    start = time.perf_counter()
    client.post("/decide", json={"year": "N", "day": 12, "hour": 3})
    assert time.perf_counter() - start < 60
    page = TestClient(web.app).get("/")
    assert page.status_code == 200 and 'id="sign"' in page.text
