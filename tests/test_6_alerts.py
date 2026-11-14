"""Alerts – test cases AL1–AL5 (REQ8, REQ23, SPEC7, SPEC9)."""
import pytest

import alerts

BASE = {"temperature_c": 2.0, "humidity_pct": 60, "water_um": 0, "wind_kmh": 10, "black_ice": False,
        "accidents": 0, "injured": 0, "near_accidents": 0, "damage_eur": 0, "hour": 20}


@pytest.fixture
def run(monkeypatch):
    def go(row=None, prev=None, nxt=()):
        monkeypatch.setattr(alerts, "neighbours", lambda *a: (prev, list(nxt)))
        return alerts.build_alerts(dict(BASE, **(row or {})), "N", 9, 20)
    return go


def types(result):
    return [a["type"] for a in result]


def test_AL1_temperature_drop_of_5_1_degrees(run):
    """2.0 -> -3.1 degC within 1 hour (drop 5.1 degC) -> weather-change alert (SPEC9)."""
    assert "weather" in types(run({"temperature_c": -3.1}, prev=dict(BASE)))


def test_AL2_temperature_drop_of_exactly_5_degrees(run):
    """2.0 -> -3.0 degC within 1 hour (drop exactly 5 degC) -> no alert: SPEC9 says 'more than 5 degC'."""
    assert "weather" not in types(run({"temperature_c": -3.0}, prev=dict(BASE)))


def test_AL3_black_ice_now_sends_salt_trucks(run):
    """Black ice in this hour -> salt-truck alert to the winter service, level high (REQ23)."""
    res = run({"black_ice": True, "temperature_c": -2.0, "water_um": 1200})
    assert any(a["type"] == "winter" and a["level"] == "high" for a in res)


def test_AL4_black_ice_expected_soon(run):
    """No ice now, black ice expected in 2 hours -> preventive salting alert with that hour (REQ23, SPEC7)."""
    res = run(nxt=[dict(BASE, hour=21, black_ice=False), dict(BASE, hour=22, black_ice=True)])
    winter = [a for a in res if a["type"] == "winter"]
    assert winter and winter[0]["level"] == "medium" and "22:00" in winter[0]["text"]


def test_AL5_accident_with_injured(run):
    """Accident with 2 injured -> alert to ambulance 144 and police 133 (REQ8)."""
    res = run({"accidents": 1, "injured": 2})
    emergency = [a for a in res if a["type"] == "emergency"]
    assert emergency and "144" in emergency[0]["to"] and "133" in emergency[0]["to"]
