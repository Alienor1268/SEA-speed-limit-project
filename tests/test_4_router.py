"""Speed Limit Service (router) – test cases SL1–SL5 (SPEC1, SPEC2, SPEC5, REQ1, lowest limit)."""
from conftest import FakeModel, feature_row

SAFE = FakeModel({40: .5, 50: .6, 60: .7, 70: .8, 80: .9})


def decide_with(router, monkeypatch, row, model=SAFE, reduction=0):
    monkeypatch.setitem(router.STATE, "model", model)
    monkeypatch.setattr(router, "get_hour", lambda *a: row)
    monkeypatch.setattr(router, "ask_air_quality", lambda *a: {"aq_index": 125 if reduction else 40, "reduction": reduction,
                                                              "poor": bool(reduction), "source": "test"})
    return router.decide("N", 1, row["hour"])


def test_SL1_dark_boundary_0_5_lux(router, monkeypatch):
    """Light 0.49 lux -> dark -> NN used; exactly 0.50 lux -> not dark (SPEC1: below 500 millilux)."""
    assert decide_with(router, monkeypatch, feature_row(light=0.49))["nn_used"] is True
    res = decide_with(router, monkeypatch, feature_row(light=0.50))
    assert res["dark"] is False and res["nn_used"] is False


def test_SL2_black_ice_boundary(router, monkeypatch):
    """Water 1,000 um at -0.1 degC -> no black ice -> 80 km/h; 1,001 um at -0.1 degC -> black ice (SPEC2)."""
    res = decide_with(router, monkeypatch, feature_row(water=1000, temp=-0.1))
    assert res["black_ice"] is False and res["final_limit"] == 80
    assert decide_with(router, monkeypatch, feature_row(water=1001, temp=-0.1))["black_ice"] is True
    assert decide_with(router, monkeypatch, feature_row(water=1001, temp=0.0))["black_ice"] is False


def test_SL3_prediction_exactly_1_is_accepted(router, monkeypatch):
    """Predictions 0.8 / 0.9 / 1.0 / 1.2 / 1.4 -> NN limit 60 km/h (1.0 is accepted, REQ1 and SPEC5)."""
    model = FakeModel({40: .8, 50: .9, 60: 1.0, 70: 1.2, 80: 1.4})
    assert decide_with(router, monkeypatch, feature_row(light=0.1), model)["nn_limit"] == 60


def test_SL4_no_safe_limit_gives_40(router, monkeypatch):
    """All predictions above 1.0 -> 40 km/h and the reason says that no limit meets REQ1 (SPEC5)."""
    model = FakeModel({40: 1.3, 50: 1.4, 60: 1.5, 70: 1.6, 80: 1.7})
    res = decide_with(router, monkeypatch, feature_row(light=0.1), model)
    assert res["final_limit"] == 40 and "no limit meets REQ1" in res["reason"]


def test_SL5_lowest_limit_wins(router, monkeypatch):
    """NN 70 km/h and air-quality reduction 20 (= 60 km/h) -> final limit 60 km/h, never above 80 (REQ4)."""
    model = FakeModel({40: .5, 50: .6, 60: .7, 70: .9, 80: 1.3})
    res = decide_with(router, monkeypatch, feature_row(light=0.1), model, reduction=20)
    assert res["nn_limit"] == 70 and res["final_limit"] == 60
