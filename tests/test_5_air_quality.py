"""Air Quality Service – test cases AQ1–AQ5 (REQ2, REQ13, SPEC6)."""
import pytest

import airquality_service as aq


@pytest.fixture
def service(monkeypatch):
    def setup(index, key="", llm=None):
        monkeypatch.setattr(aq, "get_air_quality", lambda *a: index)
        monkeypatch.setattr(aq, "API_KEY", key)
        monkeypatch.setattr(aq, "ask_llm", llm or (lambda *a: pytest.fail("the LLM must not be called")))
        return aq.assess("N", 9, 12)
    return setup


def test_AQ1_index_100_is_not_poor(service):
    """Index 100 (exactly the threshold) -> not poor, LLM not called, reduction 0 (SPEC6)."""
    res = service(100, key="test-key")
    assert res["poor"] is False and res["reduction"] == 0


def test_AQ2_valid_llm_answer_is_used(service):
    """Index 101, LLM answers '10' -> reduction 10, source = LLM (REQ2)."""
    res = service(101, key="test-key", llm=lambda *a: ("10", []))
    assert res["reduction"] == 10 and res["source"].startswith("LLM")


def test_AQ3_invalid_llm_answer_falls_back_to_rule(service):
    """Index 101, LLM answers '25' -> invalid (only 0/10/20/30/40) -> fallback rule -> 10 (SPEC6)."""
    res = service(101, key="test-key", llm=lambda *a: ("25", []))
    assert res["reduction"] == 10 and res["source"] == "rule: invalid LLM answer"


def test_AQ4_no_api_key_uses_the_rule(service):
    """Index 140, no API key -> fallback rule -> reduction 30 (REQ13)."""
    res = service(140)
    assert res["reduction"] == 30 and "no LLM key" in res["source"]


def test_AQ5_llm_timeout_uses_the_rule(service):
    """Index 170, LLM times out -> fallback rule -> reduction 40, the maximum (REQ13)."""
    def timeout(*a):
        raise TimeoutError("LLM did not answer")
    res = service(170, key="test-key", llm=timeout)
    assert res["reduction"] == 40 and "LLM error" in res["source"]
