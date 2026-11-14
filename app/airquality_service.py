"""Air Quality Service (REQ2, REQ13, SPEC6, SPEC12).

1. Reads the predicted air-quality index (API). Without a real API (AQ_API_URL empty) a
   deterministic simulation is used, with some episodes of poor air.
2. If the index is above the "poor" threshold, an LLM agent decides the reduction.
   The LLM does not call anything itself: it asks for tool calls (get_air_quality,
   get_recent_accidents), this service runs them and returns the results (function calling).
3. The answer is validated: only 0, 10, 20, 30 or 40 km/h are accepted (SPEC6).
   Without an API key, or if the LLM fails or answers something invalid, a fixed rule is used.
"""
import math
import os
import re

import httpx
from fastapi import FastAPI

from common import fetch_one

AQ_API_URL = os.getenv("AQ_API_URL", "")                 # real air-quality API (optional)
POOR_THRESHOLD = int(os.getenv("POOR_THRESHOLD", "100"))  # national "poor" threshold (assumption)
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ALLOWED = [0, 10, 20, 30, 40]

app = FastAPI(title="Air Quality Service")


# ---------- tools ----------
def get_air_quality(year: str, day: int, hour: int) -> int:
    """Predicted air-quality index for that hour (higher = worse)."""
    if AQ_API_URL:
        r = httpx.get(AQ_API_URL, params={"year": year, "day": day, "hour": hour}, timeout=10)
        r.raise_for_status()
        return int(r.json()["index"])
    # simulation: normal level 30-70, rush hours higher, two pollution episodes per month
    base = 50 + 15 * math.sin(2 * math.pi * (hour - 8) / 24) + 10 * math.sin(day * 1.7)
    if (8 <= day <= 10 or 21 <= day <= 22) and 7 <= hour <= 20:
        base += 55 + 10 * math.sin(hour)
    if year == "N+1":
        base -= 5
    return int(round(base))


def get_recent_accidents(year: str, day: int, hour: int) -> dict:
    """Accidents and near-accidents in the 24 hours before this hour."""
    t = (day - 1) * 24 + hour
    row = fetch_one(
        "SELECT COALESCE(SUM(near_accidents),0) AS near, COALESCE(SUM(accidents),0) AS acc FROM hourly "
        "WHERE year = :y AND ((day - 1) * 24 + hour) >= :a AND ((day - 1) * 24 + hour) < :b",
        y=year, a=t - 24, b=t)
    return {"near_accidents_last_24h": int(row["near"]), "accidents_last_24h": int(row["acc"])}


TOOLS = [
    {"name": "get_air_quality",
     "description": "Returns the predicted air-quality index for the given hour of the highway (higher is worse; "
                    f"above {POOR_THRESHOLD} is poor).",
     "input_schema": {"type": "object", "properties": {
         "year": {"type": "string"}, "day": {"type": "integer"}, "hour": {"type": "integer"}},
         "required": ["year", "day", "hour"]}},
    {"name": "get_recent_accidents",
     "description": "Returns the number of accidents and near-accidents on the highway in the 24 hours before the given hour.",
     "input_schema": {"type": "object", "properties": {
         "year": {"type": "string"}, "day": {"type": "integer"}, "hour": {"type": "integer"}},
         "required": ["year", "day", "hour"]}},
]

SYSTEM = f"""You decide whether the speed limit on a highway must be reduced because of poor air quality.
The normal limit is 80 km/h. Lower speed means fewer emissions. Use the tools to get the predicted
air-quality index and the recent accidents before you decide. Air quality is poor above {POOR_THRESHOLD}.
Answer with ONE number only, the reduction in km/h: 0, 10, 20, 30 or 40. No other text.

Examples of earlier decisions reviewed by traffic control:
- index 105, few accidents -> 10
- index 125 -> 20
- index 140 -> 30
- index 160 or more -> 40"""


def rule_reduction(index: int) -> int:
    """Fallback rule when no LLM is available (same scale as the examples)."""
    if index <= POOR_THRESHOLD:
        return 0
    return min(40, 10 * (1 + (index - POOR_THRESHOLD - 1) // 15))


def run_tool(name: str, args: dict):
    if name == "get_air_quality":
        return {"index": get_air_quality(args["year"], int(args["day"]), int(args["hour"]))}
    if name == "get_recent_accidents":
        return get_recent_accidents(args["year"], int(args["day"]), int(args["hour"]))
    raise ValueError(f"unknown tool {name}")


def ask_llm(year: str, day: int, hour: int):
    """Agent loop: the LLM requests tools, we run them and send the results back."""
    import anthropic

    client = anthropic.Anthropic(api_key=API_KEY)
    messages = [{"role": "user", "content":
                 f"New sensor data arrived for year {year}, day {day} of January, hour {hour}:00. "
                 "Should we reduce the speed limit because of air quality?"}]
    trace = []
    for _ in range(5):
        resp = client.messages.create(model=LLM_MODEL, max_tokens=200, system=SYSTEM, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return text, trace
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                out = run_tool(block.name, block.input)
                trace.append({"tool": block.name, "input": block.input, "result": out})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(out)})
        messages.append({"role": "user", "content": results})
    raise RuntimeError("LLM did not finish")


@app.get("/health")
def health():
    return {"status": "ok", "llm": bool(API_KEY), "model": LLM_MODEL if API_KEY else None}


@app.get("/assess")
def assess(year: str, day: int, hour: int):
    index = get_air_quality(year, day, hour)
    result = {"aq_index": index, "poor_threshold": POOR_THRESHOLD, "poor": index > POOR_THRESHOLD,
              "reduction": 0, "source": "rule: air quality good", "llm_answer": None, "tool_calls": []}
    if not result["poor"]:
        return result                                    # SPEC6: the LLM is only asked when air is poor
    if API_KEY:
        try:
            answer, trace = ask_llm(year, day, hour)
            result.update(llm_answer=answer, tool_calls=trace)
            m = re.fullmatch(r"\s*(\d+)\s*(km/h)?\s*\.?\s*", answer or "")
            value = int(m.group(1)) if m else None
            if value in ALLOWED:                          # SPEC6 / NFR: reject anything else
                result.update(reduction=value, source=f"LLM ({LLM_MODEL})")
                return result
            result["source"] = "rule: invalid LLM answer"
        except Exception as exc:  # noqa: BLE001 - REQ13: never block the decision
            result["source"] = f"rule: LLM error ({exc.__class__.__name__})"
    else:
        result["source"] = "rule: no LLM key configured"
    result["reduction"] = rule_reduction(index)
    return result
