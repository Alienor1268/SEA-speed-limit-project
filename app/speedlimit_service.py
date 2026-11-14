"""Speed Limit Service - the router (REQ1-REQ5, SPEC1, SPEC2, SPEC5, REQ15).

For one hour:
  1. rule check: dark (SPEC1) or danger of black ice (SPEC2)?
  2. yes -> the neural network predicts near-accidents for 40-80 km/h and the highest limit
     with at most 1 near-accident per hour is chosen (REQ1, SPEC5); no -> 80 km/h (REQ3)
  3. the Air Quality Service is always asked (REQ2)
  4. final limit = lowest of both, never above 80 km/h (REQ4, REQ5), saved with its reason (REQ15)
"""
import json
import os

import httpx
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from alerts import build_alerts, co2_saved
from common import (CANDIDATE_LIMITS, MAX_LIMIT, MAX_NEAR_ACCIDENTS_PER_HOUR, MODEL_DIR, YEARS, engine,
                    fetch_all, fetch_one)

AIRQUALITY_URL = os.getenv("AIRQUALITY_URL", "http://localhost:8002")

app = FastAPI(title="Speed Limit Service")
STATE = {"model": None, "meta": None}


def load_model():
    with open(os.path.join(MODEL_DIR, "current.json")) as f:
        meta = json.load(f)
    STATE["model"] = joblib.load(os.path.join(MODEL_DIR, meta["file"]))
    STATE["meta"] = meta
    return meta


@app.on_event("startup")
def startup():
    load_model()


class HourRequest(BaseModel):
    year: str = "N"
    day: int
    hour: int


class DayRequest(BaseModel):
    year: str = "N"
    day: int


def get_hour(year: str, day: int, hour: int) -> dict:
    if year not in YEARS or not 1 <= day <= 31 or not 0 <= hour <= 23:
        raise HTTPException(400, "year must be N or N+1, day 1-31, hour 0-23")
    row = fetch_one("SELECT * FROM hourly WHERE year = :y AND day = :d AND hour = :h", y=year, d=day, h=hour)
    if row is None:
        raise HTTPException(404, "no data for this hour")
    return row


def nn_decision(row: dict):
    """SPEC5 with a safety guard: a higher limit is never predicted to be safer than a lower one
    (the raw network learned the opposite because of confounding, see slide 20)."""
    X = pd.DataFrame([{**row, "speed_limit": lim} for lim in CANDIDATE_LIMITS])
    raw = [float(v) for v in STATE["model"].predict(X)]
    guarded, running = [], 0.0
    for v in raw:                                     # monotonic: cumulative maximum from 40 to 80 km/h
        running = max(running, v)
        guarded.append(running)
    ok = [lim for lim, v in zip(CANDIDATE_LIMITS, guarded) if v <= MAX_NEAR_ACCIDENTS_PER_HOUR]
    limit = max(ok) if ok else min(CANDIDATE_LIMITS)  # no limit qualifies -> lowest (REQ12)
    preds = [{"limit": lim, "raw": round(r, 3), "guarded": round(g, 3)}
             for lim, r, g in zip(CANDIDATE_LIMITS, raw, guarded)]
    return limit, preds, bool(ok)


def ask_air_quality(year: str, day: int, hour: int) -> dict:
    try:
        r = httpx.get(f"{AIRQUALITY_URL}/assess", params={"year": year, "day": day, "hour": hour}, timeout=90)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001 - REQ13: the decision must still be made
        return {"aq_index": None, "reduction": 0, "source": f"service unavailable ({exc.__class__.__name__})",
                "poor": None, "llm_answer": None, "tool_calls": []}


def decide(year: str, day: int, hour: int) -> dict:
    row = get_hour(year, day, hour)
    dark, ice = bool(row["dark"]), bool(row["black_ice"])
    reasons = []
    if dark or ice:
        nn_limit, preds, found = nn_decision(row)
        cond = " and ".join(x for x, f in [("dark", dark), ("black ice", ice)] if f)
        reasons.append(f"{cond}: NN limit {nn_limit} km/h" + ("" if found else " (no limit meets REQ1)"))
    else:
        nn_limit, preds = MAX_LIMIT, []
        reasons.append("light and no black ice: NN not used")
    aq = ask_air_quality(year, day, hour)
    aq_limit = MAX_LIMIT - int(aq.get("reduction") or 0)
    if aq.get("reduction"):
        reasons.append(f"poor air quality (index {aq['aq_index']}): -{aq['reduction']} km/h")
    final = max(min(CANDIDATE_LIMITS), min(nn_limit, aq_limit, MAX_LIMIT))   # REQ4, REQ5
    if final == MAX_LIMIT:
        reasons.append("no reason to reduce: 80 km/h")
    alerts = build_alerts(row, year, day, hour)
    co2 = co2_saved(final, row["traffic_cps"])
    result = {
        "year": year, "day": day, "hour": hour, "final_limit": final, "reason": "; ".join(reasons),
        "dark": dark, "black_ice": ice, "nn_used": dark or ice, "nn_limit": nn_limit, "nn_predictions": preds,
        "air_quality": aq, "model_version": STATE["meta"]["version"], "alerts": alerts, "co2": co2,
        "conditions": {k: row[k] for k in ["light_lux", "temperature_c", "water_um", "humidity_pct", "wind_kmh",
                                            "traffic_cps", "speed_limit", "near_accidents", "accidents", "injured"]},
    }
    with engine.begin() as c:                          # REQ15: log every decision
        c.execute(text(
            "INSERT INTO decisions (year, day, hour, dark, black_ice, nn_used, nn_limit, aq_index, aq_reduction, "
            "aq_source, final_limit, reason, model_version, co2_saved_kg, alerts, details) VALUES (:year, :day, :hour, :dark, "
            ":black_ice, :nn_used, :nn_limit, :aq_index, :aq_reduction, :aq_source, :final_limit, :reason, :model_version, "
            ":co2, :alerts, :details)"),
            {**{k: result[k] for k in ["year", "day", "hour", "dark", "black_ice", "nn_used", "nn_limit",
                                        "final_limit", "reason", "model_version"]},
             "aq_index": aq.get("aq_index"), "aq_reduction": int(aq.get("reduction") or 0),
             "aq_source": str(aq.get("source"))[:60], "co2": co2["saved_kg"],
             "alerts": json.dumps(alerts, ensure_ascii=False), "details": json.dumps(result, default=str)})
    return result


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE["meta"]}


@app.post("/decide")
def decide_hour(req: HourRequest):
    return decide(req.year, req.day, req.hour)


@app.post("/decide_day")
def decide_day(req: DayRequest):
    return [decide(req.year, req.day, h) for h in range(24)]


@app.get("/decisions")
def decisions(year: str = "N", day: int = 1):
    rows = fetch_all("SELECT hour, final_limit, reason, dark, black_ice, nn_used, aq_reduction, co2_saved_kg, alerts, "
                     "created_at FROM decisions WHERE year = :y AND day = :d ORDER BY id", y=year, d=day)
    for r in rows:
        r["alerts"] = json.loads(r["alerts"] or "[]")
    latest = {r["hour"]: r for r in rows}               # latest decision per hour
    return [latest[h] for h in sorted(latest)]


@app.get("/model")
def model_info():
    return STATE["meta"]


@app.post("/model/reload")
def model_reload():
    return load_model()
