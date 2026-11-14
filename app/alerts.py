"""Alerts and CO2 for one hour.

* Winter service (salt trucks / snow ploughs): black ice now, or expected in the next 3 hours
  (the following hours are used as the weather forecast, SPEC7), or frost risk from the dew point.
* Emergency services: an accident with injured people -> ambulance (144) and police (133);
  an accident without injuries -> police and tow truck; many near-accidents -> traffic control.
* Sharp weather change (REQ8, SPEC9): temperature falls by more than 5 degC or wind rises by more than 30 km/h within 1 hour.
* CO2: simple physical approximation of the emissions of an average car per km
  (rolling resistance + air drag + engine idle): g/km = A + B*v^2 + C/v. It is an
  assumption for the demo, not an official emission model.
"""
import math
import os

from common import fetch_all

SEGMENT_KM = float(os.getenv("SEGMENT_KM", "1"))     # length of highway controlled by one sign
TREE_G_PER_DAY = 60                                    # a tree absorbs about 22 kg CO2 per year
ICE_WATER_UM, FORECAST_H = 1000, 3


def co2_g_per_km(v: float) -> float:
    return 80 + 0.006 * v ** 2 + 1800 / v


def co2_saved(limit: int, traffic_cps: float) -> dict:
    cars_per_hour = max(0.0, float(traffic_cps or 0)) * 3600
    saved_g = cars_per_hour * SEGMENT_KM * (co2_g_per_km(80) - co2_g_per_km(limit))
    return {"cars_per_hour": round(cars_per_hour), "segment_km": SEGMENT_KM,
            "g_per_km_at_limit": round(co2_g_per_km(limit), 1), "g_per_km_at_80": round(co2_g_per_km(80), 1),
            "saved_kg": round(saved_g / 1000, 1), "trees_per_day": round(saved_g / TREE_G_PER_DAY)}


def dew_point(t: float, rh: float) -> float:
    """Magnus formula (degC)."""
    rh = min(max(rh, 1.0), 100.0)
    g = math.log(rh / 100) + 17.62 * t / (243.12 + t)
    return 243.12 * g / (17.62 - g)


def neighbours(year: str, day: int, hour: int):
    t = (day - 1) * 24 + hour
    rows = fetch_all("SELECT * FROM hourly WHERE year = :y AND ((day - 1) * 24 + hour) BETWEEN :a AND :b "
                     "ORDER BY day, hour", y=year, a=t - 1, b=t + FORECAST_H)
    prev = [r for r in rows if (r["day"] - 1) * 24 + r["hour"] == t - 1]
    nxt = [r for r in rows if (r["day"] - 1) * 24 + r["hour"] > t]
    return (prev[0] if prev else None), nxt


def build_alerts(row: dict, year: str, day: int, hour: int) -> list:
    alerts = []
    prev, nxt = neighbours(year, day, hour)
    t, rh, water = float(row["temperature_c"]), float(row["humidity_pct"]), float(row["water_um"])

    # --- winter service ---
    if row["black_ice"]:
        alerts.append({"type": "winter", "level": "high", "icon": "🧂",
                       "to": "Winter service (salt trucks)",
                       "text": f"Black ice now ({t:.1f} °C, {water:.0f} µm water): send salt trucks immediately."})
    else:
        coming = [r for r in nxt if r["black_ice"]]
        if coming:
            r = coming[0]
            alerts.append({"type": "winter", "level": "medium", "icon": "🧂",
                           "to": "Winter service (salt trucks)",
                           "text": f"Black ice expected at {int(r['hour']):02d}:00: salt the road preventively."})
        else:
            td = dew_point(t, rh)
            if t <= 1.0 and t - td <= 1.0:
                alerts.append({"type": "winter", "level": "low", "icon": "❄️",
                               "to": "Winter service (stand-by)",
                               "text": f"Frost risk: {t:.1f} °C, dew point {td:.1f} °C. Keep a salt truck ready."})

    # --- emergency services ---
    acc, inj, near = int(row.get("accidents") or 0), int(row.get("injured") or 0), int(row.get("near_accidents") or 0)
    if acc and inj:
        alerts.append({"type": "emergency", "level": "high", "icon": "🚑",
                       "to": "Ambulance 144 · Police 133",
                       "text": f"{acc} accident(s) with {inj} injured: first aid on the way, lanes protected."})
    elif acc:
        alerts.append({"type": "emergency", "level": "medium", "icon": "👮",
                       "to": "Police 133 · tow truck",
                       "text": f"{acc} accident(s) without injuries (damage {float(row.get('damage_eur') or 0):,.0f} €)."})
    if near >= 3:
        alerts.append({"type": "traffic", "level": "medium", "icon": "⚠️", "to": "Traffic control",
                       "text": f"{near} near-accidents in this hour: check the cameras."})

    # --- sharp weather change (REQ8, SPEC9) ---
    if prev:
        dt = t - float(prev["temperature_c"])
        dw = float(row["wind_kmh"] or 0) - float(prev["wind_kmh"] or 0)
        if dt < -5 or dw > 30:                                           # SPEC9: more than 5 degC / 30 km/h
            what = f"temperature {dt:+.1f} °C" if dt < -5 else f"wind {dw:+.0f} km/h"
            alerts.append({"type": "weather", "level": "medium", "icon": "🌪️",
                           "to": "Traffic control · emergency services", "text": f"Sharp weather change: {what} in 1 hour."})
    return alerts
