"""
Case Study Speed Limit - data repair and feature preparation (version 2).

Used by the ingestion service (ingestion.py). Same logic as prepare_speed_limit_data_v2.py.

Reads Sensors, SensorTypes, SensorReadings (year N and N+1) and Accidents (year N and N+1),
repairs the data errors of the data analysis (numbers #1-#25 refer to that table) and writes
one row per hour, ready for the neural network, for each year.

Changes compared with version 1:
  - wind strength: sensor 6 is the reference; sensor 14 is only used (rescaled) if 6 is missing (#13)
  - wind direction: vector average of sensors 5 and 13, set to 0 when there is no wind (#14, #15)
  - a repair report is printed for every year
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

READ_COLS = ["Month", "Day", "Hour", "Minute", "Second", "Sensor", "Value"]
NEAR_COLS = ["CloseCarCm", "CloseGuardrailCm", "SkidAngle"]
DARK_LUX = 0.5            # SPEC1: 500 millilux
ICE_WATER_UM = 1000       # SPEC2: more than 1,000 micrometres of water ...
ICE_TEMP_C = 0            # ... and temperature below 0 degrees Celsius
WIND14_TO_6 = 0.75        # median ratio sensor 6 / sensor 14 (#13)
MAX_GAP_H = 3             # only short gaps are interpolated


def find(folder, pattern):
    hits = sorted(Path(folder).glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


# ---------- master data (#1 - #8) ----------
def load_master(folder):
    types = pd.read_csv(find(folder, "*SensorTypes*.csv"), sep=";", encoding="utf-8-sig", dtype=str)
    types = types.apply(lambda c: c.str.strip())                       # #6 'W ' and 'degrees   '
    types.loc[types.Type == "traffic", "SensorTypeCode"] = "TR"         # #1 code T used twice
    types.loc[types.Unit == "hm/h", "Unit"] = "km/h"                    # #5 typo in unit
    types[["Min", "Max"]] = types[["Min", "Max"]].astype(float)
    types.loc[len(types)] = ["SL", "speed limit", "km/h", 40.0, 80.0]  # #8 pseudo-sensor -1

    sensors = pd.read_csv(find(folder, "*_Sensors_*.csv"), sep=";", encoding="utf-8-sig", dtype=str)
    sensors = sensors.apply(lambda c: c.str.strip())
    sensors["SensorID"] = sensors.SensorID.astype(int)
    sensors.loc[sensors.SensorID == 16, "SensorTypeCode"] = "TR"        # #2 sensor 16 is traffic
    if 19 not in sensors.SensorID.values:                              # #3 sensor 19 = temperature
        sensors.loc[len(sensors)] = [19, "T"]
    sensors.loc[len(sensors)] = [-1, "SL"]                             # #8 speed limit
    return sensors.merge(types, on="SensorTypeCode", how="left")


# ---------- readings (#9 - #12, #16) ----------
def load_readings(path):
    first = Path(path).read_text(encoding="utf-8-sig").splitlines()[0]
    has_header = first.startswith("Month")                              # #9 N+1 file has no header
    r = pd.read_csv(path, sep=";", encoding="utf-8-sig", header=0 if has_header else None,
                    names=READ_COLS, decimal=",", dtype=str)            # #11 decimal comma
    for c in READ_COLS:
        r[c] = pd.to_numeric(r[c].str.replace(",", ".", regex=False), errors="coerce")
    rep = {"rows": len(r), "header_added": not has_header}
    missing = r[["Day", "Hour", "Sensor", "Value"]].isna().any(axis=1)
    rep["missing_removed"] = int(missing.sum())
    r = r[~missing].copy()
    r["Month"] = r.Month.fillna(1)
    before = len(r)
    r = r.drop_duplicates(subset=["Month", "Day", "Hour", "Sensor"], keep="first")
    rep["duplicates_removed"] = before - len(r)
    return r, rep


def quality_gate(r, sensors):
    """SPEC10: values outside the Min-Max range of their sensor type become missing."""
    m = r.merge(sensors[["SensorID", "Min", "Max"]], left_on="Sensor", right_on="SensorID", how="left")
    unknown = m.SensorID.isna()                                         # sensor without master data
    out = (m.Value < m.Min) | (m.Value > m.Max)
    m.loc[out, "Value"] = np.nan
    m = m[~unknown]
    return m.drop(columns=["SensorID", "Min", "Max"]), int(out.sum()), int(unknown.sum())


# ---------- hourly features (#12 - #17) ----------
def hourly_table(r, sensors):
    wide = r.pivot_table(index=["Month", "Day", "Hour"], columns="Sensor", values="Value", aggfunc="mean")
    days = int(r.Day.max())
    idx = pd.MultiIndex.from_product([[1], range(1, days + 1), range(24)], names=["Month", "Day", "Hour"])
    wide = wide.reindex(idx)
    gaps = int(wide.isna().sum().sum())
    wide = wide.interpolate(limit=MAX_GAP_H, limit_direction="both")

    def ids(code):
        return [s for s in sensors.loc[sensors.SensorTypeCode == code, "SensorID"] if s in wide.columns]

    def avg(code):                                                      # #12 several sensors per type
        cols = ids(code)
        return wide[cols].mean(axis=1) if cols else pd.Series(np.nan, index=wide.index)

    f = pd.DataFrame(index=wide.index)
    f["light_lux"] = avg("L")
    f["log_light"] = np.log10(f.light_lux + 0.01)                       # #17 skewed values
    f["temperature_c"] = avg("T")
    f["water_um"] = avg("W")
    f["humidity_pct"] = avg("H")

    # #13 wind strength: sensor 6 is the reference, sensor 14 rescaled only as a fallback
    wind = wide[6] if 6 in wide.columns else pd.Series(np.nan, index=wide.index)
    if 14 in wide.columns:
        wind = wind.fillna(wide[14] * WIND14_TO_6)
    f["wind_kmh"] = wind

    # #15 wind direction: vector average of all direction sensors, encoded as sin / cos
    rad = np.deg2rad(wide[ids("WD")])
    s, c = np.sin(rad).mean(axis=1), np.cos(rad).mean(axis=1)
    norm = np.sqrt(s ** 2 + c ** 2).replace(0, np.nan)
    calm = f.wind_kmh.fillna(0) == 0                                    # #14 no wind -> no direction
    f["wind_dir_sin"] = (s / norm).where(~calm, 0.0)
    f["wind_dir_cos"] = (c / norm).where(~calm, 0.0)

    f["traffic_cps"] = avg("TR")
    hour = f.index.get_level_values("Hour")                             # #16 hour only, no minute
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["speed_limit"] = wide[-1]

    f["dark"] = f.light_lux < DARK_LUX                                  # SPEC1
    f["black_ice"] = (f.water_um > ICE_WATER_UM) & (f.temperature_c < ICE_TEMP_C)   # SPEC2
    f["nn_applies"] = f.dark | f.black_ice
    return f, gaps, int(calm.sum())


# ---------- incidents (#19 - #25) ----------
def load_incidents(path):
    a = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    a = a.drop(columns=["LicencePlate"])                                # #25 privacy (GDPR)
    empty = a[["Damage", "Injured"] + NEAR_COLS].isna().all(axis=1)     # #22 empty rows
    a = a[~empty].copy()
    a["Hour"] = a.Hour - 1                                              # #19 hours 1-24 -> 0-23
    a["Injured"] = a.Injured.fillna(0)                                  # #24 empty = 0 injured
    a["near"] = a[NEAR_COLS].notna().any(axis=1)                        # #23 derive the type
    a["accident"] = a.Damage.notna() | (a.Injured > 0)
    beyond = ((a.CloseCarCm > 20) | (a.CloseGuardrailCm > 20) | (a.SkidAngle < 30)).sum()
    g = a.groupby(["Month", "Day", "Hour"]).agg(                        # #20, #21 join on the hour
        near_accidents=("near", "sum"), accidents=("accident", "sum"),
        injured=("Injured", "sum"), damage_eur=("Damage", "sum"))
    return g, {"empty_incident_rows": int(empty.sum()), "near_accidents": int(a.near.sum()),
               "accidents": int(a.accident.sum()), "beyond_SPEC4": int(beyond)}


def build_year(folder, readings_glob, accidents_glob, sensors):
    r, rep = load_readings(find(folder, readings_glob))
    r, rep["out_of_range"], rep["unknown_sensor_rows"] = quality_gate(r, sensors)
    feat, rep["missing_cells_before_fill"], rep["calm_hours"] = hourly_table(r, sensors)
    inc, irep = load_incidents(find(folder, accidents_glob))
    rep.update(irep)
    feat = feat.join(inc).fillna({"near_accidents": 0, "accidents": 0, "injured": 0, "damage_eur": 0})
    return feat.reset_index(), rep
