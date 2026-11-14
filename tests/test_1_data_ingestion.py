"""Data Ingestion Service – test cases DI1–DI5 (REQ17, SPEC10, data errors #1–#25)."""
import pandas as pd

import prepare
from conftest import DATA, READ_COLS


def test_DI1_missing_header_is_detected():
    """Year N+1 readings file has no header line -> header detected, 12,648 readings loaded (error #9)."""
    r, rep = prepare.load_readings(prepare.find(DATA, "*SensorReadingsYearNPlus1_*.csv"))
    assert rep["header_added"] is True
    assert rep["rows"] == 12648 and len(r) == 12648


def test_DI2_decimal_comma_is_read(tmp_path):
    """Value written '46,50' (decimal comma) -> stored as 46.5 (error #11)."""
    f = tmp_path / "readings.csv"
    f.write_text("Month;Day;Hour;Minute;Second;Sensor;Value\n1;1;0;3;35;6;46,50\n", encoding="utf-8")
    r, _ = prepare.load_readings(f)
    assert r.Value.iloc[0] == 46.5


def test_DI3_duplicate_reading_is_removed(tmp_path):
    """Same sensor, day and hour delivered twice -> one reading kept (SPEC10)."""
    f = tmp_path / "readings.csv"
    f.write_text("Month;Day;Hour;Minute;Second;Sensor;Value\n1;1;0;3;35;6;10,0\n1;1;0;40;2;6;12,0\n", encoding="utf-8")
    r, rep = prepare.load_readings(f)
    assert len(r) == 1 and rep["duplicates_removed"] == 1


def test_DI4_out_of_range_value_is_set_missing_then_interpolated():
    """Temperature 71 degC (range -30 to 70) -> set to missing; the 1-hour gap is interpolated (SPEC10)."""
    sensors = prepare.load_master(DATA)
    rows = [(1, 1, h, 0, 0, 15, 71.0 if h == 10 else 5.0) for h in range(24)]
    rows += [(1, 1, h, 0, 0, -1, 80.0) for h in range(24)]   # the speed limit (pseudo-sensor -1)
    r = pd.DataFrame(rows, columns=READ_COLS)
    gated, out_of_range, _ = prepare.quality_gate(r, sensors)
    assert out_of_range == 1
    assert pd.isna(gated.loc[(gated.Hour == 10) & (gated.Sensor == 15), "Value"]).all()
    feats, _, _ = prepare.hourly_table(gated, sensors)
    assert feats.loc[(1, 1, 10), "temperature_c"] == 5.0


def test_DI5_accident_hour_24_and_licence_plate(tmp_path):
    """Accident at hour 24 with licence plate -> joined to sensor hour 23, plate removed (errors #19, #25, REQ22)."""
    f = tmp_path / "acc.csv"
    f.write_text("Month;Day;Hour;Second;LicencePlate;Damage;Injured;CloseCarCm;CloseGuardrailCm;SkidAngle\n"
                 "1;5;24;12;W-12345A;;;15;;\n", encoding="utf-8")
    g, rep = prepare.load_incidents(f)
    assert (1, 5, 23) in g.index and g.loc[(1, 5, 23), "near_accidents"] == 1
    assert not any("licence" in c.lower() for c in g.reset_index().columns)
