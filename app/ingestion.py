"""Data Ingestion Service (one-shot job).

Reads the raw CSV files, repairs the data errors (#1-#25) and applies the quality gates
(REQ17, SPEC10) with prepare.py, then stores one row per hour and year in the table `hourly`.
It also creates the table `decisions` used by the Speed Limit Service (REQ15).
"""
import os

import pandas as pd
from sqlalchemy import text

from common import engine, wait_for_db
from prepare import build_year, load_master

DATA_DIR = os.getenv("DATA_DIR", "../data")
FILES = {
    "N": ("*SensorReadingsYearN_*.csv", "*AccidentsYearN_*.csv"),
    "N+1": ("*SensorReadingsYearNPlus1_*.csv", "*AccidentsYearNPlus1_*.csv"),
}

DECISIONS_DDL = """
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY {autoinc},
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    year            VARCHAR(4),
    day             INTEGER,
    hour            INTEGER,
    dark            BOOLEAN,
    black_ice       BOOLEAN,
    nn_used         BOOLEAN,
    nn_limit        INTEGER,
    aq_index        INTEGER,
    aq_reduction    INTEGER,
    aq_source       VARCHAR(60),
    final_limit     INTEGER,
    reason          VARCHAR(300),
    model_version   VARCHAR(60),
    co2_saved_kg    REAL,
    alerts          TEXT,
    details         TEXT
)
"""


def main():
    wait_for_db()
    sensors = load_master(DATA_DIR)
    frames = []
    for year, (readings_glob, accidents_glob) in FILES.items():
        df, report = build_year(DATA_DIR, readings_glob, accidents_glob, sensors)
        df.insert(0, "year", year)
        print(f"year {year}: {len(df)} hours, {int(df.nn_applies.sum())} NN hours | report {report}", flush=True)
        frames.append(df)
    hourly = pd.concat(frames, ignore_index=True)
    hourly.columns = [c.lower() for c in hourly.columns]
    for c in ["dark", "black_ice", "nn_applies"]:
        hourly[c] = hourly[c].astype(bool)
    hourly.to_sql("hourly", engine, if_exists="replace", index=False)

    autoinc = "AUTOINCREMENT" if engine.dialect.name == "sqlite" else ""
    ddl = DECISIONS_DDL.format(autoinc=autoinc)
    if engine.dialect.name == "postgresql":
        ddl = ddl.replace("id              INTEGER PRIMARY KEY ", "id              SERIAL PRIMARY KEY")
    with engine.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS decisions"))   # demo: start each deployment with an empty log
        c.execute(text(ddl))
    print(f"ingestion done: {len(hourly)} rows in table 'hourly'", flush=True)


if __name__ == "__main__":
    main()
