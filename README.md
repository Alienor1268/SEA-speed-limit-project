# Speed Limit System – deployable application

Case study *Speed Limit* (SE4AI, Master Krems) – Aliénor Franck de Préaumont.

The application implements the simplified component diagram (7 components) and the
sequence diagram with the router: rules → neural network (only when dark or black ice)
→ air quality with an LLM agent → lowest limit → sign and dashboard.

## Quick start (Windows, macOS or Linux)

1. Install **Docker Desktop** and start it.
2. Open a terminal (PowerShell on Windows) in this folder (`speed_limit_app`).
3. Optional – let a real LLM decide the air-quality reduction:
   copy `.env.example` to `.env` and put your Claude API key in `ANTHROPIC_API_KEY`.
   Without a key, the fixed fallback rule is used (REQ13).
4. Run:

   ```
   docker compose up --build
   ```

5. Open **http://localhost:8000** in the browser. Choose a year, day and hour and press
   *🚦 Decide* (or *⏱ Simulate 24 h* for the whole day).

Stop with `Ctrl+C`, then `docker compose down` (add `-v` to delete the database and the models).

## Components

| Component (diagram)      | Container   | File                       | Requirements |
|--------------------------|-------------|----------------------------|--------------|
| Database                 | `db`        | PostgreSQL 16              | REQ15 (decisions log) |
| Data Ingestion Service   | `ingestion` | `app/ingestion.py`, `app/prepare.py` | REQ17, SPEC10, errors #1–#25 |
| Training Pipeline        | `training`  | `app/training.py`          | REQ19, REQ20, slide 20 |
| Model Store              | volume `models` | `model_<version>.joblib/.json`, `current.json`, `previous.json` | REQ20 |
| Air Quality Service      | `airquality`| `app/airquality_service.py`| REQ2, REQ13, SPEC6, SPEC12 |
| Speed Limit Service (router) | `speedlimit` | `app/speedlimit_service.py` | REQ1–REQ5, SPEC1, SPEC2, SPEC5 |
| Backend API & Dashboard  | `web`       | `app/web.py`, `app/static/index.html` | REQ6, REQ7, REQ16 |
| (inside the router)      | `speedlimit`| `app/alerts.py`            | REQ8, salt trucks, first aid, CO₂ |

Start order (handled by `docker-compose.yml`): `db` → `ingestion` (one-shot) → `training`
(one-shot) → `airquality` + `speedlimit` → `web`.

## How a decision is made (one hour)

1. **Rules:** dark = average light < 0.5 lux (SPEC1); black ice = water > 1,000 µm and
   temperature < 0 °C (SPEC2).
2. **Neural network** (only if dark or black ice): predicts near-accidents per hour for
   40, 50, 60, 70, 80 km/h; the highest limit with ≤ 1 near-accident per hour is chosen (REQ1,
   SPEC5). A safety guard makes the predictions non-decreasing with the limit, because the raw
   network learned the confounding of the data (slide 20). If no limit qualifies → 40 km/h.
   Otherwise the limit is 80 km/h (REQ3).
3. **Air quality** (always): the predicted index is read. If it is above the “poor” threshold
   (100), an **LLM agent** decides the reduction. The LLM requests the tools `get_air_quality`
   and `get_recent_accidents`; the service runs them and returns the results (function
   calling). Only 0, 10, 20, 30 or 40 km/h are accepted (SPEC6); otherwise, or without an API
   key, a fixed rule is used.
4. **Final limit** = lowest of both, never above 80 km/h (REQ4, REQ5). It is saved with its
   reason and model version in the table `decisions` (REQ15) and shown on the dashboard.

## Alerts and CO₂ (`app/alerts.py`)

| Alert | When | To whom |
|-------|------|---------|
| 🧂 Salt trucks – now | black ice in this hour (SPEC2) | winter service (Straßenmeisterei) |
| 🧂 Salt trucks – preventive | black ice expected in the next 3 hours (the following hours are used as forecast, SPEC7) | winter service |
| ❄️ Frost stand-by | temperature ≤ 1 °C and within 1 °C of the dew point (Magnus formula, from humidity) | winter service |
| 🚑 First aid | accident with injured people in this hour | ambulance 144, police 133 |
| 👮 Accident | accident without injuries | police 133, tow truck |
| ⚠️ Near-accidents | 3 or more near-accidents in this hour | traffic control |
| 🌪️ Weather change | temperature −5 °C or wind +30 km/h within 1 hour (REQ8, SPEC9) | traffic control, emergency services |

**CO₂:** emissions of an average car per km are approximated with a simple physical model
(rolling resistance + air drag + engine idle): `g/km = 80 + 0.006·v² + 1800/v`
(141 g/km at 80 km/h, minimum around 60 km/h). CO₂ avoided per hour =
cars per hour × `SEGMENT_KM` × (g/km at 80 − g/km at the chosen limit). The tree equivalent
uses 60 g CO₂ per tree and day (about 22 kg per year). These values are assumptions for the
demo, not an official emission model (e.g. HBEFA or COPERT).

## The dashboard

* 🛣️ Animated highway: day/night sky, snow or rain, fog, smog, cars driving at the chosen speed,
  the gantry sign with its message, salt truck 🧂🚛 and ambulance 🚑 when an alert is active.
* ⏱ **Simulate 24 h**: decides the 24 hours, then replays them hour by hour (1.5 s per hour).
* Plain-language explanation of each decision, neural-network bars, air-quality gauge with the
  LLM tool calls, CO₂ card, alerts feed and a colored timeline of the day.

## Retraining (slide 20)

The default model is trained on years N and N+1 (`TRAIN_YEARS=N,N+1`). To compare with the
first model trained on year N only:

```
docker compose run --rm -e TRAIN_YEARS=N training
curl -X POST http://localhost:8000/api/model/reload
```

(PowerShell: `Invoke-RestMethod -Method Post http://localhost:8000/api/model/reload`.)
Every training creates a new version in the model store; the previous one is kept in
`previous.json` for rollback.

## API (Speed Limit Service, also reachable through the dashboard under `/api/...`)

| Method | Path | Body / parameters |
|--------|------|-------------------|
| POST | `/decide` | `{"year": "N", "day": 9, "hour": 20}` |
| POST | `/decide_day` | `{"year": "N", "day": 9}` |
| GET  | `/decisions` | `?year=N&day=9` – latest decision per hour |
| GET  | `/model` | current model version and its test error |
| POST | `/model/reload` | load the newest model from the model store |

Air Quality Service: `GET /assess?year=N&day=9&hour=12` (port 8002, internal).

## Tests (40 test cases)

The folder `tests/` contains the test cases of the presentation: 8 components × 5 tests = 40.
They use a temporary SQLite database and model store, so they do not touch the running application.

| File | Component | Tests |
|------|-----------|-------|
| `test_1_data_ingestion.py` | Data Ingestion Service | DI1–DI5 |
| `test_2_training_pipeline.py` | Training Pipeline | TP1–TP5 |
| `test_3_model_store.py` | Model Store | MS1–MS5 |
| `test_4_router.py` | Speed Limit Service (router) | SL1–SL5 |
| `test_5_air_quality.py` | Air Quality Service | AQ1–AQ5 |
| `test_6_alerts.py` | Alerts | AL1–AL5 |
| `test_7_api_dashboard.py` | Backend API & Dashboard | UI1–UI5 |
| `test_8_database.py` | Database | DB1–DB5 |

**With Docker** (after `docker compose up --build` has built the image once):

```
docker compose run --rm tests
```

**Without Docker:**

```
pip install -r requirements.txt -r requirements-test.txt
python -m pytest
```

Every test name starts with its ID (e.g. `test_SL3_prediction_exactly_1_is_accepted`), so the output can be
compared with the test table of the presentation. Test AL2 found a bug: a temperature drop of exactly 5 °C
raised an alert, although SPEC9 says “more than 5 °C”; `alerts.py` was corrected.

## Run without Docker (for development)

```
pip install -r requirements.txt
cd app
set DATA_DIR=../data          (macOS/Linux: export DATA_DIR=../data)
python ingestion.py           (uses SQLite: speedlimit.db)
python training.py
uvicorn airquality_service:app --port 8002
uvicorn speedlimit_service:app --port 8001
uvicorn web:app --port 8000
```

## Limits (honest list)

* **Air-quality API:** no real API is connected; a deterministic simulation with two pollution
  episodes (days 8–10 and 21–22) is used. Set `AQ_API_URL` to plug in a real API returning
  `{"index": <int>}`.
* **Model quality:** near-accidents per hour are close to pure chance (see slide 20); the
  network is hardly better than the average, so the NN decisions are mostly 40 or 80 km/h.
* **Not implemented:** stability of the signs (REQ14), sensor-maintenance alerts (REQ18), real
  sending of alerts (they are shown on the dashboard only), the mobile app (REQ9) and access
  control (NFR4).
* **Alerts replay the data:** accidents are taken from the accident file of that hour, so the
  first-aid alert shows what the system would have sent, not a live detection.
* **Data:** the CSV files in `data/` are the V100 files of the case study (file names without
  the “(1)” suffix).
