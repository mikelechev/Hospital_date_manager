# Hospital Date Manager

Hospital Date Manager is a prototype for hospital appointment optimization. It predicts the probability that a patient will miss an appointment and uses that probability to decide when a controlled overbooking is safe.

The project combines:

- A FastAPI backend for managing appointment slots.
- A simple web dashboard served by FastAPI.
- An XGBoost model trained to estimate no-show risk.
- Backtesting and Monte Carlo scripts that compare a traditional agenda with an AI-assisted smart-slotting strategy.

The core idea is to reduce wasted hospital appointment slots without creating too many collisions in the waiting room.

## Project Context

Hospitals often lose capacity because some patients do not attend scheduled appointments. Traditional scheduling leaves those gaps empty. This project explores a smarter scheduling policy:

1. Estimate each patient's probability of absence.
2. Allow normal booking when a slot is empty.
3. Allow overbooking only when the mathematical risk is acceptable.
4. Block overbooking when two patients are too likely to attend at the same time.

The overbooking rule used by the API is conservative:

- `prob_ambos_vienen < 0.25`
- `prob_al_menos_uno_venga > 0.8`
- Maximum of two patients per time slot.

In other words, the system only doubles a slot when the risk of both patients arriving is low and the probability that at least one patient uses the slot is high.

## Repository Contents

| Path                                      | Purpose                                                                                                                          |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `src/api.py`                              | Main compatibility entrypoint for the web app. Keeps the old path working while pointing to the consolidated scheduler.          |
| `src/api_2.py`                            | Canonical ML-backed FastAPI app. Loads the XGBoost model, predicts absence probability, and applies smart-slotting rules.        |
| `src/api_chatbot.py`                      | Chatbot-driven app. The patient talks naturally, an LLM extracts model fields, and the app predicts absence risk before booking. |
| `scripts/simulacion.py`                   | Single historical backtest using the dataset and the trained model.                                                              |
| `scripts/mc.py`                           | Monte Carlo simulation over 1000 historical scenarios using the raw XGBoost model.                                               |
| `scripts/mc_2.py`                         | Monte Carlo simulation with isotonic probability calibration in memory before evaluation.                                        |
| `data/dataset_limpio.csv`                 | Clean dataset used by the simulations. Contains 22,106 rows including the target `No-show`.                                      |
| `models/modelo_campeon.json`              | Native XGBoost model used by the API and simulations.                                                                            |
| `models/mejor_modelo_xgb.joblib`          | Serialized XGBoost model artifact, likely from an earlier training/export workflow.                                              |
| `models/calibrated_isotonic_model.joblib` | Serialized calibrated model artifact. The current simulation script calibrates in memory instead of loading this file.           |
| `mikel A`                                 | Small text note file.                                                                                                            |

## Model Inputs

The XGBoost model in `modelo_campeon.json` expects these features:

- `Age`
- `Scholarship`
- `Hipertension`
- `Diabetes`
- `Alcoholism`
- `Handcap`
- `SMS_received`
- `Days_between`
- `Weekend`
- `Ratio_Faltas`
- `Gender_M`
- `Scheduled_Time_of_Day_Evening`
- `Scheduled_Time_of_Day_Morning`

The target column in the dataset is:

- `No-show`

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Web Demo

The main API demo is `src/api_2.py` (and `src/api.py` remains as a compatibility wrapper):

```bash
python3 src/api_2.py
```

Then open:

```text
http://localhost:8000
```

The dashboard lets you enter patient information, choose an appointment time, run the no-show prediction, and book the patient if the smart-slotting policy accepts the appointment.

Useful API routes:

- `GET /api/estado-agenda` returns the current in-memory agenda.
- `POST /api/evaluar-y-reservar` predicts no-show probability and tries to reserve a slot.

Example request:

```bash
curl -X POST http://localhost:8000/api/evaluar-y-reservar \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "Paciente Demo",
    "hora": "09:00",
    "age": 35,
    "days_between": 14,
    "ratio_faltas": 0.2,
    "sms_received": 1,
    "weekend": 0
  }'
```

## Running the Chatbot App

Run the conversational version:

```bash
python3 src/api_chatbot.py
```

Then open:

```text
http://localhost:8100
```

The chatbot keeps the conversation in memory and sends the latest conversation context to Gemini on every turn. Gemini returns both the assistant response and extracted patient data. The app shows the collected admission data, the consultation reason, the payload that would be sent to the XGBoost model, and the estimated no-show risk when enough model fields are available. This chatbot does not reserve appointments.

To enable the Gemini-powered conversation and extraction, set these environment variables before starting the app:

```bash
export GEMINI_API_KEY="your_api_key"
export LLM_API_BASE_URL="https://generativelanguage.googleapis.com/v1beta"
export LLM_MODEL="gemini-3.6-flash"
python3 api_chatbot.py
```

For local development, you can also place the Gemini API key in `api_key.txt`. That file is ignored by Git so it is not uploaded by accident. If both are present, `GEMINI_API_KEY` takes priority over `api_key.txt`. The older `LLM_API_KEY` name is also supported.

If no API key is set, the app still runs with fixed follow-up questions and a small local fallback extractor for demo phrases such as name, age, consultation reason, wait days, SMS, weekend, ratio, and preferred time.

Chatbot routes:

- `POST /api/chat` stores a message, sends the conversation to Gemini, returns the assistant reply, and updates extracted patient data.
- `GET /api/session/{session_id}` returns the current chat session state.

## Running the Simulations

Run a single empirical backtest:

```bash
python3 scripts/simulacion.py
```

Run the raw Monte Carlo simulation:

```bash
python3 scripts/mc.py
```

Run the calibrated Monte Carlo simulation:

```bash
python3 scripts/mc_2.py
```

The simulation scripts compare:

- A traditional hospital agenda with fixed capacity.
- An AI-assisted agenda that overbooks high-risk no-show patients.

The output reports extra patients attended, wasted appointment slots, activated overbookings, and collisions/overlaps.

## Implementation Notes

- Appointment data is stored in memory. Restarting the API resets the agenda.
- The demo agenda contains four 15-minute slots: `09:00`, `09:15`, `09:30`, and `09:45`.
- The API falls back to a simple mathematical estimate if the model file is missing or prediction fails.
- Some comments in the API mention `Days-between`, but the model artifact expects `Days_between`. Keep feature names aligned when modifying the model or API.
- The project is a prototype/demo and is not production-ready for real clinical scheduling without validation, persistence, authentication, privacy controls, and operational monitoring.

## Suggested Next Steps

- Add a `requirements.txt` or `pyproject.toml`.
- Consolidate the API variants into one main application.
- Add automated tests for the booking rules.
- Persist appointments in a database instead of memory.
- Add model training documentation so the model artifacts can be reproduced.
