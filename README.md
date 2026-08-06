# Hospital Date Manager

Hospital Date Manager is a prototype for hospital appointment optimization. It predicts the probability that a patient will miss an appointment and uses that probability to decide when a controlled overbooking is safe.

The project combines:

- A FastAPI backend for appointment scheduling and risk-based slotting.
- A conversational Streamlit chatbot UI for collecting patient data and clinical history.
- An XGBoost model trained to estimate no-show risk.
- Export tooling for conversation and clinical history data.
- Backtesting and Monte Carlo scripts that compare traditional scheduling with AI-assisted smart-slotting.

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

## Repository Contents

| Path                                      | Purpose                                                                                                                   |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `chatbot/`                                | Streamlit conversational app with LLM patient intake, clinical history interpretation, and export support.                |
| `chatbot/app.py`                          | Streamlit UI entrypoint for the chatbot and clinical history workflow.                                                    |
| `chatbot/.env.example`                    | Example environment configuration for Ollama/GPU settings and local app options.                                          |
| `chatbot/exports/`                        | Generated conversation and clinical history export files.                                                                 |
| `chatbot/conversation/`                   | Conversation management, prompt building, state, and extraction logic.                                                    |
| `chatbot/providers/`                      | LLM provider adapters for Ollama and Gemini.                                                                              |
| `src/api_2.py`                            | Canonical ML-backed FastAPI app. Loads the XGBoost model, predicts absence probability, and applies smart-slotting rules. |
| `src/chatbot.py`                          | Compatibility wrapper or alternate chatbot entrypoint.                                                                    |
| `scripts/simulacion.py`                   | Single historical backtest using the dataset and the trained model.                                                       |
| `scripts/mc.py`                           | Monte Carlo simulation over 1000 historical scenarios using the raw XGBoost model.                                        |
| `scripts/mc_2.py`                         | Monte Carlo simulation with isotonic probability calibration in memory before evaluation.                                 |
| `data/dataset_limpio.csv`                 | Clean dataset used by the simulations. Contains 22,106 rows including the target `No-show`.                               |
| `models/modelo_campeon.json`              | Native XGBoost model used by the API and simulations.                                                                     |
| `models/mejor_modelo_xgb.joblib`          | Serialized XGBoost model artifact, likely from an earlier training/export workflow.                                       |
| `models/calibrated_isotonic_model.joblib` | Serialized calibrated model artifact. The current simulation script calibrates in memory instead of loading this file.    |

## Model Inputs

The XGBoost model in `models/modelo_campeon.json` expects these features:

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

## Running the Chatbot App

Run the conversational Streamlit app from the `chatbot` folder:

```bash
cd chatbot
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The chatbot interface supports:

- Natural patient conversation and structured patient data extraction.
- Clinical history upload or text paste.
- LLM-based interpretation of clinical history using the same provider as the chat flow.
- Exporting conversation data to JSON or CSV.
- Optional anonymization of exported personal identifiers.

### Local LLM and GPU settings

Copy or edit `chatbot/.env.example` to create `chatbot/.env` and configure Ollama GPU acceleration. The app also supports Gemini via environment variables if you prefer remote LLM inference.

## Running the Web API Demo

The main API demo is `src/api_2.py`:

```bash
python3 src/api_2.py
```

Then open:

```text
http://localhost:8000
```

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
- An AI-assisted agenda that overbooks lower-risk no-show patients.

## Implementation Notes

- Appointment and chatbot session state are stored in memory. Restarting an app resets the agenda and conversation state.
- Conversation exports are saved under `chatbot/exports/`.
- Uploaded clinical histories are stored under `chatbot/exports/clinical_histories/`.
- `api_key.txt` and `.env` are ignored by Git and should be used for secrets only.
- This project is a prototype; it is not production-ready for clinical deployment without persistence, authentication, privacy safeguards, and validation.

## Suggested Next Steps

- Add automated tests for booking rules and export workflows.
- Persist appointments and session state in a database.
- Add CI checks for formatting, linting, and runtime validation.
- Document model training and reproducibility.
