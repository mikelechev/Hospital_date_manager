# Hospital Date Manager

Hospital Date Manager is a prototype for hospital appointment optimization. It combines an AI-assisted conversational intake flow with traditional scheduling and smart overbooking logic.

The project includes:

- A FastAPI smart-slotting demo in `src/api_2.py`.
- A Streamlit dashboard in `scripts/app.py` for interactive scheduling and smart overbooking simulation.
- A Streamlit chatbot UI in `chatbot/app.py` for conversational patient intake, clinical history upload, and export.
- LLM provider adapters for local Ollama and remote Gemini.
- Export tooling for JSON and CSV conversation exports, plus full data exports with patient state and clinical history.
- Backtesting and Monte Carlo scripts for comparing scheduling strategies.

## Project Context

Hospitals lose capacity when patients miss appointments. This prototype explores a risk-aware scheduling policy:

1. Estimate each patient’s absence probability.
2. Book regular appointments when the slot is empty.
3. Approve a second booking only when the combined risk is low.
4. Block overbooking when two patients are likely to arrive simultaneously.

The current booking rule in `src/api_2.py` uses the following criteria for a second patient in the same slot:

- `prob_ambos_vienen < 0.25`
- `prob_al_menos_uno_venga > 0.8`
- Maximum of two patients per time slot.

## Repository Contents

| Path                                      | Purpose                                                                                           |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `README.md`                               | Unified documentation for the repository.                                                         |
| `scripts/app.py`                          | Streamlit dashboard for interactive scheduling and smart overbooking simulation.                  |
| `chatbot/app.py`                          | Streamlit chatbot UI with conversational intake, clinical history workflows, and export controls. |
| `chatbot/.env.example`                    | Example environment file with local Ollama and Gemini settings.                                   |
| `chatbot/requirements.txt`                | Chatbot-specific Python dependencies.                                                             |
| `chatbot/exports/`                        | Generated conversation exports and saved clinical histories.                                      |
| `chatbot/conversation/`                   | Conversation state, prompt building, extraction, and LLM coordination.                            |
| `chatbot/providers/`                      | Provider adapters for Ollama and Gemini.                                                          |
| `chatbot/config.py`                       | Chatbot runtime configuration and environment loading.                                            |
| `src/api_2.py`                            | Main FastAPI app for slot booking, agenda status, and XGBoost-based risk prediction.              |
| `scripts/simulacion.py`                   | Empirical historical backtest.                                                                    |
| `scripts/mc.py`                           | Monte Carlo simulation using the raw model.                                                       |
| `scripts/mc_2.py`                         | Monte Carlo simulation with isotonic probability calibration in memory.                           |
| `data/dataset_limpio.csv`                 | Clean dataset used by the simulation scripts.                                                     |
| `models/modelo_campeon.json`              | XGBoost model artifact loaded by `src/api_2.py` when available.                                   |
| `models/modelo_definitivo.joblib`         | Joblib model loaded by `scripts/app.py` for the interactive dashboard.                            |
| `models/voting_clf.joblib`               | Alternative serialized model artifact.                                                            |
| `models/calibrated_isotonic_model.joblib` | Serialized calibration artifact for model evaluation workflows.                                   |

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the repo dependencies:

```bash
pip install -r requirements.txt
```

3. Install chatbot dependencies for the Streamlit UI:

```bash
pip install -r chatbot/requirements.txt
```

## Streamlit Dashboard: `scripts/app.py`

The file `scripts/app.py` contains the interactive Hospital Smart Slotting dashboard. It is separate from the conversational chatbot in `chatbot/app.py` and includes:

- Three operating scenarios: fixed traditional scheduling, flexible traditional scheduling, and AI-assisted overbooking.
- A configurable number of patients and simulation days.
- Appointment slots, scheduled breaks, end-of-shift administrative time, and patient no-show outcomes.
- A planned-agenda view and a real-attendance view.
- Animated monthly heatmaps showing empty slots, attendance, delays, breaks, reports, and early arrivals.
- Risk, arrival-time, waiting-room, probability-density, cumulative-probability, and ROI charts.
- The real no-show label from `data/dataset_limpio.csv` and predictions from `models/modelo_definitivo.joblib`.

The dashboard expects these project files to exist:

- `data/dataset_limpio.csv`
- `models/modelo_definitivo.joblib`

Run it from the project root (the folder containing `README.md`):

```bash
streamlit run scripts/app.py
```

Then open:

```text
http://localhost:8501
```

If the virtual environment is being used on Windows, run Streamlit explicitly through it:

```powershell
.\.venv\Scripts\python.exe -m streamlit run scripts/app.py
```

The dashboard loads the joblib model when it starts, so `catboost` must be installed even though it is not imported directly in `scripts/app.py`: the serialized model contains a CatBoost component.

## Running the Chatbot App

Start the Streamlit chatbot UI from the project root:

```bash
streamlit run chatbot/app.py
```

Then open:

```text
http://localhost:8501
```

### Chatbot features

- Conversational patient intake with natural language.
- Clinical history upload or paste.
- Same-provider LLM interpretation for clinical history and chat extraction.
- Export conversation to JSON or CSV.
- Full data export including patient state and clinical history references.
- Optional anonymization of emails, phone numbers, and long numeric identifiers.

### Export behavior

The chatbot export module is implemented in `chatbot/exports/conversation_exporter.py`, with helpers:

- `export_conversation_json(...)`
- `export_conversation_csv(...)`
- `export_full_data_csv(...)`
- `save_clinical_history(...)`

Exports are saved under `chatbot/exports/`, and saved clinical histories are stored in `chatbot/exports/clinical_histories/`.

## Local LLM and GPU Configuration

Create `chatbot/.env` from `chatbot/.env.example` when using Ollama or Gemini.

`chatbot/.env.example` includes:

- `DEFAULT_PROVIDER` (default: `ollama`)
- `DEFAULT_MODEL` (default: `qwen3:8b`)
- `GEMINI_API_KEY`
- `OLLAMA_BASE_URL`
- `OLLAMA_NO_CLOUD`
- `OLLAMA_FLASH_ATTENTION`
- `OLLAMA_IGPU_ENABLE`
- `OLLAMA_MAX_LOADED_MODELS`
- `OLLAMA_GPU_OVERHEAD`

Use these settings to enable local GPU inference for an RTX 4060 via Ollama, while keeping the same model.

### Download the Qwen Ollama model locally

If you want to run the chatbot with the same model locally, install and pull it with:

```bash
ollama pull qwen3:8b
```

If you want to create a local alias for the repo-specific setup:

```bash
ollama create hospital-model -f ./models/ollama/Modelfile
```

You can verify the installed model with:

```bash
ollama list
```

## Running the Web API Demo

Run the FastAPI app:

```bash
python3 src/api_2.py
```

Then open:

```text
http://localhost:8000
```

### API routes

- `GET /api/estado-agenda` — returns the current appointment agenda.
- `POST /api/evaluar-y-reservar` — evaluates no-show risk and tries to reserve a slot.

### Example request

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

### Model loading behavior

`src/api_2.py` loads `models/modelo_campeon.json` if present. If the file is missing, the API uses a fallback heuristic to compute absence probability.

## Running the Simulations

- `python3 scripts/simulacion.py`
- `python3 scripts/mc.py`
- `python3 scripts/mc_2.py`

These scripts compare traditional scheduling with AI-assisted overbooking and simulated risk-based performance.

## Model Inputs

The XGBoost model expects these features:

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

Target column:

- `No-show`

## Notes

- The chatbot UI is the primary conversational interface. The `src/chatbot.py` file is an empty compatibility placeholder.
- Session, agenda, and extraction state are stored in memory. Restarting an app resets state.
- `api_key.txt` and `.env` are ignored by Git and should contain secrets only.
- This project is a prototype and not production-ready for real clinical deployment without validation, persistence, authentication, and privacy controls.

## Suggested Next Steps

- Add automated tests for booking rules, chat extraction, and exports.
- Persist appointments, sessions, and clinical history in a database.
- Add CI workflows for linting, formatting, and runtime validation.
- Document model training and dataset preparation.
