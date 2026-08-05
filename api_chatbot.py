# -*- coding: utf-8 -*-
"""
Chatbot for patient intake.

The patient talks naturally. Gemini keeps the conversation going and extracts
the structured fields needed by the no-show model, plus the consultation reason.
This app does not book appointments.
"""

import json
import os
import random
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import xgboost as xgb


MODEL_PATH = "modelo_campeon.json"
API_KEY_PATH = Path("api_key.txt")
LLM_API_BASE_URL = os.getenv(
    "LLM_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")

FEATURES = [
    "Age",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "Days_between",
    "Weekend",
    "Ratio_Faltas",
    "Gender_M",
    "Scheduled_Time_of_Day_Evening",
    "Scheduled_Time_of_Day_Morning",
]

FIELD_DEFAULTS = {
    "scholarship": 0,
    "hipertension": 0,
    "diabetes": 0,
    "alcoholism": 0,
    "handcap": 0,
    "gender_m": 0,
    "scheduled_morning": 1,
    "scheduled_evening": 0,
}

MODEL_REQUIRED_FIELDS = [
    "age",
    "days_between",
    "ratio_faltas",
    "sms_received",
    "weekend",
]

INTAKE_REQUIRED_FIELDS = [
    "nombre",
    "motivo_consulta",
    *MODEL_REQUIRED_FIELDS,
]


def load_llm_api_key():
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    if env_key:
        return env_key.strip()

    if API_KEY_PATH.exists():
        return API_KEY_PATH.read_text(encoding="utf-8").strip()

    return None


LLM_API_KEY = load_llm_api_key()


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. The app will use a mathematical fallback.")
        return None

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print("XGBoost model loaded from modelo_campeon.json")
    return model


modelo_ia = load_model()


class ChatRequest(BaseModel):
    session_id: str
    message: str


app = FastAPI(title="Hospital Patient Intake Chatbot")
sessions: dict[str, dict[str, Any]] = {}


def normalize_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)

    text = str(value).strip().lower()
    if text in {"1", "yes", "si", "sí", "true", "y", "s"}:
        return 1
    if text in {"0", "no", "false", "n"}:
        return 0
    return None


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def clean_extracted_data(data):
    cleaned = {}
    if not isinstance(data, dict):
        return cleaned

    for key, value in data.items():
        if value is None or value == "":
            continue
        try:
            if key in {
                "age",
                "days_between",
                "scholarship",
                "hipertension",
                "diabetes",
                "alcoholism",
                "handcap",
                "gender_m",
                "scheduled_morning",
                "scheduled_evening",
                "sms_received",
                "weekend",
            }:
                if key in {"sms_received", "weekend"}:
                    parsed_bool = normalize_bool(value)
                    if parsed_bool is not None:
                        cleaned[key] = parsed_bool
                else:
                    cleaned[key] = int(value)
            elif key == "ratio_faltas":
                cleaned[key] = clamp(float(value), 0.0, 1.0)
            elif key in {"nombre", "motivo_consulta", "hora_preferida"}:
                cleaned[key] = str(value).strip()
        except (TypeError, ValueError):
            continue
    return cleaned


def fallback_extract(text):
    lowered = text.lower()
    extracted = {}

    name_match = re.search(
        r"(?:me llamo|soy|mi nombre es|my name is|i am)\s+([a-zA-ZÁÉÍÓÚÜÑáéíóúüñ ]{2,40})",
        text,
    )
    if name_match:
        name = re.split(
            r"\s+(?:y|and)\s+|,|\.",
            name_match.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        extracted["nombre"] = name.strip(" .")

    age_match = re.search(r"(\d{1,3})\s*(?:anos|años|years old|year old)", lowered)
    if age_match:
        extracted["age"] = int(age_match.group(1))

    wait_match = re.search(
        r"(\d{1,3})\s*(?:dias|días|days).{0,24}(?:espera|wait|waiting|cita)",
        lowered,
    )
    if not wait_match:
        wait_match = re.search(
            r"(?:espera|wait|waiting|cita).{0,24}(\d{1,3})\s*(?:dias|días|days)",
            lowered,
        )
    if wait_match:
        extracted["days_between"] = int(wait_match.group(1))

    ratio_match = re.search(
        r"(?:ratio|faltas|missed|no[- ]?show).{0,20}(0?\.\d+|1(?:\.0)?)",
        lowered,
    )
    if ratio_match:
        extracted["ratio_faltas"] = float(ratio_match.group(1))

    time_match = re.search(r"\b(\d{1,2}:[0-5]\d)\b", text)
    if time_match:
        extracted["hora_preferida"] = time_match.group(1)

    reason_match = re.search(
        r"(?:motivo|consulta|vengo|voy|quiero consultar|es por|porque)\s+(?:es\s+|por\s+|a\s+)?(.{8,160})",
        text,
        flags=re.IGNORECASE,
    )
    if reason_match:
        extracted["motivo_consulta"] = reason_match.group(1).strip(" .")

    if "sms" in lowered:
        if any(
            word in lowered
            for word in ["no sms", "sin sms", "no recibi", "no recibí"]
        ):
            extracted["sms_received"] = 0
        elif any(word in lowered for word in ["sms", "mensaje", "recordatorio"]):
            extracted["sms_received"] = 1

    if any(word in lowered for word in ["sabado", "sábado", "domingo", "weekend"]):
        extracted["weekend"] = 1
    elif any(
        word in lowered
        for word in [
            "lunes",
            "martes",
            "miercoles",
            "miércoles",
            "jueves",
            "viernes",
            "weekday",
            "entre semana",
        ]
    ):
        extracted["weekend"] = 0

    return extracted


def call_gemini_json(prompt):
    if not LLM_API_KEY:
        return None

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.85,
            "topP": 0.95,
        },
    }
    url = f"{LLM_API_BASE_URL}/models/{LLM_MODEL}:generateContent"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-goog-api-key": LLM_API_KEY,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = raw["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(content)
    except (
        KeyError,
        IndexError,
        json.JSONDecodeError,
        urllib.error.URLError,
        TimeoutError,
    ) as exc:
        print(f"Gemini chat failed, using local fallback only: {exc}")
        return None


def missing_model_fields(patient_data):
    return [field for field in MODEL_REQUIRED_FIELDS if field not in patient_data]


def missing_intake_fields(patient_data):
    return [field for field in INTAKE_REQUIRED_FIELDS if field not in patient_data]


def build_chat_prompt(messages, current_data):
    conversation = "\n".join(
        f"{message['role']}: {message['content']}" for message in messages[-14:]
    )
    missing_now = missing_intake_fields(current_data)
    payload = {
        "current_data": current_data,
        "missing_intake_fields": missing_now,
        "missing_model_fields": missing_model_fields(current_data),
        "conversation": conversation,
    }
    return (
        "Eres el asistente de admision de un hospital, hablando por chat en "
        "espanol con un paciente real. No reservas citas ni das diagnosticos "
        "ni consejos clinicos. Tu tarea de fondo es completar, sin que se "
        "note como si fuera un formulario, los datos que se listan mas "
        "abajo.\n\n"
        "COMO SONAR NATURAL Y ADAPTARTE:\n"
        "- Escribe como una persona cercana y profesional, no como un guion "
        "leido. Mira el historial de la conversacion y varia las palabras: "
        "no repitas la misma frase o estructura de pregunta que ya usaste.\n"
        "- Reacciona primero a lo que el paciente acaba de decir (un dato, "
        "una duda, una queja, un saludo, una broma) antes de seguir "
        "recopilando datos. Un reconocimiento breve basta.\n"
        "- Si el paciente da varios datos en un mismo mensaje, o los da en "
        "otro orden o de forma indirecta, extraelos todos de una vez y no "
        "vuelvas a preguntar por algo que ya contesto.\n"
        "- Si el paciente pregunta algo (por ejemplo por que necesitas un "
        "dato, o que pasara con su informacion), respondele con una frase "
        "honesta y breve y luego retoma la conversacion con naturalidad, "
        "sin forzar la siguiente pregunta si no encaja en ese momento.\n"
        "- Si el mensaje es solo un saludo, small talk, o muestra nervios o "
        "incomodidad, respondele como persona antes de pedir mas datos.\n"
        "- Adapta el tono al del paciente: si escribe formal, se formal; si "
        "escribe informal o con prisa, se breve y directo; si suena "
        "preocupado, se calido y tranquilizador.\n"
        "- Haz como maximo una pregunta nueva por turno, encadenada de "
        "forma natural a lo anterior en vez de listada aparte.\n"
        "- Nunca inventes ni asumas un dato que el paciente no dio.\n"
        "- Si ya tienes todos los campos requeridos, cierra con calidez y "
        "brevedad, y di que los datos estan listos para el sistema, sin "
        "recitar cada campo como una lista.\n\n"
        "Campos requeridos de admision: nombre, motivo_consulta, age, "
        "days_between, ratio_faltas, sms_received, weekend.\n"
        "Campos para el modelo: age, days_between, ratio_faltas, sms_received, "
        "weekend, scholarship, hipertension, diabetes, alcoholism, handcap, "
        "gender_m, scheduled_morning, scheduled_evening.\n"
        "Campos extra utiles: hora_preferida.\n"
        f"Campos que todavia faltan ahora mismo: {missing_now}.\n\n"
        "Valores permitidos:\n"
        "- ratio_faltas: numero entre 0 y 1.\n"
        "- sms_received y weekend: 0 o 1.\n"
        "- Si un dato no esta claro, usa null. No inventes valores.\n\n"
        "Devuelve exclusivamente JSON valido con esta forma:\n"
        "{"
        '"assistant_message": "respuesta natural y breve al paciente", '
        '"extracted_data": {'
        '"nombre": null, "motivo_consulta": null, "age": null, '
        '"days_between": null, "ratio_faltas": null, "sms_received": null, '
        '"weekend": null, "hora_preferida": null, "scholarship": null, '
        '"hipertension": null, "diabetes": null, "alcoholism": null, '
        '"handcap": null, "gender_m": null, "scheduled_morning": null, '
        '"scheduled_evening": null'
        "}"
        "}\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


CLOSING_MESSAGES = [
    "Perfecto, con eso ya tengo lo que necesitaba. Gracias por la paciencia, "
    "los datos estan listos para el sistema.",
    "Genial, ya esta todo completo. Gracias por contarmelo, lo dejo listo "
    "para el sistema.",
    "Listo, ya tengo todo lo necesario. Gracias, los datos quedan "
    "preparados para el sistema.",
]

FIELD_QUESTIONS = {
    "nombre": [
        "Para empezar, como te llamas?",
        "Con quien tengo el gusto de hablar?",
    ],
    "motivo_consulta": [
        "Cuentame, por que motivo vienes a consulta?",
        "Que es lo que te trae por aqui?",
    ],
    "age": [
        "Cuantos anos tienes?",
        "Me confirmas tu edad?",
    ],
    "days_between": [
        "Cuantos dias faltan hasta la cita o consulta prevista?",
        "En cuantos dias es la consulta?",
    ],
    "ratio_faltas": [
        "Mas o menos, de 0 a 1, cual dirias que es tu ratio historico de "
        "faltas a citas?",
        "Con que frecuencia sueles faltar a tus citas, en una escala de 0 a 1?",
    ],
    "sms_received": [
        "Recibiste un SMS de recordatorio para esta cita?",
        "Te llego algun mensaje recordandote la cita?",
    ],
    "weekend": [
        "La cita prevista cae en fin de semana?",
        "Esa consulta es entre semana o en fin de semana?",
    ],
}


def next_question(patient_data):
    missing = missing_intake_fields(patient_data)
    if not missing:
        return random.choice(CLOSING_MESSAGES)

    return random.choice(FIELD_QUESTIONS[missing[0]])


def fallback_chat_turn(messages, current_data):
    fallback_data = fallback_extract(messages[-1]["content"])
    merged = dict(current_data)
    merged.update(clean_extracted_data(fallback_data))
    return next_question(merged), merged


def llm_chat_turn(messages, current_data):
    gemini_response = call_gemini_json(build_chat_prompt(messages, current_data))
    if not gemini_response:
        return fallback_chat_turn(messages, current_data)

    merged = dict(current_data)
    merged.update(clean_extracted_data(gemini_response.get("extracted_data", {})))
    fallback_data = fallback_extract(messages[-1]["content"])
    for key, value in clean_extracted_data(fallback_data).items():
        if key not in merged:
            merged[key] = value

    assistant_message = str(gemini_response.get("assistant_message") or "").strip()
    if not assistant_message:
        assistant_message = next_question(merged)

    return assistant_message, merged


def patient_dataframe(patient_data):
    data = {**FIELD_DEFAULTS, **patient_data}
    return pd.DataFrame(
        [
            {
                "Age": data["age"],
                "Scholarship": data["scholarship"],
                "Hipertension": data["hipertension"],
                "Diabetes": data["diabetes"],
                "Alcoholism": data["alcoholism"],
                "Handcap": data["handcap"],
                "SMS_received": data["sms_received"],
                "Days_between": data["days_between"],
                "Weekend": data["weekend"],
                "Ratio_Faltas": data["ratio_faltas"],
                "Gender_M": data["gender_m"],
                "Scheduled_Time_of_Day_Evening": data["scheduled_evening"],
                "Scheduled_Time_of_Day_Morning": data["scheduled_morning"],
            }
        ],
        columns=FEATURES,
    )


def predict_absence(patient_data):
    if missing_model_fields(patient_data):
        return None

    if modelo_ia is not None:
        try:
            return float(modelo_ia.predict_proba(patient_dataframe(patient_data))[0][1])
        except Exception as exc:
            print(f"Model prediction failed, using fallback: {exc}")

    return min(
        0.95,
        (float(patient_data.get("ratio_faltas", 0.0)) * 0.7)
        + (int(patient_data.get("days_between", 0)) * 0.01),
    )


def model_payload(patient_data):
    if missing_model_fields(patient_data):
        return None
    return patient_dataframe(patient_data).iloc[0].to_dict()


def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "Hola, soy el asistente de admision. Te hare unas "
                        "preguntas para preparar los datos antes de la consulta. "
                        "Como te llamas?"
                    ),
                }
            ],
            "patient_data": {},
        }
    return sessions[session_id]


def session_payload(session):
    patient_data = session["patient_data"]
    return {
        "messages": session["messages"],
        "patient_data": patient_data,
        "model_payload": model_payload(patient_data),
        "missing_intake": missing_intake_fields(patient_data),
        "missing_model": missing_model_fields(patient_data),
        "ready": not missing_intake_fields(patient_data),
        "model_ready": not missing_model_fields(patient_data),
        "probabilidad": predict_absence(patient_data),
    }


@app.get("/", response_class=HTMLResponse)
def chatbot_ui():
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Admision · Chat</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-app: #eef3ef;
      --bg-panel: #ffffff;
      --ink: #1f2e2a;
      --ink-muted: #5c6b67;
      --accent: #3f6659;
      --accent-strong: #2c4a40;
      --accent-soft: #dcebe3;
      --accent-2: #c98a3a;
      --accent-2-soft: #f4e3c8;
      --border: #dbe4de;
      --danger: #c1594a;
    }
    * { font-family: 'Inter', sans-serif; }
    .font-display { font-family: 'Fraunces', serif; }
    .font-mono { font-family: 'JetBrains Mono', monospace; }
    body { background: var(--bg-app); color: var(--ink); }

    @keyframes rise-in {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .msg-row { animation: rise-in .35s ease both; }

    @keyframes dot-bounce {
      0%, 60%, 100% { transform: translateY(0); opacity: .5; }
      30% { transform: translateY(-4px); opacity: 1; }
    }
    .dot { animation: dot-bounce 1.1s ease-in-out infinite; }
    .dot:nth-child(2) { animation-delay: .15s; }
    .dot:nth-child(3) { animation-delay: .3s; }

    .field-card { transition: background-color .35s ease, border-color .35s ease, transform .2s ease; }
    .field-card.filled { background: var(--accent-soft); border-color: transparent; }

    #pulse-path {
      fill: none;
      stroke: var(--accent);
      stroke-width: 2.5;
      stroke-linecap: round;
      stroke-linejoin: round;
      transition: stroke-dashoffset .8s cubic-bezier(.4,0,.2,1);
    }
    #pulse-track { fill: none; stroke: var(--border); stroke-width: 2.5; }

    #risk-ring-fg {
      transition: stroke-dashoffset .8s cubic-bezier(.4,0,.2,1), stroke .4s ease;
      stroke-linecap: round;
    }

    @media (prefers-reduced-motion: reduce) {
      .msg-row, .dot { animation: none !important; }
      #pulse-path, #risk-ring-fg, .field-card { transition: none !important; }
    }
  </style>
</head>
<body class="min-h-screen">
  <main class="mx-auto grid min-h-screen max-w-7xl grid-cols-1 gap-4 p-4 lg:grid-cols-[minmax(0,1fr)_400px] lg:gap-6 lg:p-6">

    <section class="flex min-h-[80vh] flex-col overflow-hidden rounded-3xl border shadow-sm lg:min-h-0" style="background: var(--bg-panel); border-color: var(--border);">
      <header class="flex items-center gap-3 border-b px-6 py-5" style="border-color: var(--border);">
        <div class="flex h-10 w-10 items-center justify-center rounded-full" style="background: var(--accent-soft);">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent-strong)" stroke-width="2" stroke-linecap="round"><path d="M12 3v18M3 12h18"/></svg>
        </div>
        <div>
          <h1 class="font-display text-xl" style="color: var(--accent-strong);">Admisión, antes de tu consulta</h1>
          <p class="mt-0.5 text-sm" style="color: var(--ink-muted);">Una charla breve para preparar tus datos. No reservamos citas aquí.</p>
        </div>
      </header>

      <div id="messages" class="flex-1 space-y-4 overflow-y-auto px-6 py-6"></div>

      <div id="typing" class="hidden items-center gap-2 px-6 pb-2">
        <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm" style="background: var(--accent-soft); color: var(--accent-strong);">+</div>
        <div class="flex items-center gap-1 rounded-2xl rounded-bl-sm px-4 py-3" style="background: var(--accent-soft);">
          <span class="dot h-1.5 w-1.5 rounded-full" style="background: var(--accent-strong);"></span>
          <span class="dot h-1.5 w-1.5 rounded-full" style="background: var(--accent-strong);"></span>
          <span class="dot h-1.5 w-1.5 rounded-full" style="background: var(--accent-strong);"></span>
        </div>
      </div>

      <form id="chat-form" class="border-t p-4" style="border-color: var(--border);">
        <div class="flex gap-2">
          <input id="message-input" class="min-h-12 flex-1 rounded-full border bg-transparent px-5 text-sm outline-none transition focus:ring-2 disabled:opacity-60" style="border-color: var(--border); color: var(--ink);" autocomplete="off" placeholder="Escribe con tranquilidad..." />
          <button id="send-button" class="min-h-12 rounded-full px-6 text-sm font-semibold text-white shadow-sm transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50" style="background: var(--accent);" type="submit">Enviar</button>
        </div>
      </form>
    </section>

    <aside class="flex flex-col gap-4 lg:gap-6">

      <section class="rounded-3xl border p-5 shadow-sm" style="background: var(--bg-panel); border-color: var(--border);">
        <div class="flex items-center justify-between">
          <h2 class="font-display text-base" style="color: var(--accent-strong);">Progreso de admisión</h2>
          <span id="status-pill" class="rounded-full px-3 py-1 text-xs font-medium" style="background: var(--accent-2-soft); color: #8a5a1f;">Incompleto</span>
        </div>
        <svg id="pulse-svg" viewBox="0 0 300 40" class="mt-4 w-full">
          <path id="pulse-track" d="M0 20 L90 20 L100 5 L112 35 L124 20 L300 20"/>
          <path id="pulse-path" d="M0 20 L90 20 L100 5 L112 35 L124 20 L300 20"/>
        </svg>
        <p id="progress-label" class="mt-2 text-xs" style="color: var(--ink-muted);">0 de 7 datos recogidos</p>

        <dl id="fields" class="mt-4 grid grid-cols-2 gap-2.5 text-sm"></dl>
      </section>

      <section class="rounded-3xl border p-5 shadow-sm" style="background: var(--bg-panel); border-color: var(--border);">
        <h2 class="font-display text-base" style="color: var(--accent-strong);">Salida para el modelo</h2>
        <div class="mt-4 flex items-center gap-4 rounded-2xl p-4" style="background: var(--bg-app);">
          <svg width="72" height="72" viewBox="0 0 72 72">
            <circle cx="36" cy="36" r="30" fill="none" stroke="var(--border)" stroke-width="7"/>
            <circle id="risk-ring-fg" cx="36" cy="36" r="30" fill="none" stroke="var(--accent)" stroke-width="7" stroke-dasharray="188.5" stroke-dashoffset="188.5" transform="rotate(-90 36 36)"/>
          </svg>
          <div>
            <div class="text-xs" style="color: var(--ink-muted);">Probabilidad estimada de ausencia</div>
            <div id="risk" class="mt-1 font-display text-2xl" style="color: var(--accent-strong);">--</div>
          </div>
        </div>
        <pre id="model-json" class="font-mono mt-4 max-h-56 overflow-auto rounded-2xl border p-3 text-xs" style="border-color: var(--border); color: var(--ink-muted);"></pre>
      </section>

      <section class="rounded-3xl border p-5 shadow-sm" style="background: var(--bg-panel); border-color: var(--border);">
        <h2 class="font-display text-base" style="color: var(--accent-strong);">Todavía falta</h2>
        <ul id="missing-list" class="mt-3 flex flex-wrap gap-2 text-sm"></ul>
      </section>

    </aside>
  </main>

  <script>
    const sessionId = crypto.randomUUID();
    const messagesEl = document.getElementById("messages");
    const fieldsEl = document.getElementById("fields");
    const form = document.getElementById("chat-form");
    const input = document.getElementById("message-input");
    const sendButton = document.getElementById("send-button");
    const typing = document.getElementById("typing");
    const riskEl = document.getElementById("risk");
    const riskRing = document.getElementById("risk-ring-fg");
    const modelJson = document.getElementById("model-json");
    const statusPill = document.getElementById("status-pill");
    const missingList = document.getElementById("missing-list");
    const pulsePath = document.getElementById("pulse-path");
    const progressLabel = document.getElementById("progress-label");
    let currentMessages = [];

    const FIELD_LABELS = {
      nombre: "Nombre",
      motivo_consulta: "Motivo",
      age: "Edad",
      days_between: "Días espera",
      ratio_faltas: "Ratio faltas",
      sms_received: "SMS",
      weekend: "Fin de semana",
      hora_preferida: "Hora pref."
    };
    const INTAKE_FIELDS = ["nombre", "motivo_consulta", "age", "days_between", "ratio_faltas", "sms_received", "weekend"];

    const pulseLength = pulsePath.getTotalLength();
    pulsePath.style.strokeDasharray = String(pulseLength);
    pulsePath.style.strokeDashoffset = String(pulseLength);

    const riskCircumference = 2 * Math.PI * 30;

    function renderMessages(messages) {
      messagesEl.innerHTML = "";
      for (const message of messages) {
        const isUser = message.role === "user";
        const row = document.createElement("div");
        row.className = `msg-row flex items-end gap-2 ${isUser ? "justify-end" : "justify-start"}`;

        if (!isUser) {
          const avatar = document.createElement("div");
          avatar.className = "flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm";
          avatar.style.background = "var(--accent-soft)";
          avatar.style.color = "var(--accent-strong)";
          avatar.textContent = "+";
          row.appendChild(avatar);
        }

        const bubble = document.createElement("div");
        bubble.className = `max-w-[78%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${isUser ? "rounded-br-sm text-white" : "rounded-bl-sm"}`;
        bubble.style.background = isUser ? "var(--accent-2)" : "var(--accent-soft)";
        if (!isUser) bubble.style.color = "var(--ink)";
        bubble.textContent = message.content;
        row.appendChild(bubble);
        messagesEl.appendChild(row);
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function renderFields(data) {
      fieldsEl.innerHTML = "";
      for (const [key, label] of Object.entries(FIELD_LABELS)) {
        const hasValue = data[key] !== undefined && data[key] !== null && data[key] !== "";
        const value = hasValue ? data[key] : "Pendiente";
        const node = document.createElement("div");
        node.className = `field-card rounded-xl border px-3 py-2 ${key === "motivo_consulta" ? "col-span-2" : ""} ${hasValue ? "filled" : ""}`;
        node.style.borderColor = "var(--border)";
        const dt = document.createElement("dt");
        dt.className = "flex items-center justify-between text-xs";
        dt.style.color = "var(--ink-muted)";
        dt.textContent = label;
        if (hasValue) {
          const check = document.createElement("span");
          check.textContent = "✓";
          check.style.color = "var(--accent-strong)";
          dt.appendChild(check);
        }
        const dd = document.createElement("dd");
        dd.className = "mt-0.5 break-words text-sm font-medium";
        dd.style.color = hasValue ? "var(--ink)" : "var(--ink-muted)";
        dd.textContent = value;
        node.append(dt, dd);
        fieldsEl.appendChild(node);
      }
    }

    function setLoading(isLoading) {
      typing.classList.toggle("hidden", !isLoading);
      typing.classList.toggle("flex", isLoading);
      input.disabled = isLoading;
      sendButton.disabled = isLoading;
      if (isLoading) messagesEl.scrollTop = messagesEl.scrollHeight;
      if (!isLoading) input.focus();
    }

    function updateState(payload) {
      currentMessages = payload.messages;
      renderMessages(currentMessages);
      renderFields(payload.patient_data);

      const filledCount = INTAKE_FIELDS.filter((key) => {
        const value = payload.patient_data[key];
        return value !== undefined && value !== null && value !== "";
      }).length;
      const pct = filledCount / INTAKE_FIELDS.length;
      pulsePath.style.strokeDashoffset = String(pulseLength * (1 - pct));
      progressLabel.textContent = `${filledCount} de ${INTAKE_FIELDS.length} datos recogidos`;

      if (payload.probabilidad === null) {
        riskEl.textContent = "--";
        riskRing.style.strokeDashoffset = String(riskCircumference);
      } else {
        const p = payload.probabilidad;
        riskEl.textContent = `${(p * 100).toFixed(1)}%`;
        riskRing.style.strokeDashoffset = String(riskCircumference * (1 - p));
        riskRing.style.stroke = p > 0.6 ? "var(--danger)" : p > 0.3 ? "var(--accent-2)" : "var(--accent)";
      }

      modelJson.textContent = payload.model_payload
        ? JSON.stringify(payload.model_payload, null, 2)
        : "Aún faltan campos para generar el payload del modelo.";

      statusPill.textContent = payload.ready ? "Completo" : "Incompleto";
      statusPill.style.background = payload.ready ? "var(--accent-soft)" : "var(--accent-2-soft)";
      statusPill.style.color = payload.ready ? "var(--accent-strong)" : "#8a5a1f";

      missingList.innerHTML = "";
      if (!payload.missing_intake.length) {
        const chip = document.createElement("li");
        chip.className = "rounded-full px-3 py-1 text-xs";
        chip.style.background = "var(--accent-soft)";
        chip.style.color = "var(--accent-strong)";
        chip.textContent = "Nada, ya está todo";
        missingList.appendChild(chip);
      } else {
        for (const item of payload.missing_intake) {
          const chip = document.createElement("li");
          chip.className = "rounded-full border px-3 py-1 text-xs";
          chip.style.borderColor = "var(--border)";
          chip.style.color = "var(--ink-muted)";
          chip.textContent = FIELD_LABELS[item] || item;
          missingList.appendChild(chip);
        }
      }
    }

    async function sendMessage(message) {
      const optimisticMessages = [...currentMessages, {role: "user", content: message}];
      currentMessages = optimisticMessages;
      renderMessages(optimisticMessages);
      setLoading(true);
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({session_id: sessionId, message})
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        updateState(await response.json());
      } catch (error) {
        currentMessages = [
          ...optimisticMessages,
          {role: "assistant", content: "No he podido procesar el mensaje. Revisa la conexión o la API key e inténtalo de nuevo."}
        ];
        renderMessages(currentMessages);
      } finally {
        setLoading(false);
      }
    }

    async function loadSession() {
      const response = await fetch(`/api/session/${sessionId}`);
      updateState(await response.json());
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      input.value = "";
      await sendMessage(message);
    });

    loadSession();
  </script>
</body>
</html>
        """
    )


@app.post("/api/chat")
def chat(request: ChatRequest):
    session = get_session(request.session_id)
    if request.message.strip():
        session["messages"].append({"role": "user", "content": request.message.strip()})
        answer, session["patient_data"] = llm_chat_turn(
            session["messages"], session["patient_data"]
        )
        session["messages"].append({"role": "assistant", "content": answer})

    return session_payload(session)


@app.get("/api/session/{session_id}")
def session_state(session_id: str):
    return session_payload(get_session(session_id))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)