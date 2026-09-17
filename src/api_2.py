# -*- coding: utf-8 -*-
"""Main FastAPI app for the hospital scheduling demo."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
# Antes usaba modelo_campeon.json (XGBoost suelto, 13 variables, sin
# calibrar). Se cambia a modelo_definitivo.joblib para que esta API use el
# MISMO modelo que el dashboard (scripts/app.py), el chatbot y el portal de
# paciente (scripts/patient.py) -- tener dos modelos distintos calculando
# el riesgo del mismo paciente en distintas partes del sistema es una
# inconsistencia real, no solo cosmetica: un modelo sin calibrar dispara
# overbooking muchas mas veces de lo previsto para el mismo umbral (ver
# Claude outputs/datos_tecnicos_completados.md, Hallazgo #2).
MODEL_PATH = MODELS_DIR / "modelo_definitivo.joblib"


class AgendaInteligente:
    def __init__(self):
        self.slots = {
            "09:00": {"pacientes": [], "prob_simultaneous_absence": 1.0},
            "09:15": {"pacientes": [], "prob_simultaneous_absence": 1.0},
            "09:30": {"pacientes": [], "prob_simultaneous_absence": 1.0},
            "09:45": {"pacientes": [], "prob_simultaneous_absence": 1.0},
        }

    def _insertar(self, hora, nombre_paciente, prob_ausencia):
        self.slots[hora]["pacientes"].append(
            {"name": nombre_paciente, "prob_ausencia": prob_ausencia}
        )
        current_prob_product = 1.0
        for paciente in self.slots[hora]["pacientes"]:
            current_prob_product *= paciente["prob_ausencia"]
        self.slots[hora]["prob_simultaneous_absence"] = current_prob_product

    def book_slot(self, hora, nombre_paciente, prob_ausencia):
        if hora not in self.slots:
            return False, "❌ Hora no válida en el sistema."

        info = self.slots[hora]
        if len(info["pacientes"]) == 0:
            self._insertar(hora, nombre_paciente, prob_ausencia)
            return True, f"✅ Cita normal confirmada a las {hora}."

        if len(info["pacientes"]) == 1:
            p_aus_existente = info["pacientes"][0]["prob_ausencia"]
            prob_ambos_vienen = (1.0 - p_aus_existente) * (1.0 - prob_ausencia)
            prob_al_menos_uno = 1.0 - (p_aus_existente * prob_ausencia)
            if prob_ambos_vienen < 0.25 and prob_al_menos_uno > 0.8:
                self._insertar(hora, nombre_paciente, prob_ausencia)
                return True, f"⚠️ Overbooking aprobado en {hora}."
            return False, "⛔ Reserva bloqueada por riesgo excesivo."

        return False, "⛔ Hueco completamente saturado."


class PeticionCita(BaseModel):
    nombre: str
    hora: str
    age: int
    days_between: int
    ratio_faltas: float
    sms_received: int
    weekend: int


app = FastAPI(title="Smart-Slotting Hospitalario")
agenda = AgendaInteligente()

if MODEL_PATH.exists():
    modelo_ia = joblib.load(MODEL_PATH)
    print("✅ Modelo definitivo (Voting + Isotonic) cargado")
else:
    modelo_ia = None
    print("⚠️ Modelo no encontrado; usando fallback matemático")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang=\"es\">
        <head><meta charset=\"utf-8\"><title>Smart-Slotting</title></head>
        <body style=\"font-family:Arial;padding:24px;\">
            <h1>Smart-Slotting Hospitalario</h1>
            <p>Use /api/estado-agenda para ver la agenda y /api/evaluar-y-reservar para reservar.</p>
        </body>
        </html>
        """
    )


@app.get("/api/estado-agenda")
def ver_estado_actual():
    return agenda.slots


# Orden exacto de features con el que se entrenó modelo_definitivo.joblib
# (mismo contrato que chatbot/prediction/predictor.py y
# scripts/patient.py::_RISK_FEATURES -- las tres partes del sistema deben
# construir este vector igual para que el mismo paciente reciba el mismo
# riesgo en cualquier interfaz).
FEATURES_MODELO = [
    "Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap",
    "SMS_received", "Days_between", "Appointment_Day_of_Week",
    "Scheduled_Day_of_Week", "Weekend", "Appointment_Month",
    "Scheduled_Month", "Faltas_Previas", "Citas_Previas", "Ratio_Faltas",
    "Gender_M", "Scheduled_Time_of_Day_Evening", "Scheduled_Time_of_Day_Morning",
]


def construir_features(peticion: PeticionCita) -> pd.DataFrame:
    """Construye el vector de entrada para modelo_ia.predict_proba.

    PeticionCita hoy solo recoge age, days_between, ratio_faltas,
    sms_received y weekend. Las demás features clínicas/demográficas
    (Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, Gender_M)
    se fijan a 0 por no estar disponibles todavía en el formulario de
    reserva — pendiente de ampliar el formulario (ver roadmap), igual que
    en scripts/patient.py::_construir_features_nuevo_paciente. Los campos
    de calendario que este formulario tampoco pregunta (día/mes de la
    cita y de la solicitud) se derivan de "hoy" + days_between, con el
    mismo criterio que ya usa el resto del sistema. La agenda de esta
    demo solo tiene huecos de mañana (09:00-09:45), así que
    Scheduled_Time_of_Day_Morning=1 y ...Evening=0 siempre. Sin
    historial de citas propio en esta demo (Faltas_Previas/Citas_Previas
    = 0); Ratio_Faltas la da directamente la petición.
    """
    ahora = datetime.now()
    fecha_cita = ahora + timedelta(days=peticion.days_between)

    fila = {
        "Age": peticion.age,
        "Scholarship": 0,
        "Hipertension": 0,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handcap": 0,
        "SMS_received": peticion.sms_received,
        "Days_between": peticion.days_between,
        "Appointment_Day_of_Week": fecha_cita.weekday(),
        "Scheduled_Day_of_Week": ahora.weekday(),
        "Weekend": peticion.weekend,
        "Appointment_Month": fecha_cita.month,
        "Scheduled_Month": ahora.month,
        "Faltas_Previas": 0,
        "Citas_Previas": 0,
        "Ratio_Faltas": peticion.ratio_faltas,
        "Gender_M": 0,
        "Scheduled_Time_of_Day_Evening": 0,
        "Scheduled_Time_of_Day_Morning": 1,
    }
    return pd.DataFrame([fila])[FEATURES_MODELO]


@app.post("/api/evaluar-y-reservar")
def evaluar_y_reservar(peticion: PeticionCita):
    if modelo_ia is None:
        prob_ausencia = 0.5 + (peticion.age / 200.0) + (peticion.days_between / 400.0)
        prob_ausencia = max(0.05, min(0.95, prob_ausencia))
    else:
        features = construir_features(peticion)
        # Columna 1 = probabilidad calibrada (isotonic) de la clase
        # positiva (No-show = Sí) -- ver Hallazgo #2 en
        # Claude outputs/datos_tecnicos_completados.md sobre por qué debe
        # ser la probabilidad CALIBRADA la que decide el overbooking.
        prob_ausencia = float(modelo_ia.predict_proba(features)[0][1])

    exito, mensaje = agenda.book_slot(
        peticion.hora,
        peticion.nombre,
        prob_ausencia,
    )
    return {"exito": exito, "mensaje": mensaje, "prob_ausencia": round(prob_ausencia, 3)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
