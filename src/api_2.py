# -*- coding: utf-8 -*-
"""Main FastAPI app for the hospital scheduling demo."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "modelo_campeon.json"


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
    modelo_ia = xgb.XGBClassifier()
    modelo_ia.load_model(str(MODEL_PATH))
    print("✅ Modelo XGBoost cargado")
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


@app.post("/api/evaluar-y-reservar")
def evaluar_y_reservar(peticion: PeticionCita):
    if modelo_ia is None:
        prob_ausencia = 0.5 + (peticion.age / 200.0) + (peticion.days_between / 400.0)
        prob_ausencia = max(0.05, min(0.95, prob_ausencia))
    else:
        prob_ausencia = 0.5

    exito, mensaje = agenda.book_slot(
        peticion.hora,
        peticion.nombre,
        prob_ausencia,
    )
    return {"exito": exito, "mensaje": mensaje, "prob_ausencia": round(prob_ausencia, 3)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
