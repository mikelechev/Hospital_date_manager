# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:42:07 2026

@author: mikel
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- 1. TU LÓGICA DE NEGOCIO (Copia tu clase exacta aquí) ---
class AgendaInteligente:
    def __init__(self):
        self.slots = {
            "09:00": {"pacientes": [], "prob_simultaneous_absence": 1.0},
            "09:15": {"pacientes": [], "prob_simultaneous_absence": 1.0},
            "09:30": {"pacientes": [], "prob_simultaneous_absence": 1.0},
            "09:45": {"pacientes": [], "prob_simultaneous_absence": 1.0}
        }

    def _insertar(self, hora, nombre_paciente, prob_ausencia):
        self.slots[hora]["pacientes"].append({"name": nombre_paciente, "prob_ausencia": prob_ausencia})
        current_prob_product = 1.0
        for p in self.slots[hora]["pacientes"]:
            current_prob_product *= p["prob_ausencia"]
        self.slots[hora]["prob_simultaneous_absence"] = current_prob_product

    def get_available_slots(self, new_patient_prob_ausencia):
        available_slots = []
        p_venga_nuevo = 1.0 - new_patient_prob_ausencia
        for hora, info in self.slots.items():
            if len(info["pacientes"]) == 0:
                available_slots.append((hora, "Libre", p_venga_nuevo))
            elif len(info["pacientes"]) == 1:
                p_ausencia_existente = info["pacientes"][0]["prob_ausencia"]
                p_venga_existente = 1.0 - p_ausencia_existente
                prob_ambos_vienen = p_venga_existente * p_venga_nuevo
                prob_al_menos_uno_venga = 1.0 - (p_ausencia_existente * new_patient_prob_ausencia)
                if prob_ambos_vienen < 0.25 and prob_al_menos_uno_venga > 0.8:
                    desc = f"Overbooking (P_ambos={prob_ambos_vienen:.2f})"
                    available_slots.append((hora, desc, prob_al_menos_uno_venga))
        return available_slots

    def book_slot(self, hora, nombre_paciente, prob_ausencia):
        if hora not in self.slots: return False
        self._insertar(hora, nombre_paciente, prob_ausencia)
        return True

# --- 2. MODELOS DE DATOS (Lo que la API espera recibir) ---
class PeticionHueco(BaseModel):
    prob_ausencia: float

class PeticionReserva(BaseModel):
    hora: str
    nombre_paciente: str
    prob_ausencia: float

# --- 3. CONFIGURACIÓN DE FASTAPI ---
app = FastAPI(title="Motor Smart-Slotting Hospitalario")

# Creamos una única instancia de la agenda en la memoria del servidor
agenda = AgendaInteligente()

@app.post("/api/huecos")
def consultar_huecos(peticion: PeticionHueco):
    opciones = agenda.get_available_slots(peticion.prob_ausencia)
    
    # Formateamos la respuesta para que sea fácil de leer por una web
    resultados = []
    for hora, desc, p_venga in opciones:
        resultados.append({
            "hora": hora,
            "tipo": desc,
            "probabilidad_asistencia_total": round(p_venga, 2)
        })
    return {"disponibles": resultados}

@app.post("/api/reservar")
def reservar_cita(reserva: PeticionReserva):
    exito = agenda.book_slot(reserva.hora, reserva.nombre_paciente, reserva.prob_ausencia)
    if exito:
        return {"mensaje": f"Reserva confirmada para {reserva.nombre_paciente} a las {reserva.hora}"}
    return {"error": "No se pudo realizar la reserva. Hora no válida."}

@app.get("/api/estado-agenda")
def ver_estado_actual():
    return agenda.slots

# Código para arrancar el servidor si ejecutas este archivo
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)