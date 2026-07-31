# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:42:07 2026

@author: mikel
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# --- 1. TU LÓGICA DE NEGOCIO (Intacta) ---
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
                    desc = f"Overbooking (Riesgo Choque={prob_ambos_vienen*100:.0f}%)"
                    available_slots.append((hora, desc, prob_al_menos_uno_venga))
        return available_slots

    def book_slot(self, hora, nombre_paciente, prob_ausencia):
         # 1. Comprobar que la hora existe
         if hora not in self.slots: 
             return False, "❌ Hora no válida en el sistema."
             
         info = self.slots[hora]
         
         # 2. CASO 1: El hueco está vacío -> Entra sin preguntar
         if len(info["pacientes"]) == 0:
             self._insertar(hora, nombre_paciente, prob_ausencia)
             return True, f"✅ Reserva confirmada en hueco libre ({hora})"
             
         # 3. CASO 2: Ya hay alguien -> Pasa por el detector de matemáticas
         elif len(info["pacientes"]) == 1:
             p_ausencia_existente = info["pacientes"][0]["prob_ausencia"]
             p_venga_existente = 1.0 - p_ausencia_existente
             p_venga_nuevo = 1.0 - prob_ausencia
             
             prob_ambos_vienen = p_venga_existente * p_venga_nuevo
             prob_al_menos_uno_venga = 1.0 - (p_ausencia_existente * prob_ausencia)
             
             # Aquí aplicamos tu regla de oro del hackathon
             if prob_ambos_vienen < 0.25 and prob_al_menos_uno_venga > 0.8:
                 self._insertar(hora, nombre_paciente, prob_ausencia)
                 return True, f"⚠️ Overbooking Inteligente aprobado a las {hora}"
             else:
                 return False, "❌ Bloqueado: Matemáticamente demasiado arriesgado para overbooking."
                 
         # 4. CASO 3: Ya hay dos personas -> Límite absoluto
         else:
            return False, "⛔ Bloqueado: Hueco completamente saturado."

# --- 2. MODELOS DE DATOS ---
class PeticionReserva(BaseModel):
    hora: str
    nombre_paciente: str
    prob_ausencia: float

# --- 3. CONFIGURACIÓN DE FASTAPI ---
app = FastAPI(title="Motor Smart-Slotting Hospitalario")
agenda = AgendaInteligente()

# --- 4. LA NUEVA INTERFAZ GRÁFICA (FRONTEND) ---
@app.get("/", response_class=HTMLResponse)
def cargar_interfaz():
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard Smart-Slotting</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 font-sans leading-normal tracking-normal">
        <div class="container mx-auto mt-10 px-4">
            <h1 class="text-4xl font-bold text-center text-blue-700 mb-8">🏥 Smart-Slotting Dashboard</h1>
            
            <div class="flex flex-col md:flex-row gap-8">
                <!-- Panel de Nueva Cita -->
                <div class="bg-white shadow-lg rounded-lg p-6 flex-1 border-t-4 border-blue-500">
                    <h2 class="text-2xl font-semibold mb-4 text-gray-700">Añadir Paciente (Prueba Manual)</h2>
                    <div class="mb-4">
                        <label class="block text-gray-700 font-bold mb-2">Nombre del Paciente:</label>
                        <input type="text" id="nombre" class="shadow border rounded w-full py-2 px-3 text-gray-700" placeholder="Ej: María López">
                    </div>
                    <div class="mb-4">
                        <label class="block text-gray-700 font-bold mb-2">Prob. de Ausencia (0.0 a 1.0):</label>
                        <input type="number" id="prob" step="0.05" class="shadow border rounded w-full py-2 px-3 text-gray-700" placeholder="Ej: 0.85">
                    </div>
                    <div class="mb-6">
                        <label class="block text-gray-700 font-bold mb-2">Hora (09:00 - 09:45):</label>
                        <input type="text" id="hora" class="shadow border rounded w-full py-2 px-3 text-gray-700" placeholder="Ej: 09:15">
                    </div>
                    <button onclick="reservarCita()" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded w-full">
                        Agendar Paciente
                    </button>
                    <p id="mensaje" class="mt-4 font-bold text-center"></p>
                </div>

                <!-- Panel de la Agenda -->
                <div class="bg-white shadow-lg rounded-lg p-6 flex-1 border-t-4 border-green-500">
                    <h2 class="text-2xl font-semibold mb-4 text-gray-700">Estado de la Agenda</h2>
                    <button onclick="cargarAgenda()" class="bg-green-500 hover:bg-green-700 text-white font-bold py-1 px-3 rounded mb-4">
                        🔄 Actualizar Agenda
                    </button>
                    <div id="lista-agenda" class="space-y-4">
                        <!-- Aquí se inyectan los datos -->
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function cargarAgenda() {
                const response = await fetch('/api/estado-agenda');
                const data = await response.json();
                const contenedor = document.getElementById('lista-agenda');
                contenedor.innerHTML = '';

                for (const [hora, info] of Object.entries(data)) {
                    let pacientesHtml = info.pacientes.length === 0 
                        ? '<span class="text-gray-400 italic">Hueco Libre</span>' 
                        : info.pacientes.map(p => `<span class="bg-blue-100 text-blue-800 text-sm font-medium mr-2 px-2.5 py-0.5 rounded">${p.name} (Riesgo: ${p.prob_ausencia})</span>`).join('');
                    
                    let bgClass = info.pacientes.length > 1 ? 'bg-red-50 border-red-200' : 'bg-gray-50 border-gray-200';
                    
                    contenedor.innerHTML += `
                        <div class="p-4 border rounded ${bgClass} shadow-sm">
                            <span class="font-bold text-lg text-gray-800">${hora}</span>
                            <div class="mt-2">${pacientesHtml}</div>
                        </div>
                    `;
                }
            }

            async function reservarCita() {
                const nombre = document.getElementById('nombre').value;
                const prob = parseFloat(document.getElementById('prob').value);
                const hora = document.getElementById('hora').value;
                const msg = document.getElementById('mensaje');

                const response = await fetch('/api/reservar', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ hora: hora, nombre_paciente: nombre, prob_ausencia: prob })
                });

                const result = await response.json();
                if(response.ok) {
                    msg.innerHTML = `<span class="text-green-600">✅ ${result.mensaje}</span>`;
                    cargarAgenda(); // Refrescar visualmente
                } else {
                    msg.innerHTML = `<span class="text-red-600">❌ Error al reservar</span>`;
                }
            }

            // Cargar la agenda al abrir la página
            window.onload = cargarAgenda;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# --- RUTAS DE LA API (Backend) ---
@app.get("/api/estado-agenda")
def ver_estado_actual():
    return agenda.slots

@app.post("/api/reservar")
def reservar_cita(reserva: PeticionReserva):
    exito, mensaje = agenda.book_slot(reserva.hora, reserva.nombre_paciente, reserva.prob_ausencia)
    
    if exito:
        return {"mensaje": mensaje}
    else:
        # Devuelve un error 400 (Bad Request) camuflado para que el frontend lo lea bien
        return {"error": mensaje}
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)