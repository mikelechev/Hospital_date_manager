# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:55:35 2026

@author: mikel
"""

import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
from pydantic import BaseModel
import uvicorn
import xgboost as xgb

# --- 1. CARGA DEL CEREBRO (IA XGBOOST) ---
MODELO_PATH = 'modelo_campeon.json'  # Si al final usaste .joblib, pon su nombre aquí

if os.path.exists(MODELO_PATH):
    modelo_ia = xgb.XGBClassifier()
    modelo_ia.load_model(MODELO_PATH)
    print("✅ CEREBRO IA CARGADO CON ÉXITO: Modelo XGBoost activo.")
else:
    modelo_ia = None
    print(
        "⚠️ ATENCIÓN: No se encontró el archivo. Se usará simulación"
        " matemática."
    )


# --- 2. EL CORAZÓN: LÓGICA DE SMART-SLOTTING ---
class AgendaInteligente:

    def __init__(self):
        self.slots = {
            "09:00": {"pacientes": [], "prob_simultaneous_absence": 1.0},
            "09:15": {"pacientes": [], "prob_simultaneous_absence": 1.0},
            "09:30": {"pacientes": [], "prob_simultaneous_absence": 1.0},
            "09:45": {"pacientes": [], "prob_simultaneous_absence": 1.0},
        }

    def _insertar(self, hora, nombre_paciente, prob_ausencia):
        self.slots[hora]["pacientes"].append({
            "name": nombre_paciente,
            "prob_ausencia": prob_ausencia,
        })
        current_prob_product = 1.0
        for p in self.slots[hora]["pacientes"]:
            current_prob_product *= p["prob_ausencia"]
        self.slots[hora]["prob_simultaneous_absence"] = current_prob_product

    def book_slot(self, hora, nombre_paciente, prob_ausencia):
        if hora not in self.slots:
            return False, "❌ Hora no válida en el sistema."

        info = self.slots[hora]

        # CASO 1: Hueco libre
        if len(info["pacientes"]) == 0:
            self._insertar(hora, nombre_paciente, prob_ausencia)
            return True, f"✅ Cita normal confirmada a las {hora}."

        # CASO 2: Hueco ocupado -> Filtro matemático de Overbooking
        elif len(info["pacientes"]) == 1:
            p_aus_existente = info["pacientes"][0]["prob_ausencia"]
            prob_ambos_vienen = (1.0 - p_aus_existente) * (1.0 - prob_ausencia)
            prob_al_menos_uno = 1.0 - (p_aus_existente * prob_ausencia)

            # Regla de oro del Hackathon
            if prob_ambos_vienen < 0.25 and prob_al_menos_uno > 0.8:
                self._insertar(hora, nombre_paciente, prob_ausencia)
                return (
                    True,
                    f"⚠️ OVERBOOKING INTELIGENTE APROBADO en {hora} (Riesgo"
                    f" choque: {prob_ambos_vienen*100:.0f}%).",
                )
            else:
                return (
                    False,
                    f"⛔ BLOQUEADO en {hora}: Riesgo de colapso excesivo en"
                    " sala de espera.",
                )

        # CASO 3: Límite absoluto de 2 pacientes por hueco
        else:
            return (
                False,
                f"⛔ BLOQUEADO en {hora}: El hueco está completamente saturado.",
            )


# --- 3. MODELOS DE DATOS ---
class PeticionCita(BaseModel):
    nombre: str
    hora: str
    age: int
    days_between: int
    ratio_faltas: float
    sms_received: int
    weekend: int

    # Variables secundarias en silencio (Por defecto a 0 = "No")
    scholarship: int = 0
    hipertension: int = 0
    diabetes: int = 0
    alcoholism: int = 0
    handcap: int = 0
    gender_m: int = 0
    scheduled_morning: int = 1
    scheduled_evening: int = 0


# --- 4. CONFIGURACIÓN DE FASTAPI ---
app = FastAPI(title="Smart-Slotting Hospitalario Completo")
agenda = AgendaInteligente()


# --- 5. INTERFAZ WEB PROFESIONAL (FRONTEND AMPLIADO) ---
@app.get("/", response_class=HTMLResponse)
def cargar_interfaz():
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IA Smart-Slotting Hospitalario</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-gray-100 font-sans min-h-screen p-6">
        <div class="max-w-6xl mx-auto">
            <header class="mb-8 border-b border-slate-700 pb-4 flex justify-between items-center">
                <div>
                    <h1 class="text-3xl font-extrabold text-blue-400">🏥 Smart-Slotting AI System</h1>
                    <p class="text-slate-400 text-sm mt-1">Gestión hospitalaria basada en predicción de ausencias con XGBoost</p>
                </div>
                <div class="bg-blue-900/40 border border-blue-500/30 px-4 py-2 rounded-lg text-right">
                    <span class="text-xs text-blue-300 block font-bold">ESTADO DEL MOTOR</span>
                    <span class="text-green-400 font-mono text-sm">● IA Activa & Cortafuegos Online</span>
                </div>
            </header>
            
            <div class="grid grid-cols-1 md:grid-cols-12 gap-8">
                <!-- Panel de Entrada de Datos -->
                <div class="md:col-span-5 bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-xl">
                    <h2 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
                        <span>👤 Parámetros Clínicos del Paciente</span>
                    </h2>
                    
                    <div class="space-y-3">
                        <div>
                            <label class="block text-slate-300 text-xs font-bold mb-1 uppercase">Nombre del Paciente:</label>
                            <input type="text" id="nombre" class="w-full bg-slate-900 border border-slate-600 rounded px-3 py-1.5 text-white focus:border-blue-500" placeholder="Ej: Carlos Mendoza">
                        </div>

                        <div class="grid grid-cols-2 gap-3">
                            <div>
                                <label class="block text-slate-300 text-xs font-bold mb-1 uppercase">Edad:</label>
                                <input type="number" id="age" class="w-full bg-slate-900 border border-slate-600 rounded px-3 py-1.5 text-white focus:border-blue-500" value="35">
                            </div>
                            <div>
                                <label class="block text-slate-300 text-xs font-bold mb-1 uppercase">Días Espera:</label>
                                <input type="number" id="dias" class="w-full bg-slate-900 border border-slate-600 rounded px-3 py-1.5 text-white focus:border-blue-500" value="14">
                            </div>
                        </div>

                        <div class="grid grid-cols-2 gap-3">
                            <div>
                                <label class="block text-amber-400 text-xs font-bold mb-1 uppercase">Ratio Faltas (0.0 a 1.0):</label>
                                <input type="number" step="0.1" max="1.0" min="0.0" id="ratio" class="w-full bg-slate-900 border border-amber-500/50 rounded px-3 py-1.5 text-white focus:border-amber-500" value="0.2">
                            </div>
                            <div>
                                <label class="block text-slate-300 text-xs font-bold mb-1 uppercase">¿Recibió SMS?:</label>
                                <select id="sms" class="w-full bg-slate-900 border border-slate-600 rounded px-3 py-1.5 text-white focus:border-blue-500">
                                    <option value="1">Sí (Recordatorio)</option>
                                    <option value="0">No recibió SMS</option>
                                </select>
                            </div>
                        </div>

                        <div>
                            <label class="block text-slate-300 text-xs font-bold mb-1 uppercase">¿Es Cita en Fin de Semana?:</label>
                            <select id="finde" class="w-full bg-slate-900 border border-slate-600 rounded px-3 py-1.5 text-white focus:border-blue-500">
                                <option value="0">No (Lunes a Viernes)</option>
                                <option value="1">Sí (Sábado/Domingo)</option>
                            </select>
                        </div>

                        <div class="pt-2">
                            <label class="block text-blue-400 text-xs font-bold mb-1 uppercase">Seleccionar Hora de Reserva:</label>
                            <select id="hora" class="w-full bg-slate-900 border border-blue-500 rounded px-3 py-2 text-white font-mono text-base focus:outline-none">
                                <option value="09:00">09:00</option>
                                <option value="09:15">09:15</option>
                                <option value="09:30">09:30</option>
                                <option value="09:45">09:45</option>
                            </select>
                        </div>

                        <button onclick="procesarPaciente()" class="w-full mt-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold py-3 px-4 rounded-lg shadow-lg transform transition active:scale-95 flex justify-center items-center gap-2">
                            <span>⚡ Evaluar IA & Agendar</span>
                        </button>
                    </div>

                    <div id="panel-resultado" class="mt-4 p-4 rounded-lg bg-slate-900/50 border border-slate-700 hidden">
                        <div class="text-xs text-slate-400 uppercase tracking-wider font-bold mb-1">Diagnóstico de la IA:</div>
                        <div id="ia-prob" class="text-lg font-bold"></div>
                        <div id="ia-msg" class="text-sm mt-2 font-medium"></div>
                    </div>
                </div>

                <!-- Panel de la Agenda -->
                <div class="md:col-span-7 bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-xl">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-bold text-white">📅 Agenda Inteligente en Tiempo Real</h2>
                        <button onclick="cargarAgenda()" class="text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 px-3 py-1.5 rounded transition">
                            🔄 Refrescar Vista
                        </button>
                    </div>
                    <div id="lista-agenda" class="space-y-3"></div>
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
                        ? '<span class="text-slate-500 italic text-sm">Hueco disponible</span>' 
                        : info.pacientes.map(p => {
                            let badgeColor = p.prob_ausencia > 0.6 ? 'bg-red-900/60 text-red-300 border-red-700/50' : 'bg-blue-900/60 text-blue-300 border-blue-700/50';
                            return `<span class="inline-block border text-xs font-semibold mr-2 px-2.5 py-1 rounded-md mb-1 ${badgeColor}">${p.name} \vert{} Riesgo Faltar:${(p.prob_ausencia*100).toFixed(0)}%</span>`;
                        }).join('');
                    
                    let cardStyle = info.pacientes.length > 1 
                        ? 'border-l-4 border-l-amber-500 bg-amber-950/10 border-slate-700' 
                        : 'border-l-4 border-l-blue-500 bg-slate-900/60 border-slate-700/60';
                    
                    if (info.pacientes.length === 0) cardStyle = 'border-l-4 border-l-slate-600 bg-slate-900/30 border-slate-800';

                    contenedor.innerHTML += `
                        <div class="p-4 border rounded-lg ${cardStyle} transition">
                            <div class="flex justify-between items-start mb-2">
                                <span class="font-mono font-bold text-lg text-white bg-slate-800 px-2 py-0.5 rounded border border-slate-700">${hora}</span>
                                <span class="text-xs text-slate-400">Asistencia Asegurada: <strong class="text-slate-200">${((1 - info.prob_simultaneous_absence)*100).toFixed(0)}%</strong></span>
                            </div>
                            <div>${pacientesHtml}</div>
                        </div>
                    `;
                }
            }

            async function procesarPaciente() {
                const nombre = document.getElementById('nombre').value || "Paciente Anónimo";
                const age = parseInt(document.getElementById('age').value) || 35;
                const dias = parseInt(document.getElementById('dias').value) || 0;
                const ratio = parseFloat(document.getElementById('ratio').value) || 0.0;
                const sms = parseInt(document.getElementById('sms').value);
                const finde = parseInt(document.getElementById('finde').value);
                const hora = document.getElementById('hora').value;

                const response = await fetch('/api/evaluar-y-reservar', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        nombre: nombre,
                        hora: hora,
                        age: age,
                        days_between: dias,
                        ratio_faltas: ratio,
                        sms_received: sms,
                        weekend: finde
                    })
                });

                const result = await response.json();
                
                const panel = document.getElementById('panel-resultado');
                const iaProb = document.getElementById('ia-prob');
                const iaMsg = document.getElementById('ia-msg');
                
                panel.classList.remove('hidden');
                iaProb.innerHTML = `Probabilidad de Ausencia: <span class="${result.probabilidad > 0.6 ? 'text-red-400' : 'text-green-400'} font-mono">${(result.probabilidad * 100).toFixed(1)}%</span>`;
                
                if (result.exito) {
                    iaMsg.innerHTML = `<span class="text-emerald-400 block mt-1">${result.mensaje}</span>`;
                } else {
                    iaMsg.innerHTML = `<span class="text-amber-400 block mt-1">${result.mensaje}</span>`;
                }

                cargarAgenda();
            }

            window.onload = cargarAgenda;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# --- 6. ENDPOINTS DEL SERVIDOR ---
@app.get("/api/estado-agenda")
def ver_estado():
    return agenda.slots


@app.post("/api/evaluar-y-reservar")
def evaluar_y_reservar(peticion: PeticionCita):
    # Reconstruimos las 13 columnas EXACTAS del modelo
    df_paciente = pd.DataFrame([{
        'Age': peticion.age,
        'Scholarship': peticion.scholarship,
        'Hipertension': peticion.hipertension,
        'Diabetes': peticion.diabetes,
        'Alcoholism': peticion.alcoholism,
        'Handcap': peticion.handcap,
        'SMS_received': peticion.sms_received,
        'Days-between': peticion.days_between,
        'Weekend': peticion.weekend,
        'Ratio_Faltas': peticion.ratio_faltas,
        'Gender_M': peticion.gender_m,
        'Scheduled_Time_of_Day_Evening': peticion.scheduled_evening,
        'Scheduled_Time_of_Day_Morning': peticion.scheduled_morning,
    }])

    if modelo_ia is not None:
        try:
            prob_ausencia = float(modelo_ia.predict_proba(df_paciente)[0][1])
        except Exception as e:
            print(f"❌ ERROR CRÍTICO AL PREDECIR CON XGBOOST: {e}")
            prob_ausencia = min(
                0.95,
                (peticion.ratio_faltas * 0.7) + (peticion.days_between * 0.01),
            )
    else:
        prob_ausencia = min(
            0.95, (peticion.ratio_faltas * 0.7) + (peticion.days_between * 0.01)
        )

    exito, mensaje = agenda.book_slot(
        peticion.hora, peticion.nombre, prob_ausencia
    )

    return {"exito": exito, "probabilidad": prob_ausencia, "mensaje": mensaje}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)