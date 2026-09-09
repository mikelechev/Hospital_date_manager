# -*- coding: utf-8 -*-
"""
mc_2: Simulación de Monte Carlo (1000 escenarios)
Usando el modelo Isotonic/Voting pre-entrenado y exportado con joblib.
"""

import os
import numpy as np
import pandas as pd
import joblib

print("🏥 INICIANDO SIMULACIÓN DE MONTE CARLO (1000 ESCENARIOS EMPÍRICOS)...")
print("-" * 75)

# --- 1. CONFIGURACIÓN DE RUTAS ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")

CSV_PATH = os.path.join(DATA_DIR, "dataset_limpio.csv")
MODELO_JOBLIB_PATH = os.path.join(MODELS_DIR, "modelo_definitivo.joblib")

if not os.path.exists(CSV_PATH):
    print(f"❌ No encuentro el dataset en: {CSV_PATH}")
    exit()

if not os.path.exists(MODELO_JOBLIB_PATH):
    print(f"❌ No encuentro el modelo en: {MODELO_JOBLIB_PATH}")
    exit()

# --- 2. CARGAR Y PREPARAR DATOS ---
df_real = pd.read_csv(CSV_PATH)

# Aseguramos la variable objetivo real
columna_realidad = "No-show" if "No-show" in df_real.columns else "Falta_Real"
if df_real[columna_realidad].dtype == "object":
    df_real["Falta_Real"] = df_real[columna_realidad].map({"Yes": True, "No": False, "1": True, "0": False, 1: True, 0: False})
else:
    df_real["Falta_Real"] = df_real[columna_realidad] == 1

# LAS COLUMNAS EXACTAS DE TU NUEVO MODELO
features_ia = [
    'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 
    'Handcap', 'SMS_received', 'Days_between', 'Appointment_Day_of_Week', 
    'Scheduled_Day_of_Week', 'Weekend', 'Appointment_Month', 
    'Scheduled_Month', 'Faltas_Previas', 'Citas_Previas', 'Ratio_Faltas', 
    'Gender_M', 'Scheduled_Time_of_Day_Evening', 'Scheduled_Time_of_Day_Morning'
]

# Validar que todas las columnas existen en el CSV
faltan = [col for col in features_ia if col not in df_real.columns]
if faltan:
    print(f"❌ Faltan estas columnas en el CSV: {faltan}")
    exit()

# --- 3. CARGAR EL MODELO EXPORTADO (JOBLIB) ---
print("⏳ Cargando el modelo de IA desde el archivo .joblib...")
modelo_ia = joblib.load(MODELO_JOBLIB_PATH)
print("✅ MODELO CARGADO CON ÉXITO\n")

# Calculamos la probabilidad de TODO el dataset de golpe (mucho más rápido)
df_real["Prob_IA"] = modelo_ia.predict_proba(df_real[features_ia])[:, 1]

# --- 4. EL BUCLE DE 1000 SIMULACIONES ---
N_SIMULACIONES = 1000
HUECOS_MAX = 350  # Capacidad del hospital en cada escenario
UMBRAL_RIESGO = 0.45  # Puedes ajustarlo tras ver los resultados de mc_4

resultados_extra_curados = []
resultados_solapamientos = []
resultados_overbookings = []
tasa_vacio_trad = []
tasa_vacio_ia = []

print(f"⚡ Auditando {N_SIMULACIONES} escenarios (Total: {N_SIMULACIONES * HUECOS_MAX} citas analizadas)...")

for i in range(N_SIMULACIONES):
    df_sim = df_real.sample(n=HUECOS_MAX + 150, random_state=i).copy()

    agenda_trad = df_sim.iloc[:HUECOS_MAX].copy()
    lista_espera = df_sim.iloc[HUECOS_MAX:].copy()

    atendidos_trad = len(agenda_trad[agenda_trad["Falta_Real"] == False])
    huecos_vacios_trad = len(agenda_trad[agenda_trad["Falta_Real"] == True])

    # IA Smart-Slotting
    alto_riesgo_idx = agenda_trad[agenda_trad["Prob_IA"] > UMBRAL_RIESGO].index

    atendidos_ia = atendidos_trad
    solapamientos = 0
    overbookings = 0
    idx_espera = 0

    for idx in alto_riesgo_idx:
        if idx_espera < len(lista_espera):
            overbookings += 1
            paciente_extra = lista_espera.iloc[idx_espera]

            falta_orig = agenda_trad.loc[idx, "Falta_Real"]
            falta_extra = paciente_extra["Falta_Real"]

            if falta_orig and not falta_extra:
                atendidos_ia += 1  # ¡Salvado!
            elif not falta_orig and not falta_extra:
                atendidos_ia += 1
                solapamientos += 1  # Choque

            idx_espera += 1

    resultados_extra_curados.append(atendidos_ia - atendidos_trad)
    resultados_solapamientos.append((solapamientos / HUECOS_MAX) * 100)
    resultados_overbookings.append(overbookings)
    tasa_vacio_trad.append((huecos_vacios_trad / HUECOS_MAX) * 100)
    tasa_vacio_ia.append(((huecos_vacios_trad - (atendidos_ia - atendidos_trad)) / HUECOS_MAX) * 100)

# --- 5. RESULTADOS ---
media_extra = np.mean(resultados_extra_curados)
media_solap = np.mean(resultados_solapamientos)
media_over = np.mean(resultados_overbookings)
media_vac_trad = np.mean(tasa_vacio_trad)
media_vac_ia = np.mean(tasa_vacio_ia)

print("\n" + "=" * 75)
print("📊 ESTADÍSTICAS DEFINITIVAS DE LA SIMULACIÓN (MODELO .JOBLIB)")
print("=" * 75)
print(f"👥 PROMEDIO DE PACIENTES EXTRA ATENDIDOS:   +{media_extra:.1f} pacientes por bloque")
print(f"⚡ Promedio de Overbookings Activados:     {media_over:.1f} huecos optimizados")
print(f"🗑️ Tasa de Desperdicio de Huecos:          Tradicional: {media_vac_trad:.1f}%  ->  IA Smart: {media_vac_ia:.1f}%")
print(f"⚠️ Tasa Media de Solapamiento (Fricción):  {media_solap:.2f}% sobre el total")
print("=" * 75)