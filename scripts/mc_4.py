# -*- coding: utf-8 -*-
"""
mc_4: Simulación mensual de agenda hospitalaria con efectos de cola (Efecto Dominó),
Cortafuegos y Análisis de Trade-off (Umbrales).
Utiliza el modelo maestro exportado con joblib (Voting + Isotonic pre-calculado).
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

print("🏥 INICIANDO SIMULACIÓN VISUAL Y OPTIMIZACIÓN DE AGENDA HOSPITALARIA...")
print("-" * 75)

# --- 1. CONFIGURACIÓN DE RUTAS ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODELO_JOBLIB_PATH = MODELS_DIR / "modelo_definitivo.joblib"
CSV_PATH = DATA_DIR / "dataset_limpio.csv"
OUTPUT_DIR = ROOT_DIR / "scripts"

# --- 2. CONFIGURACIÓN DEL HOSPITAL ---
DAYS_IN_MONTH = 30
SLOT_MINUTES = 15
WORK_START_HOUR = 9
WORK_END_HOUR = 14
TOTAL_SLOTS = ((WORK_END_HOUR - WORK_START_HOUR) * 60) // SLOT_MINUTES
FIREWALL_SLOTS = {7, 15}  # Cortafuegos cada 2 horas (10:45 y 12:45)

# LAS COLUMNAS EXACTAS DE TU NUEVO MODELO
FEATURES = [
    'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 
    'Handcap', 'SMS_received', 'Days_between', 'Appointment_Day_of_Week', 
    'Scheduled_Day_of_Week', 'Weekend', 'Appointment_Month', 
    'Scheduled_Month', 'Faltas_Previas', 'Citas_Previas', 'Ratio_Faltas', 
    'Gender_M', 'Scheduled_Time_of_Day_Evening', 'Scheduled_Time_of_Day_Morning'
]

def prepare_dataset_and_model():
    """Carga el dataset y el modelo exportado en joblib."""
    if not CSV_PATH.exists():
        print(f"❌ No encuentro el archivo {CSV_PATH}.")
        exit()

    df = pd.read_csv(CSV_PATH)

    # Validar que todas las columnas existen en el CSV
    faltan = [col for col in FEATURES if col not in df.columns]
    if faltan:
        print(f"❌ Faltan estas columnas en el CSV: {faltan}")
        exit()

    # Mapeo de la variable objetivo
    columna_realidad = "No-show" if "No-show" in df.columns else "Falta_Real"
    if df[columna_realidad].dtype == "object":
        df["Falta_Real"] = df[columna_realidad].map({"Yes": True, "No": False, "1": True, "0": False, 1: True, 0: False})
    else:
        df["Falta_Real"] = df[columna_realidad] == 1

    # Cargar el modelo desde el archivo
    if not MODELO_JOBLIB_PATH.exists():
        print(f"❌ ERROR: No se encuentra '{MODELO_JOBLIB_PATH}'.")
        exit()

    print("⏳ Cargando modelo maestro desde .joblib...")
    modelo_ia = joblib.load(MODELO_JOBLIB_PATH)
    print("✅ MODELO CARGADO CON ÉXITO\n")

    # Calcular probabilidad de todo el dataset usando las features exactas
    df["Prob_IA"] = modelo_ia.predict_proba(df[FEATURES])[:, 1]
    
    return df

def get_day_patients_pool(df, day_index, num_needed):
    """Extrae un pool de pacientes aleatorios del dataset para usarlos en la agenda."""
    rng = np.random.default_rng(day_index + 42)
    pool = df.sample(n=num_needed, random_state=day_index + 7).copy()
    patients = []
    for _, row in pool.iterrows():
        patients.append({
            "real_no_show": bool(row.get("Falta_Real", False)),
            "prob_no_show": float(row.get("Prob_IA", 0.2))
        })
    return patients

def create_schedule(day_patients_pool, smart_overbooking=False, umbral_riesgo=0.40):
    """Construye la agenda de un día aplicando el umbral de riesgo seleccionado."""
    agenda = {slot: [] for slot in range(TOTAL_SLOTS)}
    pool_idx = 0
    overbookings_hechos = 0
    
    # Asignación base (1 paciente por hueco, saltando cortafuegos)
    for slot in range(TOTAL_SLOTS):
        if slot in FIREWALL_SLOTS: continue
        if pool_idx < len(day_patients_pool):
            agenda[slot].append(day_patients_pool[pool_idx])
            pool_idx += 1

    # IA Smart-Slotting (Overbooking)
    if smart_overbooking:
        for slot in range(TOTAL_SLOTS):
            if slot in FIREWALL_SLOTS: continue
            
            if agenda[slot]:
                titular = agenda[slot][0]
                if titular["prob_no_show"] > umbral_riesgo and pool_idx < len(day_patients_pool):
                    agenda[slot].append(day_patients_pool[pool_idx])
                    pool_idx += 1
                    overbookings_hechos += 1

    return agenda, overbookings_hechos

def simulate_day_queue(agenda):
    """Motor de colas con efecto dominó."""
    slots_status = np.full(TOTAL_SLOTS, 3, dtype=int)
    cola = 0
    atendidos = 0
    solapamientos = 0
    
    for slot in range(TOTAL_SLOTS):
        pacientes_citados = agenda[slot]
        pacientes_que_llegan = [p for p in pacientes_citados if not p["real_no_show"]]
        
        # Detectar solapamientos reales
        if len(pacientes_que_llegan) > 1 or (len(pacientes_que_llegan) == 1 and cola > 0):
            solapamientos += 1

        if pacientes_citados and len(pacientes_que_llegan) == 0 and cola == 0:
            slots_status[slot] = 0
        
        cola += len(pacientes_que_llegan)
        
        if cola > 0:
            if len(pacientes_que_llegan) > 0 and cola == len(pacientes_que_llegan):
                slots_status[slot] = 1 
            else:
                slots_status[slot] = 2 
            
            cola -= 1
            atendidos += 1
        else:
            if not pacientes_citados:
                slots_status[slot] = 3 

    return slots_status, atendidos, solapamientos

def simulate_month(df, overbooking_mode=False, umbral_riesgo=0.40):
    """Simula los 30 días del mes y devuelve las estadísticas."""
    month_matrix = np.zeros((DAYS_IN_MONTH, TOTAL_SLOTS), dtype=int)
    stats = {"atendidos": 0, "solapamientos": 0, "overbookings": 0, "huecos_vacios": 0}

    for day in range(DAYS_IN_MONTH):
        pool = get_day_patients_pool(df, day, num_needed=40)
        agenda, overbookings_dia = create_schedule(pool, smart_overbooking=overbooking_mode, umbral_riesgo=umbral_riesgo)
        day_matrix, atendidos_dia, solapamientos_dia = simulate_day_queue(agenda)
        
        month_matrix[day] = day_matrix
        stats["atendidos"] += atendidos_dia
        stats["solapamientos"] += solapamientos_dia
        stats["overbookings"] += overbookings_dia
        stats["huecos_vacios"] += int(np.sum((day_matrix == 0) | (day_matrix == 3)))

    return month_matrix, stats

def save_matrix_plot(matrix, title, path):
    """Genera la imagen del calendario de colores."""
    cmap = plt.matplotlib.colors.ListedColormap(["#d3d3d3", "#2ecc71", "#f39c12", "#ffffff"])
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Slots de 15 min")
    ax.set_ylabel("Día del mes")

    slot_labels = [f"{WORK_START_HOUR + (i * SLOT_MINUTES // 60)}:{(i * SLOT_MINUTES) % 60:02d}" for i in range(TOTAL_SLOTS)]
    ax.set_xticks(np.arange(TOTAL_SLOTS))
    ax.set_xticklabels(slot_labels, rotation=90)
    ax.set_yticks(np.arange(DAYS_IN_MONTH))
    ax.set_yticklabels([str(d + 1) for d in range(DAYS_IN_MONTH)])

    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#d3d3d3", markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#2ecc71", markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#f39c12", markersize=12),
    ]
    legend_labels = ["No-show / vacío", "A tiempo", "Retrasado (Efecto Dominó)"]
    ax.legend(legend_handles, legend_labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def plot_tradeoff_curve(umbrales, atendidos_extra, solapamientos, path):
    """Genera el gráfico de la Curva de Compromiso (Optimización)."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = [str(u) for u in umbrales]

    color = '#2ecc71'
    ax1.set_xlabel('Umbral de Decisión (Theta)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pacientes Extra Rescatados (+)', color=color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(x, atendidos_extra, color=color, marker='o', linewidth=3, label='Pacientes Extra')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = '#e74c3c'
    ax2.set_ylabel('Solapamientos / Riesgo de Colapso', color=color, fontsize=12, fontweight='bold')
    line2 = ax2.plot(x, solapamientos, color=color, marker='s', linewidth=3, linestyle='--', label='Solapamientos')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Curva de Compromiso (Trade-off): Beneficio vs. Riesgo Operativo', fontsize=14, fontweight='bold', pad=15)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    fig.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"\n📈 Gráfica de Trade-off guardada en: {path}")

def main():
    df = prepare_dataset_and_model()

    # 1. Escenario Base
    matrix_trad, stats_trad = simulate_month(df, overbooking_mode=False)
    print(f"📊 Tradicional (Sin IA) -> Pacientes Atendidos: {stats_trad['atendidos']} | Huecos Vacíos: {stats_trad['huecos_vacios']}")
    print("-" * 75)

    # 2. Análisis de Sensibilidad y Optimización de Umbrales
    umbrales_a_probar = [0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    lista_umbrales, lista_extra, lista_solapamientos = [], [], []
    mejor_umbral, max_score, mejor_matrix = None, -float('inf'), None

    print("🔬 PROBANDO DIFERENTES UMBRALES DE RIESGO PARA EL OVERBOOKING:")
    for umbral in umbrales_a_probar:
        matrix_ia, stats_ia = simulate_month(df, overbooking_mode=True, umbral_riesgo=umbral)
        extra = stats_ia['atendidos'] - stats_trad['atendidos']
        
        lista_umbrales.append(umbral)
        lista_extra.append(extra)
        lista_solapamientos.append(stats_ia['solapamientos'])

        # Función de Coste: Maximizar rescates penalizando los solapamientos
        score = (extra * 1.0) - (stats_ia['solapamientos'] * 0.5)
        print(f"   🎯 Umbral > {umbral:.2f} | Atendidos: {stats_ia['atendidos']} (+{extra} extra) | Solapamientos: {stats_ia['solapamientos']} | Score: {score:.1f}")

        if score > max_score:
            max_score = score
            mejor_umbral = umbral
            mejor_matrix = matrix_ia

    print("=" * 75)
    print(f"🏆 CONCLUSIÓN ÓPTIMA: El modelo matemático recomienda el umbral > {mejor_umbral}")
    print("=" * 75)

    # 3. Exportar resultados visuales
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_matrix_plot(matrix_trad, "Agenda Mensual - Tradicional (Sin IA)", OUTPUT_DIR / "mc_4_matrix_normal.png")
    save_matrix_plot(mejor_matrix, f"Agenda Mensual - Óptima IA (Umbral > {mejor_umbral})", OUTPUT_DIR / "mc_4_matrix_overbooking.png")
    plot_tradeoff_curve(lista_umbrales, lista_extra, lista_solapamientos, OUTPUT_DIR / "mc_4_tradeoff.png")
    print(f"\n✅ Todos los gráficos y simulaciones se han guardado con éxito en la carpeta 'scripts'.")

if __name__ == "__main__":
    main()