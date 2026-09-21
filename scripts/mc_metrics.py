# -*- coding: utf-8 -*-
"""
Monte Carlo simplificado (sin interfaz visual) para sacar las metricas
operativas del overbooking inteligente a distintos umbrales de riesgo.

Reutiliza la misma logica de agenda/llegadas (Johnson SU) que
scripts/app.py, pero sin Streamlit/Plotly: solo calcula y muestra
numeros. Corre N "meses" independientes por umbral (en vez de un unico
run suelto) para poder dar una media +/- intervalo de confianza al 95%,
siguiendo la misma metodologia que "Claude outputs/datos_tecnicos_completados.md"
(Hallazgo #3: con pocas realizaciones el margen de error es demasiado
grande para citar un numero suelto en la memoria).

Uso:
    python scripts/mc_metrics.py
    python scripts/mc_metrics.py --realizaciones 100 --umbrales 0.35 0.40
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import johnsonsu

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CSV_PATH = DATA_DIR / "dataset_limpio.csv"
MODEL_PATH = MODELS_DIR / "modelo_definitivo.joblib"

# Mismo orden de columnas que espera modelo_definitivo.joblib (ver
# scripts/app.py y src/api_2.py::FEATURES_MODELO).
FEATURES = [
    "Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
    "Handcap", "SMS_received", "Days_between", "Appointment_Day_of_Week",
    "Scheduled_Day_of_Week", "Weekend", "Appointment_Month",
    "Scheduled_Month", "Faltas_Previas", "Citas_Previas", "Ratio_Faltas",
    "Gender_M", "Scheduled_Time_of_Day_Evening", "Scheduled_Time_of_Day_Morning",
]

# Parametros operativos (identicos a los valores por defecto de scripts/app.py)
WORK_START_HOUR = 9
WORK_END_HOUR = 15
SLOT_MINUTES = 10
N_PACIENTES_DIA = 60
DIAS_POR_MES = 30
DESCANSO_START = 12
COSTE_HORA_MEDICO = 120.0  # euros/hora

TOTAL_SLOTS = ((WORK_END_HOUR - WORK_START_HOUR) * 60) // SLOT_MINUTES
ADMIN_SLOTS = set(range(TOTAL_SLOTS - 6, TOTAL_SLOTS))  # ultimos 6 slots: bloque de informes
BREAK_SLOTS = {DESCANSO_START, DESCANSO_START + 1}
SKIP_SLOTS = BREAK_SLOTS | ADMIN_SLOTS

# Parametros de la Johnson SU para el desfase de llegada (identicos a scripts/app.py)
J_A, J_B, J_LOC, J_SCALE = 1.5, 1.5, -2.0, 12.0


def load_data():
    df = pd.read_csv(CSV_PATH)
    col = "No-show" if "No-show" in df.columns else "Falta_Real"
    if df[col].dtype == "object":
        df["Falta_Real"] = df[col].map({"Yes": True, "No": False, "1": True, "0": False, 1: True, 0: False})
    else:
        df["Falta_Real"] = df[col] == 1
    return df


def load_model():
    return joblib.load(MODEL_PATH)


def get_day_patients_pool(df, day_index, modelo, semilla):
    pool = df.sample(n=N_PACIENTES_DIA, random_state=day_index + semilla).copy()
    pool["Prob_IA"] = modelo.predict_proba(pool[FEATURES])[:, 1]
    return [
        {"real_no_show": bool(r["Falta_Real"]), "prob_no_show": float(r["Prob_IA"])}
        for _, r in pool.iterrows()
    ]


def precompute_pools(df, modelo, semilla):
    """Un pool de pacientes por dia; se reutiliza para baseline y ambos umbrales
    (mismos pacientes/mismo riesgo real en las tres comparaciones del mes)."""
    return [get_day_patients_pool(df, day, modelo, semilla) for day in range(DIAS_POR_MES)]


def create_schedule(pool, overbooking, umbral):
    agenda = {slot: [] for slot in range(TOTAL_SLOTS)}
    idx = 0
    for slot in range(TOTAL_SLOTS):
        if slot in SKIP_SLOTS:
            continue
        if idx < len(pool):
            agenda[slot].append(pool[idx])
            idx += 1
    if overbooking:
        for slot in range(TOTAL_SLOTS):
            if slot in SKIP_SLOTS or not agenda[slot]:
                continue
            titular = agenda[slot][0]
            if idx < len(pool):
                candidato = pool[idx]
                if titular["prob_no_show"] > umbral or candidato["prob_no_show"] > umbral:
                    agenda[slot].append(candidato)
                    idx += 1
    return agenda


def simulate_day(agenda, descanso_fijo, semilla):
    """Devuelve (atendidos, lista_de_retrasos_en_minutos) para un dia."""
    rng = np.random.RandomState(semilla)
    llegadas = []
    for slot in range(TOTAL_SLOTS):
        for p in agenda[slot]:
            if not p["real_no_show"]:
                desfase = np.clip(
                    johnsonsu.rvs(a=J_A, b=J_B, loc=J_LOC, scale=J_SCALE, random_state=rng),
                    -30, 10,
                )
                minuto_citado = slot * SLOT_MINUTES
                llegadas.append({
                    **p,
                    "slot_citado": slot,
                    "minuto_citado": minuto_citado,
                    "minuto_llegada": max(0, minuto_citado + desfase),
                })
    llegadas.sort(key=lambda x: x["minuto_llegada"])

    sala_de_espera, idx_llegada, atendidos = [], 0, 0
    retrasos = []
    descansos_pendientes, en_descanso = 2, False

    for slot in range(TOTAL_SLOTS):
        minuto_actual = slot * SLOT_MINUTES
        while idx_llegada < len(llegadas) and llegadas[idx_llegada]["minuto_llegada"] <= minuto_actual:
            sala_de_espera.append(llegadas[idx_llegada])
            idx_llegada += 1

        if not sala_de_espera and idx_llegada == len(llegadas):
            continue

        if slot >= TOTAL_SLOTS - 6:
            if sala_de_espera:
                sala_de_espera.sort(key=lambda x: x["slot_citado"])
                paciente = sala_de_espera.pop(0)
                atendidos += 1
                retrasos.append(minuto_actual - paciente["minuto_citado"])
            continue

        if descanso_fijo:
            if slot in (DESCANSO_START, DESCANSO_START + 1):
                continue
        else:
            if en_descanso:
                descansos_pendientes -= 1
                if descansos_pendientes == 0:
                    en_descanso = False
                continue
            if descansos_pendientes == 2 and slot >= DESCANSO_START - 1:
                if not sala_de_espera or slot >= DESCANSO_START + 4:
                    descansos_pendientes -= 1
                    en_descanso = True
                    continue

        if sala_de_espera:
            sala_de_espera.sort(key=lambda x: x["slot_citado"])
            paciente = sala_de_espera.pop(0)
            atendidos += 1
            retrasos.append(minuto_actual - paciente["minuto_citado"])

    return atendidos, retrasos


def run_month(pools, semilla, umbral=None):
    """umbral=None -> escenario base (sin overbooking, descansos fijos)."""
    overbooking = umbral is not None
    total_atendidos = 0
    total_overbookings = 0
    todos_los_retrasos = []
    for day, pool in enumerate(pools):
        agenda = create_schedule(pool, overbooking, umbral if overbooking else 0.0)
        if overbooking:
            total_overbookings += sum(1 for s in agenda.values() if len(s) == 2)
        atendidos, retrasos = simulate_day(agenda, descanso_fijo=not overbooking, semilla=day + semilla)
        total_atendidos += atendidos
        todos_los_retrasos.extend(retrasos)
    p90 = float(np.percentile(todos_los_retrasos, 90)) if todos_los_retrasos else 0.0
    return total_atendidos, total_overbookings, p90


def ci95(values):
    values = np.asarray(values, dtype=float)
    media = values.mean()
    error = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return media, media - error, media + error


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--realizaciones", type=int, default=100, help="Meses independientes a simular por umbral (default: 100)")
    parser.add_argument("--umbrales", type=float, nargs="+", default=[0.35, 0.40], help="Umbrales de riesgo a comparar (default: 0.35 0.40)")
    parser.add_argument("--semilla-base", type=int, default=5000, help="Semilla inicial (cada realizacion usa semilla_base + i)")
    args = parser.parse_args()

    print(f"Cargando modelo ({MODEL_PATH.name}) y datos ({CSV_PATH.name})...")
    df = load_data()
    modelo = load_model()

    print(f"Simulando {args.realizaciones} meses independientes por umbral "
          f"({N_PACIENTES_DIA} pacientes/dia, {DIAS_POR_MES} dias/mes, "
          f"coste medico {COSTE_HORA_MEDICO:.0f} EUR/h, slot {SLOT_MINUTES} min)...\n")

    resultados = {umbral: {"extra": [], "overbookings": [], "p90_delta": []} for umbral in args.umbrales}

    for r in range(args.realizaciones):
        semilla = args.semilla_base + r
        pools = precompute_pools(df, modelo, semilla)
        base_atendidos, _, base_p90 = run_month(pools, semilla, umbral=None)
        for umbral in args.umbrales:
            ia_atendidos, ia_overbookings, ia_p90 = run_month(pools, semilla, umbral=umbral)
            resultados[umbral]["extra"].append(ia_atendidos - base_atendidos)
            resultados[umbral]["overbookings"].append(ia_overbookings)
            resultados[umbral]["p90_delta"].append(ia_p90 - base_p90)
        if (r + 1) % 20 == 0 or r == args.realizaciones - 1:
            print(f"  ...{r + 1}/{args.realizaciones} meses simulados")

    coste_slot = (SLOT_MINUTES / 60.0) * COSTE_HORA_MEDICO

    print("\n" + "=" * 78)
    print(f"RESULTADOS ({args.realizaciones} meses independientes por umbral, IC95%)")
    print("=" * 78)

    for umbral in args.umbrales:
        datos = resultados[umbral]
        media_extra, low_extra, high_extra = ci95(datos["extra"])
        media_over = float(np.mean(datos["overbookings"]))
        contencion = (media_extra / media_over * 100) if media_over > 0 else 0.0
        media_p90 = float(np.mean(datos["p90_delta"]))
        ahorro_medio = media_extra * coste_slot
        ahorro_low = low_extra * coste_slot
        ahorro_high = high_extra * coste_slot

        print(f"\n--- Umbral {umbral:.2f} ---")
        print(f"  Huecos recuperados/mes:        {media_extra:6.1f}  (IC95% [{low_extra:.1f} - {high_extra:.1f}])")
        print(f"  Overbookings programados/mes:  {media_over:6.1f}")
        print(f"  Contencion (recuperados/overbookings): {contencion:5.1f}%")
        print(f"  Delta P90 tiempo de espera vs. sin IA: {media_p90:+5.1f} min")
        print(f"  Ahorro mensual ({COSTE_HORA_MEDICO:.0f} EUR/h medico): {ahorro_medio:8,.0f} EUR  (IC95% [{ahorro_low:,.0f} - {ahorro_high:,.0f}])")
        print(f"  Impacto anual proyectado:      {ahorro_medio * 12:8,.0f} EUR  (IC95% [{ahorro_low * 12:,.0f} - {ahorro_high * 12:,.0f}])")

    print("\n" + "=" * 78)
    print("Metodologia: media +/- IC95% de N realizaciones (meses) independientes,")
    print("no un unico run suelto. Mismo criterio que 'Claude outputs/datos_tecnicos_completados.md'.")


if __name__ == "__main__":
    main()
