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

Ademas de las metricas originales (huecos recuperados, overbookings,
contencion, delta P90, ahorro), incluye:
  - Volumen absoluto: citados / atendidos / no-shows reales por mes.
  - Colisiones reales del overbooking (ambos pacientes se presentan).
  - Distribucion completa de tiempos de espera (media, mediana, P95, max,
    % de pacientes con retraso >15 y >30 min).
  - Precision / recall / tasa de falsos positivos / F1 del modelo en el
    umbral usado (evaluados sobre el dataset historico completo, no sobre
    la simulacion de agenda).
  - Ocupacion de la agenda y una aproximacion a las horas extra del medico
    (pacientes atendidos fuera de horario, durante el bloque de informes).

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
SLOTS_PRIMARIOS_DIA = TOTAL_SLOTS - len(SKIP_SLOTS)  # slots de cita "titular" disponibles/dia
CAPACIDAD_PRIMARIA_MES = SLOTS_PRIMARIOS_DIA * DIAS_POR_MES

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


def metricas_modelo(df, modelo, umbral):
    """Precision / recall / F1 / tasa de falsos positivos del clasificador
    sobre TODO el dataset historico, tratando 'prob_no_show > umbral' como
    la decision de overbooking (positivo = predice no-show). Es una foto de
    la calidad del modelo en ese umbral, independiente de la simulacion."""
    prob = modelo.predict_proba(df[FEATURES])[:, 1]
    pred_no_show = prob > umbral
    real_no_show = df["Falta_Real"].to_numpy()

    tp = int(np.sum(pred_no_show & real_no_show))
    fp = int(np.sum(pred_no_show & ~real_no_show))
    fn = int(np.sum(~pred_no_show & real_no_show))
    tn = int(np.sum(~pred_no_show & ~real_no_show))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "fpr": fpr, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


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
    """Devuelve (atendidos, lista_de_retrasos_en_minutos, atendidos_fuera_de_horario)
    para un dia. 'atendidos_fuera_de_horario' cuenta pacientes vistos durante el
    bloque de informes (ultimos 6 slots) porque la sala de espera no se vacio a
    tiempo: es la mejor aproximacion disponible a "horas extra" del medico."""
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

    sala_de_espera, idx_llegada, atendidos, fuera_de_horario = [], 0, 0, 0
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
                fuera_de_horario += 1
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

    return atendidos, retrasos, fuera_de_horario


def run_month(pools, semilla, umbral=None):
    """umbral=None -> escenario base (sin overbooking, descansos fijos).
    Devuelve un dict con todas las metricas del mes."""
    overbooking = umbral is not None
    total_citados = 0
    total_atendidos = 0
    total_no_shows_reales = 0
    total_overbookings = 0
    total_colisiones = 0
    total_fuera_de_horario = 0
    todos_los_retrasos = []

    for day, pool in enumerate(pools):
        agenda = create_schedule(pool, overbooking, umbral if overbooking else 0.0)
        for pacientes_slot in agenda.values():
            total_citados += len(pacientes_slot)
            total_no_shows_reales += sum(1 for p in pacientes_slot if p["real_no_show"])
            if overbooking and len(pacientes_slot) == 2:
                total_overbookings += 1
                if not pacientes_slot[0]["real_no_show"] and not pacientes_slot[1]["real_no_show"]:
                    total_colisiones += 1
        atendidos, retrasos, fuera_de_horario = simulate_day(
            agenda, descanso_fijo=not overbooking, semilla=day + semilla
        )
        total_atendidos += atendidos
        total_fuera_de_horario += fuera_de_horario
        todos_los_retrasos.extend(retrasos)

    retrasos_arr = np.asarray(todos_los_retrasos, dtype=float) if todos_los_retrasos else np.array([0.0])

    return {
        "citados": total_citados,
        "atendidos": total_atendidos,
        "no_shows_reales": total_no_shows_reales,
        "pct_no_shows": (total_no_shows_reales / total_citados * 100) if total_citados > 0 else 0.0,
        "huecos_perdidos": total_citados - total_atendidos,
        "overbookings": total_overbookings,
        "colisiones": total_colisiones,
        "fuera_de_horario": total_fuera_de_horario,
        "espera_media": float(retrasos_arr.mean()),
        "espera_mediana": float(np.median(retrasos_arr)),
        "espera_p90": float(np.percentile(retrasos_arr, 90)),
        "espera_p95": float(np.percentile(retrasos_arr, 95)),
        "espera_max": float(retrasos_arr.max()),
        "pct_retraso_15": float(np.mean(retrasos_arr > 15) * 100),
        "pct_retraso_30": float(np.mean(retrasos_arr > 30) * 100),
    }


def ci95(values):
    values = np.asarray(values, dtype=float)
    media = values.mean()
    error = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return media, media - error, media + error


def fmt_ci(media, low, high, decimales=1):
    return f"{media:.{decimales}f}  (IC95% [{low:.{decimales}f} - {high:.{decimales}f}])"


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

    metricas_dia_keys = [
        "citados", "atendidos", "no_shows_reales", "pct_no_shows", "huecos_perdidos",
        "espera_media", "espera_mediana", "espera_p90", "espera_p95",
        "espera_max", "pct_retraso_15", "pct_retraso_30",
    ]
    resultados_base = {k: [] for k in metricas_dia_keys}

    metricas_ia_keys = metricas_dia_keys + [
        "overbookings", "colisiones", "fuera_de_horario", "extra", "p90_delta",
    ]
    resultados = {umbral: {k: [] for k in metricas_ia_keys} for umbral in args.umbrales}

    for r in range(args.realizaciones):
        semilla = args.semilla_base + r
        pools = precompute_pools(df, modelo, semilla)
        base = run_month(pools, semilla, umbral=None)
        for k in metricas_dia_keys:
            resultados_base[k].append(base[k])

        for umbral in args.umbrales:
            ia = run_month(pools, semilla, umbral=umbral)
            for k in metricas_dia_keys:
                resultados[umbral][k].append(ia[k])
            resultados[umbral]["overbookings"].append(ia["overbookings"])
            resultados[umbral]["colisiones"].append(ia["colisiones"])
            resultados[umbral]["fuera_de_horario"].append(ia["fuera_de_horario"])
            resultados[umbral]["extra"].append(ia["atendidos"] - base["atendidos"])
            resultados[umbral]["p90_delta"].append(ia["espera_p90"] - base["espera_p90"])

        if (r + 1) % 20 == 0 or r == args.realizaciones - 1:
            print(f"  ...{r + 1}/{args.realizaciones} meses simulados")

    coste_slot = (SLOT_MINUTES / 60.0) * COSTE_HORA_MEDICO

    print("\n" + "=" * 78)
    print(f"ESCENARIO BASE (sin IA, {args.realizaciones} meses independientes, IC95%)")
    print("=" * 78)
    b_atendidos, b_at_low, b_at_high = ci95(resultados_base["atendidos"])
    b_noshows, _, _ = ci95(resultados_base["no_shows_reales"])
    b_pct_noshows, _, _ = ci95(resultados_base["pct_no_shows"])
    b_perdidos, b_perdidos_low, b_perdidos_high = ci95(resultados_base["huecos_perdidos"])
    b_ocup = b_atendidos / CAPACIDAD_PRIMARIA_MES * 100
    print(f"  Pacientes citados/mes (capacidad primaria): {CAPACIDAD_PRIMARIA_MES}")
    print(f"  Pacientes atendidos/mes:        {fmt_ci(b_atendidos, b_at_low, b_at_high)}")
    print(f"  No-shows reales/mes:            {b_noshows:6.1f}  ({b_pct_noshows:.1f}% de los citados)")
    print(f"  Huecos perdidos por no-show/mes: {fmt_ci(b_perdidos, b_perdidos_low, b_perdidos_high)}")
    print(f"  Ocupacion de la agenda:         {b_ocup:5.1f}%")
    e_media, _, _ = ci95(resultados_base["espera_media"])
    e_p90, _, _ = ci95(resultados_base["espera_p90"])
    e_p95, _, _ = ci95(resultados_base["espera_p95"])
    e_max, _, _ = ci95(resultados_base["espera_max"])
    p15, _, _ = ci95(resultados_base["pct_retraso_15"])
    p30, _, _ = ci95(resultados_base["pct_retraso_30"])
    print(f"  Espera media / P90 / P95 / max: {e_media:.1f} / {e_p90:.1f} / {e_p95:.1f} / {e_max:.1f} min")
    print(f"  Pacientes con retraso >15min / >30min: {p15:.1f}% / {p30:.1f}%")

    for umbral in args.umbrales:
        datos = resultados[umbral]
        m_modelo = metricas_modelo(df, modelo, umbral)

        media_extra, low_extra, high_extra = ci95(datos["extra"])
        media_over = float(np.mean(datos["overbookings"]))
        media_colisiones = float(np.mean(datos["colisiones"]))
        pct_colisiones = (media_colisiones / media_over * 100) if media_over > 0 else 0.0
        contencion_over = (media_extra / media_over * 100) if media_over > 0 else 0.0
        contencion_total = (media_extra / b_perdidos * 100) if b_perdidos > 0 else 0.0
        media_p90 = float(np.mean(datos["p90_delta"]))

        ia_citados, ia_cit_low, ia_cit_high = ci95(datos["citados"])
        ia_atendidos, ia_at_low, ia_at_high = ci95(datos["atendidos"])
        ia_noshows, _, _ = ci95(datos["no_shows_reales"])
        ia_pct_noshows, _, _ = ci95(datos["pct_no_shows"])
        ia_ocup = ia_atendidos / CAPACIDAD_PRIMARIA_MES * 100
        ia_fuera_horario, _, _ = ci95(datos["fuera_de_horario"])
        horas_extra = ia_fuera_horario * SLOT_MINUTES / 60.0

        ia_e_media, _, _ = ci95(datos["espera_media"])
        ia_e_p95, _, _ = ci95(datos["espera_p95"])
        ia_e_max, _, _ = ci95(datos["espera_max"])
        ia_p15, _, _ = ci95(datos["pct_retraso_15"])
        ia_p30, _, _ = ci95(datos["pct_retraso_30"])

        ahorro_medio = media_extra * coste_slot
        ahorro_low = low_extra * coste_slot
        ahorro_high = high_extra * coste_slot

        print("\n" + "=" * 78)
        print(f"UMBRAL {umbral:.2f} ({args.realizaciones} meses independientes, IC95%)")
        print("=" * 78)

        print("\n  -- Calidad del modelo en este umbral (dataset historico completo) --")
        print(f"  Precision: {m_modelo['precision']*100:5.1f}%   Recall (sensibilidad): {m_modelo['recall']*100:5.1f}%")
        print(f"  Tasa de falsos positivos: {m_modelo['fpr']*100:5.1f}%   F1: {m_modelo['f1']*100:5.1f}%")

        print("\n  -- Volumen --")
        print(f"  Pacientes citados/mes:          {fmt_ci(ia_citados, ia_cit_low, ia_cit_high)}")
        print(f"  Pacientes atendidos/mes:        {fmt_ci(ia_atendidos, ia_at_low, ia_at_high)}")
        print(f"  No-shows reales/mes:            {ia_noshows:6.1f}  ({ia_pct_noshows:.1f}% de los citados)")
        print(f"  Ocupacion de la agenda:         {ia_ocup:5.1f}%")

        print("\n  -- Overbooking --")
        print(f"  Huecos recuperados/mes:         {fmt_ci(media_extra, low_extra, high_extra)}")
        print(f"  Overbookings programados/mes:   {media_over:6.1f}")
        print(f"  Colisiones reales/mes (ambos se presentan): {media_colisiones:5.1f}  ({pct_colisiones:.1f}% de los overbookings)")
        print(f"  Contencion vs. overbookings programados: {contencion_over:5.1f}%")
        print(f"  Contencion vs. huecos totales perdidos por no-show: {contencion_total:5.1f}%")

        print("\n  -- Tiempos de espera --")
        print(f"  Delta P90 vs. sin IA:            {media_p90:+5.1f} min")
        print(f"  Espera media / P95 / max:        {ia_e_media:.1f} / {ia_e_p95:.1f} / {ia_e_max:.1f} min")
        print(f"  Pacientes con retraso >15min / >30min: {ia_p15:.1f}% / {ia_p30:.1f}%")
        print(f"  Atendidos fuera de horario (bloque informes): {ia_fuera_horario:.1f}/mes (~{horas_extra:.1f}h extra medico/mes)")

        print("\n  -- Economico --")
        print(f"  Ahorro mensual ({COSTE_HORA_MEDICO:.0f} EUR/h medico): {ahorro_medio:8,.0f} EUR  (IC95% [{ahorro_low:,.0f} - {ahorro_high:,.0f}])")
        print(f"  Impacto anual proyectado:       {ahorro_medio * 12:8,.0f} EUR  (IC95% [{ahorro_low * 12:,.0f} - {ahorro_high * 12:,.0f}])")

    print("\n" + "=" * 78)
    print("Metodologia: media +/- IC95% de N realizaciones (meses) independientes,")
    print("no un unico run suelto. Mismo criterio que 'Claude outputs/datos_tecnicos_completados.md'.")
    print("Precision/recall/F1 se calculan sobre el dataset historico completo con ese")
    print("umbral como corte de decision, no sobre la simulacion de agenda.")


if __name__ == "__main__":
    main()
