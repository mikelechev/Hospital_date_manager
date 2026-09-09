# -*- coding: utf-8 -*-
"""Simulación mensual de agenda hospitalaria con comparación normal vs overbooking.

Objetivo:
- Representar un mes completo (30 días)
- 5 horas de trabajo al día
- 10 minutos de cortafuegos cada 2 horas para absorber solapamientos
- Comparar escenario normal vs overbooking
- Mostrar estadísticas relevantes y una animación mensual
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = MODELS_DIR / "modelo_campeon.json"
CSV_PATH = DATA_DIR / "dataset_limpio.csv"

DAYS_IN_MONTH = 30
HOURS_PER_DAY = 5
WORK_START_HOUR = 9
BREAKS = [(11 * 60, 11 * 60 + 10), (13 * 60, 13 * 60 + 10)]


def build_daily_slots():
    """Genera los slots de consulta por día con descansos de 10 min cada 2h."""
    slots = []
    start_minute = WORK_START_HOUR * 60
    end_minute = (WORK_START_HOUR + HOURS_PER_DAY) * 60
    minute = start_minute

    while minute < end_minute:
        slot_end = minute + 15
        if any(minute < b_end and slot_end > b_start for b_start, b_end in BREAKS):
            minute = slot_end
            continue
        slots.append((minute, slot_end))
        minute = slot_end

    return slots


def load_model():
    if MODEL_PATH.exists():
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        return model
    return None


def prepare_dataset():
    df = pd.read_csv(CSV_PATH)

    if "Days_between" in df.columns and "Days-between" not in df.columns:
        df["Days-between"] = df["Days_between"]
    elif "Wait_Time" in df.columns and "Days-between" not in df.columns:
        df["Days-between"] = df["Wait_Time"]

    if "Weekend" not in df.columns:
        df["Weekend"] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
    if "Ratio_Faltas" not in df.columns:
        df["Ratio_Faltas"] = np.random.beta(0.5, 2.0, len(df))

    if df["No-show"].dtype == "object":
        df["Falta_Real"] = df["No-show"].map({"Yes": True, "No": False, "1": True, "0": False, 1: True, 0: False})
    else:
        df["Falta_Real"] = df["No-show"] == 1

    return df


def build_patient_profile(patient_record, model):
    features = [
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

    if isinstance(patient_record, pd.Series):
        row = patient_record
    else:
        row = pd.Series(patient_record)

    prob = 0.5
    if model is not None:
        try:
            feature_row = row[features].to_frame().T
            prob = float(model.predict_proba(feature_row)[0, 1])
        except Exception:
            prob = 0.5

    no_show_value = row.get("Falta_Real", row.get("No-show", 0))
    return {
        "risk": float(np.clip(prob, 0.0, 1.0)),
        "no_show": bool(no_show_value),
    }


def choose_slot_for_patient(current_slot_idx, slot_occupancy, overbooking, patient_risk):
    """Devuelve el slot donde se asigna el paciente y si hay solapamiento."""
    slot_idx = current_slot_idx
    if overbooking and slot_occupancy[slot_idx] == 1:
        p_existing = 0.55
        p_new = patient_risk
        prob_ambos_vienen = (1.0 - p_existing) * (1.0 - p_new)
        prob_al_menos_uno = 1.0 - (p_existing * p_new)
        if prob_ambos_vienen < 0.25 and prob_al_menos_uno > 0.8:
            return slot_idx, True

    elif slot_occupancy[slot_idx] >= 1:
        for next_idx in range(slot_idx + 1, len(slot_occupancy)):
            if slot_occupancy[next_idx] < 1:
                return next_idx, False
        return slot_idx, False

    return slot_idx, False


def simulate_month(overbooking_mode=False, model=None, df=None):
    slots = build_daily_slots()
    month_rows = []
    total_delay = 0
    total_empty_slots = 0
    total_overlaps = 0
    total_waiting = 0
    total_no_show = 0
    total_assigned = 0
    total_confirmed = 0
    idle_minutes = 0

    for day in range(DAYS_IN_MONTH):
        patients = df.sample(n=np.random.randint(18, 34), random_state=day + 42).copy()
        occupancy = np.zeros(len(slots), dtype=int)
        delay_by_slot = np.zeros(len(slots), dtype=int)
        day_data = []

        for idx, patient in enumerate(patients.itertuples(index=False)):
            patient_dict = patient._asdict()
            profile = build_patient_profile(patient_dict, model)
            patient_risk = profile["risk"]
            slot_idx = 0
            assigned_slot = 0
            overlap = False

            while slot_idx < len(slots):
                if occupancy[slot_idx] == 0:
                    assigned_slot = slot_idx
                    break
                if overbooking_mode and occupancy[slot_idx] == 1:
                    p_existing = max(0.05, min(0.95, patient_risk))
                    prob_ambos_vienen = (1.0 - p_existing) * (1.0 - patient_risk)
                    prob_al_menos_uno = 1.0 - (p_existing * patient_risk)
                    if prob_ambos_vienen < 0.25 and prob_al_menos_uno > 0.8:
                        assigned_slot = slot_idx
                        overlap = True
                        break
                slot_idx += 1

            if assigned_slot < 0:
                assigned_slot = min(len(slots) - 1, assigned_slot)

            occupancy[assigned_slot] += 1
            if overlap:
                delay_by_slot[assigned_slot] += 15
                total_overlaps += 1
            else:
                if occupancy[assigned_slot] > 1:
                    delay_by_slot[assigned_slot] += 15
                    total_overlaps += 1

            if profile["no_show"]:
                total_no_show += 1
                idle_minutes += 15
            else:
                total_confirmed += 1

            total_assigned += 1
            day_data.append({
                "slot_index": assigned_slot,
                "time": slots[assigned_slot][0],
                "risk": patient_risk,
                "delay_min": delay_by_slot[assigned_slot],
                "no_show": profile["no_show"],
                "overlap": overlap,
            })

        for i in range(len(slots)):
            if occupancy[i] == 0:
                total_empty_slots += 1
            total_delay += delay_by_slot[i]
            total_waiting += max(0, occupancy[i] - 1) * 15

        month_rows.append({
            "day": day + 1,
            "schedule": occupancy.copy(),
            "delay_minutes": int(delay_by_slot.sum()),
            "mean_delay": float(delay_by_slot.mean()) if len(delay_by_slot) else 0.0,
            "empty_slots": int(np.sum(occupancy == 0)),
            "utilization": float(np.mean(occupancy > 0) * 100),
            "patients": int(len(patients)),
            "confirmed": int(np.sum(np.array([p["no_show"] is False for p in day_data])) if day_data else 0),
            "no_show": int(np.sum(np.array([p["no_show"] for p in day_data])) if day_data else 0),
            "overlaps": int(np.sum(np.array([p["overlap"] for p in day_data])) if day_data else 0),
        })

    summary = {
        "scenario": "overbooking" if overbooking_mode else "normal",
        "total_days": DAYS_IN_MONTH,
        "total_assigned": total_assigned,
        "total_confirmed": total_confirmed,
        "total_no_show": total_no_show,
        "empty_slots": total_empty_slots,
        "avg_delay_minutes": total_delay / DAYS_IN_MONTH,
        "avg_waiting_minutes": total_waiting / DAYS_IN_MONTH,
        "idle_minutes": idle_minutes,
        "mean_overlaps_per_day": total_overlaps / DAYS_IN_MONTH,
        "monthly_matrix": np.array([entry["schedule"] for entry in month_rows]),
    }
    return summary, month_rows


def render_animation(matrix, title, output_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix[0:1, :], aspect="auto", cmap="YlOrRd", vmin=0, vmax=2)
    ax.set_title(f"{title} - Día 1")
    ax.set_xlabel("Slots del día")
    ax.set_ylabel("Día del mes")
    ax.set_yticks([])
    x_labels = [f"{i + 1}" for i in range(matrix.shape[1])]
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(x_labels, rotation=90)

    def update(frame):
        im.set_data(matrix[frame:frame + 1, :])
        ax.set_title(f"{title} - Día {frame + 1}")
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=range(matrix.shape[0]), interval=400, blit=True)
    anim.save(output_path, writer="pillow", fps=3)
    plt.close(fig)


def print_summary(label, summary):
    print(f"\n{'=' * 80}")
    print(f"ESCENARIO: {label.upper()}")
    print(f"{'=' * 80}")
    print(f"Pacientes asignados: {summary['total_assigned']}")
    print(f"Pacientes confirmados: {summary['total_confirmed']}")
    print(f"No-shows: {summary['total_no_show']}")
    print(f"Huecos vacíos: {summary['empty_slots']}")
    print(f"Retraso medio por día: {summary['avg_delay_minutes']:.1f} min")
    print(f"Espera media por día: {summary['avg_waiting_minutes']:.1f} min")
    print(f"Solapamientos medio por día: {summary['mean_overlaps_per_day']:.2f}")
    print(f"Tiempo improductivo (ausencias): {summary['idle_minutes']} min")
    print(f"Utilización media del día: {summary['monthly_matrix'].mean(axis=0).mean() * 100:.1f}%")


def main():
    model = load_model()
    df = prepare_dataset()

    normal_summary, _ = simulate_month(overbooking_mode=False, model=model, df=df)
    over_summary, _ = simulate_month(overbooking_mode=True, model=model, df=df)

    print_summary("normal", normal_summary)
    print_summary("overbooking", over_summary)

    normal_matrix = normal_summary["monthly_matrix"]
    over_matrix = over_summary["monthly_matrix"]

    render_animation(normal_matrix, "Agenda mensual - Sin overbooking", ROOT_DIR / "scripts" / "mc_3_normal.gif")
    render_animation(over_matrix, "Agenda mensual - Con overbooking", ROOT_DIR / "scripts" / "mc_3_overbooking.gif")

    print(f"\nAnimaciones guardadas en:")
    print(f"- {ROOT_DIR / 'scripts' / 'mc_3_normal.gif'}")
    print(f"- {ROOT_DIR / 'scripts' / 'mc_3_overbooking.gif'}")

    print("\nResumen comparativo final:")
    print(f"- Normal: retraso medio {normal_summary['avg_delay_minutes']:.1f} min / día")
    print(f"- Overbooking: retraso medio {over_summary['avg_delay_minutes']:.1f} min / día")
    print(f"- Normal: huecos vacíos {normal_summary['empty_slots']}")
    print(f"- Overbooking: huecos vacíos {over_summary['empty_slots']}")
    print(f"- Normal: solapamientos medio {normal_summary['mean_overlaps_per_day']:.2f}/día")
    print(f"- Overbooking: solapamientos medio {over_summary['mean_overlaps_per_day']:.2f}/día")


if __name__ == "__main__":
    main()
