# -*- coding: utf-8 -*-
"""
Dashboard Interactivo DEFINITIVO - Hospital Smart Slotting
Para ejecutar: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from scipy.stats import johnsonsu, gaussian_kde
from pathlib import Path
import time

# --- CONFIGURACIÓN DE LA PÁGINA WEB ---
st.set_page_config(page_title="Hospital AI Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- MEMORIA (SESSION STATE) PARA CONGELAR LA SIMULACIÓN Y ANIMACIÓN ---
if 'semilla_global' not in st.session_state:
    st.session_state.semilla_global = 42
if 'prev_config' not in st.session_state:
    st.session_state.prev_config = None

# --- RUTAS Y CONSTANTES ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODELO_JOBLIB_PATH = MODELS_DIR / "modelo_definitivo.joblib"
CSV_PATH = DATA_DIR / "dataset_limpio.csv"

WORK_START_HOUR = 9
WORK_END_HOUR = 15

FEATURES = [
    'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 
    'Handcap', 'SMS_received', 'Days_between', 'Appointment_Day_of_Week', 
    'Scheduled_Day_of_Week', 'Weekend', 'Appointment_Month', 
    'Scheduled_Month', 'Faltas_Previas', 'Citas_Previas', 'Ratio_Faltas', 
    'Gender_M', 'Scheduled_Time_of_Day_Evening', 'Scheduled_Time_of_Day_Morning'
]

# Parámetros JSU suavizados (pico en llegadas tempranas realistas)
J_A, J_B, J_LOC, J_SCALE = 1.5, 1.5, -2.0, 12.0

# --- CACHÉ DE DATOS ---
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    col = "No-show" if "No-show" in df.columns else "Falta_Real"
    df["Falta_Real"] = df[col].map({"Yes": True, "No": False, "1": True, "0": False, 1: True, 0: False}) if df[col].dtype == "object" else df[col] == 1
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODELO_JOBLIB_PATH)

# --- MOTOR DE LÓGICA ---
def get_day_patients_pool(df, day_index, modelo_ia, n_pacientes, semilla):
    pool = df.sample(n=n_pacientes, random_state=day_index + semilla).copy()
    pool["Prob_IA"] = modelo_ia.predict_proba(pool[FEATURES])[:, 1]
    
    patients = [{"real_no_show": bool(row["Falta_Real"]), "prob_no_show": float(row["Prob_IA"])} for _, row in pool.iterrows()]
    return patients, pool

def create_schedule(day_patients_pool, smart_overbooking, umbral_riesgo, total_slots, skip_slots):
    agenda = {slot: [] for slot in range(total_slots)}
    pool_idx = 0
    
    for slot in range(total_slots):
        if slot in skip_slots: continue
        if pool_idx < len(day_patients_pool):
            paciente_titular = day_patients_pool[pool_idx].copy()
            paciente_titular["rol"] = "Titular"
            agenda[slot].append(paciente_titular)
            pool_idx += 1

    if smart_overbooking:
        for slot in range(total_slots):
            if slot in skip_slots: continue
            if agenda[slot]:
                titular = agenda[slot][0]
                if pool_idx < len(day_patients_pool):
                    candidato = day_patients_pool[pool_idx].copy()
                    candidato["rol"] = "Refuerzo (Overbooking)"
                    if titular["prob_no_show"] > umbral_riesgo or candidato["prob_no_show"] > umbral_riesgo:
                        agenda[slot].append(candidato)
                        pool_idx += 1
    return agenda

def simulate_day(agenda, descanso_fijo, total_slots, slot_minutes, umbral_riesgo, semilla, descanso_start, skip_slots):
    slots_status_real = np.full(total_slots, 3, dtype=int)
    slots_status_plan = np.full(total_slots, 10, dtype=int)
    
    hover_info_real = [""] * total_slots 
    hover_info_plan = [""] * total_slots 
    pacientes_del_dia = []
    
    rng = np.random.RandomState(semilla)
    
    # 1. GENERAR MAPA DE PLANIFICACIÓN (Agenda estática)
    for slot in range(total_slots):
        if slot in skip_slots:
            if slot >= total_slots - 6:
                slots_status_plan[slot] = 14
                hover_info_plan[slot] = "📝 Bloqueo: Informes"
            else:
                slots_status_plan[slot] = 13
                hover_info_plan[slot] = "☕ Bloqueo: Descanso Programado"
        else:
            n_pacs = len(agenda[slot])
            if n_pacs == 0:
                slots_status_plan[slot] = 10
                hover_info_plan[slot] = "🪑 Hueco Libre (Sin citar)"
            elif n_pacs == 1:
                slots_status_plan[slot] = 11
                p = agenda[slot][0]
                hover_info_plan[slot] = f"👤 <b>Cita Normal</b><br>Riesgo IA del paciente: {p['prob_no_show']*100:.0f}%"
            else:
                slots_status_plan[slot] = 12
                p1, p2 = agenda[slot][0], agenda[slot][1]
                hover_info_plan[slot] = f"🔥 <b>OVERBOOKING IA (2 Pacientes)</b><br>1️⃣ Titular (Riesgo IA): {p1['prob_no_show']*100:.0f}%<br>2️⃣ Refuerzo (Riesgo IA): {p2['prob_no_show']*100:.0f}%"

        for p in agenda[slot]:
            if not p["real_no_show"]:
                desfase = np.clip(johnsonsu.rvs(a=J_A, b=J_B, loc=J_LOC, scale=J_SCALE, size=1, random_state=rng)[0], -30, 10)
                minuto_citado = slot * slot_minutes
                minuto_llegada = max(0, minuto_citado + desfase)
                p_llegada = p.copy()
                p_llegada.update({"slot_citado": slot, "minuto_citado": minuto_citado, "minuto_llegada": minuto_llegada})
                pacientes_del_dia.append(p_llegada)

    pacientes_del_dia.sort(key=lambda x: x["minuto_llegada"])
    
    sala_de_espera, idx_llegadas, atendidos, max_retraso_dia = [], 0, 0, 0
    descansos_pendientes, en_descanso = 2, False
    sala_history = []
    diferencias_agenda = [] 
    
    # 2. SIMULAR REALIDAD (Mapa Real)
    for slot_actual in range(total_slots):
        minuto_actual_reloj = slot_actual * slot_minutes
        
        while idx_llegadas < len(pacientes_del_dia) and pacientes_del_dia[idx_llegadas]["minuto_llegada"] <= minuto_actual_reloj:
            sala_de_espera.append(pacientes_del_dia[idx_llegadas])
            idx_llegadas += 1

        sala_history.append(len(sala_de_espera))

        if len(sala_de_espera) == 0 and idx_llegadas == len(pacientes_del_dia):
            slots_status_real[slot_actual] = 5
            hover_info_real[slot_actual] = "📝 Informes (Fin de Citas)"
            continue

        if slot_actual >= total_slots - 6: 
            if len(sala_de_espera) > 0:
                sala_de_espera.sort(key=lambda x: x["slot_citado"])
                paciente = sala_de_espera.pop(0)
                retraso_agenda = minuto_actual_reloj - paciente["minuto_citado"]
                max_retraso_dia = max(max_retraso_dia, retraso_agenda)
                
                slots_status_real[slot_actual] = 2; atendidos += 1
                diferencias_agenda.append(retraso_agenda)
                
                eval_ia = "✅ Acertó (Vino)" if paciente['prob_no_show'] <= umbral_riesgo else "❌ Falló (Dijo que faltaría)"
                hover_info_real[slot_actual] = f"👤 {paciente['rol']}<br>🤖 Predicción IA: {paciente['prob_no_show']*100:.0f}% falta ({eval_ia})<br>⏳ Retraso asistencial: {int(retraso_agenda)} min"
            else:
                slots_status_real[slot_actual] = 5 
                hover_info_real[slot_actual] = "📝 Informes (Fin de turno)"
            continue 

        if descanso_fijo:
            if slot_actual in [descanso_start, descanso_start + 1]: 
                slots_status_real[slot_actual] = 4
                hover_info_real[slot_actual] = "☕ Descanso por Convenio"
                continue
        else:
            if en_descanso:
                slots_status_real[slot_actual] = 4; descansos_pendientes -= 1
                hover_info_real[slot_actual] = "☕ Descanso Flexible"
                if descansos_pendientes == 0: en_descanso = False
                continue
            if descansos_pendientes == 2 and slot_actual >= descanso_start - 1:
                if len(sala_de_espera) == 0 or slot_actual >= descanso_start + 4:
                    slots_status_real[slot_actual] = 4; descansos_pendientes -= 1; en_descanso = True
                    hover_info_real[slot_actual] = "☕ Descanso Flexible"
                    continue
                
        if len(sala_de_espera) > 0:
            sala_de_espera.sort(key=lambda x: x["slot_citado"])
            paciente = sala_de_espera.pop(0)
            atendidos += 1
            
            retraso_agenda = minuto_actual_reloj - paciente["minuto_citado"]
            diferencias_agenda.append(retraso_agenda)
            
            llegada_relativa = paciente['minuto_llegada'] - paciente['minuto_citado']
            txt_llegada = f"{int(llegada_relativa)} min" if llegada_relativa < 0 else f"+{int(llegada_relativa)} min"
            eval_ia = "✅ Acertó (Vino)" if paciente['prob_no_show'] <= umbral_riesgo else "❌ Falló (Dijo que faltaría)"

            base_texto = f"👤 <b>{paciente['rol']}</b><br>🤖 Riesgo IA evaluado: {paciente['prob_no_show']*100:.0f}% ({eval_ia})<br>🚶 Llegada a sala: {txt_llegada} respecto a cita"
            
            if retraso_agenda < 0:
                slots_status_real[slot_actual] = 6 
                hover_info_real[slot_actual] = f"{base_texto}<br>🚀 Entra adelantado: {int(abs(retraso_agenda))} min"
            else:
                max_retraso_dia = max(max_retraso_dia, retraso_agenda)
                slots_status_real[slot_actual] = 1 if retraso_agenda < 5 else 2 
                estado_txt = "Puntual" if retraso_agenda < 5 else f"Retraso de {int(retraso_agenda)} min"
                hover_info_real[slot_actual] = f"{base_texto}<br>⏳ Atención: {estado_txt}"
        else:
            if len(agenda[slot_actual]) > 0:
                slots_status_real[slot_actual] = 0 
                titular = agenda[slot_actual][0]
                eval_ia = "✅ IA Acertó (Previó la falta)" if titular['prob_no_show'] > umbral_riesgo else "❌ IA Falló (No lo vio venir)"
                hover_info_real[slot_actual] = f"👻 <b>Faltó a la cita</b><br>🤖 Riesgo IA evaluado: {titular['prob_no_show']*100:.0f}%<br>Resultado: {eval_ia}"
            else:
                slots_status_real[slot_actual] = 3 
                hover_info_real[slot_actual] = "🪑 Hueco Vacío"

    return slots_status_real, atendidos, max_retraso_dia, sala_history, hover_info_real, diferencias_agenda, slots_status_plan, hover_info_plan

# --- FRONTEND ---
def main():
    st.title("🏥 Centro de Mando: Gestión Hospitalaria con IA")

    with st.spinner("Cargando base de datos y modelos..."):
        df = load_data()
        modelo = load_model()

    # --- BARRA LATERAL ---
    st.sidebar.header("🎛️ Parámetros de Simulación")
    
    if st.sidebar.button("🎲 Generar Nuevo Mes", use_container_width=True):
        st.session_state.semilla_global += 1

    st.sidebar.divider()
    
    modo = st.sidebar.radio("Escenario Operativo", ["1. Tradicional Fijo", "2. Tradicional Flexible", "3. IA Inteligente (Overbooking)"])
    umbral = 0.40
    if "IA" in modo:
        umbral = st.sidebar.slider("Umbral de Riesgo de IA", min_value=0.10, max_value=0.90, value=0.40, step=0.05)

    with st.sidebar.expander("⚙️ Ajustes del Sistema"):
        st.markdown("⚠️ *Parámetros operativos y Financieros*")
        n_pac_dia = st.number_input("Número de pacientes / día", value=60)
        slot_mins = st.number_input("Minutos por slot", value=10)
        descanso_start = st.number_input("Slot inicio descanso", value=12)
        coste_hora = st.slider("Coste del médico / hora (€)", min_value=50, max_value=300, value=120, step=10)
        velocidad = st.slider("Velocidad de Animación (seg)", 0.0, 0.5, 0.1, 0.05)
        dias_a_simular = st.number_input("Días a simular", value=30, max_value=30)

    # Recalcular constantes
    TOTAL_SLOTS = ((WORK_END_HOUR - WORK_START_HOUR) * 60) // slot_mins
    EMPTY_BREAK_SLOTS = {descanso_start, descanso_start + 1}
    ADMIN_SLOTS = set(range(TOTAL_SLOTS - 6, TOTAL_SLOTS))
    SKIP_SLOTS = EMPTY_BREAK_SLOTS.union(ADMIN_SLOTS)

    descanso_fijo = "Fijo" in modo
    overbooking = "IA" in modo
    semilla = st.session_state.semilla_global 

    # --- CONTROL DE ANIMACIÓN ---
    config_actual = (modo, umbral, n_pac_dia, slot_mins, descanso_start, dias_a_simular, semilla)
    if st.session_state.prev_config != config_actual:
        animar = True
        st.session_state.prev_config = config_actual
    else:
        animar = False

    # --- CÁLCULO DEL ROI INVISIBLE ---
    baseline_atendidos_cum = []
    baseline_retraso_cum = []
    acum_atendidos = 0
    acum_retraso = 0
    
    for day in range(dias_a_simular):
        pool_dict_base, _ = get_day_patients_pool(df, day, modelo, n_pac_dia, semilla)
        agenda_base = create_schedule(pool_dict_base, False, 0, TOTAL_SLOTS, SKIP_SLOTS)
        _, at_base, ret_base, _, _, _, _, _ = simulate_day(agenda_base, True, TOTAL_SLOTS, slot_mins, 0, day + semilla, descanso_start, SKIP_SLOTS)
        acum_atendidos += at_base
        acum_retraso = max(acum_retraso, ret_base)
        baseline_atendidos_cum.append(acum_atendidos)
        baseline_retraso_cum.append(acum_retraso)

    # --- VARIABLES PARA LA ANIMACIÓN ---
    month_matrix_real = np.zeros((dias_a_simular, TOTAL_SLOTS), dtype=int)
    month_matrix_plan = np.zeros((dias_a_simular, TOTAL_SLOTS), dtype=int)
    month_hover_real = np.empty((dias_a_simular, TOTAL_SLOTS), dtype=object)
    month_hover_plan = np.empty((dias_a_simular, TOTAL_SLOTS), dtype=object)
    
    stats_atendidos, stats_max_retraso = 0, 0
    probabilidades_mes = []
    month_sala_history = []
    month_diferencias = [] 

    # --- DISPOSICIÓN: 4 COLUMNAS PARA LOS KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    kpi_atendidos = c1.empty()
    kpi_retraso = c2.empty()
    kpi_roi_dias = c3.empty()
    kpi_roi_dinero = c4.empty()
        
    st.divider()

    col_heatmap, col_stats = st.columns([7, 3]) 
    
    with col_heatmap:
        tipo_vista = st.radio("👁️ Alternar Vista del Mapa", ["Resultados Reales (Asistencia)", "Agenda Programada (Capa IA)"], horizontal=True)
        heatmap_placeholder = st.empty() 

    # --- BUCLE DE SIMULACIÓN ANIMADO ---
    for day in range(dias_a_simular):
        pool_dict, pool_df = get_day_patients_pool(df, day, modelo, n_pac_dia, semilla)
        probabilidades_mes.extend(pool_df["Prob_IA"].tolist())
        
        agenda = create_schedule(pool_dict, overbooking, umbral, TOTAL_SLOTS, SKIP_SLOTS)
        mat_real, atendidos, max_ret, sala_hist, hov_real, dif_dia, mat_plan, hov_plan = simulate_day(agenda, descanso_fijo, TOTAL_SLOTS, slot_mins, umbral, day + semilla, descanso_start, SKIP_SLOTS)
        
        month_matrix_real[day] = mat_real
        month_matrix_plan[day] = mat_plan
        month_hover_real[day] = hov_real
        month_hover_plan[day] = hov_plan
        
        stats_atendidos += atendidos
        stats_max_retraso = max(stats_max_retraso, max_ret)
        month_sala_history.append(sala_hist)
        month_diferencias.extend(dif_dia)

        # Si toca animar o es el último día, pintamos todo
        if animar or day == dias_a_simular - 1:
            delta_atendidos = stats_atendidos - baseline_atendidos_cum[day]
            delta_retraso = stats_max_retraso - baseline_retraso_cum[day]
            
            media_diaria_base = baseline_atendidos_cum[day] / (day + 1)
            dias_ganados = delta_atendidos / media_diaria_base if media_diaria_base > 0 else 0
            ahorro_euros = delta_atendidos * (slot_mins / 60.0) * coste_hora

            kpi_atendidos.metric("👥 Total Pacientes Atendidos", f"{stats_atendidos}", delta=f"{delta_atendidos} extra vs Realidad")
            kpi_retraso.metric("⏱️ Peor Retraso del Mes", f"{stats_max_retraso:.1f} min", delta=f"{delta_retraso:.1f} min vs Realidad" if delta_retraso != 0 else None, delta_color="inverse")
            kpi_roi_dias.metric("⏳ ROI: Tiempo Médico", f"+{dias_ganados:.1f} días", delta="Días de trabajo ahorrados")
            kpi_roi_dinero.metric("💰 ROI: Impacto Económico", f"+{ahorro_euros:,.0f} €", delta=f"Cálculo base: {coste_hora}€/h")

            if "Reales" in tipo_vista:
                matriz_a_dibujar = month_matrix_real
                hover_a_dibujar = month_hover_real
                color_map = {
                    0: ("#3b3b3b", "Falta del Paciente"), 1: ("#2ecc71", "A Tiempo"), 
                    2: ("#f39c12", "Retraso"), 3: ("#1e1e1e", "Hueco Vacío"), 
                    4: ("#ffffff", "Descanso"), 5: ("#9b59b6", "Informes"), 6: ("#3498db", "Adelantado")
                }
                colorscale = [[i/6, color_map[i][0]] for i in range(7)]
            else:
                matriz_a_dibujar = month_matrix_plan
                hover_a_dibujar = month_hover_plan
                color_map = {
                    10: ("#1e1e1e", "Hueco Libre"), 11: ("#2ecc71", "Cita Normal"), 
                    12: ("#e74c3c", "Overbooking IA (Doble)"), 13: ("#ffffff", "Descanso Fijo"), 
                    14: ("#9b59b6", "Informes")
                }
                colorscale = [[(i-10)/4, color_map[i][0]] for i in range(10, 15)]

            text_matrix = np.empty_like(matriz_a_dibujar, dtype=object)
            labels_x = [f"{WORK_START_HOUR + (i * slot_mins // 60)}:{(i * slot_mins) % 60:02d}" for i in range(TOTAL_SLOTS)]
            labels_y = [f"Día {d+1}" for d in range(dias_a_simular)]

            for i in range(day + 1):
                for j in range(TOTAL_SLOTS):
                    color_hex, texto_base = color_map[matriz_a_dibujar[i, j]]
                    info_extra = hover_a_dibujar[i][j] if hover_a_dibujar[i][j] is not None else ""
                    text_matrix[i, j] = f"<b>{labels_y[i]} - {labels_x[j]}</b><br>Estado: {texto_base}<br>{info_extra}"

            fig_heat = go.Figure(data=go.Heatmap(
                z=matriz_a_dibujar, text=text_matrix, hoverinfo="text", 
                colorscale=colorscale, showscale=False, xgap=1, ygap=1
            ))
            # Mapa estirado a height=850 para hacer juego visual con las 5 gráficas
            fig_heat.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(TOTAL_SLOTS)), ticktext=labels_x, tickangle=-90), yaxis=dict(tickmode='array', tickvals=list(range(dias_a_simular)), ticktext=labels_y, autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=850, margin=dict(l=0, r=0, t=10, b=0))
            
            if animar:
                heatmap_placeholder.plotly_chart(fig_heat, use_container_width=True)
                if velocidad > 0:
                    time.sleep(velocidad)

    # Lanzamos el mapa completado rápido si no había que animar
    if not animar:
        heatmap_placeholder.plotly_chart(fig_heat, use_container_width=True)

    # --- GRÁFICAS DE LA DERECHA ---
    with col_stats:
        st.subheader("📈 Telemetría Clínica")
        
        # 1. Gráfica Radar IA
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=probabilidades_mes, nbinsx=30, marker_color='#9b59b6', opacity=0.7, name="Pacientes"))
        if overbooking:
            fig_hist.add_vline(x=umbral, line_dash="dash", line_color="#e74c3c", line_width=2, annotation_text=f"Corte IA ({umbral})", annotation_position="top right")
        fig_hist.update_layout(title="Riesgo de Ausencia (XGBoost)", xaxis_title="Probabilidad de falta", yaxis_title="Volumen", height=160, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)

        # 2. Gráfica JSU de Llegadas Reales
        x_vals = np.linspace(-30, 10, 300)
        y_vals = johnsonsu.pdf(x_vals, a=J_A, b=J_B, loc=J_LOC, scale=J_SCALE)
        fig_johnson = go.Figure()
        fig_johnson.add_trace(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy', mode='lines', line=dict(color='#3498db', width=2)))
        fig_johnson.add_vline(x=0, line_dash="solid", line_color="white", annotation_text="Hora Cita (0)")
        fig_johnson.update_layout(title="Modelo de Llegadas Físicas", xaxis_title="Minutos (Negativo = Pronto)", yaxis_title="Densidad", height=160, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_johnson, use_container_width=True)

        # 3. Gráfica Sala de Espera
        avg_sala = np.mean(month_sala_history, axis=0)
        fig_sala = go.Figure()
        fig_sala.add_trace(go.Scatter(x=labels_x, y=avg_sala, fill='tozeroy', mode='lines', line=dict(color='#f39c12', width=2)))
        fig_sala.update_layout(title="Densidad Sala de Espera Promedio", xaxis=dict(tickmode='array', tickvals=list(range(0, TOTAL_SLOTS, 6)), ticktext=labels_x[0::6]), yaxis_title="Pacientes", height=160, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sala, use_container_width=True)

        # 4. y 5. Curvas KDE (PDF Normalizada) y CDF Acumulada
        if len(month_diferencias) > 1:
            try:
                kde = gaussian_kde(month_diferencias, bw_method=0.4)
                x_min = min(-5, min(month_diferencias) - 5)
                x_max = max(5, max(month_diferencias) + 5)
                
                # --- GRÁFICA 4: PDF (PROBABILIDAD NORMALIZADA) ---
                x_adelantos = np.linspace(x_min, 0, 100)
                y_adelantos = kde(x_adelantos) # Ya no multiplica, es densidad pura
                
                x_retrasos = np.linspace(0, x_max, 100)
                y_retrasos = kde(x_retrasos) 
                
                fig_dif = go.Figure()
                fig_dif.add_trace(go.Scatter(x=x_adelantos, y=y_adelantos, fill='tozeroy', mode='lines', line=dict(color='#3498db', width=2), name="Adelantados"))
                fig_dif.add_trace(go.Scatter(x=x_retrasos, y=y_retrasos, fill='tozeroy', mode='lines', line=dict(color='#f39c12', width=2), name="Retrasados"))
                
                fig_dif.add_vline(x=0, line_dash="solid", line_color="white")
                fig_dif.update_layout(title="Probabilidad de Desviación (PDF)", xaxis_title="Minutos (Neg = Adelanto, Pos = Retraso)", yaxis_title="Densidad", height=160, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(fig_dif, use_container_width=True)

                # --- GRÁFICA 5: CDF (PROBABILIDAD ACUMULADA) ---
                x_full = np.linspace(x_min, x_max, 200)
                y_pdf = kde(x_full)
                dx = x_full[1] - x_full[0]
                y_cdf = np.cumsum(y_pdf) * dx # Integración numérica para sacar la CDF
                
                # Calcular percentiles reales para pintar líneas
                perc_50 = np.percentile(month_diferencias, 50)
                perc_90 = np.percentile(month_diferencias, 90)

                fig_cdf = go.Figure()
                fig_cdf.add_trace(go.Scatter(x=x_full, y=y_cdf, fill='tozeroy', mode='lines', line=dict(color='#9b59b6', width=2), name="Acumulada"))
                
                # Añadir líneas de percentiles clave
                fig_cdf.add_vline(x=perc_50, line_dash="dash", line_color="#2ecc71", annotation_text=f"Mediana: {perc_50:.1f} min", annotation_position="bottom right")
                fig_cdf.add_vline(x=perc_90, line_dash="dash", line_color="#e74c3c", annotation_text=f"P90: {perc_90:.1f} min", annotation_position="bottom right")
                
                fig_cdf.update_layout(title="Probabilidad Acumulada (CDF)", xaxis_title="Minutos", yaxis_title="Probabilidad (0-1)", height=160, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(fig_cdf, use_container_width=True)

            except:
                st.write("Calculando desviaciones...")

if __name__ == "__main__":
    main()