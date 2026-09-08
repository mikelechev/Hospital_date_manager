# -*- coding: utf-8 -*-
"""
Portal del Paciente V6 - E&M HealthTech
Cuadrícula de citas alineada + botones corregidos + Split-Screen.

Nota sobre la cuadrícula de horarios (el fix de esta versión):
Antes se creaba `cols = st.columns(4)` UNA sola vez por día y se
reutilizaba para las 16 citas (col 0 recibía las citas 1, 5, 9, 13...).
Como las tarjetas "Ocupado" no tenían botón (más bajas) y las tarjetas
"Libre"/"SmartSlot" sí (más altas), cada columna iba acumulando una
altura total distinta a las demás y las filas se desalineaban -> caos
visual ("una arriba, otra abajo").

La solución real es doble:
 1) Crear una fila de columnas NUEVA por cada grupo de 4 horas
    (st.columns(4) dentro del bucle de filas), para que las 4 celdas
    de una misma fila sean hermanas en el mismo contenedor flex y se
    alineen de verdad entre sí.
 2) Dar a TODAS las tarjetas la misma altura, incluidas las ocupadas:
    en vez de omitir el botón cuando no hay acción posible, se muestra
    un botón deshabilitado "No disponible" del mismo tamaño.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Portal Paciente | E&M", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# 🎨 SISTEMA DE DISEÑO
# ==========================================
COLOR_BG_APP         = "#F4F6F8"
COLOR_PATIENT_BG     = "#FFFFFF"
COLOR_PATIENT_TEXT   = "#1C1F26"
COLOR_PATIENT_MUTED  = "#6B7280"
COLOR_PATIENT_BORDER = "#E7E9EC"

COLOR_ADMIN_BG       = "#1E2128"
COLOR_ADMIN_BG_ALT   = "#262A33"
COLOR_ADMIN_TEXT     = "#F3F4F6"
COLOR_ADMIN_MUTED    = "#9CA3AF"
COLOR_ADMIN_BORDER   = "#3A3F4A"

COLOR_ACCENT_FREE    = "#16A34A"
COLOR_ACCENT_AI      = "#7C3AED"
COLOR_ACCENT_MUTED   = "#9AA1AC"

st.markdown(f"""
    <style>
    /* ---------- Base ---------- */
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {COLOR_BG_APP};
    }}

    [data-testid="stMainBlockContainer"] {{
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
    }}

    /* ---------- PANEL PACIENTE (izquierda, blanco) ---------- */
    .st-key-patient_panel {{
        background-color: {COLOR_PATIENT_BG};
        border: 1px solid {COLOR_PATIENT_BORDER};
        border-radius: 16px;
        padding: 2.25rem 2.5rem;
        box-shadow: 0px 12px 32px rgba(17, 24, 39, 0.06);
    }}
    .st-key-patient_panel p,
    .st-key-patient_panel span,
    .st-key-patient_panel label {{
        color: {COLOR_PATIENT_TEXT};
    }}
    .st-key-patient_panel h1,
    .st-key-patient_panel h2,
    .st-key-patient_panel h3 {{
        color: {COLOR_PATIENT_TEXT};
        letter-spacing: -0.01em;
    }}
    .st-key-patient_panel [data-testid="stCaptionContainer"] {{
        color: {COLOR_PATIENT_MUTED};
    }}
    .st-key-patient_panel hr {{
        border-color: {COLOR_PATIENT_BORDER};
    }}

    /* Botones Principales (Acceder) */
    .st-key-patient_panel button[kind="primary"],
    .st-key-patient_panel button[kind="primaryFormSubmit"] {{
        background-color: {COLOR_ACCENT_FREE} !important;
        border: 1px solid {COLOR_ACCENT_FREE} !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        font-weight: 600;
    }}
    .st-key-patient_panel button[kind="primary"]:hover,
    .st-key-patient_panel button[kind="primaryFormSubmit"]:hover {{
        background-color: #128A3E !important;
        border-color: #128A3E !important;
    }}

    /* Botones Secundarios (Reservar, Cerrar Sesión) */
    .st-key-patient_panel button[kind="secondary"] {{
        background-color: #FFFFFF !important;
        border-radius: 8px;
        border: 1px solid #D1D5DB !important;
        color: {COLOR_PATIENT_TEXT} !important;
        font-weight: 600;
    }}
    .st-key-patient_panel button[kind="secondary"]:hover {{
        background-color: #F3F4F6 !important;
        border-color: #9CA3AF !important;
        color: #111827 !important;
    }}

    /* Botón deshabilitado "No disponible": mismo tamaño, look apagado,
       para que las tarjetas ocupadas midan igual que las reservables */
    .st-key-patient_panel button:disabled,
    .st-key-patient_panel button[kind="secondary"]:disabled {{
        background-color: #F9FAFB !important;
        border: 1px dashed #E1E4E8 !important;
        color: #C1C6CD !important;
        font-weight: 500;
        opacity: 1 !important;
    }}

    /* ---------- PANEL ADMIN (derecha, gris oscuro) ---------- */
    .st-key-admin_panel {{
        background-color: {COLOR_ADMIN_BG};
        border: 1px solid {COLOR_ADMIN_BORDER};
        border-radius: 16px;
        padding: 2.25rem 2rem;
        box-shadow: inset 0px 0px 0px 1px rgba(255,255,255,0.02),
                    0px 12px 32px rgba(0,0,0,0.25);
    }}
    .st-key-admin_panel p,
    .st-key-admin_panel span,
    .st-key-admin_panel label {{
        color: {COLOR_ADMIN_TEXT};
    }}
    .st-key-admin_panel h1,
    .st-key-admin_panel h2,
    .st-key-admin_panel h3 {{
        color: {COLOR_ADMIN_TEXT};
        letter-spacing: -0.01em;
    }}
    .st-key-admin_panel [data-testid="stCaptionContainer"] {{
        color: {COLOR_ADMIN_MUTED};
    }}
    .st-key-admin_panel hr {{
        border-color: {COLOR_ADMIN_BORDER};
    }}
    .st-key-admin_panel [data-testid="stCodeBlock"] pre {{
        background-color: {COLOR_ADMIN_BG_ALT} !important;
        border: 1px solid {COLOR_ADMIN_BORDER};
        border-radius: 10px;
    }}
    .st-key-admin_panel [data-testid="stCodeBlock"] code {{
        color: {COLOR_ADMIN_TEXT} !important;
    }}
    .st-key-admin_panel [data-testid="stAlertContainer"] {{
        background-color: {COLOR_ADMIN_BG_ALT};
        border: 1px solid {COLOR_ADMIN_BORDER};
        color: {COLOR_ADMIN_TEXT};
        border-radius: 10px;
    }}

    .admin-eyebrow, .patient-eyebrow {{
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }}
    .admin-eyebrow {{ color: {COLOR_ADMIN_MUTED}; }}
    .patient-eyebrow {{ color: {COLOR_PATIENT_MUTED}; }}

    /* ---------- Tarjetas de slots (panel paciente) ---------- */
    /* Alto fijo + flex para que el texto quede centrado igual en las
       tres variantes (Libre / SmartSlot / Ocupado), reforzando la
       sensación de cuadrícula uniforme junto con el botón de abajo. */
    .slot-card, .slot-overbooking, .slot-full {{
        border-radius: 10px;
        padding: 10px 8px;
        text-align: center;
        margin-bottom: 8px;
        font-size: 0.88rem;
        min-height: 54px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .slot-card {{
        border: 1.5px solid {COLOR_ACCENT_FREE};
        background-color: #F0FDF4;
        color: #15803D;
    }}
    .slot-overbooking {{
        border: 1.5px solid {COLOR_ACCENT_AI};
        background-color: #F5F3FF;
        color: #6D28D9;
    }}
    .slot-full {{
        border: 1.5px dashed #D1D5DB;
        background-color: #F9FAFB;
        color: {COLOR_ACCENT_MUTED};
        opacity: 0.9;
    }}
    </style>
""", unsafe_allow_html=True)

# --- BASE DE DATOS DE PACIENTES AMPLIADA ---
PACIENTES_DB = {
    "111": {"nombre": "Mikel Ezkurdia", "edad": 22, "riesgo_propio": 0.10, "historial": "Excelente (100% asistencia)"},
    "222": {"nombre": "Ane Larrañaga", "edad": 65, "riesgo_propio": 0.85, "historial": "Crítico (Falla habitualmente)"},
    "333": {"nombre": "Jon Arretxe", "edad": 41, "riesgo_propio": 0.35, "historial": "Medio (Algún retraso previo)"},
    "444": {"nombre": "Maite Zabaleta", "edad": 29, "riesgo_propio": 0.65, "historial": "Irregular (Riesgo de cancelación)"},
    "555": {"nombre": "Aitor Ocio", "edad": 50, "riesgo_propio": 0.05, "historial": "VIP (Nunca falla)"}
}

# --- ESTADOS DE SESIÓN ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'paciente_actual' not in st.session_state:
    st.session_state.paciente_actual = None
if 'cita_confirmada' not in st.session_state:
    st.session_state.cita_confirmada = None

# --- GENERADOR DE AGENDA ---
@st.cache_data
def generar_mes_simulado():
    np.random.seed(42)
    agenda = []
    hoy = datetime.now()
    dias_generados = 0
    delta = 1
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

    while dias_generados < 5:
        fecha_eval = hoy + timedelta(days=delta)
        delta += 1
        if fecha_eval.weekday() >= 5:
            continue

        fecha_str = f"{dias_semana[fecha_eval.weekday()]} {fecha_eval.strftime('%d/%m')}"

        for hora in [9, 10, 11, 12]:
            for minuto in [0, 15, 30, 45]:
                estado = np.random.choice(["Libre", "Ocupado"], p=[0.2, 0.8])
                riesgo_titular = np.random.uniform(0.10, 0.90) if estado == "Ocupado" else 0.0
                agenda.append({
                    "fecha": fecha_str,
                    "hora_str": f"{hora:02d}:{minuto:02d}",
                    "estado_base": estado,
                    "riesgo_titular": riesgo_titular
                })
        dias_generados += 1
    return pd.DataFrame(agenda)

agenda_df = generar_mes_simulado()

# --- LAYOUT: 70% PACIENTE (IZQUIERDA, BLANCO) / 30% ADMIN (DERECHA, GRIS) ---
col_app, col_admin = st.columns([7, 3], gap="large")

# ==========================================
# ⚙️ DERECHA: PANEL ADMIN (GRIS OSCURO)
# ==========================================
with col_admin:
    with st.container(key="admin_panel"):
        st.markdown('<span class="admin-eyebrow">Supervisor</span>', unsafe_allow_html=True)
        st.markdown("### ⚙️ E&M Control IA")
        st.caption("Dashboard de Supervisor")
        st.divider()

        umbral_ia = st.slider(
            "Umbral de Overbooking",
            min_value=0.10, max_value=0.90, value=0.60, step=0.05,
            help="Si el titular o el paciente actual superan este riesgo, la IA habilita el slot."
        )

        st.divider()
        st.markdown("#### 📋 Base de Datos Demo")
        st.code("""ID: 111 (Mikel  - Riesgo: 10%)
ID: 222 (Ane    - Riesgo: 85%)
ID: 333 (Jon    - Riesgo: 35%)
ID: 444 (Maite  - Riesgo: 65%)
ID: 555 (Aitor  - Riesgo: 5%)""")
        st.info("💡 Cambia el umbral y observa cómo la IA abre o cierra huecos en tiempo real.")

# ==========================================
# 📱 IZQUIERDA: APP PACIENTE (BLANCO)
# ==========================================
with col_app:
    with st.container(key="patient_panel"):
        col_logo, col_titulo = st.columns([1, 8])
        col_logo.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=52)
        with col_titulo:
            st.markdown('<span class="patient-eyebrow">Portal del paciente</span>', unsafe_allow_html=True)
            st.title("Portal OsasunFlow")

        if not st.session_state.logged_in:
            st.markdown("##### Acceso de Pacientes")
            with st.form("login_form"):
                paciente_id = st.text_input("Introduzca su Nº de Tarjeta Sanitaria (TIS)", placeholder="Ej: 111, 222, 333...")
                submit = st.form_submit_button("Escanear TIS y Acceder", type="primary", use_container_width=True)

                if submit:
                    if paciente_id in PACIENTES_DB:
                        st.session_state.logged_in = True
                        st.session_state.paciente_actual = PACIENTES_DB[paciente_id]
                        st.rerun()
                    else:
                        st.error("❌ Paciente no encontrado en la base de datos.")

        elif st.session_state.logged_in and not st.session_state.cita_confirmada:
            paciente = st.session_state.paciente_actual
            riesgo_paciente = paciente['riesgo_propio']

            c1, c2 = st.columns([4, 1])
            c1.subheader(f"👤 Bienvenido/a, {paciente['nombre']}")
            c1.markdown(f"**Perfil:** {paciente['historial']} &nbsp;·&nbsp; **Riesgo IA:** {riesgo_paciente*100:.0f}%")

            if c2.button("Cerrar Sesión", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.paciente_actual = None
                st.rerun()

            st.divider()
            st.markdown("#### Seleccione una fecha para su cita")

            dias_unicos = agenda_df['fecha'].unique()
            SLOTS_POR_FILA = 4  # 4 columnas -> cuadrícula ordenada por hora

            for dia in dias_unicos:
                with st.expander(f"📅 {dia}", expanded=(dia == dias_unicos[0])):
                    df_dia = agenda_df[agenda_df['fecha'] == dia].reset_index(drop=True)

                    # Una fila de columnas NUEVA por cada grupo de 4 horas:
                    # así las celdas de una misma fila son hermanas en el
                    # mismo contenedor flex y se alinean de verdad.
                    for fila_inicio in range(0, len(df_dia), SLOTS_POR_FILA):
                        fila_slots = df_dia.iloc[fila_inicio:fila_inicio + SLOTS_POR_FILA]
                        row_cols = st.columns(SLOTS_POR_FILA)

                        for col_idx, (_, slot) in enumerate(fila_slots.iterrows()):
                            hora = slot['hora_str']
                            estado = slot['estado_base']
                            riesgo_titular = slot['riesgo_titular']
                            condicion_overbooking = (riesgo_titular >= umbral_ia) or (riesgo_paciente >= umbral_ia)

                            with row_cols[col_idx]:
                                if estado == "Libre":
                                    st.markdown(f'<div class="slot-card"><b>{hora}</b><br>✅ Libre</div>', unsafe_allow_html=True)
                                    if st.button("Reservar", key=f"btn_{dia}_{hora}", use_container_width=True):
                                        st.session_state.cita_confirmada = f"{dia} a las {hora}"
                                        st.rerun()

                                elif estado == "Ocupado" and condicion_overbooking:
                                    st.markdown(f'<div class="slot-overbooking"><b>{hora}</b><br>✨ SmartSlot</div>', unsafe_allow_html=True)
                                    if st.button("Reservar ", key=f"ob_{dia}_{hora}", use_container_width=True):
                                        st.session_state.cita_confirmada = f"{dia} a las {hora} (Optimizada por IA)"
                                        st.rerun()

                                else:  # Ocupado, sin overbooking
                                    st.markdown(f'<div class="slot-full"><b>{hora}</b><br>❌ Ocupado</div>', unsafe_allow_html=True)
                                    st.button("No disponible", key=f"na_{dia}_{hora}", use_container_width=True, disabled=True)

        elif st.session_state.cita_confirmada:
            st.success("🎉 Cita agendada correctamente.")
            st.markdown(f"""
            ### 🎫 Resumen de Cita
            * **Paciente:** {st.session_state.paciente_actual['nombre']}
            * **Fecha y Hora:** {st.session_state.cita_confirmada}
            """)

            if st.button("Volver al Inicio", type="primary"):
                st.session_state.logged_in = False
                st.session_state.cita_confirmada = None
                st.session_state.paciente_actual = None
                st.rerun()