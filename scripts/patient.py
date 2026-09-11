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
import joblib
from pathlib import Path
from datetime import datetime, timedelta

from chatbot.i18n import DEFAULT_LANGUAGE, t

# Idioma actual del portal: comparte st.session_state.language con el
# chatbot (chatbot/app.py). Cuando este script corre embebido en
# app_unificado.py, el selector de idioma vive en la barra lateral del
# chatbot y afecta también a esta pestaña; en ejecución independiente
# (streamlit run scripts/patient.py) no hay selector visible aquí, así que
# simplemente cae al valor por defecto (castellano).
def _current_language() -> str:
    return st.session_state.get("language", DEFAULT_LANGUAGE)

# --------------------------------------------------------------------------- #
# Riesgo real del "titular" en la agenda simulada
#
# Antes: `riesgo_titular = np.random.uniform(0.10, 0.90)` — un número
# inventado, sin relación con ningún dato ni modelo, pese a que la tarjeta
# resultante se presenta como "✨ SmartSlot" (overbooking decidido por IA).
# Ahora: se toma un paciente real de `data/dataset_limpio.csv` y se calcula
# su probabilidad de no-show con el mismo `models/modelo_definitivo.joblib`
# (VotingClassifier XGBoost+CatBoost calibrado) que ya usa de verdad
# `scripts/app.py` — mismas rutas y mismo listado de 19 features, para no
# duplicar dos versiones que puedan desincronizarse.
#
# Se mantiene un fallback a un riesgo aleatorio si el modelo o el dataset no
# están disponibles (equipo sin el .joblib copiado, csv movido, etc.): igual
# que el chatbot ya hace con su propio `is_fallback`, esto nunca debe romper
# la demo, solo perder precisión en ese caso.
_ROOT_DIR = Path(__file__).resolve().parents[1]
_RISK_MODEL_PATH = _ROOT_DIR / "models" / "modelo_definitivo.joblib"
_RISK_DATASET_PATH = _ROOT_DIR / "data" / "dataset_limpio.csv"
_RISK_FEATURES = [
    'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism',
    'Handcap', 'SMS_received', 'Days_between', 'Appointment_Day_of_Week',
    'Scheduled_Day_of_Week', 'Weekend', 'Appointment_Month',
    'Scheduled_Month', 'Faltas_Previas', 'Citas_Previas', 'Ratio_Faltas',
    'Gender_M', 'Scheduled_Time_of_Day_Evening', 'Scheduled_Time_of_Day_Morning',
]


@st.cache_resource(show_spinner=False)
def _load_risk_model():
    try:
        return joblib.load(_RISK_MODEL_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_risk_dataset():
    try:
        return pd.read_csv(_RISK_DATASET_PATH)
    except Exception:
        return None


def _riesgo_titular_real(modelo, dataset) -> float:
    """Probabilidad de no-show de un paciente histórico real (predict_proba
    del modelo entrenado). Si el modelo o el dataset no cargaron, cae a un
    riesgo aleatorio en el mismo rango que se usaba antes, para que la
    agenda simulada siga funcionando sin romperse."""
    if modelo is not None and dataset is not None:
        try:
            fila = dataset.sample(n=1)[_RISK_FEATURES]
            return float(modelo.predict_proba(fila)[0, 1])
        except Exception:
            pass
    return float(np.random.uniform(0.10, 0.90))


def _construir_features_nuevo_paciente(
    edad: int,
    genero_m: bool,
    hipertension: bool,
    diabetes: bool,
    alcoholismo: bool,
    discapacidad: bool,
    beca: bool,
) -> pd.DataFrame:
    """Construye la fila de las 19 features que espera
    `models/modelo_definitivo.joblib` para un paciente NUEVO (no está en
    PACIENTES_DB ni en el dataset histórico), a partir de las respuestas de
    un formulario corto (scripts/triaje.py: _render_riesgo_form_paso) —
    "las míticas preguntas" que hacía el chatbot de admisión clásico para
    poder calcular una probabilidad de no-show real, en vez de asumir un
    0.0 fijo para todo paciente sin historial.

    Dos grupos de columnas:
      - A nivel de PERSONA, se preguntan directamente: Age, Gender_M,
        Hipertension, Diabetes, Alcoholism, Handcap, Scholarship.
      - A nivel de CONTEXTO DE LA CITA (Days_between,
        Appointment/Scheduled_Day_of_Week, Weekend, Appointment/
        Scheduled_Month, Scheduled_Time_of_Day_*), que en este punto del
        flujo aún no se conocen (el paciente todavía no ha elegido día ni
        hora), se usan valores por defecto razonables: se asume que pide
        cita para "mañana" desde "ahora mismo", con SMS de recordatorio y
        sin historial previo de citas/faltas. Esta es la misma
        simplificación que ya existe para los 5 pacientes de la demo
        (PACIENTES_DB): su `riesgo_propio` es un único valor fijo,
        independientemente del día/hueco que acaben eligiendo.
    """
    ahora = datetime.now()
    manana = ahora + timedelta(days=1)
    hora_actual = ahora.hour

    fila = {
        "Age": edad,
        "Scholarship": int(beca),
        "Hipertension": int(hipertension),
        "Diabetes": int(diabetes),
        "Alcoholism": int(alcoholismo),
        "Handcap": int(discapacidad),
        "SMS_received": 1,
        "Days_between": 1,
        "Appointment_Day_of_Week": manana.weekday(),
        "Scheduled_Day_of_Week": ahora.weekday(),
        "Weekend": int(manana.weekday() >= 5),
        "Appointment_Month": manana.month,
        "Scheduled_Month": ahora.month,
        "Faltas_Previas": 0,
        "Citas_Previas": 0,
        "Ratio_Faltas": 0.0,
        "Gender_M": int(genero_m),
        "Scheduled_Time_of_Day_Evening": int(hora_actual >= 18),
        "Scheduled_Time_of_Day_Morning": int(hora_actual < 14),
    }
    return pd.DataFrame([fila])[_RISK_FEATURES]


def calcular_riesgo_nuevo_paciente(
    edad: int,
    genero_m: bool,
    hipertension: bool,
    diabetes: bool,
    alcoholismo: bool,
    discapacidad: bool,
    beca: bool,
) -> float:
    """Probabilidad de no-show real (predict_proba del modelo entrenado)
    para un paciente nuevo, a partir de las respuestas de su formulario de
    riesgo. Mismo fallback que `_riesgo_titular_real` si el modelo no está
    disponible: un riesgo aleatorio en el mismo rango, para que la demo
    nunca se rompa por esto, solo pierda precisión."""
    modelo = _load_risk_model()
    if modelo is not None:
        try:
            fila = _construir_features_nuevo_paciente(
                edad, genero_m, hipertension, diabetes, alcoholismo, discapacidad, beca,
            )
            return float(modelo.predict_proba(fila)[0, 1])
        except Exception:
            pass
    return float(np.random.uniform(0.10, 0.90))

# --------------------------------------------------------------------------- #
# Page config
#
# Extracted into a function (instead of executed at import time) so this
# module can be reused as one tab of the combined portal
# (see app_unificado.py at the project root) without forcing its own page
# config. Standalone execution (`streamlit run scripts/patient.py`) is
# unaffected: main() below still calls configure_page() once, exactly like
# before.
# --------------------------------------------------------------------------- #

def configure_page() -> None:
    """Sets the Streamlit page config. Call at most once per app run, before
    any other Streamlit command. Only the script that owns the process
    should call it (this file in standalone mode, or app_unificado.py when
    this tab is embedded in the combined portal)."""
    st.set_page_config(
        page_title=t("agenda_page_title", _current_language()),
        layout="wide",
        initial_sidebar_state="collapsed",
    )

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

def inject_css() -> None:
    """Injects this tab's CSS. Idempotent/cheap — safe to call on every rerun."""
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
    .st-key-patient_panel , .st-key-triaje_panel {{
        background-color: {COLOR_PATIENT_BG};
        border: 1px solid {COLOR_PATIENT_BORDER};
        border-radius: 16px;
        padding: 2.25rem 2.5rem;
        box-shadow: 0px 12px 32px rgba(17, 24, 39, 0.06);
    }}
    .st-key-patient_panel p, .st-key-triaje_panel p,
    .st-key-patient_panel span, .st-key-triaje_panel span,
    .st-key-patient_panel label , .st-key-triaje_panel label {{
        color: {COLOR_PATIENT_TEXT};
    }}
    .st-key-patient_panel h1, .st-key-triaje_panel h1,
    .st-key-patient_panel h2, .st-key-triaje_panel h2,
    .st-key-patient_panel h3 , .st-key-triaje_panel h3 {{
        color: {COLOR_PATIENT_TEXT};
        letter-spacing: -0.01em;
    }}
    .st-key-patient_panel [data-testid="stCaptionContainer"] , .st-key-triaje_panel [data-testid="stCaptionContainer"] {{
        color: {COLOR_PATIENT_MUTED};
    }}
    .st-key-patient_panel hr , .st-key-triaje_panel hr {{
        border-color: {COLOR_PATIENT_BORDER};
    }}

    /* Botones Principales (Acceder) */
    .st-key-patient_panel button[kind="primary"], .st-key-triaje_panel button[kind="primary"],
    .st-key-patient_panel button[kind="primaryFormSubmit"] , .st-key-triaje_panel button[kind="primaryFormSubmit"] {{
        background-color: {COLOR_ACCENT_FREE} !important;
        border: 1px solid {COLOR_ACCENT_FREE} !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        font-weight: 600;
    }}
    .st-key-patient_panel button[kind="primary"]:hover, .st-key-triaje_panel button[kind="primary"]:hover,
    .st-key-patient_panel button[kind="primaryFormSubmit"]:hover , .st-key-triaje_panel button[kind="primaryFormSubmit"]:hover {{
        background-color: #128A3E !important;
        border-color: #128A3E !important;
    }}

    /* Botones Secundarios (Reservar, Cerrar Sesión) */
    .st-key-patient_panel button[kind="secondary"] , .st-key-triaje_panel button[kind="secondary"] {{
        background-color: #FFFFFF !important;
        border-radius: 8px;
        border: 1px solid #D1D5DB !important;
        color: {COLOR_PATIENT_TEXT} !important;
        font-weight: 600;
    }}
    .st-key-patient_panel button[kind="secondary"]:hover , .st-key-triaje_panel button[kind="secondary"]:hover {{
        background-color: #F3F4F6 !important;
        border-color: #9CA3AF !important;
        color: #111827 !important;
    }}

    /* Botón deshabilitado "No disponible": mismo tamaño, look apagado,
       para que las tarjetas ocupadas midan igual que las reservables */
    .st-key-patient_panel button:disabled, .st-key-triaje_panel button:disabled,
    .st-key-patient_panel button[kind="secondary"]:disabled , .st-key-triaje_panel button[kind="secondary"]:disabled {{
        background-color: #F9FAFB !important;
        border: 1px dashed #E1E4E8 !important;
        color: #C1C6CD !important;
        font-weight: 500;
        opacity: 1 !important;
    }}

    /* ---------- PANEL ADMIN (derecha, gris oscuro) ---------- */
    .st-key-admin_panel , .st-key-admin_panel_triaje {{
        background-color: {COLOR_ADMIN_BG};
        border: 1px solid {COLOR_ADMIN_BORDER};
        border-radius: 16px;
        padding: 2.25rem 2rem;
        box-shadow: inset 0px 0px 0px 1px rgba(255,255,255,0.02),
                    0px 12px 32px rgba(0,0,0,0.25);
    }}
    .st-key-admin_panel p, .st-key-admin_panel_triaje p,
    .st-key-admin_panel span, .st-key-admin_panel_triaje span,
    .st-key-admin_panel label , .st-key-admin_panel_triaje label {{
        color: {COLOR_ADMIN_TEXT};
    }}
    .st-key-admin_panel h1, .st-key-admin_panel_triaje h1,
    .st-key-admin_panel h2, .st-key-admin_panel_triaje h2,
    .st-key-admin_panel h3 , .st-key-admin_panel_triaje h3 {{
        color: {COLOR_ADMIN_TEXT};
        letter-spacing: -0.01em;
    }}
    .st-key-admin_panel [data-testid="stCaptionContainer"] , .st-key-admin_panel_triaje [data-testid="stCaptionContainer"] {{
        color: {COLOR_ADMIN_MUTED};
    }}
    .st-key-admin_panel hr , .st-key-admin_panel_triaje hr {{
        border-color: {COLOR_ADMIN_BORDER};
    }}
    .st-key-admin_panel [data-testid="stCodeBlock"] pre , .st-key-admin_panel_triaje [data-testid="stCodeBlock"] pre {{
        background-color: {COLOR_ADMIN_BG_ALT} !important;
        border: 1px solid {COLOR_ADMIN_BORDER};
        border-radius: 10px;
    }}
    .st-key-admin_panel [data-testid="stCodeBlock"] code , .st-key-admin_panel_triaje [data-testid="stCodeBlock"] code {{
        color: {COLOR_ADMIN_TEXT} !important;
    }}
    .st-key-admin_panel [data-testid="stAlertContainer"] , .st-key-admin_panel_triaje [data-testid="stAlertContainer"] {{
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

    /* ---------- "Cómo funciona" (3 pasos, pantalla de login del triaje) ---------- */
    .triaje-paso {{
        text-align: center;
        padding: 0.5rem 0.75rem 1rem 0.75rem;
    }}
    .triaje-paso-icono {{
        font-size: 1.9rem;
        margin-bottom: 0.35rem;
    }}
    .triaje-paso-titulo {{
        font-weight: 700;
        font-size: 0.92rem;
        color: {COLOR_PATIENT_TEXT};
        margin-bottom: 0.2rem;
    }}
    .triaje-paso-desc {{
        font-size: 0.82rem;
        color: {COLOR_PATIENT_MUTED};
        line-height: 1.3;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- BASE DE DATOS DE PACIENTES AMPLIADA ---
# "historial_key" apunta a una clave de chatbot/i18n.py (hist_*) en vez de a
# texto fijo, para que la descripción del perfil se traduzca según el idioma
# elegido en la barra lateral del chatbot.
#
# Nota sobre el triaje (scripts/triaje.py): estos 5 TIS ya NO se saltan la
# pregunta médica del triaje — todo paciente, conocido o no, responde a esa
# pregunta y la IA decide el nivel a partir de su respuesta. Lo único que se
# salta para un TIS conocido es la pregunta de edad (ya está aquí, en
# "edad"), para no volver a pedir un dato que el sistema ya tiene.
PACIENTES_DB = {
    "111": {"nombre": "Mikel Ezkurdia", "edad": 22, "riesgo_propio": 0.10, "historial_key": "hist_excelente"},
    "222": {"nombre": "Ane Larrañaga", "edad": 65, "riesgo_propio": 0.85, "historial_key": "hist_critico"},
    "333": {"nombre": "Jon Arretxe", "edad": 41, "riesgo_propio": 0.35, "historial_key": "hist_medio"},
    "444": {"nombre": "Maite Zabaleta", "edad": 29, "riesgo_propio": 0.65, "historial_key": "hist_irregular"},
    "555": {"nombre": "Aitor Ocio", "edad": 50, "riesgo_propio": 0.05, "historial_key": "hist_vip"},
}

# Umbral de overbooking (SmartSlot) por defecto y clave de session_state del
# slider del panel admin (ver render_agenda_tab). Se usa como valor inicial
# del slider Y como valor que lee el flujo de reserva integrado en el
# triaje (scripts/triaje.py), que no muestra el panel admin pero debe
# respetar el mismo umbral si ya se ha tocado el slider en esta sesión.
UMBRAL_IA_DEFECTO = 0.60
_UMBRAL_IA_STATE_KEY = "agenda_umbral_ia"


def _get_umbral_ia() -> float:
    return st.session_state.get(_UMBRAL_IA_STATE_KEY, UMBRAL_IA_DEFECTO)


def render_admin_panel_content(lang: str) -> None:
    """Contenido del panel de administración/IA: eyebrow, umbral de
    overbooking (SmartSlot), IDs de la demo e info. Extraído de
    render_agenda_tab a una función propia para poder reutilizarlo tal
    cual en la vista de triaje en solitario (scripts/triaje.py /
    app_unificado_triaje.py), que también necesita mostrar este panel —
    antes se había sustituido ahí por un simple desplegable en la barra
    lateral con solo el slider, pero eso escondía el resto del panel (los
    IDs de la demo, la explicación del umbral) que hacía falta para poder
    enseñar/probar la IA de overbooking en esa vista.

    El LLAMADOR es responsable de envolver esto en su propio
    `st.container(key=...)` con el estilo de `.st-key-admin_panel` (ver
    inject_css) — cada sitio que la usa necesita su propia key distinta
    (Streamlit no permite dos elementos con la misma key en el mismo
    script run): render_agenda_tab usa "admin_panel", la vista de triaje
    usa "admin_panel_triaje". inject_css() ya aplica el mismo estilo a las
    dos.
    """
    st.markdown(f'<span class="admin-eyebrow">{t("agenda_admin_eyebrow", lang)}</span>', unsafe_allow_html=True)
    st.markdown(t("agenda_admin_heading", lang))
    st.caption(t("agenda_admin_caption", lang))
    st.divider()

    st.slider(
        t("agenda_admin_threshold_label", lang),
        min_value=0.10, max_value=0.90, value=UMBRAL_IA_DEFECTO, step=0.05,
        help=t("agenda_admin_threshold_help", lang),
        key=_UMBRAL_IA_STATE_KEY,
    )

    st.divider()
    st.markdown(t("agenda_admin_db_heading", lang))
    # IDs/nombres de la demo se dejan sin traducir a propósito (son
    # identificadores, no texto de interfaz).
    st.code("""ID: 111 (Mikel  - Riesgo: 10%)
ID: 222 (Ane    - Riesgo: 85%)
ID: 333 (Jon    - Riesgo: 35%)
ID: 444 (Maite  - Riesgo: 65%)
ID: 555 (Aitor  - Riesgo: 5%)""")
    st.info(t("agenda_admin_info", lang))


# --- ESTADOS DE SESIÓN ---
def _init_session_state() -> None:
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'paciente_actual' not in st.session_state:
        st.session_state.paciente_actual = None
    if 'cita_confirmada' not in st.session_state:
        st.session_state.cita_confirmada = None

# --- GENERADOR DE AGENDA ---
# `lang` forma parte de la firma para que st.cache_data guarde una entrada de
# caché distinta por idioma (Streamlit incluye los argumentos en la clave de
# caché), así el nombre del día sale ya traducido sin recalcular nada más.
@st.cache_data
def generar_mes_simulado(lang: str = DEFAULT_LANGUAGE):
    np.random.seed(42)
    modelo_riesgo = _load_risk_model()
    dataset_riesgo = _load_risk_dataset()
    agenda = []
    hoy = datetime.now()
    dias_generados = 0
    delta = 1
    dias_semana = [
        t("day_monday", lang), t("day_tuesday", lang), t("day_wednesday", lang),
        t("day_thursday", lang), t("day_friday", lang), t("day_saturday", lang),
        t("day_sunday", lang),
    ]

    while dias_generados < 5:
        fecha_eval = hoy + timedelta(days=delta)
        delta += 1
        if fecha_eval.weekday() >= 5:
            continue

        fecha_str = f"{dias_semana[fecha_eval.weekday()]} {fecha_eval.strftime('%d/%m')}"

        for hora in [9, 10, 11, 12]:
            for minuto in [0, 15, 30, 45]:
                estado = np.random.choice(["Libre", "Ocupado"], p=[0.2, 0.8])
                riesgo_titular = _riesgo_titular_real(modelo_riesgo, dataset_riesgo) if estado == "Ocupado" else 0.0
                agenda.append({
                    "fecha": fecha_str,
                    "hora_str": f"{hora:02d}:{minuto:02d}",
                    "estado_base": estado,
                    "riesgo_titular": riesgo_titular
                })
        dias_generados += 1
    return pd.DataFrame(agenda)

def render_dia_selector(
    agenda_df: pd.DataFrame,
    lang: str,
    umbral_ia: float,
    riesgo_paciente: float,
    nivel_triaje,
    key_prefix: str = "ag",
    mostrar_aviso_nivel: bool = True,
    confirmacion_key: str = "cita_confirmada",
) -> None:
    """Cuadrícula de días/huecos con el gating de triaje aplicado.

    Compartida por dos puntos de entrada: la pestaña de Agenda clásica
    (render_agenda_tab, tras el login por TIS) y el flujo integrado
    triaje -> reserva (scripts/triaje.py), que ya conoce el TIS y el nivel
    de triaje desde su propio login y quiere pasar directamente a elegir
    día sin un segundo login ni cambiar de pestaña.

    `key_prefix` evita colisiones de key entre botones: Streamlit ejecuta
    el cuerpo de TODAS las pestañas en cada rerun (no solo la visible), así
    que si la Agenda clásica y el flujo integrado del triaje están montados
    a la vez (dos pestañas del mismo app_unificado_triaje.py), cada uno
    necesita sus propias keys de botón para el mismo día/hora.

    `mostrar_aviso_nivel=False` omite el aviso de "caso prioritario" que ya
    se le ha mostrado al paciente justo antes de llamar a esta función
    (evita repetir el mismo mensaje dos veces seguidas).

    `confirmacion_key` es la clave de session_state donde se guarda la cita
    reservada. Por el mismo motivo que `key_prefix`, cada llamador usa la
    suya (en vez de compartir siempre "cita_confirmada"): como las dos
    pestañas están montadas a la vez, si compartieran la misma clave, una
    reserva hecha desde el triaje activaría también la pantalla de "cita
    confirmada" de la Agenda clásica, que espera encontrar el perfil
    completo del paciente en `st.session_state.paciente_actual` — un
    paciente nuevo llegado por triaje no tiene esa ficha, y forzarla
    rompería esa pantalla.
    """
    if nivel_triaje == "urgente":
        st.error(t("agenda_triaje_urgente_bloqueo", lang))
        return

    if nivel_triaje == "prioritario" and mostrar_aviso_nivel:
        st.warning(t("agenda_triaje_prioritario_aviso", lang))

    st.markdown(t("agenda_select_date_heading", lang))

    dias_unicos = agenda_df['fecha'].unique()
    SLOTS_POR_FILA = 4  # 4 columnas -> cuadrícula ordenada por hora

    for idx_dia, dia in enumerate(dias_unicos):
        # Nota: con nivel "prioritario" NO se bloquea ningún día — el
        # paciente puede reservar el día que prefiera, igual que con nivel
        # "normal". La única diferencia de "prioritario" es el aviso de
        # arriba recomendándole coger el hueco más próximo; antes esto
        # bloqueaba todos los días salvo el primero, pero eso obligaba al
        # paciente a un único día concreto en vez de dejarle elegir, que es
        # justo lo contrario de lo que se pretendía con "prioritario"
        # (verle ANTES, no forzarle a una fecha fija que puede no encajarle).
        titulo_dia = f"📅 {dia}"

        with st.expander(titulo_dia, expanded=(dia == dias_unicos[0])):
            df_dia = agenda_df[agenda_df['fecha'] == dia].reset_index(drop=True)

            # Una fila de columnas NUEVA por cada grupo de 4 horas: así las
            # celdas de una misma fila son hermanas en el mismo contenedor
            # flex y se alinean de verdad.
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
                            st.markdown(
                                f'<div class="slot-card"><b>{hora}</b><br>{t("agenda_slot_free", lang)}</div>',
                                unsafe_allow_html=True,
                            )
                            if st.button(t("agenda_reserve_button", lang), key=f"{key_prefix}_btn_{dia}_{hora}", use_container_width=True):
                                st.session_state[confirmacion_key] = {"dia": dia, "hora": hora, "ai": False}
                                st.rerun()

                        elif estado == "Ocupado" and condicion_overbooking:
                            st.markdown(
                                f'<div class="slot-overbooking"><b>{hora}</b><br>{t("agenda_slot_smartslot", lang)}</div>',
                                unsafe_allow_html=True,
                            )
                            if st.button(t("agenda_reserve_button", lang), key=f"{key_prefix}_ob_{dia}_{hora}", use_container_width=True):
                                st.session_state[confirmacion_key] = {"dia": dia, "hora": hora, "ai": True}
                                st.rerun()

                        else:  # Ocupado, sin overbooking
                            st.markdown(
                                f'<div class="slot-full"><b>{hora}</b><br>{t("agenda_slot_occupied", lang)}</div>',
                                unsafe_allow_html=True,
                            )
                            st.button(
                                t("agenda_unavailable_button", lang),
                                key=f"{key_prefix}_na_{dia}_{hora}", use_container_width=True, disabled=True,
                            )


def render_agenda_tab() -> None:
    """Renders the full appointment-booking portal: CSS, patient login,
    the smart-overbooking agenda grid and the admin sidebar panel. Does
    NOT call configure_page() — the host script is responsible for page
    config (main() does it below for standalone runs; app_unificado.py
    does it once for the whole portal)."""
    inject_css()
    _init_session_state()

    lang = _current_language()
    agenda_df = generar_mes_simulado(lang)

    # --- LAYOUT: 70% PACIENTE (IZQUIERDA, BLANCO) / 30% ADMIN (DERECHA, GRIS) ---
    col_app, col_admin = st.columns([7, 3], gap="large")

    # ==========================================
    # ⚙️ DERECHA: PANEL ADMIN (GRIS OSCURO)
    # ==========================================
    with col_admin:
        with st.container(key="admin_panel"):
            render_admin_panel_content(lang)
    umbral_ia = _get_umbral_ia()

    # ==========================================
    # 📱 IZQUIERDA: APP PACIENTE (BLANCO)
    # ==========================================
    with col_app:
        with st.container(key="patient_panel"):
            col_logo, col_titulo = st.columns([1, 8])
            col_logo.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=52)
            with col_titulo:
                st.markdown(f'<span class="patient-eyebrow">{t("agenda_portal_eyebrow", lang)}</span>', unsafe_allow_html=True)
                st.title(t("agenda_portal_title", lang))

            if not st.session_state.logged_in:
                st.markdown(t("agenda_login_heading", lang))
                with st.form("login_form"):
                    paciente_id = st.text_input(
                        t("agenda_tis_label", lang), placeholder=t("agenda_tis_placeholder", lang)
                    )
                    submit = st.form_submit_button(
                        t("agenda_login_button", lang), type="primary", use_container_width=True
                    )

                    if submit:
                        if paciente_id in PACIENTES_DB:
                            st.session_state.logged_in = True
                            st.session_state.paciente_actual = PACIENTES_DB[paciente_id]
                            st.rerun()
                        else:
                            st.error(t("agenda_login_error", lang))

            elif st.session_state.logged_in and not st.session_state.cita_confirmada:
                paciente = st.session_state.paciente_actual
                riesgo_paciente = paciente['riesgo_propio']
                historial_texto = t(paciente['historial_key'], lang)

                c1, c2 = st.columns([4, 1])
                c1.subheader(t("agenda_welcome", lang, name=paciente['nombre']))
                c1.markdown(
                    t("agenda_profile_line", lang, historial=historial_texto, risk=f"{riesgo_paciente*100:.0f}")
                )

                if c2.button(t("agenda_logout_button", lang), use_container_width=True):
                    st.session_state.logged_in = False
                    st.session_state.paciente_actual = None
                    st.rerun()

                st.divider()

                # Efecto del triaje previo (scripts/triaje.py) sobre esta
                # agenda. Los valores "urgente"/"prioritario"/"normal" deben
                # coincidir con las constantes NIVEL_* de ese archivo — se
                # repiten aquí en vez de importarlas para no crear un import
                # circular (triaje.py ya importa PACIENTES_DB de este
                # módulo). Limitación consciente: es una única variable de
                # sesión global, no por paciente — ver la nota en triaje.py.
                nivel_triaje = st.session_state.get("triaje_nivel")
                render_dia_selector(
                    agenda_df, lang, umbral_ia, riesgo_paciente, nivel_triaje,
                    key_prefix="ag",
                )

            elif st.session_state.cita_confirmada:
                cita = st.session_state.cita_confirmada
                cuando = f"{cita['dia']} — {cita['hora']}"
                if cita.get("ai"):
                    cuando += t("agenda_summary_ai_suffix", lang)

                st.success(t("agenda_confirmed_success", lang))
                st.markdown(
                    "\n".join([
                        t("agenda_summary_heading", lang),
                        t("agenda_summary_patient", lang, name=st.session_state.paciente_actual['nombre']),
                        t("agenda_summary_datetime", lang, when=cuando),
                    ])
                )

                if st.button(t("agenda_back_button", lang), type="primary"):
                    st.session_state.logged_in = False
                    st.session_state.cita_confirmada = None
                    st.session_state.paciente_actual = None
                    st.rerun()


def main() -> None:
    """Entry point for standalone execution: `streamlit run scripts/patient.py`."""
    configure_page()
    render_agenda_tab()


if __name__ == "__main__":
    main()
