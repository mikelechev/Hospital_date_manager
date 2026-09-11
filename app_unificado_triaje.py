# -*- coding: utf-8 -*-
"""
Vista de paciente: triaje previo a la cita.

Punto de entrada dedicado SOLO al triaje (`scripts/triaje.py`), pensado
como el POV del paciente: TIS -> pregunta médica -> resultado -> elegir
día y hora, todo en una única pantalla, sin pestañas de chat de admisión
ni de Agenda clásica que no pintan nada en el flujo del paciente (esas
son para el personal / la demo técnica, no para quien está reservando
cita). Vive en su PROPIO archivo, separado de `app_unificado.py` (el
portal de dos pestañas chat+agenda, sin triaje), a propósito — cada uno
sirve una demo distinta y un cambio en uno no puede romper el otro.

No duplica lógica: importa y reutiliza `render_triaje_tab()` de
`scripts/triaje.py`, que a su vez reutiliza `scripts/patient.py` para la
cuadrícula de horarios. `scripts/triaje.py` no tiene modo standalone
propio (no tiene sentido fuera de este flujo), así que solo funciona
embebido aquí.

El panel admin/IA (umbral de overbooking SmartSlot, IDs de la demo) se
muestra igual que en el portal de dos pestañas: una columna propia junto a
la del paciente (ver `render_admin_panel_content` en `scripts/patient.py`,
reutilizada por `render_triaje_tab` con su propia container key para no
chocar con la de la Agenda clásica).

Para lanzar esta variante, ejecutar desde la raíz del proyecto:

    streamlit run app_unificado_triaje.py
"""

import streamlit as st

from chatbot.i18n import DEFAULT_LANGUAGE, LANGUAGES, t
from scripts.triaje import render_triaje_tab

# Idioma: propio selector en la barra lateral de esta vista (no depende de
# chatbot/app.py, que ya no se monta aquí). Mismo patrón que el selector
# del chatbot (chatbot/app.py: render_sidebar) para que el comportamiento
# sea idéntico en toda la app: key="language" en session_state, así que
# scripts/patient.py y scripts/triaje.py (que leen ese mismo valor) se
# traducen igual sin ningún cambio adicional.
_lang = st.session_state.get("language", DEFAULT_LANGUAGE)

# Config de página única para toda la app (scripts/triaje.py y
# scripts/patient.py no la fijan cuando se usan embebidos: ver
# `configure_page()` en scripts/patient.py, que solo se llama en su
# propio main()).
st.set_page_config(
    page_title=t("unified_page_title", _lang),
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.segmented_control(
        t("sidebar_language_label", _lang),
        options=list(LANGUAGES.keys()),
        format_func=lambda k: LANGUAGES[k],
        key="language",
        help=t("sidebar_language_help", _lang),
    )
    # El idioma pudo cambiar en este mismo rerun: relee tras el widget para
    # que el resto de la página use ya el idioma recién elegido.
    _lang = st.session_state.get("language", DEFAULT_LANGUAGE)
    st.divider()

render_triaje_tab()
