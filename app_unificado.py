# -*- coding: utf-8 -*-
"""
Portal unificado del paciente.

Combina en una sola app de Streamlit, con tres pestañas, los puntos de
entrada que antes vivían por separado, más el triaje previo a la cita:

- chatbot/app.py     -> Asistente conversacional de admisión (intake, ficha
                         clínica, predicción de riesgo de no-show).
- scripts/triaje.py  -> Triaje previo a la cita: pregunta cómo se encuentra
                         el paciente y, según la urgencia, condiciona qué
                         días puede reservar en la pestaña de Agenda.
- scripts/patient.py -> Portal de reserva de citas (login por TIS, cuadrícula
                         de horarios, overbooking asistido por IA).

No duplica lógica: importa y reutiliza las funciones `render_*_tab()` de
los tres módulos, que fueron refactorizadas para no fijar su propia
configuración de página ni ejecutarse como efecto secundario de la
importación (ver el comentario al inicio de cada archivo). Cada script
sigue funcionando exactamente igual que antes si se ejecuta por separado:

    streamlit run chatbot/app.py
    streamlit run scripts/patient.py

Para lanzar el portal unificado, ejecutar desde la raíz del proyecto:

    streamlit run app_unificado.py

o bien:

    ./run.sh portal
"""

import streamlit as st

from chatbot.app import render_chatbot_tab
from chatbot.i18n import DEFAULT_LANGUAGE, t
from scripts.patient import render_agenda_tab
from scripts.triaje import render_triaje_tab

# Idioma: mismo st.session_state.language que fija el selector de la barra
# lateral del chatbot (chatbot/app.py). En la primerísima carga de una
# sesión nueva ese selector aún no se ha renderizado, así que cae al
# castellano por defecto hasta que el usuario elija euskera.
_lang = st.session_state.get("language", DEFAULT_LANGUAGE)

# Config de página única para toda la app (ni chatbot/app.py ni
# scripts/patient.py la fijan cuando se usan como pestañas: ver
# `configure_page()` en cada uno, que solo se llama en su propio main()).
st.set_page_config(
    page_title=t("unified_page_title", _lang),
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(t("unified_title", _lang))
st.caption(t("unified_caption", _lang))

tab_chat, tab_triaje, tab_agenda = st.tabs([
    t("unified_tab_chat", _lang),
    t("unified_tab_triaje", _lang),
    t("unified_tab_agenda", _lang),
])

with tab_chat:
    render_chatbot_tab()

with tab_triaje:
    # mostrar_control_umbral=False: tab_agenda (más abajo) ya muestra el
    # slider interactivo del umbral en su propio panel admin, montado en
    # el mismo script run que este — dos sliders con la misma key
    # explícita en la misma ejecución rompería la app (ver la nota en
    # render_admin_panel_content, scripts/patient.py).
    render_triaje_tab(mostrar_control_umbral=False)

with tab_agenda:
    render_agenda_tab()
