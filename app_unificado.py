# -*- coding: utf-8 -*-
"""
Portal unificado del paciente.

Combina en una sola app de Streamlit, con dos pestañas, los dos puntos de
entrada que antes vivían por separado:

- chatbot/app.py     -> Asistente conversacional de admisión (intake, ficha
                         clínica, predicción de riesgo de no-show).
- scripts/patient.py -> Portal de reserva de citas (login por TIS, cuadrícula
                         de horarios, overbooking asistido por IA).

No duplica lógica: importa y reutiliza las funciones `render_*_tab()` de
ambos módulos, que fueron refactorizadas para no fijar su propia
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
from scripts.patient import render_agenda_tab

# Config de página única para toda la app (ni chatbot/app.py ni
# scripts/patient.py la fijan cuando se usan como pestañas: ver
# `configure_page()` en cada uno, que solo se llama en su propio main()).
st.set_page_config(
    page_title="Portal del Paciente | Hospital",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 Portal del Paciente")
st.caption("Habla con el asistente virtual o reserva tu próxima cita, todo desde un mismo sitio.")

tab_chat, tab_agenda = st.tabs(["💬 Asistente virtual", "📅 Reservar cita"])

with tab_chat:
    render_chatbot_tab()

with tab_agenda:
    render_agenda_tab()
