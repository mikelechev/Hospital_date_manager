# -*- coding: utf-8 -*-
"""
Portal unificado del paciente + triaje previo a la cita.

Variante de `app_unificado.py` con una tercera pestaña añadida: el triaje
de `scripts/triaje.py`. Vive en su PROPIO archivo, en vez de dentro de
`app_unificado.py`, a propósito — así el portal original de dos pestañas
(el que ya funcionaba en producción/demo) queda intacto y sin ningún
riesgo de que un fallo de esta función nueva lo arrastre consigo.

Combina tres puntos de entrada:

- chatbot/app.py     -> Asistente conversacional de admisión (intake, ficha
                         clínica, predicción de riesgo de no-show).
- scripts/triaje.py  -> Triaje previo a la cita: TIS, síntomas, y la IA
                         decide si el caso es urgente/prioritario/normal.
- scripts/patient.py -> Portal de reserva de citas (login por TIS, cuadrícula
                         de horarios, overbooking asistido por IA), que lee
                         el resultado del triaje para restringir qué días
                         se pueden reservar.

No duplica lógica: importa y reutiliza las funciones `render_*_tab()` de
los tres módulos. Cada uno sigue funcionando exactamente igual si se
ejecuta por separado:

    streamlit run chatbot/app.py
    streamlit run scripts/patient.py

`scripts/triaje.py` es la excepción: al no tener sentido de forma aislada
(su función es enlazar el chat con la reserva), solo funciona embebido
aquí — no tiene un modo standalone propio.

Para lanzar esta variante, ejecutar desde la raíz del proyecto:

    streamlit run app_unificado_triaje.py
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
# scripts/patient.py ni scripts/triaje.py la fijan cuando se usan como
# pestañas: ver `configure_page()` en cada uno, que solo se llama en su
# propio main() cuando existe).
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
    render_triaje_tab()

with tab_agenda:
    render_agenda_tab()
