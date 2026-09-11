# -*- coding: utf-8 -*-
"""
Vista del médico / personal clínico.

Punto de entrada dedicado SOLO a la agenda del médico (`scripts/medico.py`):
qué citas hay hoy y en los próximos días, con quién y con qué prioridad
(nivel de triaje / riesgo de no-show), para que el personal sepa a quién
atender antes sin tener que entrar en el portal del paciente ni en el
chatbot de admisión.

Vive en su PROPIO archivo, separado de `app_unificado.py` (portal del
paciente: chat + triaje + agenda) y de `app_unificado_triaje.py` (POV
paciente, solo triaje), a propósito: es otro rol por completo (personal
clínico, no paciente), y así un cambio en cualquiera de los otros dos no
puede romper esta vista ni al revés.

Importante — lee la nota en la cabecera de scripts/medico.py: al ser un
proceso de Streamlit distinto, esta vista NO comparte `st.session_state`
con los portales del paciente; ve las citas reservadas allí a través de un
fichero compartido en disco (`data/citas_confirmadas_demo.json`), no en
tiempo real — hay que pulsar "Actualizar" tras una reserva nueva.

Para lanzar esta variante, ejecutar desde la raíz del proyecto:

    streamlit run app_unificado_medico.py
"""

import streamlit as st

from chatbot.i18n import DEFAULT_LANGUAGE, LANGUAGES, t
from scripts.medico import render_medico_tab

# Idioma: propio selector en la barra lateral de esta vista, igual que
# app_unificado_triaje.py — cada proceso de Streamlit tiene su propio
# session_state, así que no puede depender del selector de otro proceso.
_lang = st.session_state.get("language", DEFAULT_LANGUAGE)

st.set_page_config(
    page_title=t("unified_page_title", _lang),
    page_icon="🩺",
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
    _lang = st.session_state.get("language", DEFAULT_LANGUAGE)
    st.divider()

render_medico_tab()
