# -*- coding: utf-8 -*-
"""
Triaje previo a la cita — E&M HealthTech / OsasunFlow.

Conecta el asistente conversacional con la reserva de citas: en vez de
dejar que cualquiera reserve cualquier hueco libre sin más, aquí se le
pregunta al paciente cómo se encuentra y, según la urgencia detectada, se
le orienta a:

  - Urgencia real (síntomas de alarma)  -> no se ofrece cita ninguna; se
    deriva a 112/urgencias.
  - Prioritario (la IA valora que conviene verle antes, sin ser una
    emergencia) -> en la pestaña de Agenda solo puede reservar el día más
    próximo disponible.
  - Normal -> la Agenda funciona exactamente igual que hasta ahora.

Vive en un archivo propio, separado de `chatbot/app.py` y
`scripts/patient.py`, a propósito: así esta función nueva no puede romper
por accidente nada de lo que ya funcionaba en esos dos. El único punto de
contacto con `scripts/patient.py` es de solo lectura — `render_agenda_tab()`
comprueba `st.session_state.get("triaje_nivel")` para decidir qué días
mostrar como reservables — y de solo lectura también hacia `PACIENTES_DB`,
para poder saltarse las preguntas con los 5 TIS de la demo.

Limitación consciente para esta demo: el nivel de triaje se guarda en una
única variable de sesión global (no por paciente), igual que el resto del
prototipo no distingue sesiones concurrentes de verdad. Si mañana se añade
persistencia real (ver la propuesta de mejoras), este es uno de los sitios
a revisar.
"""

import json
import logging

import streamlit as st

from chatbot.i18n import DEFAULT_LANGUAGE, LLM_LANGUAGE_INSTRUCTIONS, t
from chatbot.providers.provider_factory import ProviderFactory
from scripts.patient import PACIENTES_DB

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Niveles de triaje. Los usa también scripts/patient.py (render_agenda_tab)
# para decidir qué días de la agenda se pueden reservar — no cambiar estos
# valores sin actualizar también esa comprobación.
# --------------------------------------------------------------------------- #
NIVEL_URGENTE = "urgente"
NIVEL_PRIORITARIO = "prioritario"
NIVEL_NORMAL = "normal"

# --------------------------------------------------------------------------- #
# Síntomas de alarma: lista fija y auditable, en los dos idiomas. Si el
# texto del paciente contiene cualquiera de estas frases, el nivel es
# "urgente" SIEMPRE, sin pasar por el LLM — ninguna llamada a un proveedor
# externo decide si alguien va a urgencias o no, y esta comprobación nunca
# depende de que haya conexión o de que el proveedor elegido esté
# disponible. Se comprueban las frases en los dos idiomas a la vez,
# independientemente del idioma de la interfaz: un paciente puede escribir
# en el idioma que le salga.
# --------------------------------------------------------------------------- #
ALARM_PHRASES = {
    "es": [
        "dolor en el pecho", "dolor torácico", "dolor de pecho", "opresión en el pecho",
        "no puedo respirar", "dificultad para respirar", "me ahogo", "me falta el aire",
        "pérdida de conciencia", "me desmayé", "me desmayo", "perdí el conocimiento",
        "sangrado abundante", "sangro mucho", "hemorragia",
        "un lado del cuerpo", "no puedo mover", "se me traba la lengua", "hablar raro",
        "dolor insoportable", "quiero morir", "hacerme daño",
    ],
    "eu": [
        "bularreko mina", "bularrean estutasuna", "arnasa hartzeko zailtasuna", "ezin dut arnasa hartu",
        "konortea galdu", "zorabiatu naiz", "konortea galdu dut",
        "odolustea", "asko odoltzen", "hemorragia",
        "gorputzaren alde bat", "ezin dut mugitu", "hitz egiteko zailtasuna",
        "min jasanezina",
    ],
}


def _detectar_alarma(texto: str) -> bool:
    """Compara el texto contra la lista de alarma en los dos idiomas a la
    vez. Es deliberadamente una simple búsqueda de subcadenas (no un LLM):
    predecible, auditable y sin dependencia de red."""
    texto_low = texto.lower()
    todas_las_frases = ALARM_PHRASES["es"] + ALARM_PHRASES["eu"]
    return any(frase in texto_low for frase in todas_las_frases)


SEVERITY_PROMPT_TEMPLATE = """
Eres un asistente de clasificación de urgencia (triaje) en un entorno
hospitalario. NO diagnostiques, NO recetes y NO minimices los síntomas que
te cuenten. Tu única tarea es, a partir de la conversación con el
paciente, decidir si su situación es "prioritario" (conviene verle antes
de lo normal, aunque no sea una emergencia) o "normal" (puede esperar el
turno habitual).

IDIOMA: {language_instruction}

REGLAS:
1. Haz UNA sola pregunta por turno para entender la edad, si tiene alguna
   condición relevante (hipertensión, diabetes u otra) y, sobre todo, cómo
   se encuentra y qué síntomas tiene.
2. Nunca digas que algo no es grave ni des consejo médico. Si tienes
   dudas sobre el nivel, marca "prioritario", nunca "normal".
3. En cuanto tengas edad, condición relevante (o su ausencia clara) y una
   descripción de los síntomas, da la conversación por terminada
   ("listo": true) y asigna "nivel".

FORMATO DE RESPUESTA (JSON OBLIGATORIO, nada de texto fuera del JSON):
{{
  "assistant_response": "tu respuesta natural y empática para el paciente",
  "listo": true o false,
  "nivel": "prioritario" o "normal" o null (null mientras "listo" sea false)
}}
"""


def _clasificar_con_llm(historial: list, lang: str) -> dict:
    """Una única llamada al LLM ya configurado en la barra lateral del
    asistente (Ollama/Gemini/OpenAI/Claude/Groq) para seguir la
    conversación o, cuando ya hay datos suficientes, decidir el nivel.

    Si la llamada falla por cualquier motivo (proveedor no configurado,
    sin conexión, respuesta no parseable...) se asume "prioritario" en vez
    de "normal": ante la duda, es preferible ofrecer un hueco antes de lo
    necesario que retrasar a alguien que sí lo necesitaba.
    """
    try:
        provider = ProviderFactory.get_provider()
        system_content = SEVERITY_PROMPT_TEMPLATE.format(
            language_instruction=LLM_LANGUAGE_INSTRUCTIONS.get(
                lang, LLM_LANGUAGE_INSTRUCTIONS[DEFAULT_LANGUAGE]
            ),
        )
        messages = [{"role": "system", "content": system_content}] + historial
        raw = provider.generate_response(messages)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned[:4].lower() == "json":
                cleaned = cleaned[4:]
        data = json.loads(cleaned)

        if not data.get("assistant_response"):
            raise ValueError("Respuesta del LLM sin 'assistant_response'.")
        return data

    except Exception:
        logger.exception(
            "Fallo clasificando urgencia con el LLM; se asume nivel "
            "'prioritario' por seguridad (mejor ofrecer hueco de sobra "
            "que retrasar a quien lo necesitaba)."
        )
        return {
            "assistant_response": t("triaje_llm_error_fallback", lang),
            "listo": True,
            "nivel": NIVEL_PRIORITARIO,
        }


def _init_triaje_state() -> None:
    if "triaje_paso" not in st.session_state:
        st.session_state.triaje_paso = "login"  # login -> chat -> resultado
    if "triaje_historial" not in st.session_state:
        st.session_state.triaje_historial = []
    if "triaje_nivel" not in st.session_state:
        st.session_state.triaje_nivel = None


def render_triaje_tab() -> None:
    """Renders the pre-booking triage tab. Does NOT call configure_page()
    — same convention as render_chatbot_tab()/render_agenda_tab()."""
    lang = st.session_state.get("language", DEFAULT_LANGUAGE)
    _init_triaje_state()

    st.markdown(f'<span class="patient-eyebrow">{t("triaje_eyebrow", lang)}</span>', unsafe_allow_html=True)
    st.title(t("triaje_title", lang))
    st.caption(t("triaje_caption", lang))
    st.divider()

    if st.session_state.triaje_paso == "login":
        with st.form("triaje_login_form"):
            tis = st.text_input(t("agenda_tis_label", lang), placeholder=t("agenda_tis_placeholder", lang))
            submit = st.form_submit_button(t("triaje_start_button", lang), type="primary", use_container_width=True)

            if submit:
                if tis in PACIENTES_DB:
                    # Paciente predefinido: nivel ya fijado a mano por
                    # persona (ver PACIENTES_DB en scripts/patient.py). Se
                    # salta la conversación y la llamada al LLM del todo —
                    # camino 100% determinista, para que la demo con estos
                    # 5 TIS nunca dependa de la disponibilidad del
                    # proveedor elegido.
                    st.session_state.triaje_nivel = PACIENTES_DB[tis].get("urgencia_demo", NIVEL_NORMAL)
                    st.session_state.triaje_paso = "resultado"
                else:
                    st.session_state.triaje_paso = "chat"
                    st.session_state.triaje_historial = []
                st.rerun()

    elif st.session_state.triaje_paso == "chat":
        if not st.session_state.triaje_historial:
            # Primer turno: pregunta inicial fija, sin gastar una llamada
            # al LLM solo para arrancar la conversación.
            st.session_state.triaje_historial.append(
                {"role": "assistant", "content": t("triaje_first_question", lang)}
            )

        for msg in st.session_state.triaje_historial:
            avatar = "🩺" if msg["role"] == "assistant" else "🙋"
            with st.chat_message(msg["role"], avatar=avatar):
                st.write(msg["content"])

        # key explícita: al vivir en su propia pestaña dentro de
        # app_unificado.py, este chat_input coexiste con el de
        # chatbot/app.py (Streamlit ejecuta el cuerpo de TODAS las
        # pestañas en cada rerun, no solo la visible) — con una key
        # distinta no hay ambigüedad posible entre los dos.
        user_text = st.chat_input(t("triaje_chat_placeholder", lang), key="triaje_chat_input")
        if user_text and user_text.strip():
            st.session_state.triaje_historial.append({"role": "user", "content": user_text})

            if _detectar_alarma(user_text):
                # Corte de seguridad: nunca pasa por el LLM.
                st.session_state.triaje_nivel = NIVEL_URGENTE
                st.session_state.triaje_historial.append(
                    {"role": "assistant", "content": t("triaje_alarma_respuesta", lang)}
                )
                st.session_state.triaje_paso = "resultado"
                st.rerun()

            with st.spinner(t("chat_thinking_spinner", lang)):
                resultado = _clasificar_con_llm(st.session_state.triaje_historial, lang)

            st.session_state.triaje_historial.append(
                {"role": "assistant", "content": resultado.get("assistant_response", "")}
            )
            if resultado.get("listo"):
                st.session_state.triaje_nivel = resultado.get("nivel") or NIVEL_PRIORITARIO
                st.session_state.triaje_paso = "resultado"
            st.rerun()

    elif st.session_state.triaje_paso == "resultado":
        nivel = st.session_state.triaje_nivel or NIVEL_NORMAL
        if nivel == NIVEL_URGENTE:
            st.error(t("triaje_resultado_urgente", lang))
        elif nivel == NIVEL_PRIORITARIO:
            st.warning(t("triaje_resultado_prioritario", lang))
            st.caption(t("triaje_ir_a_agenda", lang))
        else:
            st.success(t("triaje_resultado_normal", lang))
            st.caption(t("triaje_ir_a_agenda", lang))

        if st.button(t("triaje_reiniciar_button", lang)):
            for key in ("triaje_paso", "triaje_historial", "triaje_nivel"):
                st.session_state.pop(key, None)
            st.rerun()
