# -*- coding: utf-8 -*-
"""
Triaje previo a la cita — E&M HealthTech / OsasunFlow.

Conecta el asistente conversacional con la reserva de citas: en vez de
dejar que cualquiera reserve cualquier hueco libre sin más, aquí se le
pide primero el TIS y se le pregunta al paciente cómo se encuentra y,
según la urgencia detectada, se le deja pasar directamente a elegir
día y hora, ya dentro de esta misma pantalla:

  - Urgencia real (síntomas de alarma)  -> no se ofrece cita ninguna; se
    deriva a 112/urgencias.
  - Prioritario (la IA valora que conviene verle antes, sin ser una
    emergencia) -> solo puede reservar el día más próximo disponible.
  - Normal -> puede reservar cualquier día, igual que en la Agenda clásica.

Con un TIS de los 5 de la demo (ver PACIENTES_DB en scripts/patient.py) se
salta únicamente la pregunta de la edad, porque el sistema ya la conoce —
la pregunta médica (cómo se encuentra, qué síntomas tiene) se hace
SIEMPRE, sea o no un TIS conocido: es la IA, a partir de esa respuesta, la
que decide el nivel en los dos casos. Con un TIS no reconocido se hace la
conversación completa (edad, condición relevante y síntomas) porque el
sistema no tiene esos datos.

Vive en un archivo propio, separado de `chatbot/app.py` y
`scripts/patient.py`, a propósito: así esta función nueva no puede romper
por accidente nada de lo que ya funcionaba en esos dos. El único punto de
contacto con `scripts/patient.py` es la reutilización directa de
`PACIENTES_DB` (solo lectura) y de `render_dia_selector`/`generar_mes_simulado`
para no duplicar la cuadrícula de horarios ni la lógica de overbooking.

Limitación consciente para esta demo: el estado del triaje se guarda en
variables de sesión globales (no por paciente), igual que el resto del
prototipo no distingue sesiones concurrentes de verdad. Si mañana se añade
persistencia real (ver la propuesta de mejoras), este es uno de los sitios
a revisar.
"""

import json
import logging

import streamlit as st

from chatbot.i18n import DEFAULT_LANGUAGE, LLM_LANGUAGE_INSTRUCTIONS, t
from chatbot.providers.provider_factory import ProviderFactory
from scripts.patient import (
    PACIENTES_DB,
    _get_umbral_ia,
    generar_mes_simulado,
    render_dia_selector,
)

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

{contexto_paciente}

REGLAS:
1. Haz UNA sola pregunta por turno para entender lo que falte de: edad, si
   tiene alguna condición relevante (hipertensión, diabetes u otra) y,
   sobre todo, cómo se encuentra y qué síntomas tiene. Si algún dato ya
   aparece más arriba en "DATOS YA CONOCIDOS", no lo vuelvas a preguntar.
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


def _clasificar_con_llm(historial: list, lang: str, edad_conocida=None) -> dict:
    """Una única llamada al LLM ya configurado en la barra lateral del
    asistente (Ollama/Gemini/OpenAI/Claude/Groq) para seguir la
    conversación o, cuando ya hay datos suficientes, decidir el nivel.

    `edad_conocida` (solo para TIS reconocidos en PACIENTES_DB) se pasa
    como contexto para que el LLM no vuelva a preguntar un dato que el
    sistema ya tiene — el llamador, además, fuerza "listo" a True tras la
    primera respuesta en ese caso (ver render_triaje_tab), así que aquí
    solo hace falta evitar la pregunta redundante, no forzar el corte.

    Si la llamada falla por cualquier motivo (proveedor no configurado,
    sin conexión, respuesta no parseable...) se asume "prioritario" en vez
    de "normal": ante la duda, es preferible ofrecer un hueco antes de lo
    necesario que retrasar a alguien que sí lo necesitaba.
    """
    try:
        provider = ProviderFactory.get_provider()
        if edad_conocida is not None:
            contexto_paciente = (
                f"DATOS YA CONOCIDOS: el paciente ya está identificado en el "
                f"sistema, tiene {edad_conocida} años. No le preguntes la "
                f"edad. Pregúntale directamente cómo se encuentra y qué "
                f"síntomas tiene."
            )
        else:
            contexto_paciente = "DATOS YA CONOCIDOS: ninguno todavía."

        system_content = SEVERITY_PROMPT_TEMPLATE.format(
            language_instruction=LLM_LANGUAGE_INSTRUCTIONS.get(
                lang, LLM_LANGUAGE_INSTRUCTIONS[DEFAULT_LANGUAGE]
            ),
            contexto_paciente=contexto_paciente,
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
        st.session_state.triaje_paso = "login"  # login -> chat -> reserva
    if "triaje_historial" not in st.session_state:
        st.session_state.triaje_historial = []
    if "triaje_nivel" not in st.session_state:
        st.session_state.triaje_nivel = None
    if "triaje_tis" not in st.session_state:
        st.session_state.triaje_tis = None
    if "triaje_paciente_conocido" not in st.session_state:
        st.session_state.triaje_paciente_conocido = False


def _reiniciar_triaje() -> None:
    for key in (
        "triaje_paso", "triaje_historial", "triaje_nivel", "triaje_tis",
        "triaje_paciente_conocido", "triaje_cita_confirmada",
    ):
        st.session_state.pop(key, None)


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
                st.session_state.triaje_tis = tis
                st.session_state.triaje_historial = []
                # TIS conocido: se salta únicamente la pregunta de la edad
                # (ya está en PACIENTES_DB) — la pregunta médica se hace
                # igual, para conocidos y desconocidos. Ver la nota en la
                # cabecera del archivo.
                st.session_state.triaje_paciente_conocido = tis in PACIENTES_DB
                st.session_state.triaje_paso = "chat"
                st.rerun()

    elif st.session_state.triaje_paso == "chat":
        tis = st.session_state.triaje_tis
        conocido = st.session_state.triaje_paciente_conocido

        if not st.session_state.triaje_historial:
            # Primer turno: pregunta inicial fija, sin gastar una llamada
            # al LLM solo para arrancar la conversación. Para un TIS
            # conocido, personalizada con su nombre y pidiendo solo el
            # síntoma (la edad ya la tenemos); para uno nuevo, la pregunta
            # genérica de siempre.
            if conocido:
                nombre = PACIENTES_DB[tis]["nombre"]
                primera_pregunta = t("triaje_first_question_conocido", lang, name=nombre)
            else:
                primera_pregunta = t("triaje_first_question", lang)
            st.session_state.triaje_historial.append(
                {"role": "assistant", "content": primera_pregunta}
            )

        for msg in st.session_state.triaje_historial:
            avatar = "🩺" if msg["role"] == "assistant" else "🙋"
            with st.chat_message(msg["role"], avatar=avatar):
                st.write(msg["content"])

        # key explícita: al vivir en su propia pestaña dentro de
        # app_unificado_triaje.py, este chat_input coexiste con el de
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
                st.session_state.triaje_paso = "reserva"
                st.rerun()

            edad_conocida = PACIENTES_DB[tis]["edad"] if conocido else None
            with st.spinner(t("chat_thinking_spinner", lang)):
                resultado = _clasificar_con_llm(st.session_state.triaje_historial, lang, edad_conocida)

            if conocido:
                # Para un TIS conocido solo se hace "la pregunta de la
                # consulta médica": una única respuesta del paciente basta
                # para decidir, así que se fuerza el cierre aquí en vez de
                # confiar en que el LLM marque "listo" por su cuenta.
                resultado["listo"] = True
                resultado.setdefault("nivel", NIVEL_PRIORITARIO)

            st.session_state.triaje_historial.append(
                {"role": "assistant", "content": resultado.get("assistant_response", "")}
            )
            if resultado.get("listo"):
                st.session_state.triaje_nivel = resultado.get("nivel") or NIVEL_PRIORITARIO
                st.session_state.triaje_paso = "reserva"
            st.rerun()

    elif st.session_state.triaje_paso == "reserva":
        tis = st.session_state.triaje_tis
        conocido = st.session_state.triaje_paciente_conocido
        nivel = st.session_state.triaje_nivel or NIVEL_NORMAL

        if nivel == NIVEL_URGENTE:
            st.error(t("triaje_resultado_urgente", lang))
        else:
            if nivel == NIVEL_PRIORITARIO:
                st.warning(t("triaje_resultado_prioritario", lang))
            else:
                st.success(t("triaje_resultado_normal", lang))
            st.caption(t("triaje_ir_a_agenda", lang))
            st.divider()

            # Nombre a mostrar y riesgo propio para el cálculo de
            # overbooking: de PACIENTES_DB si el TIS es conocido, o un
            # valor neutro si es un paciente nuevo (no tenemos su
            # historial, así que su propio riesgo no empuja el
            # overbooking — solo lo hace el riesgo del titular del hueco).
            #
            # Deliberadamente NO se tocan aquí `st.session_state.logged_in`
            # / `paciente_actual` / `cita_confirmada` — esas son las
            # variables que usa la Agenda clásica (scripts/patient.py) y,
            # como las pestañas se ejecutan todas en cada rerun, escribir
            # en ellas desde aquí para un paciente nuevo (sin
            # `historial_key`) rompería esa otra pestaña. Se usa en su
            # lugar una clave de confirmación propia
            # ("triaje_cita_confirmada"), totalmente separada.
            if conocido:
                nombre_paciente = PACIENTES_DB[tis]["nombre"]
                riesgo_paciente = PACIENTES_DB[tis]["riesgo_propio"]
            else:
                nombre_paciente = tis or "Paciente"
                riesgo_paciente = 0.0

            cita = st.session_state.get("triaje_cita_confirmada")
            # Contenedor con la misma tarjeta blanca que la Agenda clásica
            # (scripts/patient.py, key "patient_panel") pero con SU PROPIA
            # key ("triaje_panel"): Streamlit no permite dos elementos con
            # la misma key en el mismo script run, y las tres pestañas se
            # montan a la vez, así que "patient_panel" ya está en uso por
            # render_agenda_tab(). inject_css() (llamada desde ahí, ya
            # inyectada en la página) aplica el mismo estilo a las dos keys
            # a la vez — ver el selector combinado en scripts/patient.py.
            with st.container(key="triaje_panel"):
                if cita:
                    cuando = f"{cita['dia']} — {cita['hora']}"
                    if cita.get("ai"):
                        cuando += t("agenda_summary_ai_suffix", lang)

                    st.success(t("agenda_confirmed_success", lang))
                    st.markdown(
                        "\n".join([
                            t("agenda_summary_heading", lang),
                            t("agenda_summary_patient", lang, name=nombre_paciente),
                            t("agenda_summary_datetime", lang, when=cuando),
                        ])
                    )
                else:
                    agenda_df = generar_mes_simulado(lang)
                    render_dia_selector(
                        agenda_df, lang, _get_umbral_ia(), riesgo_paciente, nivel,
                        key_prefix="tri", mostrar_aviso_nivel=False,
                        confirmacion_key="triaje_cita_confirmada",
                    )

        if st.button(t("triaje_reiniciar_button", lang)):
            _reiniciar_triaje()
            st.rerun()
