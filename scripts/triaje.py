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
    emergencia) -> puede reservar cualquier día, igual que en nivel normal;
    se le avisa y anima a coger el hueco más próximo, pero no se le
    bloquea el resto de días (bloquearle a un único día fijo iba en
    contra del propio objetivo de "prioritario": verle antes, no obligarle
    a una fecha que puede no encajarle).
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
    calcular_riesgo_nuevo_paciente,
    generar_mes_simulado,
    inject_css,
    render_admin_panel_content,
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
        "sangrado abundante", "sangro mucho", "hemorragia", "desangr", "sangrando mucho",
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
2. EXCEPCIÓN a la regla 1: si "DATOS YA CONOCIDOS" dice explícitamente que
   esta es la ÚNICA pregunta permitida, NO pidas nada más aunque falten
   datos (edad, condiciones...) — decide el nivel ya, con lo que tengas.
3. Evalúa la gravedad real de lo descrito, sin sesgo automático hacia
   ningún nivel: algo claramente banal y sin ningún signo de alarma (un
   golpe leve, una molestia menor, un catarro sin complicaciones...) es
   "normal"; algo que requiere ser visto antes de lo habitual (fiebre alta
   persistente, dolor relevante, empeoramiento progresivo, sospecha
   razonable de complicación...) es "prioritario". Solo ante una duda
   genuina entre los dos, marca "prioritario". Nunca digas que algo no es
   grave ni des consejo médico.
4. Da la conversación por terminada ("listo": true) y asigna "nivel" en
   cuanto tengas edad, condición relevante (o su ausencia clara) y una
   descripción de los síntomas — o, si aplica la EXCEPCIÓN de la regla 2,
   inmediatamente tras la respuesta del paciente sobre sus síntomas.
5. Si NO aplica la EXCEPCIÓN de la regla 2 (es decir, es un paciente
   nuevo), haz SIEMPRE al menos dos preguntas de seguimiento antes de
   poner "listo": true — nunca decidas con un único mensaje del paciente.
   Aunque su primer mensaje ya mencione síntomas, pregunta también por la
   edad y por antecedentes relevantes antes de cerrar, como haría un
   profesional haciendo una anamnesis breve.

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
    como contexto para que el LLM sepa que esta es la ÚNICA pregunta que se
    le hace al paciente (ver la EXCEPCIÓN de la regla 2 del prompt): debe
    decidir "normal" o "prioritario" ya, con esta única respuesta, en vez
    de seguir preguntando por edad/condiciones como haría normalmente.

    Importante: esto es una INSTRUCCIÓN al LLM, no una garantía. El
    llamador (render_triaje_tab) fuerza "listo" a True de todas formas tras
    esta llamada para un TIS conocido, por si el LLM no obedece — pero YA
    NO fuerza el "nivel" a "prioritario" cuando viene vacío: eso fue un
    bug real (todo TIS conocido salía siempre "prioritario", incluso una
    verruga, porque el LLM solía dejar "nivel": null al no tener aún la
    condición crónica y el corte forzado no rellenaba un nivel real). Si a
    pesar de esta instrucción el LLM sigue sin decidir, el nivel se
    resuelve más abajo, en el punto donde se guarda en session_state.

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
                f"edad ni por condiciones crónicas. ESTA ES LA ÚNICA "
                f"PREGUNTA QUE SE LE HACE — no vas a tener otro turno. Con "
                f"su respuesta sobre cómo se encuentra y qué síntomas "
                f"tiene, decide el nivel AHORA: pon \"listo\": true "
                f"siempre en tu respuesta a este mensaje (nunca false ni "
                f"null), y asigna \"nivel\" a \"normal\" o \"prioritario\" "
                f"según la regla 3 — no lo dejes vacío."
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
        st.session_state.triaje_paso = "login"  # login -> chat -> [riesgo_form] -> reserva
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
        "triaje_riesgo_paciente",
    ):
        st.session_state.pop(key, None)


def _render_como_funciona(lang: str) -> None:
    """Mini "cómo funciona" de 3 pasos, solo en la pantalla de login. Sin
    esto, la primera pantalla es literalmente un único campo de texto
    flotando sobre fondo blanco — con esto el paciente entiende de un
    vistazo qué va a pasar (identificarse, contar cómo se encuentra,
    reservar) antes de escribir nada, y la pantalla deja de sentirse vacía."""
    col1, col2, col3 = st.columns(3)
    pasos = [
        (col1, "🪪", t("triaje_paso1_titulo", lang), t("triaje_paso1_desc", lang)),
        (col2, "🩺", t("triaje_paso2_titulo", lang), t("triaje_paso2_desc", lang)),
        (col3, "📅", t("triaje_paso3_titulo", lang), t("triaje_paso3_desc", lang)),
    ]
    for col, icono, titulo, desc in pasos:
        with col:
            st.markdown(
                f'<div class="triaje-paso">'
                f'<div class="triaje-paso-icono">{icono}</div>'
                f'<div class="triaje-paso-titulo">{titulo}</div>'
                f'<div class="triaje-paso-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.divider()


def render_triaje_tab(mostrar_control_umbral: bool = True) -> None:
    """Renders the pre-booking triage tab. Does NOT call configure_page()
    — same convention as render_chatbot_tab()/render_agenda_tab().

    `mostrar_control_umbral=False` es para app_unificado.py (el portal
    clásico de 3 pestañas: chat, triaje y agenda): ahí esta pestaña y la de
    Agenda están montadas A LA VEZ en el mismo script run (Streamlit
    ejecuta el cuerpo de todas las pestañas, no solo la visible), y las dos
    llaman a render_admin_panel_content — si las dos mostraran el slider
    interactivo del umbral, habría dos widgets con la misma key en la misma
    ejecución, algo que Streamlit no permite y que rompería el portal
    entero. Ahí se deja el control interactivo solo en la pestaña Agenda y
    aquí se muestra de solo lectura. En app_unificado_triaje.py (proceso
    propio, sin pestaña de Agenda) el valor por defecto (True) es el
    correcto: es el único sitio de esa vista donde se puede ajustar."""
    lang = st.session_state.get("language", DEFAULT_LANGUAGE)
    _init_triaje_state()

    # inject_css() normalmente la llama render_agenda_tab() — pero esta
    # vista (app_unificado_triaje.py) ya NO monta esa pestaña, así que sin
    # esta línea el CSS de scripts/patient.py (la tarjeta blanca, los
    # colores de los slots, ocultar el header/footer de Streamlit...)
    # nunca se inyecta y toda la pantalla sale con el look plano por
    # defecto de Streamlit. Esta era la causa real de que la vista se
    # viera "vacía": no faltaba contenido, faltaba el estilo.
    inject_css()

    # --- LAYOUT: 70% PACIENTE (IZQUIERDA, BLANCO) / 30% ADMIN (DERECHA, GRIS) ---
    # Mismo patrón de dos columnas que render_agenda_tab (scripts/patient.py):
    # la primera versión de esta vista escondía el panel admin/IA en un
    # desplegable de la barra lateral (solo con el slider) para no estorbar
    # el POV del paciente, pero eso se cargó toda la columna donde vivía el
    # resto del panel (los IDs de la demo, la explicación del umbral) — se
    # restaura aquí, visible, reutilizando render_admin_panel_content tal
    # cual la usa render_agenda_tab, solo que con su propia container key
    # ("admin_panel_triaje") para no chocar con la de la Agenda clásica si
    # algún día ambas vistas llegaran a montarse juntas.
    col_app, col_admin = st.columns([7, 3], gap="large")

    with col_admin:
        with st.container(key="admin_panel_triaje"):
            render_admin_panel_content(lang, mostrar_control_umbral=mostrar_control_umbral)

    with col_app:
        with st.container(key="triaje_panel"):
            col_logo, col_titulo = st.columns([1, 8])
            col_logo.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=52)
            with col_titulo:
                st.markdown(f'<span class="patient-eyebrow">{t("triaje_eyebrow", lang)}</span>', unsafe_allow_html=True)
                st.title(t("triaje_title", lang))
            st.caption(t("triaje_caption", lang))
            st.divider()

            if st.session_state.triaje_paso == "login":
                _render_como_funciona(lang)

                with st.form("triaje_login_form"):
                    tis = st.text_input(t("agenda_tis_label", lang), placeholder=t("agenda_tis_placeholder", lang))
                    submit = st.form_submit_button(t("triaje_start_button", lang), type="primary", use_container_width=True)

                    if submit:
                        st.session_state.triaje_tis = tis
                        st.session_state.triaje_historial = []
                        # TIS conocido: se salta únicamente la pregunta de la
                        # edad (ya está en PACIENTES_DB) — la pregunta médica
                        # se hace igual, para conocidos y desconocidos. Ver la
                        # nota en la cabecera del archivo.
                        st.session_state.triaje_paciente_conocido = tis in PACIENTES_DB
                        st.session_state.triaje_paso = "chat"
                        st.rerun()

                st.caption(f"🔒 {t('triaje_privacidad_nota', lang)}")

            elif st.session_state.triaje_paso == "chat":
                _render_chat_paso(lang)

            elif st.session_state.triaje_paso == "riesgo_form":
                _render_riesgo_form_paso(lang)

            elif st.session_state.triaje_paso == "reserva":
                _render_reserva_paso(lang)


def _render_chat_paso(lang: str) -> None:
    tis = st.session_state.triaje_tis
    conocido = st.session_state.triaje_paciente_conocido

    if not st.session_state.triaje_historial:
        # Primer turno: pregunta inicial fija, sin gastar una llamada al
        # LLM solo para arrancar la conversación. Para un TIS conocido,
        # personalizada con su nombre y pidiendo solo el síntoma (la edad
        # ya la tenemos); para uno nuevo, la pregunta genérica de siempre.
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
    # chatbot/app.py (Streamlit ejecuta el cuerpo de TODAS las pestañas en
    # cada rerun, no solo la visible) — con una key distinta no hay
    # ambigüedad posible entre los dos.
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
            #
            # OJO: aquí hubo un bug real — `resultado.setdefault("nivel",
            # NIVEL_PRIORITARIO)` NO tiene efecto cuando la clave "nivel"
            # ya existe con valor None (que es lo que devuelve el LLM casi
            # siempre en este punto, porque su respuesta natural sin la
            # instrucción de la regla 2 era seguir pidiendo la condición
            # crónica). Resultado: todo TIS conocido salía "prioritario"
            # siempre — una verruga, un catarro, daba igual — porque el
            # nivel real (None) llegaba intacto hasta el `or
            # NIVEL_PRIORITARIO` de más abajo. Con el `if not ...` de aquí
            # SÍ se sustituye None por un valor real cuando el LLM no
            # obedece la instrucción.
            resultado["listo"] = True
            if not resultado.get("nivel"):
                resultado["nivel"] = NIVEL_PRIORITARIO

        st.session_state.triaje_historial.append(
            {"role": "assistant", "content": resultado.get("assistant_response", "")}
        )
        if resultado.get("listo"):
            st.session_state.triaje_nivel = resultado.get("nivel") or NIVEL_PRIORITARIO
            # Paciente conocido (o urgente, aunque ese ya se cortó arriba
            # antes de llegar aquí): directo a elegir día/hora, igual que
            # antes. Paciente NUEVO (no está en PACIENTES_DB): antes de
            # reservar, un breve formulario ("las míticas preguntas") para
            # poder calcularle un riesgo de no-show REAL con el modelo
            # entrenado, en vez de asumirle un 0.0 fijo sin ninguna base —
            # ver _render_riesgo_form_paso más abajo.
            if conocido:
                st.session_state.triaje_paso = "reserva"
            else:
                st.session_state.triaje_paso = "riesgo_form"
        st.rerun()


def _render_riesgo_form_paso(lang: str) -> None:
    """Formulario breve ("las míticas preguntas") para un paciente NUEVO
    (TIS no reconocido en PACIENTES_DB), justo después del triaje médico y
    antes de elegir día/hora.

    Por qué existe: para un TIS conocido, `riesgo_propio` ya está fijado en
    PACIENTES_DB (un valor de demo). Para uno nuevo no hay ningún dato
    histórico, así que antes simplemente se le asignaba un 0.0 fijo — su
    propio riesgo nunca empujaba el overbooking, por mucho que sus
    respuestas aquí dijeran lo contrario. Este formulario pide los mismos
    campos "de toda la vida" (edad, sexo, condiciones crónicas, beca) que
    usaba el chatbot de admisión clásico para calcular el riesgo de
    no-show, y con ellos se llama al modelo real
    (calcular_riesgo_nuevo_paciente, en scripts/patient.py) para obtener una
    probabilidad de verdad en vez de un número inventado.
    """
    st.markdown(t("triaje_riesgo_form_intro", lang))
    st.divider()

    with st.form("triaje_riesgo_form"):
        edad = st.number_input(t("triaje_riesgo_edad_label", lang), min_value=0, max_value=120, value=40, step=1)
        genero = st.radio(
            t("triaje_riesgo_genero_label", lang),
            options=["M", "F"],
            format_func=lambda v: t("triaje_riesgo_genero_m", lang) if v == "M" else t("triaje_riesgo_genero_f", lang),
            horizontal=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            hipertension = st.checkbox(t("triaje_riesgo_hipertension_label", lang))
            diabetes = st.checkbox(t("triaje_riesgo_diabetes_label", lang))
        with c2:
            alcoholismo = st.checkbox(t("triaje_riesgo_alcoholismo_label", lang))
            discapacidad = st.checkbox(t("triaje_riesgo_discapacidad_label", lang))
        beca = st.checkbox(t("triaje_riesgo_beca_label", lang))

        submit = st.form_submit_button(t("triaje_riesgo_form_submit", lang), type="primary", use_container_width=True)
        if submit:
            st.session_state.triaje_riesgo_paciente = calcular_riesgo_nuevo_paciente(
                edad=edad,
                genero_m=(genero == "M"),
                hipertension=hipertension,
                diabetes=diabetes,
                alcoholismo=alcoholismo,
                discapacidad=discapacidad,
                beca=beca,
            )
            st.session_state.triaje_paso = "reserva"
            st.rerun()


def _render_reserva_paso(lang: str) -> None:
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

        # Nombre a mostrar y riesgo propio para el cálculo de overbooking:
        # de PACIENTES_DB si el TIS es conocido, o el valor calculado por
        # el formulario de riesgo (_render_riesgo_form_paso, con el modelo
        # real) si es un paciente nuevo. El 0.0 de repuesto solo se usaría
        # si _render_reserva_paso se alcanzara sin pasar por ese formulario
        # (no debería ocurrir en el flujo normal, pero evita un KeyError).
        #
        # Deliberadamente NO se tocan aquí `st.session_state.logged_in` /
        # `paciente_actual` / `cita_confirmada` — esas son las variables
        # que usa la Agenda clásica (scripts/patient.py) y, si algún día
        # esta vista y la Agenda clásica volvieran a montarse juntas,
        # escribir en ellas desde aquí para un paciente nuevo (sin
        # `historial_key`) la rompería. Se usa en su lugar una clave de
        # confirmación propia ("triaje_cita_confirmada"), totalmente
        # separada.
        if conocido:
            nombre_paciente = PACIENTES_DB[tis]["nombre"]
            riesgo_paciente = PACIENTES_DB[tis]["riesgo_propio"]
        else:
            nombre_paciente = tis or "Paciente"
            riesgo_paciente = st.session_state.get("triaje_riesgo_paciente", 0.0)

        cita = st.session_state.get("triaje_cita_confirmada")
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
                nombre_paciente=nombre_paciente,
            )

    if st.button(t("triaje_reiniciar_button", lang)):
        _reiniciar_triaje()
        st.rerun()
