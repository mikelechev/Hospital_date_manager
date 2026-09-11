# -*- coding: utf-8 -*-
"""
Vista del médico / personal clínico — E&M HealthTech.

Muestra, para los próximos días de la agenda, quién tiene cita y con qué
prioridad, para que el personal sepa de un vistazo a quién atender y por
qué (nivel de triaje / riesgo de no-show) — sin tener que entrar en la
Agenda del paciente ni en el chatbot para averiguarlo.

Vive en un archivo propio, igual que scripts/triaje.py, y NO como una
pestaña más de app_unificado.py: es otro rol completamente distinto
(personal clínico, no paciente) que en un despliegue real ni siquiera
compartiría aplicación con el portal del paciente, y evita de raíz el
mismo problema de colisión de keys/CSS entre pestañas montadas a la vez
que ya ha dado trabajo en el resto del proyecto (Streamlit ejecuta el
cuerpo de TODAS las pestañas en cada rerun, no solo la visible).

Limitación consciente de la demo — y la más importante de este archivo:
`st.session_state` es propio de cada PROCESO de Streamlit. Si esta vista
se lanza como un `streamlit run` separado de app_unificado.py /
app_unificado_triaje.py (que es justo el punto de tenerla aparte), NO
puede leer directamente las citas confirmadas en la sesión de otro
proceso — son dos programas distintos sin memoria compartida. Para que el
médico vea de verdad lo que un paciente acaba de reservar en otra pestaña
del navegador, las citas confirmadas se registran en un fichero JSON
compartido en disco en el momento de reservar (ver
registrar_cita_confirmada / leer_citas_confirmadas en scripts/patient.py,
llamadas desde el único punto por el que pasa cualquier reserva:
render_dia_selector) — la primera pieza de persistencia real del
prototipo, aunque sea mínima, tal como señalaba el roadmap de
propuesta_mejoras_hospital_date_manager.md ("Cero persistencia...
inaceptable para cualquier uso más allá de la demo").

Como esta vista es de otro proceso, no se actualiza sola cuando alguien
reserva en el portal del paciente: hay un botón de refrescar que fuerza un
st.rerun() y, con él, una relectura del fichero compartido (que
deliberadamente no lleva caché).

Fuera de las citas realmente confirmadas, el resto de huecos del día se
rellena con la misma agenda simulada que ya usa el portal del paciente
(generar_mes_simulado) — marcados solo como "ocupado"/"libre", sin
inventar un nombre de paciente para ellos: mezclar un dato real (una
reserva de verdad) con uno inventado (relleno de la simulación) como si
fueran la misma cosa sería confuso y poco honesto de cara a una demo.

Para lanzar esta vista, ejecutar desde la raíz del proyecto:

    streamlit run app_unificado_medico.py
"""

import streamlit as st

from chatbot.i18n import DEFAULT_LANGUAGE, t
from scripts.patient import (
    generar_mes_simulado,
    inject_css,
    leer_citas_confirmadas,
)


def _construir_vista_medico(agenda_df, citas_confirmadas: list) -> list:
    """Combina la agenda simulada con las citas reales confirmadas en
    cualquier proceso de la demo (leídas del fichero compartido), fila a
    fila y en el mismo orden cronológico que ya trae `agenda_df`.

    Empareja por (día, hora) exactos. Si dos citas reales cayeran en el
    mismo hueco (no debería pasar en el flujo normal, cada botón de
    reserva desaparece al confirmarse), se queda con la última leída del
    fichero — no es más que orden de aparición, no un criterio clínico.
    """
    confirmadas_por_slot = {}
    for cita in citas_confirmadas:
        confirmadas_por_slot[(cita.get("dia"), cita.get("hora"))] = cita

    filas = []
    for _, slot in agenda_df.iterrows():
        clave = (slot["fecha"], slot["hora_str"])
        real = confirmadas_por_slot.get(clave)
        if real:
            filas.append({
                "fecha": slot["fecha"],
                "hora_str": slot["hora_str"],
                "confirmada": True,
                "paciente": real.get("nombre") or "Paciente",
                "riesgo": real.get("riesgo"),
                "nivel_triaje": real.get("nivel_triaje"),
                "overbooking": bool(real.get("ai")),
            })
        else:
            filas.append({
                "fecha": slot["fecha"],
                "hora_str": slot["hora_str"],
                "confirmada": False,
                "estado_base": slot["estado_base"],
            })
    return filas


def _render_fila(fila: dict, lang: str) -> None:
    hora = fila["hora_str"]

    if not fila["confirmada"]:
        # Relleno visual de la agenda simulada: nunca se le pone nombre ni
        # nivel de triaje a esto, para no aparentar que es un dato real.
        estado_txt = (
            t("agenda_slot_occupied", lang) if fila["estado_base"] == "Ocupado"
            else t("agenda_slot_free", lang)
        )
        st.markdown(
            f'<div class="medico-fila medico-fila-simulado">'
            f'<div class="medico-fila-hora">{hora}</div>'
            f'<div class="medico-fila-info medico-fila-info-muted">{estado_txt}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    nivel = fila.get("nivel_triaje")
    if nivel == "urgente":
        etiqueta_nivel, css_clase, icono = t("medico_nivel_urgente", lang), "medico-badge-urgente", "🔴"
    elif nivel == "prioritario":
        etiqueta_nivel, css_clase, icono = t("medico_nivel_prioritario", lang), "medico-badge-prioritario", "🟠"
    elif nivel == "normal":
        etiqueta_nivel, css_clase, icono = t("medico_nivel_normal", lang), "medico-badge-normal", "🟢"
    else:
        # Reservas hechas desde la Agenda clásica (sin pasar por triaje) no
        # tienen nivel: se marca aparte en vez de fingir que son "normal".
        etiqueta_nivel, css_clase, icono = t("medico_nivel_sin_triaje", lang), "medico-badge-sin-triaje", "⚪"

    riesgo = fila.get("riesgo")
    riesgo_pct = f"{(riesgo or 0) * 100:.0f}%"
    overbooking_html = ""
    if fila.get("overbooking"):
        overbooking_html = (
            f'<span class="medico-badge medico-badge-overbooking">'
            f'✨ {t("agenda_slot_smartslot", lang)}</span>'
        )

    st.markdown(
        f'<div class="medico-fila medico-fila-{nivel or "sin-triaje"}">'
        f'<div class="medico-fila-hora">{hora}</div>'
        f'<div class="medico-fila-info">'
        f'<b>{fila["paciente"]}</b>'
        f'<span class="medico-badge {css_clase}">{icono} {etiqueta_nivel}</span>'
        f'<span class="medico-riesgo">{t("medico_riesgo_label", lang, pct=riesgo_pct)}</span>'
        f'{overbooking_html}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def render_medico_tab() -> None:
    """Renders the doctor/clinical-staff schedule view. Does NOT call
    configure_page() — same convention as render_triaje_tab()/
    render_agenda_tab()."""
    lang = st.session_state.get("language", DEFAULT_LANGUAGE)
    inject_css()

    agenda_df = generar_mes_simulado(lang)
    citas_confirmadas = leer_citas_confirmadas()

    with st.container(key="medico_panel"):
        col_logo, col_titulo = st.columns([1, 8])
        col_logo.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=52)
        with col_titulo:
            st.markdown(f'<span class="patient-eyebrow">{t("medico_eyebrow", lang)}</span>', unsafe_allow_html=True)
            st.title(t("medico_title", lang))

        c_caption, c_refrescar = st.columns([5, 1])
        c_caption.caption(t("medico_caption", lang))
        if c_refrescar.button(t("medico_refrescar_button", lang), use_container_width=True):
            st.rerun()

        st.divider()

        filas = _construir_vista_medico(agenda_df, citas_confirmadas)
        dias_unicos = agenda_df["fecha"].unique()

        for dia in dias_unicos:
            filas_dia = [f for f in filas if f["fecha"] == dia]
            n_confirmadas = sum(1 for f in filas_dia if f["confirmada"])
            titulo_dia = f"📅 {dia} · {t('medico_citas_confirmadas_contador', lang, n=n_confirmadas)}"
            with st.expander(titulo_dia, expanded=(dia == dias_unicos[0])):
                for fila in filas_dia:
                    _render_fila(fila, lang)
