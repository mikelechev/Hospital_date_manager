"""
Módulo de la Interfaz de Usuario utilizando Streamlit.
Actúa exclusivamente como capa de presentación, delegando toda la lógica
al ConversationManager y al Predictor.
"""

import csv
import html
import io
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from chatbot.config import LLM_CONFIG
from chatbot.providers.provider_factory import ProviderFactory
from chatbot.conversation.conversation_manager import ConversationManager
from chatbot.prediction.predictor import Predictor

# NOTA: la exportación de datos (JSON/CSV/histórico clínico) se hizo antes
# mediante un módulo externo `exports` que nunca llegó a subirse al repo.
# Esa funcionalidad ya está reimplementada íntegramente más abajo
# (build_state_csv / append_state_to_full_csv / render_export_section),
# así que no hace falta ese import.

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Page config + CSS. Streamlit re-ejecuta este script completo en cada
# interacción del usuario (no es un import cacheado), así que este bloque
# se re-evalúa en cada rerun; es barato e idempotente, así que no supone
# ningún problema real.
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Asistente de Admisión Hospitalaria",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# La hoja de estilos ya no es estática: depende de la paleta elegida por el
# usuario, así que se construye e inyecta más abajo (inject_theme_css),
# una vez que sabemos qué preset está activo en session_state.

PROVIDERS = ["ollama", "gemini", "openai", "claude"]
# "custom" se quitó de la lista: ProviderFactory nunca implementó ese caso,
# así que elegirlo en el desplegable rompía la app con un ValueError en
# cuanto se intentaba construir el manager (ver ProviderFactory.get_provider).

# Mapeo provider -> variable de entorno de la API key, usado tanto al
# guardar credenciales en .env como al construir la config de cada proveedor.
_ENV_KEY_BY_PROVIDER = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "claude": "CLAUDE_API_KEY",
}
_ENV_BASE_URL_BY_PROVIDER = {
    "ollama": "OLLAMA_BASE_URL",
    "openai": "OPENAI_BASE_URL",
    "claude": "CLAUDE_BASE_URL",
}


# --------------------------------------------------------------------------- #
# Presets de apariencia (paleta de color)
# --------------------------------------------------------------------------- #
# Solo tocamos aquí los elementos que construimos nosotros mismos (tarjetas
# de paciente, chips, barra de progreso, badge de riesgo, fondo de la app y
# de la barra lateral). No tocamos el tema base de Streamlit (botones,
# sliders, etc. — eso vive en .streamlit/config.toml y solo se aplica
# reiniciando el servidor), así que todos los presets usan fondos claros:
# el texto por defecto de Streamlit es oscuro y así se mantiene siempre
# legible sin tener que reescribir el CSS interno de cada widget nativo.
# El propio menú "⋮ → Settings → Theme" de Streamlit sigue disponible para
# quien quiera un modo oscuro real de toda la app.
DEFAULT_PALETTE = "clinico_azul"

PALETTES: Dict[str, Dict[str, str]] = {
    "clinico_azul": {
        "name": "🔵 Clínico Azul",
        "app_bg": "#eef3fb", "sidebar_bg": "#f7fafd",
        "surface": "#ffffff", "surface_alt": "#eef4fc", "border": "#d6e2f2",
        "text": "#1b2733", "text_muted": "#5b6b7c", "primary": "#1d6fd6",
        "chat_user_bg": "#dceafd", "chat_assistant_bg": "#eef7ee",
        "risk_high": "#d3342d", "risk_medium": "#e08a20", "risk_low": "#1f8a4c",
        "conf_high": "#1f8a4c", "conf_medium": "#e08a20", "conf_low": "#d3342d",
    },
    "verde_salud": {
        "name": "🟢 Verde Salud",
        "app_bg": "#eef8f1", "sidebar_bg": "#f6fbf7",
        "surface": "#ffffff", "surface_alt": "#e9f6ec", "border": "#cfe8d6",
        "text": "#1c2b20", "text_muted": "#54685b", "primary": "#1f8a4c",
        "chat_user_bg": "#dcf3e2", "chat_assistant_bg": "#eaf1fb",
        "risk_high": "#c0392b", "risk_medium": "#d68910", "risk_low": "#1f8a4c",
        "conf_high": "#1f8a4c", "conf_medium": "#d68910", "conf_low": "#c0392b",
    },
    "calido_coral": {
        "name": "🟠 Cálido Coral",
        "app_bg": "#fdf3ee", "sidebar_bg": "#fef8f5",
        "surface": "#ffffff", "surface_alt": "#fbeae1", "border": "#f0d6c6",
        "text": "#3a281f", "text_muted": "#7a5f52", "primary": "#e0622c",
        "chat_user_bg": "#fbe0d0", "chat_assistant_bg": "#eef4fb",
        "risk_high": "#c0392b", "risk_medium": "#d9822b", "risk_low": "#2e8b57",
        "conf_high": "#2e8b57", "conf_medium": "#d9822b", "conf_low": "#c0392b",
    },
    "alto_contraste": {
        "name": "⚫ Alto Contraste",
        "app_bg": "#ffffff", "sidebar_bg": "#ffffff",
        "surface": "#ffffff", "surface_alt": "#f0f0f0", "border": "#000000",
        "text": "#000000", "text_muted": "#000000", "primary": "#0033cc",
        "chat_user_bg": "#fff176", "chat_assistant_bg": "#c8e6c9",
        "risk_high": "#b30000", "risk_medium": "#a15c00", "risk_low": "#006600",
        "conf_high": "#006600", "conf_medium": "#a15c00", "conf_low": "#b30000",
    },
}


def inject_theme_css(palette_key: str) -> None:
    """Construye e inyecta la hoja de estilos del preset de paleta elegido.

    Se llama en cada rerun con el valor actual de st.session_state.theme_palette,
    así que cambiar el desplegable de la barra lateral se refleja al instante
    (Streamlit reruns el script completo tras cada interacción, no hace falta
    reiniciar nada).
    """
    palette = PALETTES.get(palette_key, PALETTES[DEFAULT_PALETTE])
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {palette['app_bg']};
            --sidebar-bg: {palette['sidebar_bg']};
            --surface: {palette['surface']};
            --surface-alt: {palette['surface_alt']};
            --border: {palette['border']};
            --text: {palette['text']};
            --text-muted: {palette['text_muted']};
            --primary: {palette['primary']};
            --chat-user-bg: {palette['chat_user_bg']};
            --chat-assistant-bg: {palette['chat_assistant_bg']};
            --risk-high: {palette['risk_high']};
            --risk-medium: {palette['risk_medium']};
            --risk-low: {palette['risk_low']};
            --conf-high: {palette['conf_high']};
            --conf-medium: {palette['conf_medium']};
            --conf-low: {palette['conf_low']};
        }}

        .stApp {{ background: var(--app-bg); }}
        [data-testid="stSidebar"] {{ background: var(--sidebar-bg); border-right: 1px solid var(--border); }}

        [data-testid="stChatMessage"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 2px 6px;
        }}

        .risk-badge {{
            padding:10px 14px; border-radius:10px; color:#fff; font-weight:700;
            display:inline-block; box-shadow:0 1px 3px rgba(0,0,0,.15);
        }}
        .risk-high {{ background: var(--risk-high); }}
        .risk-medium {{ background: var(--risk-medium); }}
        .risk-low {{ background: var(--risk-low); }}

        .progress-wrap {{ margin: 2px 0 14px; }}
        .progress-label {{ display:flex; justify-content:space-between; font-size:.8em; color:var(--text-muted); margin-bottom:4px; }}
        .progress-track {{ background: var(--surface-alt); border:1px solid var(--border); border-radius:999px; height:10px; overflow:hidden; }}
        .progress-fill {{ height:100%; background: var(--primary); border-radius:999px; transition: width .35s ease; }}

        .patient-card {{
            display:flex; align-items:flex-start; gap:10px; background:var(--surface);
            border:1px solid var(--border); border-radius:10px; padding:8px 10px; margin:6px 0;
        }}
        .patient-card .pc-icon {{ font-size:1.25em; line-height:1.4; }}
        .patient-card .pc-body {{ flex:1; min-width:0; }}
        .patient-card .pc-label {{ font-size:.72em; text-transform:uppercase; letter-spacing:.04em; color:var(--text-muted); }}
        .patient-card .pc-value {{ font-weight:600; color:var(--text); word-break:break-word; }}
        .pc-conf-track {{ background:var(--surface-alt); border-radius:999px; height:5px; margin-top:5px; overflow:hidden; }}
        .pc-conf-fill {{ height:100%; border-radius:999px; }}
        .pc-conf-fill.high {{ background: var(--conf-high); }}
        .pc-conf-fill.medium {{ background: var(--conf-medium); }}
        .pc-conf-fill.low {{ background: var(--conf-low); }}
        .pc-conf-pct {{ font-size:.68em; color:var(--text-muted); }}

        .pending-wrap {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }}
        .pending-chip {{
            display:inline-flex; align-items:center; gap:5px; padding:5px 10px; border-radius:999px;
            background: var(--surface-alt); border:1px dashed var(--border); color:var(--text-muted); font-size:.82em;
        }}

        .small-note {{ font-size:0.9em; color:var(--text-muted); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Cached / expensive-call helpers
# --------------------------------------------------------------------------- #

def _provider_cache_key() -> tuple:
    """Fingerprint of everything that changes which provider instance is valid.

    Used so we don't rebuild the provider (and its underlying client/session)
    on every single Streamlit rerun -- only when the config actually changes.
    """
    return (
        LLM_CONFIG.default_provider,
        LLM_CONFIG.default_model,
        getattr(LLM_CONFIG, "gemini_api_key", None),
        getattr(LLM_CONFIG, "gemini_model", None),
        getattr(LLM_CONFIG, "gemini_base_url", None),
        getattr(LLM_CONFIG, "openai_api_key", None),
        getattr(LLM_CONFIG, "openai_model", None),
        getattr(LLM_CONFIG, "openai_base_url", None),
        getattr(LLM_CONFIG, "claude_api_key", None),
        getattr(LLM_CONFIG, "claude_model", None),
        getattr(LLM_CONFIG, "claude_base_url", None),
        getattr(LLM_CONFIG, "ollama_base_url", None),
        LLM_CONFIG.temperature,
        LLM_CONFIG.max_tokens,
        LLM_CONFIG.request_timeout,
    )


def get_manager() -> Optional[ConversationManager]:
    """Return a cached ConversationManager, rebuilding only when config changed.

    Avoids re-instantiating the LLM provider client on every rerun/keystroke,
    which is the single biggest avoidable cost in this app.
    """
    key = _provider_cache_key()
    if (
        "manager" not in st.session_state
        or st.session_state.get("_manager_key") != key
        or not hasattr(st.session_state.manager, "process_user_input")
        or not hasattr(st.session_state.manager, "process_clinical_history")
    ):
        try:
            provider = ProviderFactory.get_provider()
            st.session_state.manager = ConversationManager(provider)
            st.session_state._manager_key = key
        except Exception as e:
            st.error(f"Error de configuración del proveedor LLM: {e}")
            return None
    return st.session_state.manager


@st.cache_data(ttl=300, show_spinner=False)
def discover_models_cached(provider_name: str, api_key: str, base_url: str) -> List[str]:
    """Cache model discovery for 5 minutes per (provider, key, url) combo.

    Model lists rarely change; this avoids re-hitting the provider's API
    every time the user touches an unrelated widget and triggers a rerun.
    """
    LLM_CONFIG.default_provider = provider_name
    _apply_provider_credentials(provider_name, api_key, base_url)
    provider_inst = ProviderFactory.get_provider()
    return provider_inst.list_models() or []


def _apply_provider_credentials(provider: str, api_key: str, base_url: str) -> None:
    """Escribe api_key/base_url únicamente en los campos del proveedor elegido.

    Antes esta lógica escribía la misma api_key tanto en gemini_api_key como
    en openai_api_key sin mirar qué proveedor estaba seleccionado: guardar una
    clave de OpenAI terminaba pisando silenciosamente la clave de Gemini (y
    viceversa). Además Claude no tenía forma de configurarse desde la UI.
    """
    if provider == "gemini":
        if api_key:
            LLM_CONFIG.gemini_api_key = api_key
        if base_url:
            LLM_CONFIG.gemini_base_url = base_url
    elif provider == "openai":
        if api_key:
            LLM_CONFIG.openai_api_key = api_key
        if base_url:
            LLM_CONFIG.openai_base_url = base_url
    elif provider == "claude":
        if api_key:
            LLM_CONFIG.claude_api_key = api_key
        if base_url:
            LLM_CONFIG.claude_base_url = base_url
    elif provider == "ollama":
        if base_url:
            LLM_CONFIG.ollama_base_url = base_url


# Evita mandar prompts descomunales al LLM (coste/abuso) por un mensaje
# pegado por error o malicioso.
MAX_USER_MESSAGE_CHARS = 4000


def get_predictor() -> Predictor:
    if "predictor" not in st.session_state:
        st.session_state.predictor = Predictor()
    return st.session_state.predictor


# --------------------------------------------------------------------------- #
# CSV export of collected patient data
# --------------------------------------------------------------------------- #

DATA_DIR = Path(__file__).resolve().parent / "data"
FULL_DATA_CSV_PATH = DATA_DIR / "historial_pacientes.csv"
# Streamlit sirve varias sesiones de usuario en el mismo proceso; sin este
# lock, dos pacientes guardando su ficha al mismo tiempo podían intercalar
# escrituras y corromper historial_pacientes.csv.
_CSV_WRITE_LOCK = threading.Lock()


def build_state_csv(state_dump: Dict[str, Any]) -> bytes:
    """Build a downloadable CSV (campo, valor, confianza) from the current state dump."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["campo", "valor", "confianza"])
    for key, data in sorted(state_dump.items()):
        value = data.get("value")
        if value is None:
            continue
        conf = data.get("confidence")
        writer.writerow([key, value, f"{conf:.2f}" if conf is not None else ""])
    # utf-8-sig so accented characters (á, é, ñ...) open correctly in Excel.
    return buf.getvalue().encode("utf-8-sig")


def append_state_to_full_csv(state_dump: Dict[str, Any]) -> Path:
    """Append the current patient's collected data as one row to a cumulative
    dataset CSV on disk (useful for later retraining the prediction model).

    The header is a union of all fields seen so far, so it stays valid even
    as new fields appear across different patients/sessions.
    """
    row = {"timestamp": datetime.now().isoformat(timespec="seconds")}
    for key, data in state_dump.items():
        if data.get("value") is not None:
            row[key] = data["value"]

    with _CSV_WRITE_LOCK:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        fieldnames = ["timestamp"] + sorted(k for k in row if k != "timestamp")
        if FULL_DATA_CSV_PATH.exists():
            with open(FULL_DATA_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
                existing_header = next(csv.reader(f), [])
            fieldnames = sorted(set(existing_header) | set(fieldnames), key=lambda c: (c != "timestamp", c))

        write_header = not FULL_DATA_CSV_PATH.exists()
        with open(FULL_DATA_CSV_PATH, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    return FULL_DATA_CSV_PATH


def render_export_section(state_dump: Dict[str, Any]) -> None:
    """Renders CSV download + 'save to dataset' controls for the collected data."""
    st.divider()
    st.subheader("📥 Exportar datos")

    if not state_dump or not any(v.get("value") is not None for v in state_dump.values()):
        st.caption("Aún no hay datos recabados para exportar.")
        return

    csv_bytes = build_state_csv(state_dump)
    st.download_button(
        "⬇️ Descargar CSV de esta ficha",
        data=csv_bytes,
        file_name=f"paciente_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if st.button("💾 Añadir al historial acumulado (CSV)", use_container_width=True):
        try:
            path = append_state_to_full_csv(state_dump)
            st.success(f"Ficha añadida a {path}")
        except Exception as e:
            st.error(f"No se pudo guardar en el historial acumulado: {e}")


# --------------------------------------------------------------------------- #
# Clinical history ("ficha") ingestion — LLM reads it and fills the fields
# --------------------------------------------------------------------------- #

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


def extract_text_from_upload(uploaded_file) -> str:
    """Extract raw text from an uploaded .txt, .pdf or .docx clinical record."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if len(data) > MAX_UPLOAD_BYTES:
        st.error(f"El archivo supera el tamaño máximo permitido ({MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
        return ""

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            import pypdf
        except ImportError:
            st.error("Para leer PDFs instala 'pypdf' (pip install pypdf).")
            return ""
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.error(f"No se pudo leer el PDF: {e}")
            return ""

    if name.endswith(".docx"):
        try:
            import docx
        except ImportError:
            st.error("Para leer .docx instala 'python-docx' (pip install python-docx).")
            return ""
        try:
            document = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in document.paragraphs)
        except Exception as e:
            st.error(f"No se pudo leer el .docx: {e}")
            return ""

    st.error("Formato no soportado. Usa .txt, .pdf o .docx.")
    return ""


def render_clinical_history_section() -> None:
    """Lets the user upload/paste a clinical history record and has the LLM
    (via ConversationManager.process_clinical_history) extract structured
    fields from it, the same way it would from a chat message.
    """
    st.subheader("📁 Ficha de Historial Clínico")
    st.caption("Sube o pega el historial clínico del paciente para que el asistente extraiga los datos automáticamente.")

    uploaded = st.file_uploader(
        "Subir ficha (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], key="clinical_history_upload"
    )
    pasted_text = st.text_area("...o pega el texto del historial aquí", height=120, key="clinical_history_text")

    if st.button("🧠 Procesar historial con el LLM", use_container_width=True):
        text = ""
        if uploaded is not None:
            text = extract_text_from_upload(uploaded)
        elif pasted_text.strip():
            text = pasted_text.strip()
        else:
            st.warning("Sube un archivo o pega el texto del historial primero.")
            return

        if not text.strip():
            return  # extraction already reported the error, nothing to process

        manager = get_manager()
        if manager is None:
            return

        if not hasattr(manager, "process_clinical_history"):
            st.error(
                "ConversationManager no implementa todavía 'process_clinical_history(texto)'. "
                "Añade ese método (debe leer el texto del historial, extraer los campos clínicos "
                "relevantes con el LLM, actualizar `manager.state` igual que process_user_input, "
                "y devolver un resumen en texto de lo extraído) para habilitar esta función."
            )
            return

        with st.spinner("El LLM está leyendo el historial clínico..."):
            try:
                summary = manager.process_clinical_history(text)
            except Exception as e:
                st.error(f"Error procesando el historial clínico: {e}")
                return

        summary_text = _format_clinical_summary(summary)
        is_error = isinstance(summary, dict) and "error" in summary and "assistant_response" not in summary

        st.session_state.clinical_history_path = uploaded.name if uploaded is not None else "texto pegado"

        if is_error:
            st.session_state.messages_ui.append({"role": "assistant", "content": summary_text})
            st.error("No se pudo procesar el historial clínico. Revisa los logs para más detalle.")
        else:
            st.session_state.messages_ui.append(
                {
                    "role": "assistant",
                    "content": (
                        "He leído la ficha de historial clínico y actualizado los datos del paciente.\n\n"
                        + summary_text
                    ).strip(),
                }
            )
            st.success("Historial clínico procesado y datos actualizados.")
        st.rerun()


def _format_clinical_summary(summary: Any) -> str:
    """Normalize whatever process_clinical_history returns into display text.

    Handles the shape returned by ConversationManager.process_clinical_history
    ({"assistant_response": ..., "analysis": {...}, "ready_for_prediction": ...}),
    an error dict ({"error": "..."}), a plain string, a flat dict of extracted
    fields, or a list of such items. Never assumes a "confidence" value is
    numeric before formatting it.
    """
    if summary is None:
        return ""
    if isinstance(summary, str):
        return summary

    if isinstance(summary, dict):
        # Error shape from process_clinical_history's except-branch.
        if "error" in summary and "assistant_response" not in summary:
            return f"⚠️ Error del LLM al interpretar el historial: {summary['error']}"

        # Expected shape: {"assistant_response": ..., "analysis": {...}, ...}
        if "assistant_response" in summary:
            lines = [str(summary["assistant_response"])]
            analysis = summary.get("analysis") or {}
            extracted = analysis.get("extracted_data") or {}
            confidences = analysis.get("confidence") or {}

            field_lines = []
            for key, value in extracted.items():
                if value is None:
                    continue
                conf = confidences.get(key)
                conf_str = f" (confianza: {conf:.2f})" if isinstance(conf, (int, float)) else ""
                field_lines.append(f"- **{key}**: {value}{conf_str}")

            if field_lines:
                lines.append("")
                lines.append("Datos extraídos:")
                lines.extend(field_lines)
            return "\n".join(lines)

        # Generic fallback for any other dict shape — guards against a
        # non-numeric "confidence" (e.g. a nested dict) crashing the format.
        lines = []
        for key, val in summary.items():
            if isinstance(val, dict):
                value = val.get("value", val)
                conf = val.get("confidence")
                conf_str = f" (confianza: {conf:.2f})" if isinstance(conf, (int, float)) else ""
                lines.append(f"- **{key}**: {value}{conf_str}")
            else:
                lines.append(f"- **{key}**: {val}")
        return "\n".join(lines)

    if isinstance(summary, (list, tuple)):
        return "\n".join(f"- {item}" for item in summary)
    return str(summary)


# --------------------------------------------------------------------------- #
# Session lifecycle
# --------------------------------------------------------------------------- #

def initialize_session() -> None:
    """Initialize session_state defaults. Called once at the top of main()."""
    if "theme_palette" not in st.session_state:
        st.session_state.theme_palette = DEFAULT_PALETTE
    if "messages_ui" not in st.session_state:
        st.session_state.messages_ui = [
            {
                "role": "assistant",
                "content": "Hola. Soy el asistente virtual del hospital. ¿En qué te puedo ayudar hoy?",
            }
        ]
    if "clinical_history_path" not in st.session_state:
        st.session_state.clinical_history_path = None
    if "discovered_models" not in st.session_state:
        st.session_state.discovered_models = []

    # Ensure manager/predictor exist without forcing a rebuild if valid.
    get_manager()
    get_predictor()


def reset_conversation() -> None:
    """Borra el estado actual para iniciar un nuevo flujo conversacional."""
    logger.info("Reiniciando la conversación a petición del usuario.")
    for key in ("manager", "_manager_key", "predictor", "messages_ui", "clinical_history_path", "discovered_models"):
        st.session_state.pop(key, None)
    st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def render_sidebar() -> None:
    """Renderiza el panel lateral: proveedor LLM, credenciales y ayuda."""
    with st.sidebar:
        st.selectbox(
            "🎨 Paleta de color",
            options=list(PALETTES.keys()),
            format_func=lambda k: PALETTES[k]["name"],
            key="theme_palette",
            help="Cambia el aspecto visual de la app al instante (no afecta a los datos del paciente).",
        )
        st.divider()

        st.header("⚙️ Configuración del Sistema")

        provider = st.selectbox(
            "Proveedor",
            PROVIDERS,
            index=PROVIDERS.index(LLM_CONFIG.default_provider) if LLM_CONFIG.default_provider in PROVIDERS else 0,
            key="provider_select",
        )

        # Campos específicos por proveedor. Antes había un único input de
        # "API Key" compartido entre Gemini y OpenAI (y sin nada para
        # Claude), así que introducir la clave de un proveedor pisaba la
        # del otro en cuanto se aplicaba la configuración.
        api_key = ""
        base_url = ""
        if provider == "ollama":
            base_url = st.text_input("Base URL de Ollama", value=LLM_CONFIG.ollama_base_url)
            model_hint = st.text_input("Modelo (opcional)", value=LLM_CONFIG.default_model)
        elif provider == "gemini":
            api_key = st.text_input("Gemini API Key", value=LLM_CONFIG.gemini_api_key, type="password")
            base_url = st.text_input(
                "Gemini Base URL (opcional)", value=getattr(LLM_CONFIG, "gemini_base_url", "") or ""
            )
            model_hint = st.text_input("Modelo", value=LLM_CONFIG.gemini_model)
        elif provider == "openai":
            api_key = st.text_input("OpenAI API Key", value=LLM_CONFIG.openai_api_key, type="password")
            base_url = st.text_input("OpenAI Base URL (opcional)", value=LLM_CONFIG.openai_base_url or "")
            model_hint = st.text_input("Modelo", value=LLM_CONFIG.openai_model)
        else:  # claude
            api_key = st.text_input("Claude API Key", value=LLM_CONFIG.claude_api_key, type="password")
            base_url = st.text_input("Claude Base URL (opcional)", value=LLM_CONFIG.claude_base_url or "")
            model_hint = st.text_input("Modelo", value=LLM_CONFIG.claude_model)

        # NOTA: LLM_CONFIG es un singleton a nivel de módulo, compartido por
        # todas las sesiones de Streamlit del mismo proceso. Estos widgets
        # (y "Aplicar configuración") cambian la configuración para TODOS
        # los usuarios conectados, no solo para quien mueve el slider. Para
        # un despliegue multiusuario real habría que mover esta config a
        # st.session_state y pasarla explícitamente a ProviderFactory.
        LLM_CONFIG.temperature = st.slider(
            "Temperatura (Creatividad vs Precisión)",
            min_value=0.0, max_value=1.0, value=LLM_CONFIG.temperature, step=0.1,
            help="Mantenlo en 0.0 para maximizar la consistencia del JSON.",
        )
        LLM_CONFIG.max_tokens = st.number_input(
            "Max Tokens", min_value=64, max_value=4096, value=LLM_CONFIG.max_tokens, step=64
        )

        discover = st.button("🔎 Buscar modelos disponibles")
        if discover:
            try:
                with st.spinner("Buscando modelos..."):
                    models = discover_models_cached(provider, api_key, base_url)
                st.session_state.discovered_models = models
                if not models and model_hint:
                    st.warning("No se pudieron listar modelos automáticamente; se usará el nombre indicado.")
            except Exception as e:
                st.error(f"Error inicializando proveedor: {e}")

        # Persisted across reruns, unlike the original which vanished
        # as soon as `discover` went back to False on the next script run.
        if st.session_state.discovered_models:
            chosen = st.selectbox("Modelos detectados", st.session_state.discovered_models, key="chosen_model")
            if st.button("Usar este modelo"):
                _apply_llm_settings(provider, api_key, base_url, chosen)
        elif model_hint:
            if st.button("Aplicar configuración"):
                _apply_llm_settings(provider, api_key, base_url, model_hint)

        with st.expander("💾 Guardar credenciales"):
            _render_credential_persistence(provider, api_key, base_url)

        st.divider()
        with st.expander("📁 Ficha de Historial Clínico", expanded=False):
            render_clinical_history_section()

        st.divider()
        with st.expander("Ayuda rápida"):
            st.markdown(
                "**Sugerencias de prompts:**\n"
                "- 'Hola, necesito ayuda para una cita'\n"
                "- 'Tengo dolor de cabeza y fiebre desde ayer'\n"
                "- '¿Qué documentos necesito llevar?'\n\n"
                "**Consejos:**\n"
                "- Pega tu API key si usas Gemini / OpenAI / Claude.\n"
                "- Usa 'Buscar modelos' para detectar modelos disponibles."
            )

        st.divider()
        if st.button("🔄 Nueva Conversación", use_container_width=True):
            reset_conversation()


def _apply_llm_settings(provider: str, api_key: str, base_url: str, model: str) -> None:
    """Apply chosen provider/model settings and invalidate the cached manager."""
    LLM_CONFIG.default_provider = provider
    LLM_CONFIG.default_model = model
    _apply_provider_credentials(provider, api_key, base_url)
    if provider == "gemini":
        LLM_CONFIG.gemini_model = model
    elif provider == "openai":
        LLM_CONFIG.openai_model = model
    elif provider == "claude":
        LLM_CONFIG.claude_model = model
    st.success(f"Modelo aplicado: {model}")
    st.rerun()


def _render_credential_persistence(provider: str, api_key: str, base_url: str) -> None:
    """Plaintext .env save, plus optional encrypted save/load if `cryptography` is installed."""
    if st.button("Guardar en chatbot/.env"):
        try:
            env_path = Path(__file__).resolve().parent / ".env"
            lines = [f"DEFAULT_PROVIDER={provider}"]
            env_key = _ENV_KEY_BY_PROVIDER.get(provider)
            if env_key and api_key:
                lines.append(f"{env_key}={api_key}")
            base_url_env_key = _ENV_BASE_URL_BY_PROVIDER.get(provider)
            if base_url_env_key and base_url:
                lines.append(f"{base_url_env_key}={base_url}")
            if LLM_CONFIG.default_model:
                lines.append(f"DEFAULT_MODEL={LLM_CONFIG.default_model}")
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            try:
                os.chmod(env_path, 0o600)
            except Exception:
                pass
            st.success(f"Credenciales guardadas en {env_path}")
        except Exception as e:
            st.error(f"No se pudo guardar .env: {e}")

    try:
        from chatbot.utils.crypto import HAS_CRYPTO, encrypt_dict, decrypt_file
    except Exception:
        HAS_CRYPTO = False

    if not HAS_CRYPTO:
        st.caption("Instala 'cryptography' (pip install cryptography) para guardar credenciales encriptadas.")
        return

    passphrase = st.text_input("Passphrase para encriptar", type="password", key="enc_pass")
    passphrase2 = st.text_input("Confirmar passphrase", type="password", key="enc_pass2")
    if st.button("Encriptar y guardar .env.enc"):
        if not passphrase or passphrase != passphrase2:
            st.error("Las passphrases no coinciden o están vacías.")
        else:
            data = {"DEFAULT_PROVIDER": provider}
            env_key = _ENV_KEY_BY_PROVIDER.get(provider)
            if env_key and api_key:
                data[env_key] = api_key
            base_url_env_key = _ENV_BASE_URL_BY_PROVIDER.get(provider)
            if base_url_env_key and base_url:
                data[base_url_env_key] = base_url
            if LLM_CONFIG.default_model:
                data["DEFAULT_MODEL"] = LLM_CONFIG.default_model
            try:
                enc = encrypt_dict(passphrase, data)
                env_enc_path = Path(__file__).resolve().parent / ".env.enc"
                env_enc_path.write_text(json.dumps(enc, ensure_ascii=False), encoding="utf-8")
                try:
                    os.chmod(env_enc_path, 0o600)
                except Exception:
                    pass
                st.success(f"Credenciales encriptadas guardadas en {env_enc_path}")
            except Exception as e:
                st.error(f"Fallo al encriptar: {e}")

    enc_path = Path(__file__).resolve().parent / ".env.enc"
    if enc_path.exists():
        dec_pass = st.text_input("Passphrase para desencriptar", type="password", key="dec_pass")
        if st.button("Cargar .env.enc"):
            try:
                data = decrypt_file(dec_pass, str(enc_path))
                if "DEFAULT_PROVIDER" in data:
                    LLM_CONFIG.default_provider = data["DEFAULT_PROVIDER"]
                if "GEMINI_API_KEY" in data:
                    LLM_CONFIG.gemini_api_key = data["GEMINI_API_KEY"]
                if "OPENAI_API_KEY" in data:
                    LLM_CONFIG.openai_api_key = data["OPENAI_API_KEY"]
                if "CLAUDE_API_KEY" in data:
                    LLM_CONFIG.claude_api_key = data["CLAUDE_API_KEY"]
                if "OLLAMA_BASE_URL" in data:
                    LLM_CONFIG.ollama_base_url = data["OLLAMA_BASE_URL"]
                if "OPENAI_BASE_URL" in data:
                    LLM_CONFIG.openai_base_url = data["OPENAI_BASE_URL"]
                if "CLAUDE_BASE_URL" in data:
                    LLM_CONFIG.claude_base_url = data["CLAUDE_BASE_URL"]
                if "DEFAULT_MODEL" in data:
                    LLM_CONFIG.default_model = data["DEFAULT_MODEL"]
                st.success("Credenciales cargadas en la configuración de sesión")
                st.rerun()
            except Exception as e:
                st.error(f"Fallo al desencriptar: {e}")


# --------------------------------------------------------------------------- #
# Patient status panel
# --------------------------------------------------------------------------- #

# Icono + etiqueta legible para cada campo de PatientState. Un campo que no
# esté aquí (por ejemplo si se amplía PatientState más adelante) sigue
# funcionando: cae en el fallback de _field_meta().
FIELD_META: Dict[str, Dict[str, str]] = {
    "age": {"icon": "🎂", "label": "Edad"},
    "gender_m": {"icon": "🚻", "label": "Sexo"},
    "hypertension": {"icon": "❤️", "label": "Hipertensión"},
    "diabetes": {"icon": "🩸", "label": "Diabetes"},
    "alcoholism": {"icon": "🍷", "label": "Alcoholismo"},
    "handicap": {"icon": "♿", "label": "Discapacidad"},
    "scholarship": {"icon": "🎓", "label": "Beca social"},
    "sms_received": {"icon": "📩", "label": "SMS recibido"},
    "history_no_show": {"icon": "📊", "label": "Historial de faltas"},
    "days_between": {"icon": "📅", "label": "Días de antelación"},
    "weekend": {"icon": "🗓️", "label": "Cita en fin de semana"},
    "time_of_day": {"icon": "⏰", "label": "Horario de la cita"},
    "consultation_reason": {"icon": "📝", "label": "Motivo de consulta"},
}

_BOOLEAN_FIELDS = {
    "hypertension", "diabetes", "alcoholism", "handicap", "scholarship", "sms_received", "weekend",
}


def _field_meta(key: str) -> Dict[str, str]:
    return FIELD_META.get(key, {"icon": "•", "label": key.replace("_", " ").capitalize()})


def _humanize_value(key: str, value: Any) -> str:
    """Convierte el valor crudo del estado en texto legible para humanos."""
    if key == "gender_m":
        return "Masculino" if value in (1, "1", 1.0, True) else "Femenino"
    if key == "age":
        return f"{value} años"
    if key == "days_between":
        return f"{value} días"
    if key == "history_no_show":
        try:
            return f"{float(value):.0%}"
        except (TypeError, ValueError):
            return str(value)
    if key in _BOOLEAN_FIELDS:
        return "Sí" if value in (1, "1", 1.0, True) else "No"
    return str(value)


def _confidence_class(conf: float) -> str:
    if conf >= 0.8:
        return "high"
    if conf >= 0.5:
        return "medium"
    return "low"


def render_patient_status(state_dump: Dict[str, Any], missing_fields: list) -> None:
    st.subheader("📋 Estado del Paciente")

    if not state_dump:
        st.info("Aún no se ha recopilado información.")
        return

    total_fields = len(state_dump)
    extracted_data = {k: v for k, v in state_dump.items() if v.get("value") is not None}
    completed = len(extracted_data)
    pct = (completed / total_fields) if total_fields else 0.0

    # Barra de progreso general de la ficha — da una vista "de un vistazo"
    # de cuánto falta, en vez de tener que contar la lista de pendientes.
    st.markdown(
        f"""
        <div class="progress-wrap">
          <div class="progress-label">
            <span>Progreso de la ficha</span>
            <span>{completed}/{total_fields} campos · {pct:.0%}</span>
          </div>
          <div class="progress-track"><div class="progress-fill" style="width:{pct * 100:.0f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if extracted_data:
        cards = []
        for key, data in extracted_data.items():
            value = data["value"]
            conf = data.get("confidence") or 0.0
            meta = _field_meta(key)
            conf_class = _confidence_class(conf)
            # html.escape es imprescindible aquí: "value" puede venir de texto
            # que el paciente escribió en el chat y que el LLM copió tal cual
            # a "extracted_data" (p.ej. consultation_reason). Como esto se
            # renderiza con unsafe_allow_html=True, sin escapar sería una
            # inyección de HTML/JS trivial (XSS) a través del propio chat.
            safe_label = html.escape(meta["label"])
            safe_value = html.escape(_humanize_value(key, value))
            cards.append(
                f"""
                <div class="patient-card">
                  <div class="pc-icon">{meta['icon']}</div>
                  <div class="pc-body">
                    <div class="pc-label">{safe_label}</div>
                    <div class="pc-value">{safe_value}</div>
                    <div class="pc-conf-track">
                      <div class="pc-conf-fill {conf_class}" style="width:{conf * 100:.0f}%"></div>
                    </div>
                    <span class="pc-conf-pct">Confianza: {conf:.0%}</span>
                  </div>
                </div>
                """
            )
        st.markdown("".join(cards), unsafe_allow_html=True)
    else:
        st.caption("Todavía no hay campos confirmados con datos.")

    st.divider()
    st.subheader("🎯 Variables Pendientes")
    if missing_fields:
        chips = []
        for field in missing_fields:
            meta = _field_meta(field)
            chips.append(f'<span class="pending-chip">{meta["icon"]} {html.escape(meta["label"])}</span>')
        st.markdown(f'<div class="pending-wrap">{"".join(chips)}</div>', unsafe_allow_html=True)
    else:
        st.success("¡Información completada!")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    initialize_session()
    inject_theme_css(st.session_state.theme_palette)

    st.title("🏥 Asistente de Admisión Hospitalaria — Chatbot")
    st.caption("Interfaz para conversar con el asistente y ver el estado del paciente en tiempo real.")

    render_sidebar()

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Chat")

        # st.chat_message renders natively and incrementally — Streamlit only
        # diffs what changed, unlike the previous approach of rebuilding one
        # giant HTML string every rerun and force-scrolling it via an
        # injected <script> inside a components.html iframe.
        chat_box = st.container(height=500)
        with chat_box:
            for msg in st.session_state.messages_ui:
                avatar = "🩺" if msg["role"] == "assistant" else "🙋"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(msg["content"])

        user_text = st.chat_input("Escribe tu mensaje aquí...")

        if user_text and user_text.strip():
            user_text = user_text.strip()
            if len(user_text) > MAX_USER_MESSAGE_CHARS:
                st.warning(
                    f"Tu mensaje superaba los {MAX_USER_MESSAGE_CHARS} caracteres; se ha truncado."
                )
                user_text = user_text[:MAX_USER_MESSAGE_CHARS]
            st.session_state.messages_ui.append({"role": "user", "content": user_text})
            manager = get_manager()

            if manager is not None:
                try:
                    with st.spinner("El asistente está escribiendo..."):
                        reply, ready = manager.process_user_input(user_text)
                    st.session_state.messages_ui.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.session_state.messages_ui.append(
                        {"role": "assistant", "content": f"⚠️ Ocurrió un error: {e}"}
                    )
                    st.session_state.last_error = str(e)
            st.rerun()

    with col2:
        st.subheader("Estado paciente & Predicción")
        mgr = get_manager()
        if mgr is not None:
            try:
                state_dump = mgr.get_current_state()
                missing = mgr.state.get_missing_critical_fields()
                render_patient_status(state_dump, missing)
                render_export_section(state_dump)

                if mgr.state.is_ready_for_prediction():
                    pred = get_predictor().predict(mgr.state)
                    color_class = "risk-low"
                    if pred.risk_level == "ALTO":
                        color_class = "risk-high"
                    elif pred.risk_level == "MEDIO":
                        color_class = "risk-medium"
                    st.markdown(
                        f"<div class='risk-badge {color_class}'>"
                        f"Probabilidad de ausencia: {pred.probability:.2%} — {pred.risk_level}</div>",
                        unsafe_allow_html=True,
                    )
                    if pred.is_fallback:
                        st.caption("(Fallback usado — modelo ausente)")
            except Exception as e:
                st.error(f"Error mostrando estado/predicción: {e}")


if __name__ == "__main__":
    main()