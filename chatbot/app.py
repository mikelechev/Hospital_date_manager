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
from chatbot.i18n import DEFAULT_LANGUAGE, LANGUAGES, t

# NOTA: la exportación de datos (JSON/CSV/histórico clínico) se hizo antes
# mediante un módulo externo `exports` que nunca llegó a subirse al repo.
# Esa funcionalidad ya está reimplementada íntegramente más abajo
# (build_state_csv / append_state_to_full_csv / render_export_section),
# así que no hace falta ese import.

logger = logging.getLogger(__name__)


def _current_language() -> str:
    """Idioma activo de la UI/asistente ("es"/"eu"), con fallback seguro.

    Usa st.session_state.get en vez de acceder directo, porque se llama
    también antes de que initialize_session() haya corrido (p.ej. desde
    configure_page()).
    """
    return st.session_state.get("language", DEFAULT_LANGUAGE)


# --------------------------------------------------------------------------- #
# Page config + CSS
#
# Extracted into functions (instead of executed at import time) so this
# module can be reused as one tab of the combined portal
# (see app_unificado.py at the project root) without forcing its own page
# config or re-injecting CSS at import time. Standalone execution
# (`streamlit run chatbot/app.py`) is unaffected: main() below still calls
# configure_page() once, exactly like before.
#
# Note: Streamlit re-ejecuta este script completo en cada interacción del
# usuario (no es un import cacheado), así que este bloque se re-evalúa en
# cada rerun; es barato e idempotente, así que no supone ningún problema real.
# --------------------------------------------------------------------------- #

def configure_page() -> None:
    """Sets the Streamlit page config. Call at most once per app run, before
    any other Streamlit command. Only the script that owns the process
    should call it (this file in standalone mode, or app_unificado.py when
    this tab is embedded in the combined portal)."""
    st.set_page_config(
        page_title=t("page_title", _current_language()),
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_css() -> None:
    """Injects this tab's CSS. Idempotent/cheap — safe to call on every rerun."""
    st.markdown(
        """
        <style>
        .chat-bubble { padding:10px; border-radius:10px; margin:8px 0; max-width:75%; }
        .chat-user { background:#e6f2ff; margin-left:auto; }
        .chat-assistant { background:#f1f8e9; margin-right:auto; }
        .chat-meta { font-size:0.8em; color:#666; margin-bottom:6px; }
        .risk-badge { padding:8px 12px; border-radius:6px; color:#fff; font-weight:600; display:inline-block;}
        .risk-high { background:#d32f2f; }
        .risk-medium { background:#f57c00; }
        .risk-low { background:#2e7d32; }
        .small-note { font-size:0.9em; color:#666; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# La hoja de estilos ya no es estática: depende de la paleta elegida por el
# usuario, así que se construye e inyecta más abajo (inject_theme_css),
# una vez que sabemos qué preset está activo en session_state.

PROVIDERS = ["ollama", "gemini", "openai", "claude", "groq"]
# "custom" se quitó de la lista: ProviderFactory nunca implementó ese caso,
# así que elegirlo en el desplegable rompía la app con un ValueError en
# cuanto se intentaba construir el manager (ver ProviderFactory.get_provider).

# Mapeo provider -> variable de entorno de la API key, usado tanto al
# guardar credenciales en .env como al construir la config de cada proveedor.
_ENV_KEY_BY_PROVIDER = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "claude": "CLAUDE_API_KEY",
    "groq": "GROQ_API_KEY",
}
_ENV_BASE_URL_BY_PROVIDER = {
    "ollama": "OLLAMA_BASE_URL",
    "openai": "OPENAI_BASE_URL",
    "claude": "CLAUDE_BASE_URL",
    "groq": "GROQ_BASE_URL",
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
        getattr(LLM_CONFIG, "groq_api_key", None),
        getattr(LLM_CONFIG, "groq_model", None),
        getattr(LLM_CONFIG, "groq_base_url", None),
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
            st.error(t("provider_config_error", _current_language(), error=e))
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
    elif provider == "groq":
        if api_key:
            LLM_CONFIG.groq_api_key = api_key
        if base_url:
            LLM_CONFIG.groq_base_url = base_url
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
    lang = _current_language()
    st.divider()
    st.subheader(t("export_subheader", lang))

    if not state_dump or not any(v.get("value") is not None for v in state_dump.values()):
        st.caption(t("export_no_data_caption", lang))
        return

    csv_bytes = build_state_csv(state_dump)
    st.download_button(
        t("export_download_button", lang),
        data=csv_bytes,
        file_name=f"paciente_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        width="stretch",
    )

    if st.button(t("export_add_history_button", lang), width="stretch"):
        try:
            path = append_state_to_full_csv(state_dump)
            st.success(t("export_add_history_success", lang, path=path))
        except Exception as e:
            st.error(t("export_add_history_error", lang, error=e))


# --------------------------------------------------------------------------- #
# Clinical history ("ficha") ingestion — LLM reads it and fills the fields
# --------------------------------------------------------------------------- #

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


def extract_text_from_upload(uploaded_file) -> str:
    """Extract raw text from an uploaded .txt, .pdf or .docx clinical record."""
    lang = _current_language()
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if len(data) > MAX_UPLOAD_BYTES:
        st.error(t("upload_too_large_error", lang, max_mb=MAX_UPLOAD_BYTES // (1024 * 1024)))
        return ""

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            import pypdf
        except ImportError:
            st.error(t("pdf_missing_lib_error", lang))
            return ""
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.error(t("pdf_read_error", lang, error=e))
            return ""

    if name.endswith(".docx"):
        try:
            import docx
        except ImportError:
            st.error(t("docx_missing_lib_error", lang))
            return ""
        try:
            document = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in document.paragraphs)
        except Exception as e:
            st.error(t("docx_read_error", lang, error=e))
            return ""

    st.error(t("unsupported_format_error", lang))
    return ""


def render_clinical_history_section() -> None:
    """Lets the user upload/paste a clinical history record and has the LLM
    (via ConversationManager.process_clinical_history) extract structured
    fields from it, the same way it would from a chat message.
    """
    lang = _current_language()
    st.subheader(t("clinical_history_subheader", lang))
    st.caption(t("clinical_history_caption", lang))

    uploaded = st.file_uploader(
        t("clinical_history_upload_label", lang), type=["txt", "pdf", "docx"], key="clinical_history_upload"
    )
    pasted_text = st.text_area(t("clinical_history_paste_label", lang), height=120, key="clinical_history_text")

    if st.button(t("clinical_history_process_button", lang), width="stretch"):
        text = ""
        if uploaded is not None:
            text = extract_text_from_upload(uploaded)
        elif pasted_text.strip():
            text = pasted_text.strip()
        else:
            st.warning(t("clinical_history_warning_empty", lang))
            return

        if not text.strip():
            return  # extraction already reported the error, nothing to process

        manager = get_manager()
        if manager is None:
            return

        if not hasattr(manager, "process_clinical_history"):
            st.error(t("clinical_history_not_implemented_error", lang))
            return

        with st.spinner(t("clinical_history_spinner", lang)):
            try:
                summary = manager.process_clinical_history(text, lang)
            except Exception as e:
                st.error(t("clinical_history_error", lang, error=e))
                return

        summary_text = _format_clinical_summary(summary)
        is_error = isinstance(summary, dict) and "error" in summary and "assistant_response" not in summary

        st.session_state.clinical_history_path = (
            uploaded.name if uploaded is not None else t("clinical_history_path_pasted", lang)
        )

        if is_error:
            st.session_state.messages_ui.append({"role": "assistant", "content": summary_text})
            st.error(t("clinical_history_process_error_generic", lang))
        else:
            st.session_state.messages_ui.append(
                {
                    "role": "assistant",
                    "content": (
                        t("clinical_history_processed_prefix", lang) + summary_text
                    ).strip(),
                }
            )
            st.success(t("clinical_history_success", lang))
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
    if "language" not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE
    if "theme_palette" not in st.session_state:
        st.session_state.theme_palette = DEFAULT_PALETTE
    if "messages_ui" not in st.session_state:
        st.session_state.messages_ui = [
            {
                "role": "assistant",
                "content": t("initial_greeting", _current_language()),
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
    """Renderiza el panel lateral: idioma, proveedor LLM, credenciales y ayuda."""
    lang = _current_language()
    with st.sidebar:
        st.segmented_control(
            t("sidebar_language_label", lang),
            options=list(LANGUAGES.keys()),
            format_func=lambda k: LANGUAGES[k],
            key="language",
            help=t("sidebar_language_help", lang),
        )
        # El idioma pudo cambiar en este mismo rerun: relee tras el widget
        # para que el resto de la sidebar (y el resto de main()) ya use el
        # idioma recién elegido en vez del anterior.
        lang = _current_language()

        st.selectbox(
            t("sidebar_palette_label", lang),
            options=list(PALETTES.keys()),
            format_func=lambda k: PALETTES[k]["name"],
            key="theme_palette",
            help=t("sidebar_palette_help", lang),
        )
        st.divider()

        st.header(t("sidebar_config_header", lang))

        provider = st.selectbox(
            t("provider_label", lang),
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
            base_url = st.text_input(t("ollama_base_url_label", lang), value=LLM_CONFIG.ollama_base_url)
            model_hint = st.text_input(t("model_optional_label", lang), value=LLM_CONFIG.default_model)
        elif provider == "gemini":
            api_key = st.text_input(t("gemini_api_key_label", lang), value=LLM_CONFIG.gemini_api_key, type="password")
            base_url = st.text_input(
                t("gemini_base_url_label", lang), value=getattr(LLM_CONFIG, "gemini_base_url", "") or ""
            )
            model_hint = st.text_input(t("model_label", lang), value=LLM_CONFIG.gemini_model)
        elif provider == "openai":
            api_key = st.text_input(t("openai_api_key_label", lang), value=LLM_CONFIG.openai_api_key, type="password")
            base_url = st.text_input(t("openai_base_url_label", lang), value=LLM_CONFIG.openai_base_url or "")
            model_hint = st.text_input(t("model_label", lang), value=LLM_CONFIG.openai_model)
        elif provider == "claude":
            api_key = st.text_input(t("claude_api_key_label", lang), value=LLM_CONFIG.claude_api_key, type="password")
            base_url = st.text_input(t("claude_base_url_label", lang), value=LLM_CONFIG.claude_base_url or "")
            model_hint = st.text_input(t("model_label", lang), value=LLM_CONFIG.claude_model)
        else:  # groq
            api_key = st.text_input(
                t("groq_api_key_label", lang), value=LLM_CONFIG.groq_api_key, type="password",
                help=t("groq_api_key_help", lang),
            )
            base_url = st.text_input(t("groq_base_url_label", lang), value=LLM_CONFIG.groq_base_url or "")
            model_hint = st.text_input(t("model_label", lang), value=LLM_CONFIG.groq_model)

        # NOTA: LLM_CONFIG es un singleton a nivel de módulo, compartido por
        # todas las sesiones de Streamlit del mismo proceso. Estos widgets
        # (y "Aplicar configuración") cambian la configuración para TODOS
        # los usuarios conectados, no solo para quien mueve el slider. Para
        # un despliegue multiusuario real habría que mover esta config a
        # st.session_state y pasarla explícitamente a ProviderFactory.
        LLM_CONFIG.temperature = st.slider(
            t("temperature_label", lang),
            min_value=0.0, max_value=1.0, value=LLM_CONFIG.temperature, step=0.1,
            help=t("temperature_help", lang),
        )
        LLM_CONFIG.max_tokens = st.number_input(
            t("max_tokens_label", lang), min_value=64, max_value=4096, value=LLM_CONFIG.max_tokens, step=64
        )

        discover = st.button(t("discover_button", lang))
        if discover:
            try:
                with st.spinner(t("discover_spinner", lang)):
                    models = discover_models_cached(provider, api_key, base_url)
                st.session_state.discovered_models = models
                if not models and model_hint:
                    st.warning(t("discover_warning", lang))
            except Exception as e:
                st.error(t("discover_error", lang, error=e))

        # Persisted across reruns, unlike the original which vanished
        # as soon as `discover` went back to False on the next script run.
        if st.session_state.discovered_models:
            chosen = st.selectbox(t("models_detected_label", lang), st.session_state.discovered_models, key="chosen_model")
            if st.button(t("use_model_button", lang)):
                _apply_llm_settings(provider, api_key, base_url, chosen, lang)
        elif model_hint:
            if st.button(t("apply_config_button", lang)):
                _apply_llm_settings(provider, api_key, base_url, model_hint, lang)

        with st.expander(t("save_credentials_expander", lang)):
            _render_credential_persistence(provider, api_key, base_url, lang)

        st.divider()
        with st.expander(t("clinical_history_expander", lang), expanded=False):
            render_clinical_history_section()

        st.divider()
        with st.expander(t("help_expander", lang)):
            st.markdown(t("help_content", lang))

        st.divider()
        if st.button(t("new_conversation_button", lang), width="stretch"):
            reset_conversation()


def _apply_llm_settings(provider: str, api_key: str, base_url: str, model: str, lang: str = DEFAULT_LANGUAGE) -> None:
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
    elif provider == "groq":
        LLM_CONFIG.groq_model = model
    st.success(t("model_applied_success", lang, model=model))
    st.rerun()


def _render_credential_persistence(provider: str, api_key: str, base_url: str, lang: str = DEFAULT_LANGUAGE) -> None:
    """Plaintext .env save, plus optional encrypted save/load if `cryptography` is installed."""
    if st.button(t("save_env_button", lang)):
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
            st.success(t("save_env_success", lang, path=env_path))
        except Exception as e:
            st.error(t("save_env_error", lang, error=e))

    try:
        from chatbot.utils.crypto import HAS_CRYPTO, encrypt_dict, decrypt_file
    except Exception:
        HAS_CRYPTO = False

    if not HAS_CRYPTO:
        st.caption(t("crypto_missing_caption", lang))
        return

    passphrase = st.text_input(t("passphrase_label", lang), type="password", key="enc_pass")
    passphrase2 = st.text_input(t("passphrase_confirm_label", lang), type="password", key="enc_pass2")
    if st.button(t("encrypt_save_button", lang)):
        if not passphrase or passphrase != passphrase2:
            st.error(t("passphrase_mismatch_error", lang))
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
                st.success(t("encrypt_success", lang, path=env_enc_path))
            except Exception as e:
                st.error(t("encrypt_error", lang, error=e))

    enc_path = Path(__file__).resolve().parent / ".env.enc"
    if enc_path.exists():
        dec_pass = st.text_input(t("decrypt_passphrase_label", lang), type="password", key="dec_pass")
        if st.button(t("load_env_enc_button", lang)):
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
                st.success(t("load_env_enc_success", lang))
                st.rerun()
            except Exception as e:
                st.error(t("decrypt_error", lang, error=e))


# --------------------------------------------------------------------------- #
# Patient status panel
# --------------------------------------------------------------------------- #

# Icono para cada campo de PatientState; la etiqueta se traduce vía i18n
# (claves "field_<key>" en chatbot/i18n.py). Un campo que no tenga icono
# aquí, o traducción, sigue funcionando: cae en el fallback de _field_meta().
FIELD_ICONS: Dict[str, str] = {
    "age": "🎂",
    "gender_m": "🚻",
    "hypertension": "❤️",
    "diabetes": "🩸",
    "alcoholism": "🍷",
    "handicap": "♿",
    "scholarship": "🎓",
    "sms_received": "📩",
    "history_no_show": "📊",
    "days_between": "📅",
    "weekend": "🗓️",
    "time_of_day": "⏰",
    "consultation_reason": "📝",
}

_BOOLEAN_FIELDS = {
    "hypertension", "diabetes", "alcoholism", "handicap", "scholarship", "sms_received", "weekend",
}


def _field_meta(key: str, lang: str) -> Dict[str, str]:
    icon = FIELD_ICONS.get(key, "•")
    label_key = f"field_{key}"
    label = t(label_key, lang)
    if label == label_key:  # sin traducción para este campo
        label = key.replace("_", " ").capitalize()
    return {"icon": icon, "label": label}


def _humanize_value(key: str, value: Any, lang: str) -> str:
    """Convierte el valor crudo del estado en texto legible para humanos."""
    if key == "gender_m":
        return t("value_male", lang) if value in (1, "1", 1.0, True) else t("value_female", lang)
    if key == "age":
        return t("value_age_suffix", lang, value=value)
    if key == "days_between":
        return t("value_days_suffix", lang, value=value)
    if key == "history_no_show":
        try:
            return f"{float(value):.0%}"
        except (TypeError, ValueError):
            return str(value)
    if key in _BOOLEAN_FIELDS:
        return t("value_yes", lang) if value in (1, "1", 1.0, True) else t("value_no", lang)
    return str(value)


def _confidence_class(conf: float) -> str:
    if conf >= 0.8:
        return "high"
    if conf >= 0.5:
        return "medium"
    return "low"


def render_patient_status(state_dump: Dict[str, Any], missing_fields: list) -> None:
    lang = _current_language()
    st.subheader(t("patient_status_subheader", lang))

    if not state_dump:
        st.info(t("patient_status_empty_info", lang))
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
            <span>{t("progress_label", lang)}</span>
            <span>{t("progress_fields_count", lang, completed=completed, total=total_fields, pct=f"{pct:.0%}")}</span>
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
            meta = _field_meta(key, lang)
            conf_class = _confidence_class(conf)
            # html.escape es imprescindible aquí: "value" puede venir de texto
            # que el paciente escribió en el chat y que el LLM copió tal cual
            # a "extracted_data" (p.ej. consultation_reason). Como esto se
            # renderiza con unsafe_allow_html=True, sin escapar sería una
            # inyección de HTML/JS trivial (XSS) a través del propio chat.
            safe_label = html.escape(meta["label"])
            safe_value = html.escape(_humanize_value(key, value, lang))
            conf_label = html.escape(t("confidence_label", lang, pct=f"{conf:.0%}"))
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
                    <span class="pc-conf-pct">{conf_label}</span>
                  </div>
                </div>
                """
            )
        st.markdown("".join(cards), unsafe_allow_html=True)
    else:
        st.caption(t("no_confirmed_fields_caption", lang))

    st.divider()
    st.subheader(t("pending_variables_subheader", lang))
    if missing_fields:
        chips = []
        for field in missing_fields:
            meta = _field_meta(field, lang)
            chips.append(f'<span class="pending-chip">{meta["icon"]} {html.escape(meta["label"])}</span>')
        st.markdown(f'<div class="pending-wrap">{"".join(chips)}</div>', unsafe_allow_html=True)
    else:
        st.success(t("pending_variables_done_success", lang))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def render_chatbot_tab() -> None:
    """Renders the full chatbot experience: CSS, sidebar config, chat and the
    patient status/prediction panel. Does NOT call configure_page() — the
    host script is responsible for page config (main() does it below for
    standalone runs; app_unificado.py does it once for the whole portal)."""
    inject_css()
    initialize_session()
    inject_theme_css(st.session_state.theme_palette)
    lang = _current_language()

    st.title(t("app_title", lang))
    st.caption(t("app_caption", lang))

    render_sidebar()
    lang = _current_language()  # el usuario pudo cambiarlo dentro de la sidebar

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(t("chat_subheader", lang))

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

        user_text = st.chat_input(t("chat_input_placeholder", lang))

        if user_text and user_text.strip():
            user_text = user_text.strip()
            if len(user_text) > MAX_USER_MESSAGE_CHARS:
                st.warning(t("chat_truncate_warning", lang, max=MAX_USER_MESSAGE_CHARS))
                user_text = user_text[:MAX_USER_MESSAGE_CHARS]
            st.session_state.messages_ui.append({"role": "user", "content": user_text})
            manager = get_manager()

            if manager is not None:
                try:
                    with st.spinner(t("chat_thinking_spinner", lang)):
                        reply, ready = manager.process_user_input(user_text, lang)
                    st.session_state.messages_ui.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.session_state.messages_ui.append(
                        {"role": "assistant", "content": t("chat_error_message", lang, error=e)}
                    )
                    st.session_state.last_error = str(e)
            st.rerun()

    with col2:
        st.subheader(t("patient_prediction_subheader", lang))
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
                    risk_label = t(f"risk_{pred.risk_level}", lang)
                    st.markdown(
                        f"<div class='risk-badge {color_class}'>"
                        f"{t('risk_badge_text', lang, probability=f'{pred.probability:.2%}', risk_level=risk_label)}</div>",
                        unsafe_allow_html=True,
                    )
                    if pred.is_fallback:
                        st.caption(t("risk_fallback_caption", lang))
            except Exception as e:
                st.error(t("patient_prediction_error", lang, error=e))


def main() -> None:
    """Entry point for standalone execution: `streamlit run chatbot/app.py`."""
    configure_page()
    render_chatbot_tab()


if __name__ == "__main__":
    main()