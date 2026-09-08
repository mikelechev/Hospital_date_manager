"""
Módulo de la Interfaz de Usuario utilizando Streamlit.
Actúa exclusivamente como capa de presentación, delegando toda la lógica
al ConversationManager y al Predictor.
"""

import csv
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from chatbot.config import LLM_CONFIG, APP_CONFIG, PREDICT_CONFIG
from chatbot.providers.provider_factory import ProviderFactory
from chatbot.conversation.conversation_manager import ConversationManager
from chatbot.prediction.predictor import Predictor

# --- BYPASS HACKATHON: Comentamos esto porque falta el archivo en Git ---
# from exports import (
#     export_conversation_json,
#     export_conversation_csv,
#     export_full_data_csv,
#     save_clinical_history,
# )
# ------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Page config + CSS (done once, at import time, not re-injected every rerun)
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Asistente de Admisión Hospitalaria",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

PROVIDERS = ["ollama", "gemini", "openai", "claude", "custom"]


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
        getattr(LLM_CONFIG, "openai_api_key", None),
        getattr(LLM_CONFIG, "ollama_base_url", None),
        getattr(LLM_CONFIG, "gemini_base_url", None),
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
    if api_key:
        LLM_CONFIG.gemini_api_key = api_key
        LLM_CONFIG.openai_api_key = api_key
    if base_url:
        LLM_CONFIG.gemini_base_url = base_url
        LLM_CONFIG.ollama_base_url = base_url
    provider_inst = ProviderFactory.get_provider()
    return provider_inst.list_models() or []


def get_predictor() -> Predictor:
    if "predictor" not in st.session_state:
        st.session_state.predictor = Predictor()
    return st.session_state.predictor


# --------------------------------------------------------------------------- #
# CSV export of collected patient data
# --------------------------------------------------------------------------- #

DATA_DIR = Path(__file__).resolve().parent / "data"
FULL_DATA_CSV_PATH = DATA_DIR / "historial_pacientes.csv"


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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds")}
    for key, data in state_dump.items():
        if data.get("value") is not None:
            row[key] = data["value"]

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

def extract_text_from_upload(uploaded_file) -> str:
    """Extract raw text from an uploaded .txt, .pdf or .docx clinical record."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

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
        st.header("⚙️ Configuración del Sistema")

        provider = st.selectbox(
            "Proveedor",
            PROVIDERS,
            index=PROVIDERS.index(LLM_CONFIG.default_provider) if LLM_CONFIG.default_provider in PROVIDERS else 0,
            key="provider_select",
        )
        api_key = st.text_input(
            "API Key (si aplica)",
            value=LLM_CONFIG.gemini_api_key or LLM_CONFIG.openai_api_key or "",
            type="password",
        )
        base_url = st.text_input(
            "Base URL / Endpoint (si aplica)",
            value=getattr(LLM_CONFIG, "gemini_base_url", "") or LLM_CONFIG.ollama_base_url or "",
        )
        model_hint = st.text_input("Modelo (opcional)", value=LLM_CONFIG.default_model)

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
                "- Pega tu API key si usas Gemini / OpenAI.\n"
                "- Usa 'Buscar modelos' para detectar modelos disponibles."
            )

        st.divider()
        if st.button("🔄 Nueva Conversación", use_container_width=True):
            reset_conversation()


def _apply_llm_settings(provider: str, api_key: str, base_url: str, model: str) -> None:
    """Apply chosen provider/model settings and invalidate the cached manager."""
    LLM_CONFIG.default_provider = provider
    LLM_CONFIG.default_model = model
    if api_key:
        LLM_CONFIG.gemini_api_key = api_key
        LLM_CONFIG.openai_api_key = api_key
    if base_url:
        LLM_CONFIG.gemini_base_url = base_url
        LLM_CONFIG.ollama_base_url = base_url
    if provider == "gemini":
        LLM_CONFIG.gemini_model = model
    if provider == "openai":
        LLM_CONFIG.openai_model = model
    st.success(f"Modelo aplicado: {model}")
    st.rerun()


def _render_credential_persistence(provider: str, api_key: str, base_url: str) -> None:
    """Plaintext .env save, plus optional encrypted save/load if `cryptography` is installed."""
    if st.button("Guardar en chatbot/.env"):
        try:
            env_path = Path(__file__).resolve().parent / ".env"
            lines = []
            if api_key:
                lines.append(f"GEMINI_API_KEY={api_key}")
                lines.append(f"OPENAI_API_KEY={api_key}")
            if base_url:
                lines.append(f"OLLAMA_BASE_URL={base_url}")
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
            data = {}
            if api_key:
                data["GEMINI_API_KEY"] = api_key
                data["OPENAI_API_KEY"] = api_key
            if base_url:
                data["OLLAMA_BASE_URL"] = base_url
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
                if "GEMINI_API_KEY" in data:
                    LLM_CONFIG.gemini_api_key = data["GEMINI_API_KEY"]
                    LLM_CONFIG.openai_api_key = data["GEMINI_API_KEY"]
                if "OPENAI_API_KEY" in data:
                    LLM_CONFIG.openai_api_key = data["OPENAI_API_KEY"]
                if "OLLAMA_BASE_URL" in data:
                    LLM_CONFIG.ollama_base_url = data["OLLAMA_BASE_URL"]
                if "DEFAULT_MODEL" in data:
                    LLM_CONFIG.default_model = data["DEFAULT_MODEL"]
                st.success("Credenciales cargadas en la configuración de sesión")
                st.rerun()
            except Exception as e:
                st.error(f"Fallo al desencriptar: {e}")


# --------------------------------------------------------------------------- #
# Patient status panel
# --------------------------------------------------------------------------- #

def render_patient_status(state_dump: Dict[str, Any], missing_fields: list) -> None:
    st.subheader("📋 Estado del Paciente")

    if not state_dump:
        st.info("Aún no se ha recopilado información.")
        return

    extracted_data = {k: v for k, v in state_dump.items() if v.get("value") is not None}

    if extracted_data:
        # Single markdown write instead of one st.markdown call per field —
        # cheaper to render and avoids N separate DOM nodes.
        rows = []
        for key, data in extracted_data.items():
            value = data["value"]
            conf = data["confidence"]
            color = "green" if conf >= 0.8 else ("orange" if conf >= 0.5 else "red")
            rows.append(
                f"**{key}**: {value} "
                f"<span style='color:{color}; font-size:0.8em;'>(Confianza: {conf:.2f})</span>"
            )
        st.markdown("<br/>".join(rows), unsafe_allow_html=True)

    st.divider()
    st.subheader("🎯 Variables Pendientes")
    if missing_fields:
        st.markdown("\n".join(f"- `{field}`" for field in missing_fields))
    else:
        st.success("¡Información completada!")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    initialize_session()

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