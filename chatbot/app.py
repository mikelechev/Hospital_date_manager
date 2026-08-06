"""
Módulo de la Interfaz de Usuario utilizando Streamlit.
Actúa exclusivamente como capa de presentación, delegando toda la lógica
al ConversationManager y al Predictor.
"""

import logging
import streamlit as st
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

from config import LLM_CONFIG, APP_CONFIG, PREDICT_CONFIG
from providers.provider_factory import ProviderFactory
from conversation.conversation_manager import ConversationManager
from prediction.predictor import Predictor
from exports import (
    export_conversation_json,
    export_conversation_csv,
    export_full_data_csv,
    save_clinical_history,
)

# Configuración inicial de la página de Streamlit
st.set_page_config(
    page_title="Asistente de Admisión Hospitalaria",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)

def initialize_session() -> None:
    """
    Inicializa los objetos principales en la sesión de Streamlit si no existen.
    Garantiza la persistencia del estado entre recargas de la UI[cite: 1].
    """
    if "manager" not in st.session_state:
        logger.info("Inicializando nueva sesión de ConversationManager.")
        try:
            provider = ProviderFactory.get_provider()
            st.session_state.manager = ConversationManager(provider)
        except Exception as e:
            st.error(f"Error de configuración del proveedor LLM: {str(e)}")
            st.stop()
    else:
        # Si el manager está almacenado en sesión pero no define el método recién agregado,
        # recreamos la instancia para evitar errores en caliente.
        if not hasattr(st.session_state.manager, "process_clinical_history"):
            logger.warning("Recreando ConversationManager porque falta process_clinical_history.")
            try:
                provider = ProviderFactory.get_provider()
                st.session_state.manager = ConversationManager(provider)
            except Exception as e:
                st.error(f"Error de configuración del proveedor LLM: {str(e)}")
                st.stop()
            
    if "predictor" not in st.session_state:
        st.session_state.predictor = Predictor()
        
    if "messages_ui" not in st.session_state:
        # Mensaje de bienvenida inicial
        st.session_state.messages_ui = [
            {"role": "assistant", "content": "Hola. Soy el asistente virtual del hospital. ¿En qué te puedo ayudar hoy?"}
        ]
    if "clinical_history_path" not in st.session_state:
        st.session_state.clinical_history_path = None

def reset_conversation() -> None:
    """Borra el estado actual para iniciar un nuevo flujo conversacional."""
    logger.info("Reiniciando la conversación a petición del usuario.")
    for key in ["manager", "predictor", "messages_ui", "clinical_history_path"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def render_sidebar() -> None:
    """
    Renderiza el panel lateral con las configuraciones técnicas.
    Estos valores modifican en caliente la configuración global inyectada.
    """
    with st.sidebar:
        st.header("⚙️ Configuración del Sistema")
        
        provider_choice = st.selectbox(
            "Proveedor LLM",
            options=["Ollama", "Gemini"],
            index=0 if LLM_CONFIG.default_provider == "ollama" else 1
        )
        LLM_CONFIG.default_provider = provider_choice.lower()
        
        LLM_CONFIG.temperature = st.slider(
            "Temperatura (Creatividad vs Precisión)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.1,
            help="Mantenlo en 0.0 para maximizar la consistencia del JSON."
        )
        
        LLM_CONFIG.max_tokens = st.number_input(
            "Max Tokens", min_value=256, max_value=4096, value=1500, step=256
        )
        
        st.divider()
        if st.button("🔄 Nueva Conversación", use_container_width=True):
            reset_conversation()

def render_patient_status(state_dump: Dict[str, Any], missing_fields: list[str]) -> None:
    """
    Renderiza la tabla de estado del paciente en el panel lateral derecho[cite: 4].
    """
    st.subheader("📋 Estado del Paciente")
    
    if not state_dump:
        st.info("Aún no se ha recopilado información.")
        return

    # Filtramos solo las variables que tienen un valor asignado
    extracted_data = {k: v for k, v in state_dump.items() if v.get("value") is not None}
    
    if extracted_data:
        for key, data in extracted_data.items():
            value = data["value"]
            conf = data["confidence"]
            color = "green" if conf >= 0.8 else ("orange" if conf >= 0.5 else "red")
            
            st.markdown(
                f"**{key}**: {value} <span style='color:{color}; font-size:0.8em;'>(Confianza: {conf:.2f})</span>", 
                unsafe_allow_html=True
            )
    
    st.divider()
    st.subheader("🎯 Variables Pendientes")
    if missing_fields:
        for field in missing_fields:
            st.markdown(f"- `{field}`")
    else:
        st.success("¡Información completada!")

def main() -> None:
    """Función principal que orquesta la construcción de la interfaz."""
    initialize_session()
    render_sidebar()
    # Top header: title + model/provider status
    header_col, status_col = st.columns([7, 3])
    with header_col:
        st.title("🏥 Asistente de Admisión Hospitalaria")
        st.caption("Interfaz de extracción conversacional y predicción de No-Show")

    model_path = Path(PREDICT_CONFIG.model_path)
    model_present = model_path.exists()
    with status_col:
        st.metric("LLM Provider", LLM_CONFIG.default_provider.capitalize())
        if model_present:
            st.success(f"Modelo: {model_path.name}")
        else:
            st.warning("Modelo ausente — se usará fallback matemático")

    # Dividimos la pantalla: 70% chat, 30% panel de estado
    col_chat, col_status = st.columns([7, 3])
    
    with col_chat:
        st.header("💬 Consulta Virtual")
        
        # Renderizamos el historial de mensajes de la interfaz
        for msg in st.session_state.messages_ui:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Captura de input del usuario
        if prompt := st.chat_input("Escribe tu mensaje aquí..."):
            
            # 1. Mostrar input del usuario
            st.session_state.messages_ui.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # 2. Procesar con el ConversationManager (Lógica aislada)
            with st.chat_message("assistant"):
                with st.spinner("Analizando respuesta..."):
                    manager: ConversationManager = st.session_state.manager
                    reply, is_ready = manager.process_user_input(prompt)
                    
                    st.markdown(reply)
                    st.session_state.messages_ui.append({"role": "assistant", "content": reply})

    with col_status:
        # Obtenemos el estado directamente del manager para mantener una única fuente de verdad[cite: 1, 2]
        manager = st.session_state.manager
        current_state = manager.get_current_state()
        missing = manager.state.get_missing_critical_fields()
        
        render_patient_status(current_state, missing)
        
        # Evaluamos y ejecutamos el predictor si los datos están listos[cite: 4]
        if manager.state.is_ready_for_prediction():
            st.divider()
            st.subheader("⚠️ Predicción de Riesgo (No-Show)")
            with st.spinner("Calculando probabilidad mediante XGBoost..."):
                predictor: Predictor = st.session_state.predictor
                result = predictor.predict(manager.state)
                
                risk_color = "red" if result.risk_level == "ALTO" else ("orange" if result.risk_level == "MEDIO" else "green")
                
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px; background-color: rgba(0,0,0,0.1); border-left: 5px solid {risk_color};'>"
                    f"<h4>Nivel de Riesgo: {result.risk_level}</h4>"
                    f"<p>Probabilidad de ausencia: <strong>{result.probability:.2%}</strong></p>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
                
                if result.is_fallback:
                    st.caption("Nota: Se utilizó el modelo matemático de respaldo.")

        # Export controls
        st.divider()
        st.subheader("💾 Export Conversation")
        fmt = st.selectbox("Format", ("JSON", "CSV"))
        anonymize = st.checkbox("Anonymize (redact emails/phones/IDs)", value=True)

        # Clinical history manager: upload or paste
        st.markdown("**📎 Clinical History**")
        uploaded = st.file_uploader("Upload clinical history (PDF / TXT)", type=["pdf", "txt", "doc", "docx"], key="ch_upload")
        ch_text = st.text_area("Or paste clinical history text (optional)", height=120, key="ch_text")

        # Show current saved clinical history
        current_ch = st.session_state.get("clinical_history_path")
        if current_ch:
            st.info(f"Saved clinical history: {Path(current_ch).name}")

        col_btn, col_hist = st.columns([1, 2])
        with col_btn:
            if st.button("Create export", key="create_export"):
                try:
                    messages = st.session_state.get("messages_ui", [])
                    metadata = {
                        "provider": LLM_CONFIG.default_provider,
                        "exported_by": "streamlit_ui",
                    }
                    if fmt == "JSON":
                        out_path = export_conversation_json(messages, patient_state=current_state, metadata=metadata, anonymize=anonymize)
                        mime = "application/json"
                    else:
                        out_path = export_conversation_csv(messages, anonymize=anonymize)
                        mime = "text/csv"

                    with open(out_path, "rb") as fh:
                        data = fh.read()

                    st.success(f"Export creado: {out_path.name}")
                    st.download_button("Download export", data=data, file_name=out_path.name, mime=mime, key=f"dl_{out_path.name}")
                except Exception as e:
                    st.error(f"Error creando export: {e}")

            # Save clinical history if provided
            if st.button("Save clinical history", key="save_ch"):
                try:
                    if uploaded is not None:
                        # read bytes
                        content = uploaded.read()
                        outp = save_clinical_history(content, filename=uploaded.name)
                    elif ch_text and ch_text.strip():
                        outp = save_clinical_history(ch_text, filename=f"clinical_history_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.txt")
                    else:
                        st.warning("No clinical history provided to save.")
                        outp = None

                    if outp:
                        st.session_state.clinical_history_path = str(outp)
                        st.success(f"Clinical history saved: {outp.name}")
                except Exception as e:
                    st.error(f"Error saving clinical history: {e}")

            # Interpret clinical history (extract variables)
            if st.button("Interpret clinical history", key="interpret_ch"):
                try:
                    # choose source: uploaded > saved file > pasted text
                    raw_text = None
                    if uploaded is not None:
                        b = uploaded.read()
                        try:
                            raw_text = b.decode("utf-8")
                        except Exception:
                            raw_text = str(b)
                    elif st.session_state.get("clinical_history_path"):
                        p = Path(st.session_state.get("clinical_history_path"))
                        try:
                            raw_text = p.read_text(encoding="utf-8")
                        except Exception:
                            raw_text = None
                    elif ch_text and ch_text.strip():
                        raw_text = ch_text

                    if not raw_text:
                        st.warning("No clinical history text available to interpret.")
                    else:
                        summary = manager.process_clinical_history(raw_text)
                        if summary.get("error"):
                            st.error(f"Error interpretando clinical history: {summary['error']}")
                        else:
                            assistant_response = summary.get("assistant_response")
                            analysis = summary.get("analysis", {})
                            extracted = analysis.get("extracted_data", {})
                            confidences = analysis.get("confidence", {})

                            if assistant_response:
                                st.markdown(f"**LLM response:** {assistant_response}")

                            if extracted:
                                st.success(f"Extracted {len(extracted)} fields from clinical history.")
                                for field_name, value in extracted.items():
                                    confidence = confidences.get(field_name, 0.0)
                                    st.markdown(f"- **{field_name}**: {value} (Conf: {confidence:.2f})")
                            else:
                                st.info("No structured fields were extracted from this clinical history.")
                except Exception as e:
                    st.error(f"Error interpretando clinical history: {e}")

            # Full data export (CSV)
            if st.button("Export full dataset (CSV)", key="export_full"):
                try:
                    messages = st.session_state.get("messages_ui", [])
                    metadata = {"provider": LLM_CONFIG.default_provider, "exported_by": "streamlit_ui"}
                    ch_path = Path(st.session_state.get("clinical_history_path")) if st.session_state.get("clinical_history_path") else None
                    out_path = export_full_data_csv(messages, patient_state=current_state, metadata=metadata, clinical_history_path=ch_path, anonymize=anonymize)
                    b = out_path.read_bytes()
                    st.success(f"Full export creado: {out_path.name}")
                    st.download_button("Download full export", data=b, file_name=out_path.name, mime="text/csv", key=f"dl_full_{out_path.name}")
                except Exception as e:
                    st.error(f"Error creando full export: {e}")

        # Export history: show recent files in exports/
        with col_hist:
            exports_dir = Path(__file__).resolve().parent / "exports"
            st.markdown("**Recent exports**")
            if exports_dir.exists():
                files = sorted(exports_dir.glob("conversation_*.*"), key=lambda p: p.stat().st_mtime, reverse=True)[:6]
                if files:
                    for f in files:
                        try:
                            b = f.read_bytes()
                            st.download_button(f.name, data=b, file_name=f.name, mime=("application/json" if f.suffix==".json" else "text/csv"), key=f"hist_{f.name}")
                        except Exception:
                            st.write(f.name)
                else:
                    st.write("No exports yet.")
            else:
                st.write("No exports directory.")

if __name__ == "__main__":
    main()