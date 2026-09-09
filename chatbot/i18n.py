"""
Módulo de internacionalización (i18n) de la interfaz.
Soporta Castellano (es) y Euskera (eu). El español es el idioma de referencia:
si falta una clave en euskera, se cae automáticamente al texto en español en
vez de desaparecer silenciosamente.
"""

from typing import Dict

DEFAULT_LANGUAGE = "es"

LANGUAGES: Dict[str, str] = {
    "es": "🇪🇸 Castellano",
    "eu": "🌐 Euskera",
}

# LLM_LANGUAGE_INSTRUCTIONS se usa en prompt_builder para pedirle al modelo
# que responda en el idioma elegido por el usuario en la UI.
LLM_LANGUAGE_INSTRUCTIONS: Dict[str, str] = {
    "es": "Responde siempre en español (castellano), tanto en \"assistant_response\" "
          "como en cualquier otro campo de texto libre.",
    "eu": "Erantzun beti euskaraz (euskara batuan), bai \"assistant_response\" eremuan "
          "bai testu librezko beste edozein eremutan.",
}

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "es": {
        "page_title": "Asistente de Admisión Hospitalaria",
        "app_title": "🏥 Asistente de Admisión Hospitalaria — Chatbot",
        "app_caption": "Interfaz para conversar con el asistente y ver el estado del paciente en tiempo real.",

        "sidebar_language_label": "🌐 Idioma",
        "sidebar_language_help": "Cambia el idioma de la interfaz y de las respuestas del asistente.",

        "sidebar_palette_label": "🎨 Paleta de color",
        "sidebar_palette_help": "Cambia el aspecto visual de la app al instante (no afecta a los datos del paciente).",

        "sidebar_config_header": "⚙️ Configuración del Sistema",
        "provider_config_error": "Error de configuración del proveedor LLM: {error}",
        "provider_label": "Proveedor",
        "ollama_base_url_label": "Base URL de Ollama",
        "model_optional_label": "Modelo (opcional)",
        "model_label": "Modelo",
        "gemini_api_key_label": "Gemini API Key",
        "gemini_base_url_label": "Gemini Base URL (opcional)",
        "openai_api_key_label": "OpenAI API Key",
        "openai_base_url_label": "OpenAI Base URL (opcional)",
        "claude_api_key_label": "Claude API Key",
        "claude_base_url_label": "Claude Base URL (opcional)",
        "groq_api_key_label": "Groq API Key",
        "groq_api_key_help": "Gratis en https://console.groq.com",
        "groq_base_url_label": "Groq Base URL (opcional)",
        "temperature_label": "Temperatura (Creatividad vs Precisión)",
        "temperature_help": "Mantenlo en 0.0 para maximizar la consistencia del JSON.",
        "max_tokens_label": "Max Tokens",
        "discover_button": "🔎 Buscar modelos disponibles",
        "discover_spinner": "Buscando modelos...",
        "discover_warning": "No se pudieron listar modelos automáticamente; se usará el nombre indicado.",
        "discover_error": "Error inicializando proveedor: {error}",
        "models_detected_label": "Modelos detectados",
        "use_model_button": "Usar este modelo",
        "apply_config_button": "Aplicar configuración",
        "model_applied_success": "Modelo aplicado: {model}",
        "save_credentials_expander": "💾 Guardar credenciales",
        "clinical_history_expander": "📁 Ficha de Historial Clínico",
        "help_expander": "Ayuda rápida",
        "help_content": (
            "**Sugerencias de prompts:**\n"
            "- 'Hola, necesito ayuda para una cita'\n"
            "- 'Tengo dolor de cabeza y fiebre desde ayer'\n"
            "- '¿Qué documentos necesito llevar?'\n\n"
            "**Consejos:**\n"
            "- Pega tu API key si usas Gemini / OpenAI / Claude / Groq.\n"
            "- Usa 'Buscar modelos' para detectar modelos disponibles."
        ),
        "new_conversation_button": "🔄 Nueva Conversación",

        "save_env_button": "Guardar en chatbot/.env",
        "save_env_success": "Credenciales guardadas en {path}",
        "save_env_error": "No se pudo guardar .env: {error}",
        "crypto_missing_caption": "Instala 'cryptography' (pip install cryptography) para guardar credenciales encriptadas.",
        "passphrase_label": "Passphrase para encriptar",
        "passphrase_confirm_label": "Confirmar passphrase",
        "encrypt_save_button": "Encriptar y guardar .env.enc",
        "passphrase_mismatch_error": "Las passphrases no coinciden o están vacías.",
        "encrypt_success": "Credenciales encriptadas guardadas en {path}",
        "encrypt_error": "Fallo al encriptar: {error}",
        "decrypt_passphrase_label": "Passphrase para desencriptar",
        "load_env_enc_button": "Cargar .env.enc",
        "load_env_enc_success": "Credenciales cargadas en la configuración de sesión",
        "decrypt_error": "Fallo al desencriptar: {error}",

        "clinical_history_subheader": "📁 Ficha de Historial Clínico",
        "clinical_history_caption": "Sube o pega el historial clínico del paciente para que el asistente extraiga los datos automáticamente.",
        "clinical_history_upload_label": "Subir ficha (.txt, .pdf, .docx)",
        "clinical_history_paste_label": "...o pega el texto del historial aquí",
        "clinical_history_process_button": "🧠 Procesar historial con el LLM",
        "clinical_history_warning_empty": "Sube un archivo o pega el texto del historial primero.",
        "clinical_history_not_implemented_error": (
            "ConversationManager no implementa todavía 'process_clinical_history(texto)'. "
            "Añade ese método (debe leer el texto del historial, extraer los campos clínicos "
            "relevantes con el LLM, actualizar `manager.state` igual que process_user_input, "
            "y devolver un resumen en texto de lo extraído) para habilitar esta función."
        ),
        "clinical_history_spinner": "El LLM está leyendo el historial clínico...",
        "clinical_history_error": "Error procesando el historial clínico: {error}",
        "clinical_history_processed_prefix": "He leído la ficha de historial clínico y actualizado los datos del paciente.\n\n",
        "clinical_history_process_error_generic": "No se pudo procesar el historial clínico. Revisa los logs para más detalle.",
        "clinical_history_success": "Historial clínico procesado y datos actualizados.",
        "clinical_history_path_pasted": "texto pegado",
        "upload_too_large_error": "El archivo supera el tamaño máximo permitido ({max_mb} MB).",
        "pdf_missing_lib_error": "Para leer PDFs instala 'pypdf' (pip install pypdf).",
        "pdf_read_error": "No se pudo leer el PDF: {error}",
        "docx_missing_lib_error": "Para leer .docx instala 'python-docx' (pip install python-docx).",
        "docx_read_error": "No se pudo leer el .docx: {error}",
        "unsupported_format_error": "Formato no soportado. Usa .txt, .pdf o .docx.",

        "export_subheader": "📥 Exportar datos",
        "export_no_data_caption": "Aún no hay datos recabados para exportar.",
        "export_download_button": "⬇️ Descargar CSV de esta ficha",
        "export_add_history_button": "💾 Añadir al historial acumulado (CSV)",
        "export_add_history_success": "Ficha añadida a {path}",
        "export_add_history_error": "No se pudo guardar en el historial acumulado: {error}",

        "patient_status_subheader": "📋 Estado del Paciente",
        "patient_status_empty_info": "Aún no se ha recopilado información.",
        "progress_label": "Progreso de la ficha",
        "progress_fields_count": "{completed}/{total} campos · {pct}",
        "pending_variables_subheader": "🎯 Variables Pendientes",
        "pending_variables_done_success": "¡Información completada!",
        "no_confirmed_fields_caption": "Todavía no hay campos confirmados con datos.",
        "confidence_label": "Confianza: {pct}",

        "chat_subheader": "Chat",
        "chat_input_placeholder": "Escribe tu mensaje aquí...",
        "chat_truncate_warning": "Tu mensaje superaba los {max} caracteres; se ha truncado.",
        "chat_thinking_spinner": "El asistente está escribiendo...",
        "chat_error_message": "⚠️ Ocurrió un error: {error}",

        "patient_prediction_subheader": "Estado paciente & Predicción",
        "risk_badge_text": "Probabilidad de ausencia: {probability} — {risk_level}",
        "risk_fallback_caption": "(Fallback usado — modelo ausente)",
        "patient_prediction_error": "Error mostrando estado/predicción: {error}",

        "initial_greeting": "Hola. Soy el asistente virtual del hospital. ¿En qué te puedo ayudar hoy?",
        "generic_error_response": "Lo siento, ha ocurrido un error técnico interno procesando su solicitud.",

        "field_age": "Edad",
        "field_gender_m": "Sexo",
        "field_hypertension": "Hipertensión",
        "field_diabetes": "Diabetes",
        "field_alcoholism": "Alcoholismo",
        "field_handicap": "Discapacidad",
        "field_scholarship": "Beca social",
        "field_sms_received": "SMS recibido",
        "field_history_no_show": "Historial de faltas",
        "field_days_between": "Días de antelación",
        "field_weekend": "Cita en fin de semana",
        "field_time_of_day": "Horario de la cita",
        "field_consultation_reason": "Motivo de consulta",

        "value_male": "Masculino",
        "value_female": "Femenino",
        "value_yes": "Sí",
        "value_no": "No",
        "value_age_suffix": "{value} años",
        "value_days_suffix": "{value} días",

        "risk_ALTO": "ALTO",
        "risk_MEDIO": "MEDIO",
        "risk_BAJO": "BAJO",
    },
    "eu": {
        "page_title": "Ospitaleko Onarpen Laguntzailea",
        "app_title": "🏥 Ospitaleko Onarpen Laguntzailea — Txatbota",
        "app_caption": "Laguntzailearekin hitz egiteko eta pazientearen egoera denbora errealean ikusteko interfazea.",

        "sidebar_language_label": "🌐 Hizkuntza",
        "sidebar_language_help": "Aldatu interfazearen eta laguntzailearen erantzunen hizkuntza.",

        "sidebar_palette_label": "🎨 Kolore-paleta",
        "sidebar_palette_help": "Aplikazioaren itxura berehala aldatzen du (ez die eragiten pazientearen datuei).",

        "sidebar_config_header": "⚙️ Sistemaren Konfigurazioa",
        "provider_config_error": "Errorea LLM hornitzailearen konfigurazioan: {error}",
        "provider_label": "Hornitzailea",
        "ollama_base_url_label": "Ollama-ren Base URL-a",
        "model_optional_label": "Modeloa (aukerakoa)",
        "model_label": "Modeloa",
        "gemini_api_key_label": "Gemini API Key-a",
        "gemini_base_url_label": "Gemini Base URL-a (aukerakoa)",
        "openai_api_key_label": "OpenAI API Key-a",
        "openai_base_url_label": "OpenAI Base URL-a (aukerakoa)",
        "claude_api_key_label": "Claude API Key-a",
        "claude_base_url_label": "Claude Base URL-a (aukerakoa)",
        "groq_api_key_label": "Groq API Key-a",
        "groq_api_key_help": "Doan hemen: https://console.groq.com",
        "groq_base_url_label": "Groq Base URL-a (aukerakoa)",
        "temperature_label": "Tenperatura (Sormena vs Zehaztasuna)",
        "temperature_help": "Mantendu 0.0an JSONaren koherentzia maximizatzeko.",
        "max_tokens_label": "Gehienezko Token kopurua",
        "discover_button": "🔎 Modelo eskuragarriak bilatu",
        "discover_spinner": "Modeloak bilatzen...",
        "discover_warning": "Ezin izan dira modeloak automatikoki zerrendatu; adierazitako izena erabiliko da.",
        "discover_error": "Errorea hornitzailea hasieratzean: {error}",
        "models_detected_label": "Detektatutako modeloak",
        "use_model_button": "Erabili modelo hau",
        "apply_config_button": "Konfigurazioa aplikatu",
        "model_applied_success": "Modeloa aplikatuta: {model}",
        "save_credentials_expander": "💾 Kredentzialak gorde",
        "clinical_history_expander": "📁 Historial Klinikoaren Fitxa",
        "help_expander": "Laguntza azkarra",
        "help_content": (
            "**Prompt iradokizunak:**\n"
            "- 'Kaixo, hitzordu bat lortzeko laguntza behar dut'\n"
            "- 'Buruko mina eta sukarra dut atzotik'\n"
            "- 'Zer dokumentu eraman behar ditut?'\n\n"
            "**Aholkuak:**\n"
            "- Itsatsi zure API key-a Gemini / OpenAI / Claude / Groq erabiltzen baduzu.\n"
            "- Erabili 'Modeloak bilatu' modelo eskuragarriak detektatzeko."
        ),
        "new_conversation_button": "🔄 Elkarrizketa berria",

        "save_env_button": "Gorde chatbot/.env fitxategian",
        "save_env_success": "Kredentzialak {path}-n gordeta",
        "save_env_error": "Ezin izan da .env gorde: {error}",
        "crypto_missing_caption": "Instalatu 'cryptography' (pip install cryptography) kredentzial enkriptatuak gordetzeko.",
        "passphrase_label": "Enkriptatzeko pasaesaldia",
        "passphrase_confirm_label": "Berretsi pasaesaldia",
        "encrypt_save_button": "Enkriptatu eta gorde .env.enc",
        "passphrase_mismatch_error": "Pasaesaldiak ez datoz bat edo hutsik daude.",
        "encrypt_success": "Kredentzial enkriptatuak {path}-n gordeta",
        "encrypt_error": "Enkriptatzean akatsa: {error}",
        "decrypt_passphrase_label": "Desenkriptatzeko pasaesaldia",
        "load_env_enc_button": "Kargatu .env.enc",
        "load_env_enc_success": "Kredentzialak saioaren konfigurazioan kargatuta",
        "decrypt_error": "Desenkriptatzean akatsa: {error}",

        "clinical_history_subheader": "📁 Historial Klinikoaren Fitxa",
        "clinical_history_caption": "Igo edo itsatsi pazientearen historial klinikoa, laguntzaileak datuak automatikoki atera ditzan.",
        "clinical_history_upload_label": "Igo fitxa (.txt, .pdf, .docx)",
        "clinical_history_paste_label": "...edo itsatsi historialaren testua hemen",
        "clinical_history_process_button": "🧠 Prozesatu historiala LLMarekin",
        "clinical_history_warning_empty": "Igo fitxategi bat edo itsatsi historialaren testua lehenengo.",
        "clinical_history_not_implemented_error": (
            "ConversationManager-ek ez du oraindik 'process_clinical_history(testua)' inplementatzen. "
            "Gehitu metodo hori (historialaren testua irakurri, LLMarekin dagozkion eremu klinikoak "
            "atera, `manager.state` process_user_input-ek egiten duen bezala eguneratu, eta "
            "ateratakoaren laburpen testua itzuli behar du) funtzio hau gaitzeko."
        ),
        "clinical_history_spinner": "LLMa historial klinikoa irakurtzen ari da...",
        "clinical_history_error": "Errorea historial klinikoa prozesatzean: {error}",
        "clinical_history_processed_prefix": "Historial klinikoaren fitxa irakurri dut eta pazientearen datuak eguneratu ditut.\n\n",
        "clinical_history_process_error_generic": "Ezin izan da historial klinikoa prozesatu. Begiratu erregistroak (logs) xehetasun gehiagorako.",
        "clinical_history_success": "Historial klinikoa prozesatuta eta datuak eguneratuta.",
        "clinical_history_path_pasted": "itsatsitako testua",
        "upload_too_large_error": "Fitxategiak baimendutako gehienezko tamaina gainditzen du ({max_mb} MB).",
        "pdf_missing_lib_error": "PDFak irakurtzeko instalatu 'pypdf' (pip install pypdf).",
        "pdf_read_error": "Ezin izan da PDFa irakurri: {error}",
        "docx_missing_lib_error": "'.docx' fitxategiak irakurtzeko instalatu 'python-docx' (pip install python-docx).",
        "docx_read_error": "Ezin izan da .docx-a irakurri: {error}",
        "unsupported_format_error": "Formatua ez da onartzen. Erabili .txt, .pdf edo .docx.",

        "export_subheader": "📥 Datuak esportatu",
        "export_no_data_caption": "Oraindik ez dago esportatzeko bildutako daturik.",
        "export_download_button": "⬇️ Deskargatu fitxa honen CSVa",
        "export_add_history_button": "💾 Gehitu metatutako historialera (CSV)",
        "export_add_history_success": "Fitxa {path}-ra gehituta",
        "export_add_history_error": "Ezin izan da metatutako historialean gorde: {error}",

        "patient_status_subheader": "📋 Pazientearen Egoera",
        "patient_status_empty_info": "Oraindik ez da informaziorik bildu.",
        "progress_label": "Fitxaren aurrerapena",
        "progress_fields_count": "{completed}/{total} eremu · {pct}",
        "pending_variables_subheader": "🎯 Falta diren Aldagaiak",
        "pending_variables_done_success": "Informazioa osatuta!",
        "no_confirmed_fields_caption": "Oraindik ez dago daturekin berretsitako eremurik.",
        "confidence_label": "Konfiantza: {pct}",

        "chat_subheader": "Txata",
        "chat_input_placeholder": "Idatzi zure mezua hemen...",
        "chat_truncate_warning": "Zure mezuak {max} karaktere gainditzen zituen; moztu da.",
        "chat_thinking_spinner": "Laguntzailea idazten ari da...",
        "chat_error_message": "⚠️ Errore bat gertatu da: {error}",

        "patient_prediction_subheader": "Pazientearen egoera & Iragarpena",
        "risk_badge_text": "Ez agertzeko probabilitatea: {probability} — {risk_level}",
        "risk_fallback_caption": "(Fallback erabilita — modeloa falta da)",
        "patient_prediction_error": "Errorea egoera/iragarpena erakustean: {error}",

        "initial_greeting": "Kaixo. Ospitaleko laguntzaile birtuala naiz. Zertan lagundu diezazuket gaur?",
        "generic_error_response": "Sentitzen dut, barne errore tekniko bat gertatu da zure eskaera prozesatzean.",

        "field_age": "Adina",
        "field_gender_m": "Sexua",
        "field_hypertension": "Hipertentsioa",
        "field_diabetes": "Diabetesa",
        "field_alcoholism": "Alkoholismoa",
        "field_handicap": "Desgaitasuna",
        "field_scholarship": "Gizarte-beka",
        "field_sms_received": "SMS jasota",
        "field_history_no_show": "Ez-agertzeen historiala",
        "field_days_between": "Aurrerapen-egunak",
        "field_weekend": "Asteburuko hitzordua",
        "field_time_of_day": "Hitzorduaren ordutegia",
        "field_consultation_reason": "Kontsultaren arrazoia",

        "value_male": "Gizonezkoa",
        "value_female": "Emakumezkoa",
        "value_yes": "Bai",
        "value_no": "Ez",
        "value_age_suffix": "{value} urte",
        "value_days_suffix": "{value} egun",

        "risk_ALTO": "ALTUA",
        "risk_MEDIO": "ERTAINA",
        "risk_BAJO": "BAXUA",
    },
}


def t(key: str, lang: str, **kwargs) -> str:
    """Devuelve el texto traducido de `key` en el idioma `lang`.

    Si la clave no existe en `lang` cae al español; si tampoco existe ahí,
    devuelve la propia clave (evita que un texto desaparezca silenciosamente
    por una traducción olvidada).
    """
    table = TRANSLATIONS.get(lang, TRANSLATIONS[DEFAULT_LANGUAGE])
    text = table.get(key, TRANSLATIONS[DEFAULT_LANGUAGE].get(key, key))
    return text.format(**kwargs) if kwargs else text
