"""
Módulo para la construcción de prompts del sistema.
Aplica principios de Prompt Engineering.
"""

import json
import logging
from typing import List, Dict

from chatbot.conversation.patient_state import PatientState
from chatbot.conversation.history_manager import HistoryManager
from chatbot.i18n import DEFAULT_LANGUAGE, LLM_LANGUAGE_INSTRUCTIONS

logger = logging.getLogger(__name__)
    
class PromptBuilder:
    """
    Construye los mensajes para el LLM combinando instrucciones, historial y estado[cite: 2].
    """

    SYSTEM_PROMPT_TEMPLATE = """
Eres un asistente conversacional avanzado en un entorno hospitalario.
Tu objetivo NO es diagnosticar, NO es agendar citas y NO sustituyes a un médico.
Tu ÚNICA función es conversar de forma natural y empática con el paciente para obtener la siguiente información:
{missing_fields}

IDIOMA: {language_instruction}

REGLAS ESTRICTAS DE CONVERSACIÓN:
1. NUNCA parezcas un formulario. Muestra empatía y naturalidad.
2. Escucha activamente. Si el paciente da información espontánea, asúmela.
3. Haz ÚNICAMENTE UNA pregunta por turno.
4. No repitas preguntas sobre información que ya tienes.
5. Para el historial de faltas, prioriza preguntar cuántas citas médicas ha
   tenido antes y a cuántas ha faltado (rellena "citas_previas" y
   "faltas_previas" con esos números exactos). Solo si el paciente responde
   con una proporción aproximada ("falto la mitad de las veces") sin dar
   números concretos, usa "history_no_show" con ese valor entre 0 y 1.
6. El estado actual del paciente es: {current_state}

REGLAS ESTRICTAS DE FORMATO (JSON OBLIGATORIO):
Debes responder ÚNICAMENTE con un objeto JSON válido con la siguiente estructura exacta:
{{
  "assistant_response": "La respuesta natural y empática que leerá el usuario",
  "conversation_analysis": {{
    "intent": "intención del usuario",
    "emotion": "emoción detectada",
    "extracted_data": {{
      "age": int o null,
      "gender_m": 1 (hombre) o 0 (mujer) o null,
      "hypertension": 1 o 0 o null,
      "diabetes": 1 o 0 o null,
      "alcoholism": 1 o 0 o null,
      "handicap": 1 o 0 o null,
      "scholarship": 1 o 0 o null,
      "sms_received": int o null,
      "history_no_show": float o null,
      "citas_previas": int o null,
      "faltas_previas": int o null,
      "days_between": int o null,
      "weekend": 1 o 0 o null,
      "time_of_day": "string" o null,
      "consultation_reason": "string" o null
    }},
    "confidence": {{
      "age": float 0.0-1.0,
      "gender_m": float 0.0-1.0,
      "hypertension": float 0.0-1.0,
      "diabetes": float 0.0-1.0,
      "alcoholism": float 0.0-1.0,
      "handicap": float 0.0-1.0,
      "scholarship": float 0.0-1.0,
      "sms_received": float 0.0-1.0,
      "history_no_show": float 0.0-1.0,
      "citas_previas": float 0.0-1.0,
      "faltas_previas": float 0.0-1.0,
      "days_between": float 0.0-1.0,
      "weekend": float 0.0-1.0,
      "time_of_day": float 0.0-1.0,
      "consultation_reason": float 0.0-1.0
    }},
    "missing_fields": ["lista de campos que aún faltan"],
    "next_goal": "Qué vas a preguntar a continuación",
    "conversation_finished": false
  }}
}}
No incluyas texto fuera del JSON. No uses etiquetas Markdown en el output.
"""

    CLINICAL_HISTORY_PROMPT_TEMPLATE = """
Eres un asistente especializado en extracción de datos clínicos a partir de historias médicas.
Vas a recibir un texto de historia clínica del paciente. Tu tarea es extraer variables estructuradas.
No realices preguntas al paciente. No hay conversación: solo estás leyendo un documento.
Si algún campo no aparece en el texto, usa null.

IDIOMA: {language_instruction}

Estado actual del paciente (puede ayudar a completar campos faltantes): {current_state}

Historia clínica:
{clinical_history}

REGLAS ESTRICTAS DE FORMATO (JSON OBLIGATORIO):
Debes responder ÚNICAMENTE con un objeto JSON válido con la siguiente estructura exacta
(la misma que usarías en el chat normal; "assistant_response" aquí debe ser un breve
resumen en español, en tono natural, de lo que extrajiste del historial):
{{
  "assistant_response": "Resumen breve y natural de los datos extraídos del historial",
  "conversation_analysis": {{
    "intent": "extraccion_historial_clinico",
    "emotion": "neutral",
    "extracted_data": {{
      "age": int o null,
      "gender_m": 1 (hombre) o 0 (mujer) o null,
      "hypertension": 1 o 0 o null,
      "diabetes": 1 o 0 o null,
      "alcoholism": 1 o 0 o null,
      "handicap": 1 o 0 o null,
      "scholarship": 1 o 0 o null,
      "sms_received": int o null,
      "history_no_show": float o null,
      "citas_previas": int o null,
      "faltas_previas": int o null,
      "days_between": int o null,
      "weekend": 1 o 0 o null,
      "time_of_day": "string" o null,
      "consultation_reason": "string" o null
    }},
    "confidence": {{
      "age": float 0.0-1.0,
      "gender_m": float 0.0-1.0,
      "hypertension": float 0.0-1.0,
      "diabetes": float 0.0-1.0,
      "alcoholism": float 0.0-1.0,
      "handicap": float 0.0-1.0,
      "scholarship": float 0.0-1.0,
      "sms_received": float 0.0-1.0,
      "history_no_show": float 0.0-1.0,
      "citas_previas": float 0.0-1.0,
      "faltas_previas": float 0.0-1.0,
      "days_between": float 0.0-1.0,
      "weekend": float 0.0-1.0,
      "time_of_day": float 0.0-1.0,
      "consultation_reason": float 0.0-1.0
    }},
    "missing_fields": ["lista de campos que siguen faltando tras leer el historial"],
    "next_goal": "",
    "conversation_finished": false
  }}
}}
No incluyas texto fuera del JSON. No uses etiquetas Markdown en el output.
"""

    def build_messages(
        self, history: HistoryManager, state: PatientState, language: str = DEFAULT_LANGUAGE
    ) -> List[Dict[str, str]]:
        """Construye la lista final de mensajes para enviar al Provider."""
        missing = state.get_missing_critical_fields()

        # Serializamos el estado actual de forma limpia para que el LLM lo entienda.
        # model_dump() devuelve un diccionario de diccionarios, por lo que usamos .get()
        state_dict = state.model_dump()
        current_state_dict = {
            k: v.get("value") for k, v in state_dict.items()
            if isinstance(v, dict) and v.get("value") is not None
        }

        system_content = self.SYSTEM_PROMPT_TEMPLATE.format(
            missing_fields=", ".join(missing) if missing else "Ninguna, tienes todos los datos.",
            current_state=json.dumps(current_state_dict, ensure_ascii=False),
            language_instruction=LLM_LANGUAGE_INSTRUCTIONS.get(
                language, LLM_LANGUAGE_INSTRUCTIONS[DEFAULT_LANGUAGE]
            ),
        )

        messages = [{"role": "system", "content": system_content}]
        messages.extend(history.get_history())

        return messages

    def build_clinical_history_messages(
        self, state: PatientState, clinical_history: str, language: str = DEFAULT_LANGUAGE
    ) -> List[Dict[str, str]]:
        """Construye los mensajes para interpretar una historia clínica con el mismo extractor."""
        state_dict = state.model_dump()
        current_state_dict = {
            k: v.get("value") for k, v in state_dict.items()
            if isinstance(v, dict) and v.get("value") is not None
        }

        system_content = self.CLINICAL_HISTORY_PROMPT_TEMPLATE.format(
            current_state=json.dumps(current_state_dict, ensure_ascii=False),
            clinical_history=clinical_history,
            language_instruction=LLM_LANGUAGE_INSTRUCTIONS.get(
                language, LLM_LANGUAGE_INSTRUCTIONS[DEFAULT_LANGUAGE]
            ),
        )

        return [{"role": "system", "content": system_content}]