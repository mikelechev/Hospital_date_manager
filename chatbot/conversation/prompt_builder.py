"""
Módulo para la construcción de prompts del sistema.
Aplica principios de Prompt Engineering.
"""

import json
import logging
from typing import List, Dict

from .patient_state import PatientState
from .history_manager import HistoryManager

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

REGLAS ESTRICTAS DE CONVERSACIÓN:
1. NUNCA parezcas un formulario. Muestra empatía y naturalidad.
2. Escucha activamente. Si el paciente da información espontánea, asúmela.
3. Haz ÚNICAMENTE UNA pregunta por turno.
4. No repitas preguntas sobre información que ya tienes.
5. El estado actual del paciente es: {current_state}

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
      "sms_received": int o null,
      "history_no_show": float o null,
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
      "handicap": float 0.0-1.0
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
Vas a recibir un texto de historia clínica del paciente. Tu tarea es extraer variables estructuradas y solamente responder con JSON.
No realices preguntas. No escribas explicaciones humanas. Solo devuelve el JSON con los mismos campos que usas en el chat.
Si algún campo no aparece en el texto, usa null.

Estado actual del paciente (puede ayudar a completar campos faltantes): {current_state}

Historia clínica:
{clinical_history}

Recuerda responder únicamente con un objeto JSON válido y con la estructura exacta indicada.
"""

    def build_messages(self, history: HistoryManager, state: PatientState) -> List[Dict[str, str]]:
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
            current_state=json.dumps(current_state_dict, ensure_ascii=False)
        )
        
        messages = [{"role": "system", "content": system_content}]
        messages.extend(history.get_history())
        
        return messages

    def build_clinical_history_messages(self, state: PatientState, clinical_history: str) -> List[Dict[str, str]]:
        """Construye los mensajes para interpretar una historia clínica con el mismo extractor."""
        state_dict = state.model_dump()
        current_state_dict = {
            k: v.get("value") for k, v in state_dict.items()
            if isinstance(v, dict) and v.get("value") is not None
        }

        system_content = self.CLINICAL_HISTORY_PROMPT_TEMPLATE.format(
            current_state=json.dumps(current_state_dict, ensure_ascii=False),
            clinical_history=clinical_history,
        )

        return [{"role": "system", "content": system_content}]
