"""
Módulo encargado de interpretar el JSON del LLM y actualizar el estado[cite: 2].
"""

import json
import logging
from typing import Dict, Any, Tuple
from .patient_state import PatientState

logger = logging.getLogger(__name__)

class Extractor:
    """
    Procesa respuestas estructuradas e inyecta los datos en PatientState.
    No contiene lógica conversacional[cite: 2].
    """
    
    def process_llm_response(self, json_str: str, state: PatientState) -> Tuple[str, Dict[str, Any]]:
        """
        Parsea el JSON y actualiza el estado del paciente.
        
        Args:
            json_str: Cadena JSON generada por el LLM.
            state: Referencia al estado actual del paciente.
            
        Returns:
            Tuple[str, Dict[str, Any]]: La respuesta para el usuario y el análisis de la conversación.
            
        Raises:
            ValueError: Si el JSON es inválido o no cumple el esquema requerido.
        """
        try:
            data = json.loads(json_str)
            
            assistant_response = data.get("assistant_response")
            analysis = data.get("conversation_analysis", {})
            extracted_data = analysis.get("extracted_data", {})
            confidence_data = analysis.get("confidence", {})
            
            if not assistant_response:
                raise ValueError("El JSON no contiene 'assistant_response'")
                
            self._update_state(state, extracted_data, confidence_data)
            
            return assistant_response, analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parseando JSON del LLM: {str(e)}\nContenido crudo: {json_str}")
            raise ValueError("El LLM no devolvió un JSON válido.") from e

    def _update_state(self, state: PatientState, extracted: Dict[str, Any], confidences: Dict[str, float]) -> None:
        """Itera sobre los datos extraídos y actualiza las variables del estado."""
        for key, value in extracted.items():
            if value is not None and hasattr(state, key):
                confidence = confidences.get(key, 0.5)
                field: Any = getattr(state, key)
                field.update(value=value, confidence=confidence, source="llm_extraction")