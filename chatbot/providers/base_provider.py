"""
Módulo que define la interfaz base para todos los proveedores LLM.
Garantiza el cumplimiento del principio de Inversión de Dependencias (DIP).
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BaseProvider(ABC):
    """
    Clase abstracta que define el contrato para los proveedores de LLM.
    Su única responsabilidad es enviar mensajes y recibir respuestas[cite: 1, 2].
    """

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Envía una lista de mensajes al LLM y devuelve la respuesta en formato string (esperando JSON).
        
        Args:
            messages: Lista de diccionarios con claves 'role' y 'content'.
            
        Returns:
            str: Respuesta generada por el modelo, estrictamente limpiada de formato Markdown.
        """
        pass

    def _clean_json_response(self, raw_response: str) -> str:
        """
        Limpia las etiquetas Markdown que los LLMs suelen añadir al devolver JSON.
        
        Args:
            raw_response: Respuesta cruda del LLM.
            
        Returns:
            str: Respuesta limpia lista para ser parseada.
        """
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
            
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
            
        return cleaned.strip()