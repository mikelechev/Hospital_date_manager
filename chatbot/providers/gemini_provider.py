"""
Implementación del proveedor LLM utilizando la API oficial de Google Gemini.
Garantiza el aislamiento de la lógica de comunicación externa.
"""

import logging
import google.generativeai as genai
from typing import List, Dict, Any

from .base_provider import BaseProvider
from config import LLM_CONFIG

logger = logging.getLogger(__name__)

class GeminiProvider(BaseProvider):
    """
    Proveedor para comunicarse con la API de Google Gemini.
    Su única responsabilidad es enviar y recibir mensajes.
    """

    def __init__(self) -> None:
        """
        Inicializa el proveedor configurando la API Key.
        Valida que la credencial exista antes de instanciar el modelo.
        """
        api_key = LLM_CONFIG.gemini_api_key
        if not api_key:
            error_msg = "GEMINI_API_KEY no está configurada en las variables de entorno."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Configuración del SDK
        genai.configure(api_key=api_key)
        
        # Corrección del modelo según la documentación oficial del proyecto[cite: 3]
        self.model_name = "gemini-3.6-flash" 
        
        # Forzamos la salida en JSON puro a nivel de API
        self.generation_config = genai.types.GenerationConfig(
            temperature=LLM_CONFIG.temperature,
            response_mime_type="application/json"
        )
        
        self.client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config
        )
        logger.info(f"GeminiProvider inicializado con el modelo: {self.model_name}")

    def _convert_messages_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convierte el formato estándar de mensajes al formato requerido por Gemini.
        """
        gemini_messages = []
        for msg in messages:
            role = msg["role"]
            gemini_role = "model" if role == "assistant" else "user"
            gemini_messages.append({
                "role": gemini_role,
                "parts": [msg["content"]]
            })
        return gemini_messages

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Envía la petición a Gemini.
        
        Args:
            messages: Historial y contexto estructurado.
            
        Returns:
            str: Respuesta del LLM limpia y en formato JSON.
            
        Raises:
            RuntimeError: Si ocurre un error en la comunicación con la API.
        """
        try:
            formatted_messages = self._convert_messages_format(messages)
            logger.debug("Enviando petición a Gemini API")
            
            response = self.client.generate_content(formatted_messages)
            
            return self._clean_json_response(response.text)
            
        except Exception as e:
            error_msg = f"Error en la comunicación con Gemini API: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e