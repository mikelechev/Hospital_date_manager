"""
Implementación del proveedor LLM utilizando la API REST local de Ollama.
Su única responsabilidad es la comunicación externa[cite: 1, 2].
"""

import logging
import requests
from typing import List, Dict

from .base_provider import BaseProvider
from config import LLM_CONFIG

logger = logging.getLogger(__name__)

class OllamaProvider(BaseProvider):
    """
    Proveedor para comunicarse con instancias de Ollama.
    """

    def __init__(self) -> None:
        """Inicializa el proveedor utilizando la configuración central."""
        self.base_url = f"{LLM_CONFIG.ollama_base_url}/api/chat"
        self.model = LLM_CONFIG.default_model
        self.temperature = LLM_CONFIG.temperature
        self.timeout = LLM_CONFIG.request_timeout
        logger.info(f"OllamaProvider inicializado (Modelo: {self.model}, Timeout: {self.timeout}s)")

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Envía la petición a Ollama forzando el formato JSON.
        
        Args:
            messages: Historial y contexto estructurado.
            
        Returns:
            str: Respuesta del LLM en formato JSON puro.
            
        Raises:
            ConnectionError: Si hay fallos de red o tiempos de espera agotados.
            RuntimeError: Si la API de Ollama devuelve un error estructural.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature
            }
        }

        try:
            logger.debug(f"Enviando petición a Ollama en {self.base_url}")
            # Se aplica el timeout dinámico configurado[cite: 1]
            response = requests.post(self.base_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            raw_content = data.get("message", {}).get("content", "")
            
            return self._clean_json_response(raw_content)
            
        except requests.exceptions.Timeout as e:
            error_msg = f"Tiempo de espera agotado ({self.timeout}s) comunicando con Ollama: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except requests.exceptions.RequestException as e:
            error_msg = f"Error de red comunicando con Ollama: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Error inesperado procesando la respuesta de Ollama: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e