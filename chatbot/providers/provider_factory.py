"""
Módulo responsable de instanciar el proveedor correcto basándose en la configuración.
Implementa el patrón Factory.
"""

import logging
from chatbot.providers.base_provider import BaseProvider
from chatbot.providers.ollama_provider import OllamaProvider
from chatbot.providers.gemini_provider import GeminiProvider
from chatbot.providers.openai_provider import OpenAIProvider
from chatbot.providers.claude_provider import ClaudeProvider
from chatbot.config import LLM_CONFIG

logger = logging.getLogger(__name__)

class ProviderFactory:
    """
    Fábrica encargada de crear instancias de proveedores LLM.
    Desacopla la lógica de instanciación del resto de la aplicación[cite: 1, 2].
    """

    @staticmethod
    def get_provider() -> BaseProvider:
        """
        Devuelve la instancia del proveedor configurado por defecto.
        
        Returns:
            BaseProvider: Instancia del proveedor (Ollama o Gemini).
            
        Raises:
            ValueError: Si el proveedor configurado no está soportado.
        """
        provider_name = LLM_CONFIG.default_provider.lower().strip()
        
        if provider_name == "ollama":
            return OllamaProvider()
        elif provider_name == "gemini":
            return GeminiProvider()
        elif provider_name == "openai":
            return OpenAIProvider()
        elif provider_name == "claude":
            return ClaudeProvider()
        else:
            error_msg = f"Proveedor no soportado: {provider_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)