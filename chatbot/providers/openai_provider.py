"""
Proveedor para la API de OpenAI (Chat Completions).
Requiere la librería `openai` (SDK >= 1.0, cliente basado en instancia
`OpenAI(...)`; la API modular `openai.ChatCompletion.create` fue retirada
en la v1.0 del SDK y ya no existe en las versiones actuales).
"""

import logging
from typing import List, Dict, Any

from chatbot.providers.base_provider import BaseProvider
from chatbot.config import LLM_CONFIG

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseProvider):
    def __init__(self) -> None:
        try:
            import openai
        except ModuleNotFoundError as e:
            raise RuntimeError("El paquete 'openai' no está instalado. Instala con 'pip install openai'.") from e

        api_key = getattr(LLM_CONFIG, 'openai_api_key', None)
        if not api_key:
            raise ValueError("OPENAI_API_KEY no está configurada en las variables de entorno.")

        self.model = getattr(LLM_CONFIG, 'openai_model', None) or LLM_CONFIG.default_model
        self.timeout = LLM_CONFIG.request_timeout
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=getattr(LLM_CONFIG, 'openai_base_url', None) or None,
            timeout=self.timeout,
        )
        logger.info(f"OpenAIProvider inicializado (Modelo: {self.model})")

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        chat_messages = []
        for m in messages:
            role = m.get('role')
            if role not in ('assistant', 'system'):
                role = 'user'
            chat_messages.append({'role': role, 'content': m.get('content')})

        try:
            self.last_model_used = self.model

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                temperature=LLM_CONFIG.temperature,
                max_tokens=LLM_CONFIG.max_tokens,
            )
            text = resp.choices[0].message.content

            try:
                self.last_call_meta = {
                    "id": getattr(resp, "id", None),
                    "usage": getattr(resp, "usage", None),
                }
            except Exception:
                self.last_call_meta = {}

            return self._clean_json_response(text or "")
        except Exception as e:
            logger.exception('Error llamando a OpenAI API')
            raise RuntimeError(str(e)) from e

    def list_models(self) -> list:
        """Lista modelos disponibles en OpenAI a través del SDK oficial.
        Devuelve lista de model ids o [] en caso de error."""
        try:
            resp = self.client.models.list()
            models = []
            for m in getattr(resp, 'data', []) or resp:
                models.append(getattr(m, 'id', None) or str(m))
            return models
        except Exception as e:
            logger.debug(f"OpenAI list_models error: {e}")
            return []
