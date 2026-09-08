"""
Proveedor para Anthropic Claude (HTTP + fallback).
Implementa el contrato BaseProvider.
"""

import logging
import requests
from typing import List, Dict, Any

from chatbot.providers.base_provider import BaseProvider
from chatbot.config import LLM_CONFIG

logger = logging.getLogger(__name__)


class ClaudeProvider(BaseProvider):
    """Proveedor para Anthropic Claude vía API HTTP.

    Convierte el historial de mensajes en el formato "Human:/Assistant:" que
    Claude espera y llama al endpoint /v1/complete. No depende del SDK.
    """

    def __init__(self) -> None:
        self.api_key = getattr(LLM_CONFIG, "claude_api_key", None)
        if not self.api_key:
            raise ValueError("CLAUDE_API_KEY no está configurada en las variables de entorno.")

        self.model = getattr(LLM_CONFIG, "claude_model", None) or LLM_CONFIG.default_model
        self.base_url = getattr(LLM_CONFIG, "claude_base_url", "https://api.anthropic.com").rstrip('/')
        self.timeout = LLM_CONFIG.request_timeout
        logger.info(f"ClaudeProvider inicializado (Modelo: {self.model})")

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Transforma la lista de mensajes a un prompt compat con Claude.

        Usa etiquetas: "Human:" y "Assistant:" para cada turno.
        """
        parts = []
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            if role == 'assistant':
                parts.append(f"Assistant: {content}\n\n")
            else:
                # incluye 'system' o 'user' como Human
                parts.append(f"Human: {content}\n\n")
        # Añadir la señal de que esperamos la respuesta del asistente
        parts.append("Assistant:")
        return "".join(parts)

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        prompt = self._messages_to_prompt(messages)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens_to_sample": int(LLM_CONFIG.max_tokens),
            "temperature": float(LLM_CONFIG.temperature),
        }

        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        # Anthropic historically usa Authorization: Bearer, pero algunas integraciones usan x-api-key.
        # Probar ambas cabeceras si es necesario más adelante.

        try:
            self.last_model_used = self.model
            url = f"{self.base_url}/v1/complete"
            logger.debug(f"Enviando petición a Claude en {url}")
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            # Extraer resultado de forma robusta
            completion = data.get('completion') or data.get('completion', '') or data.get('response') or data.get('output') or ''
            if not completion and 'completion' in data:
                completion = data['completion']

            try:
                self.last_call_meta = {"status_code": resp.status_code, "id": data.get('id')}
            except Exception:
                self.last_call_meta = {"status_code": resp.status_code}

            return self._clean_json_response(str(completion))
        except requests.exceptions.RequestException as e:
            logger.error(f"Error comunicando con Claude API: {e}")
            raise ConnectionError(f"Error comunicando con Claude API: {e}") from e
        except Exception as e:
            logger.exception("Error inesperado en ClaudeProvider")
            raise RuntimeError(str(e)) from e

    def list_models(self) -> list:
        """Intenta listar modelos desde el endpoint público de Anthropic si está disponible.

        Devuelve una lista de ids o una lista por defecto si el endpoint no responde.
        """
        models = []
        headers = {"x-api-key": self.api_key}
        try:
            url = f"{self.base_url}/v1/models"
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            if resp.ok:
                data = resp.json()
                if isinstance(data, dict):
                    # puede venir como {'models': [...]}
                    if 'models' in data and isinstance(data['models'], list):
                        for m in data['models']:
                            if isinstance(m, dict):
                                models.append(m.get('id') or m.get('name') or str(m))
                            else:
                                models.append(str(m))
                elif isinstance(data, list):
                    for item in data:
                        models.append(item.get('id') if isinstance(item, dict) else str(item))
                if models:
                    return list(dict.fromkeys(models))
        except Exception as e:
            logger.debug(f"Claude list_models attempt failed: {e}")

        # Fallback: modelos comunes
        return ["claude-2.1", "claude-2", "claude-instant-v1"]
