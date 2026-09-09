"""
Proveedor para Anthropic Claude (HTTP directo, sin SDK).
Implementa el contrato BaseProvider usando la Messages API
(`/v1/messages`) — el endpoint `/v1/complete` (Text Completions) que usaba
esta clase antes está deprecado y no admite los modelos Claude actuales.
"""

import logging
import requests
from typing import Any, Dict, List, Tuple

from chatbot.providers.base_provider import BaseProvider
from chatbot.config import LLM_CONFIG

logger = logging.getLogger(__name__)

ANTHROPIC_API_VERSION = "2023-06-01"


class ClaudeProvider(BaseProvider):
    """Proveedor para comunicarse con la API de Anthropic Claude.

    Usa la Messages API: system prompt aparte del historial, y turnos que
    alternan estrictamente entre 'user' y 'assistant'.
    """

    def __init__(self) -> None:
        self.api_key = getattr(LLM_CONFIG, "claude_api_key", None)
        if not self.api_key:
            raise ValueError("CLAUDE_API_KEY no está configurada en las variables de entorno.")

        self.model = getattr(LLM_CONFIG, "claude_model", None) or LLM_CONFIG.default_model
        self.base_url = (getattr(LLM_CONFIG, "claude_base_url", None) or "https://api.anthropic.com").rstrip('/')
        self.timeout = LLM_CONFIG.request_timeout
        logger.info(f"ClaudeProvider inicializado (Modelo: {self.model})")

    def _split_messages(self, messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        """Separa el system prompt (la Messages API lo espera en un campo
        aparte, no como un mensaje más) y fusiona turnos consecutivos del
        mismo rol, porque Claude exige que los mensajes alternen
        estrictamente entre 'user' y 'assistant'.
        """
        system_parts = []
        turns: List[Dict[str, str]] = []
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '') or ''
            if role == 'system':
                system_parts.append(content)
                continue
            role = 'assistant' if role == 'assistant' else 'user'
            if turns and turns[-1]['role'] == role:
                turns[-1]['content'] += f"\n\n{content}"
            else:
                turns.append({'role': role, 'content': content})

        if not turns or turns[0]['role'] != 'user':
            turns.insert(0, {'role': 'user', 'content': '(inicio de conversación)'})

        return "\n\n".join(system_parts), turns

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        system_prompt, turns = self._split_messages(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "max_tokens": int(LLM_CONFIG.max_tokens),
            "temperature": float(LLM_CONFIG.temperature),
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "Content-Type": "application/json",
        }

        try:
            self.last_model_used = self.model
            url = f"{self.base_url}/v1/messages"
            logger.debug(f"Enviando petición a Claude en {url}")
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            blocks = data.get("content") or []
            text = "".join(
                b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
            )

            try:
                self.last_call_meta = {"id": data.get("id"), "usage": data.get("usage")}
            except Exception:
                self.last_call_meta = {}

            return self._clean_json_response(text)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error comunicando con Claude API: {e}")
            raise ConnectionError(f"Error comunicando con Claude API: {e}") from e
        except Exception as e:
            logger.exception("Error inesperado en ClaudeProvider")
            raise RuntimeError(str(e)) from e

    def list_models(self) -> list:
        """Lista modelos desde el endpoint público de Anthropic (`/v1/models`).

        Devuelve una lista de ids o una lista por defecto si el endpoint no responde.
        """
        headers = {"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_API_VERSION}
        try:
            url = f"{self.base_url}/v1/models"
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            if resp.ok:
                data = resp.json()
                models = [
                    m.get('id') for m in data.get('data', []) if isinstance(m, dict) and m.get('id')
                ]
                if models:
                    return models
        except Exception as e:
            logger.debug(f"Claude list_models attempt failed: {e}")

        # Fallback: modelos actuales conocidos
        return ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001"]
