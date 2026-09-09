"""
Implementación del proveedor LLM utilizando la API oficial de Google Gemini.
Garantiza el aislamiento de la lógica de comunicación externa.
"""

import logging
import requests
from google import genai
from google.genai import types
from typing import List, Dict, Any

from chatbot.providers.base_provider import BaseProvider
from chatbot.config import LLM_CONFIG

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

        self.model_name = LLM_CONFIG.gemini_model or "gemini-2.0-flash"
        self.client = genai.Client(api_key=api_key)
        self.generation_config = types.GenerateContentConfig(
            temperature=LLM_CONFIG.temperature,
            response_mime_type="application/json",
        )
        self.timeout = LLM_CONFIG.request_timeout
        logger.info(f"GeminiProvider inicializado con el modelo: {self.model_name}")

    def _convert_messages_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convierte el formato estándar de mensajes al formato requerido por Gemini.
        """
        gemini_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            gemini_role = "model" if role == "assistant" else "user"
            gemini_messages.append({
                "role": gemini_role,
                "parts": [{"text": msg["content"]}]
            })
        return gemini_messages

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Envía la petición a Gemini y reintenta con modelos alternativos si el modelo configurado no está disponible.
        """
        try:
            self.last_model_used = self.model_name
            logger.debug("Enviando petición a Gemini API")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=self._convert_messages_format(messages),
                config=self.generation_config,
            )
            text = response.text if hasattr(response, "text") else str(response)
            # record metadata for UI
            try:
                self.last_call_meta = {"model": self.model_name}
            except Exception:
                self.last_call_meta = {"model": self.model_name}
            return self._clean_json_response(text)
        except Exception as e:
            # Log original error
            msg = str(e)
            error_msg = f"Error en la comunicación con Gemini API: {msg}"
            logger.error(error_msg)

            # Attempt to extract a suggested model from the error message
            import re
            fallback_candidates = []
            m = re.search(r"models/[\w\-\.]+", msg)
            if m:
                fallback_candidates.append(m.group(0))

            # Try variants of the configured model (with/without 'models/')
            if self.model_name:
                if self.model_name.startswith("models/"):
                    fallback_candidates.append(self.model_name.replace("models/", ""))
                else:
                    fallback_candidates.append("models/" + self.model_name)

            # Recommended modern model
            fallback_candidates.append("models/gemini-3.6-flash")
            fallback_candidates.append("gemini-3.6-flash")

            tried = set()
            for candidate in fallback_candidates:
                if not candidate or candidate in tried:
                    continue
                tried.add(candidate)
                try:
                    logger.info(f"Intentando modelo alternativo: {candidate}")
                    response2 = self.client.models.generate_content(
                        model=candidate,
                        contents=self._convert_messages_format(messages),
                        config=self.generation_config,
                    )
                    text2 = response2.text if hasattr(response2, "text") else str(response2)
                    # record metadata for UI
                    try:
                        self.last_model_used = candidate
                        self.last_call_meta = {"model": candidate}
                    except Exception:
                        pass
                    return self._clean_json_response(text2)
                except Exception as e2:
                    logger.debug(f"Fallback model {candidate} failed: {e2}")
                    continue

            # If all retries failed, raise runtime error
            raise RuntimeError(error_msg) from e

    def list_models(self) -> list:
        """Lista modelos disponibles. Primero intenta la SDK, si falla usa llamadas HTTP directas.
        Devuelve lista de ids de modelos o [] en fallo."""
        models = []
        # 1) Intentar usar la SDK si está disponible
        try:
            if hasattr(self, 'client'):
                if hasattr(self.client, 'list_models'):
                    resp = self.client.list_models()
                    for m in resp:
                        models.append(getattr(m, 'name', None) or getattr(m, 'model', None) or str(m))
                    return list(dict.fromkeys(models))
                if hasattr(self.client, 'models') and hasattr(self.client.models, 'list'):
                    resp = self.client.models.list()
                    seq = getattr(resp, 'data', None) or resp
                    for m in seq:
                        models.append(getattr(m, 'name', None) or getattr(m, 'model', None) or str(m))
                    return list(dict.fromkeys(models))
        except Exception as e:
            logger.debug(f"Gemini SDK list_models attempt failed: {e}")

        # 2) Fallback HTTP endpoints (try common endpoints used by Google Generative API / Gemini)
        endpoints = []
        if getattr(LLM_CONFIG, 'gemini_base_url', None):
            endpoints.append(LLM_CONFIG.gemini_base_url.rstrip('/') + '/models')
        endpoints.extend([
            'https://generativelanguage.googleapis.com/v1/models',
            'https://gemini.googleapis.com/v1/models'
        ])

        headers = {'Authorization': f'Bearer {LLM_CONFIG.gemini_api_key}'} if LLM_CONFIG.gemini_api_key else {}

        for url in endpoints:
            try:
                resp = requests.get(url, headers=headers, timeout=getattr(self, 'timeout', 30))
                if not resp.ok:
                    # try with key as query param
                    if LLM_CONFIG.gemini_api_key:
                        resp = requests.get(url, params={'key': LLM_CONFIG.gemini_api_key}, timeout=getattr(self, 'timeout', 30))
                if not resp.ok:
                    continue
                data = resp.json()
                # parse common structures
                candidates = []
                if isinstance(data, dict):
                    if 'models' in data and isinstance(data['models'], list):
                        candidates = data['models']
                    elif 'data' in data and isinstance(data['data'], list):
                        candidates = data['data']
                    else:
                        # sometimes API returns a mapping of model -> details
                        for k, v in data.items():
                            if isinstance(v, dict) and ('name' in v or 'id' in v):
                                candidates.append(v)
                elif isinstance(data, list):
                    candidates = data

                for item in candidates:
                    if isinstance(item, dict):
                        models.append(item.get('name') or item.get('id') or item.get('model') or str(item))
                    else:
                        models.append(str(item))

                if models:
                    return list(dict.fromkeys(models))

            except Exception as e:
                logger.debug(f"Gemini HTTP list_models attempt to {url} failed: {e}")
                continue

        # Nothing found
        return []