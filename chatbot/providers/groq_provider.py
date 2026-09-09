"""
Proveedor para Groq (API compatible con OpenAI Chat Completions).
Implementa el contrato BaseProvider.

Usa el SDK oficial `groq` en lugar de reutilizar el cliente `openai` con un
`base_url` alternativo: expone tipos de excepción propios (RateLimitError,
APITimeoutError, APIConnectionError) que permiten distinguir con precisión
el límite de tasa del tier gratuito (429) de un simple fallo de red, algo
necesario para el reintento con backoff pedido más abajo.
"""

import logging
import time
from typing import Dict, List, Optional

from chatbot.providers.base_provider import BaseProvider
from chatbot.config import LLM_CONFIG

logger = logging.getLogger(__name__)

# El tier gratuito de Groq admite ~30 req/min; con 3 reintentos y backoff
# exponencial corto es suficiente para absorber una ráfaga puntual sin
# convertir un 429 pasajero en un error visible para el usuario.
MAX_RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 2.0


class GroqProvider(BaseProvider):
    """Proveedor para comunicarse con la API de Groq."""

    def __init__(self) -> None:
        try:
            import groq
        except ModuleNotFoundError as e:
            raise RuntimeError("El paquete 'groq' no está instalado. Instala con 'pip install groq'.") from e

        self.api_key = getattr(LLM_CONFIG, "groq_api_key", None)
        if not self.api_key:
            raise ValueError("GROQ_API_KEY no está configurada en las variables de entorno.")

        self._sdk = groq
        self.model = getattr(LLM_CONFIG, "groq_model", None) or LLM_CONFIG.default_model
        # OJO: solo el host. El SDK `groq` ya antepone "/openai/v1" a cada
        # endpoint por su cuenta; incluirlo aquí también duplica el segmento
        # y la API responde 404 "Unknown request URL: .../openai/v1/openai/v1/...".
        self.base_url = getattr(LLM_CONFIG, "groq_base_url", None) or "https://api.groq.com"
        self.timeout = LLM_CONFIG.request_timeout
        # max_retries=0: los reintentos por rate limit los gestionamos
        # nosotros explícitamente en generate_response (requisito de negocio:
        # backoff simple + mensaje claro en vez de fallar en silencio).
        self.client = groq.Groq(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )
        logger.info(f"GroqProvider inicializado (Modelo: {self.model})")

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        chat_messages = []
        for m in messages:
            role = m.get('role')
            if role not in ('assistant', 'system'):
                role = 'user'
            chat_messages.append({'role': role, 'content': m.get('content')})

        self.last_model_used = self.model
        attempt = 0
        while True:
            try:
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

            except self._sdk.RateLimitError as e:
                attempt += 1
                if attempt > MAX_RATE_LIMIT_RETRIES:
                    error_msg = (
                        "Groq ha alcanzado el límite de peticiones del plan gratuito "
                        "(30 req/min o ~1000 req/día). Espera unos segundos y vuelve a intentarlo, "
                        "o cambia de proveedor en la barra lateral."
                    )
                    logger.error(f"{error_msg} (detalle: {e})")
                    raise RuntimeError(error_msg) from e

                wait = self._retry_after_seconds(e) or (RATE_LIMIT_BACKOFF_SECONDS * attempt)
                logger.warning(
                    f"Rate limit de Groq alcanzado, reintentando en {wait:.1f}s "
                    f"(intento {attempt}/{MAX_RATE_LIMIT_RETRIES})"
                )
                time.sleep(wait)

            except self._sdk.APITimeoutError as e:
                error_msg = f"Tiempo de espera agotado ({self.timeout}s) comunicando con Groq: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg) from e

            except self._sdk.APIConnectionError as e:
                error_msg = f"Error de red comunicando con Groq: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg) from e

            except self._sdk.APIStatusError as e:
                error_msg = f"Error de la API de Groq ({e.status_code}): {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

            except Exception as e:
                logger.exception("Error inesperado en GroqProvider")
                raise RuntimeError(str(e)) from e

    @staticmethod
    def _retry_after_seconds(error) -> Optional[float]:
        """Respeta la cabecera Retry-After de Groq si viene informada."""
        try:
            retry_after = error.response.headers.get("retry-after")
            if retry_after is not None:
                return float(retry_after)
        except Exception:
            pass
        return None

    def list_models(self) -> list:
        """Lista modelos disponibles en Groq a través del SDK oficial.
        Devuelve lista de model ids o [] en caso de error."""
        try:
            resp = self.client.models.list()
            models = []
            for m in getattr(resp, "data", []) or resp:
                models.append(getattr(m, "id", None) or str(m))
            return models
        except Exception as e:
            logger.debug(f"Groq list_models error: {e}")
            return []
