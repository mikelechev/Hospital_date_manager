"""
Smoke test de GroqProvider.

Mockea el cliente `groq.Groq` (no depende de red ni de un GROQ_API_KEY real),
y comprueba que:
  1) generate_response() devuelve un string limpio (mismo formato que el
     resto de proveedores: sin fences ```json ni espacios sobrantes).
  2) un 429 (RateLimitError) transitorio se reintenta con backoff hasta
     tener éxito, en vez de propagar el error a la primera.
  3) si el rate limit persiste más allá de los reintentos, se lanza un
     RuntimeError con un mensaje claro para el usuario (no un traceback).

Ejecutar con: python -m chatbot.test_groq_provider
(o `pytest chatbot/test_groq_provider.py` si pytest está instalado).
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _fake_chat_completion(content: str):
    """Construye un objeto con la misma forma que devuelve el SDK de Groq
    para chat.completions.create (choices[0].message.content, id, usage)."""
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice], id="chatcmpl-test", usage={"total_tokens": 42})


class GroqProviderTest(unittest.TestCase):
    def setUp(self):
        # Import diferido: config.py lee variables de entorno al importarse,
        # así que fijamos GROQ_API_KEY antes de que chatbot.config se cargue.
        import os
        os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")

        from chatbot.config import LLM_CONFIG
        self.LLM_CONFIG = LLM_CONFIG
        self._original_api_key = LLM_CONFIG.groq_api_key
        LLM_CONFIG.groq_api_key = "test-key-not-real"

    def tearDown(self):
        self.LLM_CONFIG.groq_api_key = self._original_api_key

    def _make_provider(self, mock_client):
        from chatbot.providers.groq_provider import GroqProvider
        with patch("groq.Groq", return_value=mock_client):
            return GroqProvider()

    def test_generate_response_same_format_as_other_providers(self):
        """La respuesta debe llegar limpia, igual que OpenAI/Claude/Gemini."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_chat_completion(
            '```json\n{"campo": "valor"}\n```'
        )

        provider = self._make_provider(mock_client)
        result = provider.generate_response([{"role": "user", "content": "hola"}])

        self.assertIsInstance(result, str)
        self.assertEqual(result, '{"campo": "valor"}')
        self.assertFalse(result.startswith("```"))
        self.assertTrue(hasattr(provider, "last_model_used"))
        self.assertTrue(hasattr(provider, "last_call_meta"))

    def test_rate_limit_retries_then_succeeds(self):
        """Un 429 puntual se reintenta con backoff y no debe fallar."""
        import groq

        request = MagicMock()
        response_429 = MagicMock()
        response_429.headers = {}
        rate_limit_error = groq.RateLimitError(
            "rate limit exceeded", response=response_429, body=None
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            rate_limit_error,
            _fake_chat_completion('{"ok": true}'),
        ]

        provider = self._make_provider(mock_client)
        with patch("chatbot.providers.groq_provider.time.sleep") as mock_sleep:
            result = provider.generate_response([{"role": "user", "content": "hola"}])

        self.assertEqual(result, '{"ok": true}')
        mock_sleep.assert_called_once()
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    def test_rate_limit_exhausted_raises_clear_error(self):
        """Si el 429 persiste, se lanza un RuntimeError legible (no un traceback crudo)."""
        import groq
        from chatbot.providers.groq_provider import MAX_RATE_LIMIT_RETRIES

        request = MagicMock()
        response_429 = MagicMock()
        response_429.headers = {}
        rate_limit_error = groq.RateLimitError(
            "rate limit exceeded", response=response_429, body=None
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = rate_limit_error

        provider = self._make_provider(mock_client)
        with patch("chatbot.providers.groq_provider.time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                provider.generate_response([{"role": "user", "content": "hola"}])

        self.assertIn("límite de peticiones", str(ctx.exception))
        # 1 intento inicial + MAX_RATE_LIMIT_RETRIES reintentos
        self.assertEqual(
            mock_client.chat.completions.create.call_count, MAX_RATE_LIMIT_RETRIES + 1
        )

    def test_missing_api_key_raises_value_error(self):
        self.LLM_CONFIG.groq_api_key = ""
        from chatbot.providers.groq_provider import GroqProvider
        with self.assertRaises(ValueError):
            GroqProvider()


if __name__ == "__main__":
    unittest.main()
