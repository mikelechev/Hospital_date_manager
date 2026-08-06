"""
Módulo para la gestión del historial de la conversación.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class HistoryManager:
    """
    Gestiona la memoria a corto plazo de la conversación.
    """
    def __init__(self) -> None:
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Añade un mensaje al historial."""
        if role not in ["system", "user", "assistant"]:
            logger.warning(f"Rol no reconocido intentando ser añadido al historial: {role}")
            
        self.messages.append({"role": role, "content": content})
        logger.debug(f"Mensaje añadido al historial (Rol: {role}). Total: {len(self.messages)}")

    def get_history(self) -> List[Dict[str, str]]:
        """Devuelve el historial completo."""
        return self.messages.copy()