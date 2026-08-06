"""
Módulo coordinador central de la conversación.
Implementa el patrón Facade para interactuar con UI y Predictor[cite: 2, 4].
"""

import logging
from typing import Tuple, Dict, Any
from providers.base_provider import BaseProvider
from .patient_state import PatientState
from .history_manager import HistoryManager
from .prompt_builder import PromptBuilder
from .extractor import Extractor

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Orquesta el flujo conversacional.
    Coordina estado, historial, LLM y extracción, sin acoplarse a tecnologías específicas[cite: 2].
    """

    def __init__(self, provider: BaseProvider) -> None:
        """
        Inicializa el manager inyectando la dependencia del proveedor LLM.
        """
        self.provider = provider
        self.state = PatientState()
        self.history = HistoryManager()
        self.prompt_builder = PromptBuilder()
        self.extractor = Extractor()
        logger.info("ConversationManager inicializado correctamente.")

    def process_user_input(self, user_text: str) -> Tuple[str, bool]:
        """
        Procesa un nuevo mensaje del usuario y devuelve la respuesta del asistente.
        
        Args:
            user_text: Mensaje de texto ingresado por el paciente.
            
        Returns:
            Tuple[str, bool]: Respuesta natural del asistente y flag indicando si está listo para predecir.
        """
        logger.info("Procesando nuevo mensaje de usuario.")
        self.history.add_message("user", user_text)
        
        try:
            # 1. Construir contexto
            messages = self.prompt_builder.build_messages(self.history, self.state)
            
            # 2. Obtener respuesta del LLM
            llm_json_response = self.provider.generate_response(messages)
            
            # 3. Extraer y actualizar estado
            assistant_reply, analysis = self.extractor.process_llm_response(
                llm_json_response, self.state
            )
            
            # 4. Guardar respuesta en historial
            self.history.add_message("assistant", assistant_reply)
            
            # 5. Comprobar si hemos terminado de recopilar datos[cite: 4]
            ready_for_prediction = self.state.is_ready_for_prediction()
            if ready_for_prediction:
                logger.info("Estado del paciente suficientemente completo para predicción.")
                
            return assistant_reply, ready_for_prediction
            
        except Exception as e:
            error_msg = f"Error crítico en el flujo conversacional: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return "Lo siento, ha ocurrido un error técnico interno procesando su solicitud.", False

    def get_current_state(self) -> Dict[str, Any]:
        """Devuelve un volcado del estado actual para la interfaz o el predictor."""
        return self.state.model_dump()

    def process_clinical_history(self, text: str) -> Dict[str, Any]:
        """Use the same LLM interpretation flow as normal chat to extract structured fields."""
        logger.info("Procesando clinical history a través del intérprete de chat.")

        try:
            messages = self.prompt_builder.build_clinical_history_messages(self.state, text)
            llm_json_response = self.provider.generate_response(messages)
            assistant_reply, analysis = self.extractor.process_llm_response(llm_json_response, self.state)
            self.history.add_message("assistant", assistant_reply)
            ready = self.state.is_ready_for_prediction()
            return {
                "assistant_response": assistant_reply,
                "analysis": analysis,
                "ready_for_prediction": ready,
            }
        except Exception as e:
            logger.exception("Error interpretando clinical history con el intérprete de chat.")
            return {"error": str(e)}
