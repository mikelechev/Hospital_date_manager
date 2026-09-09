"""
Módulo responsable de cargar en memoria el modelo de Machine Learning.
Aísla la dependencia de scikit-learn/joblib del resto del sistema.
"""

import logging
from pathlib import Path
from typing import Any, Optional
import joblib

from chatbot.config import PREDICT_CONFIG

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Clase encargada de cargar y proveer el modelo de predicción de No-Show
    (un VotingClassifier calibrado con scikit-learn, serializado con joblib;
    el mismo artefacto que usa el dashboard de simulación en scripts/app.py).
    Implementa un patrón Singleton a nivel de instancia para evitar recargas innecesarias.
    """

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._model_path: Path = PREDICT_CONFIG.model_path

    def load_model(self) -> Any:
        """
        Carga el modelo desde disco (formato joblib) si no está ya en memoria.

        Returns:
            El estimador scikit-learn listo para predecir (expone predict_proba).

        Raises:
            FileNotFoundError: Si el archivo del modelo no existe en la ruta configurada.
            RuntimeError: Si ocurre un error interno al cargar el artefacto.
        """
        if self._model is not None:
            return self._model

        if not self._model_path.exists():
            error_msg = f"No se encontró el modelo de predicción en la ruta: {self._model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Cargando modelo de predicción desde {self._model_path}")
            self._model = joblib.load(self._model_path)
            logger.info("Modelo cargado exitosamente.")
            return self._model

        except Exception as e:
            error_msg = f"Error crítico al cargar el modelo de predicción: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
