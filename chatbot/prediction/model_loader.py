"""
Módulo responsable de cargar en memoria el modelo de Machine Learning.
Aísla la dependencia de XGBoost del resto del sistema.
"""

import logging
from pathlib import Path
from typing import Optional
import xgboost as xgb

from config import PREDICT_CONFIG

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Clase encargada de cargar y proveer el modelo XGBoost[cite: 2].
    Implementa un patrón Singleton a nivel de instancia para evitar recargas innecesarias.
    """

    def __init__(self) -> None:
        self._model: Optional[xgb.XGBClassifier] = None
        self._model_path: Path = PREDICT_CONFIG.model_path

    def load_model(self) -> xgb.XGBClassifier:
        """
        Carga el modelo XGBoost desde el disco si no está ya en memoria.
        
        Returns:
            xgb.XGBClassifier: Instancia del modelo lista para predecir.
            
        Raises:
            FileNotFoundError: Si el archivo del modelo no existe en la ruta configurada.
            RuntimeError: Si ocurre un error interno en XGBoost al cargar el archivo.
        """
        if self._model is not None:
            return self._model

        if not self._model_path.exists():
            error_msg = f"No se encontró el modelo XGBoost en la ruta: {self._model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Cargando modelo XGBoost desde {self._model_path}")
            # Instanciamos el clasificador e importamos el artefacto JSON
            model = xgb.XGBClassifier()
            model.load_model(self._model_path)
            self._model = model
            logger.info("Modelo XGBoost cargado exitosamente.")
            return self._model
            
        except Exception as e:
            error_msg = f"Error crítico al cargar el modelo XGBoost: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e