"""
Módulo principal de inferencia de Machine Learning.
Transforma el PatientState en el formato requerido por el modelo y ejecuta la predicción.
"""

import logging
import pandas as pd
import numpy as np

from conversation.patient_state import PatientState
from config import PREDICT_CONFIG
from .model_loader import ModelLoader
from .prediction_result import PredictionResult

logger = logging.getLogger(__name__)

class Predictor:
    """
    Clase responsable de estimar la probabilidad de No-Show.
    No interactúa con la interfaz de usuario ni con los LLMs[cite: 1, 2].
    """

    def __init__(self) -> None:
        """Inicializa el predictor inyectando su dependencia de carga de modelos."""
        self.loader = ModelLoader()
        
    def _map_state_to_features(self, state: PatientState) -> pd.DataFrame:
        """
        Convierte el PatientState en un DataFrame con las columnas exactas
        esperadas por el modelo entrenado (modelo_campeon.json)[cite: 3].
        
        Args:
            state: Estado actual del paciente.
            
        Returns:
            pd.DataFrame: Un DataFrame de una sola fila preparado para la inferencia.
        """
        # Procesamiento de variables temporales y categóricas
        time_of_day = state.time_of_day.value if state.time_of_day.value else ""
        is_morning = 1 if "mañana" in time_of_day.lower() else 0
        is_evening = 1 if "tarde" in time_of_day.lower() else 0

        # Mapeo estricto a las features que exige XGBoost[cite: 3]
        features = {
            "Age": state.age.value,
            "Scholarship": state.scholarship.value,
            "Hipertension": state.hypertension.value,
            "Diabetes": state.diabetes.value,
            "Alcoholism": state.alcoholism.value,
            "Handcap": state.handicap.value,
            "SMS_received": state.sms_received.value,
            "Days_between": state.days_between.value,
            "Weekend": state.weekend.value,
            "Ratio_Faltas": state.history_no_show.value,
            "Gender_M": state.gender_m.value,
            "Scheduled_Time_of_Day_Evening": is_evening,
            "Scheduled_Time_of_Day_Morning": is_morning
        }

        # Convertimos los valores nulos de Python (None) a NaN para que XGBoost los maneje
        clean_features = {k: (v if v is not None else np.nan) for k, v in features.items()}
        
        df = pd.DataFrame([clean_features])
        logger.debug(f"DataFrame generado para predicción:\n{df.to_dict(orient='records')}")
        return df

    def _determine_risk_level(self, probability: float) -> str:
        """Clasifica la probabilidad numérica en una categoría de riesgo."""
        if probability >= PREDICT_CONFIG.risk_threshold_high:
            return "ALTO"
        elif probability >= PREDICT_CONFIG.risk_threshold_medium:
            return "MEDIO"
        else:
            return "BAJO"

    def predict(self, patient_state: PatientState) -> PredictionResult:
        """
        Ejecuta el modelo de predicción basado en el estado del paciente.
        Si el modelo físico falla, devuelve un fallback seguro en lugar de romper la app[cite: 3].
        
        Args:
            patient_state: Objeto que contiene las variables extraídas de la conversación[cite: 2].
            
        Returns:
            PredictionResult: Probabilidad estimada y nivel de riesgo[cite: 2].
        """
        try:
            logger.info("Iniciando proceso de predicción de No-Show.")
            
            model = self.loader.load_model()
            features_df = self._map_state_to_features(patient_state)
            
            # predict_proba devuelve una matriz [prob_clase_0, prob_clase_1]
            # Seleccionamos la columna 1, que representa el evento "No-Show" = True
            probabilities = model.predict_proba(features_df)
            probability_no_show = float(probabilities[0][1])
            
            risk_level = self._determine_risk_level(probability_no_show)
            
            logger.info(f"Predicción finalizada: Riesgo {risk_level} ({probability_no_show:.2f})")
            return PredictionResult(
                probability=probability_no_show,
                risk_level=risk_level
            )
            
        except FileNotFoundError:
            logger.warning("Archivo de modelo ausente. Utilizando estimación matemática de respaldo[cite: 3].")
            return self._fallback_prediction(patient_state)
        except Exception as e:
            logger.error(f"Fallo durante la predicción: {str(e)}", exc_info=True)
            return self._fallback_prediction(patient_state)

    def _fallback_prediction(self, state: PatientState) -> PredictionResult:
        """
        Implementa una heurística básica en caso de que XGBoost no esté disponible.
        Replicamos el comportamiento de respaldo documentado en api_2.py[cite: 3].
        """
        base_risk = 0.20
        
        # Penalizaciones básicas
        if state.history_no_show.value is not None and state.history_no_show.value > 0.3:
            base_risk += 0.40
        if state.days_between.value is not None and state.days_between.value > 14:
            base_risk += 0.15
        if state.sms_received.value == 0:
            base_risk += 0.10
            
        final_risk = min(base_risk, 0.99)
        return PredictionResult(
            probability=final_risk,
            risk_level=self._determine_risk_level(final_risk),
            is_fallback=True
        )