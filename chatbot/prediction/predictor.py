"""
Módulo principal de inferencia de Machine Learning.
Transforma el PatientState en el formato requerido por el modelo y ejecuta la predicción.
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from chatbot.conversation.patient_state import PatientState
from chatbot.config import PREDICT_CONFIG
from chatbot.prediction.model_loader import ModelLoader
from chatbot.prediction.prediction_result import PredictionResult

logger = logging.getLogger(__name__)

# Orden exacto que exige modelo_definitivo.joblib (ver `feature_names_in_`
# del propio estimador). A diferencia del antiguo modelo_campeon.json, que
# solo usaba 13 variables, este modelo también necesita variables de
# calendario (día de semana/mes de la cita y de la programación) y el
# historial de faltas como conteos absolutos, no solo una proporción.
MODEL_FEATURE_ORDER = [
    "Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap",
    "SMS_received", "Days_between", "Appointment_Day_of_Week", "Scheduled_Day_of_Week",
    "Weekend", "Appointment_Month", "Scheduled_Month", "Faltas_Previas", "Citas_Previas",
    "Ratio_Faltas", "Gender_M", "Scheduled_Time_of_Day_Evening", "Scheduled_Time_of_Day_Morning",
]

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
        esperadas por el modelo entrenado (modelo_definitivo.joblib).

        Args:
            state: Estado actual del paciente.

        Returns:
            pd.DataFrame: Un DataFrame de una sola fila preparado para la inferencia.
        """
        # El chat nunca pregunta fechas de calendario en crudo; las derivamos
        # de "hoy" + los días de antelación para que el mes/día de la cita
        # queden siempre consistentes entre sí (el LLM no los inventa por
        # separado, evitando p.ej. un "fin de semana" que no encaje con el
        # día de la semana calculado).
        now = datetime.now()
        scheduled_weekday = now.weekday()  # 0=lunes ... 6=domingo
        scheduled_month = now.month

        days_between = state.days_between.value
        if days_between is not None:
            appointment_dt = now + timedelta(days=int(days_between))
            appointment_weekday = appointment_dt.weekday()
            appointment_month = appointment_dt.month
            weekend = 1 if appointment_weekday >= 5 else 0
        else:
            appointment_weekday = None
            appointment_month = None
            weekend = state.weekend.value

        # El modelo espera Faltas_Previas/Citas_Previas como conteos y
        # Ratio_Faltas por separado. Si el paciente dio los conteos exactos
        # los usamos (más fiel); si solo dio una proporción aproximada
        # ("falto la mitad de las veces", capturada en history_no_show), la
        # usamos solo para Ratio_Faltas. Sin ningún dato, 0 en todo — que es
        # justo lo que vale Ratio_Faltas en el dataset de entrenamiento
        # cuando Citas_Previas es 0 (paciente sin historial).
        citas_previas = state.citas_previas.value
        faltas_previas = state.faltas_previas.value
        if citas_previas is not None and faltas_previas is not None and citas_previas > 0:
            ratio_faltas = faltas_previas / citas_previas
        elif state.history_no_show.value is not None:
            ratio_faltas = state.history_no_show.value
        else:
            ratio_faltas = None

        # Procesamiento de variables temporales y categóricas
        time_of_day = state.time_of_day.value if state.time_of_day.value else ""
        is_morning = 1 if "mañana" in time_of_day.lower() else 0
        is_evening = 1 if "tarde" in time_of_day.lower() else 0

        features = {
            "Age": state.age.value,
            "Scholarship": state.scholarship.value,
            "Hipertension": state.hypertension.value,
            "Diabetes": state.diabetes.value,
            "Alcoholism": state.alcoholism.value,
            "Handcap": state.handicap.value,
            "SMS_received": state.sms_received.value,
            "Days_between": days_between,
            "Appointment_Day_of_Week": appointment_weekday,
            "Scheduled_Day_of_Week": scheduled_weekday,
            "Weekend": weekend,
            "Appointment_Month": appointment_month,
            "Scheduled_Month": scheduled_month,
            "Faltas_Previas": faltas_previas if faltas_previas is not None else 0,
            "Citas_Previas": citas_previas if citas_previas is not None else 0,
            "Ratio_Faltas": ratio_faltas if ratio_faltas is not None else 0.0,
            "Gender_M": state.gender_m.value,
            "Scheduled_Time_of_Day_Evening": is_evening,
            "Scheduled_Time_of_Day_Morning": is_morning,
        }

        # Convertimos los valores nulos de Python (None) a NaN para que el modelo los maneje
        clean_features = {k: (v if v is not None else np.nan) for k, v in features.items()}

        df = pd.DataFrame([clean_features])[MODEL_FEATURE_ORDER]
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
        Implementa una heurística básica en caso de que el modelo no esté disponible.
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