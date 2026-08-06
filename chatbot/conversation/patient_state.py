"""
Módulo que define el estado estructurado del paciente utilizando Pydantic.
Actúa como la única fuente de verdad del sistema.
"""

import logging
from datetime import datetime
from typing import Optional, Generic, TypeVar, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar('T')

class PatientVariable(BaseModel, Generic[T]):
    """
    Representa una variable individual del paciente con su metadata asociada.
    """
    value: Optional[T] = None
    confidence: float = 0.0
    source: str = "unknown"
    last_update: Optional[datetime] = None

    def update(self, value: T, confidence: float, source: str) -> None:
        """Actualiza el valor si la nueva confianza es mayor o igual a la actual."""
        if confidence >= self.confidence:
            self.value = value
            self.confidence = confidence
            self.source = source
            self.last_update = datetime.now()
            logger.debug(f"Variable actualizada: {value} (Confianza: {confidence})")

class PatientState(BaseModel):
    """
    Estado global estructurado del paciente.
    Contiene todas las variables necesarias para el modelo XGBoost[cite: 4].
    """
    age: PatientVariable[int] = Field(default_factory=PatientVariable)
    gender_m: PatientVariable[int] = Field(default_factory=PatientVariable)
    hypertension: PatientVariable[int] = Field(default_factory=PatientVariable)
    diabetes: PatientVariable[int] = Field(default_factory=PatientVariable)
    alcoholism: PatientVariable[int] = Field(default_factory=PatientVariable)
    handicap: PatientVariable[int] = Field(default_factory=PatientVariable)
    scholarship: PatientVariable[int] = Field(default_factory=PatientVariable)
    sms_received: PatientVariable[int] = Field(default_factory=PatientVariable)
    history_no_show: PatientVariable[float] = Field(default_factory=PatientVariable)
    days_between: PatientVariable[int] = Field(default_factory=PatientVariable)
    weekend: PatientVariable[int] = Field(default_factory=PatientVariable)
    time_of_day: PatientVariable[str] = Field(default_factory=PatientVariable)
    consultation_reason: PatientVariable[str] = Field(default_factory=PatientVariable)

    def get_missing_critical_fields(self) -> list[str]:
        """
        Evalúa qué campos críticos faltan o tienen baja confianza.
        
        Returns:
            list[str]: Lista de nombres de variables pendientes.
        """
        missing = []
        # Definimos umbral de confianza mínimo
        threshold = 0.6
        
        for field_name, field_obj in self.__dict__.items():
            if isinstance(field_obj, PatientVariable):
                if field_obj.value is None or field_obj.confidence < threshold:
                    missing.append(field_name)
                    
        return missing

    def is_ready_for_prediction(self) -> bool:
        """Determina si hay suficientes datos para ejecutar el predictor[cite: 4]."""
        return len(self.get_missing_critical_fields()) <= 3  # Permitimos cierto margen de incertidumbre