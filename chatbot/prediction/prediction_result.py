"""
Módulo que define la estructura de datos para los resultados de predicción.
Garantiza un contrato estricto entre la capa de predicción y los consumidores.
"""

from dataclasses import dataclass

@dataclass
class PredictionResult:
    """
    Encapsula el resultado de la evaluación de riesgo de No-Show.
    """
    probability: float
    risk_level: str
    is_fallback: bool = False
    
    def __str__(self) -> str:
        return f"Riesgo: {self.risk_level} (Probabilidad: {self.probability:.2%})"