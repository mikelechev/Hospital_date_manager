"""
Módulo de configuración central del proyecto.
Gestiona variables de entorno, rutas del sistema y parámetros de los modelos.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent

@dataclass
class ModelConfig:
    """Configuración inmutable para los proveedores LLM."""
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    default_provider: str = os.getenv("DEFAULT_PROVIDER", "ollama")
    default_model: str = os.getenv("DEFAULT_MODEL", "qwen3:8b")
    temperature: float = 0.0
    max_tokens: int = 1500
    # Nuevo parámetro extraído para cumplir las reglas de arquitectura[cite: 1]
    request_timeout: int = int(os.getenv("LLM_TIMEOUT", "120"))

@dataclass
class PredictionConfig:
    """Configuración para el modelo XGBoost de No-Show."""
    model_path: Path = BASE_DIR / "models" / "modelo_campeon.json"
    risk_threshold_high: float = 0.8
    risk_threshold_medium: float = 0.5

@dataclass
class AppConfig:
    """Configuración general de la aplicación y sistema de logging."""
    logs_dir: Path = BASE_DIR / "logs"
    exports_dir: Path = BASE_DIR / "exports"
    log_level: int = logging.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logging(config: AppConfig) -> None:
    """Configura el sistema de logging centralizado."""
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    config.exports_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = config.logs_dir / "chatbot_hospital.log"
    
    logging.basicConfig(
        level=config.log_level,
        format=config.log_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

LLM_CONFIG = ModelConfig()
PREDICT_CONFIG = PredictionConfig()
APP_CONFIG = AppConfig()

setup_logging(APP_CONFIG)