"""
Módulo de configuración central del proyecto.
Gestiona variables de entorno, rutas del sistema y parámetros de los modelos.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent


def load_environment() -> None:
    """Carga variables desde .env si existen en la carpeta del chatbot."""
    env_path = BASE_DIR / ".env"
    if env_path.exists() and load_dotenv is not None:
        load_dotenv(dotenv_path=env_path, override=False)


load_environment()


@dataclass
class ModelConfig:
    """Configuración inmutable para los proveedores LLM."""
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    claude_api_key: str = os.getenv("CLAUDE_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    default_provider: str = os.getenv("DEFAULT_PROVIDER", "ollama")
    default_model: str = os.getenv("DEFAULT_MODEL", "qwen3:8b")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
    claude_base_url: str = os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    # El SDK oficial `groq` ya antepone "/openai/v1" a cada endpoint
    # internamente (p.ej. resuelve a "<base_url>/openai/v1/chat/completions"),
    # así que el base_url debe ser solo el host: incluir aquí "/openai/v1"
    # duplica el segmento y produce un 404 "Unknown request URL".
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com")
    temperature: float = 0.0
    max_tokens: int = 1500
    request_timeout: int = int(os.getenv("LLM_TIMEOUT", "120"))
    # El SDK oficial `openai` espera la base con el segmento de versión
    # incluido (a diferencia del SDK `groq`, que lo antepone él mismo):
    # sin "/v1" las peticiones fallan con 404.
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

@dataclass
class PredictionConfig:
    """Configuración para el modelo de No-Show.

    Usa el mismo artefacto que el dashboard de simulación
    (scripts/app.py): models/modelo_definitivo.joblib, un
    VotingClassifier (XGBClassifier + CatBoostClassifier +
    LogisticRegression) calibrado con CalibratedClassifierCV —
    reemplaza al antiguo modelo_campeon.json (XGBoost suelto), que solo
    tenía 13 de las 19 variables que este modelo espera.
    """
    model_path: Path = BASE_DIR.parent / "models" / "modelo_definitivo.joblib"
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