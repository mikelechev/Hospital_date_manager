#!/bin/bash

# Script de integración y ejecución para entornos Linux
# Detiene la ejecución si ocurre algún error
set -e

echo "🏥 Iniciando integración del Hospital Conversational Chatbot..."

# 1. Crear entorno virtual si no existe
if [ ! -d ".venv" ]; then
    echo "⚙️ Creando entorno virtual de Python..."
    python3 -m venv .venv
fi

# 2. Activar entorno virtual
source .venv/bin/activate

# 3. Instalar dependencias requeridas
echo "📦 Instalando dependencias desde requirements.txt..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 4. Verificar configuración de entorno
if [ ! -f ".env" ]; then
    echo "⚠️ Archivo .env no encontrado. Creando uno a partir de .env.example..."
    cp .env.example .env
    echo "Por favor, actualiza el archivo .env con tus credenciales si vas a usar Gemini."
fi

# 5. Exportar variables de entorno para la sesión actual
export $(grep -v '^#' .env | xargs)

# 6. Configurar Ollama para usar GPU local cuando se use el proveedor Ollama
#    Mantiene el mismo modelo, pero fuerza aceleración en la RTX 4060.
export OLLAMA_NO_CLOUD=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_IGPU_ENABLE=0
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_GPU_OVERHEAD=2000000000

# 7. Iniciar el servidor local de Ollama si el proveedor configurado es Ollama
if [ "${DEFAULT_PROVIDER,,}" = "ollama" ]; then
    echo "🧠 Iniciando Ollama local en segundo plano..."
    ollama serve >/dev/null 2>&1 &
    OLLAMA_PID=$!
    trap 'echo "🚪 Deteniendo Ollama..."; kill "$OLLAMA_PID" 2>/dev/null || true' EXIT
    sleep 3
fi

# 8. Ejecutar la interfaz gráfica
echo "🚀 Levantando la interfaz de Streamlit..."
streamlit run app.py