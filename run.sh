#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
LOGS="$ROOT/logs"
API_PID_FILE="$LOGS/api.pid"

usage() {
  cat <<'EOF'
Uso: ./run.sh <comando>

Instalación:
  setup       Crea el entorno virtual e instala todas las dependencias (raíz + chatbot)
  install     Reinstala/actualiza dependencias en un entorno virtual ya existente

Aplicaciones:
  api         API FastAPI de reservas (src/api_2.py)              -> http://localhost:8000
  dashboard   Dashboard Streamlit de simulación (scripts/app.py)  -> http://localhost:8501
  chatbot     Chatbot de admisión (chatbot/app.py)                -> http://localhost:8501
  portal      Portal unificado: chatbot + reservas (app_unificado.py) -> http://localhost:8501

Simulaciones (sin UI):
  sim         Backtest histórico empírico (scripts/simulacion.py)
  mc          Simulación Monte Carlo, modelo crudo (scripts/mc.py)
  mc2         Simulación Monte Carlo con calibración isotónica (scripts/mc_2.py)

Todo junto:
  all         Arranca la API en segundo plano y el portal unificado en primer plano
  stop        Detiene la API que quedó corriendo en segundo plano (arrancada con 'all')

  help        Muestra esta ayuda
EOF
  exit "${1:-1}"
}

check_venv() {
  if [ ! -d "$VENV" ]; then
    echo "No existe el entorno virtual. Ejecuta primero: ./run.sh setup" >&2
    exit 1
  fi
}

ensure_chatbot_env() {
  if [ ! -f "$ROOT/chatbot/.env" ] && [ -f "$ROOT/chatbot/.env.example" ]; then
    echo "Creando chatbot/.env a partir de chatbot/.env.example (por defecto: Ollama local)"
    cp "$ROOT/chatbot/.env.example" "$ROOT/chatbot/.env"
  fi
}

case "${1:-}" in
  setup)
    echo "Creando entorno virtual en $VENV"
    python3 -m venv "$VENV"
    "$PY" -m pip install --upgrade pip
    echo "Instalando dependencias principales..."
    "$PY" -m pip install -r "$ROOT/requirements.txt"
    if [ -f "$ROOT/chatbot/requirements.txt" ]; then
      echo "Instalando dependencias del chatbot..."
      "$PY" -m pip install -r "$ROOT/chatbot/requirements.txt"
    fi
    ensure_chatbot_env
    echo "Setup completo. Activa el entorno con: source $VENV/bin/activate"
    echo "Luego usa './run.sh help' para ver los comandos disponibles."
    ;;

  install)
    check_venv
    "$PIP" install -r "$ROOT/requirements.txt"
    if [ -f "$ROOT/chatbot/requirements.txt" ]; then
      "$PIP" install -r "$ROOT/chatbot/requirements.txt"
    fi
    ;;

  api)
    check_venv
    echo "Iniciando API FastAPI (uvicorn) en http://localhost:8000"
    exec "$PY" -m uvicorn src.api_2:app --host 0.0.0.0 --port 8000 --reload
    ;;

  dashboard)
    check_venv
    echo "Iniciando dashboard Streamlit (scripts/app.py)"
    echo "Nota: requiere models/modelo_definitivo.joblib, que puede no estar presente en el repo."
    exec "$PY" -m streamlit run scripts/app.py
    ;;

  chatbot)
    check_venv
    ensure_chatbot_env
    echo "Iniciando chatbot de admisión (chatbot/app.py)"
    exec "$PY" -m streamlit run chatbot/app.py
    ;;

  portal)
    check_venv
    ensure_chatbot_env
    echo "Iniciando portal unificado (chatbot + reservas en app_unificado.py)"
    exec "$PY" -m streamlit run app_unificado.py
    ;;

  sim)
    check_venv
    exec "$PY" scripts/simulacion.py
    ;;

  mc)
    check_venv
    exec "$PY" scripts/mc.py
    ;;

  mc2)
    check_venv
    exec "$PY" scripts/mc_2.py
    ;;

  all)
    check_venv
    ensure_chatbot_env
    mkdir -p "$LOGS"
    if [ -f "$API_PID_FILE" ] && kill -0 "$(cat "$API_PID_FILE")" 2>/dev/null; then
      echo "La API ya está corriendo (PID $(cat "$API_PID_FILE"))."
    else
      echo "Iniciando API en segundo plano (log: $LOGS/api.log)..."
      nohup "$PY" -m uvicorn src.api_2:app --host 0.0.0.0 --port 8000 > "$LOGS/api.log" 2>&1 &
      echo $! > "$API_PID_FILE"
      sleep 1
    fi
    echo "Iniciando portal unificado en primer plano (Ctrl+C lo detiene; la API sigue corriendo, usa './run.sh stop' para pararla)"
    exec "$PY" -m streamlit run app_unificado.py
    ;;

  stop)
    if [ -f "$API_PID_FILE" ] && kill -0 "$(cat "$API_PID_FILE")" 2>/dev/null; then
      kill "$(cat "$API_PID_FILE")"
      rm -f "$API_PID_FILE"
      echo "API detenida."
    else
      echo "No hay ninguna API en segundo plano arrancada con './run.sh all'."
    fi
    ;;

  help)
    usage 0
    ;;

  "")
    usage 1
    ;;

  *)
    usage
    ;;
esac
