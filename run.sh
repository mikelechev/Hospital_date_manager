#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

usage() { echo "Usage: $0 {setup|install|api|dashboard|chatbot|all|help}"; exit 1; }

case "${1:-}" in
  setup)
    echo "Creating virtualenv at $VENV"
    python3 -m venv "$VENV"
    "$PY" -m pip install --upgrade pip
    echo "Installing main requirements..."
    "$PY" -m pip install -r "$ROOT/requirements.txt"
    if [ -f "$ROOT/chatbot/requirements.txt" ]; then
      echo "Installing chatbot requirements..."
      "$PY" -m pip install -r "$ROOT/chatbot/requirements.txt"
    fi
    echo "Setup complete. Activate with: source $VENV/bin/activate"
    ;;
  install)
    if [ ! -d "$VENV" ]; then echo "Run '$0 setup' first"; exit 1; fi
    "$PIP" install -r "$ROOT/requirements.txt"
    ;;
  api)
    if [ ! -d "$VENV" ]; then echo "Run '$0 setup' first"; exit 1; fi
    echo "Starting FastAPI (uvicorn) on http://0.0.0.0:8000"
    exec "$PY" -m uvicorn src.api_2:app --host 0.0.0.0 --port 8000 --reload
    ;;
  dashboard)
    if [ ! -d "$VENV" ]; then echo "Run '$0 setup' first"; exit 1; fi
    echo "Starting Streamlit dashboard (scripts/app.py)"
    exec "$PY" -m streamlit run scripts/app.py
    ;;
  chatbot)
    if [ ! -d "$VENV" ]; then echo "Run '$0 setup' first"; exit 1; fi
    echo "Starting Streamlit chatbot (chatbot/app.py)"
    exec "$PY" -m streamlit run chatbot/app.py
    ;;
  all)
    "$0" setup
    echo "Starting API in background..."
    mkdir -p "$ROOT/logs"
    nohup "$PY" -m uvicorn src.api_2:app --host 0.0.0.0 --port 8000 > "$ROOT/logs/api.log" 2>&1 &
    echo "API started; run '$0 dashboard' and '$0 chatbot' in other terminals or use tmux."
    ;;
  help|"" )
    usage
    ;;
  *)
    usage
    ;;
esac
