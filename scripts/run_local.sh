#!/usr/bin/env bash
# Atalho para rodar a API em modo desenvolvimento (reload).
# Execute ./scripts/setup_local.sh antes da primeira vez.

set -e
cd "$(dirname "$0")/.."
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Ambiente virtual não encontrado. Execute primeiro: ./scripts/setup_local.sh"
  exit 1
fi

echo "Iniciando Guarda (uvicorn)..."
exec "$VENV_DIR/bin/uvicorn" main:app --reload --host 0.0.0.0 --port 8000
