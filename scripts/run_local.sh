#!/usr/bin/env bash
# Script único: prepara o ambiente se precisar (venv, .env, data), sobe o Postgres no Docker
# se existir no projeto e inicia a API. Uso: ./scripts/run_local.sh

set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"

echo "=== Guarda - Execução local ==="
echo ""

# 1) Ambiente virtual
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[1/4] Criando ambiente virtual ($VENV_DIR)..."
  "$PYTHON" -m venv "$VENV_DIR"
else
  echo "[1/4] Ambiente virtual ok."
fi

# 2) Dependências (sempre instala/atualiza, rápido quando já está ok)
echo "[2/4] Verificando dependências..."
"$ROOT/$VENV_DIR/bin/pip" install --upgrade pip -q
"$ROOT/$VENV_DIR/bin/pip" install -r requirements.txt -q

# 3) .env
if [[ ! -f "$ROOT/.env" ]]; then
  echo "[3/4] Criando .env a partir de .env.example..."
  cp "$ROOT/.env.example" "$ROOT/.env"
  if grep -q '^# DATABASE_URL=' "$ROOT/.env" 2>/dev/null; then
    sed -i 's|^# DATABASE_URL=.*|DATABASE_URL=postgresql+asyncpg://guarda:guarda@localhost:5432/guarda|' "$ROOT/.env"
  fi
else
  echo "[3/4] .env ok."
fi

# 4) Diretório de dados
mkdir -p "$ROOT/data"
echo "[4/4] Diretório data/ ok."
echo ""

# 5) PostgreSQL no Docker (se o projeto tiver postgres no docker-compose)
if [[ -f "$ROOT/docker-compose.yml" ]] && grep -q "postgres:" "$ROOT/docker-compose.yml" 2>/dev/null; then
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    echo "Subindo PostgreSQL (Docker)..."
    (cd "$ROOT" && docker compose up -d postgres)
    echo "Aguardando Postgres ficar pronto..."
    sleep 4
    echo ""
  fi
fi

echo "Iniciando API (uvicorn)..."
echo "API: http://localhost:8000  |  Docs: http://localhost:8000/docs"
echo ""
exec "$ROOT/$VENV_DIR/bin/uvicorn" main:app --reload --host 0.0.0.0 --port 8000
