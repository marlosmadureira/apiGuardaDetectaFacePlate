#!/usr/bin/env bash
# Script único: prepara o ambiente (.env, data), sobe o Postgres no Docker se existir
# e inicia a API. Sem venv: usa o Python do sistema. Uso: ./scripts/run_local.sh

set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"
PYTHON="${PYTHON:-python3}"

echo "=== Guarda - Execução local (sem venv) ==="
echo ""

# 0) CMake é necessário para compilar dlib (face_recognition)
if ! command -v cmake >/dev/null 2>&1; then
  echo "Erro: CMake não encontrado. O pacote face_recognition (dlib) precisa do CMake para compilar."
  echo ""
  echo "No Ubuntu/Debian, instale com:"
  echo "  sudo apt update"
  echo "  sudo apt install cmake build-essential"
  echo ""
  echo "Depois rode este script de novo."
  exit 1
fi

# 1) Dependências (pip --user para não precisar de venv nem sudo)
echo "[1/3] Verificando dependências..."
"$PYTHON" -m pip install --user --upgrade pip -q
"$PYTHON" -m pip install --user -r requirements.txt -q

# 2) .env
if [[ ! -f "$ROOT/.env" ]]; then
  echo "[2/3] Criando .env a partir de .env.example..."
  cp "$ROOT/.env.example" "$ROOT/.env"
  if grep -q '^# DATABASE_URL=' "$ROOT/.env" 2>/dev/null; then
    sed -i 's|^# DATABASE_URL=.*|DATABASE_URL=postgresql+asyncpg://guarda:guarda@localhost:5432/guarda|' "$ROOT/.env"
  fi
else
  echo "[2/3] .env ok."
fi

# 3) Diretório de dados
mkdir -p "$ROOT/data"
echo "[3/3] Diretório data/ ok."
echo ""

# 4) PostgreSQL no Docker (se o projeto tiver postgres no docker-compose)
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
exec "$PYTHON" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
