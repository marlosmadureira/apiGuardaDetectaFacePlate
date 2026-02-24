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
POSTGRES_STARTED=
if [[ -f "$ROOT/docker-compose.yml" ]] && grep -q "postgres:" "$ROOT/docker-compose.yml" 2>/dev/null; then
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    echo "Subindo PostgreSQL (Docker)..."
    (cd "$ROOT" && docker compose up -d postgres)
    POSTGRES_STARTED=1
  fi
fi

# 5) Esperar Postgres aceitar conexões (quando subimos pelo Docker ou sempre, para evitar Connection refused)
wait_for_postgres() {
  echo "Aguardando Postgres ficar pronto..."
  "$PYTHON" - "$ROOT/.env" << 'PY'
import asyncio
import os
import sys

def get_dsn():
    env_path = sys.argv[1] if len(sys.argv) > 1 else None
    dsn = "postgresql://guarda:guarda@localhost:5432/guarda"
    if env_path and os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if line.startswith("DATABASE_URL=") and "=" in line:
                    url = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if url.startswith("postgresql+asyncpg://"):
                        url = "postgresql://" + url.split("://", 1)[1]
                    if url.startswith("postgresql://"):
                        dsn = url
                    break
    return dsn

async def wait():
    try:
        import asyncpg
    except ImportError:
        print("asyncpg não instalado; aguardando 8s e seguindo...", file=sys.stderr)
        await asyncio.sleep(8)
        return
    dsn = get_dsn()
    for i in range(45):
        try:
            conn = await asyncpg.connect(dsn)
            await conn.close()
            print("Postgres pronto.")
            return
        except Exception:
            await asyncio.sleep(1)
    print("Erro: Postgres não respondeu em 45s. Verifique o container.", file=sys.stderr)
    sys.exit(1)

asyncio.run(wait())
PY
}

if [[ -n "$POSTGRES_STARTED" ]]; then
  wait_for_postgres
fi
echo ""

echo "Iniciando API (uvicorn)..."
echo "API: http://localhost:8000  |  Docs: http://localhost:8000/docs"
echo ""
exec "$PYTHON" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
