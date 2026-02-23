#!/usr/bin/env bash
# Setup do ambiente para execução local do Guarda (venv, dependências, .env, dados).
# Uso: ./scripts/setup_local.sh   ou   bash scripts/setup_local.sh

set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"

echo "=== Guarda - Setup local ==="
echo "Diretório do projeto: $ROOT"
echo ""

# 1) Virtual environment
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[1/4] Criando ambiente virtual em $VENV_DIR ..."
  "$PYTHON" -m venv "$VENV_DIR"
else
  echo "[1/4] Ambiente virtual já existe: $VENV_DIR"
fi

# 2) Dependências
echo "[2/4] Instalando dependências ..."
"$ROOT/$VENV_DIR/bin/pip" install --upgrade pip -q
"$ROOT/$VENV_DIR/bin/pip" install -r requirements.txt -q
echo "      Dependências instaladas."
echo ""

# 3) .env
if [[ ! -f "$ROOT/.env" ]]; then
  echo "[3/4] Criando .env a partir de .env.example ..."
  cp "$ROOT/.env.example" "$ROOT/.env"
  # Garante DATABASE_URL ativa para local (guarda:guarda@localhost:5432/guarda)
  if grep -q '^# DATABASE_URL=' "$ROOT/.env" 2>/dev/null; then
    sed -i 's|^# DATABASE_URL=.*|DATABASE_URL=postgresql+asyncpg://guarda:guarda@localhost:5432/guarda|' "$ROOT/.env"
  fi
  echo "      Arquivo .env criado. Ajuste DATABASE_URL se o seu PostgreSQL usar outro usuário/senha/porta."
else
  echo "[3/4] .env já existe (não foi alterado)."
fi
echo ""

# 4) Diretório de dados (fotos de rostos, etc.)
DATA_DIR="$ROOT/data"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "[4/4] Criando diretório de dados: $DATA_DIR"
  mkdir -p "$DATA_DIR"
else
  echo "[4/4] Diretório de dados já existe: $DATA_DIR"
fi
echo ""

# Lembrete PostgreSQL
echo "--- PostgreSQL ---"
echo "A API espera PostgreSQL em localhost:5432 com banco 'guarda' e usuário/senha 'guarda/guarda'."
echo ""
echo "Opção A - PostgreSQL só via Docker:"
echo "  docker compose up -d postgres"
echo ""
echo "Opção B - PostgreSQL já instalado na máquina:"
echo "  Crie o banco e usuário, por exemplo:"
echo "  sudo -u postgres createuser -P guarda"
echo "  sudo -u postgres createdb -O guarda guarda"
echo ""
echo "Se usar outro host/porta/usuário, edite o arquivo .env (DATABASE_URL)."
echo ""

# Comando para rodar
echo "--- Como rodar a API ---"
echo "  source $VENV_DIR/bin/activate"
echo "  uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Ou use o atalho:"
echo "  ./scripts/run_local.sh"
echo ""
echo "=== Setup concluído ==="
