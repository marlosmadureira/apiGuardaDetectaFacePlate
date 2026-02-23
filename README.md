# Guarda – Controle de Acesso (Piloto)

Projeto piloto para **controle de entrada** em espaço privado, usando:

1. **Reconhecimento de placas** – Câmera identifica placas brasileiras (cinza/preta e Mercosul branca/azul), extrai os caracteres e pode encaminhar para um servidor externo.
2. **Reconhecimento facial** – Câmera reconhece o rosto, faz crop, gera embedding (vetor matemático) e armazena no banco para comparação futura e verificação de autorização.

Tudo rodando em **Docker** para teste local; produção pode ser configurada depois.

---

## Requisitos

- Docker e Docker Compose (para rodar tudo em containers)
- Python 3.10+ e pip (para execução local da API; o script não usa venv)
- Câmera (para uso com câmera no container: Linux com `/dev/video0`)

---

## Execução local (API na máquina, PostgreSQL no Docker ou instalado)

Um único script instala as dependências (com `pip --user`), cria `.env` e a pasta `data/` se precisar, sobe o PostgreSQL via Docker (se houver no projeto) e inicia a API — sem usar venv:

```bash
./scripts/run_local.sh
```

Na primeira execução instala as dependências e cria o `.env`; nas seguintes só confere e sobe a API (e o Postgres no Docker, se estiver parado).

API: **http://localhost:8000** | Docs: **http://localhost:8000/docs**

---

## Subir o projeto (Docker)

```bash
cd /home/madureira/Downloads/Guarda
docker compose build
docker compose up -d
```

API: **http://localhost:8000**  
Documentação: **http://localhost:8000/docs**

---

## Usar câmera no Docker (Linux)

Para os endpoints que capturam da câmera funcionarem dentro do container, passe o dispositivo de vídeo:

1. Edite `docker-compose.yml` e descomente:

```yaml
    # devices:
    #   - /dev/video0:/dev/video0
```

2. Reinicie:

```bash
docker compose up -d --build
```

Se a câmera não for detectada, use os endpoints que aceitam **upload de imagem** (veja abaixo).

---

## Fluxo resumido

### 1) Placas (Brasil)

- **Formatos suportados:**
  - **Antigo:** ABC-1234 (cinza, letras pretas)
  - **Mercosul:** ABC1D23 (branca, letras azuis) – 3 letras + 1 número + 1 letra + 2 números
- A API extrai o texto da placa e pode **encaminhar** o resultado para um endpoint externo (configurável por variável de ambiente).

### 2) Reconhecimento facial

- **Cadastro:** envia uma foto (ou usa a câmera); a API detecta o rosto, faz o crop e gera o **embedding** (128 números) e grava no banco.
- **Verificação:** nova foto ou frame da câmera é comparado com os embeddings cadastrados; retorna se há correspondência e com quem.

### 3) Controle de acesso

- **Pessoas** e **veículos** são cadastrados; **autorizações** vinculam pessoa (e opcionalmente um veículo).
- O endpoint de **verificação de acesso** usa placa + rosto (por upload ou câmera) e responde se a pessoa e o veículo estão autorizados a entrar.

---

## Endpoints principais

| Recurso | Método | Descrição |
|--------|--------|-----------|
| **Placa** | | |
| `/plate/capture` | POST | Captura da câmera, reconhece placa e opcionalmente encaminha |
| `/plate/capture/upload` | POST | Reconhece placa a partir de imagem enviada |
| **Rosto** | | |
| `/face/register/{person_id}` | POST | Cadastra rosto da pessoa (upload de imagem) |
| `/face/capture/register/{person_id}` | POST | Cadastra rosto usando câmera |
| `/face/verify` | POST | Verifica se o rosto (upload) está cadastrado |
| `/face/capture/verify` | POST | Verifica rosto usando câmera |
| **Cadastros** | | |
| `/persons` | GET/POST | Listar e criar pessoas |
| `/vehicles` | GET/POST | Listar e criar veículos (placas) |
| `/authorizations` | GET/POST | Listar e criar autorizações (pessoa + veículo) |
| **Acesso** | | |
| `/access/check` | POST | Verifica acesso (upload: `plate_image` e `face_image`) |
| `/access/check/camera` | POST | Verifica acesso usando câmera |

---

## Encaminhar placa para outro servidor

Configure no `docker-compose.yml` (ou em `.env`):

```yaml
environment:
  - PLATE_FORWARD_URL=https://seu-servidor.com/api/placas
  - PLATE_FORWARD_ENABLED=true
```

O POST enviado para esse URL terá o corpo JSON, por exemplo:

```json
{
  "plate": "ABC1D23",
  "format_type": "mercosul",
  "raw_text": "ABC1D23"
}
```

---

## Dados persistentes

O banco **PostgreSQL** roda no serviço `postgres`; os dados ficam no volume `postgres_data`. Para backup:

```bash
docker compose exec postgres pg_dump -U guarda guarda > backup_guarda.sql
```

Para restaurar:

```bash
docker compose exec -T postgres psql -U guarda guarda < backup_guarda.sql
```

O volume `guarda_data` é usado para arquivos da aplicação (ex.: fotos de rostos) em `/app/data`.

---

## Teste rápido sem câmera

1. Subir: `docker compose up -d`
2. Criar pessoa: `POST /persons` com `{"name": "Fulano", "document": "123"}`
3. Cadastrar rosto: `POST /face/register/1` com um arquivo de imagem (form-data `file`)
4. Criar veículo: `POST /vehicles` com `{"plate": "ABC1234", "description": "Carro"}`
5. Criar autorização: `POST /authorizations` com `{"person_id": 1, "vehicle_id": 1}`
6. Verificar acesso: `POST /access/check` com `plate_image` e `face_image` (arquivos)

Quando for colocar em produção, faça uma nova solicitação para ajustes de ambiente e segurança.
