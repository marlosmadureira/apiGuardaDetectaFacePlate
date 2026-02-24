# Guarda – Controle de Acesso (Piloto)

Projeto piloto para **controle de entrada** em espaço privado, usando:

1. **Reconhecimento de placas** – Câmera identifica placas brasileiras (cinza/preta e Mercosul branca/azul), extrai os caracteres e pode encaminhar para um servidor externo.
2. **Reconhecimento facial** – Câmera reconhece o rosto, faz crop, gera embedding (vetor matemático) e armazena no banco para comparação futura e verificação de autorização.

Tudo rodando em **Docker** para teste local; produção pode ser configurada depois.

---

## Requisitos

- Docker e Docker Compose (para rodar tudo em containers)
- Python 3.10+ e pip (para execução local da API; o script não usa venv)
- CMake e build-essential (para compilar `dlib`/`face_recognition`). No Ubuntu/Debian: `sudo apt install cmake build-essential`
- Câmera (para uso com câmera no container: Linux com `/dev/video0`)

---

## Execução local (API na máquina, PostgreSQL no Docker ou instalado)

Um único script instala as dependências (com `pip --user`), cria `.env` e a pasta `data/` se precisar, sobe o PostgreSQL via Docker (se houver no projeto) e inicia a API — sem usar venv:

```bash
./scripts/run_local.sh
```

Na primeira execução instala as dependências e cria o `.env`; nas seguintes só confere e sobe a API (e o Postgres no Docker, se estiver parado).

**Interface no navegador:** abra **http://localhost:8000** para cadastrar seu rosto: câmera ao vivo, botão "Capturar rosto" envia a foto para a API (que extrai o rosto e grava o embedding para futura verificação).  
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

- **Autorização** é sempre de um tipo por registro: **entrada a pé** (só pessoa) ou **entrada com veículo** (pessoa + veículo). Nunca “veículo e pessoa” obrigatórios juntos.
  - **Entrada a pé:** autorização com `vehicle_id = null` → verificação apenas **facial**.
  - **Entrada com veículo:** autorização com `vehicle_id` preenchido → verificação **facial + placa** (a captura já faz a leitura da placa).
- A **verificação de acesso** recebe rosto (e opcionalmente placa): só facial para entrada a pé; facial + placa para entrada com veículo.

---

## Endpoints principais

| Recurso | Método | Descrição |
|--------|--------|-----------|
| **Placa** | | |
| `/plate/capture` | POST | Captura da câmera, reconhece placa e opcionalmente encaminha |
| `/plate/capture/upload` | POST | Reconhece placa a partir de imagem enviada |
| **Rosto** | | |
| `/face/register/{person_id}` | POST | Cadastra rosto da pessoa (upload de imagem); salva foto para consulta |
| `/face/capture/register/{person_id}` | POST | Cadastra rosto usando câmera; salva foto para consulta |
| `/face/verify` | POST | Verifica se o rosto (upload) está cadastrado |
| `/face/capture/verify` | POST | Verifica rosto usando câmera |
| `/face/photo/{person_id}` | GET | Retorna a foto do rosto cadastrada (para consultas futuras) |
| **Cadastros** | | |
| `/persons` | GET/POST | Listar e criar pessoas |
| `/vehicles` | GET/POST | Listar e criar veículos (placas) |
| `/authorizations` | GET/POST | Listar e criar autorizações: só pessoa (a pé) ou pessoa + veículo |
| **Acesso** | | |
| `/access/check` | POST | Verifica acesso: só face_image (a pé) ou face_image + plate_image (com veículo) |
| `/access/check/camera` | POST | Verifica acesso usando câmera (placa + rosto) |

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

## Teste local: cadastro de rosto pelo navegador (recomendado)

1. Suba a API: `./scripts/run_local.sh`
2. Abra no navegador: **http://localhost:8000**
3. Crie uma pessoa (nome e documento opcional) ou selecione uma existente.
4. Clique em **Iniciar câmera**: a imagem ao vivo aparece na tela.
5. Posicione o rosto no quadro e clique em **Capturar rosto**. A foto é enviada para a API, que detecta o rosto, gera o embedding e salva a imagem para futura verificação.
6. Depois use **POST /face/capture/verify** (em /docs) ou a verificação por upload para confirmar que é você.

---

## Teste local: capturar rosto pela câmera do notebook (Swagger)

Com a API rodando **na sua máquina** (`./scripts/run_local.sh`), a câmera do notebook é usada automaticamente (índice 0). Passos:

1. **Subir a API** (e o Postgres, se usar Docker):
   ```bash
   ./scripts/run_local.sh
   ```

2. **Abrir a documentação interativa:**  
   [http://localhost:8000/docs](http://localhost:8000/docs)

3. **Criar uma pessoa**  
   Em **POST /persons**, clique em "Try it out", use o corpo:
   ```json
   {"name": "Seu Nome", "document": ""}
   ```
   Execute e anote o `id` retornado (ex.: `1`).

4. **Cadastrar seu rosto pela câmera**  
   Em **POST /face/capture/register/{person_id}**, clique em "Try it out", informe o `person_id` (ex.: `1`) e clique em "Execute".  
   A API vai capturar um frame da câmera na hora. Deixe o rosto visível e centralizado; em alguns segundos deve retornar sucesso.

5. **Testar reconhecimento (validação)**  
   Em **POST /face/capture/verify**, "Try it out" → "Execute". A API captura da câmera e responde se o rosto corresponde a alguém cadastrado (e a quem).

6. **Consultar a foto salva**  
   A captura do rosto (câmera ou upload) é salva em `data/faces/`. Para ver a foto cadastrada: **GET /face/photo/{person_id}** (ex.: abrir no navegador `http://localhost:8000/face/photo/1`).

A câmera usa um breve tempo de ajuste (~1 s) antes de capturar, para melhorar luz e foco. Se der "Câmera não disponível", verifique se outro programa não está usando a webcam e, no Linux, se o usuário tem acesso a `/dev/video0` (ex.: `ls -l /dev/video0`).

---

## Teste rápido sem câmera

1. Subir: `docker compose up -d`
2. Criar pessoa: `POST /persons` com `{"name": "Fulano", "document": "123"}`
3. Cadastrar rosto: `POST /face/register/1` com um arquivo de imagem (form-data `file`)
4. Criar veículo: `POST /vehicles` com `{"plate": "ABC1234", "description": "Carro"}`
5. Criar autorização: `POST /authorizations` com `{"person_id": 1, "vehicle_id": 1}`
6. Verificar acesso: `POST /access/check` com `plate_image` e `face_image` (arquivos)

Quando for colocar em produção, faça uma nova solicitação para ajustes de ambiente e segurança.
