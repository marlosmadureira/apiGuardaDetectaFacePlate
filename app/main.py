"""
Guarda - Controle de acesso a veículos e pessoas.
Piloto: reconhecimento de placa (Brasil/Mercosul) + reconhecimento facial.
"""
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import get_settings
from app.database import init_db
from app.routes import (
    plate_router,
    face_router,
    persons_router,
    vehicles_router,
    authorizations_router,
    access_router,
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Garante que a pasta de fotos de rosto existe para capturas futuras
    from pathlib import Path
    from app.config import get_settings
    Path(get_settings().face_photos_dir).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Guarda - Controle de Acesso",
    description="API piloto: reconhecimento de placas (Brasil/Mercosul) e reconhecimento facial para fluxo de entrada de veículos e pessoas.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(plate_router)
app.include_router(face_router)
app.include_router(persons_router)
app.include_router(vehicles_router)
app.include_router(authorizations_router)
app.include_router(access_router)


@app.get("/")
async def root():
    """Frontend: cadastro de rosto com câmera ao vivo."""
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {
        "app": get_settings().app_name,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/verificar")
async def verificar_page():
    """Tela de verificação de acesso: câmera ao vivo e indicação se a pessoa está autorizada."""
    path = STATIC_DIR / "verificar.html"
    if path.is_file():
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Página não encontrada")


@app.get("/autorizacoes")
async def autorizacoes_page():
    """Tela de cadastro de autorizações: Pedestre, Veículo ou Pedestre e Veículo."""
    path = STATIC_DIR / "autorizacoes.html"
    if path.is_file():
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Página não encontrada")


@app.get("/placas")
async def placas_page():
    """Tela de cadastro de placas (veículos): lista, formulário e captura pela câmera."""
    path = STATIC_DIR / "placas.html"
    if path.is_file():
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Página não encontrada")


@app.get("/health")
async def health():
    return {"status": "ok"}
