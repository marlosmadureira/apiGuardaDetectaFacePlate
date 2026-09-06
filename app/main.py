"""
Guarda - Controle de acesso a veículos e pessoas.
Fase 4: WebSocket ao vivo /ws/verify, métricas Prometheus /metrics.
"""
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.database import init_db, AsyncSessionLocal
from app.cache import embedding_cache
from app.logging_config import configure_logging
from app.routes import (
    plate_router,
    face_router,
    persons_router,
    vehicles_router,
    authorizations_router,
    access_router,
    stream_router,
    cameras_router,
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Headers para evitar cache do browser nas páginas HTML
_NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}

# Rate limiter (chave: IP do cliente)
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(debug=settings.debug)
    await init_db()
    Path(settings.face_photos_dir).mkdir(parents=True, exist_ok=True)
    embedding_cache._ttl = settings.embedding_cache_ttl
    # Aquecer cache de embeddings na inicialização
    async with AsyncSessionLocal() as db:
        await embedding_cache.get_embeddings(db)
    # Inicializar registry de câmeras (gera cameras.json padrão se não existir)
    from app.camera import get_registry
    get_registry(settings.cameras_config_path)
    yield
    # Liberar todas as câmeras do registry ao encerrar
    try:
        get_registry().release_all()
    except Exception:
        pass


app = FastAPI(
    title="Guarda - Controle de Acesso",
    description="API: reconhecimento de placas (Brasil/Mercosul) + reconhecimento facial (ArcFace).",
    version="4.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
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
app.include_router(stream_router)
app.include_router(cameras_router)


def _html(filename: str):
    """Retorna FileResponse com headers anti-cache para garantir sempre a versão mais recente."""
    path = STATIC_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Página não encontrada")
    return FileResponse(path, headers=_NO_CACHE)


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index, headers=_NO_CACHE)
    return {"app": settings.app_name, "docs": "/docs", "health": "/health"}


@app.get("/verificar")
async def verificar_page():
    return _html("verificar.html")


@app.get("/autorizacoes")
async def autorizacoes_page():
    return _html("autorizacoes.html")


@app.get("/placas")
async def placas_page():
    return _html("placas.html")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "4.0.0"}
