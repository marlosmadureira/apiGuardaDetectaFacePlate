"""
Guarda - Controle de acesso a veículos e pessoas.
Piloto: reconhecimento de placa (Brasil/Mercosul) + reconhecimento facial.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
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
    return {
        "app": get_settings().app_name,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
