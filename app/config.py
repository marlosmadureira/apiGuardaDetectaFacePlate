"""Configurações da aplicação."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Configurações carregadas de variáveis de ambiente."""

    # API
    app_name: str = "Guarda - Controle de Acesso"
    debug: bool = False

    # Banco (PostgreSQL)
    database_url: str = "postgresql+asyncpg://guarda:guarda@localhost:5432/guarda"

    # Endpoint externo para envio da placa reconhecida
    plate_forward_url: str = ""
    plate_forward_enabled: bool = False

    # Câmera
    camera_index: int = 0

    # Reconhecimento facial
    face_tolerance: float = 0.6  # menor = mais rigoroso
    face_embedding_dim: int = 128

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
