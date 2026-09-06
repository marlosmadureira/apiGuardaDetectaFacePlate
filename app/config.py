"""Configurações da aplicação."""
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Configurações carregadas de variáveis de ambiente."""

    # API
    app_name: str = "Guarda - Controle de Acesso"
    debug: bool = False
    log_level: str = "INFO"

    # Segurança
    api_keys: List[str] = []          # vazio = aberto (dev); ex: API_KEYS=key1,key2
    cors_origins: List[str] = ["*"]   # restringir em produção

    # Banco (PostgreSQL)
    database_url: str = "postgresql+asyncpg://guarda:guarda@localhost:5432/guarda"

    # Endpoint externo para envio da placa reconhecida
    plate_forward_url: str = ""
    plate_forward_enabled: bool = False

    # Câmeras — configuração central via JSON (editável por cliente sem rebuild)
    cameras_config_path: str = "data/cameras.json"
    camera_index: int = 0  # câmera padrão (fallback legado)

    # Reconhecimento facial (InsightFace ArcFace 512-d, similaridade coseno)
    face_tolerance: float = 0.5  # maior = mais rigoroso (similaridade coseno mínima exigida)
    face_embedding_dim: int = 512
    face_photos_dir: str = "data/faces"  # pasta para salvar crops (rosto) para consultas futuras
    embedding_cache_ttl: int = 60  # segundos — TTL do cache em memória de embeddings

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
