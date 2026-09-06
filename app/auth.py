"""Autenticação por API Key via header X-API-Key."""
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

from app.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(_api_key_header)) -> None:
    """
    Dependência FastAPI que exige X-API-Key válida quando api_keys está configurado.
    Se api_keys estiver vazio (padrão), a rota fica aberta (modo dev).
    """
    settings = get_settings()
    if not settings.api_keys:
        return  # sem chaves configuradas = aberto (modo dev/piloto)
    if not api_key or api_key not in settings.api_keys:
        raise HTTPException(status_code=403, detail="Chave de API inválida ou ausente.")
