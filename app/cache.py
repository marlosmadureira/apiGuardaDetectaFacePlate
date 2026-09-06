"""Cache em memória de embeddings faciais com TTL configurável."""
import asyncio
import time
from typing import List, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Person


class EmbeddingCache:
    def __init__(self, ttl_seconds: int = 60):
        self._data: List[Tuple[int, str, str]] = []
        self._loaded_at: float = 0.0
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    async def get_embeddings(self, db: AsyncSession) -> List[Tuple[int, str, str]]:
        """Retorna embeddings do cache; recarrega do banco se o TTL expirou."""
        async with self._lock:
            if time.monotonic() - self._loaded_at > self._ttl:
                q = select(Person.id, Person.name, Person.face_embedding).where(
                    Person.is_active == True,
                    Person.face_embedding.isnot(None),
                )
                rows = (await db.execute(q)).all()
                self._data = [(r[0], r[1], r[2]) for r in rows if r[2]]
                self._loaded_at = time.monotonic()
        return self._data

    def invalidate(self) -> None:
        """Força recarga no próximo acesso (chamar após cadastro/remoção de rosto)."""
        self._loaded_at = 0.0


embedding_cache = EmbeddingCache()
