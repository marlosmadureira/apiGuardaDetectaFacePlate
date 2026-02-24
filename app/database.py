"""Sessão e inicialização do banco de dados."""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from .config import get_settings

settings = get_settings()
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    from . import models  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Migração: garantir que face_embedding aceita NULL (evita 500 ao criar person sem rosto)
        try:
            await conn.execute(text(
                "ALTER TABLE persons ALTER COLUMN face_embedding DROP NOT NULL"
            ))
        except Exception as e:
            if "already nullable" not in str(e).lower():
                raise
