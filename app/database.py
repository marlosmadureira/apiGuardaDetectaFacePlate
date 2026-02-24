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
        # Migração: autorização pode ser só do veículo (sem pessoa)
        try:
            await conn.execute(text(
                "ALTER TABLE authorizations ALTER COLUMN person_id DROP NOT NULL"
            ))
        except Exception as e:
            if "already nullable" not in str(e).lower():
                raise
        # Migração: ON DELETE CASCADE nas FKs de authorizations (excluir pessoa/veículo exclui autorizações)
        for col, ref in [("person_id", "persons(id)"), ("vehicle_id", "vehicles(id)")]:
            cname = f"authorizations_{col}_fkey"
            try:
                await conn.execute(text(
                    f"ALTER TABLE authorizations DROP CONSTRAINT IF EXISTS {cname}"
                ))
                await conn.execute(text(
                    f"ALTER TABLE authorizations ADD CONSTRAINT {cname} "
                    f"FOREIGN KEY ({col}) REFERENCES {ref} ON DELETE CASCADE"
                ))
            except Exception:
                pass  # ignora se já existir com outro nome; o cascade em código já garante a exclusão
