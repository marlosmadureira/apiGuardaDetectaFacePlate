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

        # 1. Extensão pgvector (obrigatória antes de create_all para usar Vector(512))
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception:
            pass  # PostgreSQL sem pgvector; Vector column usará fallback Text

        # 2. Criar tabelas novas (não altera existentes)
        await conn.run_sync(Base.metadata.create_all)

        # 3. Migração: face_embedding TEXT → VECTOR(512) (Phase 1→3 upgrade)
        try:
            await conn.execute(text("""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'persons'
                          AND column_name = 'face_embedding'
                          AND data_type = 'text'
                    ) THEN
                        ALTER TABLE persons DROP COLUMN face_embedding;
                        ALTER TABLE persons ADD COLUMN face_embedding vector(512);
                    END IF;
                END $$
            """))
        except Exception:
            pass  # pgvector indisponível ou coluna já é vector

        # 4. Migração legacy: face_embedding nullable
        try:
            await conn.execute(text(
                "ALTER TABLE persons ALTER COLUMN face_embedding DROP NOT NULL"
            ))
        except Exception:
            pass

        # 5. Migração legacy: person_id nullable em authorizations
        try:
            await conn.execute(text(
                "ALTER TABLE authorizations ALTER COLUMN person_id DROP NOT NULL"
            ))
        except Exception:
            pass

        # 6. FK ON DELETE CASCADE em authorizations
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
                pass

        # 7. Índice IVFFlat para busca coseno O(log n) em embeddings faciais
        try:
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS persons_face_embedding_cosine_idx "
                "ON persons USING ivfflat (face_embedding vector_cosine_ops) "
                "WITH (lists = 10)"
            ))
        except Exception:
            pass  # Tabela vazia ou pgvector indisponível; índice pode ser criado depois
