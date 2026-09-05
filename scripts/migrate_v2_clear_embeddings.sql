-- Migração Fase 2: limpar embeddings no formato antigo (dlib 128-d CSV).
-- O novo formato é InsightFace ArcFace 512-d base64.
-- ATENÇÃO: após executar, todos os usuários precisam re-cadastrar o rosto.
--
-- Executar ANTES de fazer deploy da Fase 2:
--   docker exec guarda-postgres psql -U guarda -d guarda -f /path/to/migrate_v2_clear_embeddings.sql
-- Ou via psql direto:
--   psql postgresql://guarda:guarda@localhost:5432/guarda -f migrate_v2_clear_embeddings.sql

UPDATE persons
SET face_embedding = NULL,
    face_photo_path = NULL
WHERE face_embedding IS NOT NULL;

-- Confirmar resultado:
SELECT COUNT(*) AS persons_sem_embedding FROM persons WHERE face_embedding IS NULL;
