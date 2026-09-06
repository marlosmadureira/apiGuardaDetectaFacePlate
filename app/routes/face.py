"""
Rotas de reconhecimento facial: cadastro (crop + embedding) e verificação.
Fase 3: usa pgvector (find_best_match_pgvector) e câmera singleton.
"""
import asyncio
import os
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse
import cv2

from app.database import get_db
from app.models import Person
from app.config import get_settings
from app.face_service import (
    get_face_crop_and_embedding,
    embedding_from_image,
    find_best_match_pgvector,
    save_crop,
)
from app.auth import require_api_key
from app.schemas import FaceRegisterResponse, FaceVerifyResponse
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/face", tags=["Reconhecimento facial"])


def _sync_capture_frame(camera_index: int):
    """Captura frame usando câmera singleton (sem overhead de open/close)."""
    from app.camera import get_camera
    frame = get_camera(camera_index).read_frame()
    return frame is not None, frame


def _decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@router.post(
    "/register/{person_id}",
    response_model=FaceRegisterResponse,
    dependencies=[Depends(require_api_key)],
)
async def register_face(
    person_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Cadastra o rosto de uma pessoa já criada em /persons.
    Envie uma foto com o rosto visível; a API faz o crop, gera o embedding ArcFace
    e armazena no banco (coluna vector(512)) para comparações futuras.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    image = _decode_image(content)
    if image is None:
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    person = await db.get(Person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada.")

    loop = asyncio.get_running_loop()
    crop, embedding = await loop.run_in_executor(None, get_face_crop_and_embedding, image)
    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="Nenhum rosto detectado. Envie uma foto com o rosto visível.",
        )

    settings = get_settings()
    # Verifica duplicata: rosto já cadastrado em outra pessoa?
    existing = await find_best_match_pgvector(
        embedding, db, tolerance=settings.face_tolerance, exclude_person_id=person_id
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"⚠️ ROSTO JÁ CADASTRADO: Este rosto pertence a '{existing.name}' "
                f"(ID: {existing.person_id}). Não é possível cadastrar o mesmo rosto para mais de uma pessoa."
            ),
        )

    # Armazena embedding diretamente como vetor (pgvector cuida da serialização)
    person.face_embedding = embedding
    photo_path = save_crop(crop, settings.face_photos_dir, prefix=str(person_id))
    if photo_path:
        person.face_photo_path = photo_path
    await db.commit()
    await db.refresh(person)

    return FaceRegisterResponse(
        person_id=person.id,
        name=person.name,
        message="Rosto cadastrado com sucesso. Embedding ArcFace armazenado para comparação.",
    )


@router.post("/verify", response_model=FaceVerifyResponse)
async def verify_face(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Verifica se o rosto na imagem corresponde a alguma pessoa cadastrada.
    Usa busca vetorial indexada (ivfflat) — O(log n).
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    image = _decode_image(content)
    if image is None:
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    loop = asyncio.get_running_loop()
    embedding = await loop.run_in_executor(None, embedding_from_image, image)
    if embedding is None:
        return FaceVerifyResponse(matched=False, message="Nenhum rosto detectado na imagem.")

    settings = get_settings()
    match = await find_best_match_pgvector(embedding, db, tolerance=settings.face_tolerance)
    if match:
        return FaceVerifyResponse(
            matched=True,
            person_id=match.person_id,
            name=match.name,
            distance=match.distance,
            message=f"Rosto reconhecido: {match.name}.",
        )
    return FaceVerifyResponse(matched=False, message="Rosto não reconhecido.")


@router.post(
    "/capture/register/{person_id}",
    response_model=FaceRegisterResponse,
    dependencies=[Depends(require_api_key)],
)
async def register_face_from_camera(
    person_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Captura um frame da câmera (singleton persistente) e cadastra o rosto.
    """
    settings = get_settings()
    loop = asyncio.get_running_loop()

    ret, frame = await loop.run_in_executor(None, _sync_capture_frame, settings.camera_index)
    if not ret or frame is None:
        raise HTTPException(
            status_code=503,
            detail="Câmera não disponível. Use /face/register com upload de imagem.",
        )

    person = await db.get(Person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada.")

    crop, embedding = await loop.run_in_executor(None, get_face_crop_and_embedding, frame)
    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="Nenhum rosto detectado. Posicione o rosto na câmera e tente novamente.",
        )

    existing = await find_best_match_pgvector(
        embedding, db, tolerance=settings.face_tolerance, exclude_person_id=person_id
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"⚠️ ROSTO JÁ CADASTRADO: Este rosto pertence a '{existing.name}' "
                f"(ID: {existing.person_id})."
            ),
        )

    person.face_embedding = embedding
    photo_path = save_crop(crop, settings.face_photos_dir, prefix=str(person_id))
    if photo_path:
        person.face_photo_path = photo_path
    await db.commit()
    await db.refresh(person)

    return FaceRegisterResponse(
        person_id=person.id,
        name=person.name,
        message="Rosto cadastrado a partir da câmera.",
    )


@router.post("/capture/verify", response_model=FaceVerifyResponse)
async def verify_face_from_camera(db: AsyncSession = Depends(get_db)):
    """
    Captura da câmera e verifica se o rosto corresponde a alguma pessoa cadastrada.
    """
    settings = get_settings()
    loop = asyncio.get_running_loop()

    ret, frame = await loop.run_in_executor(None, _sync_capture_frame, settings.camera_index)
    if not ret or frame is None:
        raise HTTPException(status_code=503, detail="Câmera não disponível.")

    embedding = await loop.run_in_executor(None, embedding_from_image, frame)
    if embedding is None:
        return FaceVerifyResponse(matched=False, message="Nenhum rosto detectado.")

    match = await find_best_match_pgvector(embedding, db, tolerance=settings.face_tolerance)
    if match:
        return FaceVerifyResponse(
            matched=True,
            person_id=match.person_id,
            name=match.name,
            distance=match.distance,
            message=f"Rosto reconhecido: {match.name}.",
        )
    return FaceVerifyResponse(matched=False, message="Rosto não reconhecido.")


@router.get("/photo/{person_id}", response_class=FileResponse)
async def get_face_photo(person_id: int, db: AsyncSession = Depends(get_db)):
    """Retorna a foto do rosto cadastrada para a pessoa."""
    person = await db.get(Person, person_id)
    if person is None or not person.face_photo_path:
        raise HTTPException(
            status_code=404,
            detail="Pessoa não encontrada ou rosto ainda não cadastrado.",
        )
    path = person.face_photo_path
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Arquivo da foto não encontrado.")
    return FileResponse(path, media_type="image/jpeg")
