"""
Rotas de reconhecimento facial: cadastro (crop + embedding) e verificação (comparação).
"""
import os
import time
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import Person
from app.config import get_settings
from app.face_service import (
    get_face_crop_and_embedding,
    embedding_from_image,
    compare_face_to_embeddings,
    _embedding_to_str,
    save_crop,
)
from app.schemas import FaceRegisterResponse, FaceVerifyResponse

router = APIRouter(prefix="/face", tags=["Reconhecimento facial"])


def _capture_frame_from_camera(camera_index: int, warmup_seconds: float = 1.0):
    """Abre a câmera, espera um pouco para ajuste de luz e foco, e retorna um frame."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None, None
    try:
        # Descarta alguns frames e dá tempo da câmera ajustar
        for _ in range(5):
            cap.read()
        time.sleep(warmup_seconds)
        ret, frame = cap.read()
        return ret, frame
    finally:
        cap.release()


def _decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@router.post("/register/{person_id}", response_model=FaceRegisterResponse)
async def register_face(
    person_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Cadastra o rosto de uma pessoa já criada em /persons.
    Envie uma foto com o rosto visível; a API faz o crop, gera o embedding
    e armazena no banco para comparações futuras.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    image = _decode_image(content)
    if image is None:
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    result = await db.get(Person, person_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada.")

    crop, embedding = get_face_crop_and_embedding(image)
    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="Nenhum rosto detectado na imagem. Envie uma foto com o rosto visível.",
        )

    result.face_embedding = _embedding_to_str(embedding)
    settings = get_settings()
    photo_path = save_crop(crop, settings.face_photos_dir, prefix=str(person_id))
    if photo_path:
        result.face_photo_path = photo_path
    await db.commit()
    await db.refresh(result)

    return FaceRegisterResponse(
        person_id=result.id,
        name=result.name,
        message="Rosto cadastrado com sucesso. Embedding e foto armazenados para comparação.",
    )


@router.post("/verify", response_model=FaceVerifyResponse)
async def verify_face(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Verifica se o rosto na imagem corresponde a alguma pessoa cadastrada.
    Retorna matched=True e dados da pessoa se houver correspondência.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    image = _decode_image(content)
    if image is None:
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    embedding = embedding_from_image(image)
    if embedding is None:
        return FaceVerifyResponse(
            matched=False,
            message="Nenhum rosto detectado na imagem.",
        )

    settings = get_settings()
    q = select(Person.id, Person.name, Person.face_embedding).where(
        Person.is_active == True,
        Person.face_embedding.isnot(None),
        Person.face_embedding != "",
    )
    rows = (await db.execute(q)).all()
    stored = [(r[0], r[1], r[2]) for r in rows if r[2]]

    match = compare_face_to_embeddings(
        embedding, stored, tolerance=settings.face_tolerance
    )
    if match:
        return FaceVerifyResponse(
            matched=True,
            person_id=match.person_id,
            name=match.name,
            distance=match.distance,
            message=f"Rosto reconhecido: {match.name}.",
        )
    return FaceVerifyResponse(
        matched=False,
        message="Rosto não reconhecido. Nenhuma correspondência no banco.",
    )


@router.post("/capture/register/{person_id}", response_model=FaceRegisterResponse)
async def register_face_from_camera(
    person_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Captura um frame da câmera e cadastra o rosto para a pessoa informada.
    Salva a foto do rosto para consultas futuras (GET /face/photo/{person_id}).
    Requer câmera disponível (--device /dev/video0 no Docker).
    """
    settings = get_settings()
    ret, frame = _capture_frame_from_camera(settings.camera_index)
    if not ret or frame is None:
        raise HTTPException(
            status_code=503,
            detail="Câmera não disponível ou falha ao capturar. Use /face/register com upload de imagem ou verifique o dispositivo.",
        )

    result = await db.get(Person, person_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada.")

    crop, embedding = get_face_crop_and_embedding(frame)
    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="Nenhum rosto detectado no frame. Posicione o rosto na câmera e tente novamente.",
        )

    result.face_embedding = _embedding_to_str(embedding)
    photo_path = save_crop(crop, settings.face_photos_dir, prefix=str(person_id))
    if photo_path:
        result.face_photo_path = photo_path
    await db.commit()
    await db.refresh(result)

    return FaceRegisterResponse(
        person_id=result.id,
        name=result.name,
        message="Rosto cadastrado a partir da câmera. Foto salva para consultas futuras.",
    )


@router.post("/capture/verify", response_model=FaceVerifyResponse)
async def verify_face_from_camera(db: AsyncSession = Depends(get_db)):
    """
    Captura um frame da câmera e verifica se o rosto corresponde a alguma pessoa cadastrada.
    """
    settings = get_settings()
    ret, frame = _capture_frame_from_camera(settings.camera_index)
    if not ret or frame is None:
        raise HTTPException(status_code=503, detail="Câmera não disponível ou falha ao capturar.")

    embedding = embedding_from_image(frame)
    if embedding is None:
        return FaceVerifyResponse(matched=False, message="Nenhum rosto detectado.")

    q = select(Person.id, Person.name, Person.face_embedding).where(
        Person.is_active == True,
        Person.face_embedding.isnot(None),
        Person.face_embedding != "",
    )
    rows = (await db.execute(q)).all()
    stored = [(r[0], r[1], r[2]) for r in rows if r[2]]
    match = compare_face_to_embeddings(
        embedding, stored, tolerance=settings.face_tolerance
    )
    if match:
        return FaceVerifyResponse(
            matched=True,
            person_id=match.person_id,
            name=match.name,
            distance=match.distance,
            message=f"Rosto reconhecido: {match.name}.",
        )
    return FaceVerifyResponse(
        matched=False,
        message="Rosto não reconhecido.",
    )


@router.get("/photo/{person_id}", response_class=FileResponse)
async def get_face_photo(person_id: int, db: AsyncSession = Depends(get_db)):
    """
    Retorna a foto do rosto cadastrada para a pessoa (captura salva no cadastro).
    Útil para consultas futuras e conferência.
    """
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
