"""
Rotas de reconhecimento facial: cadastro (crop + embedding) e verificação (comparação).
"""
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
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
    # Opcional: salvar crop em disco (ex.: uploads/faces/)
    # result.face_photo_path = save_crop(crop, "uploads/faces", prefix=str(person_id))
    await db.commit()
    await db.refresh(result)

    return FaceRegisterResponse(
        person_id=result.id,
        name=result.name,
        message="Rosto cadastrado com sucesso. Embedding armazenado para comparação.",
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
    Requer câmera disponível (--device /dev/video0 no Docker).
    """
    settings = get_settings()
    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        raise HTTPException(
            status_code=503,
            detail="Câmera não disponível. Use /face/register com upload de imagem ou verifique o dispositivo.",
        )
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise HTTPException(status_code=503, detail="Falha ao capturar frame.")

    result = await db.get(Person, person_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada.")

    crop, embedding = get_face_crop_and_embedding(frame)
    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="Nenhum rosto detectado no frame. Tente novamente com o rosto visível.",
        )

    result.face_embedding = _embedding_to_str(embedding)
    await db.commit()
    await db.refresh(result)

    return FaceRegisterResponse(
        person_id=result.id,
        name=result.name,
        message="Rosto cadastrado a partir da câmera com sucesso.",
    )


@router.post("/capture/verify", response_model=FaceVerifyResponse)
async def verify_face_from_camera(db: AsyncSession = Depends(get_db)):
    """
    Captura um frame da câmera e verifica se o rosto está cadastrado.
    """
    settings = get_settings()
    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        raise HTTPException(status_code=503, detail="Câmera não disponível.")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise HTTPException(status_code=503, detail="Falha ao capturar frame.")

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
