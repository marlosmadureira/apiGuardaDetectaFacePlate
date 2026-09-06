"""
Verificação de acesso. A autorização é sempre de um tipo:
- Entrada a pé: só verificação facial (autorização com vehicle_id = null).
- Entrada com veículo: verificação facial + placa (autorização com vehicle_id preenchido).
Nunca exige os dois ao mesmo tempo; ou a pessoa entra a pé ou com aquele veículo.
"""
import asyncio
import time
import cv2
import numpy as np
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db, AsyncSessionLocal
from app.models import Person, Vehicle, Authorization, AccessLog
from app.config import get_settings
from app.plate_recognizer import recognize_plate_from_image
from app.face_service import (
    get_face_bbox_embedding_landmarks,
    embedding_from_image,
    find_best_match_pgvector,
)
from app.schemas import AccessCheckResponse

router = APIRouter(prefix="/access", tags=["Controle de acesso"])


def _decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _sync_capture_two_frames(camera_index: int):
    """Captura dois frames usando câmera singleton. Roda em thread pool."""
    from app.camera import get_camera
    cam = get_camera(camera_index)
    frame1 = cam.read_frame()
    if frame1 is None:
        return None, None
    frame2 = cam.read_frame()
    face_frame = frame2 if frame2 is not None else frame1
    return frame1, face_frame


async def _log_access(
    db: AsyncSession,
    allowed: bool,
    message: str,
    latency_ms: int,
    person_id: int = None,
    vehicle_plate: str = None,
    face_similarity: float = None,
) -> None:
    """Registra tentativa de acesso na tabela access_logs."""
    try:
        log = AccessLog(
            person_id=person_id,
            vehicle_plate=vehicle_plate,
            allowed=allowed,
            face_similarity=face_similarity,
            latency_ms=latency_ms,
            message=message,
        )
        db.add(log)
        await db.commit()
    except Exception:
        pass  # log não deve derrubar o endpoint


@router.post("/check", response_model=AccessCheckResponse)
async def check_access(
    face_image: UploadFile = File(..., description="Frame da câmera. Usado para rosto e placa."),
    db: AsyncSession = Depends(get_db),
):
    """
    Verifica acesso a partir de uma única imagem.
    Rosto e placa detectados em paralelo (asyncio.gather + thread pool).
    Registra resultado em access_logs para auditoria.
    """
    t0 = time.monotonic()
    content = await face_image.read()
    if not content:
        return AccessCheckResponse(allowed=False, message="Imagem vazia.")
    img = _decode_image(content)
    if img is None:
        return AccessCheckResponse(allowed=False, message="Imagem inválida.")

    loop = asyncio.get_running_loop()
    settings = get_settings()

    vehicle_plate = None
    vehicle_authorized = None
    plate_bbox = None
    person_id = None
    person_name = None
    face_bbox = None
    face_landmarks = None
    face_similarity = None
    allowed = False
    message_parts = []

    # Detecção paralela de placa + rosto
    result_plate, face_data = await asyncio.gather(
        loop.run_in_executor(None, recognize_plate_from_image, img),
        loop.run_in_executor(None, get_face_bbox_embedding_landmarks, img),
    )

    # 1) Placa
    if result_plate and result_plate.normalized:
        vehicle_plate = result_plate.normalized
        plate_bbox = list(result_plate.bbox) if result_plate.bbox else None
        q = select(Vehicle).where(Vehicle.plate == vehicle_plate, Vehicle.is_active == True)
        v = (await db.execute(q)).scalar_one_or_none()
        vehicle_authorized = v is not None
        if not vehicle_authorized:
            message_parts.append("Placa não cadastrada.")

    # 2) Rosto → busca pgvector
    face_bbox_tuple, embedding, face_landmarks = face_data
    if face_bbox_tuple:
        face_bbox = list(face_bbox_tuple)
    if embedding is not None:
        match = await find_best_match_pgvector(embedding, db, tolerance=settings.face_tolerance)
        if match:
            person_id = match.person_id
            person_name = match.name
            face_similarity = match.distance
        else:
            message_parts.append("Pessoa não reconhecida.")
    else:
        if not vehicle_plate:
            message_parts.append("Nenhum rosto nem placa detectados.")

    # 3) Autorização: primeira regra que casar libera
    if person_id is not None:
        q_ped = select(Authorization).where(
            Authorization.person_id == person_id,
            Authorization.is_active == True,
            Authorization.vehicle_id.is_(None),
        )
        if (await db.execute(q_ped)).first() is not None:
            allowed = True

    if not allowed and person_id is not None and vehicle_plate is not None:
        q2 = select(Authorization).where(
            Authorization.person_id == person_id,
            Authorization.is_active == True,
            Authorization.vehicle_id.isnot(None),
        )
        auths = (await db.execute(q2)).scalars().all()
        vehicle_ids = [a.vehicle_id for a in auths]
        if vehicle_ids:
            vq = select(Vehicle).where(Vehicle.plate == vehicle_plate, Vehicle.id.in_(vehicle_ids))
            if (await db.execute(vq)).scalar_one_or_none() is not None:
                allowed = True

    if not allowed and vehicle_plate and vehicle_authorized:
        vq = select(Vehicle.id).where(Vehicle.plate == vehicle_plate, Vehicle.is_active == True)
        vid = (await db.execute(vq)).scalar_one_or_none()
        if vid is not None:
            q_veh = select(Authorization).where(
                Authorization.person_id.is_(None),
                Authorization.vehicle_id == vid,
                Authorization.is_active == True,
            )
            if (await db.execute(q_veh)).first() is not None:
                allowed = True

    if not allowed and not message_parts:
        if person_id is None and not vehicle_plate:
            message_parts.append("Nenhum rosto nem placa reconhecidos.")
        elif person_id is not None:
            message_parts.append("Pessoa sem autorização.")
        elif vehicle_plate and vehicle_authorized:
            message_parts.append("Placa sem autorização (só veículo).")
        else:
            message_parts.append("Rosto ou placa não reconhecidos.")

    message = "Acesso autorizado." if allowed else (" ".join(message_parts) or "Acesso negado.")
    latency_ms = int((time.monotonic() - t0) * 1000)

    await _log_access(
        db, allowed=allowed, message=message, latency_ms=latency_ms,
        person_id=person_id, vehicle_plate=vehicle_plate, face_similarity=face_similarity,
    )

    return AccessCheckResponse(
        allowed=allowed,
        person_id=person_id,
        person_name=person_name,
        vehicle_plate=vehicle_plate,
        vehicle_authorized=vehicle_authorized,
        face_bbox=face_bbox,
        face_landmarks=face_landmarks,
        plate_bbox=plate_bbox,
        message=message,
    )


@router.post("/check/camera", response_model=AccessCheckResponse)
async def check_access_from_camera(db: AsyncSession = Depends(get_db)):
    """
    Captura da câmera singleton: lê placa e rosto.
    """
    t0 = time.monotonic()
    settings = get_settings()
    loop = asyncio.get_running_loop()

    plate_frame, face_frame = await loop.run_in_executor(
        None, _sync_capture_two_frames, settings.camera_index
    )
    if plate_frame is None:
        raise HTTPException(
            status_code=503,
            detail="Câmera não disponível. Use /access/check com upload de imagens.",
        )

    result_plate, embedding = await asyncio.gather(
        loop.run_in_executor(None, recognize_plate_from_image, plate_frame),
        loop.run_in_executor(None, embedding_from_image, face_frame),
    )

    vehicle_plate = None
    if result_plate and result_plate.normalized:
        vehicle_plate = result_plate.normalized

    vehicle_authorized = None
    if vehicle_plate:
        q = select(Vehicle).where(Vehicle.plate == vehicle_plate, Vehicle.is_active == True)
        v = (await db.execute(q)).scalar_one_or_none()
        vehicle_authorized = v is not None

    person_id = None
    person_name = None
    face_similarity = None
    if embedding is not None:
        match = await find_best_match_pgvector(embedding, db, tolerance=settings.face_tolerance)
        if match:
            person_id = match.person_id
            person_name = match.name
            face_similarity = match.distance

    allowed = False
    message_parts = []

    if person_id:
        q_ped = select(Authorization).where(
            Authorization.person_id == person_id,
            Authorization.is_active == True,
            Authorization.vehicle_id.is_(None),
        )
        if (await db.execute(q_ped)).first() is not None:
            allowed = True

    if not allowed and person_id and vehicle_plate:
        q = select(Authorization).where(
            Authorization.person_id == person_id,
            Authorization.is_active == True,
            Authorization.vehicle_id.isnot(None),
        )
        auths = (await db.execute(q)).scalars().all()
        vehicle_ids = [a.vehicle_id for a in auths]
        if vehicle_ids:
            vq = select(Vehicle).where(Vehicle.plate == vehicle_plate, Vehicle.id.in_(vehicle_ids))
            if (await db.execute(vq)).scalar_one_or_none() is not None:
                allowed = True

    if not allowed and vehicle_plate and vehicle_authorized:
        vq = select(Vehicle.id).where(Vehicle.plate == vehicle_plate, Vehicle.is_active == True)
        vid = (await db.execute(vq)).scalar_one_or_none()
        if vid is not None:
            q_veh = select(Authorization).where(
                Authorization.person_id.is_(None),
                Authorization.vehicle_id == vid,
                Authorization.is_active == True,
            )
            if (await db.execute(q_veh)).first() is not None:
                allowed = True

    if not allowed:
        if not vehicle_plate and not person_name:
            message_parts.append("Placa e rosto não identificados.")
        elif person_id:
            message_parts.append("Pessoa sem autorização.")
        elif vehicle_plate and vehicle_authorized:
            message_parts.append("Veículo sem autorização.")
        else:
            message_parts.append("Rosto ou placa não reconhecidos.")

    message = "Acesso autorizado." if allowed else (" ".join(message_parts) or "Acesso negado.")
    latency_ms = int((time.monotonic() - t0) * 1000)

    await _log_access(
        db, allowed=allowed, message=message, latency_ms=latency_ms,
        person_id=person_id, vehicle_plate=vehicle_plate, face_similarity=face_similarity,
    )

    return AccessCheckResponse(
        allowed=allowed,
        person_id=person_id,
        person_name=person_name,
        vehicle_plate=vehicle_plate,
        vehicle_authorized=vehicle_authorized,
        message=message,
    )
