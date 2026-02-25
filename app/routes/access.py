"""
Verificação de acesso. A autorização é sempre de um tipo:
- Entrada a pé: só verificação facial (autorização com vehicle_id = null).
- Entrada com veículo: verificação facial + placa (autorização com vehicle_id preenchido).
Nunca exige os dois ao mesmo tempo; ou a pessoa entra a pé ou com aquele veículo.
"""
import cv2
import numpy as np
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import Person, Vehicle, Authorization
from app.config import get_settings
from app.plate_recognizer import recognize_plate_from_image, capture_frame
from app.face_service import get_face_bbox_embedding_landmarks, compare_face_to_embeddings
from app.schemas import AccessCheckResponse

router = APIRouter(prefix="/access", tags=["Controle de acesso"])


def _decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@router.post("/check", response_model=AccessCheckResponse)
async def check_access(
    face_image: UploadFile = File(..., description="Imagem do frame (câmera). Usada para reconhecimento facial e leitura de placa."),
    db: AsyncSession = Depends(get_db),
):
    """
    Verifica acesso a partir de uma única imagem (frame da câmera).
    A mesma imagem é usada para: (1) reconhecimento facial e (2) leitura de placa (OCR).
    Libera por pedestre (rosto) ou por veículo (placa) ou por pessoa+veículo.
    """
    content = await face_image.read()
    if not content:
        return AccessCheckResponse(
            allowed=False,
            message="Imagem vazia. Envie o frame da câmera.",
        )
    img = _decode_image(content)
    if img is None:
        return AccessCheckResponse(allowed=False, message="Imagem inválida.")

    vehicle_plate = None
    vehicle_authorized = None
    plate_bbox = None
    person_id = None
    person_name = None
    face_bbox = None
    face_landmarks = None
    allowed = False
    message_parts = []

    # 1) Placa: extrai da própria imagem (câmera ao vivo pode mostrar a placa)
    result_plate = recognize_plate_from_image(img)
    if result_plate and result_plate.normalized:
        vehicle_plate = result_plate.normalized
        plate_bbox = list(result_plate.bbox) if result_plate.bbox else None
        q = select(Vehicle).where(
            Vehicle.plate == vehicle_plate, Vehicle.is_active == True
        )
        v = (await db.execute(q)).scalar_one_or_none()
        vehicle_authorized = v is not None
        if not vehicle_authorized and vehicle_plate:
            message_parts.append("Placa não cadastrada.")

    # 2) Rosto: extrai da mesma imagem (bbox, embedding e landmarks)
    face_bbox_tuple, embedding, face_landmarks = get_face_bbox_embedding_landmarks(img)
    if face_bbox_tuple:
        face_bbox = list(face_bbox_tuple)
    if embedding is not None:
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
            person_id = match.person_id
            person_name = match.name
        else:
            message_parts.append("Pessoa não reconhecida.")
    else:
        if not vehicle_plate:
            message_parts.append("Nenhum rosto nem placa detectados.")

    # 3) Autorização: a primeira que bater permite (nunca exige rosto E placa juntos)
    # 3a) Pedestre: só rosto
    if person_id is not None:
        q_ped = select(Authorization).where(
            Authorization.person_id == person_id,
            Authorization.is_active == True,
            Authorization.vehicle_id.is_(None),
        )
        if (await db.execute(q_ped)).first() is not None:
            allowed = True
    # 3b) Pessoa + veículo: rosto e placa (só se ainda não liberou por pedestre)
    if not allowed and person_id is not None and vehicle_plate is not None:
        q2 = select(Authorization).where(
            Authorization.person_id == person_id,
            Authorization.is_active == True,
            Authorization.vehicle_id.isnot(None),
        )
        auths = (await db.execute(q2)).scalars().all()
        vehicle_ids = [a.vehicle_id for a in auths]
        if vehicle_ids:
            vq = select(Vehicle).where(
                Vehicle.plate == vehicle_plate, Vehicle.id.in_(vehicle_ids)
            )
            if (await db.execute(vq)).scalar_one_or_none() is not None:
                allowed = True
    # 3c) Só veículo: só placa (não exige rosto)
    if not allowed and vehicle_plate and vehicle_authorized:
        vq = select(Vehicle.id).where(
            Vehicle.plate == vehicle_plate, Vehicle.is_active == True
        )
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
            message_parts.append("Nenhum rosto nem placa reconhecidos na imagem.")
        elif person_id is not None:
            message_parts.append("Pessoa sem autorização de entrada a pé.")
        elif vehicle_plate and vehicle_authorized:
            message_parts.append("Placa sem autorização (só veículo).")
        else:
            message_parts.append("Rosto ou placa não reconhecidos.")

    if allowed:
        message = "Acesso autorizado."
    elif message_parts:
        message = " ".join(message_parts)
    else:
        message = "Acesso negado."

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
    Captura da câmera: lê placa (se houver) e rosto.
    - Só rosto reconhecido: permite se a pessoa tiver autorização de entrada a pé (vehicle_id null).
    - Rosto + placa: permite se a pessoa tiver autorização para aquele veículo.
    """
    settings = get_settings()
    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        raise HTTPException(
            status_code=503,
            detail="Câmera não disponível. Use /access/check com upload de imagens.",
        )
    # Frame 1 - placa (ex.: veículo na frente)
    ret1, frame1 = cap.read()
    # Frame 2 - rosto (ex.: motorista)
    ret2, frame2 = cap.read()
    cap.release()
    if not ret1 or frame1 is None:
        raise HTTPException(status_code=503, detail="Falha ao capturar frame da câmera.")

    # Usar mesmo frame para ambos se só tiver um (piloto)
    plate_frame = frame1
    face_frame = frame2 if ret2 and frame2 is not None else frame1

    vehicle_plate = None
    result_plate = recognize_plate_from_image(plate_frame)
    if result_plate and result_plate.normalized:
        vehicle_plate = result_plate.normalized

    vehicle_authorized = None
    if vehicle_plate:
        q = select(Vehicle).where(
            Vehicle.plate == vehicle_plate, Vehicle.is_active == True
        )
        v = (await db.execute(q)).scalar_one_or_none()
        vehicle_authorized = v is not None

    person_id = None
    person_name = None
    embedding = embedding_from_image(face_frame)
    if embedding is not None:
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
            person_id = match.person_id
            person_name = match.name

    allowed = False
    message_parts = []
    # A primeira que bater permite: pedestre OU pessoa+veículo OU só veículo (nunca os 2 juntos)
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
            vq = select(Vehicle).where(
                Vehicle.plate == vehicle_plate, Vehicle.id.in_(vehicle_ids)
            )
            if (await db.execute(vq)).scalar_one_or_none() is not None:
                allowed = True
    if not allowed and vehicle_plate and vehicle_authorized:
        vq = select(Vehicle.id).where(
            Vehicle.plate == vehicle_plate, Vehicle.is_active == True
        )
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
            message_parts.append("Pessoa sem autorização de entrada a pé.")
        elif vehicle_plate and vehicle_authorized:
            message_parts.append("Veículo sem autorização (só placa).")
        else:
            message_parts.append("Rosto ou placa não reconhecidos.")

    if allowed:
        message = "Acesso autorizado."
    elif message_parts:
        message = " ".join(message_parts)
    else:
        message = "Acesso negado."
    return AccessCheckResponse(
        allowed=allowed,
        person_id=person_id,
        person_name=person_name,
        vehicle_plate=vehicle_plate,
        vehicle_authorized=vehicle_authorized,
        message=message,
    )
