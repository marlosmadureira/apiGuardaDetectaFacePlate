"""
Verificação de acesso: combina placa + rosto para decidir se veículo e pessoa
estão autorizados a entrar no espaço privado.
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
from app.face_service import embedding_from_image, compare_face_to_embeddings
from app.schemas import AccessCheckResponse

router = APIRouter(prefix="/access", tags=["Controle de acesso"])


def _decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@router.post("/check", response_model=AccessCheckResponse)
async def check_access(
    plate_image: UploadFile = File(None),
    face_image: UploadFile = File(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Verifica acesso com base em duas imagens (upload):
    - plate_image: foto da placa do veículo
    - face_image: foto do rosto da pessoa

    Retorna se a pessoa está autorizada e se o veículo está autorizado (e se estão
    vinculados na mesma autorização).
    """
    plate_ok = False
    vehicle_plate = None
    vehicle_authorized = None
    person_id = None
    person_name = None
    allowed = False
    message_parts = []

    # 1) Placa
    if plate_image and plate_image.filename:
        content = await plate_image.read()
        if content:
            img = _decode_image(content)
            if img is not None:
                result = recognize_plate_from_image(img)
                if result and result.normalized:
                    vehicle_plate = result.normalized
                    plate_ok = True
                    q = select(Vehicle).where(
                        Vehicle.plate == vehicle_plate, Vehicle.is_active == True
                    )
                    v = (await db.execute(q)).scalar_one_or_none()
                    vehicle_authorized = v is not None
                    if not vehicle_authorized:
                        message_parts.append("Veículo não autorizado.")

    # 2) Rosto
    if face_image and face_image.filename:
        content = await face_image.read()
        if content:
            img = _decode_image(content)
            if img is not None:
                embedding = embedding_from_image(img)
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
                    message_parts.append("Nenhum rosto detectado.")
        else:
            message_parts.append("Imagem do rosto vazia.")
    else:
        message_parts.append("Envie face_image para verificar pessoa.")

    # 3) Autorização: pessoa + veículo (se ambos fornecidos)
    if person_id is not None and vehicle_plate is not None:
        q2 = select(Authorization).where(
            Authorization.person_id == person_id, Authorization.is_active == True
        )
        auths = (await db.execute(q2)).scalars().all()
        vehicle_ids = [a.vehicle_id for a in auths if a.vehicle_id is not None]
        person_vehicle_ok = False
        if vehicle_ids:
            vq = select(Vehicle).where(
                Vehicle.plate == vehicle_plate, Vehicle.id.in_(vehicle_ids)
            )
            v_match = (await db.execute(vq)).scalar_one_or_none()
            person_vehicle_ok = v_match is not None
        has_person_only = any(a.vehicle_id is None for a in auths)
        if person_vehicle_ok or (has_person_only and vehicle_authorized):
            allowed = True
        elif person_id and vehicle_authorized and not person_vehicle_ok:
            message_parts.append("Pessoa autorizada, mas não vinculada a este veículo.")
        elif person_id:
            allowed = has_person_only
    elif person_id is not None:
        # Só pessoa: verificar se tem autorização (com ou sem veículo)
        q = select(Authorization).where(
            Authorization.person_id == person_id, Authorization.is_active == True
        )
        auths = (await db.execute(q)).scalars().all()
        allowed = len(auths) > 0
        if not allowed:
            message_parts.append("Pessoa reconhecida mas sem autorização de entrada.")
    elif vehicle_plate and vehicle_authorized:
        message_parts.append("Veículo autorizado, mas pessoa não identificada.")
    else:
        if not message_parts:
            message_parts.append("Envie plate_image e/ou face_image para verificação.")

    message = " ".join(message_parts) if message_parts else (
        "Acesso autorizado." if allowed else "Acesso negado."
    )

    return AccessCheckResponse(
        allowed=allowed,
        person_id=person_id,
        person_name=person_name,
        vehicle_plate=vehicle_plate,
        vehicle_authorized=vehicle_authorized,
        message=message,
    )


@router.post("/check/camera", response_model=AccessCheckResponse)
async def check_access_from_camera(db: AsyncSession = Depends(get_db)):
    """
    Captura dois frames da câmera (primeiro para placa, segundo para rosto)
    e verifica acesso. Em produção pode ser substituído por fluxo com duas câmeras
    ou um único frame com placa e rosto em ângulos diferentes.
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
    if person_id and vehicle_plate:
        q = select(Authorization).where(
            Authorization.person_id == person_id, Authorization.is_active == True
        )
        auths = (await db.execute(q)).scalars().all()
        vehicle_ids = [a.vehicle_id for a in auths]
        if vehicle_ids:
            vq = select(Vehicle).where(
                Vehicle.plate == vehicle_plate, Vehicle.id.in_(vehicle_ids)
            )
            person_vehicle_ok = (await db.execute(vq)).scalar_one_or_none() is not None
        else:
            person_vehicle_ok = False
        has_person_only = any(a.vehicle_id is None for a in auths)
        allowed = person_vehicle_ok or (has_person_only and vehicle_authorized)
        if not allowed and person_id and vehicle_authorized:
            message_parts.append("Pessoa não vinculada a este veículo.")
    elif person_id:
        q = select(Authorization).where(
            Authorization.person_id == person_id, Authorization.is_active == True
        )
        auths = (await db.execute(q)).scalars().all()
        allowed = len(auths) > 0
        if not allowed:
            message_parts.append("Pessoa sem autorização.")
    else:
        if not vehicle_plate:
            message_parts.append("Placa não identificada.")
        if not person_name:
            message_parts.append("Rosto não reconhecido.")

    message = " ".join(message_parts) if message_parts else (
        "Acesso autorizado." if allowed else "Acesso negado."
    )
    return AccessCheckResponse(
        allowed=allowed,
        person_id=person_id,
        person_name=person_name,
        vehicle_plate=vehicle_plate,
        vehicle_authorized=vehicle_authorized,
        message=message,
    )
