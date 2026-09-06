"""WebSocket para verificação de acesso em tempo real com backpressure natural."""
import asyncio
import json
import time
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models import Vehicle, Authorization, AccessLog
from app.config import get_settings
from app.plate_recognizer import recognize_plate_from_image
from app.face_service import get_face_bbox_embedding_landmarks, find_best_match_pgvector

router = APIRouter(prefix="/ws", tags=["Stream WebSocket"])


@router.websocket("/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket leve para detecção de rosto em tempo real (cadastro).
    Sem acesso ao banco — apenas informa se há rosto no frame e seu bbox.
    """
    await websocket.accept()
    loop = asyncio.get_running_loop()
    try:
        while True:
            data = await websocket.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                await websocket.send_text(json.dumps({"face_detected": False, "face_bbox": None}))
                continue
            face_bbox_tuple, _, _ = await loop.run_in_executor(
                None, get_face_bbox_embedding_landmarks, img
            )
            face_bbox = list(face_bbox_tuple) if face_bbox_tuple else None
            await websocket.send_text(json.dumps({
                "face_detected": face_bbox is not None,
                "face_bbox": face_bbox,
            }))
    except WebSocketDisconnect:
        pass


def _to_serializable(obj):
    """Converte numpy arrays e estruturas aninhadas para tipos JSON-serializáveis."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(i) for i in obj]
    return obj


@router.websocket("/verify")
async def websocket_verify(websocket: WebSocket):
    """
    WebSocket de verificação ao vivo.

    Protocolo: cliente envia frame JPEG (bytes) → servidor responde com JSON.
    Backpressure natural: cliente só envia o próximo frame após receber a resposta,
    evitando acúmulo de frames e garantindo processamento de cada frame enviado.
    """
    await websocket.accept()
    loop = asyncio.get_running_loop()
    settings = get_settings()

    try:
        while True:
            data = await websocket.receive_bytes()
            t0 = time.monotonic()

            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                await websocket.send_text(json.dumps({
                    "allowed": False,
                    "message": "Frame inválido.",
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }))
                continue

            # Detecção paralela de placa + rosto no thread pool
            result_plate, face_data = await asyncio.gather(
                loop.run_in_executor(None, recognize_plate_from_image, img),
                loop.run_in_executor(None, get_face_bbox_embedding_landmarks, img),
            )

            face_bbox_tuple, embedding, face_landmarks_raw = face_data
            face_bbox = list(face_bbox_tuple) if face_bbox_tuple else None
            face_landmarks = _to_serializable(face_landmarks_raw) if face_landmarks_raw else None

            vehicle_plate = None
            vehicle_authorized = None
            plate_bbox = None
            person_id = None
            person_name = None
            face_similarity = None
            message_parts = []
            allowed = False

            async with AsyncSessionLocal() as db:
                # --- Placa ---
                if result_plate and result_plate.normalized:
                    vehicle_plate = result_plate.normalized
                    plate_bbox = list(result_plate.bbox) if result_plate.bbox else None
                    q = select(Vehicle).where(
                        Vehicle.plate == vehicle_plate, Vehicle.is_active == True
                    )
                    v = (await db.execute(q)).scalar_one_or_none()
                    vehicle_authorized = v is not None
                    if not vehicle_authorized:
                        message_parts.append("Placa não cadastrada.")

                # --- Rosto ---
                if embedding is not None:
                    match = await find_best_match_pgvector(
                        embedding, db, tolerance=settings.face_tolerance
                    )
                    if match:
                        person_id = match.person_id
                        person_name = match.name
                        face_similarity = match.distance
                    else:
                        message_parts.append("Pessoa não reconhecida.")
                else:
                    if not vehicle_plate:
                        message_parts.append("Nenhum rosto nem placa detectados.")

                # --- Autorização: pedestre (sem veículo) ---
                if person_id is not None:
                    q_ped = select(Authorization).where(
                        Authorization.person_id == person_id,
                        Authorization.is_active == True,
                        Authorization.vehicle_id.is_(None),
                    )
                    if (await db.execute(q_ped)).first() is not None:
                        allowed = True

                # --- Autorização: pessoa + veículo ---
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

                # --- Autorização: só veículo ---
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
                        message_parts.append("Nenhum rosto nem placa reconhecidos.")
                    elif person_id is not None:
                        message_parts.append("Pessoa sem autorização.")
                    elif vehicle_plate and vehicle_authorized:
                        message_parts.append("Placa sem autorização (só veículo).")
                    else:
                        message_parts.append("Rosto ou placa não reconhecidos.")

                message = "Acesso autorizado." if allowed else (
                    " ".join(message_parts) or "Acesso negado."
                )
                latency_ms = int((time.monotonic() - t0) * 1000)

                try:
                    db.add(AccessLog(
                        person_id=person_id,
                        vehicle_plate=vehicle_plate,
                        allowed=allowed,
                        face_similarity=face_similarity,
                        latency_ms=latency_ms,
                        message=message,
                    ))
                    await db.commit()
                except Exception:
                    pass

            await websocket.send_text(json.dumps({
                "allowed": allowed,
                "person_id": person_id,
                "person_name": person_name,
                "vehicle_plate": vehicle_plate,
                "vehicle_authorized": vehicle_authorized,
                "face_bbox": face_bbox,
                "face_landmarks": face_landmarks,
                "plate_bbox": plate_bbox,
                "message": message,
                "latency_ms": latency_ms,
            }))

    except WebSocketDisconnect:
        pass
