"""
Rotas de captura e reconhecimento de placa (câmera ou upload).
Extrai caracteres e encaminha para endpoint externo configurável.
"""
import cv2
import numpy as np
import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.config import get_settings
from app.plate_recognizer import capture_frame, recognize_plate_from_image
from app.schemas import PlateCaptureResponse

router = APIRouter(prefix="/plate", tags=["Placa"])


def _decode_image_from_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@router.post("/capture", response_model=PlateCaptureResponse)
async def capture_plate_from_camera():
    """
    Captura um frame da câmera do computador, reconhece a placa (Brasil/Mercosul)
    e opcionalmente encaminha para um servidor externo.
    """
    settings = get_settings()
    frame = capture_frame(settings.camera_index)
    if frame is None:
        raise HTTPException(
            status_code=503,
            detail="Não foi possível acessar a câmera. Verifique se está conectada e se o Docker tem permissão (--device /dev/video0).",
        )
    result = recognize_plate_from_image(frame)
    if result is None:
        return PlateCaptureResponse(
            plate="",
            format_type="unknown",
            forwarded=False,
            message="Nenhuma placa identificada na imagem.",
        )

    forward_response = None
    forwarded = False
    if settings.plate_forward_enabled and settings.plate_forward_url:
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.post(
                    settings.plate_forward_url,
                    json={
                        "plate": result.normalized,
                        "format_type": result.format_type,
                        "raw_text": result.raw_text,
                    },
                )
                forwarded = True
                forward_response = {"status_code": r.status_code, "body": r.text}
            except Exception as e:
                forward_response = {"error": str(e)}

    return PlateCaptureResponse(
        plate=result.normalized,
        format_type=result.format_type,
        forwarded=forwarded,
        forward_response=forward_response,
        message="Placa reconhecida com sucesso.",
    )


@router.post("/capture/upload", response_model=PlateCaptureResponse)
async def capture_plate_from_upload(file: UploadFile = File(...)):
    """
    Reconhece placa a partir de uma imagem enviada (útil quando a câmera
    não está disponível no container).
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    image = _decode_image_from_upload(content)
    if image is None:
        raise HTTPException(status_code=400, detail="Imagem inválida.")
    result = recognize_plate_from_image(image)
    if result is None:
        return PlateCaptureResponse(
            plate="",
            format_type="unknown",
            forwarded=False,
            message="Nenhuma placa identificada na imagem.",
        )

    settings = get_settings()
    forward_response = None
    forwarded = False
    if settings.plate_forward_enabled and settings.plate_forward_url:
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.post(
                    settings.plate_forward_url,
                    json={
                        "plate": result.normalized,
                        "format_type": result.format_type,
                        "raw_text": result.raw_text,
                    },
                )
                forwarded = True
                forward_response = {"status_code": r.status_code, "body": r.text}
            except Exception as e:
                forward_response = {"error": str(e)}

    return PlateCaptureResponse(
        plate=result.normalized,
        format_type=result.format_type,
        forwarded=forwarded,
        forward_response=forward_response,
        message="Placa reconhecida com sucesso.",
    )
