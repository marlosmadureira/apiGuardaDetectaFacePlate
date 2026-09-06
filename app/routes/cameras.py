"""
Gerenciamento de fontes de câmera.
Permite listar câmeras configuradas, recarregar o JSON de configuração
e capturar um snapshot de qualquer fonte (local ou stream de rede).
"""
import asyncio
import cv2
from fastapi import APIRouter, HTTPException, Response

router = APIRouter(prefix="/cameras", tags=["Câmeras"])


@router.get(
    "",
    summary="Lista câmeras configuradas",
    description=(
        "Retorna todas as câmeras definidas em `data/cameras.json` "
        "com seu status atual (online / offline / disabled / reconnecting)."
    ),
)
async def list_cameras():
    from app.camera import get_registry
    return get_registry().list_cameras()


@router.post(
    "/reload",
    summary="Recarrega configuração de câmeras",
    description=(
        "Relê `data/cameras.json` sem reiniciar o servidor. "
        "Use após editar o arquivo para adicionar/remover fontes de vídeo."
    ),
)
async def reload_cameras():
    from app.camera import get_registry
    get_registry().reload()
    cameras = get_registry().list_cameras()
    return {"message": f"Configuração recarregada. {len(cameras)} câmera(s) registrada(s).", "cameras": cameras}


@router.get(
    "/{camera_id}/snapshot",
    summary="Captura snapshot JPEG de uma câmera",
    description=(
        "Abre (ou reusa) a conexão com a câmera identificada por `camera_id` "
        "e retorna um frame JPEG. Útil para testar conectividade antes de usar "
        "a câmera em modo de verificação contínua."
    ),
    response_class=Response,
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "Frame capturado com sucesso."},
        503: {"description": "Câmera indisponível."},
        404: {"description": "Câmera não encontrada na configuração."},
    },
)
async def camera_snapshot(camera_id: str):
    from app.camera import get_registry
    registry = get_registry()
    cameras = {c["id"] for c in registry.list_cameras()}
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail=f"Câmera '{camera_id}' não encontrada na configuração.")

    loop = asyncio.get_running_loop()
    frame = await loop.run_in_executor(None, registry.read_frame, camera_id)
    if frame is None:
        raise HTTPException(
            status_code=503,
            detail=f"Câmera '{camera_id}' indisponível. Verifique conexão e configuração.",
        )
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=buf.tobytes(), media_type="image/jpeg")
