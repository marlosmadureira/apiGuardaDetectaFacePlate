"""Testes do endpoint de reconhecimento de placa por upload."""
import io
import pytest
import numpy as np
import cv2


def _make_blank_image_bytes(width: int = 640, height: int = 480) -> bytes:
    """Gera um JPEG de imagem preta para testes sem câmera ou placa real."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.mark.asyncio
async def test_plate_upload_blank_image(client):
    """Imagem sem placa retorna resposta válida (plate vazio, não 500)."""
    img_bytes = _make_blank_image_bytes()
    r = await client.post(
        "/plate/capture/upload",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "plate" in data
    assert "format_type" in data
    assert "forwarded" in data


@pytest.mark.asyncio
async def test_plate_upload_empty_file(client):
    """Arquivo vazio deve retornar 400."""
    r = await client.post(
        "/plate/capture/upload",
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_plate_upload_invalid_file(client):
    """Bytes inválidos devem retornar 400."""
    r = await client.post(
        "/plate/capture/upload",
        files={"file": ("bad.jpg", io.BytesIO(b"not-an-image"), "image/jpeg")},
    )
    assert r.status_code == 400
