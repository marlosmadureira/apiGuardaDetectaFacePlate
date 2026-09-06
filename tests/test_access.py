"""Testes do endpoint de controle de acesso."""
import io
import pytest
import numpy as np
import cv2
from unittest.mock import patch, AsyncMock


def _make_blank_jpeg() -> bytes:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.mark.asyncio
async def test_access_check_blank_image(client):
    """Imagem sem rosto nem placa retorna allowed=False, sem erro 500."""
    img_bytes = _make_blank_jpeg()

    # Mock: nenhum rosto e nenhuma placa detectados
    with patch(
        "app.plate_recognizer.recognize_plate_from_image", return_value=None
    ), patch(
        "app.face_service.get_face_bbox_embedding_landmarks",
        return_value=(None, None, None),
    ):
        r = await client.post(
            "/access/check",
            files={"face_image": ("frame.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["allowed"] is False
    assert "message" in data


@pytest.mark.asyncio
async def test_access_check_empty_image(client):
    """Imagem vazia retorna allowed=False com mensagem."""
    r = await client.post(
        "/access/check",
        files={"face_image": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert r.status_code == 200
    assert r.json()["allowed"] is False


@pytest.mark.asyncio
async def test_access_check_authorized_pedestrian(client):
    """Rosto reconhecido + autorização de pedestre → allowed=True."""
    from app.face_service import FaceMatch
    from app.models import Person, Authorization

    # Cria pessoa e autorização a pé
    pr = await client.post("/persons", json={"name": "Autorizado Teste"})
    person_id = pr.json()["id"]

    auth_r = await client.post(
        "/authorizations", json={"person_id": person_id, "vehicle_id": None}
    )
    assert auth_r.status_code == 200

    img_bytes = _make_blank_jpeg()
    fake_bbox = (10, 10, 100, 100)
    fake_embedding = np.random.rand(512).astype(np.float32)
    fake_match = FaceMatch(
        person_id=person_id, name="Autorizado Teste", distance=0.88, matched=True
    )

    with patch(
        "app.plate_recognizer.recognize_plate_from_image", return_value=None
    ), patch(
        "app.face_service.get_face_bbox_embedding_landmarks",
        return_value=(fake_bbox, fake_embedding, {}),
    ), patch(
        "app.face_service.find_best_match_pgvector",
        new=AsyncMock(return_value=fake_match),
    ):
        r = await client.post(
            "/access/check",
            files={"face_image": ("frame.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )

    assert r.status_code == 200
    data = r.json()
    assert data["allowed"] is True
    assert data["person_id"] == person_id


@pytest.mark.asyncio
async def test_access_check_not_authorized(client):
    """Rosto reconhecido mas sem autorização → allowed=False."""
    from app.face_service import FaceMatch

    pr = await client.post("/persons", json={"name": "Sem Autorização"})
    person_id = pr.json()["id"]

    img_bytes = _make_blank_jpeg()
    fake_embedding = np.random.rand(512).astype(np.float32)
    fake_match = FaceMatch(
        person_id=person_id, name="Sem Autorização", distance=0.85, matched=True
    )

    with patch(
        "app.plate_recognizer.recognize_plate_from_image", return_value=None
    ), patch(
        "app.face_service.get_face_bbox_embedding_landmarks",
        return_value=((10, 10, 80, 80), fake_embedding, {}),
    ), patch(
        "app.face_service.find_best_match_pgvector",
        new=AsyncMock(return_value=fake_match),
    ):
        r = await client.post(
            "/access/check",
            files={"face_image": ("frame.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )

    assert r.status_code == 200
    assert r.json()["allowed"] is False
