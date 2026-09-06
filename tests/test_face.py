"""Testes de cadastro e verificação facial."""
import io
import pytest
import numpy as np
import cv2
from unittest.mock import patch, AsyncMock, MagicMock


def _make_blank_jpeg() -> bytes:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.mark.asyncio
async def test_register_face_person_not_found(client):
    """Cadastrar rosto em ID inexistente deve retornar 404."""
    img_bytes = _make_blank_jpeg()
    r = await client.post(
        "/face/register/999999",
        files={"file": ("face.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert r.status_code in (404, 400)


@pytest.mark.asyncio
async def test_register_face_no_face_detected(client):
    """Imagem sem rosto detectável retorna 400 (após criar a pessoa)."""
    person_r = await client.post("/persons", json={"name": "Teste Rosto"})
    person_id = person_r.json()["id"]

    img_bytes = _make_blank_jpeg()

    # Mock para simular que nenhum rosto foi detectado
    with patch(
        "app.face_service.get_face_crop_and_embedding",
        return_value=(None, None),
    ):
        r = await client.post(
            f"/face/register/{person_id}",
            files={"file": ("face.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_verify_face_no_face_detected(client):
    """Verificar imagem sem rosto retorna matched=False."""
    img_bytes = _make_blank_jpeg()

    with patch(
        "app.face_service.embedding_from_image",
        return_value=None,
    ):
        r = await client.post(
            "/face/verify",
            files={"file": ("face.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
    assert r.status_code == 200
    assert r.json()["matched"] is False


@pytest.mark.asyncio
async def test_register_and_verify_face(client):
    """Ciclo completo: criar pessoa → cadastrar rosto → verificar correspondência."""
    # Cria pessoa
    pr = await client.post("/persons", json={"name": "José Teste"})
    person_id = pr.json()["id"]

    img_bytes = _make_blank_jpeg()
    fake_embedding = np.random.rand(512).astype(np.float32)
    fake_crop = np.zeros((80, 80, 3), dtype=np.uint8)

    # Mock do InsightFace para retornar embedding sintético
    with patch(
        "app.face_service.get_face_crop_and_embedding",
        return_value=(fake_crop, fake_embedding),
    ), patch(
        "app.face_service.find_best_match_pgvector",
        new=AsyncMock(return_value=None),
    ):
        reg_r = await client.post(
            f"/face/register/{person_id}",
            files={"file": ("face.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
    assert reg_r.status_code == 200

    # Mock para verificação: retorna o match encontrado
    from app.face_service import FaceMatch
    fake_match = FaceMatch(
        person_id=person_id, name="José Teste", distance=0.92, matched=True
    )
    with patch(
        "app.face_service.embedding_from_image",
        return_value=fake_embedding,
    ), patch(
        "app.face_service.find_best_match_pgvector",
        new=AsyncMock(return_value=fake_match),
    ):
        ver_r = await client.post(
            "/face/verify",
            files={"file": ("face.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
    assert ver_r.status_code == 200
    assert ver_r.json()["matched"] is True
    assert ver_r.json()["person_id"] == person_id
