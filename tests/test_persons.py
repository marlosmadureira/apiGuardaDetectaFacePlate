"""Testes CRUD de pessoas — inclui soft delete."""
import pytest


@pytest.mark.asyncio
async def test_create_person(client):
    r = await client.post("/persons", json={"name": "Ana Silva"})
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "Ana Silva"
    assert data["is_active"] is True
    assert "id" in data


@pytest.mark.asyncio
async def test_list_persons(client):
    await client.post("/persons", json={"name": "Carlos Souza"})
    r = await client.get("/persons")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    assert len(r.json()) >= 1


@pytest.mark.asyncio
async def test_get_person(client):
    r = await client.post("/persons", json={"name": "Maria Costa"})
    person_id = r.json()["id"]
    r2 = await client.get(f"/persons/{person_id}")
    assert r2.status_code == 200
    assert r2.json()["name"] == "Maria Costa"


@pytest.mark.asyncio
async def test_get_person_not_found(client):
    r = await client.get("/persons/999999")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_soft_delete_person(client):
    """DELETE deve desativar (is_active=False), não apagar o registro."""
    r = await client.post("/persons", json={"name": "Pedro Lima"})
    person_id = r.json()["id"]

    del_r = await client.delete(f"/persons/{person_id}")
    assert del_r.status_code == 204

    # Pessoa ainda existe no banco (soft delete)
    r2 = await client.get(f"/persons/{person_id}")
    assert r2.status_code == 200
    assert r2.json()["is_active"] is False

    # Não aparece na listagem padrão (active_only=True)
    list_r = await client.get("/persons?active_only=true")
    ids = [p["id"] for p in list_r.json()]
    assert person_id not in ids
