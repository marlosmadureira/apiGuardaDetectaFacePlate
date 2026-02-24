"""
CRUD de autorizações. Cada autorização é de um único tipo:
- Pedestre: person_id + vehicle_id null (entrada a pé).
- Pessoa + veículo: person_id + vehicle_id (entrada com aquele veículo, rosto + placa).
- Só veículo: person_id null + vehicle_id (entrada apenas pela placa, sem vínculo com pessoa).
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import Authorization, Person, Vehicle
from app.schemas import AuthorizationCreate, AuthorizationResponse

router = APIRouter(prefix="/authorizations", tags=["Autorizações"])


@router.post("", response_model=AuthorizationResponse)
async def create_authorization(
    data: AuthorizationCreate, db: AsyncSession = Depends(get_db)
):
    """
    Cria autorização. Tipos:
    - Pedestre: person_id preenchido, vehicle_id null.
    - Veículo (pessoa + placa): person_id + vehicle_id.
    - Só veículo: person_id null, vehicle_id preenchido (não relaciona pessoa).
    """
    person_id = data.person_id
    vehicle_id = data.vehicle_id if data.vehicle_id else None
    if person_id is None and vehicle_id is None:
        raise HTTPException(
            status_code=400,
            detail="Informe person_id (pedestre ou pessoa+veículo) ou apenas vehicle_id (só veículo).",
        )
    if person_id is not None:
        person = await db.get(Person, person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Pessoa não encontrada.")
    if vehicle_id is not None:
        vehicle = await db.get(Vehicle, vehicle_id)
        if vehicle is None:
            raise HTTPException(status_code=404, detail="Veículo não encontrado.")
    auth = Authorization(person_id=person_id, vehicle_id=vehicle_id)
    db.add(auth)
    await db.commit()
    await db.refresh(auth)
    return auth


@router.get("", response_model=list[AuthorizationResponse])
async def list_authorizations(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    active_only: bool = True,
    db: AsyncSession = Depends(get_db),
):
    q = select(Authorization).order_by(Authorization.id).offset(skip).limit(limit)
    if active_only:
        q = q.where(Authorization.is_active == True)
    result = await db.execute(q)
    return list(result.scalars().all())


@router.delete("/{authorization_id}", status_code=204)
async def delete_authorization(
    authorization_id: int, db: AsyncSession = Depends(get_db)
):
    """
    Exclui a autorização (soft delete: is_active=False).
    """
    auth = await db.get(Authorization, authorization_id)
    if auth is None:
        raise HTTPException(status_code=404, detail="Autorização não encontrada.")
    auth.is_active = False
    await db.commit()
    return None
