"""
CRUD de autorizações. Cada autorização é de um único tipo:
- Só pessoa (vehicle_id = null): entrada a pé → verificação apenas facial.
- Pessoa + veículo (vehicle_id preenchido): entrada com veículo → verificação facial + placa.
Nunca é "veículo e pessoa" obrigatórios juntos; ou entra a pé ou com aquele veículo.
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
    Cria autorização de entrada. Envie:
    - person_id + vehicle_id = null → entrada a pé (só facial).
    - person_id + vehicle_id → entrada com veículo (facial + placa).
    """
    person = await db.get(Person, data.person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada.")
    if data.vehicle_id is not None:
        vehicle = await db.get(Vehicle, data.vehicle_id)
        if vehicle is None:
            raise HTTPException(status_code=404, detail="Veículo não encontrado.")
    auth = Authorization(person_id=data.person_id, vehicle_id=data.vehicle_id)
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
