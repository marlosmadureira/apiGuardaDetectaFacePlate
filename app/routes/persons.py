"""CRUD de pessoas (para vincular rosto e autorizações)."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.database import get_db
from app.models import Person, Authorization
from app.schemas import PersonCreate, PersonResponse

router = APIRouter(prefix="/persons", tags=["Pessoas"])


@router.post("", response_model=PersonResponse)
async def create_person(data: PersonCreate, db: AsyncSession = Depends(get_db)):
    person = Person(name=data.name, document=data.document)
    db.add(person)
    await db.commit()
    await db.refresh(person)
    return person


@router.get("", response_model=list[PersonResponse])
async def list_persons(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    active_only: bool = Query(True, description="Se True, lista só pessoas ativas (não excluídas)"),
    db: AsyncSession = Depends(get_db),
):
    q = select(Person).offset(skip).limit(limit).order_by(Person.id)
    if active_only:
        q = q.where(Person.is_active == True)
    result = await db.execute(q)
    return list(result.scalars().all())


@router.get("/{person_id}", response_model=PersonResponse)
async def get_person(person_id: int, db: AsyncSession = Depends(get_db)):
    person = await db.get(Person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada.")
    return person


@router.delete("/{person_id}", status_code=204)
async def delete_person(person_id: int, db: AsyncSession = Depends(get_db)):
    """
    Exclui a pessoa e, em cascata, todas as autorizações vinculadas a ela.
    """
    person = await db.get(Person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada.")
    await db.execute(delete(Authorization).where(Authorization.person_id == person_id))
    await db.delete(person)
    await db.commit()
    return None
