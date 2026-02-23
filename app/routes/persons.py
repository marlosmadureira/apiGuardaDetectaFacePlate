"""CRUD de pessoas (para vincular rosto e autorizações)."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import Person
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
    db: AsyncSession = Depends(get_db),
):
    q = select(Person).offset(skip).limit(limit).order_by(Person.id)
    result = await db.execute(q)
    return list(result.scalars().all())


@router.get("/{person_id}", response_model=PersonResponse)
async def get_person(person_id: int, db: AsyncSession = Depends(get_db)):
    person = await db.get(Person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada.")
    return person
