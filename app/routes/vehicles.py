"""CRUD de veículos (placas autorizadas)."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.database import get_db
from app.models import Vehicle, Authorization
from app.schemas import VehicleCreate, VehicleResponse

router = APIRouter(prefix="/vehicles", tags=["Veículos"])


@router.post("", response_model=VehicleResponse)
async def create_vehicle(data: VehicleCreate, db: AsyncSession = Depends(get_db)):
    plate_upper = data.plate.upper().strip()
    q = select(Vehicle).where(Vehicle.plate == plate_upper)
    existing = (await db.execute(q)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Placa já cadastrada.")
    vehicle = Vehicle(plate=plate_upper, description=data.description)
    db.add(vehicle)
    await db.commit()
    await db.refresh(vehicle)
    return vehicle


@router.get("", response_model=list[VehicleResponse])
async def list_vehicles(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    active_only: bool = Query(True, description="Se True, lista só veículos ativos (não excluídos)"),
    db: AsyncSession = Depends(get_db),
):
    q = select(Vehicle).offset(skip).limit(limit).order_by(Vehicle.id)
    if active_only:
        q = q.where(Vehicle.is_active == True)
    result = await db.execute(q)
    return list(result.scalars().all())


@router.get("/{vehicle_id}", response_model=VehicleResponse)
async def get_vehicle(vehicle_id: int, db: AsyncSession = Depends(get_db)):
    vehicle = await db.get(Vehicle, vehicle_id)
    if vehicle is None:
        raise HTTPException(status_code=404, detail="Veículo não encontrado.")
    return vehicle


@router.delete("/{vehicle_id}", status_code=204)
async def delete_vehicle(vehicle_id: int, db: AsyncSession = Depends(get_db)):
    """
    Exclui o veículo e, em cascata, todas as autorizações vinculadas a ele.
    """
    vehicle = await db.get(Vehicle, vehicle_id)
    if vehicle is None:
        raise HTTPException(status_code=404, detail="Veículo não encontrado.")
    await db.execute(delete(Authorization).where(Authorization.vehicle_id == vehicle_id))
    await db.delete(vehicle)
    await db.commit()
    return None


@router.get("/by-plate/{plate}", response_model=VehicleResponse)
async def get_vehicle_by_plate(plate: str, db: AsyncSession = Depends(get_db)):
    plate_upper = plate.upper().strip()
    q = select(Vehicle).where(Vehicle.plate == plate_upper, Vehicle.is_active == True)
    result = (await db.execute(q)).scalar_one_or_none()
    if result is None:
        raise HTTPException(status_code=404, detail="Veículo não encontrado.")
    return result
