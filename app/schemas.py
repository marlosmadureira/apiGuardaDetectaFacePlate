"""Schemas Pydantic para request/response."""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# --- Placa ---
class PlateCaptureResponse(BaseModel):
    plate: str
    format_type: str  # old | mercosul
    forwarded: bool = False
    forward_response: Optional[dict] = None
    message: Optional[str] = None


# --- Pessoa / Face ---
class PersonCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    document: Optional[str] = Field(None, max_length=50)


class PersonResponse(BaseModel):
    id: int
    name: str
    document: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class FaceRegisterResponse(BaseModel):
    person_id: int
    name: str
    message: str


class FaceVerifyResponse(BaseModel):
    matched: bool
    person_id: Optional[int] = None
    name: Optional[str] = None
    distance: Optional[float] = None
    message: str


# --- Veículo ---
class VehicleCreate(BaseModel):
    plate: str = Field(..., min_length=6, max_length=20)
    description: Optional[str] = None


class VehicleResponse(BaseModel):
    id: int
    plate: str
    description: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# --- Autorização ---
# Uma autorização é sempre de um tipo: entrada a pé (vehicle_id null) ou com veículo (vehicle_id preenchido).
class AuthorizationCreate(BaseModel):
    person_id: int
    vehicle_id: Optional[int] = None  # null = entrada a pé (só facial); preenchido = entrada com veículo (facial + placa)


class AuthorizationResponse(BaseModel):
    id: int
    person_id: int
    vehicle_id: Optional[int]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# --- Controle de acesso (resposta unificada) ---
class AccessCheckResponse(BaseModel):
    allowed: bool
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    vehicle_plate: Optional[str] = None
    vehicle_authorized: Optional[bool] = None
    message: str
