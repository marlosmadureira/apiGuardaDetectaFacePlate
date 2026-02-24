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
# Tipos: Pedestre (person_id + vehicle_id null), Veículo com pessoa (person_id + vehicle_id), Só veículo (person_id null + vehicle_id).
class AuthorizationCreate(BaseModel):
    person_id: Optional[int] = None  # null = autorização só do veículo (obrigatório vehicle_id)
    vehicle_id: Optional[int] = None  # null = entrada a pé; preenchido = com veículo ou só veículo


class AuthorizationResponse(BaseModel):
    id: int
    person_id: Optional[int] = None
    vehicle_id: Optional[int] = None
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
    face_bbox: Optional[List[int]] = None   # [x, y, width, height] rosto detectado
    plate_bbox: Optional[List[int]] = None  # [x, y, width, height] placa detectada
    message: str
