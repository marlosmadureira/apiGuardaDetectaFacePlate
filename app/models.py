"""Modelos SQLAlchemy para autorizações, pessoas, veículos e log de acesso."""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base

try:
    from pgvector.sqlalchemy import Vector
    _VECTOR_TYPE = Vector(512)
except ImportError:
    # Fallback para Text se pgvector não estiver disponível (nunca deve ocorrer em produção)
    from sqlalchemy import Text
    _VECTOR_TYPE = Text()


class Person(Base):
    """Pessoa cadastrada com embedding facial ArcFace 512-d para reconhecimento."""

    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    document = Column(String(50), index=True)  # CPF/RG opcional
    face_embedding = Column(_VECTOR_TYPE, nullable=True)  # ArcFace 512-d float32
    face_photo_path = Column(String(512))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    authorizations = relationship("Authorization", back_populates="person")


class Vehicle(Base):
    """Veículo com placa (apenas referência; placa pode ser validada externamente)."""

    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, index=True)
    plate = Column(String(20), unique=True, nullable=False, index=True)
    description = Column(String(255))  # ex: modelo, cor
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    authorizations = relationship("Authorization", back_populates="vehicle")


class Authorization(Base):
    """
    Autorização de entrada: uma modalidade por registro.
    - person_id preenchido, vehicle_id NULL: entrada a pé (só verificação facial).
    - person_id preenchido, vehicle_id preenchido: entrada com veículo (rosto + placa).
    - person_id NULL, vehicle_id preenchido: só veículo (entrada apenas pela placa).
    """

    __tablename__ = "authorizations"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    person = relationship("Person", back_populates="authorizations")
    vehicle = relationship("Vehicle", back_populates="authorizations")


class AccessLog(Base):
    """Log de cada tentativa de acesso para auditoria e monitoramento."""

    __tablename__ = "access_logs"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, nullable=True)
    vehicle_plate = Column(String(20), nullable=True)
    allowed = Column(Boolean, nullable=False)
    face_similarity = Column(Float, nullable=True)  # similaridade coseno [0,1]
    latency_ms = Column(Integer, nullable=False)
    message = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
