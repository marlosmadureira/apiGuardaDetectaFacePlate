"""Modelos SQLAlchemy para autorizações, pessoas e veículos."""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base


class Person(Base):
    """Pessoa cadastrada com embedding facial para reconhecimento."""

    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    document = Column(String(50), index=True)  # CPF/RG opcional
    face_embedding = Column(Text, nullable=True)  # 128 floats em texto (CSV); preenchido ao cadastrar rosto
    face_photo_path = Column(String(512))  # caminho do crop salvo (opcional)
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
    Autorização de entrada: sempre uma única modalidade por registro.
    - vehicle_id = NULL: entrada a pé (só verificação facial).
    - vehicle_id preenchido: entrada com veículo (verificação facial + placa).
    Nunca exige os dois ao mesmo tempo; ou a pessoa entra a pé ou com aquele veículo.
    """

    __tablename__ = "authorizations"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=True)  # null = só pessoa a pé
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    person = relationship("Person", back_populates="authorizations")
    vehicle = relationship("Vehicle", back_populates="authorizations")
