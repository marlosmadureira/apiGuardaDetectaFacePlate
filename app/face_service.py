"""
Reconhecimento facial: crop do rosto, geração de embedding (cálculo matemático)
e comparação com banco para autorização.
"""
import os
import base64
import face_recognition
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class FaceMatch:
    """Resultado da comparação com uma pessoa cadastrada."""
    person_id: int
    name: str
    distance: float
    matched: bool


def _embedding_to_str(embedding: np.ndarray) -> str:
    """Converte array 128-d para string (armazenar no banco)."""
    return ",".join(str(float(x)) for x in embedding)


def _str_to_embedding(s: str) -> np.ndarray:
    """Converte string do banco de volta para array."""
    return np.array([float(x) for x in s.split(",")], dtype=np.float64)


def get_face_crop_and_embedding(
    image: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detecta o maior rosto na imagem, retorna (crop do rosto, embedding 128-d).
    image: BGR (OpenCV).
    """
    rgb = image[:, :, ::-1] if len(image.shape) == 3 else image
    rgb = np.ascontiguousarray(rgb)
    face_locations = face_recognition.face_locations(rgb)
    if not face_locations:
        return None, None
    # Maior face (por área)
    top, right, bottom, left = max(
        face_locations,
        key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]),
    )
    crop = image[top:bottom, left:right]
    # num_jitters=0 evita incompatibilidade com algumas versões do dlib (TypeError em compute_face_descriptor)
    encodings = face_recognition.face_encodings(rgb, [(top, right, bottom, left)], num_jitters=0)
    if not encodings:
        return crop, None
    return crop, encodings[0]


def embedding_from_image(image: np.ndarray) -> Optional[np.ndarray]:
    """Obtém apenas o embedding do rosto predominante na imagem."""
    _, emb = get_face_crop_and_embedding(image)
    return emb


def get_face_bbox_and_embedding(
    image: np.ndarray,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """
    Detecta o maior rosto na imagem, retorna (bbox, embedding).
    bbox: (left, top, width, height) em pixels; embedding 128-d.
    """
    rgb = image[:, :, ::-1] if len(image.shape) == 3 else image
    rgb = np.ascontiguousarray(rgb)
    face_locations = face_recognition.face_locations(rgb)
    if not face_locations:
        return None, None
    top, right, bottom, left = max(
        face_locations,
        key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]),
    )
    encodings = face_recognition.face_encodings(rgb, [(top, right, bottom, left)], num_jitters=0)
    if not encodings:
        return (left, top, right - left, bottom - top), None
    return (left, top, right - left, bottom - top), encodings[0]


def compare_face_to_embeddings(
    embedding: np.ndarray,
    stored_embeddings: List[Tuple[int, str, str]],  # (person_id, name, embedding_str)
    tolerance: float = 0.6,
) -> Optional[FaceMatch]:
    """
    Compara um embedding com uma lista de (id, nome, embedding_str).
    Retorna o melhor match se distância < tolerance.
    """
    best: Optional[FaceMatch] = None
    for person_id, name, emb_str in stored_embeddings:
        try:
            stored = _str_to_embedding(emb_str)
        except Exception:
            continue
        dist = float(face_recognition.face_distance([stored], embedding)[0])
        if dist <= tolerance and (best is None or dist < best.distance):
            best = FaceMatch(person_id=person_id, name=name, distance=dist, matched=True)
    return best


def save_crop(crop: np.ndarray, directory: str, prefix: str = "face") -> Optional[str]:
    """Salva o crop em disco; retorna o caminho ou None."""
    import cv2
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = os.path.join(directory, f"{prefix}_{os.urandom(4).hex()}.jpg")
    if cv2.imwrite(path, crop):
        return path
    return None


def embedding_to_base64(embedding: np.ndarray) -> str:
    """Para enviar embedding em JSON (opcional)."""
    return base64.b64encode(embedding.astype(np.float32).tobytes()).decode("utf-8")


def base64_to_embedding(b64: str) -> np.ndarray:
    """Decodifica embedding de base64."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float32)
