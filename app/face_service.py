"""
Reconhecimento facial via InsightFace (ArcFace buffalo_sc).
Embeddings 512-d float32, L2-normalizados; similaridade coseno.
"""
import os
import base64
import threading
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

try:
    from insightface.app import FaceAnalysis as _FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _FaceAnalysis = None
    _INSIGHTFACE_AVAILABLE = False


@dataclass
class FaceMatch:
    """Resultado da comparação com uma pessoa cadastrada."""
    person_id: int
    name: str
    distance: float  # similaridade coseno [0,1] — maior = mais parecido
    matched: bool


# Singleton thread-safe do modelo InsightFace
_face_app: Optional[Any] = None
_face_lock = threading.Lock()


def _get_face_app() -> Any:
    global _face_app
    if _face_app is None:
        with _face_lock:
            if _face_app is None:
                if not _INSIGHTFACE_AVAILABLE:
                    raise RuntimeError("insightface não instalado. Execute: pip install insightface onnxruntime")
                app = _FaceAnalysis(
                    name="buffalo_sc",
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(320, 320))
                _face_app = app
    return _face_app


def _embedding_to_str(embedding: np.ndarray) -> str:
    """Serializa embedding float32 para base64 ASCII (rápido e compacto)."""
    return base64.b64encode(embedding.astype(np.float32).tobytes()).decode("ascii")


def _str_to_embedding(s: str) -> np.ndarray:
    """Deserializa embedding de base64."""
    raw = base64.b64decode(s)
    return np.frombuffer(raw, dtype=np.float32).copy()


def get_face_crop_and_embedding(
    image: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detecta o maior rosto na imagem (BGR) e retorna (crop BGR, embedding 512-d float32).
    """
    app = _get_face_app()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)
    if not faces:
        return None, None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    crop = image[y1:y2, x1:x2]
    return crop, face.embedding.astype(np.float32)


def embedding_from_image(image: np.ndarray) -> Optional[np.ndarray]:
    """Obtém apenas o embedding do rosto predominante na imagem."""
    _, emb = get_face_crop_and_embedding(image)
    return emb


def get_face_bbox_and_embedding(
    image: np.ndarray,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    bbox, emb, _ = get_face_bbox_embedding_landmarks(image)
    return bbox, emb


def get_face_bbox_embedding_landmarks(
    image: np.ndarray,
) -> Tuple[
    Optional[Tuple[int, int, int, int]],
    Optional[np.ndarray],
    Optional[Dict[str, List[List[int]]]],
]:
    """
    Detecta o maior rosto e retorna (bbox, embedding, landmarks).
    bbox: (x, y, w, h). Landmarks: 5 pontos InsightFace mapeados para dict.
    """
    app = _get_face_app()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)
    if not faces:
        return None, None, None

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = face.bbox.astype(int)
    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    embedding = face.embedding.astype(np.float32)

    # InsightFace fornece 5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth
    landmarks: Dict[str, List[List[int]]] = {
        "left_eye": [], "right_eye": [], "nose_tip": [],
        "left_mouth": [], "right_mouth": [],
    }
    if face.kps is not None:
        kps = face.kps.astype(int).tolist()
        keys = ["left_eye", "right_eye", "nose_tip", "left_mouth", "right_mouth"]
        for i, key in enumerate(keys):
            if i < len(kps):
                landmarks[key] = [[kps[i][0], kps[i][1]]]

    return bbox, embedding, landmarks


def compare_face_to_embeddings(
    embedding: np.ndarray,
    stored_embeddings: List[Tuple[int, str, str]],
    tolerance: float = 0.5,
) -> Optional[FaceMatch]:
    """
    Compara embedding com lista de (person_id, name, embedding_str).
    Usa similaridade coseno — embeddings ArcFace são L2-normalizados,
    então dot product = cosine similarity. Aceita se similarity >= tolerance.
    tolerance maior = mais rigoroso (requer maior similaridade).
    """
    best: Optional[FaceMatch] = None
    norm = np.linalg.norm(embedding)
    emb_norm = embedding / (norm + 1e-6)
    for person_id, name, emb_str in stored_embeddings:
        try:
            stored = _str_to_embedding(emb_str)
        except Exception:
            continue
        stored_n = stored / (np.linalg.norm(stored) + 1e-6)
        similarity = float(np.dot(emb_norm, stored_n))
        if similarity >= tolerance and (best is None or similarity > best.distance):
            best = FaceMatch(person_id=person_id, name=name, distance=similarity, matched=True)
    return best


import logging as _logging
_log = _logging.getLogger(__name__)


async def find_best_match_pgvector(
    embedding: np.ndarray,
    db: "AsyncSession",
    tolerance: float = 0.5,
    exclude_person_id: Optional[int] = None,
) -> Optional[FaceMatch]:
    """
    Busca no banco usando índice HNSW de pgvector (recall exato, sem probes).
    Operador <=> = distância coseno; similarity = 1 - distância.
    exclude_person_id: ignora esse ID (útil para checar duplicatas ao cadastrar).
    """
    from sqlalchemy import text as _text

    emb_str = "[" + ",".join(f"{float(x):.8f}" for x in embedding) + "]"
    where_extra = f"AND id != {exclude_person_id}" if exclude_person_id else ""

    result = await db.execute(
        _text(f"""
            SELECT id, name,
                   1.0 - (face_embedding <=> CAST(:emb AS vector)) AS similarity
            FROM persons
            WHERE is_active = true
              AND face_embedding IS NOT NULL
              {where_extra}
            ORDER BY face_embedding <=> CAST(:emb AS vector)
            LIMIT 1
        """),
        {"emb": emb_str},
    )
    row = result.first()
    if row is None:
        _log.info("pgvector: nenhuma pessoa com rosto cadastrado encontrada")
        return None
    person_id, name, similarity = int(row[0]), str(row[1]), float(row[2])
    _log.info(f"pgvector melhor match: id={person_id} name={name!r} similarity={similarity:.4f} threshold={tolerance}")
    if similarity >= tolerance:
        return FaceMatch(person_id=person_id, name=name, distance=similarity, matched=True)
    return None


def save_crop(crop: np.ndarray, directory: str, prefix: str = "face") -> Optional[str]:
    """Salva o crop em disco; retorna o caminho ou None."""
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
    return np.frombuffer(raw, dtype=np.float32).copy()
