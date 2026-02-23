"""
Reconhecimento de placas brasileiras:
- Formato antigo: ABC-1234 (cinza, letras pretas)
- Formato Mercosul: ABC1D23 (branca, letras azuis) — 3 letras, 1 número, 1 letra, 2 números
"""
import re
import cv2
import numpy as np
import pytesseract
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PlateResult:
    """Resultado do reconhecimento da placa."""
    raw_text: str
    normalized: str  # apenas letras e números, formatado
    format_type: str  # "old" ou "mercosul"
    confidence: float
    roi: Optional[np.ndarray] = None  # região da placa (opcional)


# Padrões Brasil (texto normalizado, sem hífen):
# Antigo: 3 letras + 4 dígitos (ABC1234)
# Mercosul: 3 letras + 1 dígito + 1 letra + 2 dígitos (ABC1D23)
OLD_PLATE_RE = re.compile(r"^[A-Z]{3}[0-9]{4}$")
MERCOSUL_PLATE_RE = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")


def _normalize_plate_text(text: str) -> str:
    """Remove espaços e deixa só letras/números; maiúsculas."""
    s = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return s


def _classify_plate(normalized: str) -> str:
    """Classifica como 'old' ou 'mercosul'."""
    if len(normalized) == 7 and MERCOSUL_PLATE_RE.match(normalized):
        return "mercosul"
    if len(normalized) == 7 and OLD_PLATE_RE.match(normalized):
        return "old"
    if len(normalized) == 7:
        return "mercosul"
    return "unknown"


def _format_display(normalized: str, format_type: str) -> str:
    """Formata para exibição: antigo com hífen, Mercosul sem."""
    if format_type == "old" and len(normalized) >= 7:
        return f"{normalized[:3]}-{normalized[3:7]}"
    return normalized[:7] if len(normalized) >= 7 else normalized


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Pré-processamento para melhorar OCR em placas (cinza ou branca)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Aumentar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Binarização adaptativa (funciona para fundo claro e escuro)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh


def _find_plate_contours(image: np.ndarray) -> list:
    """Encontra retângulos que podem ser placas (proporção ~2:1)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 80 or h < 20:
            continue
        aspect = w / float(h) if h else 0
        if 2.0 <= aspect <= 5.5:  # proporção típica de placa
            candidates.append((x, y, w, h))
    return sorted(candidates, key=lambda r: r[2] * r[3], reverse=True)[:5]


def recognize_plate_from_image(image: np.ndarray) -> Optional[PlateResult]:
    """
    Reconhece placa em uma imagem (BGR).
    Tenta primeiro encontrar ROI da placa por contornos; se não achar, usa imagem inteira.
    """
    if image is None or image.size == 0:
        return None

    # Tenta encontrar região da placa
    rois = _find_plate_contours(image)
    best: Optional[PlateResult] = None

    for (x, y, w, h) in rois:
        roi = image[y : y + h, x : x + w]
        proc = _preprocess_for_ocr(roi)
        config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(proc, config=config).strip()
        text = _normalize_plate_text(text)
        if len(text) >= 6:
            fmt = _classify_plate(text)
            display = _format_display(text, fmt)
            result = PlateResult(
                raw_text=text,
                normalized=display,
                format_type=fmt,
                confidence=0.8,
                roi=roi,
            )
            if best is None or len(text) >= len(best.raw_text):
                best = result

    # Fallback: imagem inteira
    if best is None:
        proc = _preprocess_for_ocr(image)
        config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(proc, config=config).strip()
        text = _normalize_plate_text(text)
        if len(text) >= 6:
            fmt = _classify_plate(text)
            display = _format_display(text, fmt)
            best = PlateResult(
                raw_text=text,
                normalized=display,
                format_type=fmt,
                confidence=0.5,
                roi=None,
            )

    return best


def capture_frame(camera_index: int = 0) -> Optional[np.ndarray]:
    """Captura um frame da câmera."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None
