"""
Reconhecimento de placas brasileiras:
- Formato antigo: ABC-1234 (3 letras + 4 dígitos)
- Formato Mercosul: ABC1D23 (3 letras + 1 dígito + 1 letra + 2 dígitos)

Motor: YOLOv8 nano (detecção de bbox) + EasyOCR (leitura dos caracteres).
Fallback: EasyOCR na imagem inteira se YOLO não detectar placa.
"""
import re
import threading
import cv2
import numpy as np
from typing import Any, Optional, Tuple
from dataclasses import dataclass

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO = None
    _YOLO_AVAILABLE = False

try:
    import easyocr as _easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _easyocr = None
    _EASYOCR_AVAILABLE = False


@dataclass
class PlateResult:
    """Resultado do reconhecimento da placa."""
    raw_text: str
    normalized: str
    format_type: str  # "old", "mercosul" ou "unknown"
    confidence: float
    roi: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)


# Padrões Brasil (texto normalizado, sem hífen):
OLD_PLATE_RE = re.compile(r"^[A-Z]{3}[0-9]{4}$")
MERCOSUL_PLATE_RE = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")

_OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
YOLO_PLATE_MODEL = "hf://keremberke/yolov8n-license-plate-detection/best.pt"
YOLO_CONFIDENCE = 0.45

# Singletons thread-safe
_plate_model: Optional[Any] = None
_ocr_reader: Optional[Any] = None
_yolo_lock = threading.Lock()
_ocr_lock = threading.Lock()


def _get_plate_model():
    global _plate_model
    if _plate_model is None:
        with _yolo_lock:
            if _plate_model is None:
                if not _YOLO_AVAILABLE:
                    return None
                try:
                    _plate_model = _YOLO(YOLO_PLATE_MODEL)
                except Exception:
                    _plate_model = None
    return _plate_model


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                if not _EASYOCR_AVAILABLE:
                    return None
                _ocr_reader = _easyocr.Reader(
                    ["en"],
                    gpu=False,
                    verbose=False,
                )
    return _ocr_reader


def _normalize_plate_text(text: str) -> str:
    """Remove caracteres não alfanuméricos e converte para maiúsculas."""
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def _classify_plate(normalized: str) -> str:
    if len(normalized) == 7 and MERCOSUL_PLATE_RE.match(normalized):
        return "mercosul"
    if len(normalized) == 7 and OLD_PLATE_RE.match(normalized):
        return "old"
    if len(normalized) == 7:
        return "mercosul"  # 7 chars sem padrão reconhecido → assume Mercosul
    return "unknown"


def _format_display(normalized: str, format_type: str) -> str:
    """Formata para exibição: antigo com hífen, Mercosul sem."""
    if format_type == "old" and len(normalized) >= 7:
        return f"{normalized[:3]}-{normalized[3:7]}"
    return normalized[:7] if len(normalized) >= 7 else normalized


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """CLAHE + binarização adaptativa para melhorar leitura dos caracteres."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh


def _ocr_on_image(img: np.ndarray) -> Optional[str]:
    """Executa EasyOCR em uma imagem (BGR ou grayscale) e retorna texto normalizado."""
    reader = _get_ocr_reader()
    if reader is None:
        return None
    results = reader.readtext(
        img,
        detail=0,
        allowlist=_OCR_ALLOWLIST,
        paragraph=False,
        batch_size=1,
    )
    if not results:
        return None
    # Junta todos os fragmentos detectados
    text = "".join(results)
    return _normalize_plate_text(text)


def _find_plate_contours(image: np.ndarray) -> list:
    """Fallback: detecção por contornos (proporção típica de placa 2:1 a 5.5:1)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 80 or h < 20:
            continue
        aspect = w / float(h) if h else 0
        if 2.0 <= aspect <= 5.5:
            candidates.append((x, y, w, h))
    return sorted(candidates, key=lambda r: r[2] * r[3], reverse=True)[:5]


def recognize_plate_from_image(image: np.ndarray) -> Optional[PlateResult]:
    """
    Reconhece placa em uma imagem (BGR).
    1. YOLO detecta bboxes de placas → EasyOCR lê os caracteres
    2. Fallback: contornos + EasyOCR se YOLO não detectar
    3. Fallback final: EasyOCR na imagem inteira
    """
    if image is None or image.size == 0:
        return None

    best: Optional[PlateResult] = None

    # --- Tentativa 1: YOLO ---
    yolo = _get_plate_model()
    if yolo is not None:
        try:
            yolo_results = yolo(image, conf=YOLO_CONFIDENCE, verbose=False)
            for r in yolo_results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    # Padding de 4px para não cortar bordas da placa
                    x1, y1 = max(0, x1 - 4), max(0, y1 - 4)
                    x2, y2 = min(image.shape[1], x2 + 4), min(image.shape[0], y2 + 4)
                    roi = image[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    proc = _preprocess_for_ocr(roi)
                    text = _ocr_on_image(proc)
                    if text and len(text) >= 6:
                        fmt = _classify_plate(text)
                        display = _format_display(text, fmt)
                        result = PlateResult(
                            raw_text=text,
                            normalized=display,
                            format_type=fmt,
                            confidence=conf,
                            roi=roi,
                            bbox=(x1, y1, x2 - x1, y2 - y1),
                        )
                        if best is None or conf > best.confidence:
                            best = result
        except Exception:
            pass  # YOLO falhou — cai no fallback de contornos

    # --- Tentativa 2: Contornos (fallback se YOLO não achou nada) ---
    if best is None:
        for (x, y, w, h) in _find_plate_contours(image):
            roi = image[y: y + h, x: x + w]
            proc = _preprocess_for_ocr(roi)
            text = _ocr_on_image(proc)
            if text and len(text) >= 6:
                fmt = _classify_plate(text)
                display = _format_display(text, fmt)
                result = PlateResult(
                    raw_text=text,
                    normalized=display,
                    format_type=fmt,
                    confidence=0.6,
                    roi=roi,
                    bbox=(x, y, w, h),
                )
                if best is None or len(text) >= len(best.raw_text):
                    best = result

    # --- Tentativa 3: Imagem inteira ---
    if best is None:
        proc = _preprocess_for_ocr(image)
        text = _ocr_on_image(proc)
        if text and len(text) >= 6:
            fmt = _classify_plate(text)
            display = _format_display(text, fmt)
            h_img, w_img = image.shape[:2]
            best = PlateResult(
                raw_text=text,
                normalized=display,
                format_type=fmt,
                confidence=0.4,
                roi=None,
                bbox=(0, 0, w_img, h_img),
            )

    return best


def capture_frame(camera_index: int = 0) -> Optional[np.ndarray]:
    """Captura um frame via câmera singleton (sem overhead de open/close)."""
    from app.camera import get_camera
    return get_camera(camera_index).read_frame()
