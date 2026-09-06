"""
Singleton de câmera persistente — mantém cv2.VideoCapture aberto entre requests.
Elimina o overhead de open/close por request (~100-300ms) e a necessidade de warmup.
"""
import threading
import cv2
import numpy as np
from typing import Optional


class CameraManager:
    def __init__(self, index: int = 0):
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._index = index

    def _ensure_open(self) -> bool:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._index)
            if self._cap.isOpened():
                # Mantém buffer de 1 frame para sempre ter o mais recente
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> Optional[np.ndarray]:
        """Retorna o frame mais recente ou None se a câmera não estiver disponível."""
        with self._lock:
            if not self._ensure_open():
                return None
            ret, frame = self._cap.read()
            return frame if ret else None

    def is_available(self) -> bool:
        with self._lock:
            return self._ensure_open()

    def release(self) -> None:
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None


_manager: Optional[CameraManager] = None
_manager_lock = threading.Lock()


def get_camera(index: int = 0) -> CameraManager:
    """Retorna o singleton CameraManager para o índice informado."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = CameraManager(index)
    return _manager
