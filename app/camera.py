"""
CameraRegistry: gerencia múltiplas fontes de vídeo configuradas via JSON.

Suporta:
  - Câmeras USB/V4L2: source = 0, 1, 2, 3  (inteiro → /dev/video0...)
  - RTSP: source = "rtsp://user:senha@ip:554/Streaming/Channels/101"
  - RTMP: source = "rtmp://server:1935/live/chave"
  - MJPEG HTTP: source = "http://ip:8080/video"

Configuração em data/cameras.json (montado como volume Docker).
Editar o arquivo + chamar POST /cameras/reload não exige restart.
"""
import json
import threading
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

# Após _MAX_FAILS leituras falhas consecutivas, espera _RETRY_DELAY antes de tentar reabrir.
_MAX_FAILS = 5
_RETRY_DELAY = 15.0  # segundos


class CameraEntry:
    """Uma fonte de vídeo individual — câmera local ou stream de rede."""

    def __init__(self, camera_id: str, name: str, source: Union[int, str], enabled: bool = True):
        self.id = camera_id
        self.name = name
        self.source = source
        self.enabled = enabled
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._fail_count = 0
        self._last_fail = 0.0

    def _should_retry(self) -> bool:
        if self._fail_count < _MAX_FAILS:
            return True
        return (time.monotonic() - self._last_fail) > _RETRY_DELAY

    def _open(self) -> bool:
        """Abre o VideoCapture. Câmeras de rede usam backend CAP_FFMPEG explicitamente."""
        try:
            if isinstance(self.source, int):
                cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG)
            if cap.isOpened():
                # Buffer de 1 frame: sempre entrega o quadro mais recente
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._cap = cap
                self._fail_count = 0
                return True
            cap.release()
        except Exception:
            pass
        return False

    def read_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self.enabled:
                return None
            if (self._cap is None or not self._cap.isOpened()) and not self._should_retry():
                return None
            if self._cap is None or not self._cap.isOpened():
                if not self._open():
                    self._fail_count += 1
                    self._last_fail = time.monotonic()
                    return None
            ret, frame = self._cap.read()
            if not ret:
                self._cap.release()
                self._cap = None
                self._fail_count += 1
                self._last_fail = time.monotonic()
                return None
            self._fail_count = 0
            return frame

    def status(self) -> str:
        with self._lock:
            if not self.enabled:
                return "disabled"
            if self._cap is not None and self._cap.isOpened() and self._fail_count < _MAX_FAILS:
                return "online"
            if self._fail_count >= _MAX_FAILS and (time.monotonic() - self._last_fail) < _RETRY_DELAY:
                return "reconnecting"
            return "offline"

    def release(self) -> None:
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None


# Template gerado automaticamente na primeira execução
_DEFAULT_CONFIG = [
    {"id": "cam0", "name": "Câmera 0 – USB/V4L2 (/dev/video0)", "source": 0, "enabled": True},
    {"id": "cam1", "name": "Câmera 1 – USB/V4L2 (/dev/video1)", "source": 1, "enabled": False},
    {"id": "cam2", "name": "Câmera 2 – USB/V4L2 (/dev/video2)", "source": 2, "enabled": False},
    {"id": "cam3", "name": "Câmera 3 – USB/V4L2 (/dev/video3)", "source": 3, "enabled": False},
    {
        "id": "rtsp1",
        "name": "Câmera IP – RTSP (exemplo)",
        "source": "rtsp://admin:senha@192.168.1.100:554/Streaming/Channels/101",
        "enabled": False,
    },
    {
        "id": "rtsp2",
        "name": "Câmera IP – RTSP 2 (exemplo)",
        "source": "rtsp://admin:senha@192.168.1.101:554/stream1",
        "enabled": False,
    },
    {
        "id": "rtmp1",
        "name": "Stream RTMP (exemplo)",
        "source": "rtmp://192.168.1.200:1935/live/stream1",
        "enabled": False,
    },
]


class CameraRegistry:
    """
    Registro central de câmeras. Carregado de data/cameras.json.
    Suporta reload em quente (POST /cameras/reload) sem restart do servidor.
    """

    def __init__(self, config_path: str = "data/cameras.json"):
        self._config_path = Path(config_path)
        self._cameras: Dict[str, CameraEntry] = {}
        self._lock = threading.Lock()
        self._ensure_default_config()
        self._load()

    def _ensure_default_config(self) -> None:
        if not self._config_path.exists():
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(
                json.dumps(_DEFAULT_CONFIG, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def _load(self) -> None:
        """Lê cameras.json e cria/atualiza entradas. Thread-safe."""
        with self._lock:
            try:
                data: list = json.loads(self._config_path.read_text(encoding="utf-8"))
            except Exception:
                data = []

            new_ids = {c.get("id", "") for c in data if c.get("id")}

            # Remove câmeras que saíram do config
            for cid in list(self._cameras):
                if cid not in new_ids:
                    self._cameras[cid].release()
                    del self._cameras[cid]

            for cfg in data:
                cid = cfg.get("id", "")
                if not cid:
                    continue
                raw = cfg.get("source", 0)
                source: Union[int, str] = int(raw) if isinstance(raw, (int, float)) else str(raw)
                name = cfg.get("name", cid)
                enabled = bool(cfg.get("enabled", True))

                if cid in self._cameras:
                    entry = self._cameras[cid]
                    if entry.source != source:
                        entry.release()
                        entry.source = source
                        entry._fail_count = 0
                    entry.name = name
                    entry.enabled = enabled
                else:
                    self._cameras[cid] = CameraEntry(cid, name, source, enabled)

    def reload(self) -> None:
        """Recarrega cameras.json em quente. Chame após editar o arquivo."""
        self._load()

    def read_frame(self, camera_id: str) -> Optional[np.ndarray]:
        entry = self._cameras.get(camera_id)
        return entry.read_frame() if entry else None

    def list_cameras(self) -> List[Dict]:
        with self._lock:
            return [
                {
                    "id": e.id,
                    "name": e.name,
                    "source": str(e.source),
                    "enabled": e.enabled,
                    "status": e.status(),
                }
                for e in self._cameras.values()
            ]

    def is_available(self, camera_id: str) -> bool:
        entry = self._cameras.get(camera_id)
        return entry is not None and entry.status() == "online"

    def release_all(self) -> None:
        with self._lock:
            for e in self._cameras.values():
                e.release()


_registry: Optional[CameraRegistry] = None
_registry_lock = threading.Lock()


def get_registry(config_path: str = "data/cameras.json") -> CameraRegistry:
    """Singleton global do CameraRegistry."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = CameraRegistry(config_path)
    return _registry


def get_camera(index: int = 0):
    """
    Compatibilidade retroativa com código que chama get_camera(index).
    Mapeia index → camera_id 'cam{index}' no registry.
    """
    registry = get_registry()
    camera_id = f"cam{index}"

    class _Compat:
        def read_frame(self):
            return registry.read_frame(camera_id)

        def is_available(self):
            return registry.is_available(camera_id)

        def release(self):
            pass  # gerenciado pelo registry; use release_all() no shutdown

    return _Compat()
