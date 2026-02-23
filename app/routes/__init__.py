from .plate import router as plate_router
from .face import router as face_router
from .persons import router as persons_router
from .vehicles import router as vehicles_router
from .authorizations import router as authorizations_router
from .access import router as access_router

__all__ = [
    "plate_router",
    "face_router",
    "persons_router",
    "vehicles_router",
    "authorizations_router",
    "access_router",
]
