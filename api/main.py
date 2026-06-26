from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes.detect import router as detect_router
from api.routes.stream import router as stream_router
from deepfake_detector import Detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector = Detector()
    warnings: list[str] = []
    warmed_up = False
    try:
        detector.warmup()
        warmed_up = True
    except Exception as exc:
        warnings.append(str(exc))
    finally:
        detector.close()

    app.state.detector_status = {"warmed_up": warmed_up, "warnings": warnings}
    yield


app = FastAPI(
    title="deepfake-detector",
    version="0.1.0",
    description="MVP deepfake detection API with CNN, liveness, and temporal layers",
    lifespan=lifespan,
)
app.include_router(detect_router)
app.include_router(stream_router)
