from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.routes.detect import router as detect_router
from api.routes.live_test import router as live_test_router
from api.routes.stream import router as stream_router
from deepfake_detector import Detector

_STATIC_DIR = Path(__file__).parent / "static"


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
app.include_router(live_test_router)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/static/index.html")
