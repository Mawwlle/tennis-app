from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from webapp.routes import router

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="Tennis Annotation Tool")
    app.include_router(router)
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app
