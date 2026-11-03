import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import ai, chat, conversion, documents, pdf_tools
from .core import config
from .core.database import init_db
from .services.embedding_service import start_loading
from .services.errors import ToolError

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    start_loading()  # load the embedding model in the background so the first question isn't slow
    yield


app = FastAPI(title="SmartPDF AI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ToolError)
async def tool_error_handler(_: Request, exc: ToolError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


for module in (documents, pdf_tools, conversion, ai, chat):
    app.include_router(module.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"ok": True}


# Serve the built React app when it exists (single-command demo: one server, one port).
if config.FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
