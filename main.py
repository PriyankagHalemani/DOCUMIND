import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.db.client import db_manager
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("docuchat")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure upload/data folders and initialize database
    settings.ensure_directories()
    await db_manager.initialize()
    logger.info("DocuChat AI backend initialized with AI_PROVIDER=%s", settings.AI_PROVIDER)
    yield
    # Shutdown
    if db_manager.client:
        db_manager.client.close()
        logger.info("MongoDB connection closed.")

app = FastAPI(
    title="DocuChat AI — Intelligent PDF Question Answering System",
    description="Full-stack AI-powered PDF Question Answering application using RAG architecture.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for development and frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)

# Mount frontend dist static files if built
frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if (frontend_dist / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Global unhandled exception on %s: %s", request.url.path, str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on the server. Please try again."}
    )

@app.get("/")
async def serve_index():
    index_html = frontend_dist / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return {
        "app": "DocuChat AI Backend",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/api/health"
    }

# SPA fallback
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    # Skip api routes
    if full_path.startswith("api/"):
        return JSONResponse(status_code=404, content={"detail": "API route not found"})
    
    file_path = frontend_dist / full_path
    if file_path.is_file():
        return FileResponse(file_path)
    
    index_html = frontend_dist / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return JSONResponse(status_code=404, content={"detail": "Not found"})
