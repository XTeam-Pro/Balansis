"""Main FastAPI application for TNSIM."""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__ as TNSIM_VERSION
from ..core import get_global_cache, get_global_parallel_processor
from ..database.connection import close_database_connection
from ..database.repository import InfiniteSetRepository
from .routes import zerosum_router

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Initialization on startup
    logger.info("Starting TNSIM API...")

    # Initialize global components
    cache = get_global_cache()
    parallel_processor = get_global_parallel_processor()

    logger.info("Cache and parallel processor initialization completed")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down TNSIM API...")
    parallel_processor.shutdown()
    await close_database_connection()
    logger.info("TNSIM API stopped")


# Create FastAPI application
app = FastAPI(
    title="TNSIM API",
    description="API for Theory of Null Sum Infinite Sets",
    version=TNSIM_VERSION,
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify concrete domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(zerosum_router, prefix="/api/zerosum", tags=["zerosum"])


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "message": "TNSIM API - Theory of Null Sum Infinite Sets",
        "version": TNSIM_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Service health check."""
    try:
        repo = InfiniteSetRepository()
        db_status = await repo.health_check()
        if db_status.get("status") != "healthy":
            raise RuntimeError(db_status.get("error", "database health check failed"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
