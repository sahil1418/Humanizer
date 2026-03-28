"""
FastAPI Application Entry  (Section 16)
────────────────────────────────────────
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from humanizer import __version__
from humanizer.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

app = FastAPI(
    title="Humanizer v2.0",
    description=(
        "Transformer-based text rewriting system — Humanizer Edition. "
        "Rewrites text to be semantically faithful, structurally original, "
        "and undetectable as AI-generated."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permissive for development; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["rewrite"])


@app.get("/")
async def root():
    return {
        "service": "Humanizer",
        "version": __version__,
        "docs": "/docs",
    }
