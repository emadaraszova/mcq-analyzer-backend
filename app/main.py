"""FastAPI application entry point.

This module configures logging, sets up CORS, registers routers, and defines
concise exception handlers that return JSON responses. Logs are emitted to
stdout so you can see them in the terminal (or container logs).
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.endpoints.clinical_analysis import (
    router as analyze_clinical_router,
)
from app.api.endpoints.generate import router as generate_router

app = FastAPI(
    title="My App",
    description="API for Clinical and AI Responses",
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a 422 response with validation details.

    Args:
        request: The incoming HTTP request (unused, for FastAPI signature).
        exc: The raised validation error.

    Returns:
        A JSON response with the error details and a 422 status code.
    """
    print("Validation error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected exceptions.

    Args:
        request: The incoming HTTP request (unused, for FastAPI signature).
        exc: The unhandled exception.

    Returns:
        A JSON response with a generic message and the error string. The
        exception is logged with a stack trace.
    """
    print("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred.",
            "error": str(exc),
        },
    )


app.include_router(
    analyze_clinical_router,
    prefix="/api",
    tags=["Clinical Analysis"],
)

app.include_router(
    generate_router,
    prefix="/api",
    tags=["Generation"],
)
