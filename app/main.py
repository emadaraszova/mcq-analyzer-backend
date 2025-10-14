# app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.endpoints.clinical_analysis import router as analyze_clinical_router
from app.api.endpoints.generate import router as generate_router 

app = FastAPI(title="My App", description="API for Clinical and AI Responses", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error": str(exc)},
    )

app.include_router(analyze_clinical_router, prefix="/api", tags=["Clinical Analysis"])
app.include_router(generate_router,         prefix="/api", tags=["Generation"]) 
