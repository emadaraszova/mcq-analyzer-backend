from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.api.endpoints.responses import router as response_router
from app.api.endpoints.clinical_analysis import router as analyze_clinical_router

# Initialize FastAPI app
app = FastAPI(title="My App", description="API for Clinical and AI Responses", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to specific domains in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Validation Error Handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom exception handler for request validation errors.
    Provides a more descriptive response for validation issues.
    """
    return JSONResponse(
        status_code=422,  # Standard status code for validation errors
        content={"detail": exc.errors()},
    )

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for uncaught exceptions.
    Ensures consistent error responses across the application.
    """
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error": str(exc)},
    )

# Include API Routers
app.include_router(response_router, prefix="/api", tags=["Responses"])
app.include_router(analyze_clinical_router, prefix="/api", tags=["Clinical Analysis"])
