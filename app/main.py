from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.api.endpoints.generate_response import router as response_router
from app.api.endpoints.analyze_clinical import router as analyze_clinical_router


app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register Exception Handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """
    Custom exception handler for request validation errors.
    """
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors(), "body": exc.body},
    )


# Include Routers
app.include_router(response_router, prefix="/api", tags=["Responses"])
app.include_router(analyze_clinical_router, prefix="/api", tags=["Clinical Analysis"])

