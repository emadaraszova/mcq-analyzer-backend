from typing import Literal, Optional
from pydantic import BaseModel, Field

# Request schema for the trigger endpoint
class GenerateRequest(BaseModel):
    # Non-empty strings for validation
    message: str = Field(..., min_length=1, description="User prompt to process.")
    model: str = Field(..., min_length=1, description="Model name, e.g., 'gpt-4o' or 'gemini-1.5-flash'.")
    number_of_questions: int = Field(..., ge=1, le=1000, description="Target number of questions.")
    
# Response schema returned by the trigger endpoint
class JobTriggerResponse(BaseModel):
    job_id: str
    enqueued: bool

# Response schema returned by the status endpoint while running
class JobRunningResponse(BaseModel):
    status: Literal["queued", "started"]

# Result payload produced by the worker
class TaskResult(BaseModel):
    status: Literal["completed"]
    session_id: str
    model: str
    message: str
    response: str

# Response schema when the job is finished
class JobFinishedResponse(BaseModel):
    status: Literal["finished"]
    result: TaskResult

# Response schema when the job failed
class JobFailedResponse(BaseModel):
    status: Literal["failed"]
    error: Optional[str] = None
