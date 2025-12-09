"""Pydantic schemas for the clinical analysis API endpoints."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request schema for triggering an analysis job.

    Attributes:
        message: User-provided text (questions or content) to analyze.
        model: Name of the model to use, e.g., 'gpt-4o' or 'gemini-2.5-flash'.
    """

    message: str = Field(..., min_length=1, description="Questions to process.")
    model: str = Field(
        ...,
        min_length=1,
        description="Model name, e.g., 'gpt-4o' or 'gemini-2.5-flash'.",
    )


class TaskResult(BaseModel):
    """Result payload returned by the analysis worker.

    Attributes:
        status: Processing status (always 'completed' for a finished job).
        session_id: Unique session identifier.
        model: Model name used for processing.
        message: Original request message or job description.
        response: The model-generated analysis text or structured output.
    """

    status: Literal["completed"]
    session_id: str
    model: str
    message: str
    response: str


class JobTriggerResponse(BaseModel):
    """Response schema returned when a new analysis job is enqueued.

    Attributes:
        job_id: Unique identifier assigned to the job.
        enqueued: Indicates whether the job was successfully added to the queue.
    """

    job_id: str
    enqueued: bool


class JobRunningResponse(BaseModel):
    """Response schema for a job that is still running or queued.

    Attributes:
        status: Job status, either 'queued' or 'started'.
    """

    status: Literal["queued", "started"]


class JobFinishedResponse(BaseModel):
    """Response schema for a successfully finished job.

    Attributes:
        status: Job status (always 'finished').
        result: The completed task result.
    """

    status: Literal["finished"]
    result: TaskResult


class JobFailedResponse(BaseModel):
    """Response schema for a failed job.

    Attributes:
        status: Job status (always 'failed').
        error: Optional error message explaining the failure reason.
    """

    status: Literal["failed"]
    error: Optional[str] = None


class ClinicalScenario(BaseModel):
    """Structured representation of a single clinical scenario.

    Attributes:
        sex: The patient's sex (e.g., 'Male', 'Female', 'Other').
        ethnicity: The patient's ethnicity.
        age: The patient's age in years.
    """

    sex: str
    ethnicity: str
    age: int


class StructuredInfo(BaseModel):
    """Aggregated clinical information extracted from analysis.

    Attributes:
        questions: A list of structured clinical scenarios extracted
            from the analyzed text, each represented by a
            :class:`ClinicalScenario` instance.
    """

    questions: list[ClinicalScenario]
