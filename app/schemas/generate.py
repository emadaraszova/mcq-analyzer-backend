"""Pydantic schemas for question generation and demographic configuration."""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request schema for triggering a question generation job.

    Attributes:
        message: User prompt or input text for question generation.
        model: Model name, e.g., 'gpt-4o' or 'gemini-2.5-flash'.
        number_of_questions: Target number of questions to generate (1–1000).
        demographicData: Optional demographic configuration for tailoring generated questions.
    """

    message: str = Field(..., min_length=1, description="User prompt to process.")
    model: str = Field(
        ...,
        min_length=1,
        description="Model name, e.g., 'gpt-4o' or 'gemini-2.5-flash'.",
    )
    number_of_questions: Optional[int] = Field(
        None, ge=1, le=1000, description="Target number of questions."
    )
    demographicData: Optional["DemographicData"] = Field(
        None, description="Optional demographic specification for question generation."
    )


class TaskResult(BaseModel):
    """Result payload returned by the question generation worker.

    Attributes:
        status: Job status (always 'completed' when finished).
        session_id: Unique identifier for the generation session.
        model: Model used to generate the questions.
        message: Original input message.
        response: Model-generated question set or formatted text.
    """

    status: Literal["completed"]
    session_id: str
    model: str
    message: str
    response: str


class JobTriggerResponse(BaseModel):
    """Response schema returned when a generation job is enqueued.

    Attributes:
        job_id: Unique job identifier.
        enqueued: Indicates if the job was successfully added to the queue.
    """

    job_id: str
    enqueued: bool


class JobRunningResponse(BaseModel):
    """Response schema for an active or queued generation job.

    Attributes:
        status: Job status, either 'queued' or 'started'.
    """

    status: Literal["queued", "started"]


class JobFinishedResponse(BaseModel):
    """Response schema for a successfully finished generation job.

    Attributes:
        status: Job status (always 'finished').
        result: Completed task result from the worker.
    """

    status: Literal["finished"]
    result: TaskResult


class JobFailedResponse(BaseModel):
    """Response schema for a failed generation job.

    Attributes:
        status: Job status (always 'failed').
        error: Optional error message indicating the cause of failure.
    """

    status: Literal["failed"]
    error: Optional[str] = None


class DistributionRow(BaseModel):
    """Represents a single demographic label-count pair.

    Attributes:
        label: Category label, e.g., 'Female' or '30–40'.
        value: Count of items belonging to that category.
    """

    label: str
    value: int


DemographicCategory = Literal["Sex", "Age", "Ethnicity"]


class DemographicData(BaseModel):
    """Represents demographic distributions for generated questions.

    Attributes:
        Sex: List of sex-based distribution rows.
        Ethnicity: List of ethnicity-based distribution rows.
        Age: List of age-based distribution rows.
    """

    Sex: List[DistributionRow] = Field(default_factory=list)
    Ethnicity: List[DistributionRow] = Field(default_factory=list)
    Age: List[DistributionRow] = Field(default_factory=list)
