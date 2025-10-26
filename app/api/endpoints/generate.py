"""Routes for triggering and monitoring text generation tasks (Redis + RQ)."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from redis import Redis
from rq import Queue
from rq.job import Job

from app.core.config import settings
from app.schemas.generate import (
    GenerateRequest,
    JobTriggerResponse,
    JobRunningResponse,
    JobFinishedResponse,
    JobFailedResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Redis / RQ setup
redis_conn: Redis = Redis.from_url(settings.REDIS_URL)
q: Queue = Queue(name="generate", connection=redis_conn)


@router.post(
    "/generate-response/trigger",
    summary="Trigger long-running text generation task",
    response_model=JobTriggerResponse,
)
def trigger_generation(req: GenerateRequest) -> JobTriggerResponse:
    """Enqueue a background text generation job using Redis/RQ.

    The job runs asynchronously via an RQ worker. The response
    includes a unique `job_id` that can be polled for completion status.

    Args:
        req: Validated request payload containing the input text and model parameters.

    Returns:
        JobTriggerResponse: Object containing job_id and enqueued flag.

    Raises:
        HTTPException: If the job cannot be enqueued successfully.
    """
    try:
        # Lazy import to avoid circular dependency on the worker
        from app.tasks.task_generate_texts import (
            generate_texts,
        )  # pylint: disable=import-outside-toplevel

        job: Job = q.enqueue(
            generate_texts,
            args=(req.model_dump(),),
            job_timeout=1200,  # 20 min max execution time
            result_ttl=3600,  # Keep results for 1 hour
            failure_ttl=24 * 3600,  # Keep failure info for 24 hours
        )
        logger.info("Enqueued text generation job %s", job.id)
        return JobTriggerResponse(job_id=job.id, enqueued=True)

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to enqueue text generation job: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to enqueue job.") from exc


@router.get(
    "/generate-response/status/{job_id}",
    summary="Check job status / retrieve result",
    responses={
        200: {
            "description": "Status payload. Could be running, finished, or failed.",
            "content": {
                "application/json": {
                    "schema": {
                        "oneOf": [
                            JobRunningResponse.model_json_schema(),
                            JobFinishedResponse.model_json_schema(),
                            JobFailedResponse.model_json_schema(),
                        ]
                    }
                }
            },
        }
    },
)
def check_status(job_id: str) -> Dict[str, Any]:
    """Retrieve the status or final result of a text generation job.

    Depending on job progress, returns:
      * `{'status': 'queued' | 'started'}` while running
      * `{'status': 'finished', 'result': <TaskResult>}` when complete
      * `{'status': 'failed', 'error': <message>}` if it failed

    Args:
        job_id: Unique Redis/RQ job identifier.

    Returns:
        A dictionary containing job status and (optionally) the result or error.

    Raises:
        HTTPException: If no job is found with the given ID.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Job fetch failed for id=%s: %s", job_id, exc)
        raise HTTPException(status_code=404, detail="Job not found.") from exc

    status = job.get_status()

    if status == "finished":
        return {"status": "finished", "result": job.result}
    if status == "failed":
        return {"status": "failed", "error": "Task failed. Check worker logs."}

    # Still running
    return {"status": status}
