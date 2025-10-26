"""Routes for triggering and monitoring clinical analysis jobs (Redis + RQ)."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from redis import Redis
from rq import Queue
from rq.job import Job

from app.core.config import settings
from app.schemas.clinical_analysis import (
    AnalyzeRequest,
    JobFinishedResponse,
    JobRunningResponse,
    JobTriggerResponse,
)
from app.schemas.generate import JobFailedResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Redis / RQ setup
redis_conn: Redis = Redis.from_url(settings.REDIS_URL)
q: Queue = Queue(name="analyze", connection=redis_conn)


@router.post(
    "/analyze-clinical/trigger",
    summary="Trigger clinical question analysis",
    response_model=JobTriggerResponse,
)
def trigger_analysis(req: AnalyzeRequest) -> JobTriggerResponse:
    """Enqueue a background clinical analysis job.

    The job is pushed to Redis/RQ and processed by a worker. The response
    includes a `job_id` that can be polled via the status endpoint.

    Args:
        req: Validated request payload containing the text and model.

    Returns:
        JobTriggerResponse: Job identifier and enqueued flag.

    Raises:
        HTTPException: If the job could not be enqueued.
    """
    try:
        # pylint: disable=import-outside-toplevel
        from app.tasks.task_clinical_analysis import analyze_clinical_questions

        job: Job = q.enqueue(
            analyze_clinical_questions,
            args=(req.model_dump(),),
            job_timeout=1200,  # seconds
            result_ttl=3600,  # keep results 1h
            failure_ttl=24 * 3600,  # keep failure info 24h
        )
        return JobTriggerResponse(job_id=job.id, enqueued=True)
    except Exception as exc:  # broad-except: surface a 500 and log
        logger.exception("Failed to enqueue clinical analysis job: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to enqueue job.") from exc


@router.get(
    "/analyze-clinical/status/{job_id}",
    summary="Check clinical analysis job status / retrieve result",
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
    """Retrieve status or result for a given job.

    Depending on the RQ job state, this returns:
      * `{'status': 'queued' | 'started'}` while running,
      * `{'status': 'finished', 'result': <TaskResult>}` when complete,
      * `{'status': 'failed', 'error': <message>}` on failure.

    Args:
        job_id: The Redis/RQ job identifier.

    Returns:
        A status dictionary matching one of the schemas declared in the route.

    Raises:
        HTTPException: If the job cannot be found.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception as exc:  # job not found or backend error
        logger.warning("Job fetch failed for id=%s: %s", job_id, exc)
        raise HTTPException(status_code=404, detail="Job not found.") from exc

    status = job.get_status()

    if status == "finished":
        return {"status": "finished", "result": job.result}
    if status == "failed":
        return {"status": "failed", "error": "Task failed. Check worker logs."}

    # Still running (queued/started/deferred etc.). We surface the raw status
    # to keep behavior consistent with the client-side union schema.
    return {"status": status}
