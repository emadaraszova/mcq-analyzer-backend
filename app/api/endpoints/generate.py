from fastapi import APIRouter, HTTPException
from redis import Redis
from rq import Queue
from app.core.config import settings
from rq.job import Job

from app.schemas.generate import (
    GenerateRequest,
    JobTriggerResponse,
    JobRunningResponse,
    JobFinishedResponse,
    JobFailedResponse,
)

router = APIRouter()

# Redis / RQ setup
redis_conn = Redis.from_url(settings.REDIS_URL)
q = Queue(connection=redis_conn)


@router.post(
    "/generate-response/trigger",
    summary="Trigger long-running text generation task",
    response_model=JobTriggerResponse,  
)
def trigger_generation(req: GenerateRequest):
    """
    Enqueue a background job (Redis+RQ) to generate AI text.
    Returns a job_id to poll via the status endpoint.
    """
    try:
        from app.tasks.task_generate_texts import generate_texts
        print((req.model_dump(),))
        job = q.enqueue(
        generate_texts,
        args=(req.model_dump(),),
        job_timeout=1200,
        result_ttl=3600,
        failure_ttl=24*3600,           
    )
        return {"job_id": job.id, "enqueued": True}
    except Exception as e:
        print(f"[enqueue error] {e}")
        raise HTTPException(status_code=500, detail="Failed to enqueue job.")


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
def check_status(job_id: str):
    """
    Poll a job by ID.
    Returns:
      - queued / started (running)
      - finished with the task result
      - failed with an error message
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found.")

    status = job.get_status()  # 'queued' | 'started' | 'finished' | 'failed' | etc.

    if status == "finished":
        return {"status": "finished", "result": job.result}
    if status == "failed":
        return {"status": "failed", "error": "Task failed. Check worker logs."}

    # still running
    return {"status": status}
