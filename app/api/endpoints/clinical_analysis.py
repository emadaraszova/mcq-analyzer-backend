from fastapi import APIRouter, HTTPException
from redis import Redis
from rq import Queue
from app.core.config import settings
from app.schemas.clinical_analysis import GenerateRequest, JobFinishedResponse, JobRunningResponse, JobTriggerResponse
from app.schemas.generate import JobFailedResponse
from rq.job import Job

router = APIRouter()

# Redis / RQ setup  
redis_conn = Redis.from_url(settings.REDIS_URL)
q = Queue(name="analyze", connection=redis_conn)


@router.post("/analyze-clinical/trigger",
            summary="Trigger clinical question analysis",
            response_model=JobTriggerResponse
)
def trigger_analysis(req: GenerateRequest):
    """
    Enqueue a background job (Redis+RQ) to analyze questions.
    Returns a job_id to poll via the status endpoint.
    """
    try:
        from app.tasks.task_clinical_analysis import analyze_clinical_questions
        print((req.model_dump(),))
        job = q.enqueue(
        analyze_clinical_questions,
        args=(req.model_dump(),),
        ob_timeout=1200,
        result_ttl=3600,
        failure_ttl=24*3600,
    )
        return {"job_id": job.id, "enqueued": True}
    except Exception as e:
        print(f"[enqueue error] {e}")
        raise HTTPException(status_code=500, detail="Failed to enqueue job.")

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
def check_status(job_id: str):
    """
    Poll a job by ID.
    Returns:
     - queued / started (running)
     - finished with result
     - failed with an error message
     """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found.")
    
    status = job.get_status()
    if status == "finished":
        return {"status": "finished", "result": job.result}
    if status == "failed":
        return {"status": "failed", "error": "Task failed. Check worker logs."}

    # still running
    return {"status": status}