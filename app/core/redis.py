import os
import redis
from rq import Queue

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = redis.Redis.from_url(REDIS_URL, decode_responses=True)
queue = Queue("default", connection=redis_conn)

def k_session(session_id: str) -> str:
    return f"session:{session_id}:messages"

def k_job_result(job_id: str) -> str:
    return f"job:{job_id}:result"