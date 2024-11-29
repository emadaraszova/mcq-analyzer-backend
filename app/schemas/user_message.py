from pydantic import BaseModel

class UserMessage(BaseModel):
    session_id: str
    message: str
    model: str
