from pydantic import BaseModel


class UserMessage(BaseModel):
    session_id: str
    message: str
    model: str

class UserMessageAnalysis(BaseModel):
    session_id: str
    message: str
    model: str
    number_of_questions: str


