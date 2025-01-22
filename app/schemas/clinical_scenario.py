from pydantic import BaseModel
from typing import List

class ClinicalScenario(BaseModel):
    gender: str
    age: str
    ethinicity: str

class StructuredInfo(BaseModel):
    questions:  list[ClinicalScenario]