from pydantic import BaseModel
from typing import List

class ClinicalScenario(BaseModel):
    gender: str
    ethinicity: str

class StructuredInfo(BaseModel):
    questions:  list[ClinicalScenario]