from pydantic import BaseModel
from typing import List

class ClinicalScenario(BaseModel):
    gender: str
    ethnicity: str

class StructuredInfo(BaseModel):
    questions:  list[ClinicalScenario]