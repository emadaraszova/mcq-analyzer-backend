from typing_extensions import TypedDict

class ClinicalScenario(TypedDict):
    gender: str
    age: str
    symptoms: str
    family_background: str

class StructuredInfo(TypedDict):
    questions: list[ClinicalScenario]