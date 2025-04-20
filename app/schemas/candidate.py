from pydantic import BaseModel

class CandidateInput(BaseModel):
    experience_years: float
    technical_score: float 