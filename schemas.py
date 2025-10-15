from pydantic import BaseModel
from typing import List

class Recommendation(BaseModel):
    product_id: str
    product_name: str
    score: float

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Recommendation]
    source: str

class ExplanationResponse(BaseModel):
    user_id: str
    product_id: str
    explanation: str
