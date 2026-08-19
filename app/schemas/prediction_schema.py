from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="News headline or article text",
    )


class PredictionResponse(BaseModel):
    prediction: str
    label: int
    confidence: float
    confidence_level: str
    disclaimer: str