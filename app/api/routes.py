from fastapi import APIRouter, HTTPException

from app.schemas.prediction_schema import (
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import (
    prediction_service,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"],
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_news(
    request: PredictionRequest,
) -> PredictionResponse:

    try:
        result = prediction_service.predict(
            request.text
        )

        confidence = result["confidence"]

        if confidence >= 80:
            confidence_level = "High"
        elif confidence >= 60:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        return PredictionResponse(
            prediction=result["prediction"],
            label=result["label"],
            confidence=confidence,
            confidence_level=confidence_level,
            disclaimer=(
                "This is an AI-based prediction based on "
                "textual patterns and should not be treated "
                "as absolute truth or factual verification."
            ),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc