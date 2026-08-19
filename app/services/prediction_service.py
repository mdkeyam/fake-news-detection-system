from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    BASE_DIR
    / "ml"
    / "artifacts"
    / "fake_news_model.pkl"
)

VECTORIZER_FILE = (
    BASE_DIR
    / "ml"
    / "artifacts"
    / "tfidf_vectorizer.pkl"
)


class PredictionService:
    """
    Service responsible for loading the trained ML artifacts
    and generating fake/genuine news predictions.
    """

    def __init__(self) -> None:
        self.model = None
        self.vectorizer = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_FILE}"
            )

        if not VECTORIZER_FILE.exists():
            raise FileNotFoundError(
                f"Vectorizer file not found: {VECTORIZER_FILE}"
            )

        self.model = joblib.load(MODEL_FILE)
        self.vectorizer = joblib.load(VECTORIZER_FILE)

    def predict(self, text: str) -> dict:
        """
        Predict whether the supplied news text is likely fake
        or likely genuine.
        """

        if not text or not text.strip():
            raise ValueError(
                "News text cannot be empty."
            )

        cleaned_text = text.strip()

        # Convert text into TF-IDF features
        text_vector = self.vectorizer.transform(
            [cleaned_text]
        )

        # Generate prediction
        prediction = int(
            self.model.predict(text_vector)[0]
        )

        # Get model probability
        probabilities = self.model.predict_proba(
            text_vector
        )[0]

        confidence = float(
            max(probabilities)
        )

        if prediction == 0:
            label = "Likely Fake"
        else:
            label = "Likely Genuine"

        return {
            "prediction": label,
            "label": prediction,
            "confidence": round(
                confidence * 100,
                2,
            ),
        }


prediction_service = PredictionService()