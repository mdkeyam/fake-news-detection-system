from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_dataset.csv"
)

ARTIFACTS_DIR = BASE_DIR / "artifacts"

VECTORIZER_FILE = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
MODEL_FILE = ARTIFACTS_DIR / "fake_news_model.pkl"


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def split_dataset(df: pd.DataFrame):
    X = df["model_text"]
    y = df["label"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


def create_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def create_model() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
    )


def main() -> None:

    print("=" * 60)
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df = load_dataset()

    print(f"Total samples: {len(df)}")

    # --------------------------------------------------
    # 2. Train/Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = split_dataset(df)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # --------------------------------------------------
    # 3. TF-IDF
    # --------------------------------------------------

    print("\nCreating TF-IDF vectorizer...")

    vectorizer = create_vectorizer()

    print("Fitting TF-IDF on training data...")

    X_train_tfidf = vectorizer.fit_transform(X_train)

    print("Transforming test data...")

    X_test_tfidf = vectorizer.transform(X_test)

    print(
        f"Training matrix: {X_train_tfidf.shape}"
    )

    print(
        f"Testing matrix: {X_test_tfidf.shape}"
    )

    # --------------------------------------------------
    # 4. Model
    # --------------------------------------------------

    print("\nCreating Logistic Regression model...")

    model = create_model()

    # --------------------------------------------------
    # 5. Training
    # --------------------------------------------------

    print("Training model...")

    model.fit(
        X_train_tfidf,
        y_train,
    )

    print("Model training completed.")

    # --------------------------------------------------
    # 6. Save artifacts
    # --------------------------------------------------

    print("\nSaving model artifacts...")

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print("\nArtifacts saved successfully:")

    print(f"Vectorizer: {VECTORIZER_FILE}")
    print(f"Model:      {MODEL_FILE}")


if __name__ == "__main__":
    main()