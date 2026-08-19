from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


DATA_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "processed"
    / "cleaned_dataset.csv"
)


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


def main() -> None:

    print("=" * 60)
    print("FAKE NEWS DETECTION - MODEL EVALUATION")
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

    # --------------------------------------------------
    # 4. Train model
    # --------------------------------------------------

    print("\nTraining Logistic Regression model...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        X_train_tfidf,
        y_train,
    )

    print("Training completed.")

    # --------------------------------------------------
    # 5. Predictions
    # --------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(X_test_tfidf)

    # --------------------------------------------------
    # 6. Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=1,
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1,
    )

    # --------------------------------------------------
    # 7. Print metrics
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # --------------------------------------------------
    # 8. Classification Report
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Fake",
                "Genuine",
            ],
        )
    )

    # --------------------------------------------------
    # 9. Confusion Matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(cm)

    print("\nMatrix interpretation:")
    print("Rows    = Actual")
    print("Columns = Predicted")

    print("\n              Predicted")
    print("              Fake  Genuine")
    print(f"Actual Fake   {cm[0][0]:5d}  {cm[0][1]:7d}")
    print(f"Actual Genuine{cm[1][0]:5d}  {cm[1][1]:7d}")


if __name__ == "__main__":
    main()