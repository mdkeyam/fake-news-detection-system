import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parent

FAKE_PATH = BASE_DIR / "data" / "Fake.csv"
TRUE_PATH = BASE_DIR / "data" / "True.csv"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_title(title):
    title = clean_text(title)
    return title.strip()


def main():
    print("=" * 60)
    print("GROUPED DATA LEAKAGE AUDIT - MODEL V2")
    print("=" * 60)

    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat(
        [fake_df, true_df],
        ignore_index=True
    )

    df = df[["title", "text", "label"]].copy()

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Remove exact title-text-label duplicates
    df = df.drop_duplicates(
        subset=["title", "text", "label"]
    ).copy()

    # Same preprocessing as V2
    df["content"] = (
        df["title"] + " " + df["text"]
    ).apply(clean_text)

    df = df[df["content"].str.len() >= 20].copy()

    # Group identical normalized titles together.
    # If title is empty, use the article content.
    df["group"] = df["title"].apply(normalize_title)

    empty_title = df["group"].eq("")
    df.loc[empty_title, "group"] = (
        df.loc[empty_title, "content"]
    )

    print(f"\nTotal records: {len(df)}")
    print(f"Unique groups: {df['group'].nunique()}")

    X = df["content"]
    y = df["label"]
    groups = df["group"]

    # Keep title groups entirely in either train or test
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    overlap = train_groups.intersection(test_groups)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Groups in both sets: {len(overlap)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_features=200000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode"
    )

    print("\nFitting TF-IDF on training data...")

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        C=2.0,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    )

    print("Training grouped-split model...")
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print("\n" + "=" * 60)
    print("GROUPED-SPLIT PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0) * 100:.2f}%")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0) * 100:.2f}%")
    print(f"F1 Score : {f1_score(y_test, y_pred, zero_division=0) * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Fake", "True"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\nAudit check:")
    print(f"Train/test group overlap: {len(overlap)}")

    if overlap:
        print("WARNING: Some groups appear in both sets.")
    else:
        print("PASS: No normalized title groups overlap.")

    print("\nAudit completed.")
    print("The existing V2 model files were not modified.")


if __name__ == "__main__":
    main()