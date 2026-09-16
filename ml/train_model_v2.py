import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
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


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAKE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Fake.csv"
)

TRUE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "True.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fake_news_model_v2.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer_v2.pkl"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Basic text normalization.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Keep letters/numbers and basic punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("FAKE NEWS DETECTION - MODEL V2 TRAINING")
print("=" * 60)

print("\nLoading datasets...")

fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

print(f"Fake dataset shape: {fake_df.shape}")
print(f"True dataset shape: {true_df.shape}")


# ============================================================
# CREATE LABELS
# ============================================================

fake_df["label"] = 0
true_df["label"] = 1


# ============================================================
# COMBINE DATA
# ============================================================

df = pd.concat(
    [fake_df, true_df],
    ignore_index=True
)

print(f"\nCombined dataset: {df.shape}")


# ============================================================
# SELECT RELEVANT COLUMNS
# ============================================================

df = df[
    ["title", "text", "label"]
].copy()


# ============================================================
# REMOVE EXACT DUPLICATE ROWS
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["title", "text", "label"]
)

after = len(df)

print(
    f"Exact duplicate rows removed: {before - after}"
)


# ============================================================
# REMOVE MISSING TEXT
# ============================================================

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")


# ============================================================
# COMBINE TITLE + ARTICLE
# ============================================================

df["content"] = (
    df["title"].astype(str)
    + " "
    + df["text"].astype(str)
)


# ============================================================
# CLEAN TEXT
# ============================================================

print("\nCleaning text...")

df["content"] = df["content"].apply(
    clean_text
)


# ============================================================
# REMOVE EMPTY TEXT
# ============================================================

before = len(df)

df = df[
    df["content"].str.len() >= 20
].copy()

after = len(df)

print(
    f"Invalid/very short rows removed: {before - after}"
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\nFinal dataset:")
print(f"Total records: {len(df)}")

print("\nClass distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
    .rename(
        index={
            0: "Fake",
            1: "True"
        }
    )
)


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df["content"]
y = df["label"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    max_features=200000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    strip_accents="unicode"
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print(
    f"TF-IDF training matrix: {X_train_tfidf.shape}"
)


# ============================================================
# MODEL
# ============================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    C=2.0,
    class_weight="balanced",
    solver="liblinear",
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

print("\nEvaluating model...")

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake",
            "True"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)


print("\nModel saved:")
print(MODEL_PATH)

print("\nVectorizer saved:")
print(VECTORIZER_PATH)

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)