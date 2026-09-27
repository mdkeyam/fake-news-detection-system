import os
import re
import joblib
import pandas as pd
import numpy as np

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

FAKE_PATH = os.path.join(BASE_DIR, "data", "Fake.csv")
TRUE_PATH = os.path.join(BASE_DIR, "data", "True.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Keep letters, numbers and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FAKE NEWS DETECTION - MODEL V3 CONTROLLED EXPERIMENT")
print("=" * 70)

print("\nLoading datasets...")

fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat(
    [fake_df, true_df],
    ignore_index=True
)

df = df[["title", "text", "label"]].copy()

print(f"Combined records: {len(df)}")


# ============================================================
# CLEAN MISSING VALUES
# ============================================================

df["title"] = df["title"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["title", "text", "label"]
).copy()

print(
    f"Duplicate rows removed: {before - len(df)}"
)


# ============================================================
# CLEAN TITLE + TEXT SEPARATELY
# ============================================================

print("\nCleaning title and article text...")

df["clean_title"] = df["title"].apply(clean_text)
df["clean_text"] = df["text"].apply(clean_text)


# ============================================================
# REMOVE VERY SHORT RECORDS
# ============================================================

before = len(df)

df = df[
    (
        df["clean_title"].str.len() >= 5
    )
    |
    (
        df["clean_text"].str.len() >= 20
    )
].copy()

print(
    f"Invalid/very short rows removed: {before - len(df)}"
)


# ============================================================
# CREATE THREE DATASETS
# ============================================================

df["combined"] = (
    df["clean_title"]
    + " "
    + df["clean_text"]
).str.strip()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nDataset split:")
print(f"Training samples: {len(train_idx)}")
print(f"Testing samples : {len(test_idx)}")


# ============================================================
# COMMON MODEL FUNCTION
# ============================================================

def train_experiment(name, X):

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = df["label"].iloc[train_idx]
    y_test = df["label"].iloc[test_idx]

    print("\nCreating TF-IDF...")

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_features=200000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(
        f"TF-IDF shape: {X_train_tfidf.shape}"
    )

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

    # Prediction
    y_pred = model.predict(X_test_tfidf)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
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

    print("\nPerformance:")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Fake", "True"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ========================================================
    # TOP FEATURES
    # ========================================================

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    fake_idx = np.argsort(coefficients)[:15]
    true_idx = np.argsort(coefficients)[-15:][::-1]

    print("\nTop Fake-associated features:")

    for i in fake_idx:
        print(
            f"{feature_names[i]:25} "
            f"{coefficients[i]:8.4f}"
        )

    print("\nTop True-associated features:")

    for i in true_idx:
        print(
            f"{feature_names[i]:25} "
            f"{coefficients[i]:8.4f}"
        )

    # ========================================================
    # SAVE MODEL + VECTORIZER
    # ========================================================

    model_path = os.path.join(
        MODEL_DIR,
        f"fake_news_model_v3_{name.lower()}.pkl"
    )

    vectorizer_path = os.path.join(
        MODEL_DIR,
        f"tfidf_vectorizer_v3_{name.lower()}.pkl"
    )

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print("\nSaved:")
    print(model_path)
    print(vectorizer_path)

    return {
        "Experiment": name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1": f1 * 100
    }


# ============================================================
# EXPERIMENT 1 - TITLE ONLY
# ============================================================

title_result = train_experiment(
    "title",
    df["clean_title"]
)


# ============================================================
# EXPERIMENT 2 - ARTICLE TEXT ONLY
# ============================================================

text_result = train_experiment(
    "text",
    df["clean_text"]
)


# ============================================================
# EXPERIMENT 3 - TITLE + ARTICLE
# ============================================================

combined_result = train_experiment(
    "combined",
    df["combined"]
)


# ============================================================
# FINAL COMPARISON
# ============================================================

results = pd.DataFrame([
    title_result,
    text_result,
    combined_result
])

print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\n" + "=" * 70)
print("V3 TRAINING COMPLETED")
print("=" * 70)