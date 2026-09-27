import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# FAKE NEWS DETECTION - MODEL V4
# Leakage-Resistant / Artifact-Controlled Experiment
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1. TEXT CLEANING
# ------------------------------------------------------------

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove obvious formatting/template artifacts.
    # IMPORTANT:
    # Meaningful entities such as US, Trump, Obama are NOT removed.
    artifact_patterns = [
        r"\bfeatured\s+image\b",
        r"\bimage\s+via\b",
        r"\bimage\s+by\b",
        r"\bread\s+more\b",
        r"\bgetty\s+images\b",
        r"\bphoto\s+by\b",
        r"\bphoto\s+via\b",
    ]

    for pattern in artifact_patterns:
        text = re.sub(pattern, " ", text)

    # Remove standalone "via" only when it behaves like
    # an attribution marker.
    text = re.sub(r"\bvia\b", " ", text)

    # Reuters source marker:
    # Remove common attribution/header patterns, but keep
    # normal words such as "said", "Washington", etc.
    text = re.sub(r"\bwashington\s*\(\s*reuters\s*\)", " ", text)
    text = re.sub(r"\breuters\b", " ", text)

    # Keep letters/numbers.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

print("=" * 70)
print("FAKE NEWS DETECTION - MODEL V4")
print("Leakage-Resistant / Artifact-Controlled Experiment")
print("=" * 70)

print("\nLoading datasets...")

fake_path = os.path.join(DATA_DIR, "Fake.csv")
true_path = os.path.join(DATA_DIR, "True.csv")

fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

print(f"Fake dataset shape: {fake_df.shape}")
print(f"True dataset shape: {true_df.shape}")


# ------------------------------------------------------------
# 3. SELECT RELEVANT COLUMNS
# ------------------------------------------------------------

fake_df = fake_df[["title", "text", "subject", "date"]].copy()
true_df = true_df[["title", "text", "subject", "date"]].copy()

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)

print(f"\nCombined dataset: {df.shape}")


# ------------------------------------------------------------
# 4. REMOVE EXACT DUPLICATES
# ------------------------------------------------------------

before = len(df)

df = df.drop_duplicates(
    subset=["title", "text", "label"]
).reset_index(drop=True)

removed = before - len(df)

print(f"Exact duplicate rows removed: {removed}")


# ------------------------------------------------------------
# 5. BUILD ORIGINAL CONTENT
# ------------------------------------------------------------

df["content"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["text"].fillna("").astype(str)
)


# ------------------------------------------------------------
# 6. CLEAN CONTENT
# ------------------------------------------------------------

print("\nCleaning text...")

df["content"] = df["content"].apply(clean_text)

before = len(df)

df = df[df["content"].str.len() >= 20].copy()

removed_short = before - len(df)

print(f"Invalid/very short rows removed: {removed_short}")

df = df.reset_index(drop=True)


# ------------------------------------------------------------
# 7. DATA SUMMARY
# ------------------------------------------------------------

print("\nFinal dataset:")
print(f"Total records: {len(df)}")

print("\nClass distribution:")

class_counts = df["label"].value_counts().sort_index()

print(f"Fake  (0): {class_counts.get(0, 0)}")
print(f"True  (1): {class_counts.get(1, 0)}")


# ------------------------------------------------------------
# 8. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test split:")
print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")


# ------------------------------------------------------------
# 9. TF-IDF
# ------------------------------------------------------------

print("\nCreating TF-IDF features...")

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

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Training matrix: {X_train_tfidf.shape}")
print(f"Testing matrix : {X_test_tfidf.shape}")


# ------------------------------------------------------------
# 10. LOGISTIC REGRESSION
# ------------------------------------------------------------

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    C=2.0,
    class_weight="balanced",
    solver="liblinear",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# ------------------------------------------------------------
# 11. EVALUATION
# ------------------------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 70)
print("V4 RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# 12. FEATURE ANALYSIS
# ------------------------------------------------------------

print("\nTop Fake-associated features:")

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

fake_indices = coefficients.argsort()[:20]

for idx in fake_indices:
    print(f"{feature_names[idx]:30s} {coefficients[idx]:10.4f}")


print("\nTop True-associated features:")

true_indices = coefficients.argsort()[-20:][::-1]

for idx in true_indices:
    print(f"{feature_names[idx]:30s} {coefficients[idx]:10.4f}")


# ------------------------------------------------------------
# 13. CHECK IMPORTANT FEATURES
# ------------------------------------------------------------

print("\nImportant feature coefficient check:")

important_terms = [
    "us",
    "trump",
    "obama",
    "reuters",
    "video",
    "via",
    "read more",
    "featured image",
    "image",
    "said"
]

vocab = vectorizer.vocabulary_

for term in important_terms:
    if term in vocab:
        idx = vocab[term]
        print(f"{term:20s} {coefficients[idx]:10.4f}")
    else:
        print(f"{term:20s} NOT PRESENT")


# ------------------------------------------------------------
# 14. SAVE MODEL
# ------------------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "fake_news_model_v4.pkl"
)

vectorizer_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer_v4.pkl"
)

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("\nModels saved:")
print(model_path)
print(vectorizer_path)


# ------------------------------------------------------------
# 15. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("V4 TRAINING COMPLETE")
print("=" * 70)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

print("\nV3 Combined baseline:")
print("Accuracy : 99.23%")
print("Precision: 99.06%")
print("Recall   : 99.53%")
print("F1 Score : 99.29%")

print("\nV4 is intentionally evaluated against the V3 baseline.")
print("Do NOT replace the production model yet.")