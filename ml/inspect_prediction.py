from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "models" / "fake_news_model_v2.pkl"
VECTORIZER_FILE = BASE_DIR / "models" / "tfidf_vectorizer_v2.pkl"

headline = (
    "Delhi court allows US national VanDyke, "
    "6 Ukrainians to travel home after NIA charge sheet"
)

# Load existing trained model and vectorizer
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# Transform headline
X = vectorizer.transform([headline])

# Predict class and probabilities
predicted_label = int(model.predict(X)[0])
probabilities = model.predict_proba(X)[0]

print("\n--- PREDICTION INSPECTION ---")
print("Headline:", headline)
print("Predicted label:", predicted_label)
print("Prediction:", "Likely Fake" if predicted_label == 0 else "Likely Genuine")

print("\n--- CLASS PROBABILITIES ---")
for label, probability in zip(model.classes_, probabilities):
    name = "Fake" if int(label) == 0 else "Genuine"
    print(f"{name}: {probability * 100:.2f}%")

# Show features contributing to the prediction.
# For binary LogisticRegression, positive coefficients favor class 1,
# and negative coefficients favor class 0.
if hasattr(model, "coef_") and len(model.classes_) == 2:
    feature_names = vectorizer.get_feature_names_out()
    contributions = X.multiply(model.coef_[0]).toarray()[0]

    nonzero_indices = X.nonzero()[1]

    fake_features = sorted(
        (
            (feature_names[i], contributions[i])
            for i in nonzero_indices
            if contributions[i] < 0
        ),
        key=lambda item: item[1]
    )[:15]

    genuine_features = sorted(
        (
            (feature_names[i], contributions[i])
            for i in nonzero_indices
            if contributions[i] > 0
        ),
        key=lambda item: item[1],
        reverse=True
    )[:15]

    print("\n--- FEATURES PUSHING TOWARD FAKE ---")
    if fake_features:
        for word, score in fake_features:
            print(f"{word:30s} {score:.4f}")
    else:
        print("No negative feature contributions.")

    print("\n--- FEATURES PUSHING TOWARD GENUINE ---")
    if genuine_features:
        for word, score in genuine_features:
            print(f"{word:30s} {score:.4f}")
    else:
        print("No positive feature contributions.")

else:
    print("\nModel does not expose binary LogisticRegression coefficients.")