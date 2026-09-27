// Backend API URL
const API_URL = "http://127.0.0.1:9000/api/v1/predict";

// HTML elements
const newsText = document.getElementById("newsText");
const charCount = document.getElementById("charCount");
const analyzeButton = document.getElementById("analyzeButton");
const clearButton = document.getElementById("clearButton");
const errorMessage = document.getElementById("errorMessage");
const loadingMessage = document.getElementById("loadingMessage");
const result = document.getElementById("result");
const resultTitle = document.getElementById("resultTitle");
const resultIcon = document.getElementById("resultIcon");
const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceLevel = document.getElementById("confidenceLevel");
const classification = document.getElementById("classification");

const resultDisclaimer =
    document.getElementById("resultDisclaimer") ||
    document.querySelector(".disclaimer p");


// ============================================================
// CHARACTER COUNTER
// ============================================================

newsText.addEventListener("input", () => {
    const length = newsText.value.length;

    charCount.textContent = `${length} / 10000`;

    hideError();

    // Visual warning near character limit
    charCount.classList.remove("warning", "danger");

    if (length >= 9000) {
        charCount.classList.add("warning");
    }

    if (length >= 9800) {
        charCount.classList.remove("warning");
        charCount.classList.add("danger");
    }
});


// ============================================================
// ERROR HANDLING
// ============================================================

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}

function hideError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}


// ============================================================
// LOADING STATE
// ============================================================

function showLoading() {
    loadingMessage.classList.remove("hidden");
    analyzeButton.disabled = true;
    clearButton.disabled = true;

    analyzeButton.setAttribute("aria-busy", "true");
}

function hideLoading() {
    loadingMessage.classList.add("hidden");
    analyzeButton.disabled = false;
    clearButton.disabled = false;

    analyzeButton.removeAttribute("aria-busy");
}


// ============================================================
// DISPLAY RESULT
// ============================================================

function showResult(data) {
    const confidenceValue = Number(data.confidence);

    const safeConfidence = Math.min(
        Math.max(confidenceValue, 0),
        100
    );

    const predictionText = String(
        data.prediction || ""
    ).toLowerCase();

    const isFake = predictionText.includes("fake");

    // Basic values
    prediction.textContent = data.prediction || "-";
    confidence.textContent = `${safeConfidence.toFixed(2)}%`;

    confidenceBar.style.width = `${safeConfidence}%`;

    confidenceLevel.textContent =
        data.confidence_level || "-";

    // Classification
    if (classification) {
        classification.textContent =
            Number(data.label) === 0
                ? "Fake"
                : "Genuine";
    }

    // Reset result classes
    result.classList.remove("fake", "genuine");

    // Fake / Genuine styling
    if (isFake) {
        result.classList.add("fake");

        resultTitle.textContent =
            "News appears potentially unreliable";

        // Unicode escape avoids encoding problems
        resultIcon.textContent = "\u26A0\uFE0F";

    } else {
        result.classList.add("genuine");

        resultTitle.textContent =
            "News appears likely genuine";

        resultIcon.textContent = "\u2713";
    }

    // Disclaimer
    if (resultDisclaimer) {
        resultDisclaimer.textContent =
            data.disclaimer ||
            "This is an AI-based prediction and should not be treated as absolute truth.";
    }

    // Show result
    result.classList.remove("hidden");

    // Smoothly bring result into view
    result.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}


// ============================================================
// CLEAR INTERFACE
// ============================================================

function clearAll() {
    newsText.value = "";

    charCount.textContent = "0 / 10000";
    charCount.classList.remove("warning", "danger");

    hideError();

    result.classList.add("hidden");
    result.classList.remove("fake", "genuine");

    confidenceBar.style.width = "0%";

    prediction.textContent = "-";
    confidence.textContent = "0%";
    confidenceLevel.textContent = "-";

    if (classification) {
        classification.textContent = "-";
    }

    if (resultDisclaimer) {
        resultDisclaimer.textContent = "";
    }

    newsText.focus();
}


// ============================================================
// ANALYZE NEWS
// ============================================================

async function analyzeNews() {
    hideError();
    result.classList.add("hidden");

    const text = newsText.value.trim();

    // Empty input
    if (text.length === 0) {
        showError("Please enter news text.");
        newsText.focus();
        return;
    }

    // Minimum length
    if (text.length < 10) {
        showError("Please enter at least 10 characters.");
        newsText.focus();
        return;
    }

    // Maximum length
    if (text.length > 10000) {
        showError("News text cannot exceed 10,000 characters.");
        newsText.focus();
        return;
    }

    showLoading();

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text
            })
        });

        let data;

        try {
            data = await response.json();
        } catch {
            throw new Error(
                "The prediction server returned an invalid response."
            );
        }

        // HTTP errors
        if (!response.ok) {
            if (response.status === 400) {
                throw new Error(
                    data.detail || "Invalid news text."
                );
            }

            if (response.status === 422) {
                throw new Error(
                    "Please enter valid news text between 10 and 10,000 characters."
                );
            }

            if (response.status >= 500) {
                throw new Error(
                    "The prediction server encountered an error. Please try again."
                );
            }

            throw new Error(
                data.detail ||
                `Request failed (${response.status}).`
            );
        }

        // Validate API response
        if (
            !data ||
            !data.prediction ||
            data.confidence === undefined ||
            !Number.isFinite(Number(data.confidence))
        ) {
            throw new Error(
                "The server returned an incomplete prediction."
            );
        }

        showResult(data);

    } catch (error) {
        console.error("Prediction error:", error);

        const errorText =
            String(error.message || "").toLowerCase();

        if (
            error instanceof TypeError ||
            errorText.includes("failed to fetch") ||
            errorText.includes("networkerror")
        ) {
            showError(
                "Unable to connect to the prediction server. " +
                "Make sure FastAPI is running on port 9000."
            );
        } else {
            showError(
                error.message ||
                "Something went wrong while analyzing the news."
            );
        }

    } finally {
        hideLoading();
    }
}


// ============================================================
// BUTTON EVENTS
// ============================================================

analyzeButton.addEventListener(
    "click",
    analyzeNews
);

clearButton.addEventListener(
    "click",
    clearAll
);


// ============================================================
// KEYBOARD SHORTCUT
// Ctrl + Enter = Analyze
// ============================================================

newsText.addEventListener("keydown", (event) => {
    if (
        event.ctrlKey &&
        event.key === "Enter"
    ) {
        event.preventDefault();

        if (!analyzeButton.disabled) {
            analyzeNews();
        }
    }
});