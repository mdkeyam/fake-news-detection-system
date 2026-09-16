const API_URL = "http://127.0.0.1:9000/api/v1/predict";


const newsText = document.getElementById("newsText");
const charCount = document.getElementById("charCount");

const analyzeButton =
    document.getElementById("analyzeButton");

const clearButton =
    document.getElementById("clearButton");

const errorMessage =
    document.getElementById("errorMessage");

const loadingMessage =
    document.getElementById("loadingMessage");

const result =
    document.getElementById("result");

const resultTitle =
    document.getElementById("resultTitle");

const resultIcon =
    document.getElementById("resultIcon");

const prediction =
    document.getElementById("prediction");

const confidence =
    document.getElementById("confidence");

const confidenceBar =
    document.getElementById("confidenceBar");

const confidenceLevel =
    document.getElementById("confidenceLevel");

const resultDisclaimer =
    document.getElementById("resultDisclaimer") ||
    document.querySelector(".disclaimer p");


/**
 * Update character counter.
 */
newsText.addEventListener("input", () => {

    charCount.textContent =
        `${newsText.value.length} / 10000`;

});


/**
 * Show error message.
 */
function showError(message) {

    errorMessage.textContent = message;

    errorMessage.classList.remove("hidden");
}


/**
 * Hide error message.
 */
function hideError() {

    errorMessage.textContent = "";

    errorMessage.classList.add("hidden");
}


/**
 * Show loading state.
 */
function showLoading() {

    loadingMessage.classList.remove("hidden");

    analyzeButton.disabled = true;
}


/**
 * Hide loading state.
 */
function hideLoading() {

    loadingMessage.classList.add("hidden");

    analyzeButton.disabled = false;
}


/**
 * Display prediction result.
 */
function showResult(data) {

    const confidenceValue =
        Number(data.confidence);


    prediction.textContent =
        data.prediction;


    confidence.textContent =
        `${confidenceValue.toFixed(2)}%`;


    confidenceBar.style.width =
        `${Math.min(Math.max(confidenceValue, 0), 100)}%`;


    confidenceLevel.textContent =
        data.confidence_level;


   if (resultDisclaimer) {

    if (data.disclaimer) {

        resultDisclaimer.textContent =
            data.disclaimer;

    } else {

        resultDisclaimer.textContent =
            "This is an AI-based prediction and should not be treated as absolute truth.";

    }

}


    /*
     * Determine whether the result is fake
     * or genuine.
     *
     * Supports:
     * "Likely Fake"
     * "Likely Genuine"
     */
    const predictionText =
        String(data.prediction).toLowerCase();


    const isFake =
        predictionText.includes("fake");


    result.classList.remove(
        "fake",
        "genuine"
    );


    if (isFake) {

        result.classList.add("fake");

        resultTitle.textContent =
            "News appears potentially unreliable";

        resultIcon.textContent =
            "⚠️";

    } else {

        result.classList.add("genuine");

        resultTitle.textContent =
            "News appears likely genuine";

        resultIcon.textContent =
            "✓";
    }


    result.classList.remove("hidden");
}


/**
 * Clear the complete interface.
 */
function clearAll() {

    newsText.value = "";

    charCount.textContent =
        "0 / 10000";

    hideError();

    result.classList.add("hidden");

    result.classList.remove(
        "fake",
        "genuine"
    );

    confidenceBar.style.width =
        "0%";

    prediction.textContent =
        "-";

    confidence.textContent =
        "0%";

    confidenceLevel.textContent =
        "-";

    resultDisclaimer.textContent =
        "";
}


/**
 * Send news text to backend API.
 */
async function analyzeNews() {

    hideError();

    result.classList.add("hidden");

    const text =
        newsText.value.trim();


    /*
     * Frontend validation
     */
    if (text.length < 10) {

        showError(
            "Please enter at least 10 characters."
        );

        newsText.focus();

        return;
    }


    if (text.length > 10000) {

        showError(
            "News text cannot exceed 10,000 characters."
        );

        return;
    }


    showLoading();


    try {

        const response =
            await fetch(
                API_URL,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        text: text
                    })
                }
            );


        /*
         * Try to parse JSON safely.
         */
        let data;

        try {

            data =
                await response.json();

        } catch {

            throw new Error(
                "The prediction server returned an invalid response."
            );
        }


        /*
         * Handle HTTP errors properly.
         */
        if (!response.ok) {

            if (response.status === 400) {

                throw new Error(
                    data.detail ||
                    "Invalid news text."
                );
            }


            if (response.status === 422) {

                throw new Error(
                    "Please enter valid news text between 10 and 10,000 characters."
                );
            }


            throw new Error(
                data.detail ||
                "Prediction request failed."
            );
        }


        /*
         * Validate expected response.
         */
        if (
            !data.prediction ||
            data.confidence === undefined
        ) {

            throw new Error(
                "The server returned an incomplete prediction."
            );
        }


        showResult(data);


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        /*
         * Show meaningful error.
         */
        if (
            error instanceof TypeError &&
            error.message.includes("fetch")
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


/*
 * Button events
 */
analyzeButton.addEventListener(
    "click",
    analyzeNews
);


clearButton.addEventListener(
    "click",
    clearAll
);


/*
 * Allow Ctrl + Enter to analyze.
 */
newsText.addEventListener(
    "keydown",
    (event) => {

        if (
            event.ctrlKey &&
            event.key === "Enter"
        ) {

            analyzeNews();
        }
    }
);