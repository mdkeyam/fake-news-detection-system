const API_URL = "http://127.0.0.1:8000/api/v1/predict";


const newsText = document.getElementById("newsText");

const charCount = document.getElementById("charCount");

const analyzeButton =
    document.getElementById("analyzeButton");

const errorMessage =
    document.getElementById("errorMessage");

const loadingMessage =
    document.getElementById("loadingMessage");

const result =
    document.getElementById("result");

const prediction =
    document.getElementById("prediction");

const confidence =
    document.getElementById("confidence");

const confidenceLevel =
    document.getElementById("confidenceLevel");


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

    prediction.textContent =
        data.prediction;

    confidence.textContent =
        `${data.confidence}%`;

    confidenceLevel.textContent =
        data.confidence_level;

    result.classList.remove("hidden");
}


/**
 * Send news text to backend API.
 */
async function analyzeNews() {

    hideError();

    result.classList.add("hidden");

    const text = newsText.value.trim();


    if (text.length < 10) {

        showError(
            "Please enter at least 10 characters."
        );

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

        const response = await fetch(
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


        const data = await response.json();


        if (!response.ok) {

            if (data.detail) {

                throw new Error(
                    "Invalid request."
                );
            }

            throw new Error(
                "Prediction request failed."
            );
        }


        showResult(data);


    } catch (error) {

        console.error(error);

        showError(
            "Unable to connect to the prediction server. " +
            "Make sure the FastAPI server is running."
        );


    } finally {

        hideLoading();
    }
}


analyzeButton.addEventListener(
    "click",
    analyzeNews
);