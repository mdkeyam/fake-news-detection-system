from fastapi import FastAPI

app = FastAPI(
    title="Fake News Detection API",
    description="API for AI-based fake news prediction",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Fake News Detection API is running",
        "status": "success"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }