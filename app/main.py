from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router


app = FastAPI(
    title="Fake News Detection API",
    description=(
        "AI-based fake news detection system using "
        "TF-IDF and Logistic Regression."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Fake News Detection API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }