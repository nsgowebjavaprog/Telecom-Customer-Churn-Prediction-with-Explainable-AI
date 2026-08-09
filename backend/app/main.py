"""
main.py
-------
FastAPI application entrypoint.
Run locally:  uvicorn app.main:app --reload --port 8000
Docs (Swagger UI): http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import predict, customers
from app.model_utils import MODEL_NAME

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="ML-powered API to predict customer churn - single & batch (CSV) modes.",
    version="1.0.0",
)

# Allow the React frontend (served separately) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(customers.router)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Churn Prediction API is running"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_in_use": MODEL_NAME}
