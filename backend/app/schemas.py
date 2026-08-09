"""
schemas.py
----------
Pydantic models = the "contract" for every request/response.
FastAPI uses these to:
  - auto-validate incoming JSON (reject bad types/missing fields with 422)
  - auto-generate the OpenAPI/Swagger docs at /docs
  - serialize outgoing responses safely (never leak extra fields)
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from datetime import datetime


class CustomerIn(BaseModel):
    """Single customer record for a real-time prediction request."""
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100, description="Months as a customer")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    ]
    MonthlyCharges: float = Field(gt=0)
    TotalCharges: float = Field(ge=0)

    @field_validator("TotalCharges")
    @classmethod
    def total_not_less_than_monthly_times_zero(cls, v):
        if v < 0:
            raise ValueError("TotalCharges cannot be negative")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "TechSupport": "No", "StreamingTV": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5,
                "TotalCharges": 450.0,
            }
        }


class PredictionOut(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float
    risk_level: Literal["Low", "Medium", "High"]


class PredictionRecord(BaseModel):
    """What we store in the DB / return from CRUD endpoints."""
    id: int
    customer_id: Optional[str] = None
    churn_prediction: str
    churn_probability: float
    risk_level: str
    created_at: datetime

    class Config:
        from_attributes = True  # allows SQLAlchemy ORM -> pydantic conversion


class PredictionUpdate(BaseModel):
    """Partial update payload for PUT /customers/{id}."""
    customer_id: Optional[str] = None
    risk_level: Optional[Literal["Low", "Medium", "High"]] = None


class BatchSummary(BaseModel):
    total_rows: int
    predicted_churn_yes: int
    predicted_churn_no: int
    download_url: str
