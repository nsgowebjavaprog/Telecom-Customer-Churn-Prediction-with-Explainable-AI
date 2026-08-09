"""
model_utils.py
--------------
Loads the trained sklearn pipeline ONCE at startup (not per-request - that
would be slow) and exposes helper functions used by the routers.
"""
import os
import joblib
import pandas as pd
from pathlib import Path

# Local dev: backend/app/model_utils.py -> project_root/ml/models/...
# Docker: overridden via MODEL_PATH env var set in docker-compose.yml
_DEFAULT_LOCAL_PATH = Path(__file__).resolve().parents[2] / "ml" / "models" / "churn_pipeline.joblib"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(_DEFAULT_LOCAL_PATH)))

_artifact = joblib.load(MODEL_PATH)
PIPELINE = _artifact["pipeline"]
FEATURE_COLUMNS = _artifact["feature_columns"]
MODEL_NAME = _artifact["model_name"]

REQUIRED_CSV_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "TechSupport", "StreamingTV", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must mirror the feature engineering done in ml/train.py exactly,
    otherwise the pipeline sees a different schema at inference time than
    it was trained on (a very common real-world bug!)."""
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"]
    )
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["num_addon_services"] = (
        (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
        + (df["StreamingTV"] == "Yes").astype(int)
    )
    return df


def risk_bucket(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    if prob < 0.66:
        return "Medium"
    return "High"


def predict_single(record: dict) -> dict:
    df = pd.DataFrame([record])
    df = engineer_features(df)
    proba = PIPELINE.predict_proba(df)[0][1]
    pred = "Yes" if proba >= 0.5 else "No"
    return {
        "churn_prediction": pred,
        "churn_probability": round(float(proba), 4),
        "risk_level": risk_bucket(proba),
    }


def validate_csv_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    return missing


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = engineer_features(df)
    proba = PIPELINE.predict_proba(df_feat)[:, 1]
    df_out = df.copy()
    df_out["churn_probability"] = proba.round(4)
    df_out["churn_prediction"] = ["Yes" if p >= 0.5 else "No" for p in proba]
    df_out["risk_level"] = [risk_bucket(p) for p in proba]
    return df_out
