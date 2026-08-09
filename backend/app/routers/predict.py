"""
routers/predict.py
-------------------
Two prediction endpoints:
  POST /predict/single   -> JSON in, JSON prediction out (pydantic validated)
  POST /predict/batch    -> CSV file in, validates schema, returns a CSV
                             file with 3 new predicted columns appended.

This is the "File-Uploading" requirement: upload CSV -> we check the format
(required columns present, correct dtypes) -> if valid, run predictions and
return the same CSV with predicted columns; if invalid, return a 400 with a
clear error listing exactly what's wrong.
"""
import io
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.schemas import CustomerIn, PredictionOut, BatchSummary
from app.model_utils import predict_single, predict_batch, validate_csv_columns
from app.database import get_db, PredictionDB

router = APIRouter(prefix="/predict", tags=["Prediction"])

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "batch_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


@router.post("/single", response_model=PredictionOut)
def predict_single_customer(customer: CustomerIn, db: Session = Depends(get_db)):
    """Real-time single prediction. Pydantic already validated types/ranges
    before this function body even runs."""
    result = predict_single(customer.model_dump())

    # log every prediction to the DB (also demonstrates the "C" of CRUD)
    record = PredictionDB(
        customer_id=None,
        churn_prediction=result["churn_prediction"],
        churn_probability=result["churn_probability"],
        risk_level=result["risk_level"],
    )
    db.add(record)
    db.commit()

    return result


@router.post("/batch", response_model=BatchSummary)
async def predict_batch_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a CSV -> validate format -> predict -> save result CSV ->
    return a summary + a download link."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows.")

    missing_cols = validate_csv_columns(df)
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required column(s): {missing_cols}. "
                   f"Please match the expected template.",
        )

    # basic dtype sanity check on numeric columns
    for col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        if not pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce")):
            raise HTTPException(status_code=400, detail=f"Column '{col}' must be numeric.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["tenure", "MonthlyCharges", "TotalCharges"]].isnull().any().any():
        raise HTTPException(
            status_code=400,
            detail="Numeric columns contain values that could not be parsed as numbers.",
        )

    result_df = predict_batch(df)

    # persist a light-weight log row per prediction (bulk insert)
    for _, row in result_df.iterrows():
        db.add(PredictionDB(
            customer_id=row.get("customerID"),
            churn_prediction=row["churn_prediction"],
            churn_probability=float(row["churn_probability"]),
            risk_level=row["risk_level"],
        ))
    db.commit()

    file_id = f"predictions_{uuid.uuid4().hex[:8]}.csv"
    out_path = OUTPUT_DIR / file_id
    result_df.to_csv(out_path, index=False)

    return {
        "total_rows": len(result_df),
        "predicted_churn_yes": int((result_df["churn_prediction"] == "Yes").sum()),
        "predicted_churn_no": int((result_df["churn_prediction"] == "No").sum()),
        "download_url": f"/predict/download/{file_id}",
    }


@router.get("/download/{file_id}")
def download_result(file_id: str):
    path = OUTPUT_DIR / file_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found or expired.")
    return FileResponse(path, media_type="text/csv", filename=file_id)
