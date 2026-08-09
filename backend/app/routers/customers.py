"""
routers/customers.py
---------------------
Full CRUD over stored prediction records (SQLite via SQLAlchemy).
  GET    /customers/            -> list (with pagination via query params)
  GET    /customers/{id}        -> read one
  PUT    /customers/{id}        -> update (e.g. correct customer_id, override risk_level)
  DELETE /customers/{id}        -> delete
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db, PredictionDB
from app.schemas import PredictionRecord, PredictionUpdate

router = APIRouter(prefix="/customers", tags=["Prediction Records (CRUD)"])


@router.get("/", response_model=List[PredictionRecord])
def list_predictions(
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=200, description="Page size"),
    risk_level: str | None = Query(None, description="Filter: Low / Medium / High"),
    db: Session = Depends(get_db),
):
    query = db.query(PredictionDB)
    if risk_level:
        query = query.filter(PredictionDB.risk_level == risk_level)
    return query.order_by(PredictionDB.id.desc()).offset(skip).limit(limit).all()


@router.get("/{record_id}", response_model=PredictionRecord)
def get_prediction(record_id: int, db: Session = Depends(get_db)):
    record = db.query(PredictionDB).filter(PredictionDB.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@router.put("/{record_id}", response_model=PredictionRecord)
def update_prediction(record_id: int, payload: PredictionUpdate, db: Session = Depends(get_db)):
    record = db.query(PredictionDB).filter(PredictionDB.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(record, field, value)

    db.commit()
    db.refresh(record)
    return record


@router.delete("/{record_id}")
def delete_prediction(record_id: int, db: Session = Depends(get_db)):
    record = db.query(PredictionDB).filter(PredictionDB.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    db.delete(record)
    db.commit()
    return {"message": f"Record {record_id} deleted"}
