"""
database.py
-----------
Minimal SQLAlchemy setup with SQLite (zero external DB server needed - easy
to demo/run anywhere). Swap SQLALCHEMY_DATABASE_URL for Postgres in prod
(e.g. postgresql://user:pass@host/db) without changing any other code.
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./churn_predictions.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionDB(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, nullable=True)
    churn_prediction = Column(String, nullable=False)
    churn_probability = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a DB session per-request, closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
