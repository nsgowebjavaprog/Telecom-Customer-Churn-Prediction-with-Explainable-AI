# --- Backend Dockerfile ---
FROM python:3.11-slim

WORKDIR /code

# Install system deps needed by scikit-learn/pandas wheels
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy and install python deps first (better Docker layer caching)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model artifact + app code
COPY ml/models /code/ml/models
COPY backend/app /code/app

# Tell the app exactly where the model lives inside the container
ENV MODEL_PATH=/code/ml/models/churn_pipeline.joblib

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
