# Telecom-Customer-Churn-Prediction-with-Explainable-AI
[HTML,CSS,JS,React] 💻💻 [Python,FastAPI,Docker] 💻💻 [ML, Model, EDA, CSV, AWS] 💻💻 AI-ML Application

### --------------------------------------------------------------------------------------------------------

![alt text](image-0.png)

### --------------------------------------------------------------------------------------------------------

![alt text](image-1.png)

### --------------------------------------------------------------------------------------------------------

![alt text](image-2.png)

### --------------------------------------------------------------------------------------------------------

### Project: Telecom Customer Churn Prediction — full stack, tested end to end.

I built like;-->  Trained the model --> API --> Endpoint

**ML (ml/train.py):** EDA --> feature engineering (tenure buckets, spend ratio, addon count) --> trains Logistic Regression vs Random Forest --> picks the winner by ROC-AUC (Random Forest won, 0.79 AUC) --> saves one pipeline artifact

**FastAPI (backend/):** Pydantic-validated /predict/single, CSV /predict/batch (validates schema, returns predictions CSV), full CRUD (/customers/) over SQLite prediction history

**Docker:** Dockerfile for backend + frontend, docker-compose.yml to **run both with one command**

**Frontend:** React (CDN-based, zero build step) — single-prediction form, CSV upload, records table with delete


### --------------------------------------------------------------------------------------------------------

![alt text](image-3.png)

### --------------------------------------------------------------------------------------------------------

![alt text](image.png)

### --------------------------------------------------------------------------------------------------------

`
python -m venv churn_myenv
`


### --------------------------------------------------------------------------------------------------------




### --------------------------------------------------------------------------------------------------------



### --------------------------------------------------------------------------------------------------------




### --------------------------------------------------------------------------------------------------------




