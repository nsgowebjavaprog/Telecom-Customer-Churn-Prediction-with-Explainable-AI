"""
train.py
--------
End-to-end ML pipeline for Telecom Customer Churn Prediction.

Steps (this is the story you tell in interviews):
1. Load data
2. EDA (nulls, class balance, correlations) -> printed + saved as text summary
3. Data cleaning (missing values)
4. Feature engineering (new features + encoding via ColumnTransformer)
5. Train/test split (stratified, because classes matter for churn)
6. Train 2 algorithms: Logistic Regression (baseline, interpretable)
                        Random Forest (non-linear, usually stronger)
7. Evaluate both with multiple metrics (accuracy is NOT enough for churn -
   classes are imbalanced-ish and business cares about RECALL of churners)
8. Select the better model automatically based on ROC-AUC
9. Save: the winning pipeline (preprocessing + model) as one joblib artifact,
   so FastAPI only needs to load ONE file.

Run:
    python train.py
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

DATA_PATH = "../data/telecom_churn.csv"
MODEL_OUT = "models/churn_pipeline.joblib"
METRICS_OUT = "models/metrics.json"

# --------------------------------------------------------------------------
# 1. LOAD
# --------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# --------------------------------------------------------------------------
# 2. EDA (kept lightweight/print-based so this script is also a report)
# --------------------------------------------------------------------------
print("\n--- Nulls per column ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n--- Class balance (Churn) ---")
print(df["Churn"].value_counts(normalize=True))

print("\n--- Numeric summary ---")
print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe())

# --------------------------------------------------------------------------
# 3. CLEANING
# --------------------------------------------------------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.drop(columns=["customerID"])  # identifier, not predictive
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# --------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------------------------------
# New engineered features (this is what interviewers love to hear about):
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

target = "Churn"
y = df[target]
X = df.drop(columns=[target])

categorical_cols = X.select_dtypes(include="object").columns.tolist() + ["tenure_bucket"]
categorical_cols = list(dict.fromkeys(categorical_cols))  # dedupe, keep order
numeric_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

# --------------------------------------------------------------------------
# 5. TRAIN/TEST SPLIT
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------------------------------
# 6. TWO ALGORITHMS
# --------------------------------------------------------------------------
candidates = {
    "logistic_regression": Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]),
    "random_forest": Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
        )),
    ]),
}

results = {}
for name, pipe in candidates.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    results[name] = {"pipeline": pipe, "metrics": metrics}

    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", round(metrics["roc_auc"], 4))

# --------------------------------------------------------------------------
# 7 & 8. SELECT BEST MODEL BY ROC-AUC (robust to class imbalance)
# --------------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
best_pipeline = results[best_name]["pipeline"]
print(f"\n>>> Selected best model: {best_name} "
      f"(ROC-AUC={results[best_name]['metrics']['roc_auc']:.4f})")

# --------------------------------------------------------------------------
# 9. SAVE ARTIFACTS
# --------------------------------------------------------------------------
joblib.dump({
    "pipeline": best_pipeline,
    "feature_columns": X.columns.tolist(),
    "model_name": best_name,
}, MODEL_OUT)

metrics_summary = {k: v["metrics"] for k, v in results.items()}
metrics_summary["selected_model"] = best_name
with open(METRICS_OUT, "w") as f:
    json.dump(metrics_summary, f, indent=2, default=str)

print(f"\nSaved model -> {MODEL_OUT}")
print(f"Saved metrics -> {METRICS_OUT}")
