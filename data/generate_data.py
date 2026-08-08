"""
generate_data.py
-----------------
Creates a realistic, synthetic Telecom Customer Churn dataset and saves it as
data/telecom_churn.csv

Why synthetic data?
- No internet download needed (works offline / behind firewalls)
- You fully understand every column -> easier to explain in interviews
- Mirrors the real-world "Telco Customer Churn" dataset schema (IBM sample),
  so the same code works if you swap in the real Kaggle CSV later.

Run:
    python generate_data.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000  # number of customers

gender = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N)
dependents = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])
tenure = np.random.randint(0, 73, N)  # months with company
phone_service = np.random.choice(["Yes", "No"], N, p=[0.9, 0.1])
multiple_lines = np.random.choice(["Yes", "No", "No phone service"], N, p=[0.42, 0.48, 0.1])
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])
online_security = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.29, 0.49, 0.22])
tech_support = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.29, 0.49, 0.22])
streaming_tv = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.38, 0.40, 0.22])
contract = np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.24, 0.21])
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N
)
monthly_charges = np.round(np.random.uniform(18, 120, N), 2)
total_charges = np.round(monthly_charges * tenure + np.random.uniform(-50, 50, N), 2)
total_charges = np.clip(total_charges, 0, None)

# ---- Build churn probability from realistic business logic (not random!) ----
# Month-to-month + fiber + high monthly charge + low tenure => higher churn risk
churn_score = (
    (contract == "Month-to-month") * 0.35
    + (internet_service == "Fiber optic") * 0.20
    + (payment_method == "Electronic check") * 0.15
    + (monthly_charges > 80) * 0.15
    + (tenure < 12) * 0.25
    - (tenure > 48) * 0.20
    - (contract == "Two year") * 0.30
    + np.random.normal(0, 0.15, N)  # noise
)
churn_prob = 1 / (1 + np.exp(-((churn_score - 0.3) * 4)))  # sigmoid squashing
churn = (np.random.rand(N) < churn_prob).astype(int)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame({
    "customerID": [f"CUST-{i:05d}" for i in range(N)],
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn_label,
})

# introduce a few missing values (realistic messiness -> forces real EDA/cleaning)
mask = np.random.choice(df.index, size=30, replace=False)
df.loc[mask, "TotalCharges"] = np.nan

out_path = "telecom_churn.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows -> {out_path}")
print(df["Churn"].value_counts(normalize=True))
