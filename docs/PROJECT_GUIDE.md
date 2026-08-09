# Telecom Customer Churn Prediction
## Complete Project Guide, Code Walkthrough & Interview Preparation

---

# PART 1 — WHY THIS PROJECT STANDS OUT

Most AI/ML job applicants show a Jupyter notebook: load CSV, `.fit()`, print
accuracy, done. That proves you can call `sklearn` — it does **not** prove
you can build something a startup could actually ship.

This project proves the second thing. It has every layer a real ML product
needs:

| Layer | What it shows a hiring manager |
|---|---|
| EDA + Feature Engineering | You understand the data, not just the API |
| 2 algorithms + metric-based selection | You reason about model choice, don't guess |
| Multiple metrics (not just accuracy) | You understand imbalanced-class evaluation |
| FastAPI + Pydantic validation | You can build a production API, not a demo script |
| CRUD + SQLite/SQLAlchemy | You understand persistence, not just inference |
| CSV upload with format validation | You think about real users sending messy data |
| Docker + docker-compose | You can ship something that "just runs" anywhere |
| React frontend | You're not backend-only — you can deliver an MVP end-to-end |

Startups (especially small AI/ML startups in Bangalore) usually have **no
dedicated MLOps or platform team**. They want one person who can take a
model from a notebook to something a non-technical founder can click
through. That is exactly what this project demonstrates.

---

# PART 2 — ARCHITECTURE

```
                     ┌───────────────────────┐
                     │   React Frontend       │
                     │ (index.html + app.js)  │
                     │  - Single prediction   │
                     │  - CSV batch upload    │
                     │  - CRUD records view   │
                     └───────────┬───────────┘
                                 │ fetch() / REST (JSON, multipart)
                                 ▼
                     ┌───────────────────────┐
                     │     FastAPI Backend    │
                     │  main.py + routers/    │
                     │  - /predict/single     │
                     │  - /predict/batch      │
                     │  - /customers (CRUD)   │
                     └─────┬─────────────┬────┘
                           │             │
             loads once    │             │  reads/writes
                           ▼             ▼
              ┌─────────────────┐   ┌─────────────┐
              │ churn_pipeline   │   │  SQLite DB   │
              │   .joblib        │   │ predictions  │
              │ (sklearn Pipeline│   │   table      │
              │  = preprocessing │   └─────────────┘
              │  + trained model)│
              └────────┬─────────┘
                       ▲
                       │ produced by
              ┌────────┴─────────┐
              │   ml/train.py     │
              │  EDA → FE →       │
              │  2 models →       │
              │  pick best (AUC)  │
              └───────────────────┘
```

Everything is containerized via `docker-compose.yml` into two services:
`backend` (FastAPI + model) and `frontend` (nginx serving static React).

---

# PART 3 — FILE-BY-FILE WALKTHROUGH

### `data/generate_data.py`
Generates a synthetic but *realistic* telecom dataset (5000 rows) mirroring
the well-known IBM Telco Churn schema. Churn probability is built from
actual business logic (month-to-month contracts, fiber internet, high
charges, low tenure → higher churn) plus noise — not pure randomness. This
matters: if asked "how did you validate your synthetic data is realistic,"
you can explain the logic instead of just saying "I used `np.random`."

**Interview talking point:** *"I used synthetic data because it let me
control ground truth and explain every column, but the schema and code work
identically on the real Kaggle Telco dataset — I designed it to be a
drop-in replacement."*

### `ml/train.py` — the ML core
Runs, in order:
1. **Load** → `pd.read_csv`
2. **EDA** → null counts, class balance, numeric summary (printed as a
   lightweight "report" — in a real job you'd also do this in a notebook
   with plots, but the pipeline itself must be script-based for
   reproducibility)
3. **Cleaning** → coerce `TotalCharges` to numeric (it has blanks in the
   real dataset too — this is a famous real-world gotcha), drop the ID
   column
4. **Feature Engineering** → `tenure_bucket` (binned), `avg_monthly_spend`
   (derived ratio), `is_new_customer` (flag), `num_addon_services` (count)
5. **Preprocessing pipeline** → `ColumnTransformer` with
   `SimpleImputer + StandardScaler` for numeric columns and
   `SimpleImputer + OneHotEncoder` for categorical columns — bundled into
   one sklearn `Pipeline` so **the exact same transform is applied at
   inference time with zero manual re-coding**
6. **Train/test split** → stratified (important: keeps class ratio equal
   in train/test since churn is a classification target)
7. **Two algorithms**:
   - `LogisticRegression` (baseline — linear, interpretable, fast,
     `class_weight="balanced"` to not ignore the minority class)
   - `RandomForestClassifier` (non-linear, captures feature interactions,
     usually stronger on tabular data)
8. **Metrics** — accuracy, precision, recall, F1, ROC-AUC, confusion
   matrix. **Why not just accuracy?** For churn, missing a churner
   (false negative) is expensive (lost revenue) — recall matters. ROC-AUC
   is used for model *selection* because it's threshold-independent and
   robust when classes are roughly balanced-but-not-identical.
9. **Model selection** → picks whichever model has the higher ROC-AUC
   automatically (`max(results, key=...)`) — not a hardcoded choice.
10. **Save** → one `joblib` file containing the *entire pipeline*
    (preprocessing + model) + the feature column order + which model won.
    FastAPI only ever loads **one artifact**.

### `backend/app/schemas.py` — Pydantic
Defines `CustomerIn` with `Literal[...]` types for every categorical field
(so FastAPI rejects e.g. `"InternetService": "Fibre"` typo with a 422 before
it ever reaches the model), numeric `Field(ge=..., le=...)` constraints, and
a custom `@field_validator`. Also defines the response models
(`PredictionOut`, `PredictionRecord`) — Pydantic validates **outgoing** data
too, which prevents accidentally leaking internal fields.

### `backend/app/database.py`
SQLAlchemy + SQLite. One table `predictions` logs every prediction (single
or batch) with a timestamp. `get_db()` is a FastAPI **dependency** —
`Depends(get_db)` gives each request its own DB session and guarantees it's
closed after, even on error (via `try/finally`).

### `backend/app/model_utils.py`
Loads the joblib artifact **once, at import time** (not per-request — a
classic beginner mistake that tanks API latency). Contains
`engineer_features()` which **must mirror `ml/train.py`'s feature
engineering exactly** — this is a real and common production bug
(training/serving skew) and worth explicitly mentioning in interviews.

### `backend/app/routers/predict.py`
- `POST /predict/single` — pydantic-validated JSON in, prediction out,
  logs to DB.
- `POST /predict/batch` — accepts a CSV `UploadFile`, validates: is it a
  `.csv`? does it parse? is it non-empty? does it have all required
  columns? are the numeric columns actually numeric? Only if **all** checks
  pass does it run predictions, append 3 columns
  (`churn_probability`, `churn_prediction`, `risk_level`), save to disk, and
  return a `download_url`.
- `GET /predict/download/{file_id}` — serves the result CSV as a
  `FileResponse`.

### `backend/app/routers/customers.py` — CRUD
Textbook REST CRUD: `GET /customers/` (list + pagination + filter by
`risk_level`), `GET /customers/{id}`, `PUT /customers/{id}` (partial update
via `exclude_unset=True`), `DELETE /customers/{id}`. All backed by
SQLAlchemy ORM queries.

### `backend/app/main.py`
Wires routers together, adds CORS middleware (needed because the React
frontend runs on a different port/origin than the API), and exposes
`/health` for container orchestration health checks.

### `backend/Dockerfile`
Multi-step: install deps first (Docker layer caching — deps rarely change,
so this layer is cached across rebuilds), then copy the trained model and
app code, set `MODEL_PATH` via `ENV`, expose port 8000.

### `frontend/` — React (CDN build, no npm step)
`index.html` loads React, ReactDOM, and Babel Standalone from a CDN so JSX
compiles **in the browser** — zero build tooling required to run this
project (great for demos). `app.js` has three components: `SinglePrediction`
(a form → calls `/predict/single`), `BatchUpload` (drag/click a CSV →
calls `/predict/batch` → shows a download link), and `Records` (calls
`GET /customers/` and supports delete). Uses `fetch()`, React `useState`/
`useEffect` — no Redux/Router needed for this scope.

**Interview talking point:** *"I used the CDN + Babel-standalone approach
deliberately, to keep the whole project runnable with zero build step for
demos. In a production app I'd switch to Vite + a proper build pipeline —
I know how to set that up too."*

### `docker-compose.yml`
Two services (`backend`, `frontend`), each built from its own Dockerfile,
`frontend` depends on `backend`. One command (`docker compose up --build`)
runs the entire stack.

---

# PART 4 — SETUP & RUN (Step-by-Step)

## Option A — Local, no Docker

```bash
# Clone/open the project folder
cd churn-prediction-project

# 1) Create the dataset
cd data
pip install -r ../ml/requirements.txt
python generate_data.py

# 2) Train + save the model
cd ../ml
python train.py

# 3) Start the API
cd ../backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# Visit http://localhost:8000/docs for interactive Swagger UI

# 4) Start the frontend (new terminal)
cd ../frontend
python -m http.server 3000
# Visit http://localhost:3000
```

## Option B — Docker Compose (one command)

```bash
cd churn-prediction-project
docker compose up --build
# Backend:  http://localhost:8000/docs
# Frontend: http://localhost:3000
```

## Testing the CSV upload
Use `data/telecom_churn.csv` itself (drop the `Churn` column or leave it —
extra columns are ignored) as a sample upload, or craft a small CSV with the
16 required columns to see the validation errors in action.

---

# PART 5 — HOW TO PRESENT THIS IN AN INTERVIEW

## The 60-second pitch (memorize this shape, not the exact words)
> "I built an end-to-end churn prediction system for a telecom use case —
> not just a model, but the full product path a startup would actually
> need. I did EDA and feature engineering, trained and compared two
> algorithms — logistic regression as an interpretable baseline and random
> forest for non-linear patterns — and selected the better one using
> ROC-AUC because accuracy alone is misleading here. I wrapped the winning
> model in a FastAPI service with Pydantic validation and full CRUD over
> prediction history in SQLite, containerized it with Docker, and built a
> React frontend that supports both single predictions and CSV batch
> uploads with format validation — so a non-technical user could upload a
> spreadsheet and download predictions without touching code."

## Anticipate the follow-ups
- *"Walk me through a wrong prediction / how would you debug it?"* → talk
  about training/serving skew, checking `engineer_features()` consistency,
  logging inputs alongside predictions (which you already do via the DB).
- *"Why not deep learning?"* → tabular data with <20 features and 5k rows;
  tree-based models (RF/GBM) reliably beat deep nets on small tabular data;
  DL adds complexity without benefit here.
- *"How would you scale this?"* → move SQLite → Postgres, add a message
  queue (Celery/RQ) for batch jobs instead of synchronous CSV processing,
  add model versioning (MLflow), add auth, deploy behind a load balancer,
  add caching for repeated single predictions.
- *"How would you monitor this in production?"* → log prediction
  distributions over time to detect data drift, track model performance
  against ground-truth churn as it becomes known (delayed labels problem),
  add alerting if the % predicted "High risk" shifts sharply.

## What to say if asked "did you deploy this / is it live?"
Be honest — say it's built to run locally/via Docker end-to-end, and if
asked, mention you *could* deploy it in minutes to Render/Railway/AWS
ECS/Fly.io since it's already containerized — that's the whole point of
the Docker layer. Don't claim a live URL you don't have.

---

# PART 6 — INTERVIEW Q&A BANK

## A. Machine Learning Fundamentals

**Q1. Why did you use both Logistic Regression and Random Forest instead of
just picking one?**
Because model choice should be evidence-based, not assumed. Logistic
Regression gives an interpretable, fast baseline (coefficients show
direction/magnitude of each feature's effect). Random Forest captures
non-linear interactions (e.g., "fiber + month-to-month" combined effect)
that logistic regression can't. Comparing them with the same metrics tells
you whether the added complexity of RF is actually worth it.

**Q2. Why ROC-AUC to pick the winner, not accuracy?**
Accuracy is misleading when class costs are asymmetric or classes aren't
perfectly balanced — a model predicting "No churn" for everyone can still
score high accuracy on a ~51/49 split, but that's useless for the business.
ROC-AUC measures ranking quality across *all* thresholds, so it's a better
signal for "is this model actually separating churners from non-churners"
independent of the specific cutoff you'll deploy with.

**Q3. What is precision vs recall here, and which matters more for churn?**
Precision = of everyone we flagged as "will churn," how many actually
churn. Recall = of everyone who actually churns, how many did we catch.
For churn, missing an actual churner (low recall) is usually worse than a
false alarm (low precision), because a false alarm just costs a retention
email while a missed churner is lost revenue. That's why I used
`class_weight="balanced"` to bias the model toward not ignoring the
minority/costly class.

**Q4. Explain your feature engineering choices.**
`tenure_bucket` — captures non-linear relationship between tenure and
churn (new customers churn a lot, this flattens after ~1 year — binning
lets even linear models pick this up). `avg_monthly_spend` — normalizes
`TotalCharges` by tenure, catching customers whose spend rate looks
anomalous relative to their tenure. `is_new_customer` — explicit flag
since new customers are disproportionately high risk. `num_addon_services`
— aggregates 3 sparse yes/no columns into one signal of "how invested is
this customer in our ecosystem."

**Q5. How do you handle the missing values in `TotalCharges`?**
`pd.to_numeric(..., errors="coerce")` turns unparseable strings into NaN,
then the `ColumnTransformer`'s `SimpleImputer(strategy="median")` fills
numeric NaNs during the pipeline fit — importantly, median is computed
*only from training data* and reused at inference, avoiding data leakage.

**Q6. What is data leakage and did you avoid it?**
Leakage = information from outside the training set (often the test set,
or the future) sneaking into training. I avoided it by: (a) fitting all
preprocessing (imputer, scaler, encoder) inside a `Pipeline` that's fit
only on the training split, never on the full dataset before splitting;
(b) using `train_test_split(..., stratify=y)` after feature engineering
that only uses per-row information, not aggregate statistics computed
across the whole dataset.

**Q7. Why OneHotEncoding instead of label encoding for categoricals?**
These are nominal categories with no ordinal relationship (e.g.,
`PaymentMethod`) — label encoding would falsely imply an order (e.g.
"Bank transfer" > "Mailed check"), which would mislead a linear model in
particular. `handle_unknown="ignore"` also protects inference from
crashing if a new category appears that wasn't seen in training.

**Q8. How would you improve this model further?**
Hyperparameter tuning (GridSearchCV/Optuna) on the RF, try
gradient boosting (XGBoost/LightGBM), add SHAP for explainability, engineer
interaction features, try SMOTE for the minority class instead of only
`class_weight`, and add more real-world features (support tickets, usage
trends over time) if available.

**Q9. What's the difference between `predict()` and `predict_proba()` and
why do you use the latter?**
`predict()` returns the hard 0/1 label using a default 0.5 threshold.
`predict_proba()` returns the underlying probability — I use this because
(a) it's needed for ROC-AUC, and (b) it lets the business choose their own
risk threshold (e.g., only act on >70% probability) instead of being locked
into 0.5, and it enables the 3-tier `risk_level` bucketing.

**Q10. How do you know your model isn't overfitting?**
Compared train vs held-out test metrics (not shown printed but easy to
add), used a proper stratified train/test split, and capped Random
Forest's `max_depth=8` specifically to control variance/overfitting on a
relatively small (5k row) dataset. Cross-validation would be the next step
for a more rigorous estimate.

## B. FastAPI & Backend

**Q11. Why FastAPI over Flask/Django?**
Native async support, automatic OpenAPI/Swagger docs generation, and
built-in request/response validation via Pydantic — which for an ML API
means malformed input is rejected with a clear 422 error before it ever
reaches the model (avoiding cryptic sklearn errors reaching the user).

**Q12. What does Pydantic actually do for you here?**
Type + constraint validation (`Literal`, `Field(ge=, le=)`), automatic
JSON serialization/deserialization, auto-generates the OpenAPI schema so
`/docs` is always accurate, and validates *outgoing* responses too so
internal fields never leak.

**Q13. Explain FastAPI's dependency injection (`Depends(get_db)`).**
`Depends()` tells FastAPI to call `get_db()` before the endpoint runs and
inject its return value as an argument. Since `get_db` is a generator using
`yield`, FastAPI treats everything before `yield` as setup and everything
after as teardown (a `finally: db.close()`), guaranteeing every request
gets a fresh session that's always closed — even if the endpoint raises an
exception.

**Q14. How does your API validate CSV uploads?**
Layered checks in `/predict/batch`: file extension check, parse-as-CSV
check (catches malformed files), empty-dataframe check, required-columns
check (diffs against a known schema list and reports exactly which
columns are missing), then per-column numeric-type coercion checks. Any
failure returns an HTTP 400 with a specific, actionable message rather
than a generic 500.

**Q15. Explain the CRUD endpoints — what does each HTTP verb mean here?**
`GET /customers/` (list, paginated via `skip`/`limit`, filterable) = Read
many. `GET /customers/{id}` = Read one. `PUT /customers/{id}` = Update
(partial, via `exclude_unset=True` so only provided fields change).
`DELETE /customers/{id}` = Delete. Prediction *creation* happens
implicitly as a side-effect of `/predict/single` and `/predict/batch`
(logging every prediction), which is a deliberate design choice — the
create endpoint is prediction-driven, not a raw manual insert.

**Q16. Why load the model at import time instead of inside each endpoint
function?**
Loading a joblib/pickle model involves disk I/O and deserialization —
doing that on every request would add significant, unnecessary latency and
load. Loading once at module import (when the app starts) means every
request just reuses the already-in-memory object.

**Q17. How do you prevent the API from crashing on a bad CSV upload?**
Every failure path raises `HTTPException` with a specific status code and
message rather than letting an unhandled exception propagate — FastAPI
converts unhandled exceptions to opaque 500s, which is bad UX and can leak
stack traces; explicit validation with clear 400s is much better for API
consumers (including the CSV-format check that runs *before* touching the
model).

## C. Docker & Deployment

**Q18. Why Docker for an ML project?**
Reproducibility ("works on my machine" eliminated), environment isolation
(exact Python + library versions baked into the image), and portability —
the same image runs identically on a laptop, CI server, or cloud VM. For a
startup, it also means a new team member (or interviewer!) can run the
whole stack with one command.

**Q19. Explain your Dockerfile layer ordering.**
`COPY requirements.txt` + `pip install` happens *before* `COPY app code` —
Docker caches each layer, and since dependencies change far less often than
application code, this ordering means most rebuilds only re-run the fast
"copy code" layer instead of reinstalling every package from scratch.

**Q20. What does `docker-compose.yml` do that `docker run` doesn't?**
Orchestrates multiple containers (backend + frontend here) as one unit —
defines how they're built, which ports they expose, dependency order
(`depends_on`), restart policy, and volumes — all with a single
`docker compose up` instead of manually running/networking containers by
hand.

**Q21. How would you move this to production (e.g. AWS/GCP)?**
Push images to a container registry (ECR/GCR), deploy via ECS/Cloud Run/
Kubernetes, put a managed Postgres behind the backend instead of SQLite,
add a reverse proxy/HTTPS (via ALB or nginx+certbot), add environment-based
secrets management, and add CI/CD (GitHub Actions) to build/push/deploy on
merge to main.

**Q22. Why nginx for the frontend instead of just Python's http.server?**
`http.server` is fine for local demos but not production-grade — no
gzip, no proper caching headers, single-threaded. nginx is a lightweight,
battle-tested static file server that's the standard choice, and the image
(`nginx:alpine`) is tiny.

## D. System Design / Trade-offs

**Q23. Why SQLite instead of Postgres/MySQL?**
Zero setup — no external DB server needed, so anyone (including an
interviewer) can run the whole project with no extra infrastructure.
SQLAlchemy abstracts the DB layer, so swapping the connection string to
Postgres for production is a one-line change with no other code changes.

**Q24. How would you handle a CSV with 1 million rows?**
Synchronous request/response (as built) wouldn't scale — I'd move batch
processing to a background job (Celery/RQ + Redis, or FastAPI
`BackgroundTasks` for smaller scale), return a job ID immediately, and let
the client poll a `/jobs/{id}` status endpoint or use webhooks/websockets
for completion. I'd also chunk the CSV read (`pd.read_csv(chunksize=...)`)
to avoid loading it all into memory at once.

**Q25. How would you version your ML model?**
Store each trained model with a version tag/timestamp (or use MLflow's
model registry), keep the `metrics.json` per version for comparison, and
let the API's `/health` endpoint report which model version is currently
serving — so you can roll back instantly if a new model underperforms in
production.

**Q26. What would you monitor once this is live?**
Prediction volume and latency, the distribution of `risk_level` outputs
over time (sudden shifts can indicate data drift or an upstream data bug),
and — once ground truth churn events are known weeks later — actual model
performance (precision/recall) in production vs offline test metrics, to
catch model decay.

**Q27. What's the biggest weakness of this project as-is, and how would
you fix it if given more time?**
Be honest in interviews — good answers: (1) synthetic data, not real
customer data — would validate on the real Kaggle Telco dataset or actual
company data; (2) no authentication on the API — would add API keys/JWT
before any real deployment; (3) SQLite won't handle concurrent writes at
scale — would move to Postgres; (4) no automated tests — would add
pytest unit tests for the FastAPI endpoints and the feature engineering
function.

## E. Behavioral — How to Frame This Project

**Q28. "Tell me about a project you're proud of."**
Use the 60-second pitch from Part 5, then be ready to go **deep** on
whichever layer they poke at (ML metrics, API design, or Docker) —
interviewers often pick the thread that matches their own role (a
data scientist will push on metrics; a backend engineer will push on
API/DB design).

**Q29. "What was the hardest part?"**
Good honest answer: keeping the feature engineering *identical* between
training (`ml/train.py`) and serving (`model_utils.py`) — this
training/serving skew is a very real, very common production ML bug, and
explicitly calling it out shows production maturity, not just modeling
skill.

**Q30. "What would you do differently next time?"**
Write the feature engineering function *once*, import it in both
`train.py` and `model_utils.py` (rather than duplicating the logic) to
make skew impossible by construction — a good example of showing
self-critical engineering judgment.

---

# PART 7 — QUICK REFERENCE: SAMPLE cURL COMMANDS

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5,
    "TotalCharges": 450.0
  }'

# Batch CSV upload
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@../data/telecom_churn.csv"

# List prediction records
curl "http://localhost:8000/customers/?limit=10"

# Delete a record
curl -X DELETE http://localhost:8000/customers/1
```

---

# PART 8 — SUGGESTED ENHANCEMENTS (mention as "next steps" in interviews)

1. **SHAP explainability** — add per-prediction feature importance, return
   it alongside `risk_level` (huge differentiator, shows you understand
   interpretability, not just black-box scoring).
2. **Auth** — API key or JWT-based auth on all endpoints.
3. **Automated tests** — `pytest` + `TestClient` for the FastAPI app,
   unit tests for `engineer_features()` and `predict_single()`.
4. **CI/CD** — GitHub Actions: lint → test → build Docker image → push.
5. **Model registry** — MLflow to track experiments and versions.
6. **Frontend polish** — charts (e.g. Recharts) showing risk distribution,
   proper React build (Vite) instead of CDN for production.
7. **Rate limiting** — protect `/predict/batch` from abuse.

---

*This guide, plus the full working codebase (data generator, training
script, FastAPI backend, Dockerfiles, React frontend), gives you a
complete, runnable, and defensible project for AI/ML interviews. Practice
explaining Part 3 (the code) out loud, and know Part 6 (Q&A) cold — that
combination is what separates "I followed a tutorial" from "I can build
this at your company."*
