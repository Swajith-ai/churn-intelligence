# Churn Intelligence — Customer Churn Prediction Dashboard (Streamlit)

**Churn Intelligence** is a product-style **Streamlit web dashboard** that predicts **customer churn probability** using a **pre-trained ML model**.  
It supports **single customer scoring**, **paste-one scoring**, **CSV batch scoring**, **dataset insights**, a **retention playbook**, and **local scoring history**.

---

## Live Demo
https://churn-intelligence-dnahdem2j87nspxabz7ewg.streamlit.app/

---

## What this project does

### 1) Loads dataset + trained artifacts
The app expects:
- **Dataset**: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Model artifacts**:
  - `artifacts/customer_churn_model.pkl` (trained model + feature order)
  - `artifacts/encoders.pkl` (LabelEncoders for categorical columns)

These artifacts ensure that inputs are encoded and ordered exactly as required by the model during prediction.

### 2) Predicts churn in 3 ways
- **Manual input** (single customer form)
- **Paste one customer** (1-row CSV with header)
- **Batch scoring** (upload CSV → adds results → download scored CSV)

### 3) Shows decision + guidance
After scoring, the UI shows:
- Churn probability
- Risk label (Low Risk / At Risk / High Risk) based on a threshold slider
- Recommended SLA (how fast to contact)
- “Signals to check” (rule-based guidance)
- Retention playbook + ready-to-use message template (downloadable)

### 4) Insights dashboard
The Insights page visualizes churn trends from the dataset, such as:
- Churn vs MonthlyCharges (binned)
- Churn by tenure stage
- Churn by contract type

### 5) Scoring history
The app stores scoring activity locally in:
- `data/history.db`

So you can revisit recent scoring and download history.

---

## Key Features

### ✅ Single Customer Scoring
From **Score (Single)** page:
- Manual input form (dropdown + numeric inputs)
- Paste one customer (1-row CSV with header)

Outputs include:
- Churn probability (0–1)
- Risk badge (Low Risk / At Risk / High Risk)
- Likelihood label
- Recommended SLA
- Signals to check (rule-based)
- Retention playbook + message template + download message as `.txt`

### ✅ Batch Scoring (CSV Upload)
Upload a CSV dataset and get:
- `churn_prob` column (predicted churn probability)
- `churn_pred` column (0/1 based on threshold)
- Probability distribution chart
- Download scored CSV

### ✅ Insights Dashboard
Charts generated from the dataset:
- Churn rate vs MonthlyCharges
- Churn rate by tenure
- Churn by contract type

---

## Risk Policy (Threshold)
A sidebar slider controls the risk threshold:

- If `churn_prob >= threshold` → **At Risk**
- If `churn_prob >= threshold + 0.15` → **High Risk**
- Otherwise → **Low Risk**

This helps you control retention workload:
- Higher threshold → fewer flagged customers
- Lower threshold → more flagged customers

---

## Tech Stack
- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn (model inference)
- Plotly (charts)

---

## Project Structure (Recommended)

churn-intelligence/
│── app.py
│── requirements.txt
│── README.md
│
├── .streamlit/
│ └── config.toml
│
├── artifacts/
│ ├── customer_churn_model.pkl
│ └── encoders.pkl
│
├── data/
│ ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│ └── history.db
│
└── src/
├── preprocess.py
├── train.py
└── predict.py


> Optional logo  
> If your app is configured to show a logo, place it here:
> `assets/logo.png`

---

## Setup & Run (Windows)

### 1) Open terminal in the project folder
```powershell
cd C:\Users\swaji\Desktop\project2
2) Create & activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
3) Install dependencies
python -m pip install -r requirements.txt
4) Run the app
python -m streamlit run app.py
Input Formats
A) Paste Single Customer (1-row CSV)
Paste exactly one row with header.

Example:

gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
Female,0,No,No,4,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,94.65,378.60
B) Batch CSV Upload
Your CSV must contain the same feature columns used during training.
The app will add:

churn_prob

churn_pred

If you see “missing required columns”, your CSV headers don’t match the model’s expected features.

Model Artifacts (How prediction works)
The app uses:

encoders.pkl to convert categorical values → numeric values

the model’s saved feature list to ensure correct column order

predict_proba() to compute churn probability

Training Pipeline (Optional — src/)
If you want to retrain the model and regenerate artifacts:

src/preprocess.py
Loads and cleans dataset

Drops customerID

Fixes blanks in TotalCharges and converts to float

Encodes categorical columns

Saves: artifacts/encoders.pkl

src/train.py
Splits train/test

Applies SMOTE to handle class imbalance

Trains and compares models (e.g., Decision Tree / Random Forest)

Saves: artifacts/customer_churn_model.pkl

src/predict.py
Example script to run prediction using saved artifacts

If training requires SMOTE and it’s missing:

python -m pip install imbalanced-learn
Troubleshooting
“streamlit is not recognized”
Use:

python -m streamlit run app.py
“Model artifacts not found”
Confirm these exist in your repo and deployment:

artifacts/customer_churn_model.pkl

artifacts/encoders.pkl

“Dataset not found”
Confirm:

data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Author
Swajith S S