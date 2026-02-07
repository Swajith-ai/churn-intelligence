# Customer Churn Prediction Dashboard (Streamlit)

This project implements a **Customer Churn Prediction System** using a trained Machine Learning model and a modern **Streamlit web dashboard**.

The application is designed for real usage:
- Predict churn probability for a **single customer**
- Predict churn probability for **multiple customers (CSV upload)**
- Show churn insights and trends from the dataset
- Provide an **action playbook** and ready-to-use retention message templates

---

## Problem Statement

In subscription-based businesses (example: telecom), customers may leave due to pricing, service issues, poor support, or better competitors.

Instead of reacting after churn happens, this project helps estimate **churn probability** early so a retention team can:
- prioritize outreach
- fix friction points
- improve customer experience and retention

---

## Solution Approach

This solution follows a practical ML deployment workflow:

1. Load a churn dataset to understand customer attributes and churn behavior
2. Use a trained model for churn prediction (saved as artifacts)
3. Encode categorical values consistently using saved encoders
4. Build a Streamlit dashboard for:
   - manual input scoring
   - paste-based single-customer scoring
   - CSV batch scoring and export
   - insights charts
   - retention playbooks

---

## Key Features

### Prediction
- **Churn probability prediction** (`predict_proba`)
- Risk labeling using an adjustable **threshold slider**
  - **Low Risk**
  - **At Risk**
  - **High Risk**

### Web Dashboard (Streamlit)
Supports:
- Manual customer input (form)
- Paste one customer as **1-row CSV**
- Upload dataset as CSV for **batch scoring**
- Download scored CSV results
- Insights dashboard (trends and comparisons)
- Retention playbook with message templates

---

## Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn (model inference)
- Plotly (interactive charts)
- Pickle (loading saved artifacts)

---

## Dataset

This project expects the Telco Customer Churn dataset file:

- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset is used for:
- Dropdown options (categorical fields like Contract, InternetService, etc.)
- Insights charts (churn trends by tenure, charges, contract type)

---

## Model Artifacts

This project expects the trained artifacts inside:

- `artifacts/customer_churn_model.pkl`
- `artifacts/encoders.pkl`

These are required to:
- Load the trained churn model
- Encode categorical values consistently
- Ensure feature order matches the model’s training feature order

---

## Project Structure

project2/
│── app.py
│── requirements.txt
│── README.md
│
├── artifacts/
│ ├── customer_churn_model.pkl
│ └── encoders.pkl
│
└── data/
└── WA_Fn-UseC_-Telco-Customer-Churn.csv


---

## Run the User Interface (Streamlit)

### 1) Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
2) Install dependencies
python -m pip install -r requirements.txt
3) Run the application
python -m streamlit run app.py
Open in browser (if not automatic):

http://localhost:8501

CSV Input Options
A) Paste single customer (1-row CSV)
Paste exactly one row with header, for example:

gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
Female,0,No,No,4,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,94.65,378.60
B) Batch upload CSV
Upload a CSV containing the same feature columns expected by the model.
The app also provides a template CSV download in the UI to help create valid input files.

Risk Threshold (How Customers Are Flagged)
The sidebar contains a threshold slider:

If churn_prob >= threshold → customer is flagged At Risk

If churn_prob >= threshold + 0.15 → customer is flagged High Risk

Otherwise → Low Risk

This lets you control retention workload:

Higher threshold → fewer flagged customers

Lower threshold → more flagged customers

Troubleshooting
1) Dataset not found
Make sure this exists:

data/WA_Fn-UseC_-Telco-Customer-Churn.csv

2) Artifacts not found
Make sure these exist:

artifacts/customer_churn_model.pkl

artifacts/encoders.pkl

3) Streamlit command not recognized
Use:

python -m streamlit run app.py
4) Missing required columns in CSV upload
Use the template CSV download button inside the app and fill the values correctly.

Author
Swajith S S