# 📊 Churn Intelligence

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://churn-intelligence-dnahdem2j87nspxabz7ewg.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#-license)

**Churn Intelligence** is a memory-safe, production-ready customer churn prediction dashboard built with **Streamlit** and **scikit-learn**.  
It helps teams move from reactive churn analysis to **early, actionable churn detection**.

✅ Single scoring (manual + paste 1-row CSV)  
✅ Batch scoring (CSV upload)  
✅ Optional insights (toggle-on only)  
✅ Scoring history (SQLite)

---

## 🚀 Live App

Open the app here:  
**Streamlit Cloud:** https://churn-intelligence-dnahdem2j87nspxabz7ewg.streamlit.app/

---

## ✨ Features

### 🔹 Single Customer Scoring
- Manual input scoring (one customer at a time)
- **Paste 1-row CSV** scoring for fast testing / integrations
- Shows churn probability + risk label

### 🔹 Batch Scoring (CSV Upload)
- Upload a CSV and score up to **20,000 rows safely** (Streamlit Cloud friendly)
- Appends:
  - `churn_prob`
  - `churn_pred`
- Download the scored CSV instantly

### 🔹 Insights (Optional)
- Churn rate by **Contract**
- Churn vs **MonthlyCharges (binned)**
- Dataset loads **only when toggled ON** to avoid memory spikes

### 🔹 History
- Stores scoring events using lightweight **SQLite**
- View history inside the app

---

## 🛡️ Memory-Safe by Design

Optimized for Streamlit Cloud resource limits:

- `@st.cache_resource` for **model + encoders**
- `@st.cache_data` for **dataset loading**
- Avoids loading big datasets at startup
- Avoids heavy sorts (uses `nlargest`)
- `.venv`, cache folders, and local artifacts should not be committed

---

## 📂 Project Structure

```txt
churn-intelligence/
├── app.py
├── requirements.txt
├── artifacts/
│   ├── customer_churn_model.pkl
│   └── encoders.pkl
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # optional (for Insights)
│   └── history.db                             # created at runtime
├── assets/
│   └── logo.png
├── .streamlit/
│   └── config.toml
└── README.md
🧠 Model Artifacts Required
The app expects:

artifacts/customer_churn_model.pkl
artifacts/encoders.pkl
The model pickle must contain:

model

feature list under one of:

features_names

feature_names

features

▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py
Then open:

http://localhost:8501
📄 Example: Paste 1-Row CSV
Use this inside Single Score → Paste 1-row CSV:

gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
Female,0,No,No,4,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,94.65,378.60
🧪 Tech Stack
Python

Streamlit

scikit-learn

pandas, numpy

SQLite

🧭 Roadmap (Next Improvements)
Planned upgrades:

✅ Replace manual text inputs with dropdowns / numeric inputs (better UX + fewer input errors)

✅ Add explanations per prediction (risk drivers / simple rationale)

✅ Add PDF report export (single customer + batch summary)

✅ Add optional authentication / login

✅ Add retention playbooks + messaging templates per risk segment

✅ Add better validation for uploaded CSVs + column mapping

If you want any of these next, open an issue or message me.

🤝 Contributing
Contributions are welcome!

Fork the repo

Create a feature branch: git checkout -b feature/my-change

Commit: git commit -m "Add feature"

Push: git push origin feature/my-change

Open a Pull Request

📜 License
MIT License.

🙌 Author
Built by Swajith
If you found this useful, please ⭐ the repo!