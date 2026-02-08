import json
import sqlite3
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
APP_TITLE = "Churn Intelligence"

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "customer_churn_model.pkl"
ENCODERS_PATH = ARTIFACTS_DIR / "encoders.pkl"

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
DB_PATH = Path("data/history.db")

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
st.title(APP_TITLE)
st.caption("Memory-safe churn prediction dashboard")

# =========================
# Database
# =========================
def db_connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def db_init():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scoring_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                mode TEXT,
                churn_prob REAL,
                risk_label TEXT,
                threshold REAL,
                payload TEXT
            )
        """)
        conn.commit()

def db_insert(mode, prob, label, threshold, payload):
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO scoring_history VALUES (NULL,?,?,?,?,?,?)",
            (
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                mode,
                float(prob),
                label,
                float(threshold),
                json.dumps(payload),
            ),
        )
        conn.commit()

def db_read(limit=500):
    with db_connect() as conn:
        return pd.read_sql(
            "SELECT ts, mode, churn_prob, risk_label FROM scoring_history ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )

db_init()

# =========================
# Load model + encoders
# =========================
@st.cache_resource
def load_artifacts():
    import pickle

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    model = model_data.get("model")
    features = (
        model_data.get("features_names")
        or model_data.get("feature_names")
        or model_data.get("features")
    )

    if model is None or features is None:
        raise ValueError(f"Invalid model pickle keys: {list(model_data.keys())}")

    return model, list(features), encoders

@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"].astype(str)
            .str.strip()
            .replace({"": "0.0", " ": "0.0"})
        )
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    if "Churn" in df.columns:
        df["Churn_bin"] = df["Churn"].replace({"Yes": 1, "No": 0})
    else:
        df["Churn_bin"] = 0

    return df

# =========================
# Helpers
# =========================
def encode(df, features, encoders):
    for col, le in encoders.items():
        if col in df.columns:
            s = df[col].astype(str)
            known = set(le.classes_)
            s = s.apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(s)
    return df[features]

def risk_label(prob, threshold):
    if prob >= threshold + 0.15:
        return "High Risk"
    if prob >= threshold:
        return "At Risk"
    return "Low Risk"

def score_single_csv(model, features, encoders, text):
    df = pd.read_csv(StringIO(text))
    if len(df) != 1:
        raise ValueError("Paste exactly ONE row")
    df_enc = encode(df.copy(), features, encoders)
    return float(model.predict_proba(df_enc)[0][1])

# =========================
# Sidebar
# =========================
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Single Score", "Batch Score", "Insights", "History"]
)

threshold = st.sidebar.slider("Risk Threshold", 0.10, 0.90, 0.50, 0.05)

# =========================
# Load model
# =========================
model, features, encoders = load_artifacts()

# =========================
# Pages
# =========================
if page == "Home":
    hist = db_read()
    st.metric("Total scored", len(hist))
    st.dataframe(hist, use_container_width=True)

elif page == "Single Score":
    tab1, tab2 = st.tabs(["Manual Input", "Paste 1-row CSV"])

    with tab1:
        inputs = {f: st.text_input(f) for f in features}
        if st.button("Score"):
            df = pd.DataFrame([inputs])
            prob = float(model.predict_proba(encode(df, features, encoders))[0][1])
            label = risk_label(prob, threshold)
            db_insert("single_manual", prob, label, threshold, inputs)
            st.success(f"{label} | Probability: {prob:.3f}")

    with tab2:
        sample = (
            "gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,"
            "OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,"
            "Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
            "Female,0,No,No,4,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,"
            "Yes,Electronic check,94.65,378.60"
        )
        text = st.text_area("Paste 1-row CSV", value=sample, height=160)
        if st.button("Score CSV"):
            prob = score_single_csv(model, features, encoders, text)
            label = risk_label(prob, threshold)
            db_insert("single_csv", prob, label, threshold, {"csv": text})
            st.success(f"{label} | Probability: {prob:.3f}")

elif page == "Batch Score":
    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        df = pd.read_csv(up, nrows=20000)
        df_enc = encode(df.copy(), features, encoders)
        probs = model.predict_proba(df_enc)[:, 1]
        df["churn_prob"] = probs
        df["churn_pred"] = (probs >= threshold).astype(int)
        st.dataframe(df.nlargest(50, "churn_prob"), use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "scored.csv")

elif page == "Insights":
    if DATA_PATH.exists():
        if st.toggle("Load insights"):
            df = load_dataset(str(DATA_PATH))

            st.subheader("Churn rate by Contract")
            st.bar_chart(df.groupby("Contract")["Churn_bin"].mean())

            st.subheader("Churn rate vs MonthlyCharges")
            bins = np.linspace(df["MonthlyCharges"].min(), df["MonthlyCharges"].max(), 12)
            df["mc_bin"] = pd.cut(df["MonthlyCharges"], bins=bins, include_lowest=True)
            t2 = df.groupby("mc_bin", observed=True)["Churn_bin"].mean()
            t2.index = t2.index.astype(str)   # 🔑 CRASH FIX
            st.line_chart(t2)
    else:
        st.info("Dataset not available")

elif page == "History":
    st.dataframe(db_read(1000), use_container_width=True)
