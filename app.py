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
LOGO_PATH = Path("assets/logo.png")

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")

# =========================
# Styling
# =========================
st.markdown(
    """
<style>
.block-container { max-width: 1200px; padding-top: 1rem; }
.panel {
  background:#fff; border-radius:16px; padding:18px;
  border:1px solid rgba(0,0,0,0.08);
}
.badge-high { color:#b91c1c; font-weight:700; }
.badge-mid  { color:#92400e; font-weight:700; }
.badge-low  { color:#047857; font-weight:700; }
</style>
""",
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption("Memory-safe churn prediction dashboard")

# =========================
# Database
# =========================
def db_connect():
    DB_PATH.parent.mkdir(exist_ok=True)
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
                datetime.utcnow().isoformat(),
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
            "SELECT ts, mode, churn_prob, risk_label FROM scoring_history ORDER BY ts DESC LIMIT ?",
            conn,
            params=(limit,),
        )

db_init()

# =========================
# Cached Loaders (CRITICAL)
# =========================
@st.cache_resource
def load_artifacts():
    import pickle

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    model = model_data["model"]
    features = model_data.get("feature_names") or model_data.get("features")
    return model, features, encoders

@st.cache_data
def load_dataset(path: str):
    df = pd.read_csv(path)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"].astype(str).str.strip()
            .replace({"": "0.0", " ": "0.0"})
            .astype(float)
        )

    if "Churn" in df.columns:
        df["Churn_bin"] = df["Churn"].replace({"Yes": 1, "No": 0})
    else:
        df["Churn_bin"] = 0

    return df

# =========================
# Helpers
# =========================
def encode_input(df, features, encoders):
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].where(df[col].isin(le.classes_), le.classes_[0])
            df[col] = le.transform(df[col])
    return df[features]

def risk_label(prob, threshold):
    if prob >= threshold + 0.15:
        return "High Risk", "badge-high"
    if prob >= threshold:
        return "At Risk", "badge-mid"
    return "Low Risk", "badge-low"

# =========================
# Sidebar
# =========================
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Single Score", "Batch Score", "Insights", "History"],
)

threshold = st.sidebar.slider("Risk Threshold", 0.1, 0.9, 0.5, 0.05)

# =========================
# Load model once
# =========================
try:
    model, features, encoders = load_artifacts()
except Exception as e:
    st.error("Failed to load model")
    st.stop()

# =========================
# Pages
# =========================
if page == "Home":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    hist = db_read()
    st.metric("Total scored", len(hist))
    st.dataframe(hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Single Score":
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    inputs = {}
    for f in features:
        inputs[f] = st.text_input(f)

    if st.button("Score"):
        df = pd.DataFrame([inputs])
        df_enc = encode_input(df, features, encoders)
        prob = float(model.predict_proba(df_enc)[0][1])

        label, css = risk_label(prob, threshold)
        db_insert("single", prob, label, threshold, inputs)

        st.markdown(f"### Result: **{label}**")
        st.write("Churn probability:", round(prob, 3))

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Batch Score":
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    max_rows = st.number_input("Max rows", 1000, 50000, 20000, 1000)
    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file, nrows=int(max_rows))
        df_enc = encode_input(df.copy(), features, encoders)

        probs = model.predict_proba(df_enc)[:, 1]
        df["churn_prob"] = probs
        df["churn_pred"] = (probs >= threshold).astype(int)

        st.dataframe(df.nlargest(50, "churn_prob"), use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode(),
            "scored.csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Insights":
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    if not DATA_PATH.exists():
        st.info("Dataset not included in this deployment.")
    else:
        if st.toggle("Load insights"):
            df = load_dataset(str(DATA_PATH))

            st.subheader("Churn by Contract")
            st.bar_chart(df.groupby("Contract")["Churn_bin"].mean())

            st.subheader("Churn vs Monthly Charges")
            st.line_chart(
                df.groupby(pd.cut(df["MonthlyCharges"], 10))["Churn_bin"].mean()
            )

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "History":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.dataframe(db_read(1000), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
