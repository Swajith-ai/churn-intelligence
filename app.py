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
.small { color: rgba(0,0,0,0.65); font-size: 13px; }
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
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def db_init():
    with db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scoring_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                mode TEXT,
                churn_prob REAL,
                risk_label TEXT,
                threshold REAL,
                payload TEXT
            )
            """
        )
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
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        conn.commit()

def db_read(limit=500):
    with db_connect() as conn:
        df = pd.read_sql(
            "SELECT ts, mode, churn_prob, risk_label, threshold FROM scoring_history ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    return df

db_init()

# =========================
# Cached Loaders
# =========================
@st.cache_resource
def load_artifacts():
    import pickle

    if not MODEL_PATH.exists() or not ENCODERS_PATH.exists():
        raise FileNotFoundError(
            "Missing artifacts. Ensure these exist:\n"
            "- artifacts/customer_churn_model.pkl\n"
            "- artifacts/encoders.pkl"
        )

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    model = model_data.get("model")

    # handle different key names safely
    features = (
        model_data.get("features_names")
        or model_data.get("feature_names")
        or model_data.get("features")
    )

    if model is None:
        raise ValueError(f"'model' key not found in pickle. Keys: {list(model_data.keys())}")

    if features is None:
        raise ValueError(
            "Feature list not found. Expected one of: features_names / feature_names / features. "
            f"Keys: {list(model_data.keys())}"
        )

    return model, list(features), encoders

@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"].astype(str).str.strip()
            .replace({"": "0.0", " ": "0.0"})
        )
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    if "Churn" in df.columns:
        df["Churn_bin"] = df["Churn"].replace({"Yes": 1, "No": 0}).astype(int)
    else:
        df["Churn_bin"] = 0

    return df

# =========================
# Helpers
# =========================
def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"].astype(str).str.strip()
            .replace({"": "0.0", " ": "0.0"})
        )
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df

def encode_input(df: pd.DataFrame, features, encoders) -> pd.DataFrame:
    for col, le in encoders.items():
        if col in df.columns:
            s = df[col].astype(str)
            known = set(le.classes_.tolist())
            fallback = le.classes_[0]
            s = s.apply(lambda v: v if v in known else fallback)
            df[col] = le.transform(s)
    return df[features]

def risk_label(prob: float, threshold: float):
    if prob >= threshold + 0.15:
        return "High Risk"
    if prob >= threshold:
        return "At Risk"
    return "Low Risk"

def score_one_row_csv(model, features, encoders, text: str) -> float:
    df = pd.read_csv(StringIO(text))
    if df.shape[0] != 1:
        raise ValueError("Paste exactly ONE data row under the header.")

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = clean_total_charges(df)
    df_enc = encode_input(df.copy(), features, encoders)
    prob = float(model.predict_proba(df_enc)[0][1])
    return prob

# =========================
# Sidebar
# =========================
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)

st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Go to", ["Home", "Single Score", "Batch Score", "Insights", "History"])

st.sidebar.divider()
threshold = st.sidebar.slider("Risk Threshold", 0.10, 0.90, 0.50, 0.05)

# =========================
# Load model once
# =========================
try:
    model, features, encoders = load_artifacts()
except Exception as e:
    st.error("Failed to load model artifacts.")
    st.code(str(e))
    st.stop()

# =========================
# Pages
# =========================
if page == "Home":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    hist = db_read(500)
    st.metric("Total scored", len(hist))
    if hist.empty:
        st.info("No scoring history yet.")
    else:
        st.dataframe(hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Single Score":
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Manual Input", "Paste 1-row CSV"])

    # ---- Manual input ----
    with tab1:
        st.caption("Fill values manually, then click Score.")
        inputs = {}
        for f in features:
            inputs[f] = st.text_input(f)

        if st.button("Score (Manual)"):
            df = pd.DataFrame([inputs])
            df = clean_total_charges(df)

            try:
                df_enc = encode_input(df.copy(), features, encoders)
                prob = float(model.predict_proba(df_enc)[0][1])
                label = risk_label(prob, threshold)
                db_insert("single_manual", prob, label, threshold, inputs)

                st.success(f"Result: {label}")
                st.write("Churn probability:", round(prob, 3))
            except Exception as e:
                st.error("Scoring failed.")
                st.code(str(e))

    # ---- Paste 1-row CSV ----
    with tab2:
        st.caption("Paste a CSV with header + exactly 1 data row.")

        sample = (
            "gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,"
            "OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,"
            "Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
            "Female,0,No,No,4,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,"
            "Yes,Electronic check,94.65,378.60"
        )

        text = st.text_area("Paste 1-row CSV", value=sample, height=170)

        if st.button("Score (Paste CSV)"):
            try:
                prob = score_one_row_csv(model, features, encoders, text)
                label = risk_label(prob, threshold)
                db_insert("single_csv", prob, label, threshold, {"csv": text[:5000]})

                st.success(f"Result: {label}")
                st.write("Churn probability:", round(prob, 3))
            except Exception as e:
                st.error("Invalid CSV.")
                st.code(str(e))

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Batch Score":
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.caption("Upload a CSV with required columns. Memory-safe: limits rows and avoids heavy sorts.")
    max_rows = st.number_input("Max rows", 1000, 50000, 20000, 1000)
    up = st.file_uploader("Upload CSV", type=["csv"])

    if up is None:
        st.info("Upload a CSV to score.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        try:
            df = pd.read_csv(up, nrows=int(max_rows))
        except Exception as e:
            st.error("Could not read CSV.")
            st.code(str(e))
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        df = clean_total_charges(df)

        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error("CSV missing required columns:")
            st.code(str(missing))
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        try:
            df_enc = encode_input(df.copy(), features, encoders)
            probs = model.predict_proba(df_enc)[:, 1].astype(float)

            df["churn_prob"] = probs
            df["churn_pred"] = (probs >= threshold).astype(int)

            flagged = int((df["churn_pred"] == 1).sum())
            st.write(f"<span class='small'>Scored: {len(df)} | Flagged: {flagged}</span>", unsafe_allow_html=True)

            st.subheader("Top 50 by churn probability")
            st.dataframe(df.nlargest(50, "churn_prob"), use_container_width=True)

            st.download_button(
                "⬇️ Download scored CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="churn_scored.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error("Batch scoring failed.")
            st.code(str(e))

        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Insights":
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    if not DATA_PATH.exists():
        st.info("Dataset not included in this deployment (`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`).")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        load = st.toggle("Load insights dataset", value=False)
        if not load:
            st.info("Toggle ON to load dataset and render charts.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            df = load_dataset(str(DATA_PATH))

            st.subheader("Churn rate by Contract")
            if "Contract" in df.columns:
                t = df.groupby("Contract")["Churn_bin"].mean().sort_values(ascending=False)
                st.bar_chart(t)
            else:
                st.info("Contract column missing.")

            st.subheader("Churn rate vs MonthlyCharges (binned)")
            if "MonthlyCharges" in df.columns:
                bins = np.linspace(df["MonthlyCharges"].min(), df["MonthlyCharges"].max(), 12)
                df["mc_bin"] = pd.cut(df["MonthlyCharges"], bins=bins, include_lowest=True)
                t2 = df.groupby("mc_bin")["Churn_bin"].mean()
                st.line_chart(t2)
            else:
                st.info("MonthlyCharges column missing.")

            st.markdown("</div>", unsafe_allow_html=True)

elif page == "History":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    hist = db_read(1000)
    if hist.empty:
        st.info("No history yet.")
    else:
        st.dataframe(hist, use_container_width=True)
        st.download_button(
            "⬇️ Download history CSV",
            data=hist.to_csv(index=False).encode("utf-8"),
            file_name="scoring_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
