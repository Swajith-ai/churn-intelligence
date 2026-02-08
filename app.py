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

# IMPORTANT: We do NOT load the big dataset at startup on Streamlit Cloud.
# Insights will be optional and will only run if the dataset exists.
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

DB_PATH = Path("data") / "history.db"
LOGO_PATH = Path("assets/logo.png")


st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")


# =========================
# Lightweight Styling
# =========================
st.markdown(
    """
<style>
.block-container { max-width: 1200px; padding-top: 1.1rem; padding-bottom: 2rem; }
.hero {
  padding: 16px 18px; border-radius: 18px;
  background: linear-gradient(135deg, rgba(79,70,229,0.10), rgba(14,165,233,0.05));
  border: 1px solid rgba(2,6,23,0.08); margin-bottom: 12px;
}
.hero h1 { margin:0; font-size: 26px; color:#0F172A; letter-spacing: -0.2px; }
.hero p { margin:6px 0 0 0; font-size:14px; color: rgba(15,23,42,0.72); }
.panel {
  background:#fff; border-radius:18px; padding:18px;
  border:1px solid rgba(2,6,23,0.08); box-shadow: 0 10px 26px rgba(2,6,23,0.06);
}
.badge { display:inline-flex; align-items:center; gap:8px; padding:8px 14px; border-radius:999px;
  font-weight:800; font-size:13px; border:1px solid rgba(2,6,23,0.12); }
.badge-high { background: rgba(239,68,68,0.12); color:#B91C1C; border-color: rgba(239,68,68,0.32); }
.badge-mid  { background: rgba(245,158,11,0.14); color:#92400E; border-color: rgba(245,158,11,0.30); }
.badge-low  { background: rgba(34,197,94,0.12); color:#047857; border-color: rgba(34,197,94,0.32); }
.small { color: rgba(15,23,42,0.70); font-size: 13px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="hero">
  <h1>{APP_TITLE}</h1>
  <p>Memory-safe churn scoring dashboard — single scoring, batch scoring, and playbooks.</p>
</div>
""",
    unsafe_allow_html=True,
)


# =========================
# DB (History)
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
                ts TEXT NOT NULL,
                mode TEXT NOT NULL,
                churn_prob REAL NOT NULL,
                risk_label TEXT NOT NULL,
                threshold REAL NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def db_insert(mode: str, churn_prob: float, risk_label: str, threshold: float, payload: dict):
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO scoring_history (ts, mode, churn_prob, risk_label, threshold, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                mode,
                float(churn_prob),
                risk_label,
                float(threshold),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        conn.commit()


def db_read(limit: int = 500) -> pd.DataFrame:
    with db_connect() as conn:
        df = pd.read_sql_query(
            "SELECT ts, mode, churn_prob, risk_label, threshold, payload_json FROM scoring_history ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    if df.empty:
        return df
    df["payload_preview"] = df["payload_json"].apply(lambda s: str(s)[:140] + ("..." if len(str(s)) > 140 else ""))
    return df.drop(columns=["payload_json"])


db_init()


# =========================
# Load artifacts (cached resource)
# =========================
@st.cache_resource
def load_artifacts():
    import pickle

    if not MODEL_PATH.exists() or not ENCODERS_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Required:\n"
            "- artifacts/customer_churn_model.pkl\n"
            "- artifacts/encoders.pkl"
        )

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    model = model_data.get("model")
    feature_names = model_data.get("features_names") or model_data.get("feature_names") or model_data.get("features")

    if model is None or feature_names is None:
        raise ValueError("Model file must contain: 'model' and feature list (features_names/feature_names/features).")

    return model, list(feature_names), encoders


def safe_label_transform(le, values: pd.Series) -> np.ndarray:
    known = set(le.classes_.tolist())
    fallback = le.classes_[0]
    cleaned = values.astype(str).apply(lambda v: v if v in known else fallback)
    return le.transform(cleaned)


def encode_and_order(df: pd.DataFrame, feature_names, encoders) -> pd.DataFrame:
    # Encode categorical columns using encoders
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = safe_label_transform(le, df[col])

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[feature_names]


def predict_prob(model, feature_names, encoders, input_dict: dict) -> float:
    df = pd.DataFrame([input_dict])

    # TotalCharges cleanup if present
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"].astype(str).str.strip().replace({"": "0.0", " ": "0.0"}).astype(float)
        )

    df_enc = encode_and_order(df, feature_names, encoders)
    prob = float(model.predict_proba(df_enc)[0][1])
    return prob


def likelihood_label(prob: float) -> str:
    if prob >= 0.80:
        return "Very likely"
    if prob >= 0.60:
        return "Likely"
    if prob >= 0.40:
        return "Uncertain"
    if prob >= 0.20:
        return "Unlikely"
    return "Very unlikely"


def risk_badge(prob: float, threshold: float):
    if prob >= threshold + 0.15:
        return "High Risk", "badge badge-high"
    if prob >= threshold:
        return "At Risk", "badge badge-mid"
    return "Low Risk", "badge badge-low"


def sla_recommendation(prob: float, threshold: float) -> str:
    if prob >= threshold + 0.15:
        return "contact within 2–6 hours"
    if prob >= threshold:
        return "contact within 24–48 hours"
    return "normal monitoring"


def build_signals(inp: dict) -> list[str]:
    signals = []
    contract = str(inp.get("Contract", "")).lower()
    tenure = int(float(inp.get("tenure", 0) or 0))
    monthly = float(inp.get("MonthlyCharges", 0) or 0)
    payment = str(inp.get("PaymentMethod", "")).lower()
    internet = str(inp.get("InternetService", "")).lower()
    tech = str(inp.get("TechSupport", "")).lower()
    sec = str(inp.get("OnlineSecurity", "")).lower()

    if "month" in contract:
        signals.append("Month-to-month contract → low commitment; easier to switch.")
    if tenure <= 6:
        signals.append("Tenure ≤ 6 months → higher early churn risk.")
    if monthly >= 80:
        signals.append("High MonthlyCharges → friction hurts more.")
    if "electronic check" in payment:
        signals.append("Electronic check → often correlates with higher churn.")
    if "fiber" in internet:
        signals.append("Fiber customers can be more price-sensitive (depends on quality).")
    if tech in ("no", "false", "0"):
        signals.append("No TechSupport → support friction can increase churn.")
    if sec in ("no", "false", "0"):
        signals.append("No OnlineSecurity → bundles/add-ons may improve stickiness.")

    if not signals:
        signals.append("No strong red flags detected from basic profile signals.")
    return signals


def playbook_text(level: str) -> dict:
    if level == "High Risk":
        return {
            "title": "Retention Playbook (High Risk)",
            "steps": [
                "Contact within 2–6 hours if possible.",
                "Identify the real issue: billing, speed/outages, plan fit, support experience.",
                "Fix friction first, then reinforce value (optional discount/upgrade).",
                "If month-to-month: suggest longer plan only after pain points are fixed.",
            ],
            "template": (
                "Hi {name},\n\n"
                "Just checking in to make sure your service is running smoothly. "
                "If you’ve faced any issues (billing, speed, outages, plan fit), I can help resolve them today.\n\n"
                "If it helps, we can also review a plan option that better matches your usage.\n\n"
                "Thanks,\nSupport Team"
            ),
        }
    if level == "At Risk":
        return {
            "title": "Retention Playbook (At Risk)",
            "steps": [
                "Contact within 24–48 hours.",
                "Check plan fit and billing clarity.",
                "Offer value reinforcement (trial add-on / plan optimization / loyalty perk).",
            ],
            "template": (
                "Hi {name},\n\n"
                "Quick check-in — how has your service been recently? "
                "If anything feels off (billing, support, speed), I can help.\n\n"
                "We can also review your plan to ensure you’re getting the best value.\n\n"
                "Thanks,\nSupport Team"
            ),
        }
    return {
        "title": "Customer Care (Low Risk)",
        "steps": [
            "No urgent action needed — monitor normally.",
            "Proactively share a value tip or usage suggestion.",
        ],
        "template": (
            "Hi {name},\n\n"
            "Thanks for being with us. If you need anything, we’re here to help.\n\n"
            "Tip: Check your account for add-on features that may improve your experience.\n\n"
            "Thanks,\nSupport Team"
        ),
    }


# =========================
# Sidebar
# =========================
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)

st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Score (Single)", "Batch Scoring", "Insights (Optional)", "History"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown("## ⚙️ Risk Policy")
threshold = st.sidebar.slider("High-risk threshold", 0.05, 0.95, 0.50, 0.01)
st.sidebar.caption("Probability ≥ threshold → flagged")


# =========================
# Load model + encoders only (no dataset at startup)
# =========================
try:
    model, feature_names, encoders = load_artifacts()
except Exception as e:
    st.error("Failed to load model artifacts.")
    st.code(str(e))
    st.stop()

# Build dropdown options from encoder classes (no dataset needed)
def opt(col: str):
    le = encoders.get(col)
    if le is None:
        return []
    return list(le.classes_)


# =========================
# Pages
# =========================
def render_home():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Overview")
    st.write("Use the sidebar to score customers or upload a CSV for batch scoring.")

    hist = db_read(500)
    if hist.empty:
        st.info("No scoring history yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    total = len(hist)
    flagged = int(hist["risk_label"].isin(["At Risk", "High Risk"]).sum())
    high = int((hist["risk_label"] == "High Risk").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total scored", total)
    c2.metric("Flagged", flagged)
    c3.metric("High Risk", high)

    st.markdown("#### Recent scoring")
    st.dataframe(hist.head(30), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def score_manual():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Score a customer — Manual input")

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("gender", opt("gender"))
        SeniorCitizen = st.number_input("SeniorCitizen (0/1)", 0, 1, 0, 1)
        Partner = st.selectbox("Partner", opt("Partner"))
        Dependents = st.selectbox("Dependents", opt("Dependents"))
    with c2:
        Contract = st.selectbox("Contract", opt("Contract"))
        tenure = st.number_input("tenure (months)", 0, 1000, 12, 1)
        PaperlessBilling = st.selectbox("PaperlessBilling", opt("PaperlessBilling"))
        PaymentMethod = st.selectbox("PaymentMethod", opt("PaymentMethod"))
    with c3:
        InternetService = st.selectbox("InternetService", opt("InternetService"))
        PhoneService = st.selectbox("PhoneService", opt("PhoneService"))
        MultipleLines = st.selectbox("MultipleLines", opt("MultipleLines"))
        MonthlyCharges = st.number_input("MonthlyCharges", 0.0, 1000.0, 70.0, 1.0)

    TotalCharges = st.number_input("TotalCharges", 0.0, 1_000_000.0, 800.0, 10.0)

    with st.expander("Advanced: add-on services"):
        a1, a2, a3 = st.columns(3)
        with a1:
            OnlineSecurity = st.selectbox("OnlineSecurity", opt("OnlineSecurity"))
            OnlineBackup = st.selectbox("OnlineBackup", opt("OnlineBackup"))
        with a2:
            DeviceProtection = st.selectbox("DeviceProtection", opt("DeviceProtection"))
            TechSupport = st.selectbox("TechSupport", opt("TechSupport"))
        with a3:
            StreamingTV = st.selectbox("StreamingTV", opt("StreamingTV"))
            StreamingMovies = st.selectbox("StreamingMovies", opt("StreamingMovies"))

    run = st.button("Score customer", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not run:
        return

    inp = {
        "gender": gender,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    prob = predict_prob(model, feature_names, encoders, inp)
    label, css = risk_badge(prob, threshold)
    sla = sla_recommendation(prob, threshold)
    signals = build_signals(inp)

    db_insert("single_manual", prob, label, threshold, inp)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f'<span class="{css}">{label}</span>', unsafe_allow_html=True)
    st.markdown("### Decision Summary")
    st.write(f"**Churn probability:** `{prob:.3f}`")
    st.write(f"**Likelihood:** `{likelihood_label(prob)}`")
    st.write(f"**Policy threshold:** `{threshold:.2f}`")
    st.write(f"**Recommended SLA:** `{sla}`")

    st.markdown("#### Signals to check")
    for s in signals:
        st.write(f"- {s}")

    pb = playbook_text(label)
    st.markdown(f"### {pb['title']}")
    for s in pb["steps"]:
        st.write(f"- {s}")

    name = st.text_input("Customer name (optional)", value="Customer")
    message = pb["template"].replace("{name}", name)
    st.text_area("Copy/paste message", value=message, height=190)

    st.download_button(
        "⬇️ Download message (.txt)",
        data=message.encode("utf-8"),
        file_name="retention_message.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def score_paste_one():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Score a customer — Paste one row (CSV)")

    sample = (
        "gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,"
        "OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,"
        "Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
        "Female,0,No,No,4,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,"
        "Yes,Electronic check,94.65,378.60"
    )
    text = st.text_area("Paste 1-row CSV with header", height=160, value=sample)
    run = st.button("Score pasted customer", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not run:
        return

    try:
        df = pd.read_csv(StringIO(text))
        if df.shape[0] != 1:
            st.error("Paste exactly ONE row.")
            return
        inp = df.iloc[0].to_dict()
        prob = predict_prob(model, feature_names, encoders, inp)
    except Exception as e:
        st.error("Invalid CSV.")
        st.code(str(e))
        return

    label, css = risk_badge(prob, threshold)
    sla = sla_recommendation(prob, threshold)
    signals = build_signals(inp)

    db_insert("single_paste", prob, label, threshold, inp)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f'<span class="{css}">{label}</span>', unsafe_allow_html=True)
    st.markdown("### Decision Summary")
    st.write(f"**Churn probability:** `{prob:.3f}`")
    st.write(f"**Likelihood:** `{likelihood_label(prob)}`")
    st.write(f"**Recommended SLA:** `{sla}`")

    st.markdown("#### Signals to check")
    for s in signals:
        st.write(f"- {s}")

    pb = playbook_text(label)
    st.markdown(f"### {pb['title']}")
    for s in pb["steps"]:
        st.write(f"- {s}")

    name = st.text_input("Customer name (optional)", value="Customer")
    message = pb["template"].replace("{name}", name)
    st.text_area("Copy/paste message", value=message, height=190)

    st.download_button(
        "⬇️ Download message (.txt)",
        data=message.encode("utf-8"),
        file_name="retention_message.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def batch_scoring():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Batch scoring — Upload CSV")

    st.caption("Memory-safe mode: limits the number of rows processed on Streamlit Cloud.")
    max_rows = st.number_input("Max rows to process (safety limit)", 1000, 200000, 50000, 1000)

    up = st.file_uploader("Upload CSV", type=["csv"])
    template = pd.DataFrame([{c: "" for c in feature_names}])
    st.download_button(
        "⬇️ Download template CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="churn_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if up is None:
        return

    # Read only up to max_rows to prevent memory crashes
    try:
        df = pd.read_csv(up, nrows=int(max_rows))
    except Exception as e:
        st.error("Could not read CSV.")
        st.code(str(e))
        return

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"].astype(str).str.strip().replace({"": "0.0", " ": "0.0"}).astype(float)
        )

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.error("CSV missing required columns:")
        st.code(str(missing))
        return

    df_enc = encode_and_order(df.copy(), feature_names, encoders)
    probs = model.predict_proba(df_enc)[:, 1].astype(float)
    preds = (probs >= threshold).astype(int)

    out = df.copy()
    out["churn_prob"] = probs
    out["churn_pred"] = preds

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Results (Top 50 by risk)")
    st.write(f"<span class='small'>Scored rows: {len(out)} | Flagged: {(out['churn_pred']==1).sum()}</span>", unsafe_allow_html=True)

    top = out.sort_values("churn_prob", ascending=False).head(50)
    st.dataframe(top, use_container_width=True)

    st.download_button(
        "⬇️ Download scored CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="churn_scored.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def insights_optional():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Insights (Optional)")
    st.caption("This page loads the dataset and can use more memory. Turn it on only if needed.")

    if not DATA_PATH.exists():
        st.info("Dataset not found in `data/`. Insights are disabled on this deployment.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    load = st.toggle("Load dataset & show insights", value=False)
    if not load:
        st.info("Toggle ON to load dataset and render charts.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("</div>", unsafe_allow_html=True)

    # Load dataset only when toggled ON
    df = pd.read_csv(DATA_PATH)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip().replace({"": "0.0", " ": "0.0"}).astype(float)

    if "Churn" in df.columns:
        df["Churn_bin"] = df["Churn"].replace({"Yes": 1, "No": 0})
    else:
        df["Churn_bin"] = 0

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Churn rate by Contract")
    if "Contract" in df.columns:
        t = df.groupby("Contract")["Churn_bin"].mean().sort_values(ascending=False)
        st.bar_chart(t)
    else:
        st.info("Contract column not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Churn rate vs MonthlyCharges (binned)")
    if "MonthlyCharges" in df.columns:
        bins = np.linspace(df["MonthlyCharges"].min(), df["MonthlyCharges"].max(), 12)
        df["mc_bin"] = pd.cut(df["MonthlyCharges"], bins=bins, include_lowest=True)
        t2 = df.groupby("mc_bin")["Churn_bin"].mean()
        st.line_chart(t2)
    else:
        st.info("MonthlyCharges column not available.")
    st.markdown("</div>", unsafe_allow_html=True)


def history_page():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("History")
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


# Router
if page == "Home":
    render_home()

elif page == "Score (Single)":
    tabs = st.tabs(["Manual input", "Paste single customer"])
    with tabs[0]:
        score_manual()
    with tabs[1]:
        score_paste_one()

elif page == "Batch Scoring":
    batch_scoring()

elif page == "Insights (Optional)":
    insights_optional()

elif page == "History":
    history_page()
