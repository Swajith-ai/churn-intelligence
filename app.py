import json
import sqlite3
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.inspection import permutation_importance


# =========================
# Paths
# =========================
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "customer_churn_model.pkl"
ENCODERS_PATH = ARTIFACTS_DIR / "encoders.pkl"

LOGO_PATH = Path("assets/logo.png")  # optional logo
DB_PATH = Path("data") / "history.db"

APP_TITLE = "Churn Intelligence"


# =========================
# Page config
# =========================
st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")


# =========================
# Theme Toggle (Light/Dark)
# =========================
if "theme_dark" not in st.session_state:
    st.session_state["theme_dark"] = False


def inject_css(dark: bool):
    if dark:
        css = """
        <style>
          :root { color-scheme: dark; }
          .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1220px; }
          html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont; }

          .hero {
            padding: 18px 18px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(14,165,233,0.08));
            border: 1px solid rgba(148,163,184,0.18);
            margin-bottom: 14px;
          }
          .hero h1 { margin: 0; font-size: 26px; color: #E2E8F0; letter-spacing: -0.2px;}
          .hero p  { margin: 6px 0 0 0; font-size: 14px; color: rgba(226,232,240,0.74); }

          .panel {
            background: rgba(2,6,23,0.72);
            border-radius: 18px;
            padding: 18px;
            border: 1px solid rgba(148,163,184,0.16);
            box-shadow: 0 10px 26px rgba(0,0,0,0.25);
          }

          .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            font-weight: 800;
            font-size: 13px;
            border: 1px solid rgba(148,163,184,0.20);
          }
          .badge-high { background: rgba(239,68,68,0.18); color: #FCA5A5; border-color: rgba(239,68,68,0.35); }
          .badge-mid  { background: rgba(245,158,11,0.18); color: #FCD34D; border-color: rgba(245,158,11,0.35); }
          .badge-low  { background: rgba(34,197,94,0.18); color: #86EFAC; border-color: rgba(34,197,94,0.35); }

          .note {
            background: rgba(148,163,184,0.08);
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: 16px;
            padding: 14px 14px;
            line-height: 1.55;
            color: rgba(226,232,240,0.86);
          }
          .note h4 { margin: 0 0 8px 0; color: #E2E8F0; }
          .note ul { margin: 8px 0 0 18px; }
          .note li { margin: 6px 0; }

          div[data-testid="stMetric"] {
            background: rgba(2,6,23,0.72);
            padding: 14px;
            border-radius: 16px;
            border: 1px solid rgba(148,163,184,0.16);
            box-shadow: 0 8px 18px rgba(0,0,0,0.22);
          }

          .stButton button { border-radius: 14px; padding: 0.62rem 1.1rem; font-weight: 800; }
          .stSelectbox div, .stNumberInput input, textarea { border-radius: 12px !important; }
          section[data-testid="stSidebar"] { border-right: 1px solid rgba(148,163,184,0.10); }
        </style>
        """
    else:
        css = """
        <style>
          .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1220px; }
          html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont; }

          .hero {
            padding: 18px 18px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(79,70,229,0.10), rgba(14,165,233,0.05));
            border: 1px solid rgba(2,6,23,0.08);
            margin-bottom: 14px;
          }
          .hero h1 { margin: 0; font-size: 26px; color: #0F172A; letter-spacing: -0.2px;}
          .hero p  { margin: 6px 0 0 0; font-size: 14px; color: rgba(15,23,42,0.72); }

          .panel {
            background: #FFFFFF;
            border-radius: 18px;
            padding: 18px;
            border: 1px solid rgba(2,6,23,0.08);
            box-shadow: 0 10px 26px rgba(2,6,23,0.06);
          }

          .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            font-weight: 800;
            font-size: 13px;
            border: 1px solid rgba(2,6,23,0.12);
          }
          .badge-high { background: rgba(239,68,68,0.12); color: #B91C1C; border-color: rgba(239,68,68,0.32); }
          .badge-mid  { background: rgba(245,158,11,0.14); color: #92400E; border-color: rgba(245,158,11,0.30); }
          .badge-low  { background: rgba(34,197,94,0.12); color: #047857; border-color: rgba(34,197,94,0.32); }

          .note {
            background: rgba(15,23,42,0.03);
            border: 1px solid rgba(2,6,23,0.08);
            border-radius: 16px;
            padding: 14px 14px;
            line-height: 1.55;
            color: rgba(15,23,42,0.86);
          }
          .note h4 { margin: 0 0 8px 0; color: #0F172A; }
          .note ul { margin: 8px 0 0 18px; }
          .note li { margin: 6px 0; }

          div[data-testid="stMetric"] {
            background: #FFFFFF;
            padding: 14px;
            border-radius: 16px;
            border: 1px solid rgba(2,6,23,0.08);
            box-shadow: 0 8px 18px rgba(2,6,23,0.06);
          }

          .stButton button { border-radius: 14px; padding: 0.62rem 1.1rem; font-weight: 800; }
          .stSelectbox div, .stNumberInput input, textarea { border-radius: 12px !important; }
          section[data-testid="stSidebar"] { border-right: 1px solid rgba(2,6,23,0.06); }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def explain_box(title: str, points: list[str]) -> str:
    lis = "".join([f"<li>{p}</li>" for p in points])
    return f"""
    <div class="note">
      <h4>{title}</h4>
      <ul>{lis}</ul>
    </div>
    """


def make_layout(fig, height=520, title=None):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=56, b=10),
        font=dict(family="Inter, system-ui", size=13),
        title=dict(text=title or fig.layout.title.text, x=0.01, xanchor="left"),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(2,6,23,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(2,6,23,0.06)")
    return fig


def make_gauge(prob: float, threshold: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"valueformat": ".3f"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "rgba(79,70,229,0.78)"},
                "threshold": {"line": {"color": "rgba(239,68,68,0.85)", "width": 4}, "value": threshold},
            },
        )
    )
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=10, b=10))
    return fig


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


# =========================
# DB: History
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


# =========================
# Data/Model loading
# =========================
@st.cache_data
def load_raw():
    df = pd.read_csv(DATA_PATH)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip().replace({"": "0.0", " ": "0.0"}).astype(float)
    return df


@st.cache_resource
def load_artifacts():
    import pickle
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    model = model_data.get("model")
    feature_names = model_data.get("features_names") or model_data.get("feature_names") or model_data.get("features")
    if model is None or feature_names is None:
        raise ValueError("Model file must contain: 'model' and feature list (features_names/feature_names/features).")

    return model, feature_names, encoders


def add_churn_bin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Churn" in out.columns and out["Churn"].dtype == object:
        out["Churn_bin"] = out["Churn"].replace({"Yes": 1, "No": 0})
    else:
        out["Churn_bin"] = out.get("Churn", 0)
    return out


def safe_label_transform(le, values: pd.Series) -> np.ndarray:
    known = set(le.classes_.tolist())
    fallback = le.classes_[0]
    cleaned = values.astype(str).apply(lambda v: v if v in known else fallback)
    return le.transform(cleaned)


def encode_and_order(df: pd.DataFrame, feature_names, encoders) -> pd.DataFrame:
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = safe_label_transform(le, df[col])

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[feature_names]


def predict_prob(model, feature_names, encoders, input_dict):
    df = pd.DataFrame([input_dict])
    df_enc = encode_and_order(df, feature_names, encoders)
    prob = float(model.predict_proba(df_enc)[0][1])
    return prob


# ✅ FINAL FIX: no caching here → no hashing issues at all
def compute_global_importance(raw_df: pd.DataFrame, model, feature_names, encoders, sample_n: int = 800):
    df = raw_df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "Churn" in df.columns:
        y = df["Churn"].replace({"Yes": 1, "No": 0})
        X = df.drop(columns=["Churn"])
    elif "Churn_bin" in df.columns:
        y = df["Churn_bin"]
        X = df.drop(columns=["Churn_bin"])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = (
            X["TotalCharges"]
            .astype(str)
            .str.strip()
            .replace({"": "0.0", " ": "0.0"})
            .astype(float)
        )

    if len(X) > sample_n:
        X = X.sample(sample_n, random_state=42)
        y = y.loc[X.index]

    X_enc = encode_and_order(X.copy(), feature_names, encoders)

    try:
        r = permutation_importance(model, X_enc, y, n_repeats=4, random_state=42, n_jobs=-1)
        imp = pd.DataFrame({"feature": feature_names, "importance": r.importances_mean})
        return imp.sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


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


def playbook_text(badge: str) -> dict:
    if badge == "High Risk":
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
    if badge == "At Risk":
        return {
            "title": "Retention Playbook (At Risk)",
            "steps": [
                "Contact within 24–48 hours.",
                "Check plan fit and billing clarity (avoid surprises).",
                "Offer a small value reinforcement (trial add-on / plan optimization / loyalty perk).",
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


def churn_rate_by_bins(df, x_col, bins):
    temp = df[[x_col, "Churn_bin"]].dropna().copy()
    temp["bin"] = pd.cut(temp[x_col], bins=bins, include_lowest=True)
    grp = temp.groupby("bin")["Churn_bin"].mean().reset_index()
    grp["bin_label"] = grp["bin"].astype(str)
    return grp[["bin_label", "Churn_bin"]]


# =========================
# Startup checks
# =========================
if not DATA_PATH.exists():
    st.error("Dataset not found. Put it here: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`")
    st.stop()

if not MODEL_PATH.exists() or not ENCODERS_PATH.exists():
    st.error("Model artifacts not found. Required:\n- artifacts/customer_churn_model.pkl\n- artifacts/encoders.pkl")
    st.stop()

db_init()

st.sidebar.markdown("## 🎛️ Appearance")
st.session_state["theme_dark"] = st.sidebar.toggle("Dark mode", value=st.session_state["theme_dark"])
inject_css(st.session_state["theme_dark"])

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)

st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Score (Single)", "Batch Scoring", "Insights", "Playbook", "History"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown("## ⚙️ Risk Policy")
threshold = st.sidebar.slider("High-risk threshold", 0.05, 0.95, 0.50, 0.01)
st.sidebar.caption("Probability ≥ threshold → flagged for retention")

if "last_scored" not in st.session_state:
    st.session_state["last_scored"] = None

with st.spinner("Loading app assets..."):
    raw_df = load_raw()
    df_i = add_churn_bin(raw_df)
    model, feature_names, encoders = load_artifacts()

cat_cols = raw_df.select_dtypes(include="object").columns.tolist()
if "customerID" in cat_cols:
    cat_cols.remove("customerID")
cat_options = {c: sorted(raw_df[c].dropna().unique().tolist()) for c in cat_cols}

global_imp = compute_global_importance(raw_df, model, feature_names, encoders)

st.markdown(
    f"""
    <div class="hero">
      <h1>{APP_TITLE}</h1>
      <p>Product-grade churn scoring — single scoring, batch scoring, insights and playbooks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_decision(inp: dict, prob: float):
    label, css = risk_badge(prob, threshold)
    sla = sla_recommendation(prob, threshold)
    signals = build_signals(inp)

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    colA, colB = st.columns([1.2, 1.0], vertical_alignment="top")
    with colA:
        st.markdown(f'<span class="{css}">{label}</span>', unsafe_allow_html=True)
        st.markdown("### Decision Summary")
        st.caption("Use this to prioritize retention workflows.")
    with colB:
        st.plotly_chart(make_gauge(prob, threshold), use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Churn probability", f"{prob:.3f}")
    m2.metric("Likelihood", likelihood_label(prob))
    m3.metric("Policy threshold", f"{threshold:.2f}")
    m4.metric("Recommended SLA", sla)

    st.markdown(explain_box("Signals to check", signals), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return label


def render_explainability():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Explainability (Global drivers)")

    if global_imp is not None and not global_imp.empty:
        top = global_imp.head(12)
        fig = px.bar(top, x="importance", y="feature", orientation="h", title="Top global drivers")
        fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=56, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Global importance not available for this model/dataset.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_playbook(label: str):
    pb = playbook_text(label)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader(pb["title"])
    st.markdown(explain_box("Recommended steps", pb["steps"]), unsafe_allow_html=True)

    st.markdown("### Ready-to-use message template")
    name = st.text_input("Customer name (optional)", value="Customer")
    filled = pb["template"].replace("{name}", name)
    st.text_area("Copy/paste this", value=filled, height=210)

    st.download_button(
        "⬇️ Download message as .txt",
        data=filled.encode("utf-8"),
        file_name="retention_message.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def manual_form():
    with st.form("manual_form", clear_on_submit=False):
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Customer profile")

        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", cat_options["gender"])
            senior = st.number_input("SeniorCitizen (0/1)", 0, 1, 0, 1)
            partner = st.selectbox("Partner", cat_options["Partner"])
            dependents = st.selectbox("Dependents", cat_options["Dependents"])
        with c2:
            contract = st.selectbox("Contract", cat_options["Contract"])
            tenure = st.number_input("Tenure (months)", 0, 1000, 12, 1)
            paperless = st.selectbox("PaperlessBilling", cat_options["PaperlessBilling"])
            payment = st.selectbox("PaymentMethod", cat_options["PaymentMethod"])
        with c3:
            internet = st.selectbox("InternetService", cat_options["InternetService"])
            phone = st.selectbox("PhoneService", cat_options["PhoneService"])
            multiple = st.selectbox("MultipleLines", cat_options["MultipleLines"])
            monthly = st.number_input("MonthlyCharges", 0.0, 1000.0, 70.0, 1.0)

        total = st.number_input("TotalCharges", 0.0, 1_000_000.0, 800.0, 10.0)

        with st.expander("Advanced: add-on services", expanded=False):
            a1, a2, a3 = st.columns(3)
            with a1:
                online_security = st.selectbox("OnlineSecurity", cat_options["OnlineSecurity"])
                online_backup = st.selectbox("OnlineBackup", cat_options["OnlineBackup"])
            with a2:
                device_protection = st.selectbox("DeviceProtection", cat_options["DeviceProtection"])
                tech_support = st.selectbox("TechSupport", cat_options["TechSupport"])
            with a3:
                streaming_tv = st.selectbox("StreamingTV", cat_options["StreamingTV"])
                streaming_movies = st.selectbox("StreamingMovies", cat_options["StreamingMovies"])

        submitted = st.form_submit_button("Score customer", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    inp = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }
    return submitted, inp


def paste_one():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Paste one customer (fast)")
    sample = (
        "gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
        "Female,0,No,No,4,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,94.65,378.60"
    )
    pasted = st.text_area("Paste 1-row CSV (with header)", height=160, value=sample)
    run = st.button("Score pasted customer", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not run:
        return False, None

    try:
        df = pd.read_csv(StringIO(pasted))
        if df.shape[0] != 1:
            st.error("Paste exactly ONE row.")
            return False, None
        return True, df.iloc[0].to_dict()
    except Exception as e:
        st.error("Could not parse input.")
        st.code(str(e))
        return False, None


def batch_upload():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Upload CSV (batch scoring)")
    up = st.file_uploader("Upload CSV", type=["csv"])

    template_df = pd.DataFrame([{c: "" for c in feature_names}])
    st.download_button(
        "⬇️ Download template CSV",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="churn_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if up is None:
        return None

    df = pd.read_csv(up)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip().replace({"": "0.0", " ": "0.0"}).astype(float)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.error("CSV missing required columns:")
        st.code(str(missing))
        return None

    df_enc = encode_and_order(df.copy(), feature_names, encoders)
    probs = model.predict_proba(df_enc)[:, 1].astype(float)
    preds = (probs >= threshold).astype(int)

    out = df.copy()
    out["churn_prob"] = probs
    out["churn_pred"] = preds
    return out


def render_home():
    st.markdown("## Home / Overview")
    hist = db_read(limit=800)
    if hist.empty:
        st.info("No history yet. Score a customer or run batch scoring to start.")
        return

    total_scored = len(hist)
    flagged = int(hist["risk_label"].isin(["At Risk", "High Risk"]).sum())
    high = int((hist["risk_label"] == "High Risk").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total scored (history)", total_scored)
    c2.metric("Flagged", flagged)
    c3.metric("High Risk", high)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Recent activity")
    st.dataframe(hist.head(30), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_history():
    st.markdown("## History")
    hist = db_read(limit=1000)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
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


def render_insights():
    st.markdown("## Insights Dashboard")

    bins_mc = np.linspace(df_i["MonthlyCharges"].min(), df_i["MonthlyCharges"].max(), 12)
    mc_trend = churn_rate_by_bins(df_i, "MonthlyCharges", bins_mc)
    fig1 = px.line(mc_trend, x="bin_label", y="Churn_bin", markers=True, title="Churn rate vs MonthlyCharges (binned)")
    fig1.update_traces(line=dict(width=3))
    st.plotly_chart(make_layout(fig1, 520), use_container_width=True)

    bins_t = np.array([0, 3, 6, 12, 24, 36, 48, 60, 72, 1000])
    t_trend = churn_rate_by_bins(df_i, "tenure", bins_t)
    fig2 = px.bar(t_trend, x="bin_label", y="Churn_bin", title="Churn rate by tenure stage")
    st.plotly_chart(make_layout(fig2, 520), use_container_width=True)

    contract_churn = df_i.groupby("Contract")["Churn_bin"].mean().sort_values(ascending=False).reset_index()
    contract_churn.columns = ["Contract", "AvgChurnRate"]
    fig3 = px.bar(contract_churn, x="Contract", y="AvgChurnRate", title="Average churn rate by contract type", text="AvgChurnRate")
    fig3.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    st.plotly_chart(make_layout(fig3, 460), use_container_width=True)


# =========================
# Sidebar navigation
# =========================
if page == "Home":
    render_home()

elif page == "Score (Single)":
    st.markdown("### Score customer")
    tabs = st.tabs(["Manual input", "Paste single customer", "Upload CSV (batch)"])

    with tabs[0]:
        run, inp = manual_form()
        if run:
            prob = predict_prob(model, feature_names, encoders, inp)
            label, _ = risk_badge(prob, threshold)
            st.session_state["last_scored"] = ("single_manual", inp, prob, label)
            db_insert("single_manual", prob, label, threshold, inp)

    with tabs[1]:
        run, inp = paste_one()
        if run and inp is not None:
            prob = predict_prob(model, feature_names, encoders, inp)
            label, _ = risk_badge(prob, threshold)
            st.session_state["last_scored"] = ("single_paste", inp, prob, label)
            db_insert("single_paste", prob, label, threshold, inp)

    with tabs[2]:
        out = batch_upload()
        if out is not None:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Batch results")

            total = len(out)
            flagged = int((out["churn_prob"] >= threshold).sum())

            a, b, c = st.columns(3)
            a.metric("Rows scored", total)
            b.metric(f"Flagged (≥ {threshold:.2f})", flagged)
            c.metric("Not flagged", total - flagged)

            show_flagged = st.toggle("Show only flagged rows", value=False)
            top_n = st.slider("Top N highest risk", 10, 200, 50, 10)

            view = out.copy()
            if show_flagged:
                view = view[view["churn_prob"] >= threshold]

            top_view = view.sort_values("churn_prob", ascending=False).head(top_n)
            st.dataframe(top_view, use_container_width=True, height=380)

            fig = px.histogram(out, x="churn_prob", nbins=30, title="Batch distribution: churn probabilities")
            fig.add_vline(x=threshold, line_width=4, line_dash="dash")
            fig.update_layout(template="plotly_white", height=520, margin=dict(l=10, r=10, t=56, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "⬇️ Download scored CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="churn_scored.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            db_insert("batch_upload", float(out["churn_prob"].mean()), "Batch", threshold, {"rows": int(total), "flagged": int(flagged)})

    if st.session_state["last_scored"] is not None:
        _, inp, prob, _label = st.session_state["last_scored"]
        decision = render_decision(inp, prob)
        render_explainability()
        render_playbook(decision)

elif page == "Batch Scoring":
    out = batch_upload()
    if out is not None:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Batch results")
        st.dataframe(out.sort_values("churn_prob", ascending=False).head(80), use_container_width=True, height=420)
        st.download_button(
            "⬇️ Download scored CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="churn_scored.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Insights":
    render_insights()

elif page == "Playbook":
    if st.session_state["last_scored"] is None:
        st.info("Score a customer first — the playbook adapts to the risk level.")
    else:
        _, inp, prob, _label = st.session_state["last_scored"]
        decision = render_decision(inp, prob)
        render_explainability()
        render_playbook(decision)

elif page == "History":
    render_history()
