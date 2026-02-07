import pickle
import pandas as pd
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "customer_churn_model.pkl"
ENCODERS_PATH = ARTIFACTS_DIR / "encoders.pkl"


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    model = model_data["model"]
    feature_names = model_data["features_names"]

    return model, feature_names, encoders


def prepare_input(input_dict, feature_names, encoders):
    df = pd.DataFrame([input_dict])

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing input fields: {missing}")

    # Apply the same encoding used during training
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])

    # Arrange columns in exact training order
    df = df[feature_names]
    return df


def predict_churn(input_dict):
    model, feature_names, encoders = load_artifacts()
    input_df = prepare_input(input_dict, feature_names, encoders)

    pred = int(model.predict(input_df)[0])
    prob = float(model.predict_proba(input_df)[0][1])

    return pred, prob


if __name__ == "__main__":
    # Example input
    sample_input = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    prediction, probability = predict_churn(sample_input)

    print("\nPrediction:", "Churn" if prediction == 1 else "No Churn")
    print("Churn Probability:", probability)
