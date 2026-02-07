import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Loads Telco churn dataset and applies cleaning:
    - Drop customerID
    - Fix TotalCharges blank strings and convert to float
    - Convert Churn Yes/No -> 1/0
    """
    df = pd.read_csv(csv_path)

    # Drop ID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges (some rows have " " in the dataset)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
        df["TotalCharges"] = df["TotalCharges"].astype(float)

    # Convert target
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

    return df


def encode_categoricals(df: pd.DataFrame, encoders_path: str = "artifacts/encoders.pkl") -> pd.DataFrame:
    """
    Label-encodes all object columns and saves encoders for later prediction.
    """
    object_columns = df.select_dtypes(include="object").columns

    encoders = {}
    for col in object_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Save encoders
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)

    return df


def preprocess(csv_path: str):
    """
    Full preprocessing pipeline:
    - load + clean
    - encode categoricals
    - split X/y
    """
    df = load_and_clean_data(csv_path)
    df = encode_categoricals(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y
