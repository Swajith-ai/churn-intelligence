import pickle
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocess import preprocess


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "customer_churn_model.pkl"


def train_and_select_model(X, y):
    print("✅ Starting train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("✅ Split done")

    print("✅ Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("✅ SMOTE done")

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }
    print("✅ Models ready: Decision Tree, Random Forest")

    cv_means = {}
    for name, model in models.items():
        print(f"\n🔄 Running CV for: {name} ...")
        scores = cross_val_score(model, X_train_smote, y_train_smote, cv=3, scoring="accuracy")
        cv_means[name] = float(np.mean(scores))
        print(f"✅ {name} CV Accuracy (mean): {cv_means[name]:.4f}")

    best_name = max(cv_means, key=cv_means.get)
    best_model = models[best_name]
    print("\n🏆 Selected Best Model:", best_name)

    print("✅ Training best model on SMOTE data...")
    best_model.fit(X_train_smote, y_train_smote)
    print("✅ Training complete")

    return best_model, X_test, y_test, best_name


def evaluate(model, X_test, y_test):
    print("\n📊 Evaluating model on test set...")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


def save_model(model, feature_names):
    model_data = {"model": model, "features_names": feature_names}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n✅ Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    CSV_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    print("📥 Loading + preprocessing data...")
    X, y = preprocess(CSV_PATH)
    print("✅ Preprocessing complete")

    model, X_test, y_test, model_name = train_and_select_model(X, y)
    evaluate(model, X_test, y_test)
    save_model(model, X.columns.tolist())
