# ==============================================
# Predictive Pulse - Hypertension Model Training
# Professional Pipeline Version
# ==============================================

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data():
    try:
        df = pd.read_csv("data/cardio_train.csv", sep=";")
        print("Dataset loaded successfully.\n")
        return df
    except Exception as e:
        print("Error loading dataset:", e)
        return None


def basic_preprocessing(df):
    print("Starting preprocessing...\n")

    df = df.drop(columns=["id"])
    df["age"] = df["age"] / 365

    df = df[(df["ap_hi"] > 50) & (df["ap_hi"] < 250)]
    df = df[(df["ap_lo"] > 30) & (df["ap_lo"] < 200)]
    df = df[df["ap_hi"] > df["ap_lo"]]

    print("Preprocessing completed.\n")
    return df


def train_models(df):
    print("Starting model comparison using pipelines...\n")

    X = df.drop(columns=["cardio"])
    y = df["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Logistic Regression with Scaling
    logistic_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000))
    ])

    # Tree models (no scaling needed)
    decision_tree = DecisionTreeClassifier(random_state=42)
    random_forest = RandomForestClassifier(random_state=42)

    models = {
        "Logistic Regression (Scaled)": logistic_pipeline,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{name} Accuracy: {accuracy:.4f}\n")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    print("===================================")
    print(f"Best Model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print("===================================\n")

    return best_model, best_model_name


def save_model(model):
    with open("models/hypertension_model.pkl", "wb") as file:
        pickle.dump(model, file)

    print("Best model saved successfully in models/ folder.\n")


if __name__ == "__main__":
    df = load_data()

    if df is not None:
        df = basic_preprocessing(df)

        best_model, best_model_name = train_models(df)

        save_model(best_model)