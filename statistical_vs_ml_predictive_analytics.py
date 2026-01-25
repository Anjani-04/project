# statistical_vs_ml_predictive_analytics.py
# Performance Evaluation of Statistical and Machine Learning Models for Predictive Analytics
# Domain: Predictive Modeling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# -----------------------------
# 1. Synthetic Dataset Generation
# -----------------------------
def generate_predictive_data(samples=2000):
    np.random.seed(42)

    x1 = np.random.normal(100, 20, samples)
    x2 = np.random.normal(50, 10, samples)
    x3 = np.random.normal(30, 5, samples)

    noise = np.random.normal(0, 10, samples)

    target = 2.5 * x1 + 1.8 * x2 + 3.2 * x3 + noise

    df = pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "feature_3": x3,
        "target": target
    })

    return df


# -----------------------------
# 2. Dataset Creation & Saving
# -----------------------------
def create_and_save_dataset():
    df = generate_predictive_data()
    df.to_csv("predictive_analytics_data.csv", index=False)
    print("Dataset saved as predictive_analytics_data.csv")


# -----------------------------
# 3. Data Loading & Inspection
# -----------------------------
def load_and_inspect_data():
    df = pd.read_csv("predictive_analytics_data.csv")
    print("\nDataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())
    return df


# -----------------------------
# 4. Preprocessing
# -----------------------------
def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved")

    return X_scaled, y


# -----------------------------
# 5. Model Training
# -----------------------------
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    models = {
        "Linear Regression (Statistical)": LinearRegression(),
        "Decision Tree (ML)": DecisionTreeRegressor(random_state=42),
        "Random Forest (ML)": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        joblib.dump(model, name.replace(" ", "_").replace("(", "").replace(")", "").lower() + ".pkl")

    return trained, X_test, y_test


# -----------------------------
# 6. Evaluation
# -----------------------------
def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = r2

        print(f"\n{name} Performance")
        print("MAE :", mae)
        print("RMSE:", rmse)
        print("R2  :", r2)

    return results


# -----------------------------
# 7. Visualization
# -----------------------------
def plot_comparison(results):
    plt.figure(figsize=(7, 4))
    plt.bar(results.keys(), results.values())
    plt.ylabel("R2 Score")
    plt.title("Statistical vs Machine Learning Model Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("model_performance_comparison.png")
    plt.show()

    print("Comparison plot saved")


# -----------------------------
# 8. Prediction System
# -----------------------------
def predict_value(input_features):
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("random_forest_ml.pkl")

    input_features = np.array(input_features).reshape(1, -1)
    input_scaled = scaler.transform(input_features)

    return model.predict(input_scaled)[0]


# -----------------------------
# 9. Main Execution
# -----------------------------
def main():
    if not os.path.exists("predictive_analytics_data.csv"):
        create_and_save_dataset()

    df = load_and_inspect_data()
    X, y = preprocess_data(df)

    models, X_test, y_test = train_models(X, y)
    results = evaluate_models(models, X_test, y_test)
    plot_comparison(results)

    sample = X_test[0]
    prediction = predict_value(sample)
    print("\nSample Prediction Output:", prediction)


if _name_ == "_main_":
    main(
