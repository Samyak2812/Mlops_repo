import pandas as pd
import numpy as np
import pickle, mlflow
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp

# Load model
model = pickle.load(open("models/risk_model.pkl", "rb"))

# Load old and new data
train = pd.read_csv("data/processed/train.csv")
new = pd.read_csv("data/new_data/production_incidents.csv")

# Features and labels
features = ["severity", "likelihood", "impact"]

# --- 1️⃣ Accuracy Monitoring ---
X_new = new[features]
y_true = new["risk_level"]
y_pred = model.predict(X_new)
accuracy = accuracy_score(y_true, y_pred)

# --- 2️⃣ Data Drift Detection (using KS Test) ---
drift_results = {}
for feature in features:
    stat, pval = ks_2samp(train[feature], new[feature])
    drift_results[feature] = {"p_value": pval, "drift_detected": pval < 0.05}

# --- 3️⃣ Log to MLflow ---
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # optional DB backend
mlflow.set_experiment("GRC_Model_Monitoring")

with mlflow.start_run():
    mlflow.log_metric("accuracy", accuracy)
    for feature, result in drift_results.items():
        mlflow.log_metric(f"{feature}_drift_pval", result["p_value"])
        mlflow.log_param(f"{feature}_drift_detected", result["drift_detected"])

# --- 4️⃣ Print Summary ---
print(f"✅ Accuracy on new data: {accuracy:.2f}")
for f, result in drift_results.items():
    print(f"Feature {f}: p={result['p_value']:.3f} | Drift: {result['drift_detected']}")
        
import os
import pandas as pd

# Ensure metrics folder exists
os.makedirs("metrics", exist_ok=True)

# Build a small monitoring report
report = pd.DataFrame({
    "metric": ["accuracy"] + list(drift_results.keys()),
    "value": [accuracy] + [r["p_value"] for r in drift_results.values()],
    "drift_detected": ["-"] + [r["drift_detected"] for r in drift_results.values()]
})

# Save report
report.to_csv("metrics/monitoring_report.csv", index=False)
print("✅ Monitoring report saved to metrics/monitoring_report.csv")
