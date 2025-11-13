import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow, pickle, os

mlflow.set_experiment("GRC_Risk_Model")

train = pd.read_csv("data/processed/train.csv")
X = train[["severity", "likelihood", "impact"]]
y = train["risk_level"]

model = RandomForestClassifier(n_estimators=10, random_state=42)

with mlflow.start_run():
    model.fit(X, y)
    mlflow.log_param("n_estimators", 10)
    mlflow.sklearn.log_model(model, "model")

os.makedirs("models", exist_ok=True)
with open("models/risk_model.pkl", "wb") as f:
    pickle.dump(model, f)
