import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow, pickle, os

mlflow.set_tracking_uri("file:./mlruns")  # ensures relative local path
mlflow.set_experiment("GRC_Risk_Model")

train = pd.read_csv("data/processed/train.csv")
X = train[["severity", "likelihood", "impact"]]
y = train["risk_level"]

model = RandomForestClassifier(n_estimators=10, random_state=42)

with mlflow.start_run():
    model.fit(X, y)
    mlflow.log_param("n_estimators", 10)
    mlflow.sklearn.log_model(model, "model")

MODEL_DIR = os.path.join(os.getcwd(), "models")

os.makedirs(MODEL_DIR, exist_ok=True)
 
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")
 
with open(MODEL_PATH, "wb") as f:

    pickle.dump(model, f)
 
