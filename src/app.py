from fastapi import FastAPI
import numpy as np, pickle

app = FastAPI()
model = pickle.load(open("models/risk_model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "GRC Risk Prediction API is running"}

@app.get("/predict")
def predict(severity: int, likelihood: int, impact: int):
    pred = model.predict(np.array([[severity, likelihood, impact]]))
    return {"predicted_risk_level": pred[0]}
