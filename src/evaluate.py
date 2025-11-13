import pandas as pd, pickle
from sklearn.metrics import accuracy_score, classification_report

model = pickle.load(open("models/risk_model.pkl", "rb"))
test = pd.read_csv("data/processed/test.csv")

X_test = test[["severity", "likelihood", "impact"]]
y_test = test["risk_level"]

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
