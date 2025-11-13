import pandas as pd
from sklearn.model_selection import train_test_split
import yaml, os

params = yaml.safe_load(open("params.yaml"))["train"]

df = pd.read_csv("data/raw/incidents.csv")

train, test = train_test_split(df, test_size=params["test_size"], random_state=params["random_state"])

os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
