import pandas as pd
import os

BASE = "data/ieee-fraud-detection"

train_path = os.path.join(BASE, "train_transaction.csv")
identity_path = os.path.join(BASE, "train_identity.csv")

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Missing: {train_path}")

if not os.path.exists(identity_path):
    raise FileNotFoundError(f"Missing: {identity_path}")

df = pd.read_csv(train_path, nrows=5000)

missing_pct = df.isnull().mean().mean() * 100

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing %:", missing_pct)

assert missing_pct < 60
assert "TransactionID" in df.columns

print("Validation OK")
