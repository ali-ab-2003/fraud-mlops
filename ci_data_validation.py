import pandas as pd

df = pd.read_csv("data/train_transaction.csv", nrows=5000)

required_cols = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise Exception(f"Missing columns: {missing}")

missing_pct = df.isnull().mean().mean()

print("CI Validation OK")
print("Missing %:", missing_pct)
