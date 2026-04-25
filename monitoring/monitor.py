import json
import os

LOG_PATH = "/mnt/fraud-artifacts/logs/validation.json"

# fallback if running in CI
if not os.path.exists(LOG_PATH):
    print("No metrics found — skipping monitoring")
    exit(0)

with open(LOG_PATH, "r") as f:
    metrics = json.load(f)

print("Loaded metrics:", metrics)

# -----------------------
# THRESHOLDS (TUNE THIS)
# -----------------------
FRAUD_RATE_THRESHOLD = 0.05
MISSING_THRESHOLD = 0.5

fraud_rate = metrics.get("fraud_rate", 0)
missing = metrics.get("missing_value_pct", 0)

# -----------------------
# DRIFT / PERFORMANCE CHECK
# -----------------------
if fraud_rate < FRAUD_RATE_THRESHOLD or missing > MISSING_THRESHOLD:
    print("🚨 Model drift detected → triggering retraining pipeline")

    # OPTION 1: trigger Kubeflow pipeline
    os.system("python fraud_pipeline.py")

else:
    print("✅ Model performance stable")
