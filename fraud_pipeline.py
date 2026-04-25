# fraud_pipeline.py

import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics
from kfp import kubernetes

# ─────────────────────────────────────────────
# COMPONENT 1: Data Ingestion
# ─────────────────────────────────────────────
@component(
    base_image="python:3.9",
    packages_to_install=["pandas"]
)
def data_ingestion(output_dataset: Output[Dataset]):
    import pandas as pd
    import os
    import gc

    # ─────────────────────────────
    # PVC DATA PATH (KEY CHANGE)
    # ─────────────────────────────
    data_path = "/mnt/fraud-artifacts/data"

    train_path = os.path.join(data_path, "train_transaction.csv")
    identity_path = os.path.join(data_path, "train_identity.csv")

    # ─────────────────────────────
    # VALIDATION (IMPORTANT FOR DEBUGGING)
    # ─────────────────────────────
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")

    if not os.path.exists(identity_path):
        raise FileNotFoundError(f"Missing file: {identity_path}")

    # ─────────────────────────────
    # LOAD DATA
    # ─────────────────────────────
    print("Loading identity (small file)...")
    identity = pd.read_csv(identity_path, low_memory=True)
    identity = identity.drop_duplicates("TransactionID")
    identity = identity.astype("float32", errors="ignore")

    print("Loading train in chunks and merging...")
    first_chunk = True

    identity = identity.sample(frac=0.3, random_state=42)
    for chunk in pd.read_csv(train_path, chunksize=50000, low_memory=True):

        merged_chunk = chunk.merge(identity, on="TransactionID", how="left")
    # write directly instead of storing
        if first_chunk:
            merged_chunk.to_csv(output_dataset.path, index=False)
            first_chunk = False
        else:

            merged_chunk.to_csv(output_dataset.path, mode="a", header=False, index=False)

    del chunk, merged_chunk
    gc.collect()

    del identity
    gc.collect()

    print("✅ Data ingestion successful")


# ─────────────────────────────────────────────
# COMPONENT 2: Data Validation
# ─────────────────────────────────────────────
@component(
    base_image="python:3.9",
    packages_to_install=["pandas"]
)
def data_validation(
    input_dataset: Input[Dataset],
    validation_report: Output[Dataset]
):
    import pandas as pd
    import json
    import os

    total_rows = 0
    total_missing = 0
    total_values = 0
    fraud_sum = 0
    columns = None

    for chunk in pd.read_csv(input_dataset.path, chunksize=50000):
        if columns is None:
            columns = chunk.columns.tolist()

        total_rows += len(chunk)
        total_missing += chunk.isnull().sum().sum()
        total_values += chunk.size

        if "isFraud" in chunk.columns:
            fraud_sum += chunk["isFraud"].sum()

    fraud_rate = fraud_sum / total_rows if total_rows > 0 else 0

    report = {
        "total_rows": total_rows,
        "total_columns": len(columns),
        "missing_value_pct": (total_missing / total_values * 100),
        "fraud_rate": float(fraud_rate),
        "schema_valid": "isFraud" in columns and "TransactionID" in columns
    }

    print("Validation Report:", json.dumps(report, indent=2))

    with open(validation_report.path, "w") as f:
        json.dump(report, f)

    BASE = "/mnt/fraud-artifacts"
    log_dir = f"{BASE}/logs"

    os.system(f"mkdir -p {log_dir}")
    os.system(f"chmod -R 777 {BASE} || true")

    log_file = os.path.join(log_dir, "validation.json")
    with open(log_file, "w") as f:
        json.dump(report, f)


# ─────────────────────────────────────────────
# COMPONENT 3: Data Preprocessing
# ─────────────────────────────────────────────
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn"]
)
def data_preprocessing(
    input_dataset: Input[Dataset],
    processed_dataset: Output[Dataset]
):
    import pandas as pd
    from sklearn.impute import SimpleImputer
    import numpy as np
    import os

    df_sample = pd.read_csv(input_dataset.path, nrows=100000)
    num_cols = df_sample.select_dtypes(include=["number"]).columns
    num_medians = df_sample[num_cols].median()

    first = True

   # BASE = "/mnt/fraud-artifacts"
   # os.system(f"mkdir -p {BASE}/data")
   # os.system(f"chmod -R 777 {BASE} || true")
   # data_dir = f"{BASE}/data"

    for chunk in pd.read_csv(input_dataset.path, chunksize=50000):

        chunk = chunk.convert_dtypes()

        chunk[num_cols] = chunk[num_cols].fillna(num_medians)

        cat_cols = chunk.select_dtypes(include=["object"]).columns

        for col in cat_cols:
            mode_val = chunk[col].mode()
            if len(mode_val) > 0:
                chunk[col] = chunk[col].fillna(mode_val[0])

        chunk.to_csv(processed_dataset.path, mode="w" if first else "a", header=first, index=False)

     #   chunk.to_csv(f"{data_dir}/preprocessed.csv", mode="w" if first else "a", header=first, index=False)

        first = False

    print(f"Preprocessing done")


# ─────────────────────────────────────────────
# COMPONENT 4: Feature Engineering
# ─────────────────────────────────────────────
@component(
    base_image="python:3.9",
    packages_to_install=["pandas"]
)
def feature_engineering(
    input_dataset: Input[Dataset],
    engineered_dataset: Output[Dataset]
):
    import pandas as pd
    import os

    BASE = "/mnt/fraud-artifacts"

    #os.system(f"mkdir -p {BASE}/data")
    #os.system(f"chmod -R 777 {BASE} || true")

    data_dir = f"{BASE}/data"

    first = True

    for chunk in pd.read_csv(input_dataset.path, chunksize=30000):

        chunk = pd.get_dummies(chunk, drop_first=True, sparse=True)

        chunk.to_csv(engineered_dataset.path, mode="w" if first else "a", header=first, index=False)
#        chunk.to_csv(f"{data_dir}/features.csv", mode="w" if first else "a", header=first, index=False)

        first = False

    print(f"Feature engineering done.")


# ─────────────────────────────────────────────
# COMPONENT 5: Model Training
# ─────────────────────────────────────────────
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "xgboost", "joblib"]
)
def model_training(
    input_dataset: Input[Dataset],
    model_artifact: Output[Model],
    metrics_output: Output[Metrics]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score
    import joblib
    import os

    BASE = "/mnt/fraud-artifacts"

    #os.system(f"mkdir -p {BASE}/models")
    #os.system(f"chmod -R 777 {BASE} || true")

    model_dir = f"{BASE}/models"

    df = pd.read_csv(input_dataset.path, nrows=100000, on_bad_lines="skip", low_memory=True)
    X = df.drop(columns=["isFraud"]).astype("float32")
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = X_train.sample(frac=0.3, random_state=42)
    y_train = y_train.loc[X_train.index]

    import gc
    gc.collect()

    # Train XGBoost with cost-sensitive weights
    model = XGBClassifier(
        scale_pos_weight=10,   # penalize false negatives
        n_estimators=70,
        max_depth=4,
        tree_method='hist',
        random_state=42,
        eval_metric="auc"
    )
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"AUC-ROC: {auc:.4f}")

    metrics_output.log_metric("auc_roc", auc)

    os.makedirs(model_artifact.path, exist_ok=True)

    # PVC Storage
    #joblib.dump(model, f"{model_dir}/model.pkl")

    # Kubeflow artifact for pipeline UI tracking
    joblib.dump(model, f"{model_artifact.path}/model.pkl")


# ─────────────────────────────────────────────
# COMPONENT 6: Model Evaluation
# ─────────────────────────────────────────────
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib", "numpy", "xgboost"]
)
def model_evaluation(
    input_dataset: Input[Dataset],
    model_artifact: Input[Model],
    eval_metrics: Output[Metrics]
) -> float:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        precision_score, recall_score,
        f1_score, roc_auc_score
    )
    import joblib
    import os

    BASE = "/mnt/fraud-artifacts"

    #os.system(f"mkdir -p {BASE}/logs")
    #os.system(f"chmod -R 777 {BASE} || true")

    log_dir = f"{BASE}/logs"

    df = pd.read_csv(input_dataset.path, nrows=30000, on_bad_lines="skip")
    df = df.astype("float32")
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    # _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    split = int(0.8 * len(df))
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    import gc
    del df
    gc.collect()

    model = joblib.load(f"{model_artifact.path}/model.pkl")
    y_pred = model.predict(X_test)
    # y_prob = model.predict_proba(X_test)[:, 1]

    import numpy as np

    batch_size = 5000
    probs = []

    for i in range(0, len(X_test), batch_size):
        batch = X_test.iloc[i:i+batch_size]
        probs.append(model.predict_proba(batch)[:, 1])

    y_prob = np.concatenate(probs)

    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_prob)

    eval_metrics.log_metric("precision", precision)
    eval_metrics.log_metric("recall", recall)
    eval_metrics.log_metric("f1_score", f1)
    eval_metrics.log_metric("auc_roc", auc)

    import json

    log = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc)
    }

    with open(f"{log_dir}/eval.json", "w") as f:
        json.dump(log, f)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    return auc


# ─────────────────────────────────────────────
# COMPONENT 7: Conditional Deployment
# ─────────────────────────────────────────────
@component(base_image="python:3.9")
def conditional_deployment(auc_score: float, threshold: float = 0.85):
    if auc_score >= threshold:
        print(f"✅ AUC {auc_score:.4f} >= {threshold}. Deploying model...")
        # In real setup: call kubectl or serving API here
    else:
        print(f"❌ AUC {auc_score:.4f} < {threshold}. Skipping deployment.")


# ─────────────────────────────────────────────
# PIPELINE DEFINITION
# ─────────────────────────────────────────────
@pipeline(
    name="fraud-detection-pipeline",
    description="End-to-end fraud detection MLOps pipeline"
)
def fraud_detection_pipeline(deploy_threshold: float = 0.85):

    # Step 1: Ingest
    ingest_task = data_ingestion()
    ingest_task.set_retry(num_retries=3)
    ingest_task.set_memory_limit("3G")
    ingest_task.set_cpu_limit("2")

    kubernetes.mount_pvc(
	ingest_task,
	pvc_name="fraud-artifacts-pvc",
	mount_path="/mnt/fraud-artifacts"
    )

    # Step 2: Validate
    validate_task = data_validation(
        input_dataset=ingest_task.outputs["output_dataset"]
    )
    validate_task.set_retry(num_retries=2)
    validate_task.set_memory_limit("3G")
    kubernetes.mount_pvc(
        validate_task,
        pvc_name="fraud-artifacts-pvc",
        mount_path="/mnt/fraud-artifacts"
    )

    # Step 3: Preprocess
    preprocess_task = data_preprocessing(
        input_dataset=ingest_task.outputs["output_dataset"]
    ).after(validate_task)
    preprocess_task.set_retry(num_retries=2)
    preprocess_task.set_memory_limit("3G")
    kubernetes.mount_pvc(
        preprocess_task,
        pvc_name="fraud-artifacts-pvc",
        mount_path="/mnt/fraud-artifacts"
    )

    # Step 4: Feature Engineering
    feature_task = feature_engineering(
        input_dataset=preprocess_task.outputs["processed_dataset"]
    ).after(preprocess_task)
    feature_task.set_memory_limit("3G")
    kubernetes.mount_pvc(
        feature_task,
        pvc_name="fraud-artifacts-pvc",
        mount_path="/mnt/fraud-artifacts"
    )

    # Step 5: Train
    train_task = model_training(
        input_dataset=feature_task.outputs["engineered_dataset"]
    )
    train_task.set_cpu_limit("2")
    train_task.set_memory_limit("4G")
    train_task.set_retry(num_retries=3)
    kubernetes.mount_pvc(
        train_task,
        pvc_name="fraud-artifacts-pvc",
        mount_path="/mnt/fraud-artifacts"
    )

    # Step 6: Evaluate
    eval_task = model_evaluation(
        input_dataset=feature_task.outputs["engineered_dataset"],
        model_artifact=train_task.outputs["model_artifact"]
    )
    eval_task.set_memory_limit("4G")
    kubernetes.mount_pvc(
        eval_task,
        pvc_name="fraud-artifacts-pvc",
        mount_path="/mnt/fraud-artifacts"
    )

    # Step 7: Deploy conditionally
    deploy_task = conditional_deployment(
    auc_score=eval_task.outputs["Output"],
    threshold=deploy_threshold
    )


# ─────────────────────────────────────────────
# COMPILE & SUBMIT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path="fraud_pipeline.yaml"
    )
    print("Pipeline compiled → fraud_pipeline.yaml")
