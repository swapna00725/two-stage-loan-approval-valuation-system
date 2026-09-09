from pathlib import Path
import json
import logging

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.feature_engineering.feature_engineering import (
    get_stage1_features,
)
from src.preprocessing.preprocessing import (
    get_preprocessor,
)


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "loan_data_processed.csv"
)

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"
LOG_DIR = PROJECT_ROOT / "artifacts" / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    filename=LOG_DIR / "stage1_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# Evaluation function
# =========================================================

def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate a classification model using multiple metrics.

    ROC-AUC is calculated using the probability corresponding
    specifically to the 'Approved' class.
    """

    predictions = model.predict(X_test)

    # -----------------------------------------------------
    # Get probability of the APPROVED class
    # -----------------------------------------------------

    probabilities = model.predict_proba(X_test)

    classes = model.classes_

    approved_index = list(classes).index("Approved")

    approved_probabilities = probabilities[:, approved_index]

    # -----------------------------------------------------
    # Convert target to binary labels
    # -----------------------------------------------------

    y_test_binary = y_test.map(
        {
            "Rejected": 0,
            "Approved": 1,
        }
    )

    # -----------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),

        "precision": precision_score(
            y_test,
            predictions,
            pos_label="Approved",
        ),

        "recall": recall_score(
            y_test,
            predictions,
            pos_label="Approved",
        ),

        "f1": f1_score(
            y_test,
            predictions,
            pos_label="Approved",
        ),

        "roc_auc": roc_auc_score(
            y_test_binary,
            approved_probabilities,
        ),
    }

    return metrics

# =========================================================
# Training function
# =========================================================

def train_stage1():
    """
    Train and compare Stage 1 loan approval models.

    Baseline:
        Logistic Regression

    Candidate:
        Random Forest Classifier
    """

    logger.info("Starting Stage 1 training.")

    # -----------------------------------------------------
    # 1. Load processed data
    # -----------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    logger.info(
        f"Loaded processed dataset with {len(df)} rows."
    )

    # -----------------------------------------------------
    # 2. Create features and target
    # -----------------------------------------------------

    X = get_stage1_features(df)
    y = df["loan_status"].str.strip()

    logger.info(
        f"Stage 1 feature count before preprocessing: {X.shape[1]}"
    )

    # -----------------------------------------------------
    # 3. Train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    logger.info(
        f"Training rows: {len(X_train)}"
    )

    logger.info(
        f"Testing rows: {len(X_test)}"
    )

    # -----------------------------------------------------
    # 4. Baseline model
    # -----------------------------------------------------

    baseline_model = Pipeline(
        steps=[
            (
                "preprocessor",
                get_preprocessor(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    logger.info(
        "Training baseline Logistic Regression."
    )

    baseline_model.fit(X_train, y_train)

    baseline_metrics = evaluate_classifier(
        baseline_model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # 5. Candidate model
    # -----------------------------------------------------

    candidate_model = Pipeline(
        steps=[
            (
                "preprocessor",
                get_preprocessor(),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    logger.info(
        "Training candidate Random Forest."
    )

    candidate_model.fit(X_train, y_train)

    candidate_metrics = evaluate_classifier(
        candidate_model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # 6. Compare models
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE 1 — LOAN APPROVAL MODEL COMPARISON")
    print("=" * 70)

    print("\nBaseline — Logistic Regression")
    for metric, value in baseline_metrics.items():
        print(f"{metric:12}: {value:.4f}")

    print("\nCandidate — Random Forest")
    for metric, value in candidate_metrics.items():
        print(f"{metric:12}: {value:.4f}")

    # -----------------------------------------------------
    # 7. Promotion gate
    # -----------------------------------------------------

    baseline_auc = baseline_metrics["roc_auc"]
    candidate_auc = candidate_metrics["roc_auc"]

    candidate_is_better = (
        candidate_auc >= 0.80
        and candidate_auc >= baseline_auc - 0.01
    )

    if candidate_is_better:
        promoted_model = candidate_model
        promoted_name = "RandomForestClassifier"
        promoted_metrics = candidate_metrics

        logger.info(
            "Candidate model PROMOTED."
        )
    else:
        promoted_model = baseline_model
        promoted_name = "LogisticRegression"
        promoted_metrics = baseline_metrics

        logger.info(
            "Baseline model retained."
        )

    # -----------------------------------------------------
    # 8. Save model
    # -----------------------------------------------------

    model_path = (
        MODEL_DIR
        / "stage1_approval_model.joblib"
    )

    joblib.dump(
        promoted_model,
        model_path,
    )

    logger.info(
        f"Promoted model saved to {model_path}"
    )

    # -----------------------------------------------------
    # 9. Save evaluation report
    # -----------------------------------------------------

    report = {
        "stage": "stage1_loan_approval",
        "baseline": {
            "model": "LogisticRegression",
            "metrics": baseline_metrics,
        },
        "candidate": {
            "model": "RandomForestClassifier",
            "metrics": candidate_metrics,
        },
        "promoted_model": promoted_name,
        "promoted_metrics": promoted_metrics,
    }

    report_path = (
        REPORT_DIR
        / "stage1_evaluation.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    logger.info(
        f"Evaluation report saved to {report_path}"
    )

    # -----------------------------------------------------
    # 10. Final output
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL PROMOTION")
    print("=" * 70)
    print(f"Selected model : {promoted_name}")
    print(f"Model path     : {model_path}")
    print(f"Report path    : {report_path}")
    print("Status         : SUCCESS")
    print("=" * 70)

    return report


# =========================================================
# Script entry point
# =========================================================

if __name__ == "__main__":
    train_stage1()