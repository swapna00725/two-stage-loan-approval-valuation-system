from pathlib import Path
import json
import logging

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.feature_engineering.feature_engineering import (
    get_stage2_features,
)
from src.preprocessing.preprocessing import (
    get_stage2_preprocessor,
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
    filename=LOG_DIR / "stage2_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# Evaluation function
# =========================================================

def evaluate_regressor(model, X_test, y_test):
    """
    Evaluate a regression model using MAE, RMSE and R².
    """

    predictions = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(
            y_test,
            predictions,
        ),

        "rmse": mean_squared_error(
            y_test,
            predictions,
        ) ** 0.5,

        "r2": r2_score(
            y_test,
            predictions,
        ),
    }

    return metrics


# =========================================================
# Training function
# =========================================================

def train_stage2():
    """
    Train and compare Stage 2 loan valuation models.

    Baseline:
        Linear Regression

    Candidate:
        Random Forest Regressor
    """

    logger.info("Starting Stage 2 training.")

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
    # 2. Create Stage 2 features
    # -----------------------------------------------------

    X = get_stage2_features(df)

    # Target = loan amount
    y = df["loan_amount"]

    logger.info(
        f"Stage 2 feature count before preprocessing: {X.shape[1]}"
    )

    logger.info(
        "Confirmed loan_amount is excluded from Stage 2 features."
    )

    # -----------------------------------------------------
    # 3. Train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
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
                 get_stage2_preprocessor(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    logger.info(
        "Training baseline Linear Regression."
    )

    baseline_model.fit(
        X_train,
        y_train,
    )

    baseline_metrics = evaluate_regressor(
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
                 get_stage2_preprocessor(),
            ),
            (
                "model",
                RandomForestRegressor(
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
        "Training candidate Random Forest Regressor."
    )

    candidate_model.fit(
        X_train,
        y_train,
    )

    candidate_metrics = evaluate_regressor(
        candidate_model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # 6. Display comparison
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE 2 — LOAN VALUATION MODEL COMPARISON")
    print("=" * 70)

    print("\nBaseline — Linear Regression")

    for metric, value in baseline_metrics.items():
        print(f"{metric:12}: {value:,.4f}")

    print("\nCandidate — Random Forest Regressor")

    for metric, value in candidate_metrics.items():
        print(f"{metric:12}: {value:,.4f}")

    # -----------------------------------------------------
    # 7. Promotion gate
    # -----------------------------------------------------

    baseline_rmse = baseline_metrics["rmse"]
    candidate_rmse = candidate_metrics["rmse"]

    baseline_r2 = baseline_metrics["r2"]
    candidate_r2 = candidate_metrics["r2"]

    candidate_is_better = (
        candidate_rmse <= baseline_rmse
        and candidate_r2 >= baseline_r2
    )

    if candidate_is_better:

        promoted_model = candidate_model
        promoted_name = "RandomForestRegressor"
        promoted_metrics = candidate_metrics

        logger.info(
            "Candidate model PROMOTED."
        )

    else:

        promoted_model = baseline_model
        promoted_name = "LinearRegression"
        promoted_metrics = baseline_metrics

        logger.info(
            "Baseline model retained."
        )

    # -----------------------------------------------------
    # 8. Save model
    # -----------------------------------------------------

    model_path = (
        MODEL_DIR
        / "stage2_valuation_model.joblib"
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
        "stage": "stage2_loan_valuation",

        "target": "loan_amount",

        "baseline": {
            "model": "LinearRegression",
            "metrics": baseline_metrics,
        },

        "candidate": {
            "model": "RandomForestRegressor",
            "metrics": candidate_metrics,
        },

        "promoted_model": promoted_name,

        "promoted_metrics": promoted_metrics,
    }

    report_path = (
        REPORT_DIR
        / "stage2_evaluation.json"
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

    print(
        f"Selected model : {promoted_name}"
    )

    print(
        f"Model path     : {model_path}"
    )

    print(
        f"Report path    : {report_path}"
    )

    print(
        "Status         : SUCCESS"
    )

    print("=" * 70)

    return report


# =========================================================
# Script entry point
# =========================================================

if __name__ == "__main__":
    train_stage2()