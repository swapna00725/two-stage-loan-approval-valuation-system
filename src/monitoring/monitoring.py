"""
Monitoring utilities for the
Two Stage Loan Approval & Valuation System.

Provides:
- Data quality checks
- Feature drift checks
- Prediction distribution monitoring
- Retraining trigger evaluation
"""

from pathlib import Path

import json
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

REFERENCE_DATA = Path(
    "data/processed/loan_data_processed.csv"
)

RECENT_DATA = Path(
    "data/recent_batch/sample_applications.csv"
)

PREDICTIONS_DATA = Path(
    "data/recent_batch/predictions.csv"
)

REPORT_DIR = Path(
    "artifacts/monitoring"
)

REPORT_FILE = (
    REPORT_DIR / "monitoring_report.json"
)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

REQUIRED_COLUMNS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


# =========================================================
# NUMERIC FEATURES FOR DRIFT
# =========================================================

DRIFT_FEATURES = [
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


# =========================================================
# DATA QUALITY CHECK
# =========================================================

def check_data_quality(df):
    """
    Check basic data quality issues.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    null_counts = (
        df[REQUIRED_COLUMNS]
        .isnull()
        .sum()
        .to_dict()
    )

    total_nulls = int(
        sum(null_counts.values())
    )

    quality_issues = []

    if missing_columns:

        quality_issues.append(
            "Missing required columns"
        )

    if total_nulls > 0:

        quality_issues.append(
            "Missing values detected"
        )

    # -----------------------------------------------------
    # Range checks
    # -----------------------------------------------------

    if "cibil_score" in df.columns:

        invalid_cibil = (
            (df["cibil_score"] < 0)
            | (df["cibil_score"] > 900)
        ).sum()

        if invalid_cibil > 0:

            quality_issues.append(
                "Invalid CIBIL score detected"
            )

    if "income_annum" in df.columns:

        invalid_income = (
            df["income_annum"] <= 0
        ).sum()

        if invalid_income > 0:

            quality_issues.append(
                "Invalid income detected"
            )

    if "loan_amount" in df.columns:

        invalid_loan = (
            df["loan_amount"] <= 0
        ).sum()

        if invalid_loan > 0:

            quality_issues.append(
                "Invalid loan amount detected"
            )

    if "loan_term" in df.columns:

        invalid_term = (
            df["loan_term"] <= 0
        ).sum()

        if invalid_term > 0:

            quality_issues.append(
                "Invalid loan term detected"
            )

    return {
        "row_count": int(len(df)),
        "missing_columns": missing_columns,
        "null_counts": null_counts,
        "total_nulls": total_nulls,
        "quality_issues": quality_issues,
        "quality_status": (
            "WARNING"
            if quality_issues
            else "PASS"
        ),
    }


# =========================================================
# DRIFT CHECK
# =========================================================

def calculate_drift(
    reference_df,
    recent_df,
):
    """
    Compare means of selected numeric features.

    Drift is calculated as:

        abs(recent_mean - reference_mean)
        / abs(reference_mean)
    """

    drift_results = {}

    for feature in DRIFT_FEATURES:

        if feature not in reference_df.columns:
            continue

        if feature not in recent_df.columns:
            continue

        reference_mean = float(
            reference_df[feature].mean()
        )

        recent_mean = float(
            recent_df[feature].mean()
        )

        if reference_mean == 0:

            drift_percentage = 0.0

        else:

            drift_percentage = (
                abs(
                    recent_mean
                    - reference_mean
                )
                / abs(reference_mean)
                * 100
            )

        drift_results[feature] = {
            "reference_mean": reference_mean,
            "recent_mean": recent_mean,
            "drift_percentage": (
                drift_percentage
            ),
        }

    return drift_results


# =========================================================
# DRIFT STATUS
# =========================================================

def evaluate_drift(
    drift_results,
    threshold=20.0,
):
    """
    Flag features whose mean changed by
    more than the configured threshold.
    """

    drifted_features = []

    for feature, result in drift_results.items():

        if (
            result["drift_percentage"]
            > threshold
        ):

            drifted_features.append(
                feature
            )

    return {
        "threshold_percentage": threshold,
        "drifted_features": drifted_features,
        "drift_detected": (
            len(drifted_features) > 0
        ),
    }


# =========================================================
# PREDICTION MONITORING
# =========================================================

def monitor_predictions(
    predictions_df,
):
    """
    Monitor Stage 1 prediction distribution.
    """

    if (
        "approval_decision"
        not in predictions_df.columns
    ):

        return {
            "status": "UNAVAILABLE"
        }

    counts = (
        predictions_df[
            "approval_decision"
        ]
        .value_counts()
        .to_dict()
    )

    total = len(predictions_df)

    percentages = {}

    for decision, count in counts.items():

        percentages[decision] = (
            float(count)
            / total
            * 100
        )

    return {
        "status": "PASS",
        "prediction_counts": counts,
        "prediction_percentages": percentages,
    }


# =========================================================
# RETRAINING TRIGGER
# =========================================================

def should_retrain(
    drift_detected,
    model_performance_drop=False,
    sufficient_new_data=False,
):
    """
    Determine whether model retraining
    should be considered.
    """

    reasons = []

    if drift_detected:

        reasons.append(
            "Feature drift detected"
        )

    if model_performance_drop:

        reasons.append(
            "Model performance degradation"
        )

    if sufficient_new_data:

        reasons.append(
            "Sufficient new training data"
        )

    return {
        "retrain_required": (
            len(reasons) > 0
        ),
        "reasons": reasons,
    }


# =========================================================
# MAIN MONITORING PIPELINE
# =========================================================

def run_monitoring():

    print("=" * 60)

    print(
        "TWO STAGE LOAN SYSTEM - MONITORING"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    reference_df = pd.read_csv(
        REFERENCE_DATA
    )

    recent_df = pd.read_csv(
        RECENT_DATA
    )

    # -----------------------------------------------------
    # Data quality
    # -----------------------------------------------------

    quality_result = check_data_quality(
        recent_df
    )

    # -----------------------------------------------------
    # Drift
    # -----------------------------------------------------

    drift_results = calculate_drift(
        reference_df,
        recent_df,
    )

    drift_status = evaluate_drift(
        drift_results
    )

    # -----------------------------------------------------
    # Prediction monitoring
    # -----------------------------------------------------

    if PREDICTIONS_DATA.exists():

        predictions_df = pd.read_csv(
            PREDICTIONS_DATA
        )

        prediction_result = (
            monitor_predictions(
                predictions_df
            )
        )

    else:

        prediction_result = {
            "status": "UNAVAILABLE"
        }

    # -----------------------------------------------------
    # Retraining trigger
    # -----------------------------------------------------

    retraining_result = should_retrain(
        drift_detected=(
            drift_status[
                "drift_detected"
            ]
        )
    )

    # -----------------------------------------------------
    # Complete report
    # -----------------------------------------------------

    report = {

        "reference_rows": int(
            len(reference_df)
        ),

        "recent_rows": int(
            len(recent_df)
        ),

        "data_quality": quality_result,

        "drift": {
            "features": drift_results,
            "summary": drift_status,
        },

        "predictions": prediction_result,

        "retraining": retraining_result,
    }

    # -----------------------------------------------------
    # Save report
    # -----------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print()

    print("1. DATA QUALITY")
    print("-" * 60)

    print(
        f"Recent rows       : "
        f"{quality_result['row_count']}"
    )

    print(
        f"Missing values    : "
        f"{quality_result['total_nulls']}"
    )

    print(
        f"Quality status    : "
        f"{quality_result['quality_status']}"
    )

    print()

    print("2. FEATURE DRIFT")
    print("-" * 60)

    for feature, result in (
        drift_results.items()
    ):

        print(
            f"{feature:30s} "
            f"{result['drift_percentage']:8.2f}%"
        )

    print()

    print(
        f"Drift threshold   : "
        f"{drift_status['threshold_percentage']:.2f}%"
    )

    print(
        f"Drift detected    : "
        f"{drift_status['drift_detected']}"
    )

    print()

    print("3. PREDICTION MONITORING")
    print("-" * 60)

    if prediction_result.get(
        "status"
    ) == "PASS":

        print(
            "Prediction counts:"
        )

        for (
            decision,
            count,
        ) in prediction_result[
            "prediction_counts"
        ].items():

            percentage = (
                prediction_result[
                    "prediction_percentages"
                ][decision]
            )

            print(
                f"  {decision:10s}: "
                f"{count} "
                f"({percentage:.2f}%)"
            )

    else:

        print(
            "Prediction monitoring "
            "unavailable."
        )

    print()

    print("4. RETRAINING DECISION")
    print("-" * 60)

    print(
        f"Retrain required : "
        f"{retraining_result['retrain_required']}"
    )

    if retraining_result["reasons"]:

        for reason in (
            retraining_result["reasons"]
        ):

            print(
                f"  - {reason}"
            )

    else:

        print(
            "  No retraining trigger."
        )

    print()

    print(
        f"Report saved      : "
        f"{REPORT_FILE}"
    )

    print()

    print("=" * 60)

    print(
        "MONITORING COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":

    run_monitoring()