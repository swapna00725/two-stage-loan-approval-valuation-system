from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering.feature_engineering import (
    get_stage1_features,
    get_stage2_features,
)


class LoanApprovalValuationPipeline:
    """
    Master orchestration pipeline for the
    Two Stage Loan Approval & Valuation System.

    Stage 1:
        Determines whether the loan should be approved.

    Stage 2:
        Estimates the loan amount for approved applications.
    """

    STAGE1_VERSION = "1.0.0"
    STAGE2_VERSION = "1.0.0"

    def __init__(self):

        # -------------------------------------------------
        # Project root
        # -------------------------------------------------

        self.project_root = Path(__file__).resolve().parents[2]

        # -------------------------------------------------
        # Model paths
        # -------------------------------------------------

        self.stage1_model_path = (
            self.project_root
            / "artifacts"
            / "models"
            / "stage1_approval_model.joblib"
        )

        self.stage2_model_path = (
            self.project_root
            / "artifacts"
            / "models"
            / "stage2_valuation_model.joblib"
        )

        # -------------------------------------------------
        # Validate model artifacts
        # -------------------------------------------------

        if not self.stage1_model_path.exists():

            raise FileNotFoundError(
                f"Stage 1 model not found: "
                f"{self.stage1_model_path}"
            )

        if not self.stage2_model_path.exists():

            raise FileNotFoundError(
                f"Stage 2 model not found: "
                f"{self.stage2_model_path}"
            )

        # -------------------------------------------------
        # Load trained models
        # -------------------------------------------------

        self.stage1_model = joblib.load(
            self.stage1_model_path
        )

        self.stage2_model = joblib.load(
            self.stage2_model_path
        )

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_input(self, application: dict):
        """
        Validate the raw loan application.

        The pipeline expects the raw applicant fields.
        Engineered features are created internally.
        """

        required_columns = {
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
        }

        missing_columns = (
            required_columns
            - set(application.keys())
        )

        if missing_columns:

            raise ValueError(
                "Missing required input fields: "
                + ", ".join(sorted(missing_columns))
            )

    # =====================================================
    # MAIN PREDICTION
    # =====================================================

    def predict(self, application: dict) -> dict:
        """
        Run the complete two-stage prediction.

        Parameters
        ----------
        application : dict
            Raw loan application fields.

        Returns
        -------
        dict
            Structured two-stage prediction result.
        """

        # -------------------------------------------------
        # 1. Validate input
        # -------------------------------------------------

        self._validate_input(application)

        # -------------------------------------------------
        # 2. Convert input to DataFrame
        # -------------------------------------------------

        df = pd.DataFrame([application])

        # -------------------------------------------------
        # 3. Stage 1 feature engineering
        # -------------------------------------------------

        stage1_features = get_stage1_features(df)

        # -------------------------------------------------
        # 4. Stage 1 approval prediction
        # -------------------------------------------------

        stage1_prediction = self.stage1_model.predict(
            stage1_features
        )[0]

        # -------------------------------------------------
        # 5. Stage 1 approval probability
        # -------------------------------------------------

        if hasattr(
            self.stage1_model,
            "predict_proba",
        ):

            probabilities = (
                self.stage1_model.predict_proba(
                    stage1_features
                )[0]
            )

            classes = list(
                self.stage1_model.classes_
            )

            approved_index = classes.index(
                "Approved"
            )

            approval_probability = float(
                probabilities[approved_index]
            )

        else:

            approval_probability = None

        # -------------------------------------------------
        # 6. Normalize approval decision
        # -------------------------------------------------

        approval_decision = str(
            stage1_prediction
        ).strip()

        # -------------------------------------------------
        # 7. Rejected application
        # -------------------------------------------------

        if approval_decision != "Approved":

            return {
                "approval_decision": "Rejected",
                "approval_probability": (
                    approval_probability
                ),
                "recommended_loan_amount": None,
                "stage1_model_version": (
                    self.STAGE1_VERSION
                ),
                "stage2_model_version": None,
            }

        # -------------------------------------------------
        # 8. Stage 2 feature engineering
        # -------------------------------------------------

        stage2_features = get_stage2_features(df)

        # -------------------------------------------------
        # 9. Stage 2 valuation
        # -------------------------------------------------

        valuation = self.stage2_model.predict(
            stage2_features
        )[0]

        # -------------------------------------------------
        # 10. Ensure valid valuation
        # -------------------------------------------------

        recommended_loan_amount = max(
            0.0,
            float(valuation),
        )

        # -------------------------------------------------
        # 11. Final response
        # -------------------------------------------------

        return {
            "approval_decision": "Approved",
            "approval_probability": (
                approval_probability
            ),
            "recommended_loan_amount": (
                recommended_loan_amount
            ),
            "stage1_model_version": (
                self.STAGE1_VERSION
            ),
            "stage2_model_version": (
                self.STAGE2_VERSION
            ),
        }