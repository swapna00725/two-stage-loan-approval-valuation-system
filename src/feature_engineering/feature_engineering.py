import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create reusable financial features for the loan ML system.

    The function does not modify the original dataframe.
    A new dataframe is returned.
    """

    data = df.copy()

    # -----------------------------------------------------
    # Clean categorical values
    # -----------------------------------------------------

    data["education"] = data["education"].astype(str).str.strip()
    data["self_employed"] = (
        data["self_employed"].astype(str).str.strip()
    )
     # loan_status is available during training,
     # but is not available during inference.
    if "loan_status" in data.columns:
        data["loan_status"] = (
        data["loan_status"].astype(str).str.strip()
    )

    # -----------------------------------------------------
    # Total assets
    # -----------------------------------------------------

    data["total_assets"] = (
        data["residential_assets_value"]
        + data["commercial_assets_value"]
        + data["luxury_assets_value"]
        + data["bank_asset_value"]
    )

    # -----------------------------------------------------
    # Loan-to-income ratio
    # -----------------------------------------------------

    data["loan_to_income_ratio"] = (
        data["loan_amount"]
        / data["income_annum"].replace(0, np.nan)
    )

    # -----------------------------------------------------
    # Assets-to-income ratio
    # -----------------------------------------------------

    data["assets_to_income_ratio"] = (
        data["total_assets"]
        / data["income_annum"].replace(0, np.nan)
    )

    # -----------------------------------------------------
    # Income per dependent
    # -----------------------------------------------------

    data["income_per_dependent"] = (
        data["income_annum"]
        / (data["no_of_dependents"] + 1)
    )

    # -----------------------------------------------------
    # Requested loan amount per term
    # -----------------------------------------------------

    data["requested_loan_per_term"] = (
        data["loan_amount"]
        / data["loan_term"].replace(0, np.nan)
    )

    # -----------------------------------------------------
    # Bank asset ratio
    # -----------------------------------------------------

    data["bank_asset_ratio"] = (
        data["bank_asset_value"]
        / data["total_assets"].replace(0, np.nan)
    )

    # -----------------------------------------------------
    # Luxury asset ratio
    # -----------------------------------------------------

    data["luxury_asset_ratio"] = (
        data["luxury_assets_value"]
        / data["total_assets"].replace(0, np.nan)
    )

    # -----------------------------------------------------
    # Asset coverage ratio
    # -----------------------------------------------------

    data["asset_coverage_ratio"] = (
        data["total_assets"]
        / data["loan_amount"].replace(0, np.nan)
    )

    # -----------------------------------------------------
    # CIBIL band
    # -----------------------------------------------------

    data["cibil_band"] = pd.cut(
        data["cibil_score"],
        bins=[
            -np.inf,
            549,
            649,
            749,
            np.inf,
        ],
        labels=[
            "Poor",
            "Fair",
            "Good",
            "Excellent",
        ],
    )

    return data


def get_stage1_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return features used by the Stage 1 loan approval model.

    Stage 1 predicts loan_status, so requested loan_amount
    is allowed as an input feature.
    """

    data = create_features(df)

    columns_to_drop = [
        "loan_id",
        "loan_status",
    ]

    return data.drop(
        columns=columns_to_drop,
        errors="ignore",
    )


def get_stage2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return features used by the Stage 2 loan valuation model.

    Stage 2 predicts loan_amount.

    Therefore:
    - loan_amount itself cannot be used
    - any feature derived from loan_amount cannot be used
    """

    data = create_features(df)

    target_dependent_features = [
        "loan_amount",
        "loan_to_income_ratio",
        "requested_loan_per_term",
        "asset_coverage_ratio",
    ]

    columns_to_drop = [
        "loan_id",
        "loan_status",
        *target_dependent_features,
    ]

    return data.drop(
        columns=columns_to_drop,
        errors="ignore",
    )