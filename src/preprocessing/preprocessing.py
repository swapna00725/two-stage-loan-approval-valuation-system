from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# Stage 1 / General feature definitions
# =========================================================

NUMERICAL_COLUMNS = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "total_assets",
    "loan_to_income_ratio",
    "assets_to_income_ratio",
    "income_per_dependent",
    "requested_loan_per_term",
    "bank_asset_ratio",
    "luxury_asset_ratio",
    "asset_coverage_ratio",
]


CATEGORICAL_COLUMNS = [
    "education",
    "self_employed",
    "cibil_band",
]


# =========================================================
# Stage 2 feature definitions
# =========================================================

STAGE2_NUMERICAL_COLUMNS = [
    "no_of_dependents",
    "income_annum",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "total_assets",
    "assets_to_income_ratio",
    "income_per_dependent",
    "bank_asset_ratio",
    "luxury_asset_ratio",
]


STAGE2_CATEGORICAL_COLUMNS = [
    "education",
    "self_employed",
    "cibil_band",
]


# =========================================================
# Numerical pipeline
# =========================================================

def create_numerical_pipeline():
    """
    Create preprocessing pipeline for numerical features.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


# =========================================================
# Categorical pipeline
# =========================================================

def create_categorical_pipeline():
    """
    Create preprocessing pipeline for categorical features.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


# =========================================================
# Generic preprocessor builder
# =========================================================

def build_preprocessor(
    numerical_columns,
    categorical_columns,
):
    """
    Build a ColumnTransformer using supplied feature lists.
    """

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                create_numerical_pipeline(),
                numerical_columns,
            ),
            (
                "categorical",
                create_categorical_pipeline(),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


# =========================================================
# Stage 1 / general preprocessor
# =========================================================

def get_preprocessor():
    """
    Return the preprocessing pipeline used by Stage 1.
    """

    return build_preprocessor(
        NUMERICAL_COLUMNS,
        CATEGORICAL_COLUMNS,
    )


# =========================================================
# Stage 2 preprocessor
# =========================================================

def get_stage2_preprocessor():
    """
    Return the preprocessing pipeline used by Stage 2.

    loan_amount and all features derived from loan_amount
    are intentionally excluded.
    """

    return build_preprocessor(
        STAGE2_NUMERICAL_COLUMNS,
        STAGE2_CATEGORICAL_COLUMNS,
    )