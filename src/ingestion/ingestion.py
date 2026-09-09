from pathlib import Path
import logging

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_approval_dataset.csv"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "loan_data_processed.csv"
)
LOG_DIR = PROJECT_ROOT / "artifacts" / "logs"


# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "ingestion.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Expected schema
# ---------------------------------------------------------

EXPECTED_COLUMNS = [
    "loan_id",
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
    "loan_status",
]


# ---------------------------------------------------------
# Ingestion function
# ---------------------------------------------------------

def ingest_data():
    """
    Read the raw loan dataset, clean column names,
    validate the schema, perform basic quality checks,
    and save the processed dataset.
    """

    logger.info("Starting loan dataset ingestion.")

    # 1. Check that raw file exists
    if not RAW_DATA_PATH.exists():
        logger.error(f"Raw dataset not found: {RAW_DATA_PATH}")
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    # 2. Read raw CSV
    df = pd.read_csv(RAW_DATA_PATH)

    logger.info(f"Raw rows read: {len(df)}")
    logger.info(f"Raw columns found: {len(df.columns)}")

    # 3. Clean column names
    df.columns = df.columns.str.strip()

    logger.info("Column whitespace cleaned.")

    # 4. Validate schema
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    unexpected_columns = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing_columns:
        logger.error(
            f"Missing expected columns: {sorted(missing_columns)}"
        )
        raise ValueError(
            f"Missing expected columns: {sorted(missing_columns)}"
        )

    if unexpected_columns:
        logger.warning(
            f"Unexpected columns detected: {sorted(unexpected_columns)}"
        )

    logger.info("Schema validation completed.")

    # 5. Check duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        logger.warning(
            f"Duplicate rows detected: {duplicate_count}"
        )
    else:
        logger.info("No duplicate rows detected.")

    # 6. Check missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()

    if total_missing > 0:
        logger.warning(
            f"Missing values detected: {total_missing}"
        )
    else:
        logger.info("No missing values detected.")

    # 7. Save processed dataset
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    logger.info(
        f"Processed dataset saved to: {PROCESSED_DATA_PATH}"
    )

    logger.info("Loan dataset ingestion completed successfully.")

    # 8. Print useful information to terminal
    print("=" * 60)
    print("LOAN DATA INGESTION")
    print("=" * 60)
    print(f"Input file       : {RAW_DATA_PATH}")
    print(f"Rows ingested    : {len(df)}")
    print(f"Columns          : {len(df.columns)}")
    print(f"Duplicates       : {duplicate_count}")
    print(f"Missing values   : {total_missing}")
    print(f"Output file      : {PROCESSED_DATA_PATH}")
    print("Status           : SUCCESS")
    print("=" * 60)


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    ingest_data()