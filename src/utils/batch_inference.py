"""
Batch inference for the
Two Stage Loan Approval & Valuation System.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline.pipeline import (
    LoanApprovalValuationPipeline,
)


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


def validate_input_columns(df):
    """
    Validate that the batch input contains
    all required application fields.
    """

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def run_batch_inference(
    input_path,
    output_path,
):
    """
    Run two-stage predictions over a CSV file.
    """

    input_path = Path(input_path)

    output_path = Path(output_path)

    print("=" * 60)
    print("BATCH LOAN INFERENCE")
    print("=" * 60)

    print(
        f"Input file : {input_path}"
    )

    df = pd.read_csv(
        input_path
    )

    print(
        f"Rows read  : {len(df)}"
    )

    validate_input_columns(
        df
    )

    pipeline = (
        LoanApprovalValuationPipeline()
    )

    results = []

    for _, row in df.iterrows():

        application = {
            column: row[column]
            for column in REQUIRED_COLUMNS
        }

        result = pipeline.predict(
            application
        )

        results.append(result)

    predictions = pd.DataFrame(
        results
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Predictions : {len(predictions)}"
    )

    print(
        f"Output file : {output_path}"
    )

    print(
        f"Status      : SUCCESS"
    )

    print("=" * 60)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run batch inference for "
            "loan approval and valuation."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV path",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path",
    )

    args = parser.parse_args()

    run_batch_inference(
        args.input,
        args.output,
    )


if __name__ == "__main__":
    main()