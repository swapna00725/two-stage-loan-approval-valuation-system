from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.feature_engineering.feature_engineering import (
    get_stage1_features,
)
from src.preprocessing.preprocessing import (
    get_preprocessor,
)

def cv_precision(model, X, y):
    predictions = model.predict(X)

    return precision_score(
        y,
        predictions,
        pos_label="Approved",
        zero_division=0,
    )


def cv_recall(model, X, y):
    predictions = model.predict(X)

    return recall_score(
        y,
        predictions,
        pos_label="Approved",
        zero_division=0,
    )


def cv_f1(model, X, y):
    predictions = model.predict(X)

    return f1_score(
        y,
        predictions,
        pos_label="Approved",
        zero_division=0,
    )


def cv_roc_auc(model, X, y):
    probabilities = model.predict_proba(X)

    classes = model.classes_

    approved_index = list(classes).index("Approved")

    approved_probabilities = probabilities[
        :,
        approved_index,
    ]

    y_binary = y.map(
        {
            "Rejected": 0,
            "Approved": 1,
        }
    )

    return roc_auc_score(
        y_binary,
        approved_probabilities,
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


# =========================================================
# Load data
# =========================================================

df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 70)
print("STAGE 1 — MODEL DIAGNOSTICS")
print("=" * 70)


# =========================================================
# 1. Duplicate check
# =========================================================

duplicate_count = df.duplicated().sum()

print("\n1. DUPLICATE CHECK")
print("-" * 70)
print(f"Duplicate rows: {duplicate_count}")


# =========================================================
# 2. Target distribution
# =========================================================

print("\n2. TARGET DISTRIBUTION")
print("-" * 70)

print(
    df["loan_status"]
    .value_counts()
    .to_string()
)


# =========================================================
# 3. Feature creation
# =========================================================

X = get_stage1_features(df)
y = df["loan_status"].str.strip()

print("\n3. STAGE 1 FEATURES")
print("-" * 70)

print(f"Feature count: {X.shape[1]}")

for column in X.columns:
    print(f" - {column}")


# =========================================================
# 4. Check suspicious feature relationships
# =========================================================

print("\n4. FEATURE GROUPS")
print("-" * 70)

for column in X.columns:

    if "loan" in column.lower():
        print(f"Loan-related feature: {column}")

    if "asset" in column.lower():
        print(f"Asset-related feature: {column}")

    if "cibil" in column.lower():
        print(f"Credit-related feature: {column}")


# =========================================================
# 5. Build model
# =========================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            get_preprocessor(),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# =========================================================
# 6. Cross-validation
# =========================================================

print("\n5. CROSS-VALIDATION")
print("-" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "precision": cv_precision,
        "recall": cv_recall,
        "f1": cv_f1,
        "roc_auc": cv_roc_auc,
    },
    n_jobs=-1,
)

for metric in [
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
]:

    values = scores[metric]

    print(
        f"{metric:20}: "
        f"mean={values.mean():.4f}, "
        f"std={values.std():.4f}"
    )


# =========================================================
# 7. Train full model for feature importance
# =========================================================

model.fit(X, y)


# =========================================================
# 8. Permutation importance
# =========================================================

print("\n6. PERMUTATION IMPORTANCE")
print("-" * 70)

importance = permutation_importance(
    model,
    X,
    y,
    scoring="accuracy",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

importance_df = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": importance.importances_mean,
    }
).sort_values(
    "importance",
    ascending=False,
)

print(
    importance_df.to_string(index=False)
)


print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)