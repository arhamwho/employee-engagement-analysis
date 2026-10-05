"""
Load and inspect the IBM HR Analytics Employee Attrition & Performance CSV.

Rules:
- Read the file from data/ only.
- Never write back to the original CSV.
- Never invent employee rows if the file is missing.
- Return a copy of the table so later cleaning cannot touch the file on disk.

This module does not train a model.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    ALTERNATE_CSV_FILENAMES,
    DATA_DIR,
    EXPECTED_CSV_FILENAME,
    EXPECTED_CSV_PATH,
)


def missing_dataset_message() -> str:
    """Clear instructions for where the real CSV must go."""
    alternate_lines = "\n".join(
        f"    - {DATA_DIR / name}" for name in ALTERNATE_CSV_FILENAMES
    )
    return (
        "DATASET NOT FOUND.\n"
        "\n"
        "The IBM HR Analytics Employee Attrition & Performance CSV is not in this project.\n"
        "No fake employee records were created.\n"
        "\n"
        "Place the real CSV here:\n"
        f"    {EXPECTED_CSV_PATH}\n"
        "\n"
        f"Expected filename:\n"
        f"    {EXPECTED_CSV_FILENAME}\n"
        "\n"
        "If your download still has the common Kaggle name, you may instead place:\n"
        f"{alternate_lines}\n"
        "\n"
        "Do not edit or overwrite that file after you add it. This program only reads it.\n"
    )


def resolve_dataset_path() -> Optional[Path]:
    """
    Return the CSV path if a real file is present, otherwise None.

    Checks the expected filename first, then known Kaggle alternate names.
    """
    if EXPECTED_CSV_PATH.is_file():
        return EXPECTED_CSV_PATH

    for name in ALTERNATE_CSV_FILENAMES:
        candidate = DATA_DIR / name
        if candidate.is_file():
            return candidate

    return None


def dataset_is_available() -> bool:
    """True only when a real CSV file exists in data/."""
    return resolve_dataset_path() is not None


def validate_dataset_file() -> dict:
    """
    Report whether the dataset file exists.

    Returns a dictionary so the notebook and Streamlit app can show
    the same status without guessing any row counts.
    """
    found_path = resolve_dataset_path()
    if found_path is None:
        return {
            "available": False,
            "path": None,
            "message": missing_dataset_message(),
        }

    return {
        "available": True,
        "path": str(found_path),
        "message": (
            "Dataset file found. "
            f"Loading a read-only copy from:\n    {found_path}"
        ),
    }


def load_raw_dataset() -> pd.DataFrame:
    """
    Load the original CSV into memory.

    - Raises FileNotFoundError with a full how-to message if the file is missing.
    - Uses pandas.read_csv (read only).
    - Returns a copy so later cells can clean data without changing the CSV.
    """
    status = validate_dataset_file()
    if not status["available"]:
        raise FileNotFoundError(status["message"])

    csv_path = status["path"]

    try:
        raw_df = pd.read_csv(csv_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not read the CSV at {csv_path}. "
            f"pandas error: {exc}"
        ) from exc

    if raw_df.empty:
        raise ValueError(f"The file at {csv_path} has no rows.")

    return raw_df.copy()


def identify_column_types(df: pd.DataFrame) -> dict:
    """
    Split columns using pandas dtypes (automatic, no hand-picked list).

    - numerical: int and float columns
    - categorical: object, category, and bool columns
    - other: anything left (rare; for example datetime)
    """
    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    labeled = set(numerical_columns) | set(categorical_columns)
    other_columns = [col for col in df.columns if col not in labeled]

    return {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "other_columns": other_columns,
    }


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Build an inspection summary from the loaded DataFrame only.

    All numbers come from pandas. Nothing is hard-coded.
    """
    column_types = identify_column_types(df)
    missing_counts = df.isna().sum()
    duplicate_count = int(df.duplicated().sum())

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str),
        "missing_values": missing_counts,
        "missing_values_total": int(missing_counts.sum()),
        "duplicate_rows": duplicate_count,
        "numeric_summary": df.describe(include="number"),
        "categorical_summary": (
            df.describe(include=["object", "category", "bool"])
            if column_types["categorical_columns"]
            else None
        ),
        **column_types,
    }
