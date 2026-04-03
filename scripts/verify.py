"""
============================================================
  Material Property Prediction — Dataset Verification Script
  Run this BEFORE main.py to confirm datasets are ready.
============================================================
"""

import os
import sys
import numpy as np
import pandas as pd

# ─────────────────────────────────────────
#  UPDATE THESE FILENAMES IF NEEDED
# ─────────────────────────────────────────
RAW_DATA_DIR = r"C:\Users\HP\Downloads\material_pp\material_ml_project\data\raw"

CONFIG = {
    "dataset_a": os.path.join(RAW_DATA_DIR, "Concrete_Data.xls"),
    "dataset_b": os.path.join(RAW_DATA_DIR, "steel_strength.csv"),
    "dataset_c": os.path.join(RAW_DATA_DIR, "mechanical_material.csv"),
}

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"


def section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def check_file_exists(path: str, label: str) -> bool:
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"{PASS} {label} found  ({size:.1f} KB)")
        return True
    else:
        print(f"{FAIL} {label} NOT FOUND at path: '{path}'")
        print(f"        → Place the file in the same folder as verify.py")
        return False


def check_shape(df: pd.DataFrame, label: str,
                min_rows=50, min_cols=2):
    r, c = df.shape
    ok = r >= min_rows and c >= min_cols
    status = PASS if ok else FAIL
    print(f"{status} {label} shape: {r} rows × {c} cols")
    return ok


def check_missing(df: pd.DataFrame, label: str) -> bool:
    missing = df.isna().sum()
    total   = missing.sum()
    if total == 0:
        print(f"{PASS} {label}: No missing values")
        return True
    else:
        print(f"{WARN} {label}: {total} missing values detected")
        print(f"         Columns with nulls:\n{missing[missing > 0].to_string()}")
        print(f"         → main.py will fill these with column mean (auto-handled)")
        return True   # handled automatically; not a blocker


def check_duplicates(df: pd.DataFrame, label: str) -> bool:
    n_dup = df.duplicated().sum()
    if n_dup == 0:
        print(f"{PASS} {label}: No duplicate rows")
        return True
    else:
        print(f"{WARN} {label}: {n_dup} duplicate rows  (auto-removed in main.py)")
        return True


def check_dtypes(df: pd.DataFrame, label: str,
                 expected_numeric: list) -> bool:
    """Confirm expected columns exist and are numeric."""
    all_ok = True
    for col in expected_numeric:
        if col not in df.columns:
            print(f"{FAIL} {label}: Column '{col}' not found")
            all_ok = False
        elif not np.issubdtype(df[col].dtype, np.number):
            print(f"{FAIL} {label}: Column '{col}' is not numeric "
                  f"(dtype={df[col].dtype})")
            all_ok = False
    if all_ok:
        print(f"{PASS} {label}: All expected columns present and numeric")
    return all_ok


def check_target_range(series: pd.Series, label: str,
                        lo=0, hi=1e9) -> bool:
    series = pd.to_numeric(series, errors="coerce").dropna()
    mn, mx = series.min(), series.max()
    ok = lo <= mn and mx <= hi and not series.isna().all()
    status = PASS if ok else WARN
    print(f"{status} {label}: min={mn:.2f}  max={mx:.2f}  mean={series.mean():.2f}")
    return ok


def check_object_columns(df: pd.DataFrame, label: str,
                          allowed_text: list = None) -> bool:
    """Warn about any text/object columns that are NOT in the allowed list."""
    allowed_text = allowed_text or []
    obj_cols = [c for c in df.select_dtypes(include=["object", "str"]).columns
                if c not in allowed_text]
    if not obj_cols:
        print(f"{PASS} {label}: No unexpected text columns in features")
        return True
    else:
        print(f"{WARN} {label}: Text columns found → {obj_cols}")
        print(f"         → These will be dropped automatically in main.py")
        return True


def check_class_balance(series: pd.Series, label: str) -> bool:
    vc = series.value_counts()
    print(f"{PASS} {label} class distribution:")
    for val, cnt in vc.items():
        pct = 100 * cnt / len(series)
        print(f"         {val}: {cnt} rows ({pct:.1f}%)")
    if len(vc) < 2:
        print(f"{FAIL} {label}: Only ONE class found — classification will fail!")
        return False
    return True


# ══════════════════════════════════════════════════════════════
#  DATASET-SPECIFIC CHECKS
# ══════════════════════════════════════════════════════════════

def verify_dataset_a(path: str) -> bool:
    section("DATASET A — Concrete Compressive Strength")
    if not check_file_exists(path, "Concrete_Data.xls"):
        return False

    try:
        df = pd.read_excel(path, header=0)
        df.columns = [
            "Cement", "Slag", "FlyAsh", "Water",
            "Superplasticizer", "CoarseAgg", "FineAgg",
            "Age", "Strength"
        ]
    except Exception as e:
        print(f"{FAIL} Could not load dataset A: {e}")
        return False

    ok = True
    ok &= check_shape(df, "Dataset A", min_rows=100, min_cols=9)
    ok &= check_missing(df, "Dataset A")
    ok &= check_duplicates(df, "Dataset A")
    ok &= check_dtypes(
        df, "Dataset A",
        expected_numeric=[
            "Cement", "Slag", "FlyAsh", "Water",
            "Superplasticizer", "CoarseAgg", "FineAgg",
            "Age", "Strength"
        ]
    )
    ok &= check_target_range(df["Strength"], "Dataset A — Strength (MPa)", lo=0, hi=200)
    print(f"\n  Sample rows:\n{df.head(3).to_string(index=False)}")
    return ok


def verify_dataset_b(path: str) -> bool:
    section("DATASET B — Steel Ultimate Tensile Strength")
    if not check_file_exists(path, "steel_strength.csv"):
        return False

    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
    except Exception as e:
        print(f"{FAIL} Could not load dataset B: {e}")
        return False

    ok = True
    ok &= check_shape(df, "Dataset B", min_rows=50, min_cols=5)
    ok &= check_missing(df, "Dataset B")
    ok &= check_duplicates(df, "Dataset B")

    # Check target column exists
    if "tensile strength" not in df.columns:
        print(f"{FAIL} Dataset B: 'tensile strength' column not found")
        print(f"       Available columns: {df.columns.tolist()}")
        ok = False
    else:
        ok &= check_target_range(
            df["tensile strength"].dropna(),
            "Dataset B — Tensile Strength (MPa)", lo=100, hi=5000
        )

    # Warn about text columns (formula etc.)
    check_object_columns(df, "Dataset B", allowed_text=["formula"])

    print(f"\n  Sample rows:\n{df.head(3).to_string(index=False)}")
    return ok


def verify_dataset_c(path: str) -> bool:
    section("DATASET C — Materials Mechanical Properties")
    if not check_file_exists(path, "mechanical_material.csv"):
        return False

    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"{FAIL} Could not load dataset C: {e}")
        return False

    ok = True
    ok &= check_shape(df, "Dataset C", min_rows=100, min_cols=5)
    ok &= check_missing(df, "Dataset C")
    ok &= check_duplicates(df, "Dataset C")

    # Check regression target columns
    for col in ["Su", "Sy"]:
        if col not in df.columns:
            print(f"{FAIL} Dataset C: Regression target '{col}' not found")
            ok = False
        else:
            check_target_range(
                df[col].dropna(), f"Dataset C — {col} (MPa)", lo=0, hi=10000
            )

    # Check classification target
    if "Use" not in df.columns:
        print(f"{FAIL} Dataset C: Classification target 'Use' not found")
        ok = False
    else:
        # Encode for balance check
        use_col = df["Use"].map(
            {True: 1, False: 0, "TRUE": 1, "FALSE": 0,
             "true": 1, "false": 0}
        ).fillna(df["Use"])
        ok &= check_class_balance(use_col, "Dataset C — Use")

    check_object_columns(df, "Dataset C", allowed_text=["Material", "Use"])
    print(f"\n  Sample rows:\n{df.head(3).to_string(index=False)}")
    return ok


# ══════════════════════════════════════════════════════════════
#  DEPENDENCY CHECK
# ══════════════════════════════════════════════════════════════

def check_dependencies():
    section("PYTHON DEPENDENCIES")
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "openpyxl": "openpyxl",
    }
    all_ok = True
    for module, pkg in required.items():
        try:
            __import__(module)
            print(f"{PASS} {pkg}")
        except ImportError:
            print(f"{FAIL} {pkg} not installed  → run: pip install {pkg}")
            all_ok = False
    return all_ok


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*55)
    print("  Dataset Verification Script")
    print("  Material Property Prediction Project")
    print("="*55)

    dep_ok = check_dependencies()
    a_ok   = verify_dataset_a(CONFIG["dataset_a"])
    b_ok   = verify_dataset_b(CONFIG["dataset_b"])
    c_ok   = verify_dataset_c(CONFIG["dataset_c"])

    section("FINAL REPORT")
    statuses = {
        "Dependencies": dep_ok,
        "Dataset A (Concrete)": a_ok,
        "Dataset B (Steel UTS)": b_ok,
        "Dataset C (Materials)": c_ok,
    }
    all_clear = True
    for name, status in statuses.items():
        icon = "✓" if status else "✗"
        print(f"  [{icon}]  {name}")
        if not status:
            all_clear = False

    print()
    if all_clear:
        print("  All checks passed. You are ready to run main.py!")
    else:
        print("  Some checks FAILED. Please fix the issues above,")
        print("  then re-run verify.py before running main.py.")
    print()

    return 0 if all_clear else 1


if __name__ == "__main__":
    sys.exit(main())
