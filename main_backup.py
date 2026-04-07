"""
============================================================
  Material Property Prediction using Machine Learning
  Full End-to-End Implementation
  Datasets: Concrete (A), Steel UTS (B), Materials (C)
============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, classification_report,
    ConfusionMatrixDisplay
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  CONFIGURATION  (update filenames here)
# ─────────────────────────────────────────
BASE_DIR = r"C:\Users\HP\Downloads\material_pp\material_ml_project"

CONFIG = {
    "dataset_a": os.path.join(BASE_DIR, "data", "raw", "Concrete_Data.xls"),
    "dataset_b": os.path.join(BASE_DIR, "data", "raw", "steel_strength.csv"),
    "dataset_c": os.path.join(BASE_DIR, "data", "raw", "mechanical_material.csv"),
    "test_size": 0.2,
    "random_state": 42,
    "output_dir": os.path.join(BASE_DIR, "outputs"),
    "model_dir": os.path.join(BASE_DIR, "models"),
    "processed_dir": os.path.join(BASE_DIR, "data", "processed"),
}

# ─────────────────────────────────────────
#  COLUMN DEFINITIONS
# ─────────────────────────────────────────

# Dataset A ─ Concrete
A_FEATURES = [
    "Cement", "Slag", "FlyAsh", "Water",
    "Superplasticizer", "CoarseAgg", "FineAgg", "Age"
]
A_TARGET = "Strength"

# Dataset B ─ Steel UTS
B_DROP     = ["formula", "yield strength", "elongation"]   # drop: text + other targets
B_TARGET   = "tensile strength"

# Dataset C ─ Materials (multi-task)
C_DROP     = ["Material"]          # text column to drop
C_REG_TARGETS = ["Su", "Sy"]       # multi-output regression targets
C_CLS_TARGET  = "Use"             # classification target


# ══════════════════════════════════════════════════════════════
#  SECTION 1 ─ DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_dataset_a(path: str) -> pd.DataFrame:
    """Load Concrete Compressive Strength dataset (.xls)."""
    df = pd.read_excel(path, header=0)
    df.columns = A_FEATURES + [A_TARGET]
    print(f"[A] Loaded Concrete dataset  → {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_dataset_b(path: str) -> pd.DataFrame:
    """Load Steel Ultimate Tensile Strength dataset (.csv)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"[B] Loaded Steel dataset     → {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_dataset_c(path: str) -> pd.DataFrame:
    """Load Materials Mechanical Properties dataset (.csv)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    print(f"[C] Loaded Materials dataset → {df.shape[0]} rows, {df.shape[1]} cols")
    return df


# ══════════════════════════════════════════════════════════════
#  SECTION 2 ─ DATA PREPROCESSING & CLEANING
# ══════════════════════════════════════════════════════════════

def clean_dataset_a(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset A: already clean; remove duplicates only."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    print(f"[A] After cleaning: {len(df)} rows  (removed {before - len(df)})")
    return df


def clean_dataset_b(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset B: drop text/id columns, handle missing values."""
    # Drop irrelevant columns that exist in the dataframe
    cols_to_drop = [c for c in B_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    before = len(df)
    # Drop rows where the target is missing
    df = df.dropna(subset=[B_TARGET]).reset_index(drop=True)
    # Fill remaining missing numeric values with column mean
    df = df.fillna(df.mean(numeric_only=True))
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[B] After cleaning: {len(df)} rows  (removed {before - len(df)})")
    return df


def clean_dataset_c(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset C: drop text columns, encode classification target."""
    # Drop non-numeric identifier columns
    cols_to_drop = [c for c in C_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    before = len(df)
    df = df.dropna(subset=C_REG_TARGETS + [C_CLS_TARGET]).reset_index(drop=True)
    df = df.fillna(df.mean(numeric_only=True))
    df = df.drop_duplicates().reset_index(drop=True)

    # Encode classification target: TRUE→1, FALSE→0
    if df[C_CLS_TARGET].dtype == object or df[C_CLS_TARGET].dtype == bool:
        df[C_CLS_TARGET] = df[C_CLS_TARGET].map(
            {True: 1, False: 0, "TRUE": 1, "FALSE": 0, "true": 1, "false": 0}
        ).astype(int)

    print(f"[C] After cleaning: {len(df)} rows  (removed {before - len(df)})")
    print(f"[C] Use column distribution:\n{df[C_CLS_TARGET].value_counts().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════
#  SECTION 3 ─ FEATURE / TARGET SPLIT & SCALING
# ══════════════════════════════════════════════════════════════

def split_and_scale_a(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Prepare Dataset A: single-output regression."""
    X = df[A_FEATURES]
    y = df[A_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"\n[A] Train: {X_train_sc.shape}  Test: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, y_train, y_test, scaler


def split_and_scale_b(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Prepare Dataset B: single-output regression."""
    feature_cols = [c for c in df.columns if c != B_TARGET]
    X = df[feature_cols]
    y = df[B_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[B] Train: {X_train_sc.shape}  Test: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, y_train, y_test, scaler, feature_cols


def split_and_scale_c(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Prepare Dataset C: multi-output regression + classification."""
    all_targets = C_REG_TARGETS + [C_CLS_TARGET]
    X        = df.drop(columns=all_targets)
    y_reg    = df[C_REG_TARGETS]
    y_cls    = df[C_CLS_TARGET]

    # Shared split — same rows for both tasks
    X_train, X_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=test_size, random_state=random_state
    )
    yc_train = y_cls.loc[X_train.index]
    yc_test  = y_cls.loc[X_test.index]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[C] Train: {X_train_sc.shape}  Test: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, yr_train, yr_test, yc_train, yc_test, scaler


# ══════════════════════════════════════════════════════════════
#  SECTION 4 ─ MODEL TRAINING
# ══════════════════════════════════════════════════════════════

def train_module_a(X_train, y_train, random_state=42):
    """Module 1 — Concrete: Linear Regression (baseline) + Random Forest."""
    models = {}

    # Baseline: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["Linear Regression"] = lr

    # Main model: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    print("[A] Module 1 training complete.")
    return models


def train_module_b(X_train, y_train, random_state=42):
    """Module 2 — Steel: LR (baseline) + Random Forest + XGBoost."""
    models = {}

    # Baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["Linear Regression"] = lr

    # Main model
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    # Upgrade: XGBoost
    xgb = XGBRegressor(
        n_estimators=100, max_depth=4,
        learning_rate=0.1, random_state=random_state,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb

    print("[B] Module 2 training complete.")
    return models


def train_module_c_regression(X_train, y_train, random_state=42):
    """Module 3a — Multi-output regression: RF + XGBoost wrapped."""
    models = {}

    # Multi-output Random Forest
    rf_multi = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=random_state)
    )
    rf_multi.fit(X_train, y_train)
    models["MultiOutput Random Forest"] = rf_multi

    # Multi-output XGBoost
    xgb_multi = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, random_state=random_state,
            verbosity=0
        )
    )
    xgb_multi.fit(X_train, y_train)
    models["MultiOutput XGBoost"] = xgb_multi

    print("[C] Module 3a (multi-output regression) training complete.")
    return models


def train_module_c_classification(X_train, y_train, random_state=42):
    """Module 3b — Classification: Random Forest Classifier."""
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    print("[C] Module 3b (classification) training complete.")
    return clf


# ══════════════════════════════════════════════════════════════
#  SECTION 5 ─ EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_regression(models: dict, X_test, y_test, module_name: str):
    """Evaluate single-output regression models and print metrics."""
    print(f"\n{'='*55}")
    print(f"  {module_name} — Regression Evaluation")
    print(f"{'='*55}")
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)
        print(f"  {name:<28}  MAE: {mae:8.3f}   R²: {r2:.4f}")
        results[name] = {"MAE": mae, "R2": r2, "preds": preds}
    return results


def evaluate_multi_output_regression(models: dict, X_test, y_test):
    """Evaluate multi-output regression models per target column."""
    print(f"\n{'='*55}")
    print("  Module 3a — Multi-Output Regression Evaluation")
    print(f"{'='*55}")
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        print(f"\n  [{name}]")
        target_results = {}
        for i, col in enumerate(y_test.columns):
            mae = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
            r2  = r2_score(y_test.iloc[:, i], preds[:, i])
            print(f"    {col:<20}  MAE: {mae:8.3f}   R²: {r2:.4f}")
            target_results[col] = {"MAE": mae, "R2": r2}
        results[name] = target_results
    return results


def evaluate_classification(clf, X_test, y_test, output_dir: str):
    """Evaluate classification model and save confusion matrix."""
    print(f"\n{'='*55}")
    print("  Module 3b — Classification Evaluation")
    print(f"{'='*55}")
    preds = clf.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"  Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, preds, target_names=["Not Used", "Used"]))

    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=["Not Used", "Used"], ax=ax
    )
    ax.set_title("Module 3b — Confusion Matrix (Material Use)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_module3b.png"), dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → outputs/confusion_matrix_module3b.png")
    return acc, preds


# ══════════════════════════════════════════════════════════════
#  SECTION 6 ─ VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

def plot_feature_importance(model, feature_names: list, title: str,
                            filename: str, output_dir: str):
    """Bar chart of Random Forest feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"  Feature importance chart saved → outputs/{filename}")


def plot_pred_vs_actual(y_test, preds, title: str,
                        filename: str, output_dir: str):
    """Scatter plot of predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, preds, alpha=0.5, color="darkorange", edgecolors="k", s=20)
    mn = min(y_test.min(), preds.min())
    mx = max(y_test.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual (MPa)")
    ax.set_ylabel("Predicted (MPa)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"  Predicted vs Actual chart saved → outputs/{filename}")


def plot_correlation_heatmap(df: pd.DataFrame, title: str,
                              filename: str, output_dir: str):
    """Correlation heatmap for a dataframe."""
    numeric_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        numeric_df.corr(), annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5, ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"  Correlation heatmap saved → outputs/{filename}")


def plot_metrics_comparison(results: dict, title: str,
                             filename: str, output_dir: str):
    """Bar chart comparing MAE and R² across models."""
    names = list(results.keys())
    maes  = [results[n]["MAE"] for n in names]
    r2s   = [results[n]["R2"]  for n in names]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x, maes, color=["#2196F3", "#FF9800", "#4CAF50"][:len(names)])
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_title("MAE (lower is better)")
    axes[0].set_ylabel("MAE (MPa)")

    axes[1].bar(x, r2s, color=["#2196F3", "#FF9800", "#4CAF50"][:len(names)])
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=15, ha="right")
    axes[1].set_title("R² Score (higher is better)")
    axes[1].set_ylabel("R²")
    axes[1].set_ylim(0, 1.05)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"  Metrics comparison chart saved → outputs/{filename}")


# ══════════════════════════════════════════════════════════════
#  SECTION 7 ─ RESULTS EXPORT
# ══════════════════════════════════════════════════════════════

def save_results_csv(results_a, results_b, results_c_reg,
                     acc_c, output_dir: str):
    """Save all metrics to a single CSV summary file."""
    rows = []

    # Module A
    for model_name, metrics in results_a.items():
        rows.append({
            "Module": "A - Concrete",
            "Model": model_name,
            "Task": "Regression",
            "MAE": round(metrics["MAE"], 4),
            "R2": round(metrics["R2"], 4),
            "Accuracy": None
        })

    # Module B
    for model_name, metrics in results_b.items():
        rows.append({
            "Module": "B - Steel UTS",
            "Model": model_name,
            "Task": "Regression",
            "MAE": round(metrics["MAE"], 4),
            "R2": round(metrics["R2"], 4),
            "Accuracy": None
        })

    # Module C — Regression (best model per target)
    for model_name, target_metrics in results_c_reg.items():
        for target, metrics in target_metrics.items():
            rows.append({
                "Module": f"C - Materials ({target})",
                "Model": model_name,
                "Task": "Multi-Output Regression",
                "MAE": round(metrics["MAE"], 4),
                "R2": round(metrics["R2"], 4),
                "Accuracy": None
            })

    # Module C — Classification
    rows.append({
        "Module": "C - Materials (Use)",
        "Model": "Random Forest Classifier",
        "Task": "Classification",
        "MAE": None,
        "R2": None,
        "Accuracy": round(acc_c, 4)
    })

    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "results_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\n  Results summary saved → outputs/results_summary.csv")
    return summary_df


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    os.makedirs(CONFIG["processed_dir"], exist_ok=True)
    rs  = CONFIG["random_state"]
    ts  = CONFIG["test_size"]
    out = CONFIG["output_dir"]
    mdl = CONFIG["model_dir"]
    prc = CONFIG["processed_dir"]

    print("\n" + "="*60)
    print("  Material Property Prediction — Full Pipeline")
    print("="*60 + "\n")

    # ── Check dataset files exist ──
    for key in ["dataset_a", "dataset_b", "dataset_c"]:
        path = CONFIG[key]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found: '{path}'\n"
                f"Please place the file in the same folder as main.py\n"
                f"and update CONFIG['{key}'] if the filename differs."
            )

    # ──────────────────────────────
    #  MODULE 1 — CONCRETE (A)
    # ──────────────────────────────
    print("\n── MODULE 1: Concrete Compressive Strength ─────────────")
    df_a  = load_dataset_a(CONFIG["dataset_a"])
    df_a  = clean_dataset_a(df_a)
    plot_correlation_heatmap(df_a, "Dataset A — Concrete Correlation",
                             "heatmap_concrete.png", out)

    Xa_tr, Xa_te, ya_tr, ya_te, scaler_a = split_and_scale_a(df_a, ts, rs)
    joblib.dump(
        {"X_train": Xa_tr, "X_test": Xa_te, "y_train": ya_tr, "y_test": ya_te, "scaler": scaler_a},
        os.path.join(prc, "concrete_split.pkl")
    )
    print(f"  Split saved → data/processed/concrete_split.pkl")

    models_a = train_module_a(Xa_tr, ya_tr, rs)
    joblib.dump(models_a["Random Forest"], os.path.join(mdl, "module_a_rf.pkl"))
    print(f"  Model saved → models/module_a_rf.pkl")

    results_a = evaluate_regression(models_a, Xa_te, ya_te, "Module 1 — Concrete")

    # Visualizations
    plot_feature_importance(
        models_a["Random Forest"], A_FEATURES,
        "Concrete — Feature Importance (Random Forest)",
        "feat_imp_concrete.png", out
    )
    plot_pred_vs_actual(
        ya_te, models_a["Random Forest"].predict(Xa_te),
        "Concrete — Predicted vs Actual Strength",
        "pred_vs_actual_concrete.png", out
    )
    plot_metrics_comparison(results_a, "Module 1 — Model Comparison (Concrete)",
                            "metrics_concrete.png", out)

    # ──────────────────────────────
    #  MODULE 2 — STEEL UTS (B)
    # ──────────────────────────────
    print("\n── MODULE 2: Steel Ultimate Tensile Strength ───────────")
    df_b  = load_dataset_b(CONFIG["dataset_b"])
    df_b  = clean_dataset_b(df_b)
    plot_correlation_heatmap(df_b, "Dataset B — Steel Correlation",
                             "heatmap_steel.png", out)

    Xb_tr, Xb_te, yb_tr, yb_te, scaler_b, feat_b = split_and_scale_b(df_b, ts, rs)
    joblib.dump(
        {"X_train": Xb_tr, "X_test": Xb_te, "y_train": yb_tr, "y_test": yb_te,
         "scaler": scaler_b, "feature_cols": feat_b},
        os.path.join(prc, "steel_split.pkl")
    )
    print(f"  Split saved → data/processed/steel_split.pkl")

    models_b = train_module_b(Xb_tr, yb_tr, rs)
    joblib.dump(models_b["XGBoost"], os.path.join(mdl, "module_b_xgb.pkl"))
    print(f"  Model saved → models/module_b_xgb.pkl")

    results_b = evaluate_regression(models_b, Xb_te, yb_te, "Module 2 — Steel UTS")

    best_b = max(results_b, key=lambda k: results_b[k]["R2"])
    plot_pred_vs_actual(
        yb_te, results_b[best_b]["preds"],
        f"Steel UTS — Predicted vs Actual ({best_b})",
        "pred_vs_actual_steel.png", out
    )
    plot_metrics_comparison(results_b, "Module 2 — Model Comparison (Steel UTS)",
                            "metrics_steel.png", out)

    # ──────────────────────────────
    #  MODULE 3 — MATERIALS (C)
    # ──────────────────────────────
    print("\n── MODULE 3: Materials Mechanical Properties ───────────")
    df_c  = load_dataset_c(CONFIG["dataset_c"])
    df_c  = clean_dataset_c(df_c)
    plot_correlation_heatmap(df_c, "Dataset C — Materials Correlation",
                             "heatmap_materials.png", out)

    Xc_tr, Xc_te, yc_reg_tr, yc_reg_te, yc_cls_tr, yc_cls_te, scaler_c = \
        split_and_scale_c(df_c, ts, rs)
    joblib.dump(
        {"X_train": Xc_tr, "X_test": Xc_te,
         "y_reg_train": yc_reg_tr, "y_reg_test": yc_reg_te,
         "y_cls_train": yc_cls_tr, "y_cls_test": yc_cls_te,
         "scaler": scaler_c},
        os.path.join(prc, "materials_split.pkl")
    )
    print(f"  Split saved → data/processed/materials_split.pkl")

    # 3a: Multi-output Regression
    models_c_reg = train_module_c_regression(Xc_tr, yc_reg_tr, rs)
    joblib.dump(models_c_reg["MultiOutput Random Forest"], os.path.join(mdl, "module_c_reg.pkl"))
    print(f"  Model saved → models/module_c_reg.pkl")

    results_c_reg = evaluate_multi_output_regression(
        models_c_reg, Xc_te, yc_reg_te
    )

    # 3b: Classification
    clf_c = train_module_c_classification(Xc_tr, yc_cls_tr, rs)
    joblib.dump(clf_c, os.path.join(mdl, "module_c_clf.pkl"))
    print(f"  Model saved → models/module_c_clf.pkl")

    acc_c, preds_cls = evaluate_classification(clf_c, Xc_te, yc_cls_te, out)

    # ──────────────────────────────
    #  RESULTS SUMMARY
    # ──────────────────────────────
    print("\n── FINAL RESULTS SUMMARY ───────────────────────────────")
    summary_df = save_results_csv(results_a, results_b, results_c_reg, acc_c, out)
    print("\n" + summary_df.to_string(index=False))

    print("\n" + "="*60)
    print("  Pipeline complete. All outputs saved in 'outputs/' folder.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
