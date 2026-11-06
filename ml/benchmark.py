from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def load_or_generate_data(data_path: Path):
    if data_path.exists():
        return pd.read_csv(data_path)
    
    from ml.train import generate_sample_data
    df = generate_sample_data(n_samples=2000, random_state=RANDOM_STATE)
    df.to_csv(data_path, index=False)
    return df


def main():
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "ml" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "synthetic_meter_data.csv"
    benchmark_table_path = data_dir / "model_comparison_table.csv"
    benchmark_chart_path = data_dir / "model_benchmark_chart.png"

    df = load_or_generate_data(data_path)

    feature_cols = [
        "previous_month_kwh",
        "current_month_kwh",
        "avg_daily_kwh",
        "usage_change_pct",
        "night_usage_ratio",
        "weekend_usage_ratio",
        "zero_reading_days",
    ]

    X = df[feature_cols]
    y = df["theft_label"]

    # 70% Train, 30% Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    # 3 Model Candidates
    candidates = {
        "Logistic Regression (Baseline)": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4, random_state=RANDOM_STATE
        )
    }

    results = []

    print("=" * 75)
    print("GRIDGUARD - MULTI-MODEL BENCHMARKING (7TH SEMESTER CSE MAJOR PROJECT)")
    print("=" * 75)

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        print(f"\nModel: {name}")
        print(f"Accuracy:  {acc * 100:.2f}% | Precision: {prec * 100:.2f}%")
        print(f"Recall:    {rec * 100:.2f}% | F1-Score:  {f1 * 100:.2f}%")
        print(f"ROC-AUC:   {auc:.4f}  | False Neg Rate (Missed Theft): {fnr * 100:.2f}%")

        results.append({
            "Model": name,
            "Accuracy": round(acc * 100, 2),
            "Precision": round(prec * 100, 2),
            "Recall": round(rec * 100, 2),
            "F1_Score": round(f1 * 100, 2),
            "ROC_AUC": round(auc, 4),
            "False_Negative_Rate": round(fnr * 100, 2)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(benchmark_table_path, index=False)
    print(f"\n[Saved] Comparison table written to: {benchmark_table_path}")

    # Generate academic comparison chart for presentation & report
    plt.figure(figsize=(10, 5), dpi=300)
    bar_width = 0.22
    indices = np.arange(len(results))

    plt.bar(indices - bar_width, results_df["Recall"], width=bar_width, label="Recall (Catching Theft)", color="#10b981")
    plt.bar(indices, results_df["Precision"], width=bar_width, label="Precision", color="#6366f1")
    plt.bar(indices + bar_width, results_df["F1_Score"], width=bar_width, label="F1-Score", color="#06b6d4")

    plt.xlabel("Algorithms", fontweight="bold", labelpad=10)
    plt.ylabel("Score Percentage (%)", fontweight="bold")
    plt.title("GridGuard Comparative Model Evaluation (Test Set)", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(indices, results_df["Model"], fontsize=9)
    plt.ylim(0, 105)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig(benchmark_chart_path)
    plt.close()
    print(f"[Saved] Publication chart written to: {benchmark_chart_path}")


if __name__ == "__main__":
    main()