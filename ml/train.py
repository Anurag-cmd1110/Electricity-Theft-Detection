
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def generate_sample_data(n_samples=2000, random_state=RANDOM_STATE):
    """Generate reproducible synthetic meter data for demonstration."""
    rng = np.random.default_rng(random_state)

    previous_month_kwh = rng.uniform(80, 600, n_samples)

    usage_change_pct = rng.normal(0, 15, n_samples)
    usage_change_pct = np.clip(usage_change_pct, -60, 60)

    current_month_kwh = (
        previous_month_kwh * (1 + usage_change_pct / 100)
    )
    current_month_kwh = np.maximum(current_month_kwh, 1)

    avg_daily_kwh = current_month_kwh / 30
    night_usage_ratio = rng.uniform(0.1, 0.6, n_samples)
    weekend_usage_ratio = rng.uniform(0.5, 1.5, n_samples)
    zero_reading_days = rng.integers(0, 5, n_samples)

    suspicious_score = (
        (usage_change_pct < -30).astype(int)
        + (
            current_month_kwh < previous_month_kwh * 0.65
        ).astype(int)
        + (zero_reading_days >= 3).astype(int)
    )

    theft_label = (
        (suspicious_score >= 2)
        & (rng.random(n_samples) > 0.10)
    ).astype(int)

    # Introduce a small amount of label noise.
    noise_indices = rng.choice(
        n_samples,
        size=int(n_samples * 0.03),
        replace=False,
    )
    theft_label[noise_indices] = 1 - theft_label[noise_indices]

    return pd.DataFrame({
        "previous_month_kwh": previous_month_kwh.round(2),
        "current_month_kwh": current_month_kwh.round(2),
        "avg_daily_kwh": avg_daily_kwh.round(2),
        "usage_change_pct": usage_change_pct.round(2),
        "night_usage_ratio": night_usage_ratio.round(3),
        "weekend_usage_ratio": weekend_usage_ratio.round(3),
        "zero_reading_days": zero_reading_days,
        "theft_label": theft_label,
    })


def main():
    # Project directories.
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "ml" / "data"
    model_dir = project_dir / "ml" / "models"

    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Output paths.
    data_path = data_dir / "synthetic_meter_data.csv"
    model_path = model_dir / "theft_detection_model.joblib"
    metrics_path = data_dir / "evaluation_metrics.json"
    report_path = data_dir / "classification_report.csv"
    confusion_path = data_dir / "confusion_matrix.csv"
    curve_path = data_dir / "precision_recall_curve.csv"
    importance_path = data_dir / "feature_importance.csv"

    # --------------------------------------------------
    # 1. Generate and save demonstration dataset
    # --------------------------------------------------
    df = generate_sample_data()
    df.to_csv(data_path, index=False)

    feature_columns = [
        "previous_month_kwh",
        "current_month_kwh",
        "avg_daily_kwh",
        "usage_change_pct",
        "night_usage_ratio",
        "weekend_usage_ratio",
        "zero_reading_days",
    ]

    X = df[feature_columns]
    y = df["theft_label"]

    print("=" * 60)
    print("GRIDGUARD - RANDOM FOREST MODEL EVALUATION")
    print("=" * 60)
    print(f"Dataset: {data_path}")
    print(f"Total samples: {len(df)}")
    print(f"Normal samples: {(y == 0).sum()}")
    print(f"Suspicious samples: {(y == 1).sum()}")
    print("Dataset type: Synthetic demonstration data")

    # --------------------------------------------------
    # 2. Split data into training, validation and test sets
    # --------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print("\nDataset split:")
    print(f"Training samples:   {len(X_train)} (60%)")
    print(f"Validation samples: {len(X_val)} (20%)")
    print(f"Test samples:       {len(X_test)} (20%)")

    # --------------------------------------------------
    # 3. Compare candidate models using validation data
    # --------------------------------------------------
    candidates = [
        ("balanced", "balanced"),
        ("balanced_subsample", "balanced_subsample"),
    ]

    best_model = None
    best_name = None
    best_ap = -1.0

    for name, weight in candidates:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            class_weight=weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        val_scores = model.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, val_scores)

        print(f"\n{name} validation average precision: {ap:.4f}")

        if ap > best_ap:
            best_ap = ap
            best_model = model
            best_name = name

    # --------------------------------------------------
    # 4. Select a decision threshold using validation data
    # --------------------------------------------------
    val_scores = best_model.predict_proba(X_val)[:, 1]

    precision_values, recall_values, thresholds = (
        precision_recall_curve(y_val, val_scores)
    )

    if len(thresholds) > 0:
        f1_values = (
            2 * precision_values[:-1] * recall_values[:-1]
            / (
                precision_values[:-1]
                + recall_values[:-1]
                + 1e-12
            )
        )

        best_index = int(np.argmax(f1_values))
        threshold = float(thresholds[best_index])
    else:
        threshold = 0.5

    print(f"\nSelected model: {best_name}")
    print(f"Validation average precision: {best_ap:.4f}")
    print(f"Validation-selected threshold: {threshold:.4f}")

    # --------------------------------------------------
    # 5. Final evaluation on the held-out test set
    # --------------------------------------------------
    test_scores = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_scores >= threshold).astype(int)

    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(
        y_test, test_predictions, zero_division=0
    )
    recall = recall_score(
        y_test, test_predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, test_predictions, zero_division=0
    )
    average_precision = average_precision_score(y_test, test_scores)

    try:
        roc_auc = roc_auc_score(y_test, test_scores)
    except ValueError:
        roc_auc = None

    # Explicit label order:
    # [[true normal predicted normal, true normal predicted suspicious],
    #  [true suspicious predicted normal, true suspicious predicted suspicious]]
    cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = (
        float(fp / (fp + tn)) if (fp + tn) else 0.0
    )
    false_negative_rate = (
        float(fn / (fn + tp)) if (fn + tp) else 0.0
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1-score:             {f1:.4f}")
    print(f"Average precision:    {average_precision:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC:              {roc_auc:.4f}")

    print(f"False positive rate:  {false_positive_rate:.4f}")
    print(f"False negative rate:  {false_negative_rate:.4f}")

    print("\nConfusion matrix:")
    print("Rows = actual class; columns = predicted class")
    print("Class 0 = Normal; Class 1 = Suspicious")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["Normal", "Suspicious"],
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # 6. Save evaluation metrics as JSON
    # --------------------------------------------------
    metrics = {
        "project": "GridGuard",
        "model": "RandomForestClassifier",
        "selected_class_weight": best_name,
        "dataset_type": "synthetic_demonstration",
        "total_samples": int(len(df)),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "decision_threshold": float(threshold),
        "validation_average_precision": float(best_ap),
        "test_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "average_precision": float(average_precision),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "false_positive_rate": float(false_positive_rate),
            "false_negative_rate": float(false_negative_rate),
        },
        "confusion_matrix": {
            "true_normal_predicted_normal": int(tn),
            "true_normal_predicted_suspicious": int(fp),
            "true_suspicious_predicted_normal": int(fn),
            "true_suspicious_predicted_suspicious": int(tp),
        },
        "important_note": (
            "Results use synthetic demonstration data and do not "
            "establish real-world electricity theft detection accuracy."
        ),
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # --------------------------------------------------
    # 7. Save the detailed classification report
    # --------------------------------------------------
    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Normal", "Suspicious"],
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(report_path)

    # --------------------------------------------------
    # 8. Save confusion matrix as a CSV file
    # --------------------------------------------------
    confusion_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Suspicious"],
        columns=["Predicted Normal", "Predicted Suspicious"],
    )
    confusion_df.to_csv(confusion_path)

    # --------------------------------------------------
    # 9. Save precision-recall curve data
    # --------------------------------------------------
    curve_precision, curve_recall, curve_thresholds = (
        precision_recall_curve(y_test, test_scores)
    )

    curve_df = pd.DataFrame({
        "precision": curve_precision,
        "recall": curve_recall,
    })

    # The curve contains one more precision/recall point than thresholds.
    curve_df["threshold"] = np.append(
        curve_thresholds,
        np.nan,
    )

    curve_df.to_csv(curve_path, index=False)

    # --------------------------------------------------
    # 10. Save feature importance
    # --------------------------------------------------
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": best_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(importance_path, index=False)

    # --------------------------------------------------
    # 11. Save the model and prediction configuration
    # --------------------------------------------------
    joblib.dump(
        {
            "model": best_model,
            "feature_columns": feature_columns,
            "threshold": threshold,
            "selected_class_weight": best_name,
            "dataset_type": "synthetic_demonstration",
            "model_name": "RandomForestClassifier",
        },
        model_path,
    )

    # --------------------------------------------------
    # 12. Display generated files
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("GRIDGUARD TRAINING COMPLETED")
    print("=" * 60)

    output_paths = [
        data_path,
        model_path,
        metrics_path,
        report_path,
        confusion_path,
        curve_path,
        importance_path,
    ]

    for path in output_paths:
        print(f"Saved: {path}")

    print("\nRemember: synthetic-data results are for demonstration only.")


if __name__ == "__main__":
    main()
