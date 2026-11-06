from pathlib import Path
import math

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "ml" / "models" / "theft_detection_model.joblib"


def get_reason_codes(features: dict) -> list[str]:
    """Provide rule-based explanations for why a pattern resembles theft."""
    reasons = []
    usage_change = features.get("usage_change_pct", 0)
    zero_days = features.get("zero_reading_days", 0)
    night_ratio = features.get("night_usage_ratio", 0)
    previous_kwh = features.get("previous_month_kwh", 0)
    current_kwh = features.get("current_month_kwh", 0)

    if usage_change <= -35:
        reasons.append(f"Severe consumption reduction ({usage_change:.1f}%)")
    elif previous_kwh > 0 and current_kwh < previous_kwh * 0.6:
        reasons.append("Current consumption dropped below 60% of previous period")

    if zero_days >= 3:
        reasons.append(f"Frequent zero-reading days recorded ({zero_days} days)")

    if night_ratio < 0.15:
        reasons.append(f"Unusually low night-time baseline usage ({night_ratio:.2f})")
    elif night_ratio > 0.55:
        reasons.append(f"Abnormally high night consumption ratio ({night_ratio:.2f})")

    if not reasons:
        reasons.append("Mild consumption anomaly based on multi-feature profile")

    return reasons


def predict_meter(features: dict) -> dict:
    if not isinstance(features, dict):
        raise ValueError("Features must be provided as a dictionary.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. Run ml/train.py first."
        )

    artifact = joblib.load(MODEL_PATH)

    required_artifact_keys = {"model", "feature_columns", "threshold"}
    missing_artifact_keys = required_artifact_keys - set(artifact.keys())

    if missing_artifact_keys:
        raise ValueError(
            "Model artifact is missing required keys: "
            + ", ".join(sorted(missing_artifact_keys))
        )

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = float(artifact["threshold"])

    missing_features = [col for col in feature_columns if col not in features]
    if missing_features:
        raise ValueError("Missing required features: " + ", ".join(missing_features))

    converted_features = {}
    for column in feature_columns:
        value = features[column]
        if value is None or isinstance(value, bool):
            raise ValueError(f"Feature '{column}' must be a valid numeric value.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{column}' must be numeric.") from None

        if not math.isfinite(numeric_value):
            raise ValueError(f"Feature '{column}' must be finite.")
        converted_features[column] = numeric_value

    input_data = pd.DataFrame([converted_features], columns=feature_columns)
    probabilities = model.predict_proba(input_data)

    class_labels = list(model.classes_)
    if 1 not in class_labels:
        raise ValueError("The trained model does not contain the suspicious class (1).")

    suspicious_class_index = class_labels.index(1)
    risk_score = float(probabilities[0][suspicious_class_index])
    is_suspicious = risk_score >= threshold

    score_pct = round(risk_score * 100, 2)
    severity = "High" if score_pct >= 70 else "Medium" if score_pct >= 40 else "Low"
    reasons = get_reason_codes(converted_features) if is_suspicious else []

    return {
        "risk_score": score_pct,
        "threshold": round(threshold, 4),
        "severity": severity,
        "is_suspicious": is_suspicious,
        "prediction": "suspicious_pattern" if is_suspicious else "normal_pattern",
        "reasons": reasons,
        "model_type": artifact.get("dataset_type", "synthetic_demonstration"),
        "warning": (
            "Demonstration model trained on synthetic data. This score indicates pattern similarity "
            "and must not be treated as legal proof of electricity theft."
        ),
    }